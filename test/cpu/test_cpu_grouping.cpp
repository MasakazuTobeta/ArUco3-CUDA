// SPDX-License-Identifier: Apache-2.0
#include "cpu_grouping.hpp"

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "aruco3cuda/config.hpp"
#include "cpu_candidates.hpp"

namespace {

using aruco3cuda::DetectorConfig;
using aruco3cuda::hybrid::CandidateNode;
using aruco3cuda::hybrid::MarkerCandidate;
using Quads = std::vector<std::vector<cv::Point2f>>;
using Contours = std::vector<std::vector<cv::Point>>;

std::vector<cv::Point2f> rectangle(float x, float y, float width, float height) {
    return {cv::Point2f(x, y), cv::Point2f(x + width, y), cv::Point2f(x + width, y + height),
            cv::Point2f(x, y + height)};
}

void expect_candidate_equal(const MarkerCandidate& actual, const MarkerCandidate& expected) {
    EXPECT_EQ(actual.corners_, expected.corners_);
    EXPECT_EQ(actual.contour_, expected.contour_);
    EXPECT_EQ(actual.perimeter_, expected.perimeter_);
}

void expect_grouping_equal(const Quads& input, const DetectorConfig& config, int marker_size = 6,
                           cv::Size image_size = cv::Size(641, 479)) {
    Contours contours;
    for (std::size_t index = 0; index < input.size(); ++index) {
        // Give every input a distinct path, so stable representative and retry
        // ordering are checked even when quadrilaterals have equal perimeters.
        contours.push_back({cv::Point(static_cast<int>(index), 1),
                            cv::Point(static_cast<int>(index), 2),
                            cv::Point(static_cast<int>(index), 3)});
    }
    Quads reference_quads = input;
    Quads actual_quads = input;
    Contours reference_contours = contours;
    Contours actual_contours = contours;
    const std::vector<CandidateNode> expected = aruco3cuda::hybrid::filter_too_close_candidates(
            image_size, reference_quads, reference_contours, config, marker_size);
    const std::vector<CandidateNode> actual = aruco3cuda::cpu::detail::filter_spatial_candidates(
            image_size, actual_quads, actual_contours, config, marker_size);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t index = 0; index < actual.size(); ++index) {
        SCOPED_TRACE(index);
        expect_candidate_equal(actual[index], expected[index]);
        EXPECT_EQ(actual[index].parent_, expected[index].parent_);
        EXPECT_EQ(actual[index].depth_, expected[index].depth_);
        ASSERT_EQ(actual[index].close_contours_.size(), expected[index].close_contours_.size());
        for (std::size_t retry = 0; retry < actual[index].close_contours_.size(); ++retry) {
            SCOPED_TRACE(retry);
            expect_candidate_equal(actual[index].close_contours_[retry],
                                   expected[index].close_contours_[retry]);
        }
    }
}

TEST(CpuGroupingTest, shuffled_nested_duplicate_and_border_candidates_preserve_complete_nodes) {
    Quads quads;
    for (int level = 0; level < 8; ++level) {
        const float inset = static_cast<float>(level) * 7.25F;
        quads.push_back(rectangle(100.0F + inset, 90.0F + inset, 190.0F - 2.0F * inset,
                                  170.0F - 2.0F * inset));
    }
    quads.push_back(quads[0]);
    quads.push_back(quads[3]);
    quads.push_back(rectangle(311.0F, 120.0F, 100.0F, 100.0F));
    quads.push_back(rectangle(315.0F, 124.0F, 100.0F, 100.0F));
    for (float border : {std::nextafter(3.0F, 0.0F), 3.0F, std::nextafter(3.0F, 4.0F)}) {
        quads.push_back(rectangle(border, 20.0F, 50.0F, 50.0F));
        quads.push_back(rectangle(640.0F - border - 50.0F, 400.0F, 50.0F, 50.0F));
    }
    std::mt19937 random(20261117U);
    for (int iteration = 0; iteration < 60; ++iteration) {
        SCOPED_TRACE(iteration);
        std::shuffle(quads.begin(), quads.end(), random);
        for (std::size_t index = 0; index < quads.size(); ++index) {
            const int rotation =
                    static_cast<int>((index + static_cast<std::size_t>(iteration)) % 4U);
            std::rotate(quads[index].begin(), quads[index].begin() + rotation, quads[index].end());
        }
        DetectorConfig config;
        config.min_marker_distance_rate_ = iteration % 2 == 0 ? 0.125 : 0.02;
        config.min_group_distance_ = iteration % 3 == 0 ? 0.0 : 0.21;
        expect_grouping_equal(quads, config);
    }
}

TEST(CpuGroupingTest, float_threshold_neighbors_preserve_strict_boundary_decisions) {
    for (float origin : {-64.0F, 0.03125F, 64.0F, 16384.0F, 65400.0F, 65535.0F}) {
        for (float gap : {0.0001220703125F, 0.25F, 1.0F, 16.0F, 64.0F}) {
            // Zero-width quads make the AABB separation bound equal the full
            // distance. Thin nonzero quads cover its nearby realistic cases.
            for (float width :
                 {0.0F, std::nextafter(origin, std::numeric_limits<float>::infinity()) - origin,
                  0.25F}) {
                Quads quads{rectangle(origin, 128.0F, width, 16.0F),
                            rectangle(origin + width + gap, 128.0F, width, 16.0F)};
                const float distance =
                        aruco3cuda::hybrid::average_quad_distance(quads[0], quads[1]);
                const float perimeter = aruco3cuda::hybrid::quad_perimeter(quads[1]);
                const float rate = distance / perimeter;
                for (float threshold_rate :
                     {std::nextafter(rate, 0.0F), rate,
                      std::nextafter(rate, std::numeric_limits<float>::infinity())}) {
                    DetectorConfig config;
                    config.min_distance_to_border_px_ = 0;
                    config.min_marker_distance_rate_ = static_cast<double>(threshold_rate);
                    SCOPED_TRACE("origin=" + std::to_string(origin) +
                                 " gap=" + std::to_string(gap) + " width=" + std::to_string(width));
                    expect_grouping_equal(quads, config, 6, cv::Size(65536, 65536));
                }
            }
        }
    }
}

TEST(CpuGroupingTest, seeded_fractional_quad_batches_match_unpruned_grouping) {
    cv::RNG random(1729U);
    std::mt19937 shuffle_random(314159U);
    for (int iteration = 0; iteration < 160; ++iteration) {
        SCOPED_TRACE(iteration);
        Quads quads;
        for (int index = 0; index < 48; ++index) {
            const float x = random.uniform(-12.0F, 630.0F);
            const float y = random.uniform(-12.0F, 468.0F);
            const float width = random.uniform(0.125F, 96.0F);
            const float height = random.uniform(0.125F, 96.0F);
            std::vector<cv::Point2f> quad = rectangle(x, y, width, height);
            for (cv::Point2f& point : quad) {
                point.x += random.uniform(-2.0F, 2.0F);
                point.y += random.uniform(-2.0F, 2.0F);
            }
            quads.push_back(quad);
            if (index % 6 == 0) {
                for (cv::Point2f& point : quad) {
                    point += cv::Point2f(0.125F, -0.25F);
                }
                quads.push_back(std::move(quad));
            }
        }
        std::shuffle(quads.begin(), quads.end(), shuffle_random);
        DetectorConfig config;
        config.min_marker_distance_rate_ = random.uniform(0.001, 0.4);
        config.min_group_distance_ = random.uniform(0.0, 0.7);
        config.min_distance_to_border_px_ = iteration % 7;
        expect_grouping_equal(quads, config, 4 + iteration % 4);
    }
}

TEST(CpuGroupingTest, integer_centroid_bound_preserves_threshold_neighbors_and_rotations) {
    for (float origin : {-256.0F, 0.0F, 16384.0F, 65000.0F, 65300.0F}) {
        for (int shift : {1, 2, 7, 31, 127}) {
            for (int perturbation : {0, 1, 3}) {
                Quads quads{rectangle(origin, origin, 128.0F, 128.0F),
                            rectangle(origin + static_cast<float>(shift), origin, 128.0F, 128.0F)};
                // Overlapping boxes leave the centroid as the useful bound.
                // One changed integer corner also exercises quarter-pixel sums.
                quads[1][2].x += static_cast<float>(perturbation);
                for (int rotation = 0; rotation < 4; ++rotation) {
                    std::rotate(quads[1].begin(), quads[1].begin() + 1, quads[1].end());
                    const float distance =
                            aruco3cuda::hybrid::average_quad_distance(quads[0], quads[1]);
                    const float perimeter = std::min(aruco3cuda::hybrid::quad_perimeter(quads[0]),
                                                     aruco3cuda::hybrid::quad_perimeter(quads[1]));
                    const float centroid_bound = static_cast<float>(shift) +
                                                 static_cast<float>(perturbation) * 0.25F - 1.0F;
                    for (float threshold : {distance, std::max(0.0F, centroid_bound)}) {
                        const float rate = threshold / perimeter;
                        for (float neighbor :
                             {std::nextafter(rate, 0.0F), rate,
                              std::nextafter(rate, std::numeric_limits<float>::infinity())}) {
                            DetectorConfig config;
                            config.min_distance_to_border_px_ = 0;
                            config.min_marker_distance_rate_ = static_cast<double>(neighbor);
                            SCOPED_TRACE("origin=" + std::to_string(origin) +
                                         " shift=" + std::to_string(shift) +
                                         " perturbation=" + std::to_string(perturbation) +
                                         " rotation=" + std::to_string(rotation));
                            expect_grouping_equal(quads, config, 6, cv::Size(65536, 65536));
                        }
                    }
                }
            }
        }
    }
}

}  // namespace
