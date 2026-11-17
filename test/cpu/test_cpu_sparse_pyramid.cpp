// SPDX-License-Identifier: Apache-2.0
#include "cpu_sparse_pyramid.hpp"

#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace {

using aruco3cuda::cpu::detail::SparsePyramid;

std::vector<cv::Size> sizes_of(const std::vector<cv::Mat>& pyramid) {
    std::vector<cv::Size> sizes;
    for (const cv::Mat& image : pyramid) {
        sizes.push_back(image.size());
    }
    return sizes;
}

TEST(CpuSparsePyramidTest, regional_pixels_match_full_pyramids_after_same_pointer_frame_changes) {
    const int previous_threads = cv::getNumThreads();
    cv::setNumThreads(1);
    struct RestoreThreads {
        int value_;
        ~RestoreThreads() { cv::setNumThreads(this->value_); }
    } restore{previous_threads};
    SparsePyramid sparse;
    cv::RNG random(173205U);
    for (cv::Size size : {cv::Size(64, 65), cv::Size(641, 479), cv::Size(1281, 721),
                          cv::Size(4097, 2049), cv::Size(65536, 33)}) {
        const std::size_t pitch = static_cast<std::size_t>(size.width) + 37U;
        std::vector<std::uint8_t> storage(pitch * static_cast<std::size_t>(size.height));
        cv::Mat image(size.height, size.width, CV_8UC1, storage.data(), pitch);
        ASSERT_FALSE(image.isSubmatrix());
        ASSERT_FALSE(image.isContinuous());
        for (int frame = 0; frame < 3; ++frame) {
            if (frame == 1) {
                image.setTo(255);
            } else {
                random.fill(image, cv::RNG::UNIFORM, 0, 256);
            }
            std::vector<cv::Mat> pyramid;
            cv::buildPyramid(image, pyramid, 3);
            const std::vector<cv::Size> sizes = sizes_of(pyramid);
            for (int tile : {1, 3, 64, 65536}) {
                sparse.reset(tile);
                for (int level = 0; level < static_cast<int>(pyramid.size()); ++level) {
                    const cv::Size extent = sizes[static_cast<std::size_t>(level)];
                    const int width = std::min(19, extent.width);
                    const int height = std::min(13, extent.height);
                    const std::array<cv::Rect, 5> requests = {
                            cv::Rect(0, 0, width, height),
                            cv::Rect(extent.width - width, extent.height - height, width, height),
                            cv::Rect((extent.width - width) / 3, (extent.height - height) / 3,
                                     width, height),
                            cv::Rect((extent.width - 1) / 2, (extent.height - 1) / 2, 1, 1),
                            cv::Rect(0, 0, 1, 1)};
                    for (const cv::Rect& request : requests) {
                        SCOPED_TRACE("size=" + std::to_string(size.width) + "x" +
                                     std::to_string(size.height) + " frame=" +
                                     std::to_string(frame) + " tile=" + std::to_string(tile) +
                                     " level=" + std::to_string(level));
                        const cv::Mat actual = sparse.region(image, sizes, level, request);
                        const cv::Mat expected = pyramid[static_cast<std::size_t>(level)](request);
                        ASSERT_EQ(actual.size(), expected.size());
                        ASSERT_EQ(cv::countNonZero(actual != expected), 0);
                    }
                }
            }
        }
    }
}

TEST(CpuSparsePyramidTest, canonical_warps_match_global_maps_at_fractional_and_border_coordinates) {
    SparsePyramid sparse;
    cv::RNG random(223606U);
    for (cv::Size size : {cv::Size(641, 479), cv::Size(1280, 720), cv::Size(65536, 33)}) {
        cv::Mat image(size, CV_8UC1);
        random.fill(image, cv::RNG::UNIFORM, 0, 256);
        std::vector<cv::Mat> pyramid;
        cv::buildPyramid(image, pyramid, 3);
        const std::vector<cv::Size> sizes = sizes_of(pyramid);
        for (int level = 0; level < 3; ++level) {
            const cv::Size extent = sizes[static_cast<std::size_t>(level)];
            for (float fraction : {0.0F, 0.125F, 0.49999997F, 0.5F, 0.50000006F, 0.875F}) {
                const float width = static_cast<float>(extent.width);
                const float height = static_cast<float>(extent.height);
                const std::vector<std::vector<cv::Point2f>> quads = {
                        {{width * 0.32F + fraction, height * 0.30F},
                         {width * 0.65F, height * 0.34F + fraction},
                         {width * 0.62F + fraction, height * 0.73F},
                         {width * 0.28F, height * 0.68F + fraction}},
                        {{-3.5F, -2.5F},
                         {31.5F + fraction, -0.5F},
                         {30.5F, std::min(height - 0.5F, 25.5F)},
                         {0.5F, 22.5F}},
                        {{width - 29.5F, height - 21.5F},
                         {width + 2.5F, height - 20.5F},
                         {width + 3.5F, height + 1.5F},
                         {width - 28.5F, height + 2.5F}}};
                for (cv::Size canonical_size :
                     {cv::Size(5, 8), cv::Size(32, 32), cv::Size(65, 65)}) {
                    const float last_x = static_cast<float>(canonical_size.width - 1);
                    const float last_y = static_cast<float>(canonical_size.height - 1);
                    const std::vector<cv::Point2f> destination = {
                            {0.0F, 0.0F}, {last_x, 0.0F}, {last_x, last_y}, {0.0F, last_y}};
                    for (const auto& quad : quads) {
                        cv::Mat expected;
                        bool reference_rejected = false;
                        try {
                            cv::warpPerspective(pyramid[static_cast<std::size_t>(level)], expected,
                                                cv::getPerspectiveTransform(quad, destination),
                                                canonical_size, cv::INTER_NEAREST);
                        } catch (const cv::Exception&) {
                            // Large source dimensions can use different installed
                            // HAL paths. Match the actual operation's success or
                            // rejection, rather than imposing a blanket short limit.
                            reference_rejected = true;
                        }
                        for (int tile : {1, 3, 64, 65536}) {
                            SCOPED_TRACE("width=" + std::to_string(size.width) +
                                         " level=" + std::to_string(level) +
                                         " fraction=" + std::to_string(fraction) +
                                         " tile=" + std::to_string(tile) +
                                         " canonical=" + std::to_string(canonical_size.width));
                            sparse.reset(tile);
                            cv::Mat actual;
                            if (reference_rejected) {
                                ASSERT_THROW(sparse.warp(image, sizes, level, quad, canonical_size,
                                                         &actual),
                                             cv::Exception);
                            } else {
                                ASSERT_NO_THROW(sparse.warp(image, sizes, level, quad,
                                                            canonical_size, &actual));
                                ASSERT_EQ(cv::countNonZero(actual != expected), 0);
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST(CpuSparsePyramidTest, monitored_refinement_matches_global_corner_subpix_exactly) {
    SparsePyramid sparse;
    cv::RNG random(264575U);
    for (cv::Size size : {cv::Size(641, 479), cv::Size(1280, 721), cv::Size(4097, 2049)}) {
        cv::Mat image(size, CV_8UC1);
        random.fill(image, cv::RNG::UNIFORM, 0, 256);
        std::vector<cv::Mat> pyramid;
        cv::buildPyramid(image, pyramid, 3);
        const std::vector<cv::Size> sizes = sizes_of(pyramid);
        for (int level = 0; level < 3; ++level) {
            const cv::Size extent = sizes[static_cast<std::size_t>(level)];
            const std::vector<cv::Point2f> initial = {
                    {0.25F, 0.375F},
                    {3.875F, 4.125F},
                    {extent.width * 0.5F + 0.125F, extent.height * 0.5F + 0.375F},
                    {static_cast<float>(extent.width) - 1.25F,
                     static_cast<float>(extent.height) - 1.375F}};
            for (int radius : {1, 3, 5}) {
                for (int iterations : {1, 2, 30, 101}) {
                    for (double epsilon : {0.0, 0.01, 0.1}) {
                        std::vector<cv::Point2f> expected = initial;
                        cv::cornerSubPix(
                                pyramid[static_cast<std::size_t>(level)], expected,
                                cv::Size(radius, radius), cv::Size(-1, -1),
                                cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
                                                 iterations, epsilon));
                        for (int tile : {1, 3, 64, 65536}) {
                            sparse.reset(tile);
                            std::vector<cv::Point2f> actual = initial;
                            sparse.refine(image, sizes, level, &actual, radius, iterations,
                                          epsilon);
                            SCOPED_TRACE("width=" + std::to_string(size.width) +
                                         " level=" + std::to_string(level) +
                                         " radius=" + std::to_string(radius) +
                                         " iterations=" + std::to_string(iterations) + " epsilon=" +
                                         std::to_string(epsilon) + " tile=" + std::to_string(tile));
                            ASSERT_EQ(actual, expected);
                        }
                    }
                }
            }
        }
    }
}

TEST(CpuSparsePyramidTest, changed_frame_warp_footprints_do_not_read_unrequested_stale_pixels) {
    std::array<SparsePyramid, 2> sparse;
    cv::Mat image(479, 641, CV_8UC1);
    std::vector<cv::Mat> pyramid;
    cv::buildPyramid(image, pyramid, 3);
    const auto sizes = sizes_of(pyramid);
    for (std::size_t cache = 0; cache < sparse.size(); ++cache) {
        image.setTo(cache == 0U ? 17 : 239);
        sparse[cache].reset(3, 1.0);
        for (int level = 1; level < 4; ++level) {
            (void)sparse[cache].region(
                    image, sizes, level,
                    cv::Rect(cv::Point(), sizes[static_cast<std::size_t>(level)]));
        }
    }
    cv::RNG random(316227U);
    random.fill(image, cv::RNG::UNIFORM, 0, 256);
    cv::buildPyramid(image, pyramid, 3);
    for (int level = 1; level < 4; ++level) {
        const cv::Size size = sizes[static_cast<std::size_t>(level)];
        const std::vector<cv::Point2f> corners = {
                {size.width * 0.31F + 0.49999997F, size.height * 0.32F},
                {size.width * 0.71F, size.height * 0.37F + 0.50000006F},
                {size.width * 0.68F, size.height * 0.72F},
                {size.width * 0.27F, size.height * 0.67F}};
        const std::vector<cv::Point2f> destination{
                {0.0F, 0.0F}, {32.0F, 0.0F}, {32.0F, 32.0F}, {0.0F, 32.0F}};
        cv::Mat expected;
        cv::warpPerspective(pyramid[static_cast<std::size_t>(level)], expected,
                            cv::getPerspectiveTransform(corners, destination), cv::Size(33, 33),
                            cv::INTER_NEAREST);
        for (std::size_t cache = 0; cache < sparse.size(); ++cache) {
            SCOPED_TRACE("cache=" + std::to_string(cache) + " level=" + std::to_string(level));
            sparse[cache].reset(3, 1.0);
            cv::Mat actual;
            sparse[cache].warp(image, sizes, level, corners, cv::Size(33, 33), &actual);
            ASSERT_EQ(cv::countNonZero(actual != expected), 0);
        }
    }
}

}  // namespace
