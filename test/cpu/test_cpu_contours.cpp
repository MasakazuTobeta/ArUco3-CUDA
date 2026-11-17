// SPDX-License-Identifier: Apache-2.0
#include "cpu_contours.hpp"

#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "aruco3cuda/config.hpp"
#include "cpu_candidates.hpp"

namespace {

using Quads = std::vector<std::vector<cv::Point2f>>;
using Contours = std::vector<std::vector<cv::Point>>;
using aruco3cuda::DetectorConfig;
using aruco3cuda::cpu::detail::ContourWorkspace;

class CpuContoursTest : public ::testing::Test {
protected:
    void SetUp() override {
        this->previous_threads_ = cv::getNumThreads();
        cv::setNumThreads(1);
    }
    void TearDown() override { cv::setNumThreads(this->previous_threads_); }

private:
    int previous_threads_ = 1;
};

void expect_same_candidates(ContourWorkspace& workspace, const cv::Mat& image,
                            const DetectorConfig& config, Quads* actual, Contours* paths) {
    Quads expected = *actual;
    Contours expected_paths = *paths;
    const cv::Mat original = image.clone();
    // The pinned OpenCV TRU implementation requires an isolated binary image;
    // feeding a raw ROI can read its parent's nonzero border and is not a
    // reliable oracle. The custom workspace must isolate the supplied ROI.
    aruco3cuda::hybrid::find_quad_candidates(original, config, expected, expected_paths);
    workspace.append(image, config, *actual, *paths);
    ASSERT_EQ(*actual, expected);
    ASSERT_EQ(*paths, expected_paths);
    ASSERT_EQ(cv::countNonZero(image != original), 0);
}

TEST_F(CpuContoursTest, reused_scratch_preserves_exact_paths_on_strided_random_binary_inputs) {
    ContourWorkspace workspace;
    cv::RNG random(2026111703U);
    for (cv::Size size : {cv::Size(1, 1), cv::Size(2, 3), cv::Size(31, 17), cv::Size(127, 65),
                          cv::Size(321, 239)}) {
        cv::Mat owner(size.height + 9, size.width + 31, CV_8UC1, cv::Scalar(255));
        cv::Mat image = owner(cv::Rect(5, 3, size.width, size.height));
        for (int sample = 0; sample < 12; ++sample) {
            random.fill(image, cv::RNG::UNIFORM, 0, 100);
            cv::threshold(image, image, (sample * 13) % 100,
                          sample % 3 == 0 ? 1.0 : (sample % 3 == 1 ? 128.0 : 255.0),
                          cv::THRESH_BINARY);
            for (bool aruco3 : {false, true}) {
                DetectorConfig config;
                config.use_aruco3_detection_ = aruco3;
                config.min_side_length_canonical_img_px_ = 1 + sample % 9;
                config.min_marker_perimeter_rate_ = sample % 2 == 0 ? 0.001 : 0.03;
                config.max_marker_perimeter_rate_ = sample % 3 == 0 ? 1.5 : 4.0;
                config.polygonal_approx_accuracy_rate_ = sample % 2 == 0 ? 0.005 : 0.03;
                config.min_corner_distance_rate_ = sample % 3 == 0 ? 0.0 : 0.05;
                SCOPED_TRACE("size=" + std::to_string(size.width) + "x" +
                             std::to_string(size.height) + " sample=" + std::to_string(sample) +
                             " aruco3=" + std::to_string(aruco3));
                Quads candidates;
                Contours paths;
                expect_same_candidates(workspace, image, config, &candidates, &paths);
                ASSERT_FALSE(::testing::Test::HasFailure());
            }
        }
    }
}

TEST_F(CpuContoursTest, holes_diagonal_contacts_and_append_order_match_serial_opencv) {
    ContourWorkspace workspace;
    cv::Mat image(239, 321, CV_8UC1, cv::Scalar(0));
    for (int inset : {7, 23, 41, 61, 83}) {
        cv::rectangle(image, cv::Point(inset, inset),
                      cv::Point(image.cols - 1 - inset, image.rows - 1 - inset), cv::Scalar(255),
                      1 + inset % 3);
    }
    cv::rectangle(image, cv::Rect(15, 15, 16, 16), cv::Scalar(255), cv::FILLED);
    cv::rectangle(image, cv::Rect(31, 31, 16, 16), cv::Scalar(255), cv::FILLED);
    cv::line(image, cv::Point(4, 201), cv::Point(200, 5), cv::Scalar(255), 1);
    DetectorConfig config;
    config.use_aruco3_detection_ = false;
    config.min_marker_perimeter_rate_ = 0.001;
    Quads candidates{{cv::Point2f(1.0F, 2.0F), cv::Point2f(3.0F, 4.0F), cv::Point2f(5.0F, 6.0F),
                      cv::Point2f(7.0F, 8.0F)}};
    Contours paths{{cv::Point(17, 23)}};
    for (int repeat = 0; repeat < 4; ++repeat) {
        expect_same_candidates(workspace, image, config, &candidates, &paths);
        ASSERT_FALSE(::testing::Test::HasFailure());
        cv::bitwise_not(image, image);
    }
    EXPECT_EQ(candidates.front().front(), cv::Point2f(1.0F, 2.0F));
    EXPECT_EQ(paths.front(), std::vector<cv::Point>({cv::Point(17, 23)}));
}

TEST_F(CpuContoursTest, invalid_images_fail_before_scratch_access) {
    ContourWorkspace workspace;
    DetectorConfig config;
    Quads candidates;
    Contours paths;
    EXPECT_THROW(workspace.append(cv::Mat(), config, candidates, paths), cv::Exception);
    EXPECT_THROW(workspace.append(cv::Mat(17, 19, CV_16UC1), config, candidates, paths),
                 cv::Exception);
    cv::Mat oversized(1, 1, CV_8UC1);
    oversized.rows = 65536;
    oversized.cols = 65536;
    oversized.step[0] = 65536U;
    EXPECT_THROW(workspace.append(oversized, config, candidates, paths), cv::Exception);
}

}  // namespace
