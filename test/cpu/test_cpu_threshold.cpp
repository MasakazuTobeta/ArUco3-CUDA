// SPDX-License-Identifier: Apache-2.0
#include "cpu_threshold.hpp"

#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace {

using aruco3cuda::cpu::detail::ThresholdWorkspace;

void expect_same_thresholds(ThresholdWorkspace& workspace, const cv::Mat& image,
                            const std::vector<int>& windows, double constant,
                            std::size_t max_prefix_bytes = 2U * 1024U * 1024U) {
    std::vector<cv::Mat> actual;
    workspace.apply(image, windows, constant, &actual, 32768, max_prefix_bytes);
    ASSERT_EQ(actual.size(), windows.size());
    for (std::size_t index = 0; index < windows.size(); ++index) {
        SCOPED_TRACE("window=" + std::to_string(windows[index]) +
                     " constant=" + std::to_string(constant));
        cv::Mat expected;
        const int odd_window = windows[index] | 1;
        cv::adaptiveThreshold(image, expected, 255, cv::ADAPTIVE_THRESH_MEAN_C,
                              cv::THRESH_BINARY_INV, odd_window, constant);
        ASSERT_EQ(actual[index].size(), expected.size());
        ASSERT_EQ(actual[index].type(), expected.type());
        EXPECT_EQ(cv::countNonZero(actual[index] != expected), 0);
    }
}

TEST(CpuThresholdTest, random_images_tiny_images_and_fractional_constants_match_opencv) {
    ThresholdWorkspace workspace;
    cv::RNG random(20261116U);
    const std::vector<int> windows{3, 4, 13, 23, 31};
    for (const cv::Size& size :
         {cv::Size(1, 1), cv::Size(1, 19), cv::Size(29, 1), cv::Size(2, 3), cv::Size(17, 13),
          cv::Size(97, 61), cv::Size(427, 240), cv::Size(641, 479)}) {
        SCOPED_TRACE(size);
        cv::Mat image(size, CV_8UC1);
        for (int repetition = 0; repetition < 3; ++repetition) {
            random.fill(image, cv::RNG::UNIFORM, 0, 256);
            for (double constant :
                 {-255.0, -254.9, -7.1, -1.0, -0.1, 0.0, 0.1, 1.0, 1.1, 7.0, 7.9, 254.9, 255.0}) {
                expect_same_thresholds(workspace, image, windows, constant);
            }
        }
    }
}

TEST(CpuThresholdTest, roi_pixels_outside_the_view_do_not_affect_results) {
    ThresholdWorkspace workspace;
    cv::RNG random(314159U);
    cv::Mat owner(67, 113, CV_8UC1, cv::Scalar(0));
    cv::Mat roi = owner(cv::Rect(7, 5, 91, 47));
    random.fill(roi, cv::RNG::UNIFORM, 0, 256);
    ASSERT_FALSE(roi.isContinuous());
    const cv::Mat cloned = roi.clone();
    for (double constant : {-7.5, 0.0, 7.5}) {
        expect_same_thresholds(workspace, roi, {3, 13, 23, 49}, constant);
        std::vector<cv::Mat> roi_results;
        std::vector<cv::Mat> clone_results;
        workspace.apply(roi, {3, 13, 23, 49}, constant, &roi_results);
        workspace.apply(cloned, {3, 13, 23, 49}, constant, &clone_results);
        for (std::size_t i = 0; i < roi_results.size(); ++i) {
            EXPECT_EQ(cv::countNonZero(roi_results[i] != clone_results[i]), 0);
        }
    }
}

TEST(CpuThresholdTest, uniform_extremes_and_checkerboards_match_opencv) {
    ThresholdWorkspace workspace;
    cv::Mat image(51, 73, CV_8UC1);
    for (int level : {0, 1, 127, 128, 254, 255}) {
        image.setTo(level);
        for (double constant : {-1.1, -0.1, 0.0, 0.1, 1.1}) {
            expect_same_thresholds(workspace, image, {3, 5, 13, 23, 75}, constant);
        }
    }
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            image.at<unsigned char>(y, x) = ((x + y) & 1) == 0 ? 0U : 255U;
        }
    }
    expect_same_thresholds(workspace, image, {3, 13, 23, 75}, 7.0);
}

TEST(CpuThresholdTest, large_windows_and_repeated_shape_changes_match_opencv) {
    ThresholdWorkspace workspace;
    cv::RNG random(161803U);
    for (const cv::Size& size : {cv::Size(27, 19), cv::Size(69, 31), cv::Size(27, 19)}) {
        cv::Mat image(size, CV_8UC1);
        random.fill(image, cv::RNG::UNIFORM, 0, 256);
        expect_same_thresholds(workspace, image, {3, 255, 256, 4096}, 7.0);
    }
}

TEST(CpuThresholdTest, window_sums_next_to_half_integer_means_match_opencv) {
    ThresholdWorkspace workspace;
    // Odd window areas have no exact half-integer mean. These inputs place
    // the sum immediately on both sides of the rounding boundary, including
    // where a float reciprocal can lose the distinction for large windows.
    for (int window = 3; window <= 129; window += 2) {
        const int area = window * window;
        for (int base : {0, 127, 254}) {
            for (int offset : {-1, 0, 1, 2}) {
                cv::Mat image(window, window, CV_8UC1, cv::Scalar(base));
                const int raised_pixels = area / 2 + offset;
                for (int index = 0; index < raised_pixels; ++index) {
                    image.ptr<unsigned char>(index / window)[index % window] =
                            static_cast<unsigned char>(base + 1);
                }
                SCOPED_TRACE("base=" + std::to_string(base) + " offset=" + std::to_string(offset));
                expect_same_thresholds(workspace, image, {window}, 0.0);
            }
        }
    }
}

TEST(CpuThresholdTest, invalid_inputs_fail_explicitly_and_empty_window_set_clears_output) {
    ThresholdWorkspace workspace;
    const cv::Mat image(19, 27, CV_8UC1, cv::Scalar(127));
    std::vector<cv::Mat> outputs{image};
    workspace.apply(image, {}, 7.0, &outputs);
    EXPECT_TRUE(outputs.empty());
    EXPECT_THROW(workspace.apply(cv::Mat{}, {3}, 7.0, &outputs), cv::Exception);
    EXPECT_THROW(workspace.apply(cv::Mat(19, 27, CV_8UC3), {3}, 7.0, &outputs), cv::Exception);
    EXPECT_THROW(workspace.apply(image, {2}, 7.0, &outputs), cv::Exception);
    EXPECT_THROW(workspace.apply(image, {4098}, 7.0, &outputs), cv::Exception);
    EXPECT_THROW(workspace.apply(image, {3}, std::numeric_limits<double>::quiet_NaN(), &outputs),
                 cv::Exception);
    EXPECT_THROW(workspace.apply(image, {3}, 256.0, &outputs), cv::Exception);
    EXPECT_THROW(workspace.apply(image, {3}, 7.0, nullptr), cv::Exception);
    EXPECT_THROW(workspace.apply(image, {3}, 7.0, &outputs, -1), cv::Exception);
}

TEST(CpuThresholdTest, strip_integral_prevents_overflow_on_4k_white_and_random_rois) {
    ThresholdWorkspace workspace;
    const int previous_threads = cv::getNumThreads();
    cv::setNumThreads(4);
    struct RestoreThreads {
        int value_;
        ~RestoreThreads() { cv::setNumThreads(this->value_); }
    } restore{previous_threads};
    // Including the replicated halo makes a single full CV_32S integral
    // overflow on this all-white image, although the unpadded image fits.
    cv::Mat white(2161, 3840, CV_8UC1, cv::Scalar(255));
    for (double constant : {0.0, 7.0}) {
        expect_same_thresholds(workspace, white, {3, 13, 23}, constant);
    }
    cv::Mat owner(2173, 3889, CV_8UC1, cv::Scalar(0));
    cv::Mat roi = owner(cv::Rect(7, 3, 3839, 2161));
    cv::RNG random(271828U);
    random.fill(roi, cv::RNG::UNIFORM, 0, 256);
    for (double constant : {-0.1, 0.0, 7.0}) {
        expect_same_thresholds(workspace, roi, {3, 13, 23}, constant);
    }
}

TEST(CpuThresholdTest, prefix_budgets_preserve_4k_roi_pixels_and_repeated_shape_changes) {
    ThresholdWorkspace workspace;
    const int previous_threads = cv::getNumThreads();
    cv::setNumThreads(4);
    struct RestoreThreads {
        int value_;
        ~RestoreThreads() { cv::setNumThreads(this->value_); }
    } restore{previous_threads};
    const std::vector<std::size_t> budgets = {0U,
                                              1U,
                                              1024U,
                                              64U * 1024U,
                                              2U * 1024U * 1024U,
                                              4U * 1024U * 1024U,
                                              std::numeric_limits<std::size_t>::max()};
    cv::RNG random(20261117U);
    const auto check_budgets = [&](const cv::Mat& image, const std::vector<double>& constants) {
        for (std::size_t budget : budgets) {
            SCOPED_TRACE("budget=" + std::to_string(budget));
            for (double constant : constants) {
                expect_same_thresholds(workspace, image, {3, 13, 23}, constant, budget);
            }
        }
    };
    cv::Mat small(17, 13, CV_8UC1);
    random.fill(small, cv::RNG::UNIFORM, 0, 256);
    check_budgets(small, {-0.1, 7.0});
    cv::Mat owner(2173, 3889, CV_8UC1, cv::Scalar(0));
    cv::Mat roi = owner(cv::Rect(7, 3, 3839, 2161));
    ASSERT_FALSE(roi.isContinuous());
    roi.setTo(255);
    check_budgets(roi, {0.0, 7.0});
    random.fill(roi, cv::RNG::UNIFORM, 0, 256);
    check_budgets(roi, {-0.1, 7.0});
    cv::Mat changed(29, 37, CV_8UC1);
    random.fill(changed, cv::RNG::UNIFORM, 0, 256);
    check_budgets(changed, {-0.1, 7.0});
    check_budgets(small, {-0.1, 7.0});
}

TEST(CpuThresholdTest, budgeted_stripes_preserve_in_place_threshold_results) {
    ThresholdWorkspace workspace;
    cv::RNG random(65537U);
    cv::Mat original(481, 643, CV_8UC1);
    random.fill(original, cv::RNG::UNIFORM, 0, 256);
    for (std::size_t budget :
         {std::size_t{0}, std::size_t{64U * 1024U}, std::size_t{2U * 1024U * 1024U}}) {
        SCOPED_TRACE("budget=" + std::to_string(budget));
        cv::Mat owner(original.rows + 11, original.cols + 29, CV_8UC1, cv::Scalar(0));
        cv::Mat image = owner(cv::Rect(5, 3, original.cols, original.rows));
        original.copyTo(image);
        cv::Mat expected;
        cv::adaptiveThreshold(original, expected, 255.0, cv::ADAPTIVE_THRESH_MEAN_C,
                              cv::THRESH_BINARY_INV, 23, 7.0);
        std::vector<cv::Mat> outputs{image};
        workspace.apply(image, {23}, 7.0, &outputs, 0, budget);
        ASSERT_EQ(outputs.size(), 1U);
        EXPECT_EQ(cv::countNonZero(outputs[0] != expected), 0);
    }
}

}  // namespace
