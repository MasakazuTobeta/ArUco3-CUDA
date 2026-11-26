// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "aruco3cuda/apriltag/small_cpu/detector.hpp"

#include <array>
#include <cstdint>

namespace aruco3cuda::apriltag_small_cpu::detail {

/// Trace dark boundaries meeting a seed rectangle on the native stride-two sampling grid.
class EnteringContours {
public:
    /// Seed and returned points use sampled coordinates; samples at or below threshold are dark.
    const std::vector<std::vector<cv::Point>>& find(const cv::Mat& gray, int threshold,
                                                    cv::Rect seed);

private:
    struct Grid;
    void trace(const Grid& grid, cv::Point start, int background);
    std::vector<std::uint8_t> visited_;
    std::vector<cv::Point> chain_;
    std::vector<std::vector<cv::Point>> contours_;
};

/// Searches leading image edges where camera translation can reveal a stationary tag.
class EnteringStripSearch {
public:
    /// A zero width disables discovery. The mask uses a Gray8 inverse threshold.
    explicit EnteringStripSearch(int width_px = 288, int threshold = 120,
                                 int minimum_contrast = 28);
    /// Returns decoded proposals in frame coordinates. The caller merges overlapping proposals.
    std::vector<Detection> detect(const cv::Mat& gray, cv::Point translation);

private:
    struct ExposureRange {
        int first;
        int end;
        int threshold;
    };
    void exposure_ranges(const cv::Mat& gray, int shift, bool vertical);
    void append(const cv::Mat& gray, int shift, bool vertical, int threshold, cv::Point offset,
                std::vector<Detection>& result);
    int width_px_;
    int minimum_contrast_;
    std::vector<ExposureRange> ranges_;
    Config config_;
    Detector decoder_;
    EnteringContours contours_;
};

}  // namespace aruco3cuda::apriltag_small_cpu::detail
