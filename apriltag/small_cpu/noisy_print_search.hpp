// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>

#include "aruco3cuda/apriltag/small_cpu/video_detector.hpp"
#include "cpu_contour_scan.hpp"

namespace aruco3cuda::apriltag_small_cpu::detail {

// Adds current-frame quadrilateral candidates measured on a noise-reduced image.
// Only source-supported, oriented tag36h11 words reach the caller.
class NoisyPrintSearch {
public:
    explicit NoisyPrintSearch(const NoisyPrintConfig& config);
    bool has_native_noise(const cv::Mat& gray) const;
    // Reuse area4 when it already contains all 4x4 block means of the current frame.
    // A supplied noise decision comes from has_native_noise on the same frame.
    void append(const cv::Mat& gray, const VideoConfig& video, std::vector<Detection>& tags,
                const cv::Mat& area4 = {}, std::optional<bool> known_noise = std::nullopt);

private:
    NoisyPrintConfig config_;
    Detector decoder_;
    cv::Mat guide_, local_maximum_, mask_, kernel_;
    aruco3cuda::cpu::detail::ContourScanner contour_scanner_;
    std::vector<cv::Point> polygon_;
    std::vector<std::vector<cv::Point>> contours_, candidates_;
};

}  // namespace aruco3cuda::apriltag_small_cpu::detail
