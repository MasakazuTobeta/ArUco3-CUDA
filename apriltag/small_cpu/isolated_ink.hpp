// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cpu_contour_scan.hpp"
#include "outline_hull.hpp"

#include "aruco3cuda/apriltag/small_cpu/video_detector.hpp"

namespace aruco3cuda::apriltag_small_cpu::detail {

// Refines the returned corners only after checking the same native payload and rotation.
void refine_diffuse_tags(const cv::Mat& gray, int minimum_contrast, std::vector<Detection>& tags,
                         float maximum_reseed_cells);

enum class FaintGuide { exterior, adaptive_fallback, sparse };

// Finds closed regions on a smooth surround, then verifies their payload in the source image.
// An ambiguous region requests the ordinary full-image search.
class IsolatedInk {
public:
    // Adaptive proposals may supplement ordinary detections from this same frame.
    std::vector<Detection> find_faint(const cv::Mat& gray, const VideoConfig& settings,
                                      FaintGuide mode = FaintGuide::exterior,
                                      const std::vector<Detection>* explained = nullptr,
                                      bool* used_local_mean = nullptr,
                                      cv::Mat* shared_area = nullptr);
    bool quiet_outside(const cv::Mat& gray, const std::vector<Detection>& tags, int maximum_step);
    bool find(const cv::Mat& gray, int low, const VideoConfig& settings, Detector& native,
              Detector& patch, std::vector<Detection>& tags);

private:
    aruco3cuda::cpu::detail::ContourScanner guide_contours_;
    OutlineHull outline_hull_;
    cv::Mat guide_, mask_, variation_;
    std::vector<std::uint8_t> obscured_columns_;
};
}  // namespace aruco3cuda::apriltag_small_cpu::detail
