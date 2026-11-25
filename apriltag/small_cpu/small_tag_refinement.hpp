// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "aruco3cuda/apriltag/small_cpu/video_detector.hpp"

namespace aruco3cuda::apriltag_small_cpu::detail {

/// Remeasure published corners independently of the video search history.
void refine_small_tag_corners(const cv::Mat& gray, std::vector<Detection>& tags,
                              const SmallTagRefinementConfig& config);

}  // namespace aruco3cuda::apriltag_small_cpu::detail
