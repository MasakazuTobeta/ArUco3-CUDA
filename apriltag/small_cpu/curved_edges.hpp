// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "geometry.hpp"

namespace aruco3cuda::apriltag_small_cpu {
struct LargeTagRefinementConfig;
}

namespace aruco3cuda::apriltag_small_cpu::detail {
// Fits signed boundary crossings on the current native image and rechecks the oriented code.
bool fit_curved_edges(const cv::Mat& gray, const cv::Matx<std::uint8_t, 8, 8>& cells,
                      Quad& corners);
// Proposes intersections of measured native rails. The caller verifies the current payload.
bool fit_long_tag_edges(const cv::Mat& gray, const Quad& corners, Quad& fitted,
                        const LargeTagRefinementConfig& config, bool straight_rails);
}  // namespace aruco3cuda::apriltag_small_cpu::detail
