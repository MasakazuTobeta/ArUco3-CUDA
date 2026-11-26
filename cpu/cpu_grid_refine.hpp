// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace aruco3cuda::cpu::detail {
// Fit a small projective correction to the visible transitions of a decoded tag36h11.
// Reads original pixels, including on a strided ROI. On failure the corners are unchanged.
bool fit_decoded_grid(const cv::Mat& gray, const cv::Mat& cells, int radius_limit_px,
                      std::vector<cv::Point2f>& corners);
// Fit a decoded tag36h11 whose outer edges have already been aligned. Each visible
// grid transition contributes a midpoint sample. Failure leaves the input corners intact.
bool fit_decoded_payload_grid(const cv::Mat& gray, const cv::Mat& cells, int radius_limit_px,
                              std::vector<cv::Point2f>& corners);
// Fits source-supported grid transitions when part of the decoded tag is off image.
bool fit_visible_payload_grid(const cv::Mat& gray, const cv::Mat& cells, int radius_limit_px,
                              std::vector<cv::Point2f>& corners);
// Validate a near-border observation from the real image, including the ring and oriented payload.
bool visible_grid_matches(const cv::Mat& gray, const cv::Mat& cells,
                          const std::vector<cv::Point2f>& corners, int correction_budget,
                          int border_budget);
}  // namespace aruco3cuda::cpu::detail
