// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>

namespace aruco3cuda::cpu::detail {
/// Find grayscale regions around dark runs on physical image edges.
/// @param gray Gray8 image. Returned coordinates are relative to this view.
/// @param max_pixels Maximum combined area of all proposed rectangles in pixels.
/// @return Rectangles inside gray, in processing order.
/// Ownership: borrows gray for the call and returns an owning vector.
/// Synchronization: synchronous host operation.
/// Example input: a uniform 640x480 grayscale frame and a 64000-pixel budget.
/// Example output: an empty vector because there are no contrasting edge runs.
std::vector<cv::Rect> edge_regions(const cv::Mat& gray, std::size_t max_pixels);
}  // namespace aruco3cuda::cpu::detail
