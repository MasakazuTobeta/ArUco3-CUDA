// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <opencv2/core.hpp>
namespace aruco3cuda::cpu::detail {
/// Average the central 2x2 samples of each cell, replicating partial image edges.
/// @param image Nonempty Gray8 source image, including a strided region.
/// @param output Destination Gray8 matrix, allocated or reused by this call.
/// @param size Positive output dimensions in sample cells.
/// @param step Cell spacing in source pixels, from 2 through 8.
/// Ownership: the caller owns both matrices and provides nonoverlapping storage.
/// Synchronization: synchronous host operation.
/// Example input: a constant 12x12 Gray8 image, size 2x2 and step 6.
/// Example output: a 2x2 Gray8 matrix of the same intensity.
void sample_box_grid(const cv::Mat& image, cv::Mat& output, cv::Size size, int step);
}  // namespace aruco3cuda::cpu::detail
