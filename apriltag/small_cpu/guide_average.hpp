// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <opencv2/core.hpp>

namespace aruco3cuda::apriltag_small_cpu::detail {

// Average nonoverlapping 4×4 blocks of a Gray8 image.
// An incomplete block uses the visible pixels of the input view.
// The caller supplies separate input and output storage, including for strided views.
void average_gray4x4(const cv::Mat& gray, cv::Mat& guide);

enum class QuarterGuideSample { nearest, central_area };

// Attempt to write both full 4x4 area means and the requested phase from the same Gray8 view.
// Input and outputs have separate storage. An unsupported shape or CPU leaves outputs untouched.
bool try_sample_gray4x4(const cv::Mat& gray, cv::Mat& area, cv::Mat& guide,
                        QuarterGuideSample sample);

}  // namespace aruco3cuda::apriltag_small_cpu::detail
