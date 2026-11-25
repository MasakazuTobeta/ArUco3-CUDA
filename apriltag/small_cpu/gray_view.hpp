// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <limits>
#include <opencv2/core.hpp>
#include <stdexcept>

namespace aruco3cuda::apriltag_small_cpu::detail {

inline void validate_image(const cv::Mat& gray) {
    if (gray.dims != 2 || gray.type() != CV_8UC1 || !gray.data || gray.cols < 1 || gray.rows < 1 ||
        gray.cols >= 32768 || gray.rows >= 32768 || gray.step[1] != 1 ||
        gray.step[0] < static_cast<std::size_t>(gray.cols) ||
        gray.step[0] > static_cast<std::size_t>(std::numeric_limits<int>::max() / gray.rows)) {
        throw std::invalid_argument("Invalid small AprilTag Gray8 dimensions or stride");
    }
    const auto first = reinterpret_cast<std::uintptr_t>(gray.data);
    const auto end = reinterpret_cast<std::uintptr_t>(gray.dataend);
    const std::size_t visible_bytes =
            static_cast<std::size_t>(gray.rows - 1) * gray.step[0] + gray.cols;
    if (end < first || end - first < visible_bytes) {
        throw std::invalid_argument("Small AprilTag image storage does not cover visible pixels");
    }
}

}  // namespace aruco3cuda::apriltag_small_cpu::detail
