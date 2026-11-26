// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <cmath>
#include <cstdint>

namespace aruco3cuda::apriltag_small_cpu::detail {

using Point = cv::Point2f;
using Quad = std::array<Point, 4>;

inline float length(Point value) {
    return std::sqrt(value.dot(value));
}
inline float cross(Point a, Point b) {
    return a.x * b.y - a.y * b.x;
}

inline float bilinear(const cv::Mat& image, Point point) {
    if (!(point.x >= 0 && point.y >= 0 && point.x < image.cols - 1 && point.y < image.rows - 1)) {
        return -1;
    }
    const int x = static_cast<int>(point.x);
    const int y = static_cast<int>(point.y);
    const float u = point.x - x;
    const float v = point.y - y;
    const auto* top = image.ptr<std::uint8_t>(y) + x;
    const auto* bottom = image.ptr<std::uint8_t>(y + 1) + x;
    return (top[0] + u * (top[1] - top[0])) * (1 - v) +
           (bottom[0] + u * (bottom[1] - bottom[0])) * v;
}

// Updates the ordered quad when all four refined lines meet within the search bounds.
bool refine_quad(const cv::Mat& gray, Quad& corners, bool uniform_surround, int coarse_stride = 0);

}  // namespace aruco3cuda::apriltag_small_cpu::detail
