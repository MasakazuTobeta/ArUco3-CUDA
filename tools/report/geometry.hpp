// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_TOOLS_REPORT_GEOMETRY_HPP
#define ARUCO3CUDA_TOOLS_REPORT_GEOMETRY_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <utility>

/// Geometry helpers for matching detections that carry four corners.
///
/// Purpose:
///     Classifying differences against the baseline (report_diff) and matching
///     against the ground truth (evaluate) must pair detections by the same rule. If
///     the rule changes on only one side, the two reports disagree and there is no
///     way to tell which one is right. They are kept in one shared place.
namespace aruco3cuda::report {

/// Corner ordering: x0, y0, x1, y1, x2, y2, x3, y3.
using Quad = std::array<double, 8>;

/// Number of corners.
constexpr std::size_t kQuadCorners = 4U;

/// Returns the centroid of the four corners.
///
/// @param quad The four corners. Only read, never retained.
/// @return An (x, y) pair in pixels.
///
/// Ownership: the argument is not retained. The result is returned by value.
/// Synchronization: none. Reentrant.
///
/// Example input: {0,0, 2,0, 2,2, 0,2}
/// Example output: (1.0, 1.0)
inline std::pair<double, double> quad_centroid(const Quad& quad) {
    double x = 0.0;
    double y = 0.0;
    for (std::size_t c = 0; c < kQuadCorners; ++c) {
        x += quad[c * 2U];
        y += quad[(c * 2U) + 1U];
    }
    return {x / static_cast<double>(kQuadCorners), y / static_cast<double>(kQuadCorners)};
}

/// Returns the mean side length of the quadrilateral formed by the four corners.
///
/// Used so that the matching radius is a ratio of the side length rather than an
/// absolute value, because the reasonable distance changes with the resolution and
/// the size of the marker.
///
/// @param quad The four corners. Only read, never retained.
/// @return The mean side length in pixels.
///
/// Ownership: the argument is not retained.
/// Synchronization: none. Reentrant.
///
/// Example input: {0,0, 2,0, 2,2, 0,2}
/// Example output: 2.0
inline double quad_average_side(const Quad& quad) {
    double total = 0.0;
    for (std::size_t c = 0; c < kQuadCorners; ++c) {
        const std::size_t next = (c + 1U) % kQuadCorners;
        const double dx = quad[c * 2U] - quad[next * 2U];
        const double dy = quad[(c * 2U) + 1U] - quad[(next * 2U) + 1U];
        total += std::sqrt((dx * dx) + (dy * dy));
    }
    return total / static_cast<double>(kQuadCorners);
}

/// Returns the largest distance between corresponding corners after rotating the
/// target corners by steps positions.
///
/// @param baseline The baseline corners. Only read, never retained.
/// @param target The target corners. Only read, never retained.
/// @param steps Number of rotation steps, from 0 to 3.
/// @return The maximum of the four corner distances, in pixels.
///
/// Ownership: the arguments are not retained.
/// Synchronization: none. Reentrant.
///
/// Example input: two identical corner sets, steps = 0
/// Example output: 0.0
inline double quad_corner_error(const Quad& baseline, const Quad& target, int steps) {
    double worst = 0.0;
    for (std::size_t c = 0; c < kQuadCorners; ++c) {
        const std::size_t shifted = (c + static_cast<std::size_t>(steps)) % kQuadCorners;
        const double dx = baseline[c * 2U] - target[shifted * 2U];
        const double dy = baseline[(c * 2U) + 1U] - target[(shifted * 2U) + 1U];
        worst = std::max(worst, std::sqrt((dx * dx) + (dy * dy)));
    }
    return worst;
}

/// Returns the sum of squared distances between corresponding corners after
/// rotating the target corners by steps positions.
///
/// The RMSE carries information the maximum does not. The sum of squares is exposed
/// as well so that one badly displaced corner can be told apart from four corners
/// displaced uniformly.
///
/// @param baseline The baseline corners. Only read, never retained.
/// @param target The target corners. Only read, never retained.
/// @param steps Number of rotation steps, from 0 to 3.
/// @return The sum of the squared corner distances, in pixel^2.
///
/// Ownership: the arguments are not retained.
/// Synchronization: none. Reentrant.
///
/// Example input: corners with a single corner displaced by 2 pixels, steps = 0
/// Example output: 4.0
inline double quad_squared_error(const Quad& baseline, const Quad& target, int steps) {
    double total = 0.0;
    for (std::size_t c = 0; c < kQuadCorners; ++c) {
        const std::size_t shifted = (c + static_cast<std::size_t>(steps)) % kQuadCorners;
        const double dx = baseline[c * 2U] - target[shifted * 2U];
        const double dy = baseline[(c * 2U) + 1U] - target[(shifted * 2U) + 1U];
        total += (dx * dx) + (dy * dy);
    }
    return total;
}

}  // namespace aruco3cuda::report

#endif  // ARUCO3CUDA_TOOLS_REPORT_GEOMETRY_HPP
