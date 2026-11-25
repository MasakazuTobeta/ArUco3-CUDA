// SPDX-License-Identifier: BSD-3-Clause
// Closed split ordering and cleanup follow OpenCV approx.cpp at
// 0654a42e19215ef25b1d367d822f3c630447e7c7, through the existing supplemental
// closed-hull helper. See LICENSE.opencv-approx.
#pragma once

#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace aruco3cuda::apriltag_small_cpu::detail {

// For integer paths of this extent, all squared geometric distances used in
// the pinned double-precision approximation are exactly representable.
constexpr int kClosedPathMaximumExtent = 4096;
constexpr int kClosedPathMaximumVertices = 64;

template <int Minimum, int Maximum>
inline bool approximate_closed_bounded(const std::vector<cv::Point>& contour, double epsilon,
                                       std::vector<cv::Point>& corners) {
    const auto dot = [](cv::Point a, cv::Point b) { return a.x * b.x + a.y * b.y; };
    const auto squared = [&](cv::Point a) { return dot(a, a); };
    const auto cross = [](cv::Point a, cv::Point b) { return a.x * b.y - a.y * b.x; };
    const auto wrap = [](int index, int count) { return index < count ? index : index - count; };
    const int count = static_cast<int>(contour.size());
    int origin = 0, opposite = 0;
    int maximum_distance = 0;
    const double threshold = epsilon * epsilon;
    for (int round = 0; round < 3; ++round) {
        origin = opposite;
        maximum_distance = 0;
        for (int offset = 1; offset < count; ++offset) {
            const int at = wrap(origin + offset, count);
            const int distance = squared(contour[at] - contour[origin]);
            if (distance > maximum_distance) {
                maximum_distance = distance;
                opposite = at;
            }
        }
    }
    if (maximum_distance <= threshold)
        return false;
    std::array<cv::Point, kClosedPathMaximumVertices + 1> ordered;
    std::copy(contour.begin() + origin, contour.end(), ordered.begin());
    std::copy(contour.begin(), contour.begin() + origin + 1, ordered.begin() + count - origin);
    const int split = opposite > origin ? opposite - origin : opposite - origin + count;
    struct Arc {
        int begin;
        int end;
    };
    std::array<Arc, kClosedPathMaximumVertices> arcs;
    int pending = 0;
    arcs[pending++] = {split, count};
    arcs[pending++] = {0, split};
    std::array<cv::Point, 2 * Maximum + 2> output;
    int vertices = 0;
    while (pending > 0) {
        const auto arc = arcs[--pending];
        const auto first = ordered[arc.begin];
        const auto last = ordered[arc.end];
        const auto direction = last - first;
        const int span = squared(direction);
        std::int64_t maximum = 0;
        int pivot = arc.begin;
        for (int i = arc.begin + 1; i < arc.end; ++i) {
            const auto offset = ordered[i] - first;
            const int projection = dot(offset, direction);
            std::int64_t metric;
            if (projection < 0) {
                metric = std::int64_t{squared(offset)} * span;
            } else if (projection > span) {
                metric = std::int64_t{squared(ordered[i] - last)} * span;
            } else {
                const std::int64_t normal = cross(direction, offset);
                metric = normal * normal;
            }
            if (metric > maximum) {
                maximum = metric;
                pivot = i;
            }
        }
        if (static_cast<double>(maximum) > threshold * span) {
            arcs[pending++] = {pivot, arc.end};
            arcs[pending++] = {arc.begin, pivot};
        } else {
            output[vertices++] = first;
            // Cleanup can remove at most one point per two input positions.
            if (vertices == static_cast<int>(output.size()))
                return false;
        }
    }
    if (vertices < Minimum)
        return false;
    int remaining = vertices;
    auto previous = output[vertices - 1];
    auto middle = output[0];
    int write = 0, read = 1;
    for (int visited = 0; visited < vertices && remaining > 2; ++visited) {
        const auto next = output[read];
        read = wrap(read + 1, vertices);
        const auto chord = next - previous;
        const double normal = cross(middle - previous, chord);
        if (chord.x != 0 && chord.y != 0 && normal * normal <= .5 * threshold * squared(chord) &&
            dot(middle - previous, next - middle) >= 0) {
            --remaining;
            previous = next;
            middle = output[read];
            read = wrap(read + 1, vertices);
            ++visited;
        } else {
            previous = middle;
            middle = next;
        }
        output[write] = previous;
        write = wrap(write + 1, vertices);
    }
    if (remaining < Minimum || remaining > Maximum)
        return false;
    corners.assign(output.begin(), output.begin() + remaining);
    return true;
}

inline bool approximate_closed_quad(const std::vector<cv::Point>& contour, double epsilon,
                                    std::vector<cv::Point>& corners) {
    return approximate_closed_bounded<4, 4>(contour, epsilon, corners);
}

}  // namespace aruco3cuda::apriltag_small_cpu::detail
