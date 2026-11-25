// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "tag_decode.hpp"

#include <algorithm>
#include <array>

namespace aruco3cuda::apriltag_small_cpu::detail {
// Read the oriented 36-bit word independently of the reference word. Nine native
// samples per cell reduce read noise. Nearby light and dark levels can follow a
// curved illumination field that a plane fitted to the printed border misses.
inline bool matches_local_payload(const cv::Mat& gray, const Quad& corners, MarkerCode expected,
                                  float minimum_contrast) {
    Projection projection;
    if (!project(corners, projection))
        return false;
    constexpr int side = 8;
    constexpr int payload = 6;
    std::array<std::array<float, side>, side> means{};
    for (int y = 0; y < side; ++y) {
        for (int x = 0; x < side; ++x) {
            float sum = 0;
            for (float dy : {-.2F, 0.F, .2F}) {
                for (float dx : {-.2F, 0.F, .2F}) {
                    const float value =
                            projection.sample(gray, (x + .5F + dx) / side, (y + .5F + dy) / side);
                    if (value < 0)
                        return false;
                    sum += value;
                }
            }
            means[y][x] = sum / 9;
        }
    }
    for (int y = 1; y <= payload; ++y) {
        for (int x = 1; x <= payload; ++x) {
            float low = 255, high = 0;
            for (int yy = y - 1; yy <= y + 1; ++yy) {
                for (int xx = x - 1; xx <= x + 1; ++xx) {
                    low = std::min(low, means[yy][xx]);
                    high = std::max(high, means[yy][xx]);
                }
            }
            if (high - low < minimum_contrast)
                return false;
            const bool white = means[y][x] > .5F * (low + high);
            if (white != ((expected >> (payload * (y - 1) + x - 1)) & 1U))
                return false;
        }
    }
    return true;
}
}  // namespace aruco3cuda::apriltag_small_cpu::detail
