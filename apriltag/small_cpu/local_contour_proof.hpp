// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "guide_contours.hpp"

#include <opencv2/core/hal/intrin.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>

namespace aruco3cuda::apriltag_small_cpu::detail {
#if CV_SIMD128
// A retained eight-connected contour must cross three complete rows in at
// least one sampled band. Generate the exact local classifier for its central
// row, then inspect only neighbours of its positive runs on adjacent rows.
inline bool may_have_local_contour15(const cv::Mat& guide, int difference) {
    constexpr int radius = 7;
    constexpr int side = 2 * radius + 1;
    constexpr int area = side * side;
    constexpr int period = kMinimumGuideContourSide - 2;
    static_assert(period >= 2 && period <= side);
    const int width = guide.cols;
    const int height = guide.rows;
    if (width < 16 || width % 16 || height < side || difference < 1 ||
        difference > std::numeric_limits<std::uint16_t>::max() / area - 255)
        return true;

    // Each value is the exact replicated-border vertical sum for one guide
    // column. Keep a padded row for the horizontal box recurrence.
    cv::AutoBuffer<std::uint16_t, 1024> storage(2 * width + 2 * radius);
    auto* columns = storage.data() + radius;
    auto* boxes = columns + width + radius;
    cv::AutoBuffer<std::uint8_t, 512> foreground(width);
    const auto row = [&](int y) { return guide.ptr<std::uint8_t>(std::clamp(y, 0, height - 1)); };
    const auto area_lanes = cv::v_setall_u16(area);
    const auto cutoff = cv::v_setall_u16(area * difference - area / 2);
    const auto horizontal_sum = [&](const std::uint8_t* samples, int x) {
        if (x >= radius && x + radius + 1 < width)
            return static_cast<int>(cv::v_reduce_sum(cv::v_load(samples + x - radius))) -
                   samples[x + radius + 1];
        int sum = 0;
        for (int offset = -radius; offset <= radius; ++offset)
            sum += samples[std::clamp(x + offset, 0, width - 1)];
        return sum;
    };
    // Uncertainty costs two complete source-row reductions per neighbour.
    // After the equivalent of two guide rows, let the regular sliding box and
    // contour scan decide. This cap changes only how the same search is executed.
    int remaining_neighbours = 2 * width;
    for (int y = period - 1; y + 1 < height; y += period) {
        std::array<const std::uint8_t*, side> entering;
        std::array<const std::uint8_t*, period> leaving;
        const bool initial = y == period - 1;
        const int count = initial ? side : period;
        for (int i = 0; i < count; ++i)
            entering[i] = row(initial ? y - radius + i : y + radius - period + 1 + i);
        if (!initial) {
            for (int i = 0; i < period; ++i)
                leaving[i] = row(y - period - radius + i);
        }
        for (int x = 0; x < width; x += 8) {
            auto sums = initial ? cv::v_setzero_u16() : cv::v_load(columns + x);
            for (int i = 0; i < count; ++i)
                sums = cv::v_add_wrap(sums, cv::v_load_expand(entering[i] + x));
            if (!initial) {
                for (int i = 0; i < period; ++i)
                    sums = cv::v_sub_wrap(sums, cv::v_load_expand(leaving[i] + x));
            }
            cv::v_store(columns + x, sums);
        }
        std::fill_n(columns - radius, radius, columns[0]);
        std::fill_n(columns + width, radius, columns[width - 1]);
        int total = 0;
        for (int j = -radius; j <= radius; ++j)
            total += columns[j];
        boxes[0] = total;
        for (int x = 1; x < 8; ++x) {
            total += int(columns[x + radius]) - columns[x - radius - 1];
            boxes[x] = total;
        }
        const auto* center = row(y);
        for (int x = 0; x < width; x += 8) {
            auto sums = cv::v_load(boxes);
            if (x) {
                auto delta = cv::v_sub_wrap(cv::v_load(columns + x + radius),
                                            cv::v_load(columns + x - radius - 1));
                delta = cv::v_add_wrap(delta, cv::v_rotate_left<1>(delta));
                delta = cv::v_add_wrap(delta, cv::v_rotate_left<2>(delta));
                delta = cv::v_add_wrap(delta, cv::v_rotate_left<4>(delta));
                sums = cv::v_add_wrap(cv::v_setall_u16(total), delta);
                total = cv::v_extract_n<7>(sums);
                cv::v_store(boxes + x, sums);
            }
            const auto limits = cv::v_add_wrap(
                    cv::v_mul_wrap(cv::v_load_expand(center + x), area_lanes), cutoff);
            const auto pixels = cv::v_ge(sums, limits);
            cv::v_pack_store(foreground.data() + x, pixels);
        }
        const auto touches = [&](int first, int end, int dy) {
            const auto* samples = row(y + dy);
            const auto* add = row(dy > 0 ? y + radius + 1 : y - radius - 1);
            const auto* remove = row(dy > 0 ? y - radius : y + radius);
            for (int x = first; x < end; ++x) {
                const int shifted = boxes[x] + horizontal_sum(add, x) - horizontal_sum(remove, x);
                if (shifted >= area * (samples[x] + difference) - area / 2)
                    return true;
            }
            return false;
        };
        int x = 0;
        while (x < width) {
            while (width - x >= 8) {
                std::uint64_t word;
                std::memcpy(&word, foreground.data() + x, sizeof(word));
                if (word)
                    break;
                x += 8;
            }
            while (x < width && !foreground[x])
                ++x;
            const int first = x;
            while (x < width && foreground[x])
                ++x;
            if (first == x)
                break;
            const int left = std::max(0, first - 1);
            const int right = std::min(width, x + 1);
            remaining_neighbours -= 2 * (right - left);
            if (remaining_neighbours < 0)
                return true;
            if (touches(left, right, -1) && touches(left, right, 1))
                return true;
        }
    }
    return false;
}
#endif
}  // namespace aruco3cuda::apriltag_small_cpu::detail
