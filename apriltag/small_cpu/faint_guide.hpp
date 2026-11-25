// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace aruco3cuda::apriltag_small_cpu::detail {
// A foreground point with eight background neighbours with a clear ray to the image border
// contributes its own external contour. This can prove the count exceeds the
// configured cap without constructing thousands of one-pixel contours.
inline bool exceeds_external_limit(const cv::Mat& mask, std::size_t limit,
                                   std::vector<std::uint8_t>& blocked) {
    if (mask.cols < 3 || mask.rows < 3)
        return false;
    blocked.assign(mask.cols, 0);
    std::size_t witnesses = 0;
    for (int y = 1; y < std::min(40, mask.rows - 1); ++y) {
        const auto* above = mask.ptr<std::uint8_t>(y - 1);
        const auto* row = mask.ptr<std::uint8_t>(y);
        const auto* below = mask.ptr<std::uint8_t>(y + 1);
        int x = 1;
#if CV_SIMD128
        const auto zero = cv::v_setzero_u8();
        const auto one = cv::v_setall_u8(1);
        for (; x + 16 < mask.cols; x += 16) {
            const auto current = cv::v_load(row + x);
            const auto prev = cv::v_load(above + x);
            const auto obscured = cv::v_or(cv::v_load(blocked.data() + x), prev);
            cv::v_store(blocked.data() + x, obscured);
            if (!cv::v_check_any(current))
                continue;
            const auto upper = cv::v_or(cv::v_load(above + x - 1), cv::v_load(above + x + 1));
            const auto middle = cv::v_or(cv::v_load(row + x - 1), cv::v_load(row + x + 1));
            const auto lower =
                    cv::v_or(cv::v_or(cv::v_load(below + x - 1), cv::v_load(below + x + 1)),
                             cv::v_load(below + x));
            const auto neighbours = cv::v_or(cv::v_or(upper, middle), cv::v_or(lower, obscured));
            const auto isolated = cv::v_and(cv::v_ne(current, zero), cv::v_eq(neighbours, zero));
            witnesses += cv::v_reduce_sum(cv::v_and(isolated, one));
            if (witnesses > limit)
                return true;
        }
#endif
        for (; x < mask.cols - 1; ++x) {
            blocked[x] |= above[x];
            if (row[x] && !blocked[x] &&
                !(above[x - 1] | above[x + 1] | row[x - 1] | row[x + 1] | below[x - 1] | below[x] |
                  below[x + 1]) &&
                ++witnesses > limit)
                return true;
        }
    }
    return false;
}

#if CV_SIMD128
// The integer box sum rounded to the nearest byte passes the threshold iff
// sum >= area*(source+cutoff) - floor(area/2). Default 15x15 sums fit in 16 bits.
constexpr int kLocalWindow = 15;
constexpr int kLocalArea = kLocalWindow * kLocalWindow;

inline void local_ink_mask_15(const cv::Mat& guide, int difference, cv::Mat& columns,
                              cv::Mat& mask) {
    constexpr int kRadius = kLocalWindow / 2;
    const int width = guide.cols, height = guide.rows;
    columns.create(1, width + 2 * kRadius, CV_16U);
    mask.create(guide.size(), CV_8U);
    auto* sums = columns.ptr<std::uint16_t>();
    std::fill_n(sums + kRadius, width, 0);
    for (int row = -kRadius; row <= kRadius; ++row) {
        const auto* samples = guide.ptr<std::uint8_t>(std::clamp(row, 0, height - 1));
        int x = 0;
        for (; x + 8 <= width; x += 8) {
            auto old = cv::v_load(sums + kRadius + x);
            cv::v_store(sums + kRadius + x, cv::v_add_wrap(old, cv::v_load_expand(samples + x)));
        }
        for (; x < width; ++x)
            sums[kRadius + x] += samples[x];
    }
    const auto area = cv::v_setall_u16(kLocalArea);
    const auto offset = cv::v_setall_u16(difference * kLocalArea - kLocalArea / 2);
    for (int y = 0; y < height; ++y) {
        const auto* input = guide.ptr<std::uint8_t>(y);
        auto* output = mask.ptr<std::uint8_t>(y);
        std::fill_n(sums, kRadius, sums[kRadius]);
        std::fill_n(sums + width + kRadius, kRadius, sums[width + kRadius - 1]);
        int total = 0;
        for (int j = 0; j < kLocalWindow; ++j)
            total += sums[j];
        output[0] = (total >= kLocalArea * (input[0] + difference) - kLocalArea / 2) ? 255 : 0;
        int x = 1;
        for (; x + 16 <= width; x += 16) {
            const auto eight = [&](int first) {
                auto delta = cv::v_sub_wrap(cv::v_load(sums + first + kLocalWindow - 1),
                                            cv::v_load(sums + first - 1));
                // Signed entering-minus-leaving sums wrap modulo 2^16. Adding
                // the preceding box total restores the exact unsigned area sum.
                delta = cv::v_add_wrap(delta, cv::v_rotate_left<1>(delta));
                delta = cv::v_add_wrap(delta, cv::v_rotate_left<2>(delta));
                delta = cv::v_add_wrap(delta, cv::v_rotate_left<4>(delta));
                auto box = cv::v_add_wrap(cv::v_setall_u16(total), delta);
                total = cv::v_extract_n<7>(box);
                auto limit = cv::v_add_wrap(cv::v_mul_wrap(cv::v_load_expand(input + first), area),
                                            offset);
                return cv::v_ge(box, limit);
            };
            auto lo = eight(x);
            auto hi = eight(x + 8);
            cv::v_store(output + x, cv::v_pack(lo, hi));
        }
        for (; x < width; ++x) {
            total += int(sums[x + kLocalWindow - 1]) - sums[x - 1];
            output[x] = (total >= kLocalArea * (input[x] + difference) - kLocalArea / 2) ? 255 : 0;
        }
        if (y + 1 == height)
            break;
        const auto* next = guide.ptr<std::uint8_t>(std::min(y + kRadius + 1, height - 1));
        const auto* prev = guide.ptr<std::uint8_t>(std::max(y - kRadius, 0));
        x = 0;
        for (; x + 8 <= width; x += 8) {
            auto old = cv::v_load(sums + kRadius + x);
            old = cv::v_add_wrap(old, cv::v_load_expand(next + x));
            cv::v_store(sums + kRadius + x, cv::v_sub_wrap(old, cv::v_load_expand(prev + x)));
        }
        for (; x < width; ++x)
            sums[kRadius + x] += int(next[x]) - prev[x];
    }
}
#endif

#if CV_SIMD128
template <int phase>
inline cv::v_uint8x16 load_nearest4(const std::uint8_t* row) {
    static_assert(phase == 0 || phase == 2);
    cv::v_uint8x16 first, second, third, fourth;
    cv::v_load_deinterleave(row, first, second, third, fourth);
    if constexpr (phase == 2)
        return third;
    return first;
}

template <int phase>
inline void reduce_nearest4(const cv::Mat& gray, cv::Mat& guide) {
    guide.create(gray.rows / 4, gray.cols / 4, CV_8U);
    for (int y = 0; y < guide.rows; ++y) {
        const auto* row = gray.ptr<std::uint8_t>(4 * y + phase);
        auto* samples = guide.ptr<std::uint8_t>(y);
        int x = 0;
        for (; x + 16 <= guide.cols; x += 16)
            cv::v_store(samples + x, load_nearest4<phase>(row + 4 * x));
        for (; x < guide.cols; ++x)
            samples[x] = row[4 * x + phase];
    }
}
#endif

inline void resize_nearest_guide(const cv::Mat& gray, cv::Mat& guide, int step,
                                 cv::InterpolationFlags mode) {
#if CV_SIMD128
    if (step == 4 && gray.cols % 4 == 0 && gray.rows % 4 == 0 && cv::getNumThreads() <= 1) {
        // Exact quarter sizes use a fixed phase in OpenCV. Floor sampling uses
        // the first pixel and nearest-exact uses the pixel at the block midpoint.
        if (mode == cv::INTER_NEAREST_EXACT) {
            reduce_nearest4<2>(gray, guide);
            return;
        }
        if (mode == cv::INTER_NEAREST) {
            reduce_nearest4<0>(gray, guide);
            return;
        }
    }
#endif
    cv::resize(gray, guide, {(gray.cols + step - 1) / step, (gray.rows + step - 1) / step}, 0, 0,
               mode);
}

// Bounds every nonzero value in the phase-zero quarter-resolution binary guide.
// The horizontal bounds round outward to SIMD blocks. The caller expands this
// support by the dilation radius before extracting contours.
inline cv::Rect threshold_quarter_guide(const cv::Mat& gray, int level, cv::Mat& mask) {
    CV_Assert(gray.cols % 4 == 0 && gray.rows % 4 == 0);
    const int width = gray.cols / 4, height = gray.rows / 4;
    if (level < 0)
        return {};
    mask.create(height, width, CV_8U);
    const auto cutoff = static_cast<std::uint8_t>(std::min(level, 255));
    int left = width, top = height, right = 0, bottom = 0;
    for (int y = 0; y < height; ++y) {
        const auto* source = gray.ptr<std::uint8_t>(4 * y);
        auto* row = mask.ptr<std::uint8_t>(y);
        int first = width, last = 0, x = 0;
#if CV_SIMD128
        const auto threshold = cv::v_setall_u8(cutoff);
        for (; x + 16 <= width; x += 16) {
            const auto dark = cv::v_le(load_nearest4<0>(source + 4 * x), threshold);
            cv::v_store(row + x, dark);
            if (cv::v_check_any(dark)) {
                first = std::min(first, x);
                last = x + 16;
            }
        }
#endif
        for (; x < width; ++x) {
            const bool dark = source[4 * x] <= cutoff;
            row[x] = dark ? 255 : 0;
            if (dark) {
                first = std::min(first, x);
                last = x + 1;
            }
        }
        if (last > first) {
            left = std::min(left, first);
            right = std::max(right, last);
            top = std::min(top, y);
            bottom = y + 1;
        }
    }
    return bottom > top ? cv::Rect(left, top, right - left, bottom - top) : cv::Rect{};
}

// Samples the average light at the geometric centre of a 4x4 thumbnail cell.
// Using the four central pixels attenuates read noise before contour extraction.
inline void sample_area4(const cv::Mat& gray, cv::Mat& guide) {
    guide.create(gray.rows / 4, gray.cols / 4, CV_8U);
    for (int y = 0; y < guide.rows; ++y) {
        const auto* upper = gray.ptr<std::uint8_t>(4 * y + 1);
        const auto* lower = gray.ptr<std::uint8_t>(4 * y + 2);
        auto* out = guide.ptr<std::uint8_t>(y);
        int x = 0;
#if CV_SIMD128
        const auto round = cv::v_setall_u16(2);
        for (; x + 16 <= guide.cols; x += 16) {
            cv::v_uint8x16 u0, u1, u2, u3, v0, v1, v2, v3;
            cv::v_load_deinterleave(upper + 4 * x, u0, u1, u2, u3);
            cv::v_load_deinterleave(lower + 4 * x, v0, v1, v2, v3);
            const auto low =
                    cv::v_add_wrap(cv::v_add_wrap(cv::v_expand_low(u1), cv::v_expand_low(u2)),
                                   cv::v_add_wrap(cv::v_expand_low(v1), cv::v_expand_low(v2)));
            const auto high =
                    cv::v_add_wrap(cv::v_add_wrap(cv::v_expand_high(u1), cv::v_expand_high(u2)),
                                   cv::v_add_wrap(cv::v_expand_high(v1), cv::v_expand_high(v2)));
            cv::v_store(out + x, cv::v_pack(cv::v_shr<2>(cv::v_add_wrap(low, round)),
                                            cv::v_shr<2>(cv::v_add_wrap(high, round))));
        }
#endif
        for (; x < guide.cols; ++x)
            out[x] = (upper[4 * x + 1] + upper[4 * x + 2] + lower[4 * x + 1] + lower[4 * x + 2] +
                      2) /
                     4;
    }
}

// The inverted adaptive mean test is mean - sample >= difference, for
// positive integral difference. The intermediate subtraction saturates at zero.
inline void local_ink_mask(const cv::Mat& guide, int window, int difference, cv::Mat& scratch,
                           cv::Mat& mask) {
#if CV_SIMD128
    constexpr int maximum_difference = std::numeric_limits<std::uint16_t>::max() / kLocalArea - 255;
    if (window == kLocalWindow && difference >= 1 && difference <= maximum_difference) {
        local_ink_mask_15(guide, difference, scratch, mask);
        return;
    }
#endif
    cv::boxFilter(guide, scratch, CV_8U, {window, window}, {-1, -1}, true,
                  cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);
    cv::subtract(scratch, guide, mask);
    cv::compare(mask, difference, mask, cv::CMP_GE);
}
}  // namespace aruco3cuda::apriltag_small_cpu::detail
