// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace aruco3cuda::apriltag_small_cpu::detail {

#if CV_SIMD128 && defined(__aarch64__) && defined(__AARCH64EL__)
namespace cubic_patch {

using Sample = std::int32_t;
using Pixel = std::uint8_t;

// For Keys cubic with A = -0.75, the two quarter phases of 2x enlargement
// are exact multiples of 1/256. The second phase reflects the first.
constexpr std::array<Sample, 4> kEarlyWeights{-9, 67, 225, -27};
constexpr std::array<Sample, 4> kLateWeights{-27, 225, 67, -9};
constexpr int kPixelScale = 256 * 256;

inline void horizontal(const Pixel* input, int width, Sample* output) {
    const auto boundary = [&](int x) {
        const auto at = [&](int i) { return input[std::clamp(i, 0, width - 1)]; };
        const auto& a = kEarlyWeights;
        const auto& b = kLateWeights;
        output[2 * x] = a[0] * at(x - 2) + a[1] * at(x - 1) + a[2] * at(x) + a[3] * at(x + 1);
        output[2 * x + 1] = b[0] * at(x - 1) + b[1] * at(x) + b[2] * at(x + 1) + b[3] * at(x + 2);
    };
    int x = 0;
    for (; x < std::min(2, width); ++x)
        boundary(x);
    const auto outer = cv::v_setall_s32(kEarlyWeights[0]);
    const auto inner = cv::v_setall_s32(kEarlyWeights[1]);
    const auto peak = cv::v_setall_s32(kEarlyWeights[2]);
    const auto shoulder = cv::v_setall_s32(kEarlyWeights[3]);
    for (; x + 6 <= width; x += 4) {
        const auto load = [&](int index) {
            return cv::v_reinterpret_as_s32(cv::v_load_expand_q(input + index));
        };
        const auto a = load(x - 2), b = load(x - 1), c = load(x), d = load(x + 1), e = load(x + 2);
        const auto center = cv::v_mul(c, peak);
        const auto early =
                cv::v_muladd(a, outer, cv::v_muladd(b, inner, cv::v_muladd(d, shoulder, center)));
        const auto late =
                cv::v_muladd(b, shoulder, cv::v_muladd(d, inner, cv::v_muladd(e, outer, center)));
        cv::v_store_interleave(output + 2 * x, early, late);
    }
    for (; x < width; ++x)
        boundary(x);
}

template <bool early>
inline void vertical(const std::array<const Sample*, 4>& rows, Pixel* output, int width) {
    constexpr auto w = early ? kEarlyWeights : kLateWeights;
    const auto a = cv::v_setall_f32(float(w[0]) / kPixelScale);
    const auto b = cv::v_setall_f32(float(w[1]) / kPixelScale);
    const auto c = cv::v_setall_f32(float(w[2]) / kPixelScale);
    const auto d = cv::v_setall_f32(float(w[3]) / kPixelScale);
    int x = 0;
    // OpenCV 4.14.0 rounds 8-byte SIMD groups with float FMA and ties to even.
    // Its scalar remainder rounds half up in fixed point.
    for (; x + 8 <= width; x += 8) {
        const auto evaluate = [&](int j) {
            const auto p = cv::v_cvt_f32(cv::v_load(rows[0] + j));
            const auto q = cv::v_cvt_f32(cv::v_load(rows[1] + j));
            const auto r = cv::v_cvt_f32(cv::v_load(rows[2] + j));
            const auto s = cv::v_cvt_f32(cv::v_load(rows[3] + j));
            return cv::v_round(
                    cv::v_muladd(p, a, cv::v_muladd(q, b, cv::v_muladd(r, c, cv::v_mul(s, d)))));
        };
        cv::v_pack_u_store(output + x, cv::v_pack(evaluate(x), evaluate(x + 4)));
    }
    for (; x < width; ++x) {
        const int sum =
                w[0] * rows[0][x] + w[1] * rows[1][x] + w[2] * rows[2][x] + w[3] * rows[3][x];
        output[x] = cv::saturate_cast<Pixel>((sum + kPixelScale / 2) / kPixelScale);
    }
}

}  // namespace cubic_patch
#endif

// Source is a nonempty Gray8 crop. The caller owns the five reusable filtered rows.
inline void enlarge_patch_cubic(const cv::Mat& source, cv::Mat& target,
                                std::vector<std::int32_t>& rows) {
#if CV_SIMD128 && defined(__aarch64__) && defined(__AARCH64EL__)
    if (cv::useOptimized() && cv::getNumThreads() == 1) {
        constexpr int kRows = 5;
        const int width = 2 * source.cols;
        target.create(2 * source.rows, width, CV_8UC1);
        rows.resize(std::size_t{kRows} * width);
        const auto row = [&](int y) {
            const int slot = std::clamp(y, 0, source.rows - 1) % kRows;
            return rows.data() + std::size_t(slot) * width;
        };
        int next = 0;
        for (int y = 0; y < source.rows; ++y) {
            // Both phases need rows from y-2 through y+2. Compute each source row once.
            for (; next <= y + 2 && next < source.rows; ++next)
                cubic_patch::horizontal(source.ptr<std::uint8_t>(next), source.cols, row(next));
            cubic_patch::vertical<true>({row(y - 2), row(y - 1), row(y), row(y + 1)},
                                        target.ptr<std::uint8_t>(2 * y), width);
            cubic_patch::vertical<false>({row(y - 1), row(y), row(y + 1), row(y + 2)},
                                         target.ptr<std::uint8_t>(2 * y + 1), width);
        }
        return;
    }
#else
    static_cast<void>(rows);
#endif
    cv::resize(source, target, {}, 2., 2., cv::INTER_CUBIC);
}

}  // namespace aruco3cuda::apriltag_small_cpu::detail
