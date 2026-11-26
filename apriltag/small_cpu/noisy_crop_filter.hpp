// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#if defined(__aarch64__) && defined(__AARCH64EL__)
#include <arm_neon.h>
#endif

#include <algorithm>
#include <cstdint>
#include <vector>

namespace aruco3cuda::apriltag_small_cpu::detail {

// The half-area byte is rounded before either spatial filter. With the pinned
// 8-bit, isolated 3x3 Gaussian at sigma .8, the two exact separable scaling
// stages are [61, 134, 61]/256, rounded once after the vertical stage.
class NoisyCropFilter {
public:
    void filter(const cv::Mat& source, float sigma, cv::Mat& output) {
#if defined(__aarch64__) && defined(__AARCH64EL__)
        if (sigma == .8F && cv::useOptimized() && cv::getNumThreads() == 1) {
            this->filter_default(source, output);
            return;
        }
#endif
        cv::resize(source, output, {source.cols / 2, source.rows / 2}, 0, 0, cv::INTER_AREA);
        cv::GaussianBlur(output, output, {3, 3}, sigma, sigma);
    }

private:
    void filter_default(const cv::Mat& source, cv::Mat& output) {
        CV_DbgAssert(source.type() == CV_8UC1 && source.cols >= 2 && source.rows >= 2 &&
                     source.cols % 2 == 0 && source.rows % 2 == 0);
        const int cols = source.cols / 2;
        const int rows = source.rows / 2;
        output.create(rows, cols, CV_8UC1);
        this->half_row_.resize(cols + 2);
        this->horizontal_.resize(3 * cols);
        const auto line = [&](int y) { return this->horizontal_.data() + y % 3 * cols; };
        this->prepare(source, 0, cols, line(0));
        if (rows > 1)
            this->prepare(source, 1, cols, line(1));
        for (int y = 0; y < rows; ++y) {
            if (y > 0 && y + 1 < rows)
                this->prepare(source, y + 1, cols, line(y + 1));
            const auto* previous = line(y == 0 ? std::min(1, rows - 1) : y - 1);
            const auto* next = line(y + 1 < rows ? y + 1 : std::max(0, y - 1));
            auto* dst = output.ptr<std::uint8_t>(y);
            this->write_column(previous, line(y), next, cols, dst);
        }
    }

    static constexpr int kEdge = 61;
    static constexpr int kCenter = 134;
    static_assert(2 * kEdge + kCenter == 256);

    void prepare(const cv::Mat& input, int y, int width, std::uint16_t* filtered) {
        const auto* a = input.ptr<std::uint8_t>(2 * y);
        const auto* b = input.ptr<std::uint8_t>(2 * y + 1);
        auto* area = this->half_row_.data() + 1;
        int x = 0;
#if defined(__aarch64__) && defined(__AARCH64EL__)
        for (; x + 16 <= width; x += 16) {
            const auto p = vld2q_u8(a + 2 * x);
            const auto q = vld2q_u8(b + 2 * x);
            const auto lo = vaddq_u16(vaddl_u8(vget_low_u8(p.val[0]), vget_low_u8(p.val[1])),
                                      vaddl_u8(vget_low_u8(q.val[0]), vget_low_u8(q.val[1])));
            const auto hi =
                    vaddq_u16(vaddl_high_u8(p.val[0], p.val[1]), vaddl_high_u8(q.val[0], q.val[1]));
            vst1q_u8(area + x, vcombine_u8(vrshrn_n_u16(lo, 2), vrshrn_n_u16(hi, 2)));
        }
#endif
        for (; x < width; ++x)
            area[x] = static_cast<std::uint8_t>(
                    (a[2 * x] + a[2 * x + 1] + b[2 * x] + b[2 * x + 1] + 2) >> 2);
        area[-1] = area[std::min(1, width - 1)];
        area[width] = area[std::max(0, width - 2)];
        x = 0;
#if defined(__aarch64__) && defined(__AARCH64EL__)
        for (; x + 8 <= width; x += 8) {
            const auto left = vld1_u8(area + x - 1);
            const auto center = vld1_u8(area + x);
            const auto right = vld1_u8(area + x + 1);
            const auto total =
                    vmlal_u8(vmulq_n_u16(vaddl_u8(left, right), kEdge), center, vdup_n_u8(kCenter));
            vst1q_u16(filtered + x, total);
        }
#endif
        for (; x < width; ++x)
            filtered[x] = kEdge * (area[x - 1] + area[x + 1]) + kCenter * area[x];
    }

    static void write_column(const std::uint16_t* before, const std::uint16_t* center,
                             const std::uint16_t* after, int width, std::uint8_t* output) {
        int x = 0;
#if defined(__aarch64__) && defined(__AARCH64EL__)
        for (; x + 8 <= width; x += 8) {
            const auto a = vld1q_u16(before + x);
            const auto b = vld1q_u16(center + x);
            const auto c = vld1q_u16(after + x);
            const auto lo =
                    vmlal_n_u16(vmulq_n_u32(vaddl_u16(vget_low_u16(a), vget_low_u16(c)), kEdge),
                                vget_low_u16(b), kCenter);
            const auto hi = vmlal_high_n_u16(vmulq_n_u32(vaddl_high_u16(a, c), kEdge), b, kCenter);
            const auto rounded = vcombine_u16(vrshrn_n_u32(lo, 16), vrshrn_n_u32(hi, 16));
            vst1_u8(output + x, vmovn_u16(rounded));
        }
#endif
        for (; x < width; ++x) {
            const std::uint32_t sum = kEdge * (before[x] + after[x]) + kCenter * center[x];
            output[x] = static_cast<std::uint8_t>((sum + 32768) >> 16);
        }
    }

    std::vector<std::uint8_t> half_row_;
    std::vector<std::uint16_t> horizontal_;
};

}  // namespace aruco3cuda::apriltag_small_cpu::detail
