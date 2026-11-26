// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cstdint>

namespace aruco3cuda::apriltag_small_cpu::detail {

// A 3x3 median on a 0/255 mask requires five of nine foreground samples.
// The outermost row and column supply replicated border samples.
// Source and destination have separate storage.
inline void median_binary3(const cv::Mat& input, cv::Mat& output) {
#if CV_SIMD128 && defined(__aarch64__) && defined(__AARCH64EL__)
    if (!cv::useOptimized() || cv::getNumThreads() != 1) {
        cv::medianBlur(input, output, 3);
        return;
    }
    using Byte = std::uint8_t;
    output.create(input.size(), CV_8UC1);
    for (int y = 0; y < input.rows; ++y) {
        const std::array<const Byte*, 3> rows{input.ptr<Byte>(std::max(y - 1, 0)),
                                              input.ptr<Byte>(y),
                                              input.ptr<Byte>(std::min(y + 1, input.rows - 1))};
        const auto count_column = [&](int x) {
            return (rows[0][x] != 0) + (rows[1][x] != 0) + (rows[2][x] != 0);
        };
        auto* dst = output.ptr<Byte>(y);
        int x = 0;
        constexpr int kLanes = 16;
        if (input.cols >= kLanes) {
            const auto zero = cv::v_setzero_u8();
            const auto five = cv::v_setall_u8(5);
            const auto counts = [&](int pos) {
                // A foreground byte is -1 modulo 256. Negating the sum gives 0–3.
                const auto negatives = cv::v_add_wrap(
                        cv::v_add_wrap(cv::v_load(rows[0] + pos), cv::v_load(rows[1] + pos)),
                        cv::v_load(rows[2] + pos));
                return cv::v_sub_wrap(zero, negatives);
            };
            auto left = cv::v_setall_u8(static_cast<Byte>(count_column(0)));
            auto current = counts(0);
            for (; x + kLanes <= input.cols; x += kLanes) {
                const auto right = x + 2 * kLanes <= input.cols
                                           ? counts(x + kLanes)
                                           : cv::v_setall_u8(static_cast<Byte>(count_column(
                                                     std::min(x + kLanes, input.cols - 1))));
                const auto a = cv::v_extract<15>(left, current);
                const auto b = cv::v_extract<1>(current, right);
                cv::v_store(dst + x, cv::v_ge(cv::v_add_wrap(cv::v_add_wrap(a, current), b), five));
                left = current;
                current = right;
            }
        }
        for (; x < input.cols; ++x) {
            const int count = count_column(std::max(0, x - 1)) + count_column(x) +
                              count_column(std::min(x + 1, input.cols - 1));
            dst[x] = count >= 5 ? 255 : 0;
        }
    }
#else
    cv::medianBlur(input, output, 3);
#endif
}

}  // namespace aruco3cuda::apriltag_small_cpu::detail
