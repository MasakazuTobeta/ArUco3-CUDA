// SPDX-License-Identifier: Apache-2.0
#include "cpu_box_sample.hpp"

#include <algorithm>
#include <cstdint>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace aruco3cuda::cpu::detail {
void sample_box_grid(const cv::Mat& input, cv::Mat& output, cv::Size size, int step) {
    output.create(size, CV_8UC1);
    const int center = (step - 2) / 2;
    for (int y = 0; y < size.height; ++y) {
        const int ay = std::min(y * step + center, input.rows - 1);
        const int by = std::min(ay + 1, input.rows - 1);
        const auto* ra = input.ptr<uint8_t>(ay);
        const auto* rb = input.ptr<uint8_t>(by);
        auto* dst = output.ptr<uint8_t>(y);
        int x = 0;
#if defined(__aarch64__)
        for (; x + 16 <= size.width && (x + 16) * step <= input.cols; x += 16) {
            const auto* a = ra + x * step;
            const auto* b = rb + x * step;
            if (step == 3) {
                const auto v = vld3q_u8(a), u = vld3q_u8(b);
                const auto sa = vrhaddq_u8(v.val[0], v.val[1]);
                const auto sb = vrhaddq_u8(u.val[0], u.val[1]);
                vst1q_u8(dst + x, vrhaddq_u8(sa, sb));
            } else if (step == 4) {
                const auto v = vld4q_u8(a), u = vld4q_u8(b);
                const auto sa = vrhaddq_u8(v.val[1], v.val[2]);
                const auto sb = vrhaddq_u8(u.val[1], u.val[2]);
                vst1q_u8(dst + x, vrhaddq_u8(sa, sb));
            } else if (step == 6) {
                if ((x + 16) * step + 2 > input.cols)
                    break;
                const auto v0 = vld3q_u8(a + 2), v1 = vld3q_u8(a + 50);
                const auto u0 = vld3q_u8(b + 2), u1 = vld3q_u8(b + 50);
                const auto sa = vuzp1q_u8(vrhaddq_u8(v0.val[0], v0.val[1]),
                                          vrhaddq_u8(v1.val[0], v1.val[1]));
                const auto sb = vuzp1q_u8(vrhaddq_u8(u0.val[0], u0.val[1]),
                                          vrhaddq_u8(u1.val[0], u1.val[1]));
                vst1q_u8(dst + x, vrhaddq_u8(sa, sb));
            } else {
                break;
            }
        }
#endif
        for (; x < size.width; ++x) {
            const int a = std::min(x * step + center, input.cols - 1),
                      b = std::min(a + 1, input.cols - 1);
            dst[x] = (((static_cast<int>(ra[a]) + ra[b] + 1) / 2) +
                      ((static_cast<int>(rb[a]) + rb[b] + 1) / 2) + 1) /
                     2;
        }
    }
}
}  // namespace aruco3cuda::cpu::detail
