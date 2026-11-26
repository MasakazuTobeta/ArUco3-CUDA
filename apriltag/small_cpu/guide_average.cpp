// SPDX-License-Identifier: Apache-2.0
#include "guide_average.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cfenv>
#include <cstddef>
#include <cstdint>

#if defined(__aarch64__) && defined(__linux__) && defined(__AARCH64EL__)
#include <arm_neon.h>
#include <asm/hwcap.h>
#include <sys/auxv.h>
#elif defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#endif

namespace aruco3cuda::apriltag_small_cpu::detail {
namespace {

constexpr int kBlock = 4;
using Pixel = std::uint8_t;

[[maybe_unused]] void average_tail(const Pixel* row, std::size_t step, int begin, int end,
                                   Pixel* dst) {
    for (int x = begin; x < end; ++x) {
        const int left = kBlock * x;
        int sum = 0;
        for (int y = 0; y < kBlock; ++y) {
            const auto* src = row + y * step + left;
            sum += src[0] + src[1] + src[2] + src[3];
        }
        // 0 ≤ sum ≤ 4080. At a half tie, use the even destination intensity.
        dst[x] = static_cast<Pixel>((sum + 7 + ((sum >> 4) & 1)) >> 4);
    }
}

#if defined(__aarch64__) && defined(__linux__) && defined(__AARCH64EL__)
bool has_arm_dot_product() {
    static const bool available = (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
    return available;
}

template <int stage>
inline void add_row_groups(const Pixel* source, std::uint16_t* saved, Pixel* output, int width) {
    const auto sum_four = [](auto a, auto b) { return vpaddq_u16(vpaddlq_u8(a), vpaddlq_u8(b)); };
    const auto round = [](uint16x8_t sum) {
        const auto odd = vandq_u16(vshrq_n_u16(sum, 4), vdupq_n_u16(1));
        return vshrn_n_u16(vaddq_u16(vaddq_u16(sum, vdupq_n_u16(7)), odd), 4);
    };
    for (int x = 0; x < width; x += 16) {
        if (x + 96 < width)
            __builtin_prefetch(source + 4 * (x + 96));
        const auto pixels = vld1q_u8_x4(source + kBlock * x);
        auto low = sum_four(pixels.val[0], pixels.val[1]);
        auto high = sum_four(pixels.val[2], pixels.val[3]);
        if constexpr (stage != 0) {
            low = vaddq_u16(low, vld1q_u16(saved + x));
            high = vaddq_u16(high, vld1q_u16(saved + x + 8));
        }
        if constexpr (stage == 3) {
            vst1q_u8(output + x, vcombine_u8(round(low), round(high)));
        } else {
            vst1q_u16(saved + x, low);
            vst1q_u16(saved + x + 8, high);
        }
    }
}

inline void average_arm_rows(const cv::Mat& source, cv::Size complete, cv::Mat& guide) {
    // Accumulate four source rows in a small intermediate row to stream one native row at a time.
    cv::AutoBuffer<std::uint16_t, 1024> partial(complete.width);
    for (int y = 0; y < complete.height; ++y) {
        const auto* row = source.ptr<Pixel>(kBlock * y);
        const auto step = source.step[0];
        auto* output = guide.ptr<Pixel>(y);
        add_row_groups<0>(row, partial.data(), output, complete.width);
        add_row_groups<1>(row + step, partial.data(), output, complete.width);
        add_row_groups<2>(row + 2 * step, partial.data(), output, complete.width);
        add_row_groups<3>(row + 3 * step, partial.data(), output, complete.width);
    }
}

__attribute__((target("arch=armv8.2-a+dotprod"))) void average_arm(const cv::Mat& gray,
                                                                   cv::Size complete,
                                                                   cv::Mat& guide) {
    const std::size_t step = gray.step[0];
    const auto ones = vdupq_n_u8(1);
    const auto zero = vdupq_n_u32(0);
    const auto seven = vdupq_n_u16(7);
    const auto one = vdupq_n_u16(1);
    for (int y = 0; y < complete.height; ++y) {
        const auto* a = gray.ptr<Pixel>(kBlock * y);
        const auto* b = a + step;
        const auto* c = b + step;
        const auto* d = c + step;
        auto* dst = guide.ptr<Pixel>(y);
        const auto sum_four = [&](int at) __attribute__((target("arch=armv8.2-a+dotprod"))) {
            auto sum = vdotq_u32(zero, vld1q_u8(a + at), ones);
            sum = vdotq_u32(sum, vld1q_u8(b + at), ones);
            sum = vdotq_u32(sum, vld1q_u8(c + at), ones);
            return vreinterpretq_u16_u32(vdotq_u32(sum, vld1q_u8(d + at), ones));
        };
        const auto round_eight = [&](uint16x8_t low, uint16x8_t high) {
            const auto sum = vuzp1q_u16(low, high);
            const auto parity = vandq_u16(vshrq_n_u16(sum, 4), one);
            return vshrn_n_u16(vaddq_u16(vaddq_u16(sum, seven), parity), 4);
        };
        int x = 0;
        for (; complete.width - x >= 16; x += 16) {
            const int offset = kBlock * x;
            constexpr int kLookaheadBytes = 128;
            if (gray.cols - offset > kLookaheadBytes) {
                __builtin_prefetch(a + offset + kLookaheadBytes);
                __builtin_prefetch(b + offset + kLookaheadBytes);
                __builtin_prefetch(c + offset + kLookaheadBytes);
                __builtin_prefetch(d + offset + kLookaheadBytes);
            }
            const auto low = round_eight(sum_four(offset), sum_four(offset + 16));
            const auto high = round_eight(sum_four(offset + 32), sum_four(offset + 48));
            vst1q_u8(dst + x, vcombine_u8(low, high));
        }
        average_tail(a, step, x, complete.width, dst);
    }
}
template <bool centered>
__attribute__((target("arch=armv8.2-a+dotprod"))) void average_and_sample(const cv::Mat& gray,
                                                                          cv::Mat& area,
                                                                          cv::Mat& guide) {
    const cv::Size reduced(gray.cols / kBlock, gray.rows / kBlock);
    area.create(reduced, CV_8UC1);
    guide.create(reduced, CV_8UC1);
    const auto full_weights = vdupq_n_u8(1);
    // Little-endian byte weights select the two central columns of each four-column block.
    const auto center_weights = vreinterpretq_u8_u32(vdupq_n_u32(0x00010100));
    const auto nearest_columns =
            uint8x16_t{2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62};
    for (int y = 0; y < reduced.height; ++y) {
        const auto* first = gray.ptr<Pixel>(kBlock * y);
        const auto* second = first + gray.step[0];
        const auto* third = second + gray.step[0];
        const auto* fourth = third + gray.step[0];
        auto* means = area.ptr<Pixel>(y);
        auto* samples = guide.ptr<Pixel>(y);
        for (int x = 0; x < reduced.width; x += 16) {
            const int offset = kBlock * x;
            const auto first_pixels = vld1q_u8_x4(first + offset);
            const auto second_pixels = vld1q_u8_x4(second + offset);
            const auto third_pixels = vld1q_u8_x4(third + offset);
            const auto fourth_pixels = vld1q_u8_x4(fourth + offset);
            uint16x8x4_t full_sums, center_sums;
            for (int group = 0; group < 4; ++group) {
                auto sum = vdotq_u32(vdupq_n_u32(0), first_pixels.val[group], full_weights);
                sum = vdotq_u32(sum, second_pixels.val[group], full_weights);
                sum = vdotq_u32(sum, third_pixels.val[group], full_weights);
                sum = vdotq_u32(sum, fourth_pixels.val[group], full_weights);
                full_sums.val[group] = vreinterpretq_u16_u32(sum);
                if constexpr (centered) {
                    const auto middle =
                            vdotq_u32(vdupq_n_u32(0), second_pixels.val[group], center_weights);
                    center_sums.val[group] = vreinterpretq_u16_u32(
                            vdotq_u32(middle, third_pixels.val[group], center_weights));
                }
            }
            const auto area_means = [&](int group) {
                const auto sum = vuzp1q_u16(full_sums.val[group], full_sums.val[group + 1]);
                const auto odd = vandq_u16(vshrq_n_u16(sum, 4), vdupq_n_u16(1));
                return vshrn_n_u16(vaddq_u16(vaddq_u16(sum, vdupq_n_u16(7)), odd), 4);
            };
            vst1q_u8(means + x, vcombine_u8(area_means(0), area_means(2)));
            if constexpr (centered) {
                const auto center_means = [&](int group) {
                    return vrshrn_n_u16(
                            vuzp1q_u16(center_sums.val[group], center_sums.val[group + 1]), 2);
                };
                vst1q_u8(samples + x, vcombine_u8(center_means(0), center_means(2)));
            } else {
                vst1q_u8(samples + x, vqtbl4q_u8(third_pixels, nearest_columns));
            }
        }
    }
}
#elif defined(__x86_64__) && defined(__GNUC__)
__attribute__((target("avx2"))) void average_avx2(const cv::Mat& gray, cv::Size complete,
                                                  cv::Mat& guide) {
    const auto ones8 = _mm256_set1_epi8(1);
    const auto ones16 = _mm256_set1_epi16(1);
    const auto one = _mm256_set1_epi32(1);
    const auto seven = _mm256_set1_epi32(7);
    const std::size_t step = gray.step[0];
    for (int y = 0; y < complete.height; ++y) {
        const auto* row = gray.ptr<Pixel>(kBlock * y);
        auto* dst = guide.ptr<Pixel>(y);
        int x = 0;
        for (; complete.width - x >= 8; x += 8) {
            const auto* block = row + kBlock * x;
            auto pairs = _mm256_setzero_si256();
            for (int dy = 0; dy < kBlock; ++dy) {
                const auto* src = reinterpret_cast<const __m256i*>(block + dy * step);
                pairs = _mm256_add_epi16(pairs,
                                         _mm256_maddubs_epi16(_mm256_loadu_si256(src), ones8));
            }
            const auto sum = _mm256_madd_epi16(pairs, ones16);
            const auto parity = _mm256_and_si256(_mm256_srli_epi32(sum, 4), one);
            const auto rounded =
                    _mm256_srli_epi32(_mm256_add_epi32(_mm256_add_epi32(sum, seven), parity), 4);
            const auto words = _mm_packus_epi32(_mm256_castsi256_si128(rounded),
                                                _mm256_extracti128_si256(rounded, 1));
            _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + x),
                             _mm_packus_epi16(words, _mm_setzero_si128()));
        }
        average_tail(row, step, x, complete.width, dst);
    }
}
#endif

bool average_vector([[maybe_unused]] const cv::Mat& gray, [[maybe_unused]] cv::Size complete,
                    [[maybe_unused]] cv::Mat& guide) {
    if (std::fegetround() != FE_TONEAREST)
        return false;
#if defined(__aarch64__) && defined(__linux__) && defined(__AARCH64EL__)
    const bool has_dot = has_arm_dot_product();
    constexpr int kStreamingMinimumWidth = 512;
    constexpr int kStreamingMinimumHeight = 256;
    if (has_dot && complete.width >= kStreamingMinimumWidth &&
        complete.height >= kStreamingMinimumHeight && complete.width % 16 == 0) {
        average_arm_rows(gray, complete, guide);
        return true;
    }
    if (has_dot && complete.width >= 16) {
        average_arm(gray, complete, guide);
        return true;
    }
#elif defined(__x86_64__) && defined(__GNUC__)
    if (_MM_GET_ROUNDING_MODE() == _MM_ROUND_NEAREST && __builtin_cpu_supports("avx2") &&
        complete.width >= 8) {
        average_avx2(gray, complete, guide);
        return true;
    }
#endif
    return false;
}

}  // namespace

void average_gray4x4(const cv::Mat& gray, cv::Mat& guide) {
    CV_Assert(gray.dims == 2 && !gray.empty() && gray.type() == CV_8UC1);
    const cv::Size complete(gray.cols / kBlock, gray.rows / kBlock);
    const cv::Size extent(complete.width + (gray.cols % kBlock != 0),
                          complete.height + (gray.rows % kBlock != 0));
    guide.create(extent, CV_8UC1);
    if (!complete.empty() && !average_vector(gray, complete, guide)) {
        auto interior = guide(cv::Rect({}, complete));
        const cv::Rect native(0, 0, complete.width * kBlock, complete.height * kBlock);
        cv::resize(gray(native), interior, complete, 0, 0, cv::INTER_AREA);
    }
    const auto partial = [&](int col, int row) {
        const int x = col * kBlock;
        const int y = row * kBlock;
        const int width = std::min(kBlock, gray.cols - x);
        const int height = std::min(kBlock, gray.rows - y);
        int sum = 0;
        for (int dy = 0; dy < height; ++dy) {
            const auto* source = gray.ptr<Pixel>(y + dy) + x;
            for (int dx = 0; dx < width; ++dx)
                sum += source[dx];
        }
        return cv::saturate_cast<Pixel>(float(sum) / (width * height));
    };
    if (extent.width != complete.width) {
        for (int row = 0; row < complete.height; ++row)
            guide.ptr<Pixel>(row)[complete.width] = partial(complete.width, row);
    }
    if (extent.height != complete.height) {
        auto* last = guide.ptr<Pixel>(complete.height);
        for (int col = 0; col < extent.width; ++col)
            last[col] = partial(col, complete.height);
    }
}

bool try_sample_gray4x4([[maybe_unused]] const cv::Mat& gray, [[maybe_unused]] cv::Mat& area,
                        [[maybe_unused]] cv::Mat& guide,
                        [[maybe_unused]] QuarterGuideSample sample) {
#if defined(__aarch64__) && defined(__linux__) && defined(__AARCH64EL__)
    // The two outputs are produced together only for full vector-sized groups.
    if (!has_arm_dot_product() || gray.empty() || gray.cols % 64 != 0 || gray.rows % 4 != 0 ||
        std::fegetround() != FE_TONEAREST)
        return false;
    if (sample == QuarterGuideSample::central_area)
        average_and_sample<true>(gray, area, guide);
    else
        average_and_sample<false>(gray, area, guide);
    return true;
#else
    return false;
#endif
}

}  // namespace aruco3cuda::apriltag_small_cpu::detail
