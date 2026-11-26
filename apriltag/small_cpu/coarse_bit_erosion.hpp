// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/core/mat.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace aruco3cuda::apriltag_small_cpu::detail {

// The caller supplies neutral one bits beyond the right edge of the input.
inline void erode_coarse_packed(cv::Size size, const std::vector<std::uint64_t>& source,
                                std::vector<std::uint64_t>& eroded) {
    using Word = std::uint64_t;
    const int words = (size.width + 63) / 64;
    const int tail = size.width % 64;
    const Word tail_mask = tail == 0 ? ~Word{0} : (Word{1} << tail) - 1;
    eroded.resize(static_cast<std::size_t>(words) * size.height);
    for (int y = 0; y < size.height; ++y) {
        const auto* previous = source.data() + static_cast<std::size_t>(std::max(y - 1, 0)) * words;
        const auto* here = source.data() + static_cast<std::size_t>(y) * words;
        const auto* next =
                source.data() + static_cast<std::size_t>(std::min(y + 1, size.height - 1)) * words;
        auto* output = eroded.data() + static_cast<std::size_t>(y) * words;
        Word left = ~Word{0};
        Word vertical = previous[0] & here[0] & next[0];
        for (int j = 0; j < words; ++j) {
            const Word right =
                    j + 1 == words ? ~Word{0} : previous[j + 1] & here[j + 1] & next[j + 1];
            output[j] =
                    vertical & ((vertical << 1) | (left >> 63)) & ((vertical >> 1) | (right << 63));
            left = vertical;
            vertical = right;
        }
        output[words - 1] &= tail_mask;
    }
}

// Input is a binary 0/255 raster. Erosion sees the neutral value outside
// the isolated crop, as with the default morphology border.
inline void erode_coarse_bits(const cv::Mat& input, std::vector<std::uint64_t>& source,
                              std::vector<std::uint64_t>& eroded) {
    using Word = std::uint64_t;
    const int words = (input.cols + 63) / 64;
    const int tail = input.cols % 64;
    const Word tail_mask = tail == 0 ? ~Word{0} : (Word{1} << tail) - 1;
    const auto count = static_cast<std::size_t>(words) * input.rows;
    source.resize(count);
    for (int y = 0; y < input.rows; ++y) {
        const auto* row = input.ptr<std::uint8_t>(y);
        auto* bits = source.data() + static_cast<std::size_t>(y) * words;
        for (int j = 0; j < words; ++j) {
            Word word = 0;
            const int start = 64 * j;
            const int width = std::min(64, input.cols - start);
            int x = 0;
#if CV_SIMD128
            for (; x + 16 <= width; x += 16) {
                const auto mask = cv::v_signmask(cv::v_load(row + start + x));
                word |= static_cast<Word>(mask) << x;
            }
#endif
            for (; x < width; ++x)
                word |= static_cast<Word>(row[start + x] != 0) << x;
            bits[j] = word;
        }
        bits[words - 1] |= ~tail_mask;
    }
    erode_coarse_packed(input.size(), source, eroded);
}

}  // namespace aruco3cuda::apriltag_small_cpu::detail
