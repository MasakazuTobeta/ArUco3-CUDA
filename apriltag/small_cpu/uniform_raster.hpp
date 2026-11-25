// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "tag_codebook.hpp"

#include <algorithm>
#include <cstring>

namespace aruco3cuda::apriltag_small_cpu::detail {

// Recognize an axis-aligned digital tag on a uniform field. Every input pixel
// must match a reconstructed row before a positive or negative is conclusive.
class UniformRaster {
public:
    bool read(const cv::Mat& gray, std::vector<Detection>& results) {
        const auto* blank = gray.ptr<std::uint8_t>();
        const auto background = blank[0];
        if (!std::all_of(blank + 1, blank + gray.cols, [&](auto p) { return p == background; }))
            return false;
        int top = 1;
        while (top < gray.rows && !std::memcmp(blank, gray.ptr(top), gray.cols))
            ++top;
        if (top == gray.rows)
            return true;
        const auto* first = gray.ptr<std::uint8_t>(top);
        int left = 0, right = gray.cols;
        while (first[left] == background)
            ++left;
        while (first[right - 1] == background)
            --right;
        const bool halo = first[left] > background;
        const int blocks = halo ? 10 : 8;
        const int span = right - left;
        if (span < 2 * blocks || span % blocks || top + span > gray.rows)
            return false;
        const int cell = span / blocks;
        const int offset = halo ? cell : 0;
        const auto black = gray.ptr<std::uint8_t>(top + offset)[left + offset];
        int white = halo ? first[left] : -1;
        if (black >= background || (halo && white - black < 24))
            return false;
        line_.assign(gray.cols, background);
        MarkerCode word = 0;
        for (int y = 0; y < blocks; ++y) {
            const auto* row = gray.ptr<std::uint8_t>(top + y * cell);
            for (int x = 0; x < blocks; ++x) {
                const auto value = row[left + x * cell];
                const int u = x - int(halo), v = y - int(halo);
                if (u < 0 || v < 0 || u > 7 || v > 7) {
                    if (value != white)
                        return false;
                } else if (value != black) {
                    if (u == 0 || v == 0 || u == 7 || v == 7 || value - black < 24 ||
                        (white >= 0 && value != white))
                        return false;
                    white = value;
                    word |= MarkerCode{1} << ((v - 1) * 6 + u - 1);
                }
                std::fill_n(line_.begin() + left + x * cell, cell, value);
            }
            for (int j = 0; j < cell; ++j)
                if (std::memcmp(gray.ptr(top + y * cell + j), line_.data(), gray.cols))
                    return false;
        }
        for (int y = top + span; y < gray.rows; ++y)
            if (std::memcmp(gray.ptr(y), blank, gray.cols))
                return false;
        if (white < 0)
            return false;
        const auto [code, distance] = CodeIndex::match(word, Config{}.maximum_corrected_bits_);
        if (code >= 0) {
            const Point origin(left + offset - .5F, top + offset - .5F);
            const float side = 8 * cell;
            Quad q{origin, origin + Point(side, 0), origin + Point(side, side),
                   origin + Point(0, side)};
            std::rotate(q.begin(), q.begin() + (4 - code % 4) % 4, q.end());
            results.push_back({code / 4, q, distance});
        }
        return true;
    }

private:
    std::vector<std::uint8_t> line_;
};

}  // namespace aruco3cuda::apriltag_small_cpu::detail
