// SPDX-License-Identifier: Apache-2.0
// Tracing and vectorized scan adapted from OpenCV contours_truco.cpp at
// 0654a42e19215ef25b1d367d822f3c630447e7c7, under Apache-2.0; see NOTICE.
// This version uses one sequential scan per independent threshold window,
// reusable storage, and a callback before output materialization.
#pragma once

#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>
#if defined(__aarch64__) && defined(__AARCH64EL__)
#include "cpu_gray_bits.hpp"
#endif

namespace aruco3cuda::cpu::detail {

/// Trace an isolated binary image and visit its paths in discovery order.
/// Reverse the collected paths for OpenCV RETR_LIST order.
/// Ownership applies to all public member functions. Callers own the source and any saved paths.
/// Synchronization applies to all public member functions. Calls run on the caller's thread and
/// each concurrent caller needs its own scanner.
/// Example input is a threshold image containing separated printed borders.
/// Example output is the sequence of outer and inner paths passed to a callback.
class ContourScanner {
public:
    template <cv::ContourApproximationModes approximation, bool packed_scan = false,
              bool skip_single_pixel_holes = false, int kMinimumPackedColumns = 640, class Consumer>
    /// Scan the image with full points or with straight runs compressed to their ends.
    /// With skip_single_pixel_holes, omit 3x3 diamond outlines around a single background pixel.
    /// @param binary Current CV_8UC1 threshold image. Nonzero values indicate foreground.
    /// @param skip_isolated Skip visits for one-pixel foreground components when true.
    /// @param accept Receives a temporary path reference valid only during this call to the
    /// callback. A multi-point full path includes its repeated start at closure.
    /// @return No value. OpenCV reports invalid inputs with cv::Exception.
    /// Example input is a mask containing a single closed outline.
    /// Example output is an ordered contour passed to the caller's callback.
    void for_each(const cv::Mat& binary, bool skip_isolated, Consumer&& accept) {
        static_assert(approximation == cv::CHAIN_APPROX_NONE ||
                      approximation == cv::CHAIN_APPROX_SIMPLE);
        CV_Assert(!binary.empty() && binary.dims == 2 && binary.type() == CV_8UC1 &&
                  binary.cols <= 65536 && binary.rows <= 65536 &&
                  binary.total() <= static_cast<std::size_t>(std::numeric_limits<int>::max()));
        const auto stride = this->prepare(binary.size());
        // The scanner never writes its zero border. Every frame replaces the complete center.
        cv::Mat center = this->padded_(cv::Rect(1, 1, binary.cols, binary.rows));
        cv::threshold(binary, center, 0.0, 255.0, cv::THRESH_BINARY);
#if defined(__aarch64__) && defined(__AARCH64EL__)
        if constexpr (packed_scan) {
            if (binary.cols >= kMinimumPackedColumns) {
                this->trace_packed_rows<approximation, skip_single_pixel_holes>(skip_isolated,
                                                                                stride, accept);
                return;
            }
        }
#endif
        for (int row = 1; row < this->padded_.rows - 1; ++row) {
            std::uint8_t* pixels = this->padded_.ptr<std::uint8_t>(row);
            for (int column = 1; column < this->padded_.cols - 1;) {
                column = find_transition(pixels, this->padded_.cols, column, true);
                if (column >= this->padded_.cols) {
                    break;
                }
                if (pixels[column] == kForeground) {
                    if (skip_isolated && isolated(pixels + column, stride)) {
                        // Match the tracer's marking of a one-point contour rejected by length.
                        pixels[column] = kVisitedRight;
                    } else if (this->trace_one<approximation>(row, column, true)) {
                        accept(this->trace_);
                    }
                }
                column = find_transition(pixels, this->padded_.cols, column + 1, false);
                if (column >= this->padded_.cols) {
                    break;
                }
                if (pixels[column - 1] > kVisitedRight && row > 1 &&
                    !(skip_single_pixel_holes && mark_single_pixel_hole(pixels + column, stride)) &&
                    this->trace_one<approximation>(row, column - 1, false)) {
                    accept(this->trace_);
                }
            }
        }
    }

    template <cv::ContourApproximationModes approximation, bool skip_single_pixel_holes,
              class Consumer>
    /// Trace an isolated, packed foreground mask in discovery order.
    /// @tparam skip_single_pixel_holes Omit outlines whose box is exactly 3 by 3.
    /// @param size Extent in sampled pixels.
    /// @param input Foreground words in row order with the leftmost pixel in bit zero.
    /// Unused high bits are zero.
    /// @param accept Receives a temporary path reference, valid until the callback returns.
    /// @return No value. OpenCV reports invalid extents with cv::Exception.
    /// Example input is one row with its first eight bits set. Example output is its path.
    void for_each_packed(cv::Size size, const std::vector<std::uint64_t>& input,
                         Consumer&& accept) {
        static_assert(approximation == cv::CHAIN_APPROX_NONE ||
                      approximation == cv::CHAIN_APPROX_SIMPLE);
        CV_Assert(size.width > 0 && size.height > 0 && size.width <= 65536 &&
                  size.height <= 65536 &&
                  std::size_t(size.width) * size.height <=
                          static_cast<std::size_t>(std::numeric_limits<int>::max()));
        const int words = (size.width + 63) / 64;
        CV_Assert(input.size() == std::size_t(words) * size.height);
        const auto stride = prepare(size);
        static constexpr auto expanded = [] {
            std::array<std::array<std::uint8_t, 8>, 256> bytes{};
            for (int value = 0; value < 256; ++value)
                for (int i = 0; i < 8; ++i)
                    bytes[value][i] = (value & (1U << i)) ? 255 : 0;
            return bytes;
        }();
        for (int y = 0; y < size.height; ++y) {
            auto* dst = padded_.ptr<std::uint8_t>(y + 1) + 1;
            const auto* row = input.data() + std::size_t(y) * words;
            for (int j = 0; j < words; ++j) {
                const int x = 64 * j;
                const int count = std::min(64, size.width - x);
                const auto bits = row[j];
                if (count == 64) {
                    if (!bits || bits == ~std::uint64_t{0})
                        std::memset(dst + x, bits ? 255 : 0, 64);
                    else
                        for (int i = 0; i < 8; ++i)
                            std::memcpy(dst + x + 8 * i, expanded[(bits >> (8 * i)) & 255].data(),
                                        8);
                } else {
                    for (int i = 0; i < count; ++i)
                        dst[x + i] = ((bits >> i) & 1) ? 255 : 0;
                }
            }
        }
        for (int y = 0; y < size.height; ++y) {
            auto* pixels = padded_.ptr<std::uint8_t>(y + 1);
            const auto* row = input.data() + std::size_t(y) * words;
            std::uint64_t carry = 0;
            for (int j = 0; j < words; ++j) {
                const auto bits = row[j];
                const auto crossings = bits ^ ((bits << 1) | carry);
                visit_word<approximation, skip_single_pixel_holes>(y + 1, 1 + 64 * j, pixels, bits,
                                                                   crossings, true, stride, accept);
                carry = bits >> 63;
            }
            if (carry)
                visit_word<approximation, skip_single_pixel_holes>(y + 1, size.width + 1, pixels, 0,
                                                                   1, true, stride, accept);
        }
    }

private:
    std::ptrdiff_t prepare(cv::Size size) {
        const cv::Size padded_size(size.width + 2, size.height + 2);
        if (this->padded_.size() != padded_size) {
            // Construct before replacing the old owner so failed allocation cannot retain an
            // uninitialized border with the requested shape.
            this->padded_ = cv::Mat(padded_size, CV_8UC1, cv::Scalar(0));
        }
        const std::ptrdiff_t stride = static_cast<std::ptrdiff_t>(this->padded_.step[0]);
        for (std::size_t direction = 0; direction < 8U; ++direction) {
            this->offsets_[direction] = kDy[direction] * stride + kDx[direction];
            this->offsets_[direction + 8U] = this->offsets_[direction];
        }
        return stride;
    }

    template <cv::ContourApproximationModes approximation, bool skip_single_pixel_holes,
              class Consumer>
    void visit_word(int row, int x, std::uint8_t* pixels, std::uint64_t occupancy,
                    std::uint64_t transitions, bool skip_isolated, std::ptrdiff_t stride,
                    Consumer& accept) {
        while (transitions != 0) {
#if defined(__GNUC__) || defined(__clang__)
            const int bit = __builtin_ctzll(transitions);
#else
            int bit = 0;
            while (((transitions >> bit) & 1U) == 0U)
                ++bit;
#endif
            transitions &= transitions - 1;
            const int column = x + bit;
            if ((occupancy >> bit) & 1U) {
                if (pixels[column] != kForeground)
                    continue;
                if (skip_isolated && isolated(pixels + column, stride))
                    pixels[column] = kVisitedRight;
                else if (this->trace_one<approximation>(row, column, true))
                    accept(this->trace_);
            } else if (pixels[column - 1] > kVisitedRight && row > 1 &&
                       !(skip_single_pixel_holes &&
                         mark_single_pixel_hole(pixels + column, stride)) &&
                       this->trace_one<approximation>(row, column - 1, false)) {
                accept(this->trace_);
            }
        }
    }
#if defined(__aarch64__) && defined(__AARCH64EL__)
    template <cv::ContourApproximationModes approximation, bool skip_single_pixel_holes,
              class Consumer>
    void trace_packed_rows(bool skip_isolated, std::ptrdiff_t stride, Consumer& accept) {
        const GrayComparisonBits pack;
        const auto state_bit = vdupq_n_u8(64);
        const int width = this->padded_.cols;
        for (int row = 1; row < this->padded_.rows - 1; ++row) {
            auto* pixels = this->padded_.ptr<std::uint8_t>(row);
            std::uint64_t carry = 0;
            // Tracing marks foreground bytes without changing their occupancy.
            // The transition positions stay valid while each trace updates its visited state.
            const auto visit = [&](int x, std::uint64_t occupancy) {
                std::uint64_t transitions = occupancy ^ ((occupancy << 1) | carry);
                carry = occupancy >> 63;
                this->visit_word<approximation, skip_single_pixel_holes>(
                        row, x, pixels, occupancy, transitions, skip_isolated, stride, accept);
            };
            int x = 0;
            for (; x + 64 <= width; x += 64) {
                const auto* current = pixels + x;
                visit(x, pack.pack(vtstq_u8(vld1q_u8(current), state_bit),
                                   vtstq_u8(vld1q_u8(current + 16), state_bit),
                                   vtstq_u8(vld1q_u8(current + 32), state_bit),
                                   vtstq_u8(vld1q_u8(current + 48), state_bit)));
            }
            if (x < width) {
                std::uint64_t occupancy = 0;
                for (int i = 0; i < width - x; ++i)
                    occupancy |= std::uint64_t{pixels[x + i] != 0U} << i;
                visit(x, occupancy);
            }
        }
    }
#endif

    static constexpr std::uint8_t kForeground = 255U;
    static constexpr std::uint8_t kVisitedRight = 100U;
    static constexpr std::uint8_t kVisited = 200U;
    static constexpr std::array<int, 8> kDx = {1, 1, 0, -1, -1, -1, 0, 1};
    static constexpr std::array<int, 8> kDy = {0, -1, -1, -1, 0, 1, 1, 1};

    static int find_transition(const std::uint8_t* row, int width, int column, bool foreground) {
#if defined(__aarch64__) && defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && \
        (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
        // All nonzero states (foreground and both visited states) have bit 6 set.
        static_assert((kForeground & kVisited & kVisitedRight & 64U) != 0);
        constexpr std::uint64_t kStateBit = 0x4040404040404040ULL;
        constexpr int kWordBytes = sizeof(std::uint64_t);
        for (; column <= width - kWordBytes; column += kWordBytes) {
            std::uint64_t word;
            std::memcpy(&word, row + column, sizeof(word));
            const std::uint64_t match = (foreground ? word : ~word) & kStateBit;
            if (match != 0U)
                return column + __builtin_ctzll(match) / 8;
        }
#elif (CV_SIMD || CV_SIMD_SCALABLE)
        const cv::v_uint8 zero = cv::vx_setzero_u8();
        const int lanes = cv::VTraits<cv::v_uint8>::vlanes();
        for (; column <= width - lanes; column += lanes) {
            const cv::v_uint8 pixels = cv::vx_load(row + column);
            const cv::v_uint8 mask = foreground ? cv::v_ne(pixels, zero) : cv::v_eq(pixels, zero);
            if (cv::v_check_any(mask)) {
                return column + cv::v_scan_forward(mask);
            }
        }
#endif
        for (; column < width && ((row[column] != 0U) != foreground); ++column) {
        }
        return column;
    }
    // The padded image always supplies all eight neighbors of a foreground pixel.
    static bool isolated(const std::uint8_t* p, std::ptrdiff_t stride) {
        const auto* upper = p - stride;
        const auto* lower = p + stride;
        std::uint16_t upper_left, lower_left;
        std::memcpy(&upper_left, upper - 1, sizeof(upper_left));
        std::memcpy(&lower_left, lower - 1, sizeof(lower_left));
        return (upper_left | lower_left | upper[1] | lower[1] | p[-1] | p[1]) == 0U;
    }

    // The left sample is an eligible hole start. For an isolated background
    // pixel, the trace visits west, north, east and south, then closes at west.
    // Preserve the exact marks that the tracer leaves for subsequent outlines.
    static bool mark_single_pixel_hole(std::uint8_t* zero, std::ptrdiff_t stride) {
        auto* north = zero - stride;
        auto* south = zero + stride;
        auto* east = zero + 1;
        if (*north == 0U || *east == 0U || *south == 0U)
            return false;
        zero[-1] = kVisitedRight;
        for (auto* sample : {north, east, south})
            if (*sample == kForeground)
                *sample = kVisited;
        return true;
    }

    template <cv::ContourApproximationModes approximation>
    bool trace_one(int row, int column, bool external) {
        this->trace_.clear();
        int x = column;
        int y = row;
        int start_direction = -1;
        int search = external ? 5 : 1;
        std::uint8_t* current = this->padded_.ptr<std::uint8_t>(row) + column;
        const std::uint8_t* start = current;
        int direction = -1;
        bool first = true;

        if (!external) {
            int neighbor = 0;
            for (; neighbor < 8; ++neighbor) {
                const int index = search + neighbor;
                if (*(current + this->offsets_[static_cast<std::size_t>(index)]) == 0U) {
                    continue;
                }
                if (x + kDx[static_cast<std::size_t>(index & 7)] != column + 1 ||
                    y + kDy[static_cast<std::size_t>(index & 7)] != row - 1) {
                    return false;
                }
                break;
            }
            if (neighbor == 8) {
                return false;
            }
        }

        int previous_direction = -1;
        for (;;) {
            if constexpr (approximation == cv::CHAIN_APPROX_NONE) {
                this->trace_.emplace_back(x - 1, y - 1);
            }
            for (int neighbor = 0; neighbor < 8; ++neighbor) {
                const int index = search + neighbor;
                std::uint8_t* next = current + this->offsets_[static_cast<std::size_t>(index)];
                if (*next == 0U) {
                    continue;
                }
                direction = index & 7;
                if constexpr (approximation == cv::CHAIN_APPROX_SIMPLE) {
                    if (first || direction != previous_direction)
                        this->trace_.emplace_back(x - 1, y - 1);
                }
                y += kDy[static_cast<std::size_t>(direction)];
                x += kDx[static_cast<std::size_t>(direction)];
                if (search <= 1 || direction <= search - 2) {
                    *current = kVisitedRight;
                } else if (*current == kForeground) {
                    *current = kVisited;
                }
                if (current == start && !first && direction == start_direction) {
                    if constexpr (approximation == cv::CHAIN_APPROX_SIMPLE) {
                        // The closed path has no extra endpoint. Drop its starting sample
                        // when the incoming and outgoing steps lie on the same run.
                        if (direction != previous_direction)
                            this->trace_.pop_back();
                        else if (!this->trace_.empty())
                            this->trace_.erase(this->trace_.begin());
                    }
                    return true;
                }
                if constexpr (approximation == cv::CHAIN_APPROX_SIMPLE)
                    previous_direction = direction;
                current = next;
                search = (direction + 6) & 7;
                break;
            }
            if (first) {
                if (direction == -1) {
                    if constexpr (approximation == cv::CHAIN_APPROX_SIMPLE)
                        this->trace_.emplace_back(x - 1, y - 1);
                    *current = kVisitedRight;
                    return true;
                }
                start_direction = direction;
                first = false;
            }
        }
    }

    cv::Mat padded_;
    std::array<std::ptrdiff_t, 16> offsets_{};
    std::vector<cv::Point> trace_;
};

}  // namespace aruco3cuda::cpu::detail
