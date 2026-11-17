// SPDX-License-Identifier: Apache-2.0
// Tracing and vectorized scan adapted from OpenCV contours_truco.cpp at
// 0654a42e19215ef25b1d367d822f3c630447e7c7, under Apache-2.0; see NOTICE.
// This version uses one sequential scan per independent threshold window,
// reusable storage, and quadrilateral filtering before output materialization.
#include "cpu_contours.hpp"

#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace aruco3cuda::cpu::detail {
namespace {
constexpr std::uint8_t kForeground = 255U;
constexpr std::uint8_t kVisitedRight = 100U;
constexpr std::uint8_t kVisited = 200U;
constexpr std::array<int, 8> kDx = {1, 1, 0, -1, -1, -1, 0, 1};
constexpr std::array<int, 8> kDy = {0, -1, -1, -1, 0, 1, 1, 1};

int find_transition(const std::uint8_t* row, int width, int column, bool foreground) {
#if (CV_SIMD || CV_SIMD_SCALABLE)
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
}  // namespace

bool ContourWorkspace::trace(int row, int column, bool external) {
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

    for (;;) {
        this->trace_.emplace_back(x - 1, y - 1);
        for (int neighbor = 0; neighbor < 8; ++neighbor) {
            const int index = search + neighbor;
            std::uint8_t* next = current + this->offsets_[static_cast<std::size_t>(index)];
            if (*next == 0U) {
                continue;
            }
            direction = index & 7;
            y += kDy[static_cast<std::size_t>(direction)];
            x += kDx[static_cast<std::size_t>(direction)];
            if (search <= 1 || direction <= search - 2) {
                *current = kVisitedRight;
            } else if (*current == kForeground) {
                *current = kVisited;
            }
            if (current == start && !first && direction == start_direction) {
                return true;
            }
            current = next;
            search = (direction + 6) & 7;
            break;
        }
        if (first) {
            if (direction == -1) {
                *current = kVisitedRight;
                return true;
            }
            start_direction = direction;
            first = false;
        }
    }
}

void ContourWorkspace::append_quad(const DetectorConfig& config, int longest, std::size_t minimum,
                                   std::size_t maximum,
                                   std::vector<std::vector<cv::Point2f>>& candidates,
                                   std::vector<std::vector<cv::Point>>& contours) {
    if (this->trace_.size() > 1U && this->trace_.back() == this->trace_.front()) {
        this->trace_.pop_back();
    }
    const std::size_t length = this->trace_.size();
    if (length < minimum || length > maximum) {
        return;
    }
    cv::approxPolyDP(this->trace_, this->approximation_,
                     static_cast<double>(length) * config.polygonal_approx_accuracy_rate_, true);
    if (this->approximation_.size() != 4U || !cv::isContourConvex(this->approximation_)) {
        return;
    }
    double minimum_distance_squared = static_cast<double>(longest) * longest;
    for (std::size_t corner = 0; corner < 4U; ++corner) {
        const cv::Point& first = this->approximation_[corner];
        const cv::Point& second = this->approximation_[(corner + 1U) % 4U];
        const double dx = first.x - second.x;
        const double dy = first.y - second.y;
        minimum_distance_squared = std::min(minimum_distance_squared, dx * dx + dy * dy);
    }
    const double minimum_distance = static_cast<double>(length) * config.min_corner_distance_rate_;
    if (minimum_distance_squared < minimum_distance * minimum_distance) {
        return;
    }
    std::vector<cv::Point2f> candidate(4);
    for (std::size_t corner = 0; corner < 4U; ++corner) {
        candidate[corner] = cv::Point2f(static_cast<float>(this->approximation_[corner].x),
                                        static_cast<float>(this->approximation_[corner].y));
    }
    candidates.push_back(std::move(candidate));
    contours.push_back(this->trace_);
}

void ContourWorkspace::append(const cv::Mat& binary, const DetectorConfig& config,
                              std::vector<std::vector<cv::Point2f>>& candidates,
                              std::vector<std::vector<cv::Point>>& contours) {
    CV_Assert(!binary.empty() && binary.dims == 2 && binary.type() == CV_8UC1 &&
              binary.cols <= 65536 && binary.rows <= 65536 &&
              binary.total() <= static_cast<std::size_t>(std::numeric_limits<int>::max()));
    const int longest = std::max(binary.cols, binary.rows);
    std::size_t minimum = static_cast<std::size_t>(config.min_marker_perimeter_rate_ * longest);
    const std::size_t maximum =
            static_cast<std::size_t>(config.max_marker_perimeter_rate_ * longest);
    if (config.use_aruco3_detection_ && config.min_side_length_canonical_img_px_ != 0) {
        minimum = static_cast<std::size_t>(4 * config.min_side_length_canonical_img_px_);
    }
    const cv::Size padded_size(binary.cols + 2, binary.rows + 2);
    if (this->padded_.size() != padded_size) {
        // Construct before replacing the old owner so failed allocation cannot retain an
        // uninitialized border with the requested shape.
        this->padded_ = cv::Mat(padded_size, CV_8UC1, cv::Scalar(0));
    }
    // The scanner never writes its zero border. Every frame replaces the complete center.
    cv::Mat center = this->padded_(cv::Rect(1, 1, binary.cols, binary.rows));
    cv::threshold(binary, center, 0.0, 255.0, cv::THRESH_BINARY);
    const std::ptrdiff_t stride = static_cast<std::ptrdiff_t>(this->padded_.step);
    for (std::size_t direction = 0; direction < 8U; ++direction) {
        this->offsets_[direction] = kDy[direction] * stride + kDx[direction];
        this->offsets_[direction + 8U] = this->offsets_[direction];
    }
    const std::size_t start_candidates = candidates.size();
    const std::size_t start_contours = contours.size();
    for (int row = 1; row < this->padded_.rows - 1; ++row) {
        std::uint8_t* pixels = this->padded_.ptr<std::uint8_t>(row);
        for (int column = 1; column < this->padded_.cols - 1;) {
            column = find_transition(pixels, this->padded_.cols, column, true);
            if (column >= this->padded_.cols) {
                break;
            }
            if (pixels[column] == kForeground && this->trace(row, column, true)) {
                this->append_quad(config, longest, minimum, maximum, candidates, contours);
            }
            column = find_transition(pixels, this->padded_.cols, column + 1, false);
            if (column >= this->padded_.cols) {
                break;
            }
            if (pixels[column - 1] > kVisitedRight && row > 1 &&
                this->trace(row, column - 1, false)) {
                this->append_quad(config, longest, minimum, maximum, candidates, contours);
            }
        }
    }
    // The original finder returns reverse discovery order. Filter first and
    // reverse only this call's appended range to preserve both ordering and ownership.
    std::reverse(candidates.begin() + static_cast<std::ptrdiff_t>(start_candidates),
                 candidates.end());
    std::reverse(contours.begin() + static_cast<std::ptrdiff_t>(start_contours), contours.end());
}

}  // namespace aruco3cuda::cpu::detail
