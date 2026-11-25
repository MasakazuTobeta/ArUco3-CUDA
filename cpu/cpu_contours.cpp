// SPDX-License-Identifier: Apache-2.0
// Tracing and vectorized scan adapted from OpenCV contours_truco.cpp at
// 0654a42e19215ef25b1d367d822f3c630447e7c7, under Apache-2.0; see NOTICE.
// This version uses one sequential scan per independent threshold window,
// reusable storage, and quadrilateral filtering before output materialization.
#include "cpu_contours.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>

namespace aruco3cuda::cpu::detail {
void ContourWorkspace::append_quad(std::vector<cv::Point>& trace, const DetectorConfig& config,
                                   int longest, std::size_t minimum, std::size_t maximum,
                                   std::vector<std::vector<cv::Point2f>>& candidates,
                                   std::vector<std::vector<cv::Point>>& contours) {
    if (trace.size() > 1U && trace.back() == trace.front()) {
        trace.pop_back();
    }
    const std::size_t length = trace.size();
    if (length < minimum || length > maximum) {
        return;
    }
    cv::approxPolyDP(trace, this->approximation_,
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
    contours.push_back(trace);
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
    const std::size_t start_candidates = candidates.size();
    const std::size_t start_contours = contours.size();
    this->scanner_.for_each<cv::CHAIN_APPROX_NONE>(
            binary, minimum > 1, [&](std::vector<cv::Point>& trace) {
                this->append_quad(trace, config, longest, minimum, maximum, candidates, contours);
            });
    // The original finder returns reverse discovery order. Filter first and
    // reverse only this call's appended range to preserve both ordering and ownership.
    std::reverse(candidates.begin() + static_cast<std::ptrdiff_t>(start_candidates),
                 candidates.end());
    std::reverse(contours.begin() + static_cast<std::ptrdiff_t>(start_contours), contours.end());
}

}  // namespace aruco3cuda::cpu::detail
