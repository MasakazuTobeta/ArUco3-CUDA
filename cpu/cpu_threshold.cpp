// SPDX-License-Identifier: Apache-2.0
#include "cpu_threshold.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace aruco3cuda::cpu::detail {
namespace {

void threshold_row(const std::uint8_t* __restrict__ source, std::uint8_t* __restrict__ destination,
                   const std::int32_t* __restrict__ top_left,
                   const std::int32_t* __restrict__ top_right,
                   const std::int32_t* __restrict__ bottom_left,
                   const std::int32_t* __restrict__ bottom_right, int width, int area, int delta) {
    for (int x = 0; x < width; ++x) {
        const int sum = (bottom_right[x] - top_right[x]) - (bottom_left[x] - top_left[x]);
        // An odd square area has no half-integer mean. Comparing
        // against the rounding boundary avoids division and materializing
        // an intermediate mean image, while preserving OpenCV's decision.
        const int boundary = (static_cast<int>(source[x]) + delta) * area - area / 2;
        destination[x] = sum >= boundary ? 255U : 0U;
    }
}

}  // namespace

void ThresholdWorkspace::apply(const cv::Mat& gray, const std::vector<int>& window_sizes,
                               double constant, std::vector<cv::Mat>* outputs,
                               int parallel_min_pixels, std::size_t max_prefix_bytes) {
    CV_Assert(outputs != nullptr && !gray.empty() && gray.dims == 2 && gray.type() == CV_8UC1);
    CV_Assert(std::isfinite(constant) && constant >= -255.0 && constant <= 255.0);
    CV_Assert(parallel_min_pixels >= 0);
    // A striped source is read again for the next halo. Preserve the original
    // pixels if an output aliases the input, including shallow ROI headers.
    const std::uintptr_t input_begin = reinterpret_cast<std::uintptr_t>(gray.datastart);
    const std::uintptr_t input_end = reinterpret_cast<std::uintptr_t>(gray.dataend);
    for (const cv::Mat& output : *outputs) {
        if (!output.empty()) {
            const std::uintptr_t output_begin = reinterpret_cast<std::uintptr_t>(output.datastart);
            const std::uintptr_t output_end = reinterpret_cast<std::uintptr_t>(output.dataend);
            if (input_begin < output_end && output_begin < input_end) {
                const cv::Mat source = gray.clone();
                this->apply(source, window_sizes, constant, outputs, parallel_min_pixels,
                            max_prefix_bytes);
                return;
            }
        }
    }
    int max_radius = 0;
    bool use_integral = true;
    for (const int configured_size : window_sizes) {
        CV_Assert(configured_size >= 3 && configured_size <= 4097);
        const int side = configured_size | 1;
        max_radius = std::max(max_radius, side / 2);
        // With larger areas float normalization in some OpenCV SIMD paths
        // can move a nearly-half-integer mean to an exact rounding tie.
        use_integral = use_integral && side <= 127;
    }
    outputs->resize(window_sizes.size());
    if (window_sizes.empty()) {
        if (max_prefix_bytes > 0U && !this->integral_.empty() &&
            this->integral_.total() * this->integral_.elemSize() > max_prefix_bytes) {
            this->integral_.release();
            this->padded_.release();
        }
        return;
    }
    const std::int64_t padded_width = static_cast<std::int64_t>(gray.cols) + 2 * max_radius;
    // A stripe's halo participates in its integral, so leave room for both
    // replicated edges. Deriving this limit from the accumulator range avoids
    // signed prefix-sum overflow even when the whole image does not fit.
    const std::int64_t max_padded_rows =
            std::numeric_limits<std::int32_t>::max() / (255 * padded_width);
    std::int64_t max_stripe_rows = max_padded_rows - 2 * max_radius;
    if (max_prefix_bytes > 0U) {
        const std::uint64_t row_bytes =
                static_cast<std::uint64_t>(padded_width + 1) * sizeof(std::int32_t);
        // Dividing first keeps SIZE_MAX safe before converting the row count
        // to signed arithmetic. The halo and the integral's zero row also fit.
        const std::uint64_t budget_rows = max_prefix_bytes / row_bytes;
        const std::int64_t budget_output_rows =
                static_cast<std::int64_t>(budget_rows) - (2 * max_radius + 1);
        max_stripe_rows = std::min(max_stripe_rows, budget_output_rows);
    }
    use_integral = use_integral && max_stripe_rows > 0;
    if (!use_integral) {
        if (max_prefix_bytes > 0U) {
            // Reconfiguration to a tiny budget must not retain an old large
            // owner even though the fallback does not need our scratch images.
            this->integral_.release();
            this->padded_.release();
        }
        for (std::size_t i = 0; i < window_sizes.size(); ++i) {
            cv::adaptiveThreshold(gray, (*outputs)[i], 255.0, cv::ADAPTIVE_THRESH_MEAN_C,
                                  cv::THRESH_BINARY_INV, window_sizes[i] | 1, constant);
        }
        return;
    }
    const int delta = static_cast<int>(std::floor(constant));
    for (std::size_t i = 0; i < window_sizes.size(); ++i) {
        (*outputs)[i].create(gray.size(), CV_8UC1);
    }
    const std::int64_t stripe_count =
            (static_cast<std::int64_t>(gray.rows) + max_stripe_rows - 1) / max_stripe_rows;
    const int stripe_height = static_cast<int>((gray.rows + stripe_count - 1) / stripe_count);
    // Uniform stripes keep the last stripe near the others' size. The final
    // active view can be smaller while retaining the same owned allocation.
    if (max_prefix_bytes == 0U) {
        cv::copyMakeBorder(gray, this->padded_, max_radius, max_radius, max_radius, max_radius,
                           cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);
    } else {
        this->padded_.create(stripe_height + 2 * max_radius, gray.cols + 2 * max_radius, CV_8UC1);
    }
    this->integral_.create(stripe_height + 2 * max_radius + 1, this->padded_.cols + 1, CV_32SC1);
    for (int stripe_y = 0; stripe_y < gray.rows; stripe_y += stripe_height) {
        const int rows = std::min(stripe_height, gray.rows - stripe_y);
        cv::Mat padded_stripe;
        if (max_prefix_bytes == 0U) {
            padded_stripe = this->padded_.rowRange(stripe_y, stripe_y + rows + 2 * max_radius);
        } else {
            const int input_top = std::max(0, stripe_y - max_radius);
            const int input_bottom = std::min(gray.rows, stripe_y + rows + max_radius);
            padded_stripe = this->padded_.rowRange(0, rows + 2 * max_radius);
            cv::copyMakeBorder(gray.rowRange(input_top, input_bottom), padded_stripe,
                               std::max(0, max_radius - stripe_y),
                               std::max(0, stripe_y + rows + max_radius - gray.rows), max_radius,
                               max_radius, cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);
        }
        cv::Mat integral = this->integral_.rowRange(0, rows + 2 * max_radius + 1);
        cv::integral(padded_stripe, integral, CV_32S);
        const auto process_rows = [&](const cv::Range& range) {
            for (std::size_t i = 0; i < window_sizes.size(); ++i) {
                const int side = window_sizes[i] | 1;
                const int area = side * side;
                const int offset = max_radius - side / 2;
                cv::Mat& output = (*outputs)[i];
                for (int y = range.start; y < range.end; ++y) {
                    // Read the owned padded copy, which cannot alias any output plane.
                    const std::uint8_t* source =
                            padded_stripe.ptr<std::uint8_t>(y + max_radius) + max_radius;
                    std::uint8_t* destination = output.ptr<std::uint8_t>(stripe_y + y);
                    const std::int32_t* top_left = integral.ptr<std::int32_t>(y + offset) + offset;
                    const std::int32_t* bottom_left =
                            integral.ptr<std::int32_t>(y + offset + side) + offset;
                    threshold_row(source, destination, top_left, top_left + side, bottom_left,
                                  bottom_left + side, gray.cols, area, delta);
                }
            }
        };
        if (cv::getNumThreads() > 1 &&
            static_cast<std::size_t>(rows) * static_cast<std::size_t>(gray.cols) >=
                    static_cast<std::size_t>(parallel_min_pixels)) {
            // Every worker owns a disjoint row interval in every output plane.
            // Prefix sums and the padded source remain immutable until the join.
            cv::parallel_for_(cv::Range(0, rows), process_rows, cv::getNumThreads());
        } else {
            process_rows(cv::Range(0, rows));
        }
    }
}

}  // namespace aruco3cuda::cpu::detail
