// SPDX-License-Identifier: Apache-2.0
// Monitored refinement arithmetic is adapted from OpenCV cornersubpix.cpp,
// commit 0654a42e19215ef25b1d367d822f3c630447e7c7, under BSD-3-Clause.
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// The full BSD notice is retained in NOTICE. Other code in this file is original.
#include "cpu_sparse_pyramid.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace aruco3cuda::cpu::detail {

void SparsePyramid::reset(int tile_side_px, double full_level_ratio) {
    CV_Assert(tile_side_px > 0 && tile_side_px <= 65536 && std::isfinite(full_level_ratio) &&
              full_level_ratio > 0.0 && full_level_ratio <= 1.0);
    this->tile_side_px_ = tile_side_px;
    this->full_level_ratio_ = full_level_ratio;
    for (CachedRegion& cache : this->levels_) {
        std::fill(cache.valid_.begin(), cache.valid_.end(), 0U);
        cache.valid_count_ = 0U;
    }
}

cv::Mat SparsePyramid::region(const cv::Mat& input, const std::vector<cv::Size>& sizes, int level,
                              const cv::Rect& requested) {
    CV_Assert(input.type() == CV_8UC1 && (level == 0 || !input.isSubmatrix()));
    CV_Assert(level >= 0 && static_cast<std::size_t>(level) < sizes.size());
    const cv::Size size = sizes[static_cast<std::size_t>(level)];
    CV_Assert(requested.width > 0 && requested.height > 0 &&
              (requested & cv::Rect(cv::Point(), size)) == requested);
    if (level == 0) {
        return input(requested);
    }
    if (this->levels_.size() < sizes.size()) {
        this->levels_.resize(sizes.size());
    }
    CachedRegion& cache = this->levels_[static_cast<std::size_t>(level)];
    const int tile = this->tile_side_px_;
    const int tiles_x = (size.width + tile - 1) / tile;
    const int tiles_y = (size.height + tile - 1) / tile;
    if (cache.pixels_.size() != size || cache.tile_side_px_ != tile) {
        // Zero is an invalid tile size until every allocation has succeeded.
        cache.tile_side_px_ = 0;
        cache.pixels_.create(size, CV_8UC1);
        cache.valid_.assign(static_cast<std::size_t>(tiles_x) * tiles_y, 0U);
        cache.tiles_x_ = tiles_x;
        cache.valid_count_ = 0U;
        cache.tile_side_px_ = tile;
    }
    const int first_x = requested.x / tile;
    const int first_y = requested.y / tile;
    const int last_x = (requested.br().x - 1) / tile;
    const int last_y = (requested.br().y - 1) / tile;
    std::size_t missing = 0U;
    for (int ty = first_y; ty <= last_y; ++ty) {
        for (int tx = first_x; tx <= last_x; ++tx) {
            missing += cache.valid_[static_cast<std::size_t>(ty) * tiles_x + tx] == 0U ? 1U : 0U;
        }
    }
    if (missing == 0U) {
        return cache.pixels_(requested);
    }
    if (static_cast<double>(cache.valid_count_ + missing) >=
        static_cast<double>(cache.valid_.size()) * this->full_level_ratio_) {
        const cv::Mat source =
                this->region(input, sizes, level - 1,
                             cv::Rect(cv::Point(), sizes[static_cast<std::size_t>(level - 1)]));
        cv::pyrDown(source, cache.pixels_, size);
        std::fill(cache.valid_.begin(), cache.valid_.end(), 1U);
        cache.valid_count_ = cache.valid_.size();
        return cache.pixels_(requested);
    }
    const cv::Size source_size = sizes[static_cast<std::size_t>(level - 1)];
    // A rectangular request is generated as one operation rather than a dispatch per
    // tile. Already valid overlap can be rewritten safely within this worker's cache.
    const int x = first_x * tile;
    const int y = first_y * tile;
    const cv::Rect extent(x, y, std::min(size.width, (last_x + 1) * tile) - x,
                          std::min(size.height, (last_y + 1) * tile) - y);
    const int source_x = std::max(0, 2 * extent.x - 2);
    const int source_y = std::max(0, 2 * extent.y - 2);
    const int source_end_x = std::min(source_size.width, 2 * (extent.br().x - 1) + 3);
    const int source_end_y = std::min(source_size.height, 2 * (extent.br().y - 1) + 3);
    const cv::Rect source_rect(source_x, source_y, source_end_x - source_x,
                               source_end_y - source_y);
    const cv::Mat source = this->region(input, sizes, level - 1, source_rect);
    cv::pyrDown(source, cache.downsampled_, cv::Size((source.cols + 1) / 2, (source.rows + 1) / 2));
    const cv::Rect output_rect(extent.x - source_x / 2, extent.y - source_y / 2, extent.width,
                               extent.height);
    cache.downsampled_(output_rect).copyTo(cache.pixels_(extent));
    for (int ty = first_y; ty <= last_y; ++ty) {
        for (int tx = first_x; tx <= last_x; ++tx) {
            cache.valid_[static_cast<std::size_t>(ty) * tiles_x + tx] = 1U;
        }
    }
    cache.valid_count_ += missing;
    return cache.pixels_(requested);
}

void SparsePyramid::warp(const cv::Mat& input, const std::vector<cv::Size>& sizes, int level,
                         const std::vector<cv::Point2f>& corners, cv::Size canonical_size,
                         cv::Mat* canonical) {
    CV_Assert(canonical != nullptr && corners.size() == 4U && canonical_size.width > 0 &&
              canonical_size.height > 0 && level >= 0 &&
              static_cast<std::size_t>(level) < sizes.size());
    const cv::Size checked_source_size = sizes[static_cast<std::size_t>(level)];
    std::array<cv::Point2f, 4> destination = {
            cv::Point2f(0.0F, 0.0F),
            cv::Point2f(static_cast<float>(canonical_size.width - 1), 0.0F),
            cv::Point2f(static_cast<float>(canonical_size.width - 1),
                        static_cast<float>(canonical_size.height - 1)),
            cv::Point2f(0.0F, static_cast<float>(canonical_size.height - 1))};
    const cv::Mat forward = cv::getPerspectiveTransform(corners.data(), destination.data());
    if (checked_source_size.width >= std::numeric_limits<short>::max() ||
        checked_source_size.height >= std::numeric_limits<short>::max() ||
        canonical_size.width >= std::numeric_limits<short>::max() ||
        canonical_size.height >= std::numeric_limits<short>::max()) {
        // HAL implementations can differ in their large-image limits. Generate the
        // complete source and let the installed warp preserve its own behavior.
        (void)this->region(input, sizes, level, cv::Rect(cv::Point(), checked_source_size));
        const cv::Mat& full_source =
                level == 0 ? input : this->levels_[static_cast<std::size_t>(level)].pixels_;
        cv::warpPerspective(full_source, *canonical, forward, canonical_size, cv::INTER_NEAREST);
        return;
    }
    std::array<double, 9> coefficients{};
    cv::Mat inverse(3, 3, CV_64FC1, coefficients.data());
    const double determinant = cv::invert(forward, inverse, cv::DECOMP_LU);
    const cv::Size source_size = sizes[static_cast<std::size_t>(level)];
    cv::Rect bounds(cv::Point(), source_size);
    // With a nonzero denominator of one sign, a projective coordinate's extrema
    // on the canonical rectangle occur at its vertices. Bound FP evaluation error
    // conservatively; poles or ill-conditioned transforms use a complete level.
    bool bounded = determinant != 0.0 && std::isfinite(determinant);
    long double minimum_x = std::numeric_limits<long double>::infinity();
    long double minimum_y = minimum_x;
    long double maximum_x = -minimum_x;
    long double maximum_y = -minimum_y;
    long double minimum_w = minimum_x;
    int sign = 0;
    for (const cv::Point2f& point : destination) {
        const long double x = point.x;
        const long double y = point.y;
        const long double w = coefficients[6] * x + coefficients[7] * y + coefficients[8];
        const int current_sign = (w > 0.0L) - (w < 0.0L);
        bounded = bounded && current_sign != 0 && (sign == 0 || sign == current_sign);
        sign = current_sign;
        minimum_w = std::min(minimum_w, std::fabs(w));
        if (w == 0.0L || !std::isfinite(w)) {
            bounded = false;
            continue;
        }
        const long double sx = (coefficients[0] * x + coefficients[1] * y + coefficients[2]) / w;
        const long double sy = (coefficients[3] * x + coefficients[4] * y + coefficients[5]) / w;
        bounded = bounded && std::isfinite(sx) && std::isfinite(sy);
        minimum_x = std::min(minimum_x, sx);
        minimum_y = std::min(minimum_y, sy);
        maximum_x = std::max(maximum_x, sx);
        maximum_y = std::max(maximum_y, sy);
    }
    const long double width = canonical_size.width - 1;
    const long double height = canonical_size.height - 1;
    const long double max_n =
            std::max(std::fabs(coefficients[0]) * width + std::fabs(coefficients[1]) * height +
                             std::fabs(coefficients[2]),
                     std::fabs(coefficients[3]) * width + std::fabs(coefficients[4]) * height +
                             std::fabs(coefficients[5]));
    const long double max_w = std::fabs(coefficients[6]) * width +
                              std::fabs(coefficients[7]) * height + std::fabs(coefficients[8]);
    long double error = std::numeric_limits<long double>::infinity();
    if (minimum_w > 0.0L && std::isfinite(minimum_w)) {
        error = 16.0L * std::numeric_limits<double>::epsilon() *
                (max_n / minimum_w + (max_n / minimum_w) * (max_w / minimum_w));
    }
    bounded = bounded && std::isfinite(error) && error < 0.25L;
    if (bounded) {
        // warpPerspective's nearest map uses signed-short saturation before border handling.
        const auto saturated = [](long double value) {
            return std::max(-32768.0L, std::min(32767.0L, value));
        };
        const int left = static_cast<int>(std::floor(saturated(minimum_x))) - 2;
        const int top = static_cast<int>(std::floor(saturated(minimum_y))) - 2;
        const int right = static_cast<int>(std::ceil(saturated(maximum_x))) + 3;
        const int bottom = static_cast<int>(std::ceil(saturated(maximum_y))) + 3;
        bounds = cv::Rect(left, top, right - left, bottom - top) & bounds;
    }
    if (bounds.empty()) {
        canonical->create(canonical_size, CV_8UC1);
        canonical->setTo(0);
        return;
    }
    (void)this->region(input, sizes, level, bounds);
    const cv::Mat& source =
            level == 0 ? input : this->levels_[static_cast<std::size_t>(level)].pixels_;
    // Execute the actual global-coordinate warp after its complete footprint is valid.
    // This preserves platform-specific map rounding at half-integer coordinates.
    cv::warpPerspective(source, *canonical, forward, canonical_size, cv::INTER_NEAREST);
}

void SparsePyramid::refine(const cv::Mat& input, const std::vector<cv::Size>& sizes, int level,
                           std::vector<cv::Point2f>* corners, int radius, int max_iterations,
                           double epsilon) {
    CV_Assert(corners != nullptr && radius > 0 && level >= 0 &&
              static_cast<std::size_t>(level) < sizes.size());
    const cv::Size size = sizes[static_cast<std::size_t>(level)];
    CV_Assert(size.width >= radius * 2 + 5 && size.height >= radius * 2 + 5);
    const int side = radius * 2 + 1;
    const int patch_side = side + 2;
    auto mask_entry = this->masks_.find(radius);
    if (mask_entry == this->masks_.end()) {
        cv::Mat mask(side, side, CV_32FC1);
        for (int y = 0; y < side; ++y) {
            const float py = static_cast<float>(y - radius) / radius;
            const float vy = std::exp(-py * py);
            for (int x = 0; x < side; ++x) {
                const float px = static_cast<float>(x - radius) / radius;
                mask.ptr<float>(y)[x] = vy * std::exp(-px * px);
            }
        }
        mask_entry = this->masks_.emplace(radius, mask).first;
    }
    const float* mask = mask_entry->second.ptr<float>();
    this->patch_.create(patch_side, patch_side, CV_32FC1);
    const int iterations = std::min(std::max(max_iterations, 1), 100);
    const double accuracy = std::max(epsilon, 0.0) * std::max(epsilon, 0.0);
    for (cv::Point2f& point : *corners) {
        const cv::Point2f initial = point;
        CV_Assert(initial.x >= 0.0F && initial.y >= 0.0F && initial.x < size.width &&
                  initial.y < size.height);
        cv::Point2f current = initial;
        int iteration = 0;
        double error = 0.0;
        do {
            if (level > 0) {
                const int left = cvFloor(current.x - static_cast<float>(radius + 1));
                const int top = cvFloor(current.y - static_cast<float>(radius + 1));
                const cv::Rect footprint = cv::Rect(left, top, patch_side + 1, patch_side + 1) &
                                           cv::Rect(cv::Point(), size);
                (void)this->region(input, sizes, level, footprint);
            }
            const cv::Mat& pixels =
                    level == 0 ? input : this->levels_[static_cast<std::size_t>(level)].pixels_;
            cv::getRectSubPix(pixels, cv::Size(patch_side, patch_side), current, this->patch_,
                              CV_32F);
            double a = 0.0, b = 0.0, c = 0.0, bb1 = 0.0, bb2 = 0.0;
            const float* subpix = this->patch_.ptr<float>(1) + 1;
            int k = 0;
            for (int y = 0; y < side; ++y, subpix += side + 2) {
                const double py = y - radius;
                for (int x = 0; x < side; ++x, ++k) {
                    const double weight = mask[k];
                    // Preserve the float subtraction before widening used by cornerSubPix.
                    const double gx = subpix[x + 1] - subpix[x - 1];
                    const double gy = subpix[x + side + 2] - subpix[x - side - 2];
                    const double gxx = gx * gx * weight;
                    const double gxy = gx * gy * weight;
                    const double gyy = gy * gy * weight;
                    const double px = x - radius;
                    a += gxx;
                    b += gxy;
                    c += gyy;
                    bb1 += gxx * px + gxy * py;
                    bb2 += gxy * px + gyy * py;
                }
            }
            const double determinant = a * c - b * b;
            if (std::fabs(determinant) <= DBL_EPSILON * DBL_EPSILON) {
                break;
            }
            const double scale = 1.0 / determinant;
            cv::Point2f next;
            next.x = static_cast<float>(static_cast<double>(current.x) + c * scale * bb1 -
                                        b * scale * bb2);
            next.y = static_cast<float>(static_cast<double>(current.y) - b * scale * bb1 +
                                        a * scale * bb2);
            error = (next.x - current.x) * (next.x - current.x) +
                    (next.y - current.y) * (next.y - current.y);
            if (!(next.x >= 0.0F && next.y >= 0.0F && next.x < size.width &&
                  next.y < size.height)) {
                break;
            }
            current = next;
        } while (++iteration < iterations && error > accuracy);
        if (std::fabs(current.x - initial.x) > radius ||
            std::fabs(current.y - initial.y) > radius) {
            current = initial;
        }
        point = current;
    }
}

}  // namespace aruco3cuda::cpu::detail
