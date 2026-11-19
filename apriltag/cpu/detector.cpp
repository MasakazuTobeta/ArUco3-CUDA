// SPDX-License-Identifier: Apache-2.0
#include "aruco3cuda/apriltag/cpu/detector.hpp"

#include <apriltag.h>
#include <common/homography.h>
#include <tag16h5.h>
#include <tag25h9.h>
#include <tag36h10.h>
#include <tag36h11.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <system_error>

namespace aruco3cuda::apriltag_cpu {
namespace {
using FamilyPtr = std::unique_ptr<apriltag_family_t, void (*)(apriltag_family_t*)>;

FamilyPtr create_family(Family family) {
    switch (family) {
        case Family::k16h5:
            return {tag16h5_create(), tag16h5_destroy};
        case Family::k25h9:
            return {tag25h9_create(), tag25h9_destroy};
        case Family::k36h10:
            return {tag36h10_create(), tag36h10_destroy};
        case Family::k36h11:
            return {tag36h11_create(), tag36h11_destroy};
    }
    throw std::invalid_argument("Unsupported AprilTag family");
}
// Edge refinement can move a fitted corner past its neighbor. A square tag viewed
// through a camera must still be convex, with enough pixels to resolve its cells.
bool valid_quad(const apriltag_detection_t& detection, double min_edge_px_per_cell) {
    const auto& p = detection.p;
    const double min_edge = min_edge_px_per_cell * detection.family->width_at_border;
    for (int i = 0; i < 4; ++i) {
        const auto& a = p[i];
        const auto& b = p[(i + 1) % 4];
        const auto& c = p[(i + 2) % 4];
        const double dx = b[0] - a[0];
        const double dy = b[1] - a[1];
        // Upstream corner order is counterclockwise in image coordinates.
        if (!(dx * dx + dy * dy >= min_edge * min_edge &&
              dx * (c[1] - b[1]) - dy * (c[0] - b[0]) < 0.0)) {
            return false;
        }
    }
    return true;
}
// Recheck a decoded quad against the complete printed code and its black ring.
// A single sample per payload cell can match textured backgrounds by chance.
bool confirm_cells(const apriltag_detection_t& detection, const image_u8_t& image, int budget,
                   const Config& config) {
    const auto& family = *detection.family;
    constexpr int kMaxSide = 8;  // The four supported classic families have at most 6x6 bits.
    const int side = family.width_at_border;
    std::array<bool, kMaxSide * kMaxSide> white{};
    const std::uint64_t word = family.codes[detection.id];
    for (std::uint32_t i = 0; i < family.nbits; ++i) {
        white[family.bit_y[i] * side + family.bit_x[i]] =
                ((word >> (family.nbits - 1 - i)) & 1U) != 0;
    }
    constexpr int kSamples = 4;
    constexpr int kArea = kSamples * kSamples;
    std::array<std::array<std::uint8_t, kArea>, kMaxSide * kMaxSide> samples{};
    std::array<double, 2> sums{};
    std::array<int, 2> counts{};
    for (int y = 0; y < side; ++y) {
        for (int x = 0; x < side; ++x) {
            const int cell = y * side + x;
            const int color = white[cell] ? 1 : 0;
            for (int sy = 0; sy < kSamples; ++sy) {
                for (int sx = 0; sx < kSamples; ++sx) {
                    const double u = (2.0 * (x + (sx + 0.5) / kSamples)) / side - 1.0;
                    const double v = (2.0 * (y + (sy + 0.5) / kSamples)) / side - 1.0;
                    double px, py;
                    homography_project(detection.H, u, v, &px, &py);
                    if (!(px >= 0 && px < image.width && py >= 0 && py < image.height))
                        return false;
                    auto value =
                            image.buf[static_cast<int>(py) * image.stride + static_cast<int>(px)];
                    samples[cell][sy * kSamples + sx] = value;
                    sums[color] += value;
                    ++counts[color];
                }
            }
        }
    }
    if (counts[0] == 0 || counts[1] == 0)
        return false;
    const double black = sums[0] / counts[0], light = sums[1] / counts[1];
    if (!(black < light))
        return false;
    const double threshold = 0.5 * (black + light);
    int border_errors = 0, payload_errors = 0;
    for (int y = 0; y < side; ++y) {
        for (int x = 0; x < side; ++x) {
            const int cell = y * side + x;
            int agrees = 0;
            for (const auto value : samples[cell])
                agrees += (value > threshold) == white[cell];
            if (agrees < config.min_cell_agreement_ * kArea) {
                if (x == 0 || y == 0 || x == side - 1 || y == side - 1)
                    ++border_errors;
                else
                    ++payload_errors;
            }
        }
    }
    return border_errors <= static_cast<int>(family.nbits * config.max_border_error_rate_) &&
           payload_errors <= budget;
}
}  // namespace

struct Detector::Impl {
    explicit Impl(const Config& config)
        : family_(create_family(config.family_)),
          detector_(apriltag_detector_create(), apriltag_detector_destroy),
          config_(config) {
        if (!this->family_ || !this->detector_) {
            throw std::bad_alloc();
        }
        errno = 0;
        this->budget_ = config.corrected_bits_ == -1
                                ? static_cast<int>(0.6 * ((this->family_->h - 1) / 2))
                                : config.corrected_bits_;
        apriltag_detector_add_family_bits(this->detector_.get(), this->family_.get(),
                                          this->budget_);
        if (errno || !this->family_->impl) {
            throw std::runtime_error("AprilTag decoder table initialization failed");
        }
        this->detector_->nthreads = config.threads_;
        this->detector_->quad_decimate = static_cast<float>(config.quad_decimate_);
    }

    FamilyPtr family_;
    std::unique_ptr<apriltag_detector_t, void (*)(apriltag_detector_t*)> detector_;
    Config config_;
    int budget_ = 0;
};

Detector::Detector(const Config& config) {
    if (config.threads_ < 1 || config.threads_ > 256 || config.quad_decimate_ < 1 ||
        config.quad_decimate_ > 4 || config.corrected_bits_ < -1 || config.corrected_bits_ > 3 ||
        !std::isfinite(config.min_edge_px_per_cell_) || config.min_edge_px_per_cell_ < 0.0 ||
        !(config.min_cell_agreement_ >= 0.0 && config.min_cell_agreement_ <= 1.0) ||
        !(config.max_border_error_rate_ >= 0.0 && config.max_border_error_rate_ <= 1.0)) {
        throw std::invalid_argument("Invalid AprilTag detection parameters");
    }
    this->impl_ = std::make_unique<Impl>(config);
}
Detector::~Detector() = default;

std::vector<Detection> Detector::detect(const std::uint8_t* gray, std::size_t size_bytes, int width,
                                        int height, int stride) {
    if (!gray || width < 1 || height < 1 || width >= 32768 || height >= 32768 || stride < width ||
        stride > std::numeric_limits<int>::max() / height ||
        size_bytes < static_cast<std::size_t>(height - 1) * stride + width) {
        throw std::invalid_argument(
                "Invalid AprilTag grayscale dimensions, stride or buffer length");
    }
    const int decimation = static_cast<int>(this->impl_->detector_->quad_decimate);
    if (width < std::max(8, 4 * decimation) || height < std::max(8, 4 * decimation)) {
        return {};
    }
    // A constant border row cannot contain a tag. Retain two complete 4-pixel tile
    // neighborhoods plus edge refinement support, on the original tile grid.
    const auto uniform = [width](const std::uint8_t* row) {
        return std::memcmp(row, row + 1, static_cast<std::size_t>(width - 1)) == 0;
    };
    const auto row = [gray, stride](int y) { return gray + static_cast<std::size_t>(y) * stride; };
    int first = 0;
    if (uniform(gray)) {
        while (++first < height && std::memcmp(gray, row(first), width) == 0) {
        }
        if (first == height) {
            return {};
        }
    }
    const int tile = 4 * decimation;
    const int top = std::max(0, first / tile - 4) * tile;
    int last = height - 1;
    if (uniform(row(last))) {
        while (--last > first && std::memcmp(row(height - 1), row(last), width) == 0) {
        }
    }
    const int end = std::min(height, (last / tile + 5) * tile);
    // Upstream borrows the pixels. With quad_sigma = 0 it does not filter them in place.
    image_u8_t image{width, end - top, stride, const_cast<std::uint8_t*>(row(top))};
    errno = 0;
    std::unique_ptr<zarray_t, decltype(&apriltag_detections_destroy)> detections(
            apriltag_detector_detect(this->impl_->detector_.get(), &image),
            apriltag_detections_destroy);
    if (!detections || !this->impl_->detector_->wp) {
        throw std::system_error(errno ? errno : ENOMEM, std::generic_category(),
                                "AprilTag detection");
    }
    std::vector<Detection> results;
    results.reserve(zarray_size(detections.get()));
    for (int i = 0; i < zarray_size(detections.get()); ++i) {
        apriltag_detection_t* detected = nullptr;
        zarray_get(detections.get(), i, &detected);
        if (!valid_quad(*detected, this->impl_->config_.min_edge_px_per_cell_) ||
            !confirm_cells(*detected, image, this->impl_->budget_, this->impl_->config_)) {
            continue;
        }
        Detection result{};
        result.id_ = detected->id;
        for (int k = 0; k < 4; ++k) {
            const int source = (5 - k) % 4;
            result.corners_[k] = {detected->p[source][0], detected->p[source][1] + top};
        }
        results.push_back(result);
    }
    return results;
}
}  // namespace aruco3cuda::apriltag_cpu
