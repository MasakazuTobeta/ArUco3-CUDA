// SPDX-License-Identifier: Apache-2.0
#include "aruco3cuda/apriltag/small_cpu/detector.hpp"

#include "aruco3cuda/dictionary.hpp"
#include "automatic_search.hpp"
#include "border_search.hpp"
#include "bounded_quad_approximation.hpp"
#include "coarse_mask.hpp"
#include "component_candidates.hpp"
#include "cpu_contour_scan.hpp"
#include "cpu_grid_refine.hpp"
#include "geometry.hpp"
#include "gray_view.hpp"
#include "grid_alignment.hpp"
#include "local_payload_fit.hpp"
#include "tag_codebook.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace aruco3cuda::apriltag_small_cpu {
namespace {

using detail::bilinear;
using detail::cross;
using detail::length;
using detail::Point;
using detail::Quad;

using detail::Decode;
using detail::decode;
using detail::kBorderSide;
using detail::kPayloadSide;
using detail::kRingSamples;
constexpr int kImageSideLimit = 32768;

void validate_coarse_config(const CoarseConfig& config) {
    if (config.sample_stride_px_ < 2 || config.sample_stride_px_ > 4 ||
        config.dim_sample_floor_ < 0 || config.dim_sample_floor_ > 255 ||
        config.dim_sample_percentage_ < 0 || config.dim_sample_percentage_ > 100 ||
        config.tile_side_samples_ < 3 || config.tile_side_samples_ > 32 ||
        config.tile_minimum_contrast_ < 1 || config.tile_minimum_contrast_ > 255 ||
        config.minimum_quad_side_samples_ < 3 ||
        config.minimum_quad_side_samples_ >= kImageSideLimit || config.minimum_ring_matches_ < 1 ||
        config.minimum_ring_matches_ > kRingSamples ||
        !(config.minimum_decode_contrast_ > 0 && config.minimum_decode_contrast_ <= 255) ||
        config.minimum_retry_side_px_ < 0 || config.minimum_retry_side_px_ >= kImageSideLimit ||
        config.grid_radius_px_ < 2 || config.grid_radius_px_ > 32) {
        throw std::invalid_argument("Invalid coarse AprilTag candidate parameters");
    }
}

void validate_component_config(const ComponentConfig& config) {
    if (config.local_payload_fit_) {
        const auto& fit = *config.local_payload_fit_;
        if (!(fit.minimum_side_px_ >= 8 && std::isfinite(fit.minimum_side_px_)) ||
            !(fit.minimum_displacement_px_ >= 0 && std::isfinite(fit.minimum_displacement_px_)) ||
            !(fit.maximum_displacement_cells_ >= 0 &&
              std::isfinite(fit.maximum_displacement_cells_)))
            throw std::invalid_argument("Invalid local payload fit settings");
    }
    const int side = config.local_midrange_side_samples_;
    if ((side != 0 && (side < 3 || side > 31 || side % 2 == 0 || config.exposure_ ||
                       config.alternate_contour_threshold_)) ||
        config.local_midrange_offset_ < 0 || config.local_midrange_offset_ > 127) {
        throw std::invalid_argument("Invalid AprilTag component local threshold configuration");
    }
    if (config.exposure_) {
        const auto& exposure = *config.exposure_;
        if (exposure.minimum_low_percentile_ < 0 || exposure.minimum_low_percentile_ > 255 ||
            exposure.highlight_reference_ < 0 || exposure.highlight_reference_ > 255 ||
            exposure.adjustment_percent_ < 0 || exposure.adjustment_percent_ > 100 ||
            exposure.maximum_adjustment_ < 0 || exposure.maximum_adjustment_ > 255)
            throw std::invalid_argument("Invalid AprilTag component exposure adjustment");
    }
    if (config.sample_stride_px_ < 2 || config.sample_stride_px_ > 4 || config.threshold_ < 0 ||
        config.threshold_ > 255 || config.minimum_quad_side_samples_ < 3 ||
        config.maximum_quad_side_samples_ < config.minimum_quad_side_samples_ ||
        config.maximum_quad_side_samples_ >= kImageSideLimit ||
        !(config.maximum_aspect_ratio_ >= 1 && config.maximum_aspect_ratio_ < kImageSideLimit) ||
        !(config.minimum_fill_ratio_ >= 0 &&
          config.maximum_fill_ratio_ >= config.minimum_fill_ratio_ &&
          config.maximum_fill_ratio_ <= 1) ||
        !(config.hull_tolerance_ratio_ > 0 && config.hull_tolerance_ratio_ <= 0.5) ||
        !(config.minimum_quad_hull_ratio_ >= 0 && config.minimum_quad_hull_ratio_ < 1) ||
        config.minimum_ring_matches_ < 1 || config.minimum_ring_matches_ > kRingSamples ||
        config.maximum_expansion_side_px_ < 0 ||
        config.maximum_expansion_side_px_ >= kImageSideLimit ||
        (config.grid_radius_px_ != 0 &&
         (config.grid_radius_px_ < 2 || config.grid_radius_px_ > 32)))
        throw std::invalid_argument("Invalid AprilTag component candidate parameters");
}

void validate_config(const Config& config) {
    if (config.dark_threshold_ < 0 || config.bright_threshold_ < config.dark_threshold_ ||
        config.bright_threshold_ > 255 || config.bright_sample_floor_ < config.bright_threshold_ ||
        config.bright_sample_floor_ > 255 || config.bright_sample_percentage_ < 0 ||
        config.bright_sample_percentage_ > 100 || config.minimum_ring_matches_ < 1 ||
        config.minimum_ring_matches_ > kRingSamples ||
        !(config.minimum_contrast_ > 0 && config.minimum_contrast_ <= 255) ||
        !(config.polygon_tolerance_ratio_ > 0 && config.polygon_tolerance_ratio_ <= 0.5F) ||
        config.maximum_corrected_bits_ < 0 || config.maximum_corrected_bits_ > 3) {
        throw std::invalid_argument("Invalid small AprilTag detector parameters");
    }
    if (config.border_) {
        detail::validate_border_config(*config.border_);
    }
    if (config.recovery_) {
        const auto& recovery = *config.recovery_;
        if (recovery.maximum_tracks_ == 0 || recovery.maximum_tracks_ > 256 ||
            recovery.maximum_translation_px_ < 0 || recovery.maximum_translation_px_ > 32 ||
            recovery.minimum_ring_matches_ < 16 || recovery.minimum_ring_matches_ > kRingSamples ||
            !(recovery.minimum_contrast_ > 0 && recovery.minimum_contrast_ <= 255)) {
            throw std::invalid_argument("Invalid AprilTag video recovery parameters");
        }
    }
    const int search_count = config.coarse_.has_value() + config.components_.has_value() +
                             config.automatic_.has_value();
    if (search_count > 1)
        throw std::invalid_argument("AprilTag candidate searches are mutually exclusive");
    if (config.coarse_)
        validate_coarse_config(*config.coarse_);
    if (config.components_) {
        const auto& components = *config.components_;
        validate_component_config(components);
        const auto threshold = components.alternate_contour_threshold_;
        if (threshold && (*threshold < 0 || *threshold > 255 || !config.recovery_ ||
                          components.sample_stride_px_ != 2 || components.sampled_contours_ ||
                          components.exposure_)) {
            throw std::invalid_argument(
                    "Alternate contours require explicit fixed-threshold stride-2 components "
                    "and video recovery, with an alternate threshold from 0 to 255");
        }
    }
    if (config.automatic_) {
        if (config.automatic_->components_.alternate_contour_threshold_)
            throw std::invalid_argument("Alternate contours require explicit components");
        detail::validate_automatic_config(*config.automatic_);
        validate_coarse_config(config.automatic_->coarse_);
        validate_component_config(config.automatic_->components_);
    }
}

int select_threshold(const cv::Mat& gray, const Config& config) {
    int samples = 0;
    int bright = 0;
    for (int y = 5; y < gray.rows; y += 13) {
        const auto* row = gray.ptr<std::uint8_t>(y);
        for (int x = 5; x < gray.cols; x += 17) {
            ++samples;
            bright += row[x] >= config.bright_sample_floor_;
        }
    }
    return bright * 100 > samples * config.bright_sample_percentage_ ? config.bright_threshold_
                                                                     : config.dark_threshold_;
}

cv::Rect flat_surround_crop(const cv::Mat& gray) {
    const cv::Rect full(0, 0, gray.cols, gray.rows);
    const auto* base = gray.ptr<std::uint8_t>(0);
    if (std::memcmp(base, base + 1, gray.cols - 1) != 0 ||
        std::memcmp(base, gray.ptr<std::uint8_t>(gray.rows - 1), gray.cols) != 0) {
        return full;
    }
    int top = 1;
    int bottom = gray.rows - 1;
    while (top < bottom && std::memcmp(base, gray.ptr<std::uint8_t>(top), gray.cols) == 0) {
        ++top;
    }
    if (top >= bottom) {
        return {};
    }
    while (bottom > top + 1 &&
           std::memcmp(base, gray.ptr<std::uint8_t>(bottom - 1), gray.cols) == 0) {
        --bottom;
    }
    if (bottom - top > gray.rows * 3 / 4) {
        return full;
    }

    // Find the full footprint that differs from the shared outer-row value.
    // A short uniform margin keeps the candidate contour inside the crop.
    int left = gray.cols;
    int right = 0;
    for (int y = top; y < bottom; ++y) {
        const auto* row = gray.ptr<std::uint8_t>(y);
        if (std::memcmp(row, base, left) != 0) {
            int x = 0;
            for (; x + 16 <= left && std::memcmp(row + x, base + x, 16) == 0; x += 16) {
            }
            while (x < left && row[x] == base[0]) {
                ++x;
            }
            left = x;
        }
        if (std::memcmp(row + right, base + right, gray.cols - right) != 0) {
            int x = gray.cols;
            for (; x - 16 >= right && std::memcmp(row + x - 16, base + x - 16, 16) == 0; x -= 16) {
            }
            while (x > right && row[x - 1] == base[0]) {
                --x;
            }
            right = x;
        }
    }
    constexpr int kMarginPx = 3;
    left = std::max(0, left - kMarginPx);
    top = std::max(0, top - kMarginPx);
    right = std::min(gray.cols, right + kMarginPx);
    bottom = std::min(gray.rows, bottom + kMarginPx);
    return {left, top, right - left, bottom - top};
}

using detail::project;
using detail::Projection;

// A rectangular one-sample erosion moves a straight boundary inward by the
// support of the 3x3 square along its normal. Account also for the distance
// between the continuous boundary and the center of the last dark sample.
bool undo_erosion(const Quad& corners, int stride, Quad& expanded) {
    std::array<Point, 4> normals;
    std::array<float, 4> offsets;
    for (int i = 0; i < 4; ++i) {
        const Point edge = corners[(i + 1) % 4] - corners[i];
        const Point normal = Point(edge.y, -edge.x) / length(edge);
        normals[i] = normal;
        const float x = std::abs(normal.x), y = std::abs(normal.y);
        offsets[i] = stride * (x + y + .5F * std::max(x, y));
    }
    for (int i = 0; i < 4; ++i) {
        const int previous = (i + 3) % 4;
        const Point a = normals[previous], b = normals[i];
        const float determinant = cross(a, b);
        if (std::abs(determinant) < .12F)
            return false;
        expanded[i] = corners[i] + Point(b.y * offsets[previous] - a.y * offsets[i],
                                         a.x * offsets[i] - b.x * offsets[previous]) /
                                           determinant;
    }
    return true;
}

// The code can still read correctly with a displaced corner. Check the native
// signed transition on every side before accepting the grid fit from a coarse seed.
bool outer_aligned(const cv::Mat& gray, const Quad& corners, float minimum_contrast) {
    for (int i = 0; i < 4; ++i) {
        const Point edge = corners[(i + 1) % 4] - corners[i];
        const float side = length(edge);
        const Point normal = Point(edge.y, -edge.x) / side;
        const float reach = std::clamp(side / 24.F, 2.F, 5.F);
        float signed_sum = 0, absolute_sum = 0;
        int supported = 0;
        for (const float t : {.15F, .325F, .5F, .675F, .85F}) {
            const Point p = corners[i] + t * edge;
            const float ink = bilinear(gray, p - reach * normal);
            const float paper = bilinear(gray, p + reach * normal);
            const float center_intensity = bilinear(gray, p);
            const float slope = .5F * (bilinear(gray, p + normal) - bilinear(gray, p - normal));
            if (ink < 0 || paper < 0 || center_intensity < 0 || paper - ink < minimum_contrast ||
                slope < .12F * (paper - ink))
                continue;
            const float offset = (.5F * (paper + ink) - center_intensity) / slope;
            if (std::abs(offset) > 2.F)
                continue;
            signed_sum += offset;
            absolute_sum += std::abs(offset);
            ++supported;
        }
        if (supported < 3 || std::abs(signed_sum) > .4F * supported ||
            absolute_sum > .64F * supported)
            return false;
    }
    return true;
}

Point center(const Quad& corners) {
    return 0.25F * (corners[0] + corners[1] + corners[2] + corners[3]);
}

}  // namespace

class Detector::Impl {
public:
    explicit Impl(const Config& config)
        : config_(config),
          minimum_ring_matches_(config.coarse_       ? config.coarse_->minimum_ring_matches_
                                : config.components_ ? config.components_->minimum_ring_matches_
                                                     : config.minimum_ring_matches_),
          minimum_contrast_(config.coarse_ ? config.coarse_->minimum_decode_contrast_
                                           : config.minimum_contrast_) {}

    std::vector<Detection> detect(const cv::Mat& gray, bool native_samples) {
        if (this->config_.components_)
            return this->detect_components(gray, *this->config_.components_, native_samples);
        if (this->config_.coarse_) {
            const auto& coarse = *this->config_.coarse_;
            cv::Point sample_origin;
            auto* packed = cv::getNumThreads() == 1 && coarse.minimum_quad_side_samples_ >= 4
                                   ? &this->contour_bits_
                                   : nullptr;
            const auto& mask = this->coarse_mask_.build(gray, this->config_, coarse, &sample_origin,
                                                        true, packed);
            if (mask.empty())
                return {};
            if (packed && !packed->empty()) {
                this->quads_.clear();
                const auto retain = [&](const std::vector<cv::Point>& path) {
                    this->accept_shape(path, coarse.minimum_quad_side_samples_);
                };
                this->contour_scanner_.for_each_packed<cv::CHAIN_APPROX_SIMPLE, true>(
                        mask.size(), this->contour_bits_, retain);
                std::reverse(this->quads_.begin(), this->quads_.end());
                return this->decode_quads(gray, sample_origin * coarse.sample_stride_px_,
                                          coarse.sample_stride_px_, false, true);
            }
            return this->extract(gray, mask, sample_origin * coarse.sample_stride_px_,
                                 coarse.sample_stride_px_, coarse.minimum_quad_side_samples_,
                                 false);
        }
        return this->detect_native(gray, std::nullopt);
    }

    std::vector<Detection> detect_components(const cv::Mat& gray, const ComponentConfig& config,
                                             bool native_samples) {
        if (config.sampled_contours_ && config.sample_stride_px_ == 2 &&
            config.local_midrange_side_samples_ == 0 && gray.cols >= 256 &&
            cv::getNumThreads() == 1 &&
            detail::ComponentCandidates::native_stride2_available(gray.cols) &&
            detail::ComponentCandidates::pack_trace_samples(
                    gray, detail::component_threshold(gray, config), this->contour_bits_)) {
            this->quads_.clear();
            const auto retain = [&](const auto& path) {
                this->accept_shape(path, config.minimum_quad_side_samples_);
            };
            const cv::Size size((gray.cols + 1) / 2, (gray.rows + 1) / 2);
            if (config.minimum_quad_side_samples_ > 3)
                this->contour_scanner_.for_each_packed<cv::CHAIN_APPROX_SIMPLE, true>(
                        size, this->contour_bits_, retain);
            else
                this->contour_scanner_.for_each_packed<cv::CHAIN_APPROX_SIMPLE, false>(
                        size, this->contour_bits_, retain);
            std::reverse(this->quads_.begin(), this->quads_.end());
            return this->decode_quads(gray, {}, 2, false, native_samples);
        }
        if (config.sampled_contours_)
            return this->extract(gray, this->coarse_mask_.build_components(gray, config), {},
                                 config.sample_stride_px_, config.minimum_quad_side_samples_, false,
                                 native_samples, config.local_midrange_side_samples_ == 0);
        if (config.sample_stride_px_ == 2 && config.local_midrange_side_samples_ == 0 &&
            detail::ComponentCandidates::native_stride2_available(gray.cols))
            this->components_.extract_native_stride2(
                    gray, detail::component_threshold(gray, config), config, this->contours_);
        else
            this->components_.extract(this->coarse_mask_.build_components(gray, config), config,
                                      this->contours_);
        return this->decode_candidates(gray, {}, config.sample_stride_px_,
                                       config.minimum_quad_side_samples_, false, this->contours_,
                                       native_samples);
    }

    std::vector<Detection> decode_sampled_contours(
            const cv::Mat& gray, const std::vector<std::vector<cv::Point>>& contours) {
        const auto& config = *this->config_.components_;
        return this->decode_candidates(gray, {}, config.sample_stride_px_,
                                       config.minimum_quad_side_samples_, false, contours);
    }

    std::vector<Detection> detect_native(const cv::Mat& gray, std::optional<int> threshold) {
        const cv::Rect full(0, 0, gray.cols, gray.rows);
        const cv::Rect roi = this->config_.trim_uniform_surround_ ? flat_surround_crop(gray) : full;
        if (roi.empty()) {
            return {};
        }
        const bool trimmed = roi.size() != full.size();
        const int luminance = threshold ? *threshold : select_threshold(gray, this->config_);
        cv::threshold(gray(roi), this->binary_, luminance, 255, cv::THRESH_BINARY_INV);
        return this->extract(gray, this->binary_, roi.tl(), 1, 8, trimmed);
    }

    std::optional<Detection> recover_tag(const cv::Mat& gray, Quad corners, int previous_id,
                                         const RecoveryConfig& config) const {
        for (bool uniform_surround : {true, false}) {
            if (!detail::refine_quad(gray, corners, uniform_surround))
                continue;
            const Decode tag =
                    decode(gray, corners, config.minimum_ring_matches_, config.minimum_contrast_,
                           std::min(2, this->config_.maximum_corrected_bits_));
            if (tag.id < 0 || (tag.distance != 0 && (tag.id != previous_id || tag.rotation != 0)))
                continue;
            Quad ordered;
            for (int k = 0; k < 4; ++k)
                ordered[k] = corners[(k + 4 - tag.rotation) % 4];
            return Detection{tag.id, ordered, tag.distance};
        }
        return std::nullopt;
    }

private:
    struct CandidateQuad {
        cv::Rect bounds_;
        std::array<cv::Point, 4> corners_;
    };

    void accept_shape(const std::vector<cv::Point>& contour, int minimum_side) {
        if (contour.size() < 4)
            return;
        const auto bounds = cv::boundingRect(contour);
        if (std::min(bounds.width, bounds.height) < minimum_side)
            return;
        const double epsilon =
                this->config_.polygon_tolerance_ratio_ * cv::arcLength(contour, true);
        if (contour.size() <= detail::kClosedPathMaximumVertices &&
            std::max(bounds.width, bounds.height) <= detail::kClosedPathMaximumExtent) {
            if (!detail::approximate_closed_quad(contour, epsilon, this->polygon_))
                return;
        } else {
            cv::approxPolyDP(contour, this->polygon_, epsilon, true);
        }
        if (this->polygon_.size() != 4 || !cv::isContourConvex(this->polygon_))
            return;
        if (cv::contourArea(this->polygon_, true) < 0)
            std::reverse(this->polygon_.begin(), this->polygon_.end());
        CandidateQuad quad;
        quad.bounds_ = bounds;
        std::copy(this->polygon_.begin(), this->polygon_.end(), quad.corners_.begin());
        this->quads_.push_back(quad);
    }

    std::vector<Detection> extract(const cv::Mat& gray, const cv::Mat& mask, cv::Point origin,
                                   int stride, int minimum_side, bool trimmed,
                                   bool native_samples = true, bool fixed_sampled_mask = false) {
        // The serial reader follows the one-worker RETR_LIST ordering.
        const bool use_fixed_mask_scan = fixed_sampled_mask && cv::getNumThreads() == 1;
        if (use_fixed_mask_scan || this->config_.coarse_ ||
            (this->config_.components_ &&
             this->config_.components_->local_midrange_side_samples_ == 3)) {
            this->quads_.clear();
            const auto retain = [&](const std::vector<cv::Point>& path) {
                this->accept_shape(path, minimum_side);
            };
            if (use_fixed_mask_scan || this->config_.coarse_)
                this->contour_scanner_.for_each<cv::CHAIN_APPROX_SIMPLE, true>(mask, true, retain);
            else
                this->contour_scanner_.for_each<cv::CHAIN_APPROX_SIMPLE>(mask, true, retain);
            std::reverse(this->quads_.begin(), this->quads_.end());
            return this->decode_quads(gray, origin, stride, trimmed, native_samples);
        }
        cv::findContours(mask, this->contours_, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        return this->decode_candidates(gray, origin, stride, minimum_side, trimmed, this->contours_,
                                       native_samples);
    }

    std::vector<Detection> decode_candidates(const cv::Mat& gray, cv::Point origin, int stride,
                                             int minimum_side, bool trimmed,
                                             const std::vector<std::vector<cv::Point>>& contours,
                                             bool native_samples = true) {
        this->quads_.clear();
        for (const auto& contour : contours)
            this->accept_shape(contour, minimum_side);
        return this->decode_quads(gray, origin, stride, trimmed, native_samples);
    }

    std::vector<Detection> decode_quads(const cv::Mat& gray, cv::Point origin, int stride,
                                        bool trimmed, bool native_samples) {
        const auto* components = this->config_.components_ ? &*this->config_.components_ : nullptr;
        const auto* coarse = this->config_.coarse_ ? &*this->config_.coarse_ : nullptr;
        const int sample_center = (stride - 1) / 2;
        std::vector<Detection> results;
        for (const auto& candidate : this->quads_) {
            const cv::Rect bounds = candidate.bounds_;
            const int shorter_side = std::min(bounds.width, bounds.height);
            Quad corners;
            for (int k = 0; k < 4; ++k) {
                const cv::Point point = candidate.corners_[k];
                const int x = origin.x + stride * point.x + sample_center;
                const int y = origin.y + stride * point.y + sample_center;
                corners[k] = {static_cast<float>(std::min(x, gray.cols - 1)),
                              static_cast<float>(std::min(y, gray.rows - 1))};
            }
            const float shortest_edge =
                    std::min({length(corners[1] - corners[0]), length(corners[2] - corners[1]),
                              length(corners[3] - corners[2]), length(corners[0] - corners[3])});
            if (shortest_edge < 6) {
                continue;
            }
            Decode tag = this->read_cells(gray, corners);
            if (components && tag.id < 0 &&
                std::max(bounds.width, bounds.height) * stride <=
                        components->maximum_expansion_side_px_)
                this->retry_expansion(gray, stride, tag, corners);
            const bool near_decode = tag.distance <= this->config_.maximum_corrected_bits_ + 2;
            const bool retry = coarse && coarse->minimum_retry_side_px_ > 0 &&
                               shorter_side * stride >= coarse->minimum_retry_side_px_;
            // The midpoint fit can resolve a coarse contour directly when each
            // cell spans several samples. Small prints retain the full outer-edge fit.
            const bool large_coarse = coarse && coarse->grid_radius_px_ > 0 &&
                                      shortest_edge >= 4 * kBorderSide * stride;
            bool grid_measured = false;
            if (large_coarse) {
                Quad expanded;
                if (undo_erosion(corners, stride, expanded)) {
                    Decode candidate = tag.id >= 0 ? tag : this->read_cells(gray, expanded);
                    if (candidate.id >= 0 &&
                        this->polish(gray, candidate, coarse->grid_radius_px_, expanded) &&
                        outer_aligned(gray, expanded, this->minimum_contrast_)) {
                        tag = candidate;
                        corners = expanded;
                        grid_measured = true;
                    }
                }
            }
            if (!grid_measured && ((tag.supported && (!trimmed || near_decode)) || retry)) {
                Quad refined = corners;
                if (!components || !components->sharp_edge_refinement_ ||
                    !detail::try_fit_sharp_border(gray, refined))
                    detail::refine_quad(gray, refined, trimmed,
                                        coarse && coarse->grid_radius_px_ > 0 ? stride : 0);
                Decode checked = this->read_cells(gray, refined);
                // The configured native reader must still corroborate the ID, rotation,
                // ring contrast and Hamming limit. Only its worse bit count is reconsidered.
                if (native_samples && components && components->local_payload_fit_ && tag.id >= 0 &&
                    tag.distance > 0 && checked.id == tag.id && checked.rotation == tag.rotation &&
                    checked.distance > tag.distance) {
                    const auto& fit = *components->local_payload_fit_;
                    if (fit.maximum_displacement_cells_ > 0 &&
                        shortest_edge >= fit.minimum_side_px_) {
                        float movement = 0;
                        for (int i = 0; i < 4; ++i)
                            movement = std::max(movement, length(refined[i] - corners[i]));
                        const auto word = generated::kDictApriltag36h11Table
                                                  .codes_[4 * tag.id + tag.rotation];
                        if (movement > fit.minimum_displacement_px_ &&
                            movement <=
                                    fit.maximum_displacement_cells_ * shortest_edge / kBorderSide &&
                            detail::matches_local_payload(gray, refined, word,
                                                          this->minimum_contrast_))
                            checked.distance = 0;
                    }
                }
                if (checked.id >= 0 && checked.distance <= tag.distance) {
                    tag = checked;
                    corners = refined;
                }
            }
            if (tag.id < 0) {
                continue;
            }
            const int grid_radius = coarse       ? coarse->grid_radius_px_
                                    : components ? components->grid_radius_px_
                                                 : 0;
            if (!grid_measured && grid_radius > 0)
                this->polish(gray, tag, grid_radius, corners);
            Quad ordered;
            for (int k = 0; k < 4; ++k) {
                ordered[k] = corners[(k + 4 - tag.rotation) % 4];
            }
            const auto near = [&](const Detection& other) {
                const float radius =
                        0.25F * std::min(length(ordered[1] - ordered[0]),
                                         length(other.corners_px_[1] - other.corners_px_[0]));
                return length(center(ordered) - center(other.corners_px_)) <= radius;
            };
            const auto existing = std::find_if(results.begin(), results.end(), near);
            const Detection detection{tag.id, ordered, tag.distance};
            if (existing == results.end()) {
                results.push_back(detection);
            } else if (tag.distance < existing->corrected_bits_) {
                *existing = detection;
            }
        }
        return results;
    }

    void retry_expansion(const cv::Mat& gray, int stride, Decode& tag, Quad& corners) const {
        const Point centroid = center(corners);
        const float side = std::max(1.0F, length(corners[1] - corners[0]));
        for (const float offset : {1.0F, 2.0F, -1.0F}) {
            Quad attempt;
            for (std::size_t i = 0; i < attempt.size(); ++i)
                attempt[i] = centroid + (1 + offset * stride / side) * (corners[i] - centroid);
            const Decode checked = this->read_cells(gray, attempt);
            if (checked.id >= 0) {
                tag = checked;
                corners = attempt;
                return;
            }
        }
    }

    Decode read_cells(const cv::Mat& gray, const Quad& corners) const {
        return decode(gray, corners, this->minimum_ring_matches_, this->minimum_contrast_,
                      this->config_.maximum_corrected_bits_);
    }

    bool polish(const cv::Mat& gray, Decode& tag, int radius, Quad& corners) {
        const auto code = generated::kDictApriltag36h11Table.codes_[tag.id * 4 + tag.rotation];
        cv::Matx<std::uint8_t, kBorderSide, kBorderSide> cells{};
        for (int y = 0; y < kPayloadSide; ++y)
            for (int x = 0; x < kPayloadSide; ++x)
                cells(y + 1, x + 1) = (code >> (kPayloadSide * y + x)) & 1U;
        const cv::Mat cell_view(cells, false);
        this->grid_corners_.assign(corners.begin(), corners.end());
        if (!aruco3cuda::cpu::detail::fit_decoded_payload_grid(gray, cell_view, radius,
                                                               this->grid_corners_))
            return false;

        Quad refined;
        std::copy(this->grid_corners_.begin(), this->grid_corners_.end(), refined.begin());
        if (auto errors =
                    detail::verify_payload(gray, refined, tag.id, this->minimum_ring_matches_,
                                           this->minimum_contrast_, tag.distance, tag.rotation)) {
            tag.distance = *errors;
            corners = refined;
            return true;
        }
        return false;
    }

    const Config config_;
    const int minimum_ring_matches_;
    const float minimum_contrast_;
    detail::CoarseMask coarse_mask_;
    detail::ComponentCandidates components_;
    std::vector<std::uint64_t> contour_bits_;
    std::vector<CandidateQuad> quads_;
    cpu::detail::ContourScanner contour_scanner_;
    std::vector<Point> grid_corners_;
    cv::Mat binary_;
    std::vector<std::vector<cv::Point>> contours_;
    std::vector<cv::Point> polygon_;
};

class Detector::Recovery {
public:
    explicit Recovery(const RecoveryConfig& config) : config_(config) {
        this->tracks_.reserve(config.maximum_tracks_);
    }

    void reset() noexcept {
        this->tracks_.clear();
        this->frame_size_ = {};
    }

    void append(const cv::Mat& gray, const Impl& decoder, std::vector<Detection>& tags) {
        if (this->frame_size_ == gray.size()) {
            for (const Track& track : this->tracks_) {
                if (occupied(track.tag_.corners_px_, tags))
                    continue;
                Quad proposal = track.tag_.corners_px_;
                const auto translation = this->find_translation(gray, track);
                for (auto& point : proposal)
                    point += Point(translation);
                const auto verified =
                        decoder.recover_tag(gray, proposal, track.tag_.id_, this->config_);
                if (verified && !occupied(verified->corners_px_, tags))
                    tags.push_back(*verified);
            }
        }
        this->save(gray, tags);
    }

private:
    struct Sample {
        cv::Point position_;
        std::uint8_t luma_;
    };

    struct Track {
        Detection tag_;
        std::array<Sample, kBorderSide * kBorderSide / 2> samples_{};
        std::size_t sample_count_ = 0;
    };

    static bool occupied(const Quad& quad, const std::vector<Detection>& tags) {
        const Point centroid = center(quad);
        const float side_px = length(quad[1] - quad[0]);
        return std::any_of(tags.begin(), tags.end(), [&](const Detection& tag) {
            const auto& other = tag.corners_px_;
            const float radius = 0.42F * std::min(side_px, length(other[1] - other[0]));
            const Point delta = centroid - center(other);
            return delta.dot(delta) < radius * radius;
        });
    }

    cv::Point find_translation(const cv::Mat& gray, const Track& track) const {
        constexpr int kClippedDifference = 80;
        constexpr int kGoodMeanDifference = 6;
        constexpr int kGridStepPx = 3;
        const auto& corners = track.tag_.corners_px_;
        const int radius =
                std::min(this->config_.maximum_translation_px_,
                         std::max(6, static_cast<int>(0.30F * length(corners[1] - corners[0]))));
        const auto cost = [&](cv::Point offset) {
            int sum = 0;
            for (std::size_t i = 0; i < track.sample_count_; ++i) {
                const Sample& sample = track.samples_[i];
                const cv::Point at = sample.position_ + offset;
                if (at.x >= 0 && at.x < gray.cols && at.y >= 0 && at.y < gray.rows) {
                    const int luma = gray.ptr<std::uint8_t>(at.y)[at.x];
                    sum += std::min(kClippedDifference, std::abs(luma - sample.luma_));
                } else {
                    sum += kClippedDifference;
                }
            }
            return sum;
        };
        cv::Point best;
        int best_cost = cost(best);
        if (radius == 0 || best_cost <= static_cast<int>(track.sample_count_) * kGoodMeanDifference)
            return best;
        const auto probe = [&](cv::Point offset) {
            const int value = cost(offset);
            if (value < best_cost) {
                best_cost = value;
                best = offset;
            }
        };
        for (int y = -radius; y <= radius; y += kGridStepPx)
            for (int x = -radius; x <= radius; x += kGridStepPx)
                probe({x, y});
        const cv::Point coarse = best;
        for (int y = std::max(-radius, coarse.y - 2); y <= std::min(radius, coarse.y + 2); ++y)
            for (int x = std::max(-radius, coarse.x - 2); x <= std::min(radius, coarse.x + 2); ++x)
                probe({x, y});
        return best;
    }

    void save(const cv::Mat& gray, const std::vector<Detection>& tags) {
        constexpr std::size_t kMinimumSamples = 20;
        this->tracks_.clear();
        for (const Detection& tag : tags) {
            if (this->tracks_.size() == this->config_.maximum_tracks_)
                break;
            Projection projection;
            if (!project(tag.corners_px_, projection))
                continue;
            Track track{tag};
            for (int y = 0; y < kBorderSide; ++y) {
                for (int x = y % 2; x < kBorderSide; x += 2) {
                    const float u = (x + 0.5F) / kBorderSide;
                    const float v = (y + 0.5F) / kBorderSide;
                    const float z = 1 + projection.z_x * u + projection.z_y * v;
                    if (z < 1e-5F)
                        continue;
                    const Point point =
                            (projection.initial + u * projection.x + v * projection.y) / z;
                    if (!(point.x >= 0 && point.x < gray.cols - 1 && point.y >= 0 &&
                          point.y < gray.rows - 1))
                        continue;
                    const cv::Point at(cvRound(point.x), cvRound(point.y));
                    track.samples_[track.sample_count_++] = {at,
                                                             gray.ptr<std::uint8_t>(at.y)[at.x]};
                }
            }
            if (track.sample_count_ >= kMinimumSamples)
                this->tracks_.push_back(track);
        }
        this->frame_size_ = gray.size();
    }

    const RecoveryConfig config_;
    cv::Size frame_size_;
    std::vector<Track> tracks_;
};

Detector::Detector(const Config& config) {
    validate_config(config);
    if (config.border_)
        this->border_ = std::make_unique<detail::BorderSearch>(*config.border_,
                                                               config.maximum_corrected_bits_);
    if (config.automatic_)
        this->automatic_ = std::make_unique<detail::AutomaticSearch>(config);
    else
        this->impl_ = std::make_unique<Impl>(config);
    if (config.recovery_)
        this->recovery_ = std::make_unique<Recovery>(*config.recovery_);
    if (config.components_ && config.components_->alternate_contour_threshold_) {
        this->alternate_components_ = *config.components_;
        this->alternate_components_->threshold_ = *config.components_->alternate_contour_threshold_;
        this->alternate_components_->sampled_contours_ = true;
        this->alternate_components_->alternate_contour_threshold_.reset();
    }
}

Detector::~Detector() = default;

std::optional<Detection> Detector::recover_translated(const cv::Mat& gray, const Detection& last,
                                                      cv::Point translation) const {
    auto proposal = last.corners_px_;
    for (auto& corner : proposal)
        corner += cv::Point2f(translation);
    return impl_->recover_tag(gray, proposal, last.id_, RecoveryConfig{});
}

std::vector<Detection> Detector::decode_sampled_contours(
        const cv::Mat& gray, const std::vector<std::vector<cv::Point>>& contours) {
    return this->impl_->decode_sampled_contours(gray, contours);
}

std::vector<Detection> Detector::detect_native_with_threshold(const cv::Mat& gray, int threshold) {
    return this->impl_->detect_native(gray, threshold);
}

std::vector<Detection> Detector::detect(const cv::Mat& gray) {
    return this->detect(gray, true);
}

std::vector<Detection> Detector::detect(const cv::Mat& gray, bool native_samples) {
    detail::validate_image(gray);
    const bool alternate = this->alternate_components_ && this->alternate_frame_ &&
                           this->alternate_size_ == gray.size();
    auto result = alternate ? this->impl_->detect_components(gray, *this->alternate_components_,
                                                             native_samples)
                  : this->automatic_ ? this->automatic_->detect(gray)
                                     : this->impl_->detect(gray, native_samples);
    if (this->border_)
        this->border_->append(gray, result);
    if (this->recovery_) {
        const Impl& decoder = this->automatic_ ? *this->automatic_->native_.impl_ : *this->impl_;
        this->recovery_->append(gray, decoder, result);
    }
    if (this->alternate_components_) {
        this->alternate_size_ = gray.size();
        this->alternate_frame_ = !alternate;
    }
    return result;
}

void Detector::reset() noexcept {
    if (this->recovery_)
        this->recovery_->reset();
    this->alternate_size_ = {};
    this->alternate_frame_ = false;
}

}  // namespace aruco3cuda::apriltag_small_cpu
