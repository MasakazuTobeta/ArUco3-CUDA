// SPDX-License-Identifier: Apache-2.0
#include "aruco3cuda/apriltag/small_cpu/video_detector.hpp"

#include "automatic_search.hpp"
#include "border_search.hpp"
#include "cubic_patch.hpp"
#include "curved_edges.hpp"
#include "dense_board_search.hpp"
#include "entering_strip_search.hpp"
#include "faint_guide.hpp"
#include "geometry.hpp"
#include "gray_view.hpp"
#include "grid_alignment.hpp"
#include "isolated_ink.hpp"
#include "kalibr_grid_search.hpp"
#include "large_board_search.hpp"
#include "luma_distribution.hpp"
#include "motion_regions.hpp"
#include "noisy_ink.hpp"
#include "noisy_print_search.hpp"
#include "small_tag_refinement.hpp"
#include "uniform_raster.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>

namespace aruco3cuda::apriltag_small_cpu {
namespace {
Config alternating_config(const VideoConfig& settings) {
    Config config;
    config.components_.emplace();
    config.components_->local_payload_fit_ = settings.local_payload_fit_;
    config.components_->alternate_contour_threshold_ = settings.alternate_contour_threshold_;
    config.recovery_.emplace();
    return config;
}

Config component_config(const VideoConfig& settings, int window = 0) {
    Config config;
    config.components_.emplace();
    config.components_->sharp_edge_refinement_ = true;
    config.components_->local_payload_fit_ = settings.local_payload_fit_;
    if (window) {
        config.components_->sampled_contours_ = true;
        config.components_->local_midrange_side_samples_ = window;
    }
    return config;
}

struct SceneProfile {
    bool smooth_midtone = false;
    bool bright_field = false;
    bool uniform_field = false;
    bool smooth_gradient = false;
    bool light_surround = false;
    bool compact_contrast = false;
    bool broad_contrast = false;
    bool midtone_edges = false;
    int background_low = 0;
    bool sparse_dark = false;
    bool smooth_ink = false;
    bool smooth_dark_field = false;
    bool two_cell_field = false;
    bool dark_ink_on_light_field = false;
    bool soft_gradient = false;
    bool textured_matte = false;
    bool sparse_edges = false;
    bool noisy_matte = false;
};
SceneProfile profile(const cv::Mat& gray, const VideoConfig& config, bool previously_smooth) {
    if (gray.rows < 8 || gray.cols < 8)
        return {};
    detail::LumaDistribution::Bins histogram{};
    int rough = 0, moderate = 0, fine = 0;
    const int row_step = std::max(13, gray.rows / 64);
    const int column_step = std::max(17, gray.cols / 88);
    const int count = (1 + (gray.rows - 8) / row_step) * (1 + (gray.cols - 8) / column_step);
    const int allowed_fine = count * config.isolated_fine_per_mille_ / 1000;
    const int allowed_rough = count * config.smooth_max_rough_per_mille_ / 1000;
    const auto scan_row = [&](auto check_fine, const std::uint8_t* row, const std::uint8_t* next) {
        for (int x = 4; x + 3 < gray.cols; x += column_step) {
            const int value = row[x], across = row[x + 3], below = next[x];
            ++histogram[value];
            const int change = std::max(std::abs(value - across), std::abs(value - below));
            rough += change >= 16;
            moderate += change >= config.faint_background_difference_ && change < 16;
            // A variation of at most two from the anchor bounds all three
            // pair differences by four. Reuse the same current-frame loads.
            if constexpr (decltype(check_fine)::value) {
                if (change >= 3) {
                    const int low = std::min({value, across, below});
                    const int high = std::max({value, across, below});
                    fine += (low >= 80) & (high <= 200) & (high - low >= 5);
                }
            }
        }
    };
    for (int y = 4; y + 3 < gray.rows; y += row_step) {
        const auto* row = gray.ptr<std::uint8_t>(y);
        const auto* next = gray.ptr<std::uint8_t>(y + 3);
        if (rough <= allowed_rough && fine <= allowed_fine)
            scan_row(std::true_type{}, row, next);
        else
            scan_row(std::false_type{}, row, next);
    }
    const detail::LumaDistribution luma(histogram);
    const int low = luma.percentile(10), high = luma.percentile(90);
    const int span = high - low;
    const auto at_most = [count](int samples, int per_mille) {
        return std::int64_t{samples} * 1000 <= std::int64_t{count} * per_mille;
    };
    const auto at_least = [count](int samples, int per_mille) {
        return std::int64_t{samples} * 1000 >= std::int64_t{count} * per_mille;
    };
    const bool low_texture = rough <= allowed_rough;
    const bool gradient_texture = at_most(rough, config.gradient_max_rough_per_mille_);
    const int minimum_span = config.smooth_minimum_luma_span_;
    const int relaxed_span = minimum_span - config.smooth_hysteresis_levels_;

    SceneProfile scene;
    scene.background_low = low;
    // Hysteresis keeps small exposure changes from interrupting alternating discovery.
    const int margin = previously_smooth ? config.smooth_hysteresis_levels_ : 0;
    scene.smooth_midtone = low >= config.smooth_low_percentile_floor_ - margin &&
                           high <= config.smooth_high_percentile_ceiling_ + margin &&
                           span >= minimum_span - margin &&
                           at_most(rough, config.smooth_max_rough_per_mille_ + 3 * margin);
    scene.bright_field = low >= config.bright_field_minimum_;
    scene.uniform_field = scene.bright_field && high <= config.smooth_high_percentile_ceiling_ &&
                          span <= config.uniform_field_maximum_span_;
    scene.smooth_gradient = span >= minimum_span && gradient_texture;
    scene.light_surround = low_texture && low >= config.border_surround_minimum_ &&
                           span <= config.border_surround_maximum_span_;
    scene.compact_contrast = low >= config.board_background_floor_ &&
                             low < config.border_surround_minimum_ && span >= minimum_span &&
                             high > config.smooth_low_percentile_floor_ &&
                             at_most(rough, 2 * config.smooth_max_rough_per_mille_);
    const int midtones = luma.below(180) - luma.below(30);
    scene.broad_contrast = low <= config.large_board_dark_limit_ &&
                           high >= config.large_board_bright_floor_ &&
                           at_least(midtones, config.large_board_midtones_per_mille_) &&
                           at_most(rough, config.large_board_max_rough_per_mille_);
    const bool moderate_texture = at_most(rough, 3 * config.smooth_max_rough_per_mille_);
    scene.midtone_edges = low > config.midtone_boundary_.luma_threshold_ &&
                          low < config.smooth_low_percentile_floor_ &&
                          high <= config.midtone_boundary_ceiling_ && moderate_texture;
    const int isolated_dark = luma.below(low - config.isolated_luma_margin_ + 1);
    scene.sparse_dark = at_most(isolated_dark, config.isolated_dark_per_mille_);
    scene.smooth_ink = low >= config.board_background_floor_ && span >= relaxed_span &&
                       !scene.light_surround && low_texture && fine <= allowed_fine;
    scene.smooth_dark_field =
            low_texture && low < config.smooth_low_percentile_floor_ && span >= minimum_span;
    if (config.kalibr_.enabled_) {
        const int dark = luma.below(config.kalibr_.preview_dark_level_ + 1);
        const int light = count - luma.below(config.kalibr_light_level_);
        scene.two_cell_field = low <= config.kalibr_field_low_ &&
                               high <= config.kalibr_field_high_ &&
                               at_least(rough, config.kalibr_minimum_rough_per_mille_) &&
                               at_least(dark, config.kalibr_minimum_dark_per_mille_) &&
                               at_least(light, config.kalibr_minimum_light_per_mille_);
        scene.dark_ink_on_light_field = scene.light_surround && dark > 0;
    }
    if (config.maximum_isolated_regions_ > 0 && !scene.light_surround && span >= relaxed_span &&
        gradient_texture) {
        const int darkest = luma.level_at_rank(1), lightest = luma.level_at_rank(count);
        // A faint print may share the surround with a strong print at one exposure tail.
        scene.soft_gradient = lightest - darkest <= config.faint_maximum_scene_span_ ||
                              lightest - high <= config.isolated_luma_margin_ ||
                              low - darkest <= config.isolated_luma_margin_;
    }
    scene.textured_matte = config.maximum_isolated_regions_ &&
                           span >= config.faint_decode_contrast_ &&
                           span <= config.border_surround_maximum_span_ && moderate_texture &&
                           at_least(moderate, config.faint_minimum_moderate_per_mille_);
    scene.sparse_edges = config.sparse_faint_.guide_side_px_ > 0 &&
                         at_most(moderate + rough, config.sparse_faint_.maximum_changed_per_mille_);
    scene.noisy_matte = !scene.two_cell_field && !scene.light_surround &&
                        detail::NoisyInk::plausible(gray, config, config.read_noise_, row_step,
                                                    column_step, count, rough, span);
    return scene;
}

bool same_location(const Detection& a, const Detection& b) {
    const auto center = [](const detail::Quad& q) { return (q[0] + q[1] + q[2] + q[3]) * 0.25F; };
    const auto diagonal = [](const detail::Quad& q) { return detail::length(q[0] - q[2]); };
    return detail::length(center(a.corners_px_) - center(b.corners_px_)) <
           0.25F * std::min(diagonal(a.corners_px_), diagonal(b.corners_px_));
}

void merge(std::vector<Detection>& results, std::vector<Detection> proposals, cv::Point origin = {},
           int scale = 1) {
    for (auto& tag : proposals) {
        for (auto& p : tag.corners_px_) {
            p.x = (p.x + 0.5F) / scale - 0.5F + origin.x;
            p.y = (p.y + 0.5F) / scale - 0.5F + origin.y;
        }
        if (std::none_of(results.begin(), results.end(),
                         [&](const Detection& other) { return same_location(tag, other); }))
            results.push_back(tag);
    }
}

void merge_noisy(std::vector<Detection>& results, const std::vector<Detection>& proposals) {
    for (const auto& candidate : proposals) {
        const auto prior = std::find_if(results.begin(), results.end(), [&](const Detection& tag) {
            return same_location(tag, candidate);
        });
        if (prior == results.end()) {
            results.push_back(candidate);
            continue;
        }
        if (prior->id_ != candidate.id_ || candidate.corrected_bits_ > prior->corrected_bits_)
            continue;
        const auto& corners = prior->corners_px_;
        const float diagonal = detail::length(corners[2] - corners[0]);
        bool agrees = true;
        for (int j = 0; j < 4; ++j)
            agrees &= detail::length(corners[j] - candidate.corners_px_[j]) < .18F * diagonal;
        if (agrees)
            *prior = candidate;
    }
}

cv::Rect track_region(const Detection& tag, cv::Point translation, const cv::Rect& bounds) {
    const auto& q = tag.corners_px_;
    const cv::Rect box = cv::boundingRect(cv::Mat(q, false));
    const int pad = std::clamp(std::max(box.width, box.height) / 3, 12, 48);
    return (box + translation - cv::Point(pad, pad) + cv::Size(2 * pad, 2 * pad)) & bounds;
}

}  // namespace

class VideoDetector::Impl {
public:
    explicit Impl(const VideoConfig& config)
        : settings_(config),
          motion_(config.quiet_difference_, config.noisy_difference_,
                  config.changed_region_padding_px_, 0, config.maximum_changed_regions_),
          full_(alternating_config(config)),
          patch_(component_config(config)),
          local_(component_config(config, config.patch_local_window_)),
          wide_(component_config(config, config.full_frame_local_window_)),
          entry_(config.pan_strip_width_px_, config.pan_strip_threshold_,
                 config.pan_photometric_minimum_contrast_),
          border_(config.boundary_, Config{}.maximum_corrected_bits_),
          pyramid_border_(
                  [&] {
                      auto coarse = config.boundary_;
                      coarse.fit_only_corrected_candidates_ = false;
                      return coarse;
                  }(),
                  Config{}.maximum_corrected_bits_) {
        const auto& fit = config.large_tag_refinement_;
        if (fit.maximum_tags_ > 512 ||
            !(fit.minimum_side_px_ >= 8 && fit.minimum_side_px_ < 32768) ||
            !(fit.minimum_short_to_long_ratio_ > 0 && fit.minimum_short_to_long_ratio_ <= 1) ||
            fit.search_radius_px_ < 1 || fit.search_radius_px_ > 16 ||
            !(fit.maximum_corner_motion_px_ >= 0 && fit.maximum_corner_motion_px_ <= 128) ||
            !(fit.maximum_quadratic_px_ >= 0 && fit.maximum_quadratic_px_ <= 128) ||
            !(fit.minimum_blur_ratio_ > 0 && fit.minimum_blur_ratio_ <= 64) ||
            fit.minimum_ring_matches_ < 1 || fit.minimum_ring_matches_ > detail::kRingSamples ||
            !(fit.minimum_decode_contrast_ > 0 && fit.minimum_decode_contrast_ <= 255))
            throw std::invalid_argument("Invalid large-tag refinement settings");
        const auto& small_fit = config.small_tag_refinement_;
        if (!(small_fit.minimum_side_px_ >= 8.F &&
              small_fit.minimum_side_px_ <= small_fit.maximum_side_px_ &&
              small_fit.maximum_side_px_ <= 128.F) ||
            !(small_fit.maximum_relative_residual_ > 0.F &&
              small_fit.maximum_relative_residual_ <= 1.F))
            throw std::invalid_argument("Invalid small-tag refinement settings");
        const auto& retry = config.stationary_patch_retry_;
        if (retry.minimum_change_side_px_ < 1 ||
            retry.small_change_limit_px_ < retry.minimum_change_side_px_ ||
            retry.maximum_change_side_px_ < retry.small_change_limit_px_ ||
            retry.maximum_change_side_px_ >= 32768 || retry.minimum_ring_matches_ < 1 ||
            retry.minimum_ring_matches_ > detail::kRingSamples ||
            !(retry.minimum_decode_contrast_ > 0 && retry.minimum_decode_contrast_ <= 255))
            throw std::invalid_argument("Invalid stationary patch retry settings");
        detail::NoisyInk::validate(config.read_noise_);
        if (config.kalibr_field_low_ < 0 || config.kalibr_field_high_ > 255 ||
            config.kalibr_field_low_ > config.kalibr_field_high_ ||
            config.kalibr_light_level_ < 0 || config.kalibr_light_level_ > 255 ||
            config.kalibr_minimum_rough_per_mille_ < 0 ||
            config.kalibr_minimum_rough_per_mille_ > 1000 ||
            config.kalibr_minimum_dark_per_mille_ < 0 ||
            config.kalibr_minimum_dark_per_mille_ > 1000 ||
            config.kalibr_minimum_light_per_mille_ < 0 ||
            config.kalibr_minimum_light_per_mille_ > 1000)
            throw std::invalid_argument("Invalid two-cell preview settings");
        if (!std::isfinite(config.maximum_diffuse_reseed_cells_) ||
            config.maximum_diffuse_reseed_cells_ < 0 || config.maximum_diffuse_reseed_cells_ > 1)
            throw std::invalid_argument("invalid diffuse corner displacement");
        const auto& sparse = config.sparse_faint_;
        if ((sparse.guide_side_px_ != 0 &&
             (sparse.guide_side_px_ < 32 || sparse.guide_side_px_ > 1024)) ||
            sparse.maximum_initial_step_px_ < 1 || sparse.maximum_changed_per_mille_ < 0 ||
            sparse.maximum_changed_per_mille_ > 1000 ||
            !std::isfinite(sparse.maximum_box_aspect_ratio_) ||
            sparse.maximum_box_aspect_ratio_ < 1 || !std::isfinite(sparse.minimum_hull_fill_) ||
            sparse.minimum_hull_fill_ < 0 || sparse.minimum_hull_fill_ > 1)
            throw std::invalid_argument("invalid sparse faint-guide configuration");
        if (config.faint_guide_side_px_ < 32 || config.faint_guide_side_px_ > 1024 ||
            config.faint_maximum_scene_span_ < 0 || config.faint_maximum_scene_span_ > 255 ||
            config.adaptive_average_minimum_noise_per_mille_ < 0 ||
            config.adaptive_average_minimum_noise_per_mille_ > 1000 ||
            config.faint_background_difference_ < 1 || config.faint_background_difference_ > 255 ||
            config.faint_minimum_moderate_per_mille_ < 0 ||
            config.faint_minimum_moderate_per_mille_ > 1000 || config.faint_decode_contrast_ < 1 ||
            config.faint_decode_contrast_ > 255 || config.faint_background_window_ < 3 ||
            config.faint_background_window_ > 63 || config.faint_background_window_ % 2 == 0)
            throw std::invalid_argument("Invalid faint-ink contrast");
        if (config.isolated_guide_side_px_ < 32 || config.isolated_guide_side_px_ > 32767 ||
            config.isolated_fine_per_mille_ < 0 || config.isolated_fine_per_mille_ > 1000 ||
            config.isolated_residual_contrast_ < 1 || config.isolated_residual_contrast_ > 255)
            throw std::invalid_argument("Invalid smooth-field ink search settings");
        detail::validate_border_config(config.midtone_boundary_);
        if (config.midtone_boundary_reference_side_ < 32 || config.secondary_ink_threshold_ < 0 ||
            config.secondary_ink_threshold_ > config.midtone_boundary_.luma_threshold_ ||
            config.midtone_boundary_ceiling_ < 0 || config.midtone_boundary_ceiling_ > 255 ||
            !(config.midtone_edge_refinement_margin_ratio_ >= 0 &&
              config.midtone_edge_refinement_margin_ratio_ <= 1))
            throw std::invalid_argument("Invalid midtone boundary search configuration");
        detail::validate_border_config(config.boundary_);
        if (config.isolated_luma_margin_ < 0 || config.isolated_luma_margin_ > 255 ||
            config.isolated_dark_per_mille_ < 0 || config.isolated_dark_per_mille_ > 1000 ||
            config.maximum_isolated_regions_ < 0 || config.full_search_interval_ < 1 ||
            config.maximum_changed_regions_ < 0 || config.maximum_upsample_area_px_ < 0 ||
            config.full_frame_alignment_side_px_ < 0 || config.smooth_low_percentile_floor_ < 0 ||
            config.smooth_high_percentile_ceiling_ > 255 ||
            config.smooth_low_percentile_floor_ > config.smooth_high_percentile_ceiling_ ||
            config.smooth_hysteresis_levels_ < 0 || config.smooth_hysteresis_levels_ > 64 ||
            config.bright_field_minimum_ < 0 || config.bright_field_minimum_ > 255 ||
            config.gradient_max_rough_per_mille_ < 0 ||
            config.gradient_max_rough_per_mille_ > 1000 ||
            config.border_pyramid_reference_side_ < 32 || config.border_tile_contrast_ < 1 ||
            config.border_tile_contrast_ > 255 || config.border_surround_minimum_ < 0 ||
            config.border_surround_minimum_ > 255 || config.border_surround_maximum_span_ < 0 ||
            config.border_surround_maximum_span_ > 255 || config.pan_strip_width_px_ < 0 ||
            config.pan_strip_width_px_ >= 32768 || config.uniform_field_maximum_span_ < 0 ||
            config.uniform_field_maximum_span_ > 255 || config.large_board_dark_limit_ < 0 ||
            config.large_board_bright_floor_ > 255 ||
            config.large_board_dark_limit_ >= config.large_board_bright_floor_ ||
            config.large_board_midtones_per_mille_ < 0 ||
            config.large_board_midtones_per_mille_ > 1000 ||
            config.large_board_max_rough_per_mille_ < 0 ||
            config.large_board_max_rough_per_mille_ > 1000 || config.board_background_floor_ < 0 ||
            config.board_background_floor_ > 255 || config.board_region_padding_px_ < 0 ||
            config.board_region_padding_px_ >= 32768 || config.board_dark_level_ < 0 ||
            config.board_light_level_ > 255 ||
            config.board_dark_level_ >= config.board_light_level_ ||
            !(config.compact_board_area_ratio_ >= 0 && config.compact_board_area_ratio_ <= 1) ||
            config.dense_maximum_side_px_ < 8 || config.dense_seed_count_ < 2 ||
            !std::isfinite(config.dispersed_seed_spacing_sides_) ||
            config.dispersed_seed_spacing_sides_ < 0 || config.board_search_hold_frames_ < 0 ||
            config.dense_count_ == 0 || config.smooth_minimum_luma_span_ < 0 ||
            config.smooth_minimum_luma_span_ > 255 || config.smooth_max_rough_per_mille_ < 0 ||
            config.smooth_max_rough_per_mille_ > 1000)
            throw std::invalid_argument("Invalid AprilTag video settings");
    }

    void refine_published_corners(const cv::Mat& gray, std::vector<Detection>& tags) const {
        detail::refine_small_tag_corners(gray, tags, this->settings_.small_tag_refinement_);
    }

    void reset() noexcept {
        full_.reset();
        motion_.reset();
        previous_.clear();
        size_ = {};
        previous_two_cell_field_ = false;
        frame_ = 0;
        previously_smooth_ = false;
        board_frames_left_ = 0;
    }

    std::vector<Detection> detect(const cv::Mat& gray) {
        cv::Mat area4;
        std::vector<Detection> outline_retries;
        std::optional<bool> native_noise;
        auto tags = this->detect_frame_regions(gray, area4, outline_retries, native_noise);
        this->noisy_print_.append(gray, this->settings_, tags, area4, native_noise);
        merge(tags, std::move(outline_retries));
        return tags;
    }

    std::vector<Detection> detect_frame_regions(const cv::Mat& gray, cv::Mat& area4,
                                                std::vector<Detection>& outline_retries,
                                                std::optional<bool>& native_noise) {
        if (gray.size() != size_) {
            reset();
            size_ = gray.size();
        }
        const auto scene = profile(gray, settings_, previously_smooth_);
        cv::Mat* shared_area = this->settings_.noisy_print_.enabled_ ? &area4 : nullptr;
#if defined(__aarch64__) && defined(__linux__) && defined(__AARCH64EL__)
        // An exact quarter guide can use the same samples without storing the block
        // average when the current frame's noise decision already precludes its use.
        if (shared_area && !scene.noisy_matte && (scene.soft_gradient || scene.textured_matte) &&
            gray.cols % 64 == 0 && gray.rows % 4 == 0) {
            const auto shorter = std::min(gray.cols, gray.rows);
            const auto& cfg = this->settings_;
            if (shorter == 4LL * cfg.faint_guide_side_px_ ||
                shorter == 4LL * cfg.sparse_faint_.guide_side_px_) {
                native_noise = this->noisy_print_.has_native_noise(gray);
                if (!*native_noise)
                    shared_area = nullptr;
            }
        }
#endif

        const auto refine_texture = [&](std::vector<Detection>& tags) {
            // A dense board already has native geometry. Straight outer-rail refits
            // on many adjacent prints can bias peripheral corners under lens distortion.
            if (scene.textured_matte && settings_.refine_diffuse_edges_ &&
                tags.size() < settings_.dense_count_)
                detail::refine_diffuse_tags(gray, settings_.faint_decode_contrast_, tags,
                                            settings_.maximum_diffuse_reseed_cells_);
        };
        using detail::FaintGuide;
        const bool sparse = scene.soft_gradient && scene.sparse_edges;
        const auto& limits = settings_.sparse_faint_;
        const bool sparse_first =
                sparse && std::min(gray.rows, gray.cols) <= std::int64_t{limits.guide_side_px_} *
                                                                    limits.maximum_initial_step_px_;
        bool used_local_mean = false;
        std::vector<Detection> faint;
        if (scene.soft_gradient) {
            const auto mode = sparse_first ? FaintGuide::sparse
                                           : (scene.textured_matte ? FaintGuide::adaptive_fallback
                                                                   : FaintGuide::exterior);
            faint = isolated_ink_.find_faint(gray, settings_, mode, nullptr, &used_local_mean,
                                             shared_area);
            if (!scene.two_cell_field && !scene.dark_ink_on_light_field &&
                isolated_ink_.quiet_outside(gray, faint, settings_.faint_background_difference_)) {
                full_.reset();
                motion_.reset();
                frame_ = 0;
                board_frames_left_ = 0;
                previously_smooth_ = false;
                previous_ = faint;
                previous_two_cell_field_ = false;
                refine_texture(faint);
                return faint;
            }
        }
        auto tags = detect_standard(gray, scene);
        // The ordinary tracker owns one-cell payloads. A newly verified second print
        // convention must not replace that history, including on a mixed board.
        const cv::Rect bounds({}, gray.size());
        const auto region = scene.two_cell_field            ? bounds
                            : scene.dark_ink_on_light_field ? kalibr_.uncovered_region(gray, tags)
                                                            : cv::Rect{};
        // A verified board in the preceding frame can justify the native search.
        // Every returned marker still has to decode from the current frame.
        const bool request_native = scene.two_cell_field && previous_two_cell_field_;
        previous_two_cell_field_ = false;
        if (!region.empty() && (request_native || kalibr_.has_candidate(gray(region)))) {
            const auto& two_cell = kalibr_.detect(gray(region), scene.two_cell_field);
            previous_two_cell_field_ = scene.two_cell_field && !two_cell.empty();
            merge(tags, two_cell, region.tl());
            merge(tags, kalibr_.ordinary_tags(), region.tl());
            if (scene.dark_ink_on_light_field && !two_cell.empty()) {
                const auto remaining = kalibr_.uncovered_region(gray, tags);
                if (!remaining.empty())
                    merge(tags, native_.detect(gray(remaining)), remaining.tl());
            }
        }
        if (!scene.soft_gradient && scene.textured_matte)
            merge(tags, isolated_ink_.find_faint(gray, settings_, FaintGuide::adaptive_fallback,
                                                 &tags, nullptr, shared_area));
        if (!faint.empty()) {
            merge(faint, std::move(tags));
            tags = std::move(faint);
        }
        // A coarse first pass on a large raster can perturb an already accurate native fit.
        // Preserve accepted geometry and consider the sparse guide only for unexplained ink.
        if (scene.soft_gradient && (!used_local_mean || (sparse && !sparse_first))) {
            const auto mode = sparse ? FaintGuide::sparse : FaintGuide::adaptive_fallback;
            merge(tags,
                  isolated_ink_.find_faint(gray, settings_, mode, &tags, nullptr, shared_area));
        }
        refine_texture(tags);
        if (scene.noisy_matte) {
            auto proposals = noisy_ink_.find(gray, settings_, settings_.read_noise_, &area4, true);
            merge_noisy(tags, proposals.ordinary_);
            outline_retries = std::move(proposals.outline_retries_);
        }
        return tags;
    }

private:
    void refine_large_tags(const cv::Mat& gray, std::vector<Detection>& tags,
                           bool straight_rails) const {
        const auto& config = this->settings_.large_tag_refinement_;
        if (!config.enabled_ || tags.empty() || tags.size() > config.maximum_tags_ ||
            (!straight_rails && gray.total() < config.minimum_frame_pixels_))
            return;

        for (auto& tag : tags) {
            const auto& corners = tag.corners_px_;
            float shortest_px = std::numeric_limits<float>::infinity(), longest_px = 0;
            for (int i = 0; i < 4; ++i) {
                const float edge_px = detail::length(corners[(i + 1) % 4] - corners[i]);
                shortest_px = std::min(shortest_px, edge_px);
                longest_px = std::max(longest_px, edge_px);
            }
            if (shortest_px < config.minimum_side_px_ ||
                (!straight_rails && shortest_px < config.minimum_short_to_long_ratio_ * longest_px))
                continue;
            detail::Quad refined;
            if (!detail::fit_long_tag_edges(gray, corners, refined, config, straight_rails))
                continue;
            if (detail::verify_payload(gray, refined, tag.id_, config.minimum_ring_matches_,
                                       config.minimum_decode_contrast_, tag.corrected_bits_))
                tag.corners_px_ = refined;
        }
    }

    std::vector<Detection> detect_standard(const cv::Mat& gray, const SceneProfile& scene) {
        const bool isolated_field =
                scene.smooth_ink && !scene.midtone_edges && !scene.light_surround;
        if (isolated_field) {
            std::vector<Detection> isolated;
            if (isolated_ink_.find(gray, scene.background_low, settings_, native_, patch_,
                                   isolated)) {
                full_.reset();
                motion_.reset();
                previous_ = isolated;
                previously_smooth_ = false;
                board_frames_left_ = 0;
                frame_ = 0;
                return isolated;
            }
        }
        auto tags = detect_frame(gray, scene);
        if (isolated_field) {
            for (auto& tag : tags)
                detail::try_align_curved_grid(gray, tag);
            previous_ = tags;
        }
        if (scene.midtone_edges) {
            std::vector<Detection> boundary;
            const int scale = std::max(1, cvRound(double(std::min(gray.cols, gray.rows)) /
                                                  settings_.midtone_boundary_reference_side_));
            cv::Mat small = gray;
            if (scale > 1) {
                detail::resize_nearest_guide(gray, edge_small_, scale, cv::INTER_NEAREST_EXACT);
                small = edge_small_;
            }
            ink_.append(small, boundary);
            dark_ink_.append(small, boundary);
            // The same exposure can conceal a separate marker wholly inside the frame.
            merge(boundary, ink_components_.detect(small));
            if (scale > 1) {
                const float sx = float(gray.cols) / small.cols, sy = float(gray.rows) / small.rows;
                for (auto& tag : boundary)
                    for (auto& corner : tag.corners_px_)
                        corner = {(corner.x + .5F) * sx - .5F, (corner.y + .5F) * sy - .5F};
            }
            // Sampled ink components can duplicate a native dense-board result.
            // Keep native geometry for matching interior IDs, retaining the
            // boundary-specific geometry where a corner touches the image edge.
            if (tags.size() >= settings_.dense_count_) {
                for (auto& sampled : boundary) {
                    const auto match =
                            std::find_if(tags.begin(), tags.end(), [&](const Detection& tag) {
                                return tag.id_ == sampled.id_ && same_location(tag, sampled);
                            });
                    if (match != tags.end() &&
                        std::all_of(match->corners_px_.begin(), match->corners_px_.end(),
                                    [&](cv::Point2f p) {
                                        return p.x > 2 && p.y > 2 && p.x < gray.cols - 3 &&
                                               p.y < gray.rows - 3;
                                    }))
                        sampled = *match;
                }
            }
            merge(boundary, std::move(tags));
            ink_.align_visible(gray, boundary, settings_.midtone_edge_refinement_margin_ratio_);
            return boundary;
        }
        return tags;
    }

private:
    std::vector<Detection> detect_frame(const cv::Mat& gray, const SceneProfile& scene) {
        if (scene.bright_field && scene.light_surround) {
            std::vector<Detection> exact;
            if (raster_.read(gray, exact)) {
                motion_.reset();
                frame_ = 0;
                previously_smooth_ = false;
                previous_ = exact;
                return exact;
            }
        }
        std::vector<Detection> boundary;
        bool current_edge_pyramid = false;
        if (scene.light_surround) {
            border_.append(gray, boundary);
            const int scale =
                    std::min(gray.rows, gray.cols) / settings_.border_pyramid_reference_side_;
            if (scale >= 2 && has_dark_boundary(gray)) {
                detail::resize_nearest_guide(gray, edge_small_, scale, cv::INTER_NEAREST_EXACT);
                current_edge_pyramid = true;
                std::vector<Detection> coarse;
                pyramid_border_.append(edge_small_, coarse);
                const float sx = float(gray.cols) / edge_small_.cols;
                const float sy = float(gray.rows) / edge_small_.rows;
                for (auto& tag : coarse)
                    for (auto& corner : tag.corners_px_)
                        corner = {(corner.x + .5F) * sx - .5F, (corner.y + .5F) * sy - .5F};
                merge(boundary, std::move(coarse));
            }
        }
        if (!boundary.empty()) {
            if (current_edge_pyramid)
                append_light_field_components(gray, boundary);
            else
                merge(boundary, border_components_.detect(gray));
            if (boundary.size() < 2)
                for (const auto& prior : previous_) {
                    if (std::none_of(boundary.begin(), boundary.end(), [&](const Detection& known) {
                            return same_location(prior, known);
                        })) {
                        const auto region = track_region(prior, {}, cv::Rect({}, gray.size()));
                        if (!region.empty())
                            verify(gray, region, boundary);
                    }
                }
            border_.align_visible(gray, boundary);
            motion_.reset();
            frame_ = 0;
            previously_smooth_ = false;
            previous_ = boundary;
            return boundary;
        }
        if (scene.sparse_dark && scene.light_surround && !scene.smooth_midtone &&
            !scene.smooth_gradient) {
            std::vector<Detection> isolated;
            if (find_isolated_tags(gray, scene.background_low, isolated)) {
                border_.align_visible(gray, isolated);
                motion_.reset();
                frame_ = 0;
                previously_smooth_ = false;
                previous_ = isolated;
                return isolated;
            }
        }
        previously_smooth_ = scene.smooth_midtone;
        if (small_tag_count(previous_) >= settings_.dense_count_)
            board_frames_left_ = settings_.board_search_hold_frames_;
        const bool possible_board = board_frames_left_ > 0;
        if (possible_board)
            --board_frames_left_;
        if (std::min(gray.cols, gray.rows) < 32 || (scene.smooth_midtone && !possible_board)) {
            motion_.reset();
            frame_ = 0;
            previous_.clear();
            std::vector<Detection> tags;
            if (!scene.smooth_midtone || !scene.sparse_dark ||
                !find_isolated_tags(gray, scene.background_low, tags))
                merge(tags, full_.detect(gray));
            if (scene.light_surround)
                border_.align_visible(gray, tags);
            if (small_tag_count(tags) >= settings_.dense_count_) {
                auto board = search_board(gray, board_region(gray));
                if (small_tag_count(board) >= settings_.dense_count_) {
                    merge(board, std::move(tags));
                    previous_ = board;
                    return board;
                }
            }
            return tags;
        }
        if (scene.broad_contrast) {
            auto large = large_.detect_mixed(gray);
            if (!large.empty()) {
                previous_ = large;
                motion_.reset();
                frame_ = 0;
                this->refine_large_tags(gray, large, true);
                return large;
            }
        }
        cv::Rect board_box;
        if (possible_board || scene.compact_contrast)
            board_box = board_region(gray);
        const bool compact_board =
                !board_box.empty() &&
                board_box.area() < settings_.compact_board_area_ratio_ * gray.total();
        if (scene.bright_field || scene.smooth_gradient || possible_board || compact_board) {
            motion_.reset();
            frame_ = 0;
            previous_.clear();
            if (possible_board || compact_board)
                previous_ = search_board(gray, board_box);
            const bool isolated = previous_.empty() && scene.smooth_gradient && scene.sparse_dark &&
                                  find_isolated_tags(gray, scene.background_low, previous_);
            if (!isolated && previous_.size() < settings_.dense_count_)
                merge(previous_, full_fallback(gray, !possible_board));
            if (scene.uniform_field && previous_.empty())
                previous_ = native_.detect(gray);
            if (scene.light_surround)
                border_.align_visible(gray, previous_);
            return previous_;
        }
        cv::Point translation;
        std::vector<cv::Rect> regions;
        bool reliable_overlap = false;
        const bool localized =
                motion_.update(gray, frame_ == 0, translation, regions, &reliable_overlap);
        std::vector<Detection> tags;
        if (localized) {
            const bool one_change = regions.size() == 1;
            for (const auto& region : regions)
                verify(gray, region, tags);
            verify_tracks(gray, translation, regions, tags);
            merge(tags, entry_.detect(gray, translation));
            const auto& retry = this->settings_.stationary_patch_retry_;
            if (retry.enabled_ && one_change && reliable_overlap && translation == cv::Point() &&
                gray.total() >= retry.minimum_frame_pixels_)
                merge(tags, this->retry_stationary_patch(gray, regions.front(), tags.empty()));
        } else {
            tags = full_fallback(gray, true);
            if (tags.empty() && frame_ == 0 && previous_.empty() && scene.smooth_dark_field) {
                auto board = search_board(gray, {});
                if (small_tag_count(board) >= settings_.dense_seed_count_)
                    tags = std::move(board);
            }
            if (reliable_overlap && settings_.full_frame_local_window_ > 0)
                merge(tags, wide_.detect(gray));
            // A periodic refresh changes the proposal method. Re-read tracked
            // locations that the full search missed before discarding the history.
            if (frame_ == 0) {
                for (const auto& last : previous_) {
                    if (std::any_of(tags.begin(), tags.end(), [&](const Detection& tag) {
                            return same_location(tag, last) && tag.id_ == last.id_;
                        }))
                        continue;
                    const auto box = track_region(last, {}, cv::Rect({}, gray.size()));
                    if (!box.empty())
                        verify(gray, box, tags);
                }
            }
        }
        for (auto& tag : tags) {
            if (tags.size() >= settings_.dense_count_)
                break;
            const auto& q = tag.corners_px_;
            if (localized || std::max(detail::length(q[0] - q[1]), detail::length(q[1] - q[2])) >
                                     settings_.full_frame_alignment_side_px_)
                detail::align_decoded_grid(gray, tag);
            else
                detail::try_align_sharp_grid(gray, tag);
        }
        if (scene.light_surround)
            border_.align_visible(gray, tags);
        frame_ = (frame_ + 1) % settings_.full_search_interval_;
        previous_ = tags;
        this->refine_large_tags(gray, tags, false);
        return tags;
    }

private:
    bool find_isolated_tags(const cv::Mat& gray, int background_low, std::vector<Detection>& tags) {
        if (!settings_.maximum_isolated_regions_ || std::min(gray.cols, gray.rows) < 32)
            return false;
        const int level = background_low - settings_.isolated_luma_margin_;
        cv::Rect crop;
        if (gray.cols % 4 == 0 && gray.rows % 4 == 0 && cv::getNumThreads() <= 1) {
            const auto active = detail::threshold_quarter_guide(gray, level, isolated_);
            if (active.empty())
                return false;
            crop = (active + cv::Size(2, 2) - cv::Point(1, 1)) & cv::Rect({}, isolated_.size());
        } else {
            detail::resize_nearest_guide(gray, isolated_, 4, cv::INTER_NEAREST);
            cv::threshold(isolated_, isolated_, level, 255, cv::THRESH_BINARY_INV);
            crop = cv::Rect({}, isolated_.size());
        }
        auto mask = isolated_(crop);
        cv::dilate(mask, mask, cv::Mat{}, {-1, -1}, 1, cv::BORDER_CONSTANT | cv::BORDER_ISOLATED);
        // Padding must see the zero exterior of the cropped binary mask.
        const cv::Mat view(mask.rows, mask.cols, mask.type(), mask.data, mask.step[0]);
        std::vector<std::vector<cv::Point>> candidates;
        cv::findContours(view, candidates, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, crop.tl());
        if (candidates.size() > static_cast<std::size_t>(settings_.maximum_isolated_regions_))
            return false;
        bool all_found = !candidates.empty();
        const cv::Rect bounds({}, gray.size());
        for (const auto& contour : candidates) {
            const auto box = cv::boundingRect(contour);
            if (std::min(box.width, box.height) < 3 || std::max(box.width, box.height) > 32) {
                all_found = false;
                continue;
            }
            const cv::Rect region = cv::Rect(box.x * 4 - 12, box.y * 4 - 12, box.width * 4 + 28,
                                             box.height * 4 + 28) &
                                    bounds;
            const auto before = tags.size();
            auto found = native_.detect(gray(region));
            if (found.empty())
                verify(gray, region, tags);
            else
                merge(tags, std::move(found), region.tl());
            for (std::size_t i = before; i < tags.size(); ++i)
                detail::align_isolated_grid(gray, tags[i]);
            all_found &= tags.size() > before;
        }
        return all_found;
    }

    // Sparse dark support in the current pyramid bounds the native-pixel
    // component search. Separate islands can contain tags far from the boundary.
    void append_light_field_components(const cv::Mat& gray, std::vector<Detection>& tags) {
        constexpr int tile = 12;
        const auto& small = edge_small_;
        const double sx = double(gray.cols) / small.cols, sy = double(gray.rows) / small.rows;
        islands_.create((small.rows + tile - 1) / tile, (small.cols + tile - 1) / tile, CV_8U);
        islands_.setTo(0);
        for (int by = 0; by < islands_.rows; ++by) {
            auto* active = islands_.ptr<std::uint8_t>(by);
            const int bottom = std::min(small.rows, (by + 1) * tile);
            for (int bx = 0; bx < islands_.cols; ++bx) {
                const int right = std::min(small.cols, (bx + 1) * tile);
                int low = 255, high = 0;
                for (int y = by * tile; !active[bx] && y < bottom; y += 2) {
                    const auto* row = small.ptr<std::uint8_t>(y);
                    for (int x = bx * tile; x < right; x += 2) {
                        low = std::min(low, int(row[x]));
                        high = std::max(high, int(row[x]));
                    }
                    if (low <= settings_.border_surround_minimum_ ||
                        high - low >= settings_.border_tile_contrast_)
                        active[bx] = 255;
                }
            }
        }
        cv::dilate(islands_, islands_, cv::Mat{});
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(islands_, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for (const auto& contour : contours) {
            auto box = cv::boundingRect(contour);
            const int x = cvFloor(box.x * tile * sx), y = cvFloor(box.y * tile * sy);
            const int right = std::min(gray.cols, cvCeil((box.x + box.width) * tile * sx));
            const int bottom = std::min(gray.rows, cvCeil((box.y + box.height) * tile * sy));
            const cv::Rect region(x, y, right - x, bottom - y);
            merge(tags, border_components_.detect(gray(region)), region.tl());
        }
    }

    bool has_dark_boundary(const cv::Mat& gray) const {
        const int low = settings_.boundary_.luma_threshold_;
        for (int x = 0; x < gray.cols; x += 4)
            if (gray.ptr<std::uint8_t>(0)[x] < low ||
                gray.ptr<std::uint8_t>(gray.rows - 1)[x] < low)
                return true;
        for (int y = 0; y < gray.rows; y += 4)
            if (gray.ptr<std::uint8_t>(y)[0] < low ||
                gray.ptr<std::uint8_t>(y)[gray.cols - 1] < low)
                return true;
        return false;
    }

    std::size_t small_tag_count(const std::vector<Detection>& tags) const {
        return std::count_if(tags.begin(), tags.end(), [&](const Detection& tag) {
            const auto& q = tag.corners_px_;
            return std::max(detail::length(q[1] - q[0]), detail::length(q[2] - q[1])) <=
                   settings_.dense_maximum_side_px_;
        });
    }

    // A small bright/dark board on a large frame can be localized before native
    // component extraction. Include every contrasting tile and a guard band.
    cv::Rect board_region(const cv::Mat& gray) const {
        constexpr int tile = 32;
        int x0 = gray.cols, y0 = gray.rows, x1 = 0, y1 = 0, count = 0;
        const auto& c = settings_;
        for (int y = 0; y < gray.rows; y += tile) {
            const int bottom = std::min(y + tile, gray.rows);
            for (int x = 0; x < gray.cols; x += tile) {
                const int right = std::min(x + tile, gray.cols);
                int low = 255, high = 0;
                for (int row = y;
                     row < bottom && (low >= c.board_dark_level_ || high <= c.board_light_level_);
                     row += 4) {
                    const auto* pixels = gray.ptr<std::uint8_t>(row);
                    for (int col = x; col < right; col += 4) {
                        low = std::min(low, int(pixels[col]));
                        high = std::max(high, int(pixels[col]));
                    }
                }
                if (low < c.board_dark_level_ && high > c.board_light_level_) {
                    x0 = std::min(x0, x);
                    y0 = std::min(y0, y);
                    x1 = std::max(x1, right);
                    y1 = std::max(y1, bottom);
                    ++count;
                }
            }
        }
        const cv::Rect whole({}, gray.size());
        if (count < 3)
            return {};
        const int pad = settings_.board_region_padding_px_;
        return cv::Rect(x0 - pad, y0 - pad, x1 - x0 + 2 * pad, y1 - y0 + 2 * pad) & whole;
    }

    std::vector<Detection> search_board(const cv::Mat& gray, cv::Rect region,
                                        bool verify_seeds = false) {
        if (region.empty())
            region = cv::Rect({}, gray.size());
        const auto patch = gray(region);
        auto tags = board_.detect(patch, verify_seeds);
        for (auto& tag : tags)
            for (auto& p : tag.corners_px_)
                p += cv::Point2f(region.tl());
        if (!verify_seeds)
            large_.recover_oblique(gray, tags);
        return tags;
    }

    std::vector<Detection> full_fallback(const cv::Mat& gray, bool discover_board) {
        const detail::BoardSeedPolicy seeds{settings_.dense_seed_count_, settings_.dense_count_,
                                            float(settings_.dense_maximum_side_px_),
                                            settings_.dispersed_seed_spacing_sides_};
        auto tags = fallback_.detect(gray, discover_board ? &seeds : nullptr);
        if (discover_board && small_tag_count(tags) >= settings_.dense_seed_count_) {
            const auto region = board_region(gray);
            const bool verify_seeds = seeds.dispersed(tags);
            auto board = search_board(gray, region, verify_seeds);
            // Keep the original search region: another low-contrast board may be far
            // from the seeds. Any missing or newly decoded print triggers all retries.
            if (verify_seeds &&
                (board.size() != tags.size() ||
                 !std::all_of(board.begin(), board.end(), [&](const Detection& observed) {
                     return std::any_of(tags.begin(), tags.end(), [&](const Detection& seed) {
                         return observed.id_ == seed.id_ && same_location(observed, seed);
                     });
                 })))
                board = search_board(gray, region);
            merge(board, std::move(tags));
            return board;
        }
        return tags;
    }

    void verify_tracks(const cv::Mat& gray, cv::Point translation, std::vector<cv::Rect>& searched,
                       std::vector<Detection>& tags) {
        const cv::Rect bounds({}, gray.size());
        for (const auto& last : previous_) {
            const auto box = track_region(last, translation, bounds);
            if (box.empty() || std::any_of(searched.begin(), searched.end(),
                                           [&](const cv::Rect& r) { return (r & box) == box; }))
                continue;
            // Changed crops already sought every marker they could contain. In an unsearched,
            // unchanged crop the translated quad is a proposal for current-frame decode.
            const bool overlaps =
                    std::any_of(searched.begin(), searched.end(),
                                [&](const cv::Rect& r) { return !(r & box).empty(); });
            if (!overlaps) {
                if (auto current = patch_.recover_translated(gray, last, translation)) {
                    merge(tags, {*current});
                    continue;
                }
            }
            verify(gray, box, tags);
            searched.push_back(box);
        }
    }

    std::vector<Detection> retry_stationary_patch(const cv::Mat& gray, cv::Rect change,
                                                  bool ordinary_search_empty) {
        const auto& settings = this->settings_.stationary_patch_retry_;
        const int shorter = std::min(change.width, change.height);
        const int longer = std::max(change.width, change.height);
        const bool small = longer <= settings.small_change_limit_px_;
        if (shorter < settings.minimum_change_side_px_ ||
            longer > settings.maximum_change_side_px_ || (small && !ordinary_search_empty))
            return {};

        // Payload-only changes can be much narrower than the print. Large changed
        // patches need a smaller relative margin to bound the duplicate search.
        const int margin = small ? longer : (longer + 3) / 4;
        const cv::Rect roi =
                (change - cv::Point(margin, margin) + cv::Size(2 * margin, 2 * margin)) &
                cv::Rect({}, gray.size());
        auto candidates = this->fallback_.detect(gray(roi));
        if (candidates.empty())
            candidates = this->large_.detect_mixed(gray(roi));
        for (auto& candidate : candidates)
            for (auto& corner : candidate.corners_px_)
                corner += cv::Point2f(roi.tl());
        candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                        [&](const Detection& tag) {
                                            return !detail::verify_payload<true>(
                                                    gray, tag.corners_px_, tag.id_,
                                                    settings.minimum_ring_matches_,
                                                    settings.minimum_decode_contrast_, 0);
                                        }),
                         candidates.end());
        return candidates;
    }

    void verify(const cv::Mat& gray, cv::Rect region, std::vector<Detection>& tags) {
        const cv::Mat patch = gray(region);
        auto found = patch_.detect(patch);
        if (found.empty())
            found = local_.detect(patch);
        if (!found.empty()) {
            merge(tags, std::move(found), region.tl());
            return;
        }
        if (region.area() > settings_.maximum_upsample_area_px_)
            return;
        if (std::max(region.width, region.height) < 16384) {
            detail::enlarge_patch_cubic(patch, work_, cubic_rows_);
            // Upsampled pixels cannot contribute independent local payload evidence.
            found = local_.detect(work_, false);
            if (found.empty() && std::max(region.width, region.height) <= 80)
                found = native_.detect(work_);
            if (!found.empty()) {
                merge(tags, std::move(found), region.tl(), 2);
                return;
            }
        }
        double low, high;
        cv::minMaxLoc(patch, &low, &high);
        if (high - low >= 10 && high - low < 145) {
            const double scale = 220 / (high - low);
            patch.convertTo(work_, CV_8U, scale, 16 - low * scale);
            merge(tags, patch_.detect(work_), region.tl());
        }
    }

    VideoConfig settings_;
    Detector ink_components_{[&] {
        auto config = component_config(settings_);
        config.components_->threshold_ = settings_.midtone_boundary_.luma_threshold_;
        return config;
    }()};
    detail::NoisyPrintSearch noisy_print_{this->settings_.noisy_print_};
    detail::MotionRegions motion_;
    Detector full_, patch_, local_, wide_, native_;
    Detector border_components_{[this] {
        auto config = component_config(settings_);
        config.components_->sample_stride_px_ = 3;
        return config;
    }()};
    cv::Mat islands_, isolated_;
    detail::KalibrGridSearch kalibr_{settings_.kalibr_};
    detail::IsolatedInk isolated_ink_;
    detail::NoisyInk noisy_ink_;
    detail::UniformRaster raster_;
    detail::EnteringStripSearch entry_;
    detail::BorderSearch border_, pyramid_border_;
    detail::BorderSearch ink_{settings_.midtone_boundary_, Config{}.maximum_corrected_bits_};
    detail::BorderSearch dark_ink_{[&] {
                                       auto config = settings_.midtone_boundary_;
                                       config.luma_threshold_ = settings_.secondary_ink_threshold_;
                                       return config;
                                   }(),
                                   Config{}.maximum_corrected_bits_};
    detail::DenseBoardSearch board_;
    detail::LargeBoardSearch large_;
    detail::AutomaticSearch fallback_{[] {
        Config config;
        config.automatic_.emplace();
        return config;
    }()};
    cv::Mat work_, edge_small_;
    std::vector<std::int32_t> cubic_rows_;
    std::vector<Detection> previous_;
    cv::Size size_;
    bool previously_smooth_ = false;
    bool previous_two_cell_field_ = false;
    int frame_ = 0;
    int board_frames_left_ = 0;
};

VideoDetector::VideoDetector(const VideoConfig& config) : impl_(std::make_unique<Impl>(config)) {}
VideoDetector::~VideoDetector() = default;
void VideoDetector::reset() noexcept {
    impl_->reset();
}
std::vector<Detection> VideoDetector::detect(const cv::Mat& gray) {
    detail::validate_image(gray);
    try {
        auto tags = impl_->detect(gray);
        impl_->refine_published_corners(gray, tags);
        return tags;
    } catch (...) {
        reset();
        throw;
    }
}

}  // namespace aruco3cuda::apriltag_small_cpu
