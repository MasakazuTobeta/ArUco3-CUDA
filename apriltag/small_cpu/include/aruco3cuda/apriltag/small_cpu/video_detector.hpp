// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "aruco3cuda/apriltag/small_cpu/detector.hpp"
#include "aruco3cuda/apriltag/small_cpu/read_noise_config.hpp"

namespace aruco3cuda::apriltag_small_cpu {

// The Kalibr AprilGrid layout has two black border cells around a 6 x 6 payload.
// Pitch is the distance between neighbouring centres in printed marker widths.
struct KalibrGridConfig {
    bool enabled_ = true;
    int proposal_dark_level_ = 40;
    int preview_dark_level_ = 76;
    int preview_short_side_ = 256;
    int maximum_component_side_px_ = 300;
    float grid_pitch_ = 1.30F;
    float pitch_tolerance_ = .08F;
    /// Recheck supported sites with a locally fitted native-image grid under distortion.
    bool curved_grid_completion_ = true;
    float minimum_decode_contrast_ = 12.F;
    int maximum_corrected_bits_ = 1;
};

/// Local-contrast proposals for sparse edges on smooth gradients. Geometry is verified in Gray8.
struct SparseFaintConfig {
    /// Short side of the proposal image in pixels. Zero disables sparse proposals.
    int guide_side_px_ = 120;
    /// Use the ordinary guide first above this source-pixel sampling interval.
    int maximum_initial_step_px_ = 6;
    /// Maximum fraction (per thousand) of sampled neighbour differences above the noise floor.
    int maximum_changed_per_mille_ = 50;
    float maximum_box_aspect_ratio_ = 2.2F;
    float minimum_hull_fill_ = .68F;
};

/// Supplemental source-verified search for exposed prints under distributed sensor noise.
struct NoisyPrintConfig {
    bool enabled_ = true;
    int minimum_noisy_patches_per_mille_ = 750;
    int maximum_dark_patches_per_mille_ = 100;
    float minimum_noise_sigma_ = 1.0F;
    int maximum_ink_luma_ = 127;
    int minimum_local_contrast_ = 30;
};

/// Native outer-edge fit for detections of large prints.
struct LargeTagRefinementConfig {
    bool enabled_ = true;
    /// Minimum frame area for the ordinary search. The large-board route uses straight rails.
    std::size_t minimum_frame_pixels_ = 6000000;
    std::size_t maximum_tags_ = 16;
    float minimum_side_px_ = 55.F;
    float minimum_short_to_long_ratio_ = .56F;
    int search_radius_px_ = 9;
    float maximum_corner_motion_px_ = 11.F;
    float maximum_quadratic_px_ = 4.5F;
    float minimum_blur_ratio_ = 1.35F;
    int minimum_ring_matches_ = 23;
    float minimum_decode_contrast_ = 12.F;
};

/// Native corner refinement of detected small prints on wide-angle footage.
struct SmallTagRefinementConfig {
    bool enabled_ = true;
    std::size_t minimum_frame_pixels_ = 6000000;
    float minimum_side_px_ = 10.5F;
    float maximum_side_px_ = 92.F;
    float maximum_relative_residual_ = .075F;
};

/// Native retry around a single changed region in a stationary view.
struct StationaryPatchRetryConfig {
    /// Set to false to use the ordinary changed-region search alone.
    bool enabled_ = true;
    std::size_t minimum_frame_pixels_ = 6000000;
    int minimum_change_side_px_ = 96;
    /// Smaller empty regions get a wider margin to include a whole print outline.
    int small_change_limit_px_ = 256;
    int maximum_change_side_px_ = 768;
    int minimum_ring_matches_ = 25;
    float minimum_decode_contrast_ = 12.F;
};

/// Controls for the single-threaded, ordered-frame tag36h11 search.
struct VideoConfig {
    int full_search_interval_ = 24;
    int quiet_difference_ = 2;
    int noisy_difference_ = 22;
    int changed_region_padding_px_ = 23;
    int maximum_changed_regions_ = 5;
    int maximum_upsample_area_px_ = 30000;
    int full_frame_local_window_ = 0;
    int patch_local_window_ = 3;
    int alternate_contour_threshold_ = 148;
    int full_frame_alignment_side_px_ = 90;
    int smooth_low_percentile_floor_ = 120;
    int smooth_high_percentile_ceiling_ = 214;
    int smooth_minimum_luma_span_ = 35;
    int smooth_max_rough_per_mille_ = 64;
    int smooth_hysteresis_levels_ = 8;
    int bright_field_minimum_ = 190;
    int border_pyramid_reference_side_ = 480;
    int border_tile_contrast_ = 24;
    int isolated_luma_margin_ = 32;
    int isolated_dark_per_mille_ = 4;
    int maximum_isolated_regions_ = 12;
    int kalibr_field_low_ = 40;
    int kalibr_field_high_ = 165;
    int kalibr_light_level_ = 130;
    int kalibr_minimum_dark_per_mille_ = 400;
    int kalibr_minimum_light_per_mille_ = 18;
    int kalibr_minimum_rough_per_mille_ = 80;
    int isolated_guide_side_px_ = 180;
    int isolated_fine_per_mille_ = 12;
    int isolated_residual_contrast_ = 12;
    int faint_background_difference_ = 3;
    int faint_guide_side_px_ = 180;
    int faint_maximum_scene_span_ = 128;
    int faint_background_window_ = 15;
    int faint_decode_contrast_ = 8;
    BorderConfig boundary_ = [] {
        BorderConfig config;
        config.grid_refinement_passes_ = 1;
        config.fit_only_corrected_candidates_ = true;
        return config;
    }();
    BorderConfig midtone_boundary_ = [] {
        BorderConfig config;
        config.luma_threshold_ = 75;
        config.maximum_strip_depth_px_ = 150;
        config.window_margin_px_ = 96;
        config.maximum_window_pixels_per_side_ = 180000;
        config.maximum_candidates_per_side_ = 24;
        config.grid_refinement_passes_ = 1;
        return config;
    }();
    int secondary_ink_threshold_ = 44;
    int midtone_boundary_reference_side_ = 400;
    int midtone_boundary_ceiling_ = 170;
    int border_surround_minimum_ = 150;
    int border_surround_maximum_span_ = 56;
    int pan_strip_width_px_ = 288;
    int pan_strip_threshold_ = 120;
    /// Minimum edge contrast for supplemental local thresholds. Zero disables this pass.
    int pan_photometric_minimum_contrast_ = 28;
    int gradient_max_rough_per_mille_ = 24;
    int uniform_field_maximum_span_ = 3;
    std::size_t dense_count_ = 6;
    int large_board_dark_limit_ = 24;
    int large_board_bright_floor_ = 200;
    int large_board_midtones_per_mille_ = 150;
    int large_board_max_rough_per_mille_ = 140;
    int board_background_floor_ = 80;
    int board_dark_level_ = 60;
    int board_light_level_ = 170;
    int board_region_padding_px_ = 96;
    double compact_board_area_ratio_ = 0.6;
    int dense_maximum_side_px_ = 80;
    std::size_t dense_seed_count_ = 3;
    int board_search_hold_frames_ = 7;
    KalibrGridConfig kalibr_{};
    /// Near-frame margin as a fraction of one printed side, with an eight-pixel minimum.
    /// Zero disables this native edge fit on fully visible midtone tags.
    float midtone_edge_refinement_margin_ratio_ = .20F;
    /// Minimum nearest-neighbor distance in tag widths for a sparse native board probe.
    /// Zero preserves the independent automatic and dense searches.
    float dispersed_seed_spacing_sides_ = 6.F;
    /// Minimum sampled density (per thousand) of moderate intensity steps for adaptive proposals.
    int faint_minimum_moderate_per_mille_ = 4;
    /// Fit diffuse source-pixel transitions on structured, moderate-contrast backgrounds.
    bool refine_diffuse_edges_ = true;
    SparseFaintConfig sparse_faint_{};
    /// Density of two-axis native intensity changes for averaging the local-mean guide.
    /// Zero disables this averaging.
    int adaptive_average_minimum_noise_per_mille_ = 40;
    /// Maximum additional source-verified corner displacement in printed cells (0 disables it).
    float maximum_diffuse_reseed_cells_ = .70F;
    ReadNoiseConfig read_noise_{};
    NoisyPrintConfig noisy_print_{};
    /// Recheck rejected component fits in the ordinary frame and patch searches.
    LocalPayloadFitConfig local_payload_fit_{};
    LargeTagRefinementConfig large_tag_refinement_{};
    StationaryPatchRetryConfig stationary_patch_retry_{};
    SmallTagRefinementConfig small_tag_refinement_{};
};

/// Detects tags in a Gray8 video on one calling thread. Each result is decoded
/// from the current frame. The detector owns its history and borrows no input
/// storage between calls. Use a separate instance for each concurrent stream.
class VideoDetector {
public:
    explicit VideoDetector(const VideoConfig& config = {});
    ~VideoDetector();
    VideoDetector(const VideoDetector&) = delete;
    VideoDetector& operator=(const VideoDetector&) = delete;

    /// Accepts the same nonempty, possibly pitched Gray8 inputs as Detector.
    /// A size change starts a new sequence. Invalid inputs throw std::invalid_argument.
    std::vector<Detection> detect(const cv::Mat& gray);
    /// Discards the previous stream, including the full-frame search phase.
    void reset() noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace aruco3cuda::apriltag_small_cpu
