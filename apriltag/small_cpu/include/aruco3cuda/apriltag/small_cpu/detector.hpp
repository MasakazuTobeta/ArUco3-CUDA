// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace aruco3cuda::apriltag_small_cpu {

/// Candidate controls for coarse localization with source-resolution verification.
struct CoarseConfig {
    int sample_stride_px_ = 3;
    int dim_sample_floor_ = 175;
    int dim_sample_percentage_ = 60;
    int tile_side_samples_ = 13;
    int tile_minimum_contrast_ = 16;
    int minimum_quad_side_samples_ = 4;
    int minimum_ring_matches_ = 22;
    float minimum_decode_contrast_ = 10.0F;
    int minimum_retry_side_px_ = 40;
    int grid_radius_px_ = 6;
};

/// Exposure controls for the connected-component threshold.
struct ComponentExposureConfig {
    int minimum_low_percentile_ = 66;
    int highlight_reference_ = 180;
    int adjustment_percent_ = 30;
    int maximum_adjustment_ = 39;
};

/// Current-image evidence for reconsidering a rejected fit on a sampled component.
struct LocalPayloadFitConfig {
    float minimum_side_px_ = 24.F;
    float minimum_displacement_px_ = 1.F;
    /// Set to zero to disable. A printed tag spans eight cells across.
    float maximum_displacement_cells_ = 1.28F;
};

/// Candidate settings for sampled luminance masks.
struct ComponentConfig {
    int sample_stride_px_ = 2;
    int threshold_ = 106;
    int minimum_quad_side_samples_ = 4;
    int maximum_quad_side_samples_ = 600;
    double maximum_aspect_ratio_ = 2.1;
    double minimum_fill_ratio_ = 0.22;
    double maximum_fill_ratio_ = 0.93;
    double hull_tolerance_ratio_ = 0.035;
    double minimum_quad_hull_ratio_ = 0.84;
    int minimum_ring_matches_ = 22;
    int maximum_expansion_side_px_ = 104;
    int grid_radius_px_ = 0;
    /// An empty value keeps the configured threshold fixed.
    std::optional<ComponentExposureConfig> exposure_;
    /// Trace sampled contours instead of component hulls, using the same un-eroded mask.
    bool sampled_contours_ = false;
    /// Alternates ordinary components with this Gray8 contour threshold on video.
    /// Requires explicit stride-2 components with a fixed threshold and recovery.
    std::optional<int> alternate_contour_threshold_;
    /// Odd width of the local min/max window on the sampled grid, from 3 to 31.
    /// Zero retains the ordinary threshold. Local windows exclude exposure and alternating
    /// thresholds.
    int local_midrange_side_samples_ = 0;
    /// Gray levels subtracted from the local midpoint before rounding, from 0 to 127.
    int local_midrange_offset_ = 5;
    /// Try an integer-scan outer-ring edge fit on sharp proposals before ordinary refinement.
    bool sharp_edge_refinement_ = false;
    /// Optional exact local payload check on an already decoded candidate.
    std::optional<LocalPayloadFitConfig> local_payload_fit_;
};

/// Limits for density-guided native search within one rectangular region.
struct DenseRegionConfig {
    bool enabled_ = true;
    int minimum_local_contrast_ = 90;
    int minimum_sample_count_ = 24;
    int minimum_density_per_mille_ = 100;
    int maximum_extent_per_mille_ = 850;
    int padding_px_ = 64;
    std::size_t maximum_roi_pixels_ = 8U * 1024U * 1024U;
    std::size_t maximum_upsample_source_pixels_ = 600000;  ///< Zero disables 2× resizing.
    int upsample_sharpen_percent_ = 8;  ///< Weight of each neighbor. Zero disables sharpening.
    int minimum_verified_tags_ = 6;
    int replace_minimum_tags_ = 32;
};

/// Select among the existing searches using sparse samples of the current frame.
struct AutomaticConfig {
    CoarseConfig coarse_;
    ComponentConfig components_ = [] {
        ComponentConfig config;
        config.exposure_.emplace();
        return config;
    }();

    int native_low_percentile_floor_ = 130;
    int native_high_percentile_ceiling_ = 211;
    int native_max_dark_per_mille_ = 10;
    int component_min_luma_span_ = 45;
    int component_min_rough_per_mille_ = 100;
    int component_max_flat_per_mille_ = 600;
    int component_max_bright_per_mille_ = 600;
    int dense_small_tag_count_ = 4;  ///< Zero disables the supplemental native search.
    float dense_small_tag_side_px_ = 72.0F;
    DenseRegionConfig dense_region_;
};

/// Limits for the optional native search along the four frame boundaries.
struct BorderConfig {
    int luma_threshold_ = 117;
    int maximum_strip_depth_px_ = 115;
    int maximum_clip_depth_px_ = 27;
    int maximum_candidates_per_side_ = 8;
    int maximum_window_pixels_per_side_ = 38000;
    int minimum_dark_run_px_ = 3;
    int window_margin_px_ = 52;
    int minimum_ring_support_ = 14;
    int minimum_known_cells_ = 24;
    int grid_refinement_passes_ = 2;
    /// Restrict the preliminary grid fit to candidates with payload errors.
    bool fit_only_corrected_candidates_ = false;
};

/// Limits for verified local recovery after ordinary discovery on a video frame.
struct RecoveryConfig {
    std::size_t maximum_tracks_ = 64;
    int maximum_translation_px_ = 24;
    int minimum_ring_matches_ = 22;
    float minimum_contrast_ = 10.0F;
};

/// Owns the configuration for the experimental tag36h11 detector.
/// Synchronization: plain values, with no synchronization point.
/// Example input: the default configuration. Example output: use three corrected bits.
struct Config {
    int dark_threshold_ = 85;
    int bright_threshold_ = 128;
    int bright_sample_floor_ = 138;
    int bright_sample_percentage_ = 90;
    int minimum_ring_matches_ = 25;
    float minimum_contrast_ = 15.0F;
    float polygon_tolerance_ratio_ = 0.055F;
    int maximum_corrected_bits_ = 3;
    bool trim_uniform_surround_ = true;
    /// Set to opt into coarse localization. The default uses the native-luma search.
    std::optional<CoarseConfig> coarse_;
    /// Set for sampled connected-component or contour search.
    std::optional<ComponentConfig> components_;
    /// Set for frame-based selection. The three search options are mutually exclusive.
    std::optional<AutomaticConfig> automatic_;
    /// Set to supplement the chosen search with bounded frame-edge proposals.
    std::optional<BorderConfig> border_;
    /// Set for ordered video frames. Call reset before starting a new sequence.
    std::optional<RecoveryConfig> recovery_;
};

/// Synchronous host result, with clockwise corners in OpenCV's canonical order.
/// Ownership: holds values only.
/// Example input: a decoded tag. Example output: ID 17 and its four pixel coordinates.
struct Detection {
    int id_;
    std::array<cv::Point2f, 4> corners_px_;
    int corrected_bits_;
};

namespace detail {
class AutomaticSearch;
class BorderSearch;
class EnteringStripSearch;
class NoisyPrintSearch;
}  // namespace detail

/// Detects tag36h11 markers from scene-dependent luminance masks.
/// Ownership: owns a reusable workspace and only borrows each image during detect.
/// Synchronization: all methods are synchronous. Use one instance from one caller at a time.
/// Example input: a Gray8 frame with a tag. Example output: its ID and four corners.
class Detector {
public:
    /// Initializes the code index and workspace.
    /// @param config Detector settings. Invalid values throw std::invalid_argument.
    /// Example input: Config{}. Example output: a reusable detector.
    explicit Detector(const Config& config = {});
    ~Detector();
    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;

    /// Detects markers in a nonempty Gray8 frame with valid, possibly pitched storage.
    /// @param gray A two-dimensional CV_8UC1 image, including a non-contiguous ROI.
    /// @return Owned output vector in candidate order. Invalid input throws std::invalid_argument.
    /// Example input: one visible tag in a 1280x720 frame.
    /// Example output: one tag with ordered corners relative to the supplied frame.
    std::vector<Detection> detect(const cv::Mat& gray);

    /// Discards recovery history and restarts alternation with ordinary components.
    /// Example input: a detector about to receive a different video.
    /// Example output: the next frame is handled without previous candidates.
    void reset() noexcept;

private:
    friend class detail::AutomaticSearch;
    friend class VideoDetector;
    std::vector<Detection> detect(const cv::Mat& gray, bool native_samples);
    std::optional<Detection> recover_translated(const cv::Mat& gray, const Detection& last,
                                                cv::Point translation) const;
    friend class detail::EnteringStripSearch;
    friend class detail::NoisyPrintSearch;
    std::vector<Detection> decode_sampled_contours(
            const cv::Mat& gray, const std::vector<std::vector<cv::Point>>& contours);
    std::vector<Detection> detect_native_with_threshold(const cv::Mat& gray, int threshold);
    class Impl;
    class Recovery;
    std::unique_ptr<Impl> impl_;
    std::unique_ptr<detail::AutomaticSearch> automatic_;
    std::unique_ptr<detail::BorderSearch> border_;
    std::unique_ptr<Recovery> recovery_;
    std::optional<ComponentConfig> alternate_components_;
    cv::Size alternate_size_;
    bool alternate_frame_ = false;
};

}  // namespace aruco3cuda::apriltag_small_cpu
