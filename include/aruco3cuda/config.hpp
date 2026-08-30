// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CONFIG_HPP
#define ARUCO3CUDA_CONFIG_HPP

#include <string>

#include "aruco3cuda/status.hpp"

namespace aruco3cuda {

/// Maximum number of windows swept by adaptive thresholding.
///
/// The window count is (max - min) / step + 1. The cap exists because a binarized
/// image is kept for every step, so a growing count makes the workspace expand
/// without bound.
inline constexpr int kMaxAdaptiveThresholdWindows = 16;

/// Method used for subpixel refinement of the corners.
enum class CornerRefineMethod : int {
    kNone = 0,  ///< No refinement
    kSubpix,    ///< Gradient-based subpixel refinement
};

/// Converts a CornerRefineMethod into its identifier string.
///
/// @param method Value to convert. Never returns nullptr, even for a value outside the
///               enumeration.
/// @return A string with static storage duration.
///
/// Ownership: the return value points into static storage. The caller neither frees
///            nor modifies it.
/// Synchronization: host only, with no synchronization point.
///
/// Example input: CornerRefineMethod::kSubpix
/// Example output: "kSubpix"
const char* to_string(CornerRefineMethod method);

/// Detection settings.
///
/// The defaults follow `cv::aruco::DetectorParameters` of OpenCV 4.x, except for the
/// two items tied to the ArUco3 detection strategy, whose defaults are changed to suit
/// the evaluation goals of this project. Use opencv_defaults() when the OpenCV defaults
/// are needed exactly.
///
/// Ownership:
///   Holds values only and references no external resource. It may be copied and kept.
struct DetectorConfig {
    // --- Adaptive thresholding ---
    /// Side length of the adaptive thresholding window, in pixels. OpenCV rounds an
    /// even value up to the next odd one.
    int adaptive_thresh_win_size_min_px_ = 3;
    int adaptive_thresh_win_size_max_px_ = 23;
    int adaptive_thresh_win_size_step_px_ = 10;
    double adaptive_thresh_constant_ = 7.0;

    // --- Candidate filtering ---
    double min_marker_perimeter_rate_ = 0.03;
    double max_marker_perimeter_rate_ = 4.0;
    double polygonal_approx_accuracy_rate_ = 0.03;
    double min_corner_distance_rate_ = 0.05;
    int min_distance_to_border_px_ = 3;
    double min_marker_distance_rate_ = 0.125;
    /// Minimum distance at which two entries in the same group count as distinct
    /// markers, expressed as a ratio of one cell side.
    double min_group_distance_ = 0.21;
    /// Lower bound on the fraction of component pixels that fall inside the estimated
    /// quadrilateral.
    ///
    /// This measures how much of the component the four corners found by the extreme
    /// point search actually cover. Shapes that spill outside, such as circles,
    /// ellipses, and hexagons, are rejected here. The default comes from measurements
    /// on synthetic shapes: the lowest value among shapes that must pass is 0.875
    /// (a small frame of side 32) and the highest among shapes that must be rejected
    /// is 0.665 (a hexagon), so the default sits between them.
    double min_quad_inlier_ratio_ = 0.80;
    /// Lower bound on the number of component pixels backing each side of the estimated
    /// quadrilateral, as a ratio of the chain code length of that side.
    ///
    /// The inlier ratio alone cannot reject concave shapes such as an L or a cross,
    /// because a side drawn between two extreme points can run outside the component
    /// while the component itself still fits inside the quadrilateral. Checking
    /// separately whether component pixels lie near each side fills that gap. The
    /// default comes from measurements on synthetic shapes: the lowest value among
    /// shapes that must pass is 2.52 and the highest among shapes that must be rejected
    /// is 1.71 (a cross), so the default sits between them.
    double min_edge_support_ratio_ = 2.0;

    // --- Bit reading and matching ---
    int marker_border_bits_ = 1;
    int perspective_remove_pixel_per_cell_ = 4;
    double perspective_remove_ignored_margin_per_cell_ = 0.13;
    double max_erroneous_bits_in_border_rate_ = 0.35;
    double min_otsu_std_dev_ = 5.0;
    double error_correction_rate_ = 0.6;
    /// Threshold above which a cell ratio counts as a bit.
    ///
    /// Border verification treats a cell whose ratio exceeds this value as an error.
    /// Dictionary matching treats a cell that differs from the expected bit by more
    /// than this value as a mismatch. The CPU reference implicitly uses the OpenCV
    /// DetectorParameters default (0.49), so the same value is held explicitly here.
    double valid_bit_threshold_ = 0.49;

    // --- Corner refinement ---
    CornerRefineMethod corner_refine_method_ = CornerRefineMethod::kSubpix;
    int corner_refinement_win_size_px_ = 5;
    /// How many times one cell side the refinement window may span. To keep the window
    /// from reaching into the neighboring cell on small markers, the smaller of
    /// corner_refinement_win_size_px_ and the value derived from this ratio is used.
    /// Only used when ArUco3 is disabled.
    double relative_corner_refinement_win_size_ = 0.3;
    int corner_refinement_max_iterations_ = 30;
    double corner_refinement_min_accuracy_px_ = 0.1;

    // --- ArUco3 detection strategy ---
    bool use_aruco3_detection_ = true;
    int min_side_length_canonical_img_px_ = 32;
    float min_marker_length_ratio_original_img_ = 0.05F;

    // --- CUDA specific ---
    /// Capacity of the candidate buffer. On overflow, kCandidateOverflow is returned
    /// and the results are truncated.
    ///
    /// The CPU path has no such limit, but the GPU allocates its output storage at
    /// initialization time, so a limit is required. Dropping candidates silently would
    /// make it impossible to tell whether a missed detection comes from the
    /// configuration or from the implementation, so truncation is always reported
    /// through Status.
    int max_candidates_ = 4096;
    /// Capacity of the detection buffer. On overflow, kMarkerOverflow is returned and
    /// the results are truncated.
    int max_markers_ = 1024;
    /// Reference resolution the workspace is preallocated for. Input larger than this
    /// causes a reallocation.
    int max_width_px_ = 3840;
    int max_height_px_ = 2160;
    /// Threads per block side for two-dimensional kernels.
    ///
    /// This keeps device-specific tuning out of hard-coded values in the source and
    /// lets it be overridden from the configuration. The default of 16 is a safe
    /// choice on most devices; per-device values are to be determined by measurement.
    int cuda_block_dim_ = 16;

    /// Confirms that the settings are within their valid ranges and do not contradict
    /// each other.
    ///
    /// It can be called before the detector is created. Passing an out-of-range value
    /// straight into a CUDA kernel launch configuration surfaces as an invalid block
    /// count or a negative iteration count, which obscures the fact that the
    /// configuration is the cause.
    ///
    /// @param out_message On failure, receives a reason containing "field=value".
    ///                    May be nullptr. Left unchanged on success.
    /// @return kOk if everything is valid, otherwise kInvalidConfig.
    ///
    /// Ownership: does not retain the memory behind the arguments.
    /// Synchronization: host only, with no synchronization point. Calls no CUDA API.
    ///
    /// Example input: a default-constructed DetectorConfig
    /// Example output: Status::kOk
    /// Example input: adaptive_thresh_win_size_min_px_ = 2
    /// Example output: Status::kInvalidConfig, with out_message containing
    ///                 "adaptive_thresh_win_size_min_px=2"
    Status validate(std::string* out_message = nullptr) const;

    /// Returns a configuration whose ArUco3-related defaults match OpenCV.
    ///
    /// OpenCV defaults to `useAruco3Detection = false` and
    /// `minMarkerLengthRatioOriginalImg = 0.0`. Compatibility checks need to line up
    /// with those defaults.
    ///
    /// @return A configuration matching the OpenCV defaults.
    ///
    /// Ownership: returns a value, owned by the caller.
    /// Synchronization: host only, with no synchronization point.
    ///
    /// Example input: no arguments
    /// Example output: a configuration with use_aruco3_detection_ = false and
    ///                 min_marker_length_ratio_original_img_ = 0.0F
    static DetectorConfig opencv_defaults();
};

}  // namespace aruco3cuda

#endif  // ARUCO3CUDA_CONFIG_HPP
