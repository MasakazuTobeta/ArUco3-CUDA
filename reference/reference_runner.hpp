// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_REFERENCE_RUNNER_HPP
#define ARUCO3CUDA_REFERENCE_RUNNER_HPP

#include <array>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace aruco3cuda::reference {

/// Detection settings for the CPU baseline implementation.
///
/// The defaults follow OpenCV's DetectorParameters. Only the items that relate
/// to the ArUco3 detection strategy use this project's values as the evaluation
/// defaults.
struct ReferenceConfig {
    std::string dictionary_name_ = "DICT_ARUCO_MIP_36h12";

    /// Side length of the adaptive thresholding window, in pixels.
    int adaptive_thresh_win_size_min_px_ = 3;
    int adaptive_thresh_win_size_max_px_ = 23;
    int adaptive_thresh_win_size_step_px_ = 10;
    double adaptive_thresh_constant_ = 7.0;
    double min_marker_perimeter_rate_ = 0.03;
    double max_marker_perimeter_rate_ = 4.0;
    double polygonal_approx_accuracy_rate_ = 0.03;
    double min_corner_distance_rate_ = 0.05;
    int min_distance_to_border_px_ = 3;
    double min_marker_distance_rate_ = 0.125;
    int marker_border_bits_ = 1;
    int perspective_remove_pixel_per_cell_ = 4;
    double perspective_remove_ignored_margin_per_cell_ = 0.13;
    double max_erroneous_bits_in_border_rate_ = 0.35;
    double min_otsu_std_dev_ = 5.0;
    double error_correction_rate_ = 0.6;

    /// Whether to run subpixel refinement of the four corners. When ArUco3 is
    /// enabled OpenCV refines them regardless of this setting.
    bool use_corner_subpix_refinement_ = false;
    int corner_refinement_win_size_px_ = 5;
    double relative_corner_refinement_win_size_ = 0.3;
    int corner_refinement_max_iterations_ = 30;
    double corner_refinement_min_accuracy_px_ = 0.1;

    bool use_aruco3_detection_ = true;
    int min_side_length_canonical_img_px_ = 32;
    float min_marker_length_ratio_original_img_ = 0.05F;

    /// OpenCV thread count. Fixed to 1 by default so measurements are
    /// reproducible. Specifying 0 leaves the OpenCV default in place.
    int num_threads_ = 1;

    /// Omit the execution time from the JSON. Used for byte-level comparison
    /// against a golden file. Timings vary from run to run, so set this to true
    /// whenever deterministic output is required.
    bool omit_timing_ = false;
};

/// Detection result for a single image.
struct ReferenceDetection {
    int id_ = -1;
    /// Holds the four corners in the order x0, y0, x1, y1, x2, y2, x3, y3.
    /// The ordering is exactly the one returned by OpenCV's detectMarkers.
    std::array<double, 8> corners_{};
};

/// Run result for a single image.
struct ReferenceResult {
    std::string image_path_;
    std::string image_sha256_;
    int width_px_ = 0;
    int height_px_ = 0;

    /// Effective ArUco3 downscale ratio. The evaluation plan requires it to be
    /// recorded as part of the measurement conditions.
    double fxfy_effective_ = 1.0;
    int segmentation_width_px_ = 0;
    int segmentation_height_px_ = 0;

    std::vector<ReferenceDetection> detections_;
    std::size_t rejected_count_ = 0;
    double detect_ms_ = 0.0;
};

/// Execution environment and OpenCV version information.
struct ReferenceEnvironment {
    std::string opencv_version_;
    int opencv_threads_ = 0;
    /// Raw JSON of the in-image provenance. Empty when it was not obtained.
    std::string opencv_provenance_json_;
};

/// Verify that the detection settings lie in a valid range.
///
/// Passing out-of-range values straight to OpenCV raises a cv::Exception, which
/// escapes to the caller and breaks the contract of reporting failure through
/// the bool return value and out_error. detect_image performs this validation
/// first.
///
/// @param config The settings to validate. Only referenced, never retained.
/// @param out_error Receives a reason containing "item_name=value" on failure.
///                  Must not be nullptr. The storage is owned by the caller.
/// @return true if every item is valid.
///
/// Ownership: does not retain the storage of any argument.
/// Synchronization: host only, with no synchronization point. Calls neither
/// OpenCV nor CUDA.
///
/// Example input: a config with adaptive_thresh_win_size_min_ = 2
/// Example output: false, with out_error containing "adaptive_thresh_win_size_min=2"
bool validate_config(const ReferenceConfig& config, std::string* out_error);

/// Check whether a name resolves to a predefined OpenCV dictionary.
///
/// @param name The name to look up. Only referenced, never retained.
/// @return true if it can be resolved.
///
/// Ownership: does not retain the storage of any argument.
/// Synchronization: host only, with no synchronization point.
///
/// Example input: "DICT_ARUCO_MIP_36h12"
/// Example output: true
bool is_known_dictionary(const std::string& name);

/// Return the list of supported dictionary names.
///
/// @return The names sorted in ascending order.
///
/// Ownership: the return value is a value and is owned by the caller.
/// Synchronization: host only, with no synchronization point.
///
/// Example input: no arguments
/// Example output: {"DICT_4X4_100", "DICT_4X4_1000", ..., "DICT_ARUCO_MIP_36h12"}
std::vector<std::string> known_dictionary_names();

/// Detect markers in a single image.
///
/// @param image_path Image file, loaded as 8-bit grayscale.
/// @param config Detection settings.
/// @param out_result Receives the result on success.
/// @param out_error Receives the reason on failure.
/// @return true on success.
///
/// Notes:
///   The detections are sorted into a stable order by id, then by the x and y
///   of the first corner. The order OpenCV returns depends on the order in
///   which candidates were extracted, which is awkward for comparison.
bool detect_image(const std::string& image_path, const ReferenceConfig& config,
                  ReferenceResult* out_result, std::string* out_error);

/// Prepare the image and the dictionary once, then repeat only the detection.
///
/// `detect_image` reads the file and computes its checksum on every call.
/// Repeating that during a measurement makes the load time larger than the
/// detection time, so it is no longer clear what is being compared. For the
/// 1280x720 PNGs of the synthetic corpus, loading accounts for 60 to 80 percent
/// of the measured interval. This type exists to move loading into
/// initialization so that the measured interval covers detection alone.
///
/// Ownership: owns the loaded image and the OpenCV detector.
/// Synchronization: none. A single instance must not be used from several
///                  threads at the same time. `detect` does not modify the
///                  internal state, but it is not const because the thread
///                  safety of the OpenCV detector cannot be assumed.
///
/// Example input: initialize("scene.png", default settings) followed by 200 calls to detect
/// Example output: the same detection result every time, with the file read only once
class ReferenceDetector {
public:
    ReferenceDetector();
    ~ReferenceDetector();
    ReferenceDetector(const ReferenceDetector&) = delete;
    ReferenceDetector& operator=(const ReferenceDetector&) = delete;
    ReferenceDetector(ReferenceDetector&&) noexcept;
    ReferenceDetector& operator=(ReferenceDetector&&) noexcept;

    /// Load the image and prepare the dictionary and the detector.
    ///
    /// @param image_path Image file, loaded as 8-bit grayscale.
    /// @param config Detection settings.
    /// @param out_error Receives the reason on failure. Must not be nullptr.
    /// @return true on success.
    ///
    /// Ownership: owns the loaded image. Does not retain the arguments.
    /// Synchronization: none. Performs file I/O.
    ///
    /// Example input: a 1280x720 PNG and the default settings
    /// Example output: true, after which metadata() can return the dimensions and the checksum
    bool initialize(const std::string& image_path, const ReferenceConfig& config,
                    std::string* out_error);

    /// Detect markers in the already loaded image.
    ///
    /// @param out_result Receives the result on success. Must not be nullptr.
    /// @param out_error Receives the reason on failure. Must not be nullptr.
    /// @return true on success. Returns false when called before initialize.
    ///
    /// Notes:
    ///   The sorting rule for the detections is the same as in `detect_image`.
    ///
    /// Ownership: does not retain the arguments.
    /// Synchronization: none. Performs no file I/O.
    ///
    /// Example input: an initialized instance
    /// Example output: true, with out_result holding the same result as detect_image
    bool detect(ReferenceResult* out_result, std::string* out_error);

    /// Return the information about the image. Contains no detections.
    ///
    /// @return A result holding the path, the checksum, the dimensions and the
    ///         effective fxfy value.
    ///
    /// Ownership: the referent of the return value is owned by this instance.
    /// Synchronization: none.
    ///
    /// Example input: an initialized instance
    /// Example output: a result with image_sha256_ and width_px_ filled in
    const ReferenceResult& metadata() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/// Collect information about the execution environment.
///
/// @param config Detection settings. Only used for the OpenCV thread count.
/// @return The collected environment information. Items that could not be
///         obtained are left as empty strings.
///
/// Ownership: the return value is a value and references no internal resource.
///            Does not retain the arguments.
/// Synchronization: host only, with no synchronization point.
///
/// Side effects: when config.num_threads_ is 1 or more, calls
///               cv::setNumThreads() and thereby changes the thread count of
///               all subsequent OpenCV processing. This is a deliberate side
///               effect that keeps the measurement conditions fixed. Nothing is
///               changed when 0 is specified.
///
/// Example input: a ReferenceConfig with num_threads_ = 1
/// Example output: opencv_version_ = "4.14.0", opencv_threads_ = 1
ReferenceEnvironment collect_environment(const ReferenceConfig& config);

/// Write the results out as JSON.
void write_results_json(std::ostream& out, const ReferenceConfig& config,
                        const ReferenceEnvironment& environment,
                        const std::vector<ReferenceResult>& results);

}  // namespace aruco3cuda::reference

#endif  // ARUCO3CUDA_REFERENCE_RUNNER_HPP
