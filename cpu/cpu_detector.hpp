// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CPU_DETECTOR_HPP
#define ARUCO3CUDA_CPU_DETECTOR_HPP

#include <opencv2/core.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"

namespace aruco3cuda::cpu {

/// Independently selectable CPU experiments. False selects the compatibility implementation.
/// Ownership: holds values only and owns no external resource.
/// Synchronization: plain values; no synchronization point.
/// Example input: all switches false.
/// Example output: the compatibility control is selected.
struct CpuOptions {
    bool reuse_pyramid_ = true;
    bool packed_dictionary_ = true;
    bool exact_dictionary_lookup_ = true;
    bool fused_cells_ = true;
    bool multiwindow_threshold_ = true;
    bool spatial_grouping_ = false;
    bool filtered_contours_ = true;
    bool sparse_pyramid_ = true;
    bool cached_refinement_ = true;
    int sparse_tile_side_px_ = 8;
    double sparse_full_level_ratio_ = 0.5;
    double sparse_max_work_ratio_ = 0.3;
    bool simple_contours_ = false;
    bool component_prefilter_ = false;
    bool dryrun_otsu_ = false;
    bool component_candidates_ = false;
    bool parallel_windows_ = true;
    bool parallel_decode_ = true;
    int min_parallel_candidates_ = 2;
    int threshold_parallel_min_pixels_ = 32768;
    /// Prefix-sum cache budget in bytes, excluding other workspace storage. Zero selects legacy
    /// sizing.
    std::size_t threshold_max_prefix_bytes_ = 2U * 1024U * 1024U;
    double component_edge_band_px_ = 2.0;
    bool profile_stages_ = false;
    /// Opt-in projective refinement for tag36h11 with one border cell. Zero disables it.
    int decoded_grid_radius_px_ = 0;
    /// Sample the original image for tag36h11 payloads and corner fitting.
    bool decode_at_source_resolution_ = false;
    /// With source-resolution decoding, use a 2x2 average on a grid of 2..8 pixels.
    /// Zero uses the configured ArUco3 segmentation resize.
    int candidate_box_stride_px_ = 0;
    /// Add a bright exterior only during candidate search at the frame edge.
    bool search_frame_edges_ = false;
    /// Total original-image area considered by supplemental edge searches in one frame.
    std::size_t frame_edge_max_pixels_ = 720000U;
};

/// Host wall-clock stage times; zero unless profile_stages_ is enabled.
/// Ownership: owns numeric values only.
/// Synchronization: no synchronization point.
/// Example input: detection with profiling enabled.
/// Example output: nonnegative stage durations in milliseconds.
struct CpuStageTimes {
    double preprocess_ms_ = 0.0;
    double threshold_ms_ = 0.0;
    double candidates_ms_ = 0.0;
    double decode_ms_ = 0.0;
    double refine_ms_ = 0.0;
    double trace_ms_ = 0.0;
    double group_ms_ = 0.0;
};

/// Host detections. Corners encode rotation in the same order as OpenCV detectMarkers.
/// Ownership: owns ID and corner vectors and plain timing values.
/// Synchronization: host result; no synchronization point.
/// Example input: an image with one marker.
/// Example output: one ID and one vector containing four corners.
struct CpuResult {
    std::vector<int> ids_;
    std::vector<std::vector<cv::Point2f>> corners_;
    std::size_t rejected_count_ = 0;
    std::size_t candidate_count_ = 0;
    CpuStageTimes stage_times_;
};

/// Experimental CPU detector with no CUDA execution or runtime dependency.
///
/// Ownership: owns its dictionary and reusable buffers. Borrows the input image only for detect.
/// This ownership and synchronization contract applies to all public member functions.
/// Synchronization: all public member functions are synchronous host operations. One instance
/// must not be used concurrently. OpenCV's process-wide thread setting is respected, not changed.
/// Example input: initialize("DICT_ARUCO_MIP_36h12", DetectorConfig{}, &message), then
/// detect(gray,...). Example output: kOk and host marker IDs plus four full-resolution corners per
/// marker.
class CpuDetector {
public:
    CpuDetector();
    ~CpuDetector();
    CpuDetector(const CpuDetector&) = delete;
    CpuDetector& operator=(const CpuDetector&) = delete;
    CpuDetector(CpuDetector&&) noexcept;
    CpuDetector& operator=(CpuDetector&&) noexcept;

    /// Configure a dictionary and independent CPU optimization switches.
    /// @param dictionary_name OpenCV predefined dictionary name, borrowed for the call.
    /// @param config Shared detection settings. CUDA fields are validated by the shared validator
    ///               but do not impose CPU capacity limits or launch behavior.
    /// @param out_message Optional destination for an actionable failure reason.
    /// @param options Independent experiments; all false provides the compatibility control.
    /// @return kOk on success; kInvalidConfig or kUnsupportedDictionary on invalid input.
    /// Example input: default options and a supported dictionary name.
    /// Example output: kOk and a detector ready to accept varying image sizes.
    Status initialize(const std::string& dictionary_name, const DetectorConfig& config,
                      std::string* out_message = nullptr, const CpuOptions& options = CpuOptions{});

    /// Detect markers in a grayscale host image, including non-contiguous ROIs.
    /// @param image Nonempty CV_8UC1 image with at most INT_MAX pixels and sides at most 65536.
    ///              Storage is borrowed for this call only.
    /// @param out Receives all detections on success. Must not be null; unchanged on failure.
    /// @param out_message Optional failure reason.
    /// @return kOk on success, otherwise a validation or initialization status.
    /// Example input: a 1280x720 CV_8UC1 image containing one marker.
    /// Example output: kOk with one ID and one vector of four corners in out.
    Status detect(const cv::Mat& image, CpuResult* out, std::string* out_message = nullptr);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace aruco3cuda::cpu
#endif  // ARUCO3CUDA_CPU_DETECTOR_HPP
