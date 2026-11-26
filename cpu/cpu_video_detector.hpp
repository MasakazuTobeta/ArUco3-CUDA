// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CPU_VIDEO_DETECTOR_HPP
#define ARUCO3CUDA_CPU_VIDEO_DETECTOR_HPP

#include "cpu_detector.hpp"

namespace aruco3cuda::cpu {

/// Controls the experimental history-assisted Gray8 sequence detector.
/// Ownership: owns configuration values only.
/// Synchronization: value type with no synchronization point.
/// Example input: defaults and a 1280x720 frame.
/// Example output: a full refresh at most 30 frames after the last full search.
struct CpuVideoOptions {
    int refresh_interval_ = 30;  ///< Maximum spacing between full searches.
    int tile_side_px_ = 32;      ///< Target size of a change proposal tile.
    int change_threshold_ = 16;  ///< Minimum absolute change in one sparse sample.
    int padding_px_ = 40;        ///< Context around changes and previously detected markers.
    std::size_t max_rois_ = 16;  ///< Jump to a full search above this many disjoint regions.
    double max_region_fraction_ = 0.35;  ///< Jump to a full search above this area fraction.
};

/// A frame's detections with search diagnostics.
/// Ownership: owns the underlying detection arrays.
/// Synchronization: host-only values with no synchronization point.
/// Example input: the first frame after reset.
/// Example output: full_frame_ is true and roi_count_ is zero.
struct CpuVideoResult {
    CpuResult detections_;
    bool full_frame_ = false;
    std::size_t roi_count_ = 0;
};

/// Experimental CPU detector using history only to propose current-frame search regions.
/// Ownership: owns both detectors, sparse image samples and the previous marker geometry.
/// Synchronization: synchronous host operations. Serialize calls to each instance.
/// Example input: a fixed-size sequence of CV_8UC1 video frames.
/// Example output: IDs and ordered corners in full-image coordinates.
class CpuVideoDetector {
public:
    /// Initialize both search paths and discard history on success.
    /// @param dictionary_name Name of the predefined OpenCV dictionary.
    /// @param config Base configuration used for full and cropped searches.
    /// @param out_message Optional failure reason.
    /// @param options Temporal search bounds.
    /// @return kOk on success, or a detector/option validation error.
    /// Example input: DICT_ARUCO_MIP_36h12 and a default DetectorConfig.
    /// Example output: the next frame is ready for full search.
    Status initialize(const std::string& dictionary_name, const DetectorConfig& config,
                      std::string* out_message = nullptr,
                      const CpuVideoOptions& options = CpuVideoOptions{});

    /// Detect markers in current-frame pixels using full or regional search.
    /// @param image Nonempty two-dimensional CV_8UC1 image, borrowed for the call.
    /// @param out Nonnull result destination, written on success.
    /// @param out_message Optional failure reason.
    /// @return kOk on success or a detector/validation failure. Failed frames clear history.
    /// Example input: a strided image with four visible markers.
    /// Example output: IDs and corresponding ordered corners in global coordinates.
    Status detect(const cv::Mat& image, CpuVideoResult* out, std::string* out_message = nullptr);

    /// Discard frame history while retaining detector configuration.
    /// Example input: camera switches to a different stream.
    /// Example output: the next valid frame uses full-frame detection.
    void reset() noexcept;

private:
    CpuDetector full_;
    CpuDetector regional_;
    CpuVideoOptions options_;
    cv::Size previous_size_;
    cv::Mat previous_samples_;
    cv::Mat current_samples_;
    std::vector<std::vector<cv::Point2f>> previous_corners_;
    std::vector<int> previous_ids_;
    std::vector<int> current_ids_;
    int frames_since_full_ = 0;
    bool initialized_ = false;
};

}  // namespace aruco3cuda::cpu

#endif  // ARUCO3CUDA_CPU_VIDEO_DETECTOR_HPP
