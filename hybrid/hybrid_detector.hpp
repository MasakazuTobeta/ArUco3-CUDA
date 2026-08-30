// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_HYBRID_HYBRID_DETECTOR_HPP
#define ARUCO3CUDA_HYBRID_HYBRID_DETECTOR_HPP

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"
#include "aruco3cuda/workspace.hpp"

namespace aruco3cuda::hybrid {

/// A single detected marker.
///
/// Ownership: holds values only and references no external resource.
/// Synchronization: a plain set of values, with no synchronization point.
///
/// Example input: a marker with ID 42 drawn into a synthetic image
/// Example output: id_ = 42, and corners_ holding the four corners in
/// full-resolution coordinates
struct HybridDetection {
    int id_ = -1;
    /// The four corners in the order x0, y0, ... y3, in the coordinate system
    /// of the input image. As in OpenCV's detectMarkers, the order is the one
    /// left after undoing the rotation reported by the dictionary match.
    std::array<double, 8> corners_{};
    /// The rotation reported by the dictionary match. 0 through 3.
    int rotation_ = 0;
};

/// The detection result for one frame.
///
/// Ownership: holds values only and references no external resource.
/// Synchronization: a plain set of values, with no synchronization point.
///
/// Example input: a synthetic image with four markers
/// Example output: detections_ has four elements, and candidate_count_ is the
/// number of candidates obtained from binarization
struct HybridResult {
    std::vector<HybridDetection> detections_;
    /// Total number of quad candidates obtained from the binarized images,
    /// summed over the windows.
    std::size_t candidate_count_ = 0;
    /// Processing time on the GPU side, in ms. Includes synchronization.
    double gpu_ms_ = 0.0;
    /// Processing time on the CPU side, in ms.
    double cpu_ms_ = 0.0;
};

/// Detector that runs preprocessing and binarization on the GPU and candidate
/// extraction and decoding on the CPU.
///
/// Position:
///   This is option C of the
///   [detector pipeline design](../docs/design/detector-pipeline.md). It is the
///   baseline used while the GPU stages are replaced one at a time, and it is
///   also the fallback if option A does not work out as expected. It is kept
///   even after the fully GPU-resident route is complete.
///
/// Ownership:
///   This class owns the workspace and the temporary buffers on the OpenCV
///   side. The caller owns the memory of the input image.
///   **This ownership applies to all public member functions.**
///
/// Synchronization:
///   detect() internally waits for the GPU to finish, because the binarized
///   images have to come back to the host. This synchronization is inherent to
///   the structure of option C.
///   **This synchronization applies to all public member functions.**
///   A single instance must not be used from several threads at the same time.
///
/// Example input:
///   HybridDetector detector;
///   detector.initialize(config, "DICT_ARUCO_MIP_36h12", 1920, 1080, &message);
///   detector.detect(view, &result, &message);
/// Example output:
///   result.detections_ holds the IDs and corners of the detected markers
class HybridDetector {
public:
    HybridDetector();
    ~HybridDetector();

    HybridDetector(const HybridDetector&) = delete;
    HybridDetector& operator=(const HybridDetector&) = delete;
    HybridDetector(HybridDetector&&) noexcept;
    HybridDetector& operator=(HybridDetector&&) noexcept;

    /// Initializes the detector.
    ///
    /// Allocates the workspace capacity and loads the dictionary. The largest
    /// resolution you expect is given here so that no allocation happens per
    /// frame.
    ///
    /// @param config Detector configuration. It is passed through validate().
    /// @param dictionary_name Name of a predefined dictionary.
    /// @param max_width_px Largest width expected. At least 1.
    /// @param max_height_px Largest height expected. At least 1.
    /// @param out_message Receives the reason on failure. May be nullptr.
    /// @return kOk, or kInvalidConfig, kUnsupportedDictionary or kCudaError.
    ///
    /// Example input: the default configuration, "DICT_ARUCO_MIP_36h12", 1920,
    /// 1080
    /// Example output: Status::kOk
    Status initialize(const DetectorConfig& config, const std::string& dictionary_name,
                      int max_width_px, int max_height_px, std::string* out_message = nullptr);

    /// Detects the markers in one frame.
    ///
    /// @param image Input image. Must be device-resident.
    /// @param out Receives the result on success. Must not be nullptr.
    /// @param out_message Receives the reason on failure. May be nullptr.
    /// @return kOk, or kNotInitialized, kInvalidImage or kCudaError.
    ///
    /// Example input: a 1280x720 grayscale image on the device
    /// Example output: Status::kOk. out->detections_ holds the detections
    Status detect(const ImageViewU8& image, HybridResult* out,
                  std::string* out_message = nullptr);

    /// Returns how the workspace is being used.
    ///
    /// @return A reference to the statistics. Valid until the next operation.
    ///
    /// Example input: after processing 100 frames
    /// Example output: allocation_count_ is still 1
    const WorkspaceStatistics& workspace_statistics() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace aruco3cuda::hybrid

#endif  // ARUCO3CUDA_HYBRID_HYBRID_DETECTOR_HPP
