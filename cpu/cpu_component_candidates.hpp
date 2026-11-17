// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CPU_COMPONENT_CANDIDATES_HPP
#define ARUCO3CUDA_CPU_COMPONENT_CANDIDATES_HPP

#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>

#include "aruco3cuda/config.hpp"

namespace aruco3cuda::cpu::detail {

/// Reusable host storage for the non-compatible component/extrema experiment.
/// Ownership: owns labels, statistics and the packed foreground points.
/// Synchronization: synchronous host operations; one instance per detector.
/// Example input: a binary image with a square frame.
/// Example output: one outer quadrilateral, without its inner contour as an alternate.
class ComponentCandidates {
public:
    /// Append quadrilateral candidates estimated directly from connected foreground regions.
    /// @param binary Eight-bit binary image, borrowed during the call.
    /// @param config Shared candidate filtering settings, borrowed during the call.
    /// @param edge_band_px Pixel distance counted as supporting a quad edge.
    /// @param candidates Receives four corner points per accepted component, appended.
    /// @param contours Receives a vector whose length is the estimated quad chain perimeter.
    /// @return Nothing. OpenCV allocation/processing errors propagate to CpuDetector's boundary.
    /// Ownership: outputs own their values; inputs are not retained.
    /// Synchronization: host only, synchronous.
    /// Example input: a 128-pixel-perimeter square component.
    /// Example output: one quad and a corresponding 128-element perimeter placeholder.
    void append(const cv::Mat& binary, const DetectorConfig& config, double edge_band_px,
                std::vector<std::vector<cv::Point2f>>& candidates,
                std::vector<std::vector<cv::Point>>& contours);

private:
    cv::Mat labels_;
    cv::Mat statistics_;
    cv::Mat centroids_;
    std::vector<int> active_labels_;
    std::vector<std::size_t> offsets_;
    std::vector<std::size_t> cursors_;
    std::vector<cv::Point> points_;
};

}  // namespace aruco3cuda::cpu::detail
#endif  // ARUCO3CUDA_CPU_COMPONENT_CANDIDATES_HPP
