// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CPU_CONTOURS_HPP
#define ARUCO3CUDA_CPU_CONTOURS_HPP

#include <opencv2/core.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "aruco3cuda/config.hpp"

namespace aruco3cuda::cpu::detail {

/// Reusable single-window contour tracing and quadrilateral filtering.
/// Ownership: owns scratch pixels and vectors; borrows inputs only during append.
/// Synchronization: synchronous; use a distinct workspace for each concurrent window.
/// Example input: a threshold image containing a marker and many tiny noise components.
/// Example output: the same ordered quadrilateral paths without allocating tiny contour outputs.
class ContourWorkspace {
public:
    /// Append all eligible quadrilaterals in OpenCV RETR_LIST order.
    /// @param binary Current frame's nonempty CV_8UC1 threshold image; nonzero is foreground.
    /// @param config Validated detector configuration.
    /// @param candidates Receives four corners per accepted quadrilateral, appended.
    /// @param contours Receives the corresponding full point paths, appended.
    /// @return No value. Invalid inputs or OpenCV failures throw cv::Exception.
    /// Ownership: caller owns outputs; scratch allocations are reused without previous-frame
    /// pixels. Synchronization: synchronous, with no internal worker dispatch. Example input: three
    /// separated marker borders. Example output: ordered candidates and contour paths matching the
    /// serial OpenCV finder.
    void append(const cv::Mat& binary, const DetectorConfig& config,
                std::vector<std::vector<cv::Point2f>>& candidates,
                std::vector<std::vector<cv::Point>>& contours);

private:
    bool trace(int row, int column, bool external);
    void append_quad(const DetectorConfig& config, int longest, std::size_t minimum,
                     std::size_t maximum, std::vector<std::vector<cv::Point2f>>& candidates,
                     std::vector<std::vector<cv::Point>>& contours);
    cv::Mat padded_;
    std::array<std::ptrdiff_t, 16> offsets_{};
    std::vector<cv::Point> trace_;
    std::vector<cv::Point> approximation_;
};

}  // namespace aruco3cuda::cpu::detail
#endif
