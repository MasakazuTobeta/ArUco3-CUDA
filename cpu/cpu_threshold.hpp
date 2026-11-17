// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CPU_THRESHOLD_HPP
#define ARUCO3CUDA_CPU_THRESHOLD_HPP

#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>

namespace aruco3cuda::cpu::detail {

/// Shared integral-image workspace for a sweep of adaptive threshold windows.
///
/// Ownership: owns its scratch images; input images are borrowed for one call.
/// Synchronization: all public member functions are synchronous host operations.
/// A workspace must not be used concurrently from several threads.
/// Example input: a workspace followed by apply(gray, {3, 13, 23}, 7.0, &outputs).
/// Example output: three threshold planes, with scratch reused on the next call.
class ThresholdWorkspace {
public:
    /// Compute the MEAN_C, BINARY_INV adaptive thresholds with max value 255.
    ///
    /// @param gray Nonempty CV_8UC1 image, including a strided image or ROI.
    /// @param window_sizes Window sides in pixels in [3, 4097]. Even sides
    ///                     are rounded up to the next odd side; 4097 accepts
    ///                     the normalized form of configured side 4096.
    /// @param constant Finite threshold offset in [-255, 255].
    /// @param outputs Receives one independent CV_8UC1 image per window. Must
    ///                not be null. Existing planes must not overlap one another;
    ///                a plane may alias gray. Storage is reused when possible.
    /// @param parallel_min_pixels Minimum stripe pixel count for parallel row
    ///                            groups when OpenCV has multiple threads. Must
    ///                            be nonnegative; zero always permits parallelism.
    /// @param max_prefix_bytes Maximum integral-owner payload, including halo
    ///                         and its zero row/column. Zero selects the legacy
    ///                         overflow-only stripes and whole-image padding.
    ///                         Too-small positive budgets use OpenCV instead.
    /// @return No value. Invalid input throws cv::Exception.
    ///
    /// Example input: a grayscale image, {3, 13, 23}, 7.0, &outputs
    /// Example output: three images identical to cv::adaptiveThreshold results
    void apply(const cv::Mat& gray, const std::vector<int>& window_sizes, double constant,
               std::vector<cv::Mat>* outputs, int parallel_min_pixels = 32768,
               std::size_t max_prefix_bytes = 2U * 1024U * 1024U);

private:
    cv::Mat padded_;
    cv::Mat integral_;
};

}  // namespace aruco3cuda::cpu::detail

#endif  // ARUCO3CUDA_CPU_THRESHOLD_HPP
