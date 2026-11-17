// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CPU_SPARSE_PYRAMID_HPP
#define ARUCO3CUDA_CPU_SPARSE_PYRAMID_HPP

#include <opencv2/core.hpp>

#include <cstddef>
#include <map>
#include <vector>

namespace aruco3cuda::cpu::detail {

/// Demand-generated pyramid regions and monitored subpixel refinement.
/// Ownership: owns regional image buffers, masks and coordinate maps; borrows inputs per call.
/// Synchronization: synchronous host operations; one instance per worker is required.
/// Example input: a full-size grayscale image and a small marker at pyramid level two.
/// Example output: canonical samples and refined corners without generating unrelated pixels.
class SparsePyramid {
public:
    /// Invalidate every cached image region while retaining allocation capacity.
    /// @param tile_side_px Positive region allocation granularity in pyramid pixels.
    /// @param full_level_ratio Generate the full level when requested tiles cover this fraction.
    /// @return No value. Invalid granularity or ratio throws cv::Exception.
    /// Ownership: retains no input image or detections.
    /// Synchronization: host only; call before using pixels from a new input frame.
    /// Example input: reset(64) before processing changed pixels at the same address.
    /// Example output: all next pixel requests are regenerated from the current input.
    void reset(int tile_side_px = 8, double full_level_ratio = 0.5);

    /// Read an exact region of a dyadic 8-bit image pyramid.
    /// @param input Original full image, borrowed during the call; submatrices are supported only
    /// at level zero.
    /// @param sizes Global pyramid dimensions, beginning with input.size().
    /// @param level Requested pyramid level.
    /// @param requested Nonempty rectangle inside the global level image.
    /// @return A borrowed view valid until another request at the same level or reset.
    /// Ownership: owns generated buffers; level-zero views borrow the input.
    /// Synchronization: synchronous host image processing.
    /// Example input: level two and rectangle (40, 50, 20, 20).
    /// Example output: pixels identical to buildPyramid(input)[2](rectangle).
    cv::Mat region(const cv::Mat& input, const std::vector<cv::Size>& sizes, int level,
                   const cv::Rect& requested);

    /// Generate canonical nearest-neighbor samples using OpenCV's global coordinate map.
    /// @param input Original grayscale image, borrowed during the call.
    /// @param sizes Global pyramid dimensions.
    /// @param level Source pyramid level.
    /// @param corners Four source corners in global level coordinates.
    /// @param canonical_size Output width and height.
    /// @param canonical Receives the canonical image; storage is reused.
    /// @return No value. Invalid geometry throws cv::Exception.
    /// Ownership: output storage belongs to the caller; inputs are not retained.
    /// Synchronization: synchronous host operations.
    /// Example input: a marker quad at level two and canonical size (32, 32).
    /// Example output: bytes equal to warpPerspective of the corresponding full level.
    void warp(const cv::Mat& input, const std::vector<cv::Size>& sizes, int level,
              const std::vector<cv::Point2f>& corners, cv::Size canonical_size, cv::Mat* canonical);

    /// Refine points while requesting each iteration's actual interpolation footprint.
    /// @param input Original grayscale image, borrowed during the call.
    /// @param sizes Global pyramid dimensions.
    /// @param level Source pyramid level.
    /// @param corners Points in global level coordinates, updated in place.
    /// @param radius Square refinement window radius in pixels.
    /// @param max_iterations Iteration cap, clamped like OpenCV to [1, 100].
    /// @param epsilon Convergence distance in pixels.
    /// @return No value. Invalid initial coordinates throw cv::Exception.
    /// Ownership: retains no input or corner storage.
    /// Synchronization: synchronous host operations.
    /// Example input: four approximate corners and radius three.
    /// Example output: cornerSubPix-equivalent global coordinates.
    void refine(const cv::Mat& input, const std::vector<cv::Size>& sizes, int level,
                std::vector<cv::Point2f>* corners, int radius, int max_iterations, double epsilon);

private:
    struct CachedRegion {
        cv::Mat pixels_;
        cv::Mat downsampled_;
        std::vector<unsigned char> valid_;
        int tiles_x_ = 0;
        int tile_side_px_ = 0;
        std::size_t valid_count_ = 0U;
    };
    int tile_side_px_ = 8;
    double full_level_ratio_ = 0.5;
    std::vector<CachedRegion> levels_;
    std::map<int, cv::Mat> masks_;
    cv::Mat patch_;
};

}  // namespace aruco3cuda::cpu::detail
#endif  // ARUCO3CUDA_CPU_SPARSE_PYRAMID_HPP
