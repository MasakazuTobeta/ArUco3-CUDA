// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CORE_THRESHOLD_HPP
#define ARUCO3CUDA_CORE_THRESHOLD_HPP

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"
#include "aruco3cuda/workspace.hpp"
#include "preprocess.hpp"

namespace aruco3cuda::detail {

/// Output and scratch space of the adaptive threshold.
///
/// Ownership: the regions all of the pointers refer to are owned by the
///            workspace. This struct holds references only; it neither copies nor
///            frees. Every pointer becomes invalid once the workspace is reset()
///            or destroyed.
/// Synchronization: a plain set of references, so it carries no synchronization
///                  point. The contents are not settled until the already-issued
///                  kernels complete.
///
/// Example input: the default configuration and a 427x240 segmentation image
/// Example output: window_count_ = 3, window_sizes_px_ = {3, 13, 23}
struct ThresholdBuffers {
    /// Threshold result per window. The index corresponds to window_sizes_px_.
    ImagePlaneU8 binary_[kMaxAdaptiveThresholdWindows];
    int window_count_ = 0;
    int window_sizes_px_[kMaxAdaptiveThresholdWindows] = {};
    /// Scratch space holding the row-wise sums. Reused across windows.
    std::int32_t* row_sums_ = nullptr;
    std::size_t row_sums_pitch_bytes_ = 0;
    int width_px_ = 0;
    int height_px_ = 0;
};

/// Derives the list of windows to scan from the configuration.
///
/// OpenCV rounds an even window up to an odd one. The number of scans is
/// determined by the value before that rounding, so the same window size can
/// appear more than once. OpenCV processes the duplicates as they are, so they
/// are not removed here either.
///
/// @param config Detection configuration.
/// @param out_sizes Receives the window sizes on success. Must not be nullptr.
/// @param capacity Element count of out_sizes.
/// @param out_count Receives the window count on success. Must not be nullptr.
/// @return kOk, or kInvalidArgument, kInvalidConfig.
///
/// Ownership: retains none of the regions passed as arguments.
/// Synchronization: host only, so it carries no synchronization point.
///
/// Example input: the default configuration (min 3, max 23, step 10)
/// Example output: out_count = 3, out_sizes = {3, 13, 23}
Status threshold_window_sizes(const DetectorConfig& config, int* out_sizes, int capacity,
                              int* out_count);

/// Returns the required workspace capacity.
///
/// @param config Detection configuration.
/// @param width_px Width of the segmentation image.
/// @param height_px Height of the segmentation image.
/// @return Required byte count. 0 on overflow or an invalid configuration.
///
/// Ownership: retains no resource.
/// Synchronization: host only, so it carries no synchronization point.
///
/// Example input: the default configuration and 427x240
/// Example output: the byte count that holds 3 binary planes and 1 row-sum plane
std::size_t threshold_workspace_bytes(const DetectorConfig& config, int width_px, int height_px);

/// Carves the threshold regions out of the workspace.
///
/// @param config Detection configuration.
/// @param width_px Width of the segmentation image. At least 1.
/// @param height_px Height of the segmentation image. At least 1.
/// @param workspace Source of the carve-out. Owned by the caller.
/// @param out Receives the full set of buffers on success. Must not be nullptr.
/// @return kOk. kInvalidConfig when the capacity is insufficient,
///         kInvalidArgument when an argument is invalid.
///
/// Ownership: the carved-out regions stay owned by the workspace.
/// Synchronization: host only, so it carries no synchronization point.
///
/// Example input: the default configuration, 427x240, and a workspace with
///                sufficient capacity
/// Example output: binary_ receives pointers for 3 planes
Status reserve_threshold(const DetectorConfig& config, int width_px, int height_px,
                         Workspace& workspace, ThresholdBuffers* out);

/// Runs the adaptive threshold.
///
/// Aims for the same result as calling OpenCV `adaptiveThreshold` with
/// `ADAPTIVE_THRESH_MEAN_C` and `THRESH_BINARY_INV`. The mean is `boxFilter`
/// applied with normalization, the border is BORDER_REPLICATE, and the test
/// yields 255 when (pixel - mean) <= -floor(constant) and 0 otherwise.
///
/// @param segmentation Input image. Must have the same size as passed to
///                     reserve_threshold.
/// @param buffers The set of buffers returned by reserve_threshold. Must not be
///                nullptr.
/// @param config Detection configuration.
/// @param stream Stream to issue on. Pass nullptr to use the default stream.
/// @return kOk, or kInvalidArgument, kCudaError.
///
/// Ownership: the regions buffers points at stay owned by the workspace.
/// Synchronization: only issues kernels on the stream and performs no host
///                  synchronization. The row-sum scratch space is reused across
///                  windows, so the windows run in order within the same stream.
///                  Splitting them across several streams would require separate
///                  scratch space.
///
/// Example input: a 427x240 segmentation image and the default configuration
/// Example output: binary_[0..2] are filled with the threshold results for
///                 windows 3, 13, and 23
Status build_threshold_async(const ImageViewU8& segmentation, ThresholdBuffers* buffers,
                             const DetectorConfig& config, cudaStream_t stream);

}  // namespace aruco3cuda::detail

#endif  // ARUCO3CUDA_CORE_THRESHOLD_HPP
