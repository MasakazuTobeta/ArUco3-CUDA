// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CORE_CORNER_REFINE_HPP
#define ARUCO3CUDA_CORE_CORNER_REFINE_HPP

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "detection_emit.hpp"
#include "preprocess.hpp"

namespace aruco3cuda::detail {

/// Upper bound on the side length of the window used by the refinement.
///
/// On the ArUco3 path OpenCV sets the window radius to either 3 or 5. What cornerSubPix actually
/// cuts out is a square of side (2r+1)+2 derived from that radius, which is 13 when r = 5.
inline constexpr int kMaxRefinePatchSide = 13;

/// Counters that record how the refinement progressed.
///
/// This is an iterative solver, so 1 ULP of difference in the input changes the iteration count or
/// whether a bail-out is taken. These counters make it possible to say afterwards which branch
/// diverged from the CPU reference.
///
/// Ownership: the workspace owns every region these pointers refer to.
/// Synchronization: this is only a bundle of references and holds no synchronization point.
///
/// Example input: a configuration with a detection cap of 1024
/// Example output: five counters initialized to 0
struct RefineDiagnostics {
    /// [0] Number of corners that were refined
    /// [1] Number of bail-outs caused by the window leaving the image
    /// [2] Number of corners reset to their initial position for drifting too far from it
    /// [3] Number of bail-outs caused by a determinant close to 0
    /// [4] Total number of iterations
    std::int32_t* counters_ = nullptr;
};

/// Scratch space used by the refinement.
///
/// Ownership: the workspace owns every region these pointers refer to.
/// Synchronization: this is only a bundle of references and holds no synchronization point.
///
/// Example input: the default configuration
/// Example output: diagnostics_.counters_ holding five elements
struct CornerRefineBuffers {
    RefineDiagnostics diagnostics_;
};

/// Number of counter elements.
inline constexpr int kRefineCounterCount = 5;

/// Returns the workspace size required by the refinement.
///
/// @param config Configuration. It does not affect the size at present.
/// @return Required number of bytes.
///
/// Ownership: does not retain the regions of the arguments.
/// Synchronization: host only, holds no synchronization point. Calls no CUDA API.
///
/// Example input: the default configuration
/// Example output: 256
std::size_t corner_refine_workspace_bytes(const DetectorConfig& config);

/// Allocates the scratch space used by the refinement.
///
/// @param config Configuration.
/// @param workspace Workspace in device space.
/// @param out Receives the buffers on success. The caller owns the regions.
/// @return kOk, kInvalidArgument when out is nullptr, kInvalidConfig when the capacity is
///         insufficient.
///
/// Ownership: does not retain the regions of the arguments.
/// Synchronization: host only, holds no synchronization point.
///
/// Example input: the default configuration and a workspace with room left
/// Example output: kOk, with out->diagnostics_.counters_ valid
Status reserve_corner_refine(const DetectorConfig& config, Workspace& workspace,
                             CornerRefineBuffers* out);

/// Refines the four corners while climbing the pyramid levels and returns them to full-scale
/// coordinates.
///
/// Reproduces OpenCV's `findCornerInPyrImage` and `cv::cornerSubPix`.
///
/// The procedure is as follows.
///
/// 1. Multiply the four corners, which are in segmentation coordinates, by `scale_init` to obtain
///    the coordinates of the starting level. `scale_init` is `starting level width / segmentation
///    width`.
/// 2. Every time the level index drops by one (the resolution doubles), double the corners and
///    apply `cornerSubPix` at that level. Repeat down to level 0.
/// 3. Level 0 is full scale, so once the loop ends the corners are already in full-scale
///    coordinates. The reciprocal of `fxfy` must not be applied on top of that.
///
/// The window radius is decided by the size of the level, not by the configuration. OpenCV uses
/// `max(level width, level height) > 1080 ? 5 : 3`. On the ArUco3 path neither
/// `corner_refinement_win_size_px_` nor `relative_corner_refinement_win_size_` is used.
///
/// The arithmetic follows the same order as the CPU reference and never contracts multiply-add.
/// Because this is an iterative solver, the 1 ULP that contraction introduces can change the
/// iteration count or whether a bail-out is taken.
///
/// @param pyramid Image of each level. Level 0 is full scale.
/// @param plan Downscaling plan. The starting level and the segmentation width are used.
/// @param config Configuration containing the iteration count and the convergence threshold.
/// @param block_count Number of blocks to launch. Obtained from refine_block_count().
/// @param buffers Scratch space. The counters are written into it.
/// @param detections Corners are rewritten **in place**. ids_ and count_ are read.
/// @param stream Stream the kernel is issued on.
/// @return kOk, kInvalidArgument when an argument is invalid, kInvalidConfig when the
///         configuration is inconsistent, kCudaError when the kernel launch fails.
///
/// Ownership: does not retain the regions of the arguments.
/// Synchronization: the kernel is issued asynchronously on the stream. The results are not final
///           until the caller synchronizes.
///
/// Example input: a 1280x720 pyramid, a 427x240 segmentation image and four detections
/// Example output: kOk, with corner_x_ and corner_y_ in full-scale 1280x720 coordinates
Status refine_corners_async(const PyramidRef& pyramid, const ScalePlan& plan,
                            const DetectorConfig& config, int block_count,
                            CornerRefineBuffers* buffers, DeviceDetections* detections,
                            cudaStream_t stream);

/// Decides how many blocks the refinement launches, based on the SM count of the device.
///
/// The block count should be the smaller of "how many the device can run concurrently" and "how
/// much work there is". The former is estimated as twice the SM count, and the latter (16 markers
/// = 64 corners, the upper bound of the evaluation plan) is used as the cap. One block consumes
/// about 5.5 KB of shared memory, so launching more than necessary costs time even on a frame
/// with no detections.
///
/// A fixed value turns into an accident on a machine with an order-of-magnitude different SM
/// count. When the corner cap (4096) was used directly as the block count, a Jetson AGX Orin
/// (16 SM) was 7 times slower even with no detections.
///
/// @param multi_processor_count SM count of the device. Values below 1 yield the lower bound.
/// @return Number of blocks to launch, between 32 and 64 inclusive.
///
/// Ownership: holds no resources.
/// Synchronization: host only, holds no synchronization point. Calls no CUDA API.
///
/// Example input: 16 (Jetson AGX Orin)
/// Example output: 32
int refine_block_count(int multi_processor_count);

}  // namespace aruco3cuda::detail

#endif  // ARUCO3CUDA_CORE_CORNER_REFINE_HPP
