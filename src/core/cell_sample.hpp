// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CORE_CELL_SAMPLE_HPP
#define ARUCO3CUDA_CORE_CELL_SAMPLE_HPP

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/workspace.hpp"
#include "candidate_filter.hpp"
#include "preprocess.hpp"

namespace aruco3cuda::detail {

/// Canonical image per candidate: the candidate rectified to a square by a
/// perspective transform.
///
/// One side is (marker_size + 2 * marker_border_bits_) * perspective_remove_pixel_per_cell_,
/// which is 32 for the default settings with DICT_ARUCO_MIP_36h12.
///
/// Ownership: the region the pointer refers to is owned by the workspace.
/// Synchronization: only a reference; it holds no synchronization point. The
///                  contents are not final until the submitted kernels complete.
///
/// Example input: the default settings, marker_size = 6, a candidate limit of 4096
/// Example output: side_px_ = 32, images_ holding 4096 * 32 * 32 bytes
struct CanonicalBuffers {
    /// Canonical image per candidate. The index is
    /// (candidate * side_px_ * side_px_) + (y * side_px_) + x.
    std::uint8_t* images_ = nullptr;
    int side_px_ = 0;
    int capacity_ = 0;
};

/// Returns the side length of the canonical image.
///
/// @param config Detection settings.
/// @param marker_size Cell count of the dictionary. 1 or greater.
/// @return Side length in pixels. 0 on invalid arguments.
///
/// Ownership: holds no resources.
/// Synchronization: host only; it holds no synchronization point.
///
/// Example input: the default settings and marker_size = 6
/// Example output: 32
int canonical_side_px(const DetectorConfig& config, int marker_size);

/// Returns the workspace size needed for the canonical images.
///
/// @param config Detection settings. The candidate limit is used.
/// @param marker_size Cell count of the dictionary.
/// @return Required byte count. 0 on overflow or invalid arguments.
///
/// Ownership: holds no resources.
/// Synchronization: host only; it holds no synchronization point.
///
/// Example input: the default settings and marker_size = 6
/// Example output: 4096 * 32 * 32 rounded up to a 256 byte boundary
std::size_t canonical_workspace_bytes(const DetectorConfig& config, int marker_size);

/// Carves the canonical image region out of the workspace.
///
/// @param config Detection settings. The candidate limit is used.
/// @param marker_size Cell count of the dictionary. 1 or greater.
/// @param workspace Source of the allocation. Owned by the caller.
/// @param out Receives the buffer on success. Must not be nullptr.
/// @return kOk. kInvalidConfig when the capacity is insufficient,
///         kInvalidArgument on invalid arguments.
///
/// Ownership: the carved region stays owned by the workspace.
/// Synchronization: host only; it holds no synchronization point.
///
/// Example input: the default settings, marker_size = 6 and a workspace with enough room
/// Example output: side_px_ = 32, capacity_ = 4096
Status reserve_canonical(const DetectorConfig& config, int marker_size, Workspace& workspace,
                         CanonicalBuffers* out);

/// Applies the perspective transform per candidate to build the canonical images.
///
/// The goal is to match calling OpenCV's `getPerspectiveTransform` and
/// `warpPerspective` with `INTER_NEAREST`. Their composition is the inverse map
/// from canonical back to the input image, reproduced as follows.
///
/// 1. Build the 8-unknown linear system in double precision. `a[i][6]` and
///    `a[i][7]` widen a single-precision product to double. Multiplying in
///    double precision shifts the matrix by up to 1.9e-3 relative.
/// 2. Solve by Gaussian elimination with partial pivoting. On a tie, keep the
///    row with the smaller index.
/// 3. Build the inverse with the 3x3 cofactor formula.
/// 4. Take the reciprocal first and then multiply, and round to nearest even to
///    pick the pixel index.
///
/// Multiply-add is not fused. Fusing shifts the sampled pixel by one at the
/// rounding boundaries. This translation unit is compiled with `-fmad=false`.
///
/// The pyramid level used for each candidate is chosen from the chain code
/// length of the polyline through the four corners. The CPU path uses the
/// contour pixel count, but design A has no contour. For a convex quadrilateral
/// the two nearly agree, but near the boundary the chosen level can be off by
/// one.
///
/// @param preprocess The preprocessing buffers. The pyramid is read from them.
/// @param plan The downscaling plan. The segmentation width is used to choose the level.
/// @param candidates The packed candidates. The corners and perimeters are read from them.
/// @param config Detection settings.
/// @param canonical The output region returned by reserve_canonical. Must not be nullptr.
/// @param stream Stream to submit to. Pass nullptr to use the default stream.
/// @return kOk, or kInvalidArgument, kCudaError.
///
/// Ownership: the regions the arguments point to stay owned by the workspace.
/// Synchronization: only submits kernels to the stream; it performs no host
///                  synchronization. The candidate count is read on the device
///                  and never brought back to the host.
///
/// Example input: candidates for four markers and a five-level pyramid
/// Example output: the first four entries of images_ filled with 32x32 canonical images
Status build_canonical_async(const PreprocessBuffers& preprocess, const ScalePlan& plan,
                             const DeviceCandidates& candidates, const DetectorConfig& config,
                             CanonicalBuffers* canonical, cudaStream_t stream);

}  // namespace aruco3cuda::detail

#endif  // ARUCO3CUDA_CORE_CELL_SAMPLE_HPP
