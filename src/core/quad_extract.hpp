// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CORE_QUAD_EXTRACT_HPP
#define ARUCO3CUDA_CORE_QUAD_EXTRACT_HPP

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "aruco3cuda/status.hpp"
#include "aruco3cuda/workspace.hpp"
#include "labeling.hpp"

namespace aruco3cuda::detail {

/// Number of corner points.
constexpr int kQuadCornerCount = 4;

/// Per-label corners and the scratch space used to extract them.
///
/// The corners are found by an extreme-point search. The point farthest from the
/// centroid becomes c0, the point farthest from c0 becomes c2, and the points
/// farthest from the line c0c2 on either side become c1 and c3. There is no need
/// to walk the contour in order, so the work parallelizes completely per label.
///
/// Ownership: the regions all of the pointers refer to are owned by the workspace.
/// Synchronization: a plain set of references, so it carries no synchronization
///                  point. The contents are not settled until the already-issued
///                  kernels complete.
///
/// Example input: a 427x240 label image
/// Example output: capacity_ = 25680, corner_x_ holding 4 * 25680 elements
struct QuadBuffers {
    /// x coordinates of the corners. The index is (corner * capacity_) + label.
    ///
    /// The corners are laid out as separate arrays because a single kernel
    /// traverses only one corner at a time. Being contiguous along the label axis
    /// keeps the reads and writes coalesced.
    std::int32_t* corner_x_ = nullptr;
    /// y coordinates of the corners. Laid out the same way as corner_x_.
    std::int32_t* corner_y_ = nullptr;
    /// Whether the corners could be extracted. 1 for valid, 0 for invalid.
    ///
    /// A single-pixel component or a line-shaped component has no point on one
    /// side of the line c0c2. In that case the corners are not determined, so it
    /// is marked invalid.
    std::uint8_t* valid_ = nullptr;
    /// Intermediate search result. The distance and the pixel index packed into one
    /// word.
    ///
    /// The distance sits in the upper 32 bits and the pixel's linear index in the
    /// lower 32. A single atomicMax then settles both the "farthest point" and
    /// "where it is" at once. On a tie in distance the larger index survives, so
    /// the result does not depend on execution order.
    unsigned long long* best_ = nullptr;
    unsigned long long* best_positive_ = nullptr;
    unsigned long long* best_negative_ = nullptr;
    /// Number of labels allocated for.
    int capacity_ = 0;
};

/// Returns the workspace capacity corner extraction requires.
///
/// @param width_px Image width.
/// @param height_px Image height.
/// @return Required byte count. 0 on overflow or an invalid argument.
///
/// Ownership: retains no resource.
/// Synchronization: host only, so it carries no synchronization point.
///
/// Example input: 427 and 240
/// Example output: the byte count that holds the corners and the scratch space for
///                 25680 labels
std::size_t quad_workspace_bytes(int width_px, int height_px);

/// Carves the corner-extraction regions out of the workspace.
///
/// @param width_px Image width. At least 1.
/// @param height_px Image height. At least 1.
/// @param workspace Source of the carve-out. Owned by the caller.
/// @param out Receives the full set of buffers on success. Must not be nullptr.
/// @return kOk. kInvalidConfig when the capacity is insufficient,
///         kInvalidArgument when an argument is invalid.
///
/// Ownership: the carved-out regions stay owned by the workspace.
/// Synchronization: host only, so it carries no synchronization point.
///
/// Example input: 427, 240, and a workspace with sufficient capacity
/// Example output: capacity_ = 25680 and every array receives a pointer
Status reserve_quads(int width_px, int height_px, Workspace& workspace, QuadBuffers* out);

/// Finds the corners per label by an extreme-point search.
///
/// The corner order is brought into the same orientation as OpenCV
/// `_reorderCandidatesCorners`. Which corner comes first is determined by the
/// shape of the component, so it is not necessarily the same corner the CPU
/// reference starts from. That difference in starting corner is absorbed by the
/// rotation the dictionary match reports.
///
/// @param labels The set of labels filled in by build_labels_async.
/// @param stats The statistics filled in by build_label_stats_async. The centroid
///              is used as the starting point.
/// @param quads The set of buffers returned by reserve_quads. Must not be nullptr.
/// @param stream Stream to issue on. Pass nullptr to use the default stream.
/// @return kOk, or kInvalidArgument, kCudaError.
///
/// Ownership: the regions the arguments refer to stay owned by the workspace.
/// Synchronization: only issues kernels on the stream and performs no host
///                  synchronization. The search runs in 3 stages, executed in
///                  order because each stage consumes the previous stage's result.
///
/// Example input: a label image with one square outline carrying label 0
/// Example output: valid_[0] = 1, with corner_x_/corner_y_ at the 4 corners of the
///                 outline
Status build_quads_async(const LabelBuffers& labels, const LabelStatisticsBuffers& stats,
                         QuadBuffers* quads, cudaStream_t stream);

}  // namespace aruco3cuda::detail

#endif  // ARUCO3CUDA_CORE_QUAD_EXTRACT_HPP
