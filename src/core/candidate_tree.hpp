// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CORE_CANDIDATE_TREE_HPP
#define ARUCO3CUDA_CORE_CANDIDATE_TREE_HPP

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/workspace.hpp"
#include "candidate_filter.hpp"
#include "dictionary_match.hpp"

namespace aruco3cuda::detail {

/// Containment relations between candidates, and how far the walk reached.
///
/// OpenCV holds the candidates as a nested tree and identifies them from the
/// innermost outward. Once a marker is found inside, the candidates enclosing
/// it need not be identified. Both the outer and the inner side of the black
/// border become candidates, so without this cutoff the same marker would be
/// emitted twice. This cutoff is what S9 calls duplicate cleanup.
///
/// Ownership: every region the pointers refer to is owned by the workspace.
/// Synchronization: only a set of references; it holds no synchronization
///                  point. The contents are not final until the submitted
///                  kernels complete.
///
/// Example input: settings with a candidate limit of 4096
/// Example output: parent_ and depth_ with 4096 elements, stop_depth_ with 1
struct CandidateTreeBuffers {
    /// Index of the candidate enclosing this one, or -1 if there is none.
    ///
    /// OpenCV picks the **largest** index among the candidates whose index is
    /// smaller than its own. Indices are ordered by descending perimeter, so
    /// this means "the innermost (smallest perimeter) of the enclosing
    /// candidates", not the smallest index.
    std::int32_t* parent_ = nullptr;
    /// Nesting level. Not the depth seen from a candidate that nothing
    /// encloses, but the number of levels this candidate encloses. The
    /// innermost is 0.
    std::int32_t* depth_ = nullptr;
    /// Whether the candidate was marked as an ancestor. 1 means marked. This is
    /// intermediate state of the walk, not the condition for skipping
    /// identification.
    ///
    /// One byte would be enough, but it is held as 32 bit: marking uses
    /// atomicExch, and CUDA has no 8 bit version of it.
    std::int32_t* visited_ = nullptr;
    /// Level at which the walk stopped. One element.
    ///
    /// Only candidates whose depth_ is below this value are identified. Equal
    /// to the depth when OpenCV's while loop exits.
    std::int32_t* stop_depth_ = nullptr;
    /// Number of candidates the walk reached. One element. Kept for
    /// verification and debugging.
    std::int32_t* counter_ = nullptr;
    int capacity_ = 0;
};

/// Returns the workspace size needed for the containment tree.
///
/// @param config Settings including the candidate limit.
/// @return Required byte count. 0 when config is invalid.
///
/// Ownership: does not retain the argument regions.
/// Synchronization: host only; it holds no synchronization point. Calls no CUDA API.
///
/// Example input: settings with max_candidates_ = 4096
/// Example output: 49664
std::size_t candidate_tree_workspace_bytes(const DetectorConfig& config);

/// Allocates the regions for the containment tree.
///
/// @param config Settings including the candidate limit.
/// @param workspace Workspace in device space.
/// @param out Receives the buffers on success. The regions stay owned by the caller.
/// @return kOk. kInvalidArgument on invalid arguments, kInvalidConfig when the
///         capacity is insufficient.
///
/// Ownership: does not retain the argument regions.
/// Synchronization: host only; it holds no synchronization point.
///
/// Example input: settings with max_candidates_ = 4096 and a workspace with room
/// Example output: kOk. out->capacity_ = 4096
Status reserve_candidate_tree(const DetectorConfig& config, Workspace& workspace,
                              CandidateTreeBuffers* out);

/// Derives the parent and the nesting level from candidate containment.
///
/// Containment is decided the same way as calling OpenCV's `pointPolygonTest`
/// with `measureDist = false`. A candidate counts as inside when all four of
/// its corners lie inside or on the boundary of the other. Cross products are
/// computed in 64 bit integers; in single precision the products of coordinate
/// differences round off and can change sign.
///
/// Levels are propagated sequentially in descending index order. Doing it in
/// parallel would read levels that are not settled yet and change the result.
/// There are at most a few thousand candidates, so a sequential pass is fine.
///
/// @param grouped Merged candidates. Assumed to be ordered by descending perimeter.
/// @param tree Destination.
/// @param stream Stream the kernels are submitted to.
/// @return kOk. kInvalidArgument on invalid arguments, kCudaError when a kernel
///         launch fails.
///
/// Ownership: does not retain the argument regions.
/// Synchronization: submits kernels asynchronously on the stream. The results
///                  are not final until the caller synchronizes.
///
/// Example input: two candidates nested two levels deep
/// Example output: kOk. parent_ = {-1, 0}, depth_ = {1, 0}
Status build_candidate_tree_async(const DeviceCandidates& grouped, CandidateTreeBuffers* tree,
                                  cudaStream_t stream);

/// Determines the level at which identification is cut off.
///
/// OpenCV identifies candidates starting at level 0 and stops once the number
/// of candidates reached covers them all. The reached count tallies both "the
/// candidates identified at that level" and "the ancestors of the candidates
/// that were identified". A candidate counted as an ancestor is counted again
/// when its own level comes up. This double counting works to make the cutoff
/// happen earlier, so it must not be removed.
///
/// @param grouped Buffer holding the candidate count.
/// @param matches Match results. A candidate whose ids_ is 0 or greater counts as identified.
/// @param tree Buffer with the parents and levels already stored. stop_depth_
///             and counter_ are written.
/// @param stream Stream the kernels are submitted to.
/// @return kOk. kInvalidArgument on invalid arguments, kCudaError when a kernel
///         launch fails.
///
/// Ownership: does not retain the argument regions.
/// Synchronization: submits kernels asynchronously on the stream. The results
///                  are not final until the caller synchronizes.
///
/// Example input: a two-level tree whose inner candidate was identified
/// Example output: kOk. stop_depth_ = 1. The outer candidate at level 1 is not identified
Status resolve_suppression_async(const DeviceCandidates& grouped, const MatchBuffers& matches,
                                 CandidateTreeBuffers* tree, cudaStream_t stream);

}  // namespace aruco3cuda::detail

#endif  // ARUCO3CUDA_CORE_CANDIDATE_TREE_HPP
