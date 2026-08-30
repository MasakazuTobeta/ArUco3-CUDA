// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CORE_CANDIDATE_GROUP_HPP
#define ARUCO3CUDA_CORE_CANDIDATE_GROUP_HPP

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/workspace.hpp"
#include "candidate_filter.hpp"
#include "scan.hpp"

namespace aruco3cuda::detail {

/// Scratch space used to merge nearby candidates.
///
/// Ownership: every region the pointers refer to is owned by the workspace.
/// Synchronization: only a set of references; it holds no synchronization point.
///
/// Example input: reserve_candidate_groups with a candidate limit of 4096
/// Example output: capacity_ = 4096 with a pointer in every array
struct CandidateGroupBuffers {
    /// Rank in descending perimeter order. Indexed by candidate index.
    std::int32_t* rank_ = nullptr;
    /// Map from rank back to candidate index. The inverse of rank_.
    std::int32_t* order_ = nullptr;
    /// Union-find over rank space. The root becomes the representative of the group.
    ///
    /// A smaller rank means a larger perimeter. Taking the root to be the
    /// smallest rank therefore amounts to selecting the largest perimeter in
    /// the group.
    std::int32_t* parent_ = nullptr;
    /// 1 for a representative.
    std::int32_t* selected_ = nullptr;
    /// Input and output of the exclusive scan that packs the representatives.
    std::int32_t* offsets_ = nullptr;
    ScanBuffers scan_;
    int capacity_ = 0;
};

/// Returns the workspace size needed for merging.
///
/// @param config Detection settings. The candidate limit is used.
/// @return Required byte count. 0 on invalid arguments.
///
/// Ownership: holds no resources.
/// Synchronization: host only; it holds no synchronization point.
///
/// Example input: settings with a candidate limit of 4096
/// Example output: the byte count holding the scratch space and 4096 merged candidates
std::size_t candidate_group_workspace_bytes(const DetectorConfig& config);

/// Carves the merging regions out of the workspace.
///
/// @param config Detection settings. The candidate limit is used.
/// @param workspace Source of the allocation. Owned by the caller.
/// @param out_buffers Receives the scratch space on success. Must not be nullptr.
/// @param out_grouped Receives the merged candidate region on success. Must not be nullptr.
/// @return kOk. kInvalidConfig when the capacity is insufficient,
///         kInvalidArgument on invalid arguments.
///
/// Ownership: the carved regions stay owned by the workspace.
/// Synchronization: host only; it holds no synchronization point.
///
/// Example input: the default settings and a workspace with enough room
/// Example output: capacity_ = 4096 with the pointers filled in
Status reserve_candidate_groups(const DetectorConfig& config, Workspace& workspace,
                                CandidateGroupBuffers* out_buffers, DeviceCandidates* out_grouped);

/// Merges nearby candidates into one.
///
/// Changing the binarization window yields slightly different candidates from
/// the same marker. Candidates whose mean corner distance is closer than a
/// ratio of the perimeter form one group, and the candidate with the largest
/// perimeter in the group is kept. Keeping a smaller one pulls the corners
/// inward.
///
/// There is one difference from the CPU reference. When two nearby candidates
/// already belong to separate groups, OpenCV does not merge those two. This
/// implementation takes the connected components of the nearness relation as
/// the groups, so it merges them in that case too. The two differ only for
/// layouts where three or more candidates are near each other in a chain. The
/// size of the difference is measured against the CPU reference.
///
/// @param input The candidate set filled in by build_candidates_async.
/// @param config Detection settings.
/// @param buffers The scratch space returned by reserve_candidate_groups. Must not be nullptr.
/// @param grouped The output region returned by reserve_candidate_groups. Must not be nullptr.
/// @param stream Stream to submit to. Pass nullptr to use the default stream.
/// @return kOk, or kInvalidArgument, kCudaError.
///
/// Ownership: the regions the arguments point to stay owned by the workspace.
/// Synchronization: only submits kernels to the stream; it performs no host synchronization.
///
/// Example input: three candidates from the same marker with differing perimeters
/// Example output: count_ = 1, keeping only the candidate with the largest perimeter
Status build_candidate_groups_async(const DeviceCandidates& input, const DetectorConfig& config,
                                    CandidateGroupBuffers* buffers, DeviceCandidates* grouped,
                                    cudaStream_t stream);

}  // namespace aruco3cuda::detail

#endif  // ARUCO3CUDA_CORE_CANDIDATE_GROUP_HPP
