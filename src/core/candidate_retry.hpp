// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CORE_CANDIDATE_RETRY_HPP
#define ARUCO3CUDA_CORE_CANDIDATE_RETRY_HPP

#include "candidate_group.hpp"
#include "dictionary_match.hpp"

namespace aruco3cuda::detail {

/// Workspace-owned alternatives for failed group representatives. All arrays
/// have capacity_ entries; candidates and matches have the same capacity.
/// Preparation and resolution submit device work without host synchronization.
///
/// Ownership: all referenced device regions belong to the caller's workspace.
/// Synchronization: these references carry no synchronization point.
/// Example input: reserve_candidate_retries with a capacity of 4096.
/// Example output: capacity_ is 4096 and all scratch regions are reserved.
struct CandidateRetryBuffers {
    DeviceCandidates candidates_;
    MatchBuffers matches_;
    std::int32_t* owners_ = nullptr;
    std::int32_t* flags_ = nullptr;
    std::int32_t* offsets_ = nullptr;
    std::int32_t* chosen_ = nullptr;
    ScanBuffers scan_;
    int capacity_ = 0;
};

/// Byte count for all retry scratch, candidates and match results; zero if invalid.
/// @param config Configuration whose candidate capacity bounds all alternatives.
/// @return Byte count, or zero for invalid or unrepresentable capacity.
/// Ownership: retains no resources.
/// Synchronization: host only, with no CUDA call.
/// Example input: configuration with max_candidates_ = 4096.
/// Example output: enough bytes for 4096 alternatives and their match results.
std::size_t candidate_retry_workspace_bytes(const DetectorConfig& config);

/// Reserve device workspace regions. No per-frame allocation or synchronization.
/// @param config Configuration whose candidate capacity determines region sizes.
/// @param workspace Caller-owned device arena with enough remaining capacity.
/// @param out Receives the reserved references; must not be null.
/// @return kOk, kInvalidArgument for null output, or the workspace failure status.
/// Ownership: all references remain owned by workspace.
/// Synchronization: host allocation bookkeeping only.
/// Example input: a sufficiently sized device arena and default configuration.
/// Example output: initialized retry storage with no frame contents yet.
Status reserve_candidate_retries(const DetectorConfig& config, Workspace& workspace,
                                 CandidateRetryBuffers* out);

/// Emit eligible alternatives only for representatives whose primary match failed.
/// Each raw candidate is emitted at most once, in descending quad-perimeter order.
/// The copied contour length comes from the representative so all its alternatives
/// use the same pyramid level as the CPU decode loop. The raw candidates, grouping
/// maps and primary matches must belong to the same current frame.
/// @param raw Current-frame candidates before grouping.
/// @param grouped Original group representatives.
/// @param primary Match results of the representatives.
/// @param groups Completed current-frame grouping maps and quad perimeters.
/// @param config Detection settings, including close-contour minimum distance.
/// @param marker_size Number of dictionary cells on one marker side.
/// @param retries Reserved output and scratch storage.
/// @param stream CUDA stream; null selects the default stream.
/// @return kOk, kInvalidArgument, or a CUDA submission error.
/// Ownership: input and output storage stays with its caller.
/// Synchronization: submits stream work without host synchronization.
/// Example input: one failed group containing three eligible close contours.
/// Example output: three candidates inheriting the representative's sampling length.
Status prepare_candidate_retries_async(const DeviceCandidates& raw, const DeviceCandidates& grouped,
                                       const MatchBuffers& primary,
                                       const CandidateGroupBuffers& groups,
                                       const DetectorConfig& config, int marker_size,
                                       CandidateRetryBuffers* retries, cudaStream_t stream);

/// Select the first successful eligible alternative per representative, then replace
/// its corners and match result. Build the containment tree before this call, because
/// CPU containment uses original representatives. Primary successes remain unchanged.
/// @param retries Prepared candidates with completed dictionary matches.
/// @param grouped Representatives whose successful fallback corners are updated.
/// @param primary Primary results whose failed slots may be replaced.
/// @param stream CUDA stream; null selects the default stream.
/// @return kOk, kInvalidArgument, or a CUDA submission error.
/// Ownership: references retain their original owners.
/// Synchronization: submits stream work without host synchronization.
/// Example input: two successful alternatives for a failed representative.
/// Example output: the earlier alternative supplies the representative's final result.
Status resolve_candidate_retries_async(const CandidateRetryBuffers& retries,
                                       DeviceCandidates* grouped, MatchBuffers* primary,
                                       cudaStream_t stream);

}  // namespace aruco3cuda::detail
#endif
