// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CORE_DETECTION_EMIT_HPP
#define ARUCO3CUDA_CORE_DETECTION_EMIT_HPP

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/detections.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/workspace.hpp"
#include "candidate_filter.hpp"
#include "candidate_tree.hpp"
#include "dictionary_match.hpp"
#include "scan.hpp"

namespace aruco3cuda::detail {

/// Detection results that stay on the device.
///
/// The definition lives in the public header. If the type handed between stages
/// differed from the public type, the public API would have to copy the internal
/// layout across. The same type is used instead.
using DeviceDetections = ::aruco3cuda::DeviceDetections;

/// Scratch space for the compaction.
///
/// Ownership: every region the pointers refer to is owned by the workspace.
/// Synchronization: only a set of references; it holds no synchronization point.
///
/// Example input: settings with max_candidates_ = 4096
/// Example output: offsets_ with 4096 elements
struct DetectionEmitBuffers {
    /// Verdict per candidate (0 or 1). After the scan it holds the write index.
    std::int32_t* offsets_ = nullptr;
    ScanBuffers scan_;
    int capacity_ = 0;
};

/// Returns the workspace size needed for the detections and the scratch space.
///
/// The predicate lives in candidate space (max_candidates_) and the output in
/// detection space (max_markers_), so their element counts differ. The scan runs
/// in candidate space.
///
/// @param config Settings including the candidate limit and the detection limit.
/// @return Required byte count. 0 when config is invalid.
///
/// Ownership: does not retain the argument regions.
/// Synchronization: host only; it holds no synchronization point. Calls no CUDA API.
///
/// Example input: settings with max_candidates_ = 4096 and max_markers_ = 1024
/// Example output: 62464
std::size_t detection_workspace_bytes(const DetectorConfig& config);

/// Allocates the detections and the scratch space.
///
/// @param config Settings including the candidate limit and the detection limit.
/// @param workspace Workspace in device space.
/// @param out_buffers Receives the scratch space on success. The regions stay owned by the caller.
/// @param out Receives the references to the detections on success. The regions
///            stay owned by the caller.
/// @return kOk. kInvalidArgument on invalid arguments, kInvalidConfig when the
///         capacity is insufficient.
///
/// Ownership: does not retain the argument regions.
/// Synchronization: host only; it holds no synchronization point.
///
/// Example input: the default settings and a workspace with room
/// Example output: kOk. out->capacity_ = 1024, out_buffers->capacity_ = 4096
Status reserve_detections(const DetectorConfig& config, Workspace& workspace,
                          DetectionEmitBuffers* out_buffers, DeviceDetections* out);

/// Packs the accepted candidates into the detection results.
///
/// A candidate is accepted when all three of the following hold.
///
/// 1. It lies within the candidate count
/// 2. Its level is below the cutoff level (the walk reached it)
/// 3. Dictionary matching gave it an ID
///
/// **Detections sharing an ID are not dropped.** OpenCV has no ID-based
/// duplicate removal either. Two copies of the same marker in different places
/// are both emitted. Duplicates disappear as a result of the cutoff through the
/// containment tree, not from comparing IDs.
///
/// The output preserves the order of the input candidates. Merged candidates are
/// ordered by descending perimeter, so the detections are too. This is the same
/// order as OpenCV's accepted.
///
/// @param grouped Merged candidates.
/// @param matches Match results.
/// @param tree Buffer with the levels and the cutoff level already stored.
/// @param buffers Scratch space.
/// @param out Destination.
/// @param stream Stream the kernels are submitted to.
/// @return kOk. kInvalidArgument on invalid arguments, kCudaError when a kernel
///         launch fails. Exceeding the limit is not returned here; pick it up
///         from read_detection_count.
///
/// Ownership: does not retain the argument regions.
/// Synchronization: submits kernels asynchronously on the stream. The results
///                  are not final until the caller synchronizes.
///
/// Example input: 3 candidates, 2 of which have an ID
/// Example output: kOk. count_ = 2
Status emit_detections_async(const DeviceCandidates& grouped, const MatchBuffers& matches,
                             const CandidateTreeBuffers& tree, DetectionEmitBuffers* buffers,
                             DeviceDetections* out, cudaStream_t stream);

/// Reads the detection count back to the host.
///
/// @param detections Subject.
/// @param out_count Receives the detection count on success. The region stays owned by the caller.
/// @param stream Stream to synchronize.
/// @return kOk. kMarkerOverflow when truncated at the limit. kInvalidArgument on
///         invalid arguments, kCudaError when a CUDA API call fails.
///
/// Ownership: does not retain the argument regions.
/// Synchronization: **synchronizes the stream.** This is the only function here
///                  that synchronizes with the host.
///
/// Example input: 2 detections with a limit of 1024
/// Example output: kOk. *out_count = 2
/// Example input: 3 detections with a limit of 2
/// Example output: kMarkerOverflow. *out_count = 2
Status read_detection_count(const DeviceDetections& detections, int* out_count,
                            cudaStream_t stream);

}  // namespace aruco3cuda::detail

#endif  // ARUCO3CUDA_CORE_DETECTION_EMIT_HPP
