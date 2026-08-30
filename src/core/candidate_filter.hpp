// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CORE_CANDIDATE_FILTER_HPP
#define ARUCO3CUDA_CORE_CANDIDATE_FILTER_HPP

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/workspace.hpp"
#include "labeling.hpp"
#include "quad_extract.hpp"
#include "scan.hpp"

namespace aruco3cuda::detail {

/// Candidates that passed the filter, packed into a dense array.
///
/// Labels are scattered across the whole image, so handing them downstream as
/// they are would leave most indices empty. Packing them lets every later stage
/// walk only as many entries as there are candidates.
///
/// Ownership: every region the pointers refer to is owned by the workspace.
/// Synchronization: only a set of references; it holds no synchronization point.
///
/// Example input: reserve_candidates with a candidate limit of 4096
/// Example output: capacity_ = 4096, corner_x_ holding 4 * 4096 elements
struct DeviceCandidates {
    /// x coordinates of the four corners. The index is (corner * capacity_) + candidate.
    std::int32_t* corner_x_ = nullptr;
    /// y coordinates of the four corners. Laid out like corner_x_.
    std::int32_t* corner_y_ = nullptr;
    /// Label the candidate came from, so the originating component can be traced later.
    std::int32_t* label_ = nullptr;
    /// Chain code length of the polyline through the four corners.
    std::int32_t* perimeter_ = nullptr;
    /// Number of packed candidates. One element. Holds the value after truncation at the limit.
    std::int32_t* count_ = nullptr;
    /// Number of candidates that passed the filter. One element. Larger than
    /// count_ when the limit was exceeded.
    std::int32_t* accepted_total_ = nullptr;
    int capacity_ = 0;
};

/// Scratch space used to filter and pack candidates.
///
/// Ownership: every region the pointers refer to is owned by the workspace.
/// Synchronization: only a set of references; it holds no synchronization point.
///
/// Example input: 427x240 with a candidate limit of 4096
/// Example output: verdicts for 25680 labels plus the scan scratch space
struct CandidateFilterBuffers {
    /// Verdict per label. 1 means accepted.
    std::int32_t* accepted_ = nullptr;
    /// Exclusive scan of the verdicts. Becomes the write position of each accepted label.
    std::int32_t* offsets_ = nullptr;
    /// Number of component pixels that fall inside the estimated quadrilateral.
    std::int32_t* inside_count_ = nullptr;
    /// Chain code length of the polyline through the four corners, per label.
    std::int32_t* perimeter_ = nullptr;
    /// Number of pixels supporting each edge. The index is (corner * capacity_) + label.
    std::int32_t* edge_support_ = nullptr;
    /// Position where writing starts. One element.
    ///
    /// Candidates are collected once per binarization window and concatenated
    /// into a single array, so the count so far is kept on the device. Reading
    /// it back to the host and adding there would force a synchronization per
    /// window.
    std::int32_t* base_ = nullptr;
    ScanBuffers scan_;
    int capacity_ = 0;
};

/// Returns the workspace size needed to filter and pack candidates.
///
/// @param config Detection settings. The candidate limit is used.
/// @param width_px Image width.
/// @param height_px Image height.
/// @return Required byte count. 0 on overflow or invalid arguments.
///
/// Ownership: holds no resources.
/// Synchronization: host only; it holds no synchronization point.
///
/// Example input: the default settings with 427 and 240
/// Example output: the byte count holding scratch space for 25680 labels and 4096 candidates
std::size_t candidate_workspace_bytes(const DetectorConfig& config, int width_px, int height_px);

/// Carves the filtering and packing regions out of the workspace.
///
/// @param config Detection settings. The candidate limit is used.
/// @param width_px Image width. 1 or greater.
/// @param height_px Image height. 1 or greater.
/// @param workspace Source of the allocation. Owned by the caller.
/// @param out_buffers Receives the scratch space on success. Must not be nullptr.
/// @param out_candidates Receives the output region on success. Must not be nullptr.
/// @return kOk. kInvalidConfig when the capacity is insufficient,
///         kInvalidArgument on invalid arguments.
///
/// Ownership: the carved regions stay owned by the workspace.
/// Synchronization: host only; it holds no synchronization point.
///
/// Example input: the default settings with 427 and 240 and a workspace with enough room
/// Example output: capacity_ set to the label limit and to the candidate limit respectively
Status reserve_candidates(const DetectorConfig& config, int width_px, int height_px,
                          Workspace& workspace, CandidateFilterBuffers* out_buffers,
                          DeviceCandidates* out_candidates);

/// Filters the estimated corners and packs the survivors into a dense array.
///
/// The filter maps onto the decisions the CPU path makes in
/// `_findMarkerContours`. The perimeter is not the contour pixel count but the
/// chain code length of the polyline through the four corners. Extreme point
/// search has no contour, so it cannot count contour pixels, but the chain code
/// length of a straight line follows uniquely from its endpoints. For a convex
/// quadrilateral the two nearly agree.
///
/// Quadrilateral likeness is measured by two ratios instead of
/// `polygonalApproxAccuracyRate`. The first is the fraction of component pixels
/// falling inside the estimated quadrilateral, which rejects shapes such as
/// circles and ellipses that bulge outside it. The second is the number of
/// component pixels near each edge, which rejects shapes such as an L or a plus
/// sign whose edges run outside the component. Measurements on synthetic shapes
/// confirm that neither ratio alone separates these cases.
///
/// Extreme point search accepts a triangle, and two markers that touch and
/// become one component, as quadrilaterals: both pass either test. This is a
/// known limitation of design A. Its impact is assessed by comparing candidates
/// against the CPU reference.
///
/// Write positions come from an exclusive scan. Claiming slots with atomicAdd
/// would order candidates by arrival, so the candidate order would change from
/// run to run.
///
/// @param labels The label set filled in by build_labels_async.
/// @param stats The statistics filled in by build_label_stats_async.
/// @param quads The corners filled in by build_quads_async.
/// @param config Detection settings.
/// @param buffers The scratch space returned by reserve_candidates. Must not be nullptr.
/// @param candidates The output region returned by reserve_candidates. Must not be nullptr.
/// @param append true appends after the candidates already stored; false
///               rewrites from the beginning. When calling once per
///               binarization window, pass false only for the first window.
/// @param stream Stream to submit to. Pass nullptr to use the default stream.
/// @return kOk, or kInvalidArgument, kCudaError.
///         Exceeding the limit is asynchronous and is not decided here.
///
/// Ownership: the regions the arguments point to stay owned by the workspace.
/// Synchronization: only submits kernels to the stream; it performs no host synchronization.
///
/// Example input: a label image holding the four black marker borders plus noise components
/// Example output: count_ = 4, with four sets of corners in corner_x_/corner_y_
Status build_candidates_async(const LabelBuffers& labels, const LabelStatisticsBuffers& stats,
                              const QuadBuffers& quads, const DetectorConfig& config,
                              CandidateFilterBuffers* buffers, DeviceCandidates* candidates,
                              bool append, cudaStream_t stream);

/// Reads the candidate count back to the host and reports whether it was truncated.
///
/// @param candidates The output region passed to build_candidates_async.
/// @param out_count Receives the candidate count on success. Must not be nullptr.
/// @param stream Stream to submit to. Waits until the transfer completes.
/// @return kOk. kCandidateOverflow when truncated at the limit.
///         kInvalidArgument on invalid arguments, kCudaError when the transfer fails.
///
/// Ownership: does not retain the argument regions.
/// Synchronization: waits for the stream to complete. out_count is final once this call returns.
///
/// Example input: 5000 candidates pass the filter and the limit is 4096
/// Example output: out_count = 4096 and a return value of kCandidateOverflow
Status read_candidate_count(const DeviceCandidates& candidates, int* out_count,
                            cudaStream_t stream);

}  // namespace aruco3cuda::detail

#endif  // ARUCO3CUDA_CORE_CANDIDATE_FILTER_HPP
