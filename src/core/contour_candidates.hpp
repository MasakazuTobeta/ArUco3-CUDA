// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CORE_CONTOUR_CANDIDATES_HPP
#define ARUCO3CUDA_CORE_CONTOUR_CANDIDATES_HPP

#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/workspace.hpp"
#include "candidate_filter.hpp"
#include "preprocess.hpp"
#include "scan.hpp"

namespace aruco3cuda::detail {

/// Integer pixel-center coordinates in a unit contour path.
/// Ownership: contains values only. Synchronization: none.
/// Example input: foreground pixel at column 3, row 7.
/// Example output: x_ = 3, y_ = 7.
struct ContourPoint {
    int x_ = 0;
    int y_ = 0;
};
/// Cyclic path interval used by the bounded RDP stack.
/// Ownership: contains indices only. Synchronization: none.
/// Example input: split a contour at point indices 0 and 12.
/// Example output: first_ = 0, last_ = 12.
struct ContourSlice {
    int first_ = 0;
    int last_ = 0;
};

/// Device diagnostic bits. Any nonzero value invalidates this invocation's output.
enum ContourFailure : std::int32_t {
    kContourInvalidCycle = 1,
    kContourPointOverflow = 2,
    kContourCandidateOverflow = 4,
    kContourApproximationOverflow = 8
};

/// Workspace for the experimental 8-foreground / 4-background contour route.
/// Ownership: every pointer refers to caller-owned Workspace storage.
/// Synchronization: references only; build completion requires stream synchronization.
/// Example input: 320x239 binary input and a 305920-point budget.
/// Example output: four directed-edge slots per pixel, bounded path and RDP storage.
struct ContourCandidateBuffers {
    std::int32_t* parents_ = nullptr;
    std::int32_t* previous_ = nullptr;
    /// Compact cycle roots, in increasing original edge order.
    std::int32_t* roots_ = nullptr;
    /// Device count of compact roots; aliases root_scan_.total_.
    std::int32_t* root_count_ = nullptr;
    std::int32_t* lengths_ = nullptr;
    std::int32_t* offsets_ = nullptr;
    /// Temporary root-to-descriptor map before point materialization.
    std::int32_t* count_map_ = nullptr;
    /// Temporary exact area and canonical hole seed before materialization.
    long long* count_area_ = nullptr;
    std::int32_t* count_seed_ = nullptr;
    /// Per-cycle crack-edge length; aliases RDP stack storage until approximation.
    std::int32_t* count_edges_ = nullptr;
    /// Thirty-two-step predecessor map; aliases expired count aggregation scratch.
    std::int32_t* jump_target_ = nullptr;
    /// Unordered long-cycle edge IDs; aliases stack scratch before approximation.
    std::int32_t* jump_edges_ = nullptr;
    /// Number of compact long-cycle edge IDs in the current invocation.
    std::int32_t* jump_count_ = nullptr;
    int jump_capacity_ = 0;
    ContourPoint* points_ = nullptr;
    ContourPoint* approximation_ = nullptr;
    ContourSlice* stack_ = nullptr;
    /// One scalar: persistent OR of failures across appended threshold windows.
    std::int32_t* failure_ = nullptr;
    /// Current invocation's path point count, retained after scan scratch is reused.
    std::int32_t* point_total_ = nullptr;
    std::int32_t* base_ = nullptr;
    ScanBuffers scan_;
    ScanBuffers root_scan_;
    int width_px_ = 0;
    int height_px_ = 0;
    int edge_capacity_ = 0;
    /// Minimum of the requested budget and the proven 2WH+W+H interface bound.
    int point_capacity_ = 0;
    int root_capacity_ = 0;
};

/// Computes required storage, with checked multiplication and addition.
/// @param width_px Positive binary image width.
/// @param height_px Positive binary image height.
/// @param max_points Positive point budget, internally capped at 2WH+W+H, which covers all paths.
/// @return Byte count, or zero for invalid/overflowing sizes.
/// Ownership: retains nothing. Synchronization: host arithmetic only.
/// Example input: 320, 239, 305920.
/// Example output: sufficient bytes for all contour paths and bounded approximation scratch.
std::size_t contour_candidate_workspace_bytes(int width_px, int height_px, int max_points);

/// Reserves bounded reusable contour storage without allocating inside detection.
/// @param width_px Positive binary image width.
/// @param height_px Positive binary image height.
/// @param max_points Positive configured point-pool bound.
/// @param workspace Caller-owned arena with sufficient capacity.
/// @param out Receives references on success, unchanged on failure.
/// @return kOk, kInvalidArgument or the workspace failure.
/// Ownership: all regions remain owned by workspace. Synchronization: host only.
/// Example input: a preallocated workspace and dimensions 320x239.
/// Example output: out points to edge, path, scan and approximation scratch.
Status reserve_contour_candidates(int width_px, int height_px, int max_points, Workspace& workspace,
                                  ContourCandidateBuffers* out);

/// Extracts outer and hole contour quadrilaterals on the device.
/// @param binary Current-frame binary plane; zero is background.
/// @param config Validated contour perimeter, approximation and corner-distance settings.
/// @param buffers Scratch returned by reserve_contour_candidates.
/// @param candidates Existing candidate output storage.
/// @param append Whether to append after an earlier threshold window.
/// @param stream Stream on which binary is ready.
/// @return Submission status. Caller must additionally inspect failure_ after synchronization.
/// Ownership: retains no new regions. Synchronization: submits kernels with no host readback.
/// Example input: two outlined rectangles connected by a foreground line.
/// Example output: both eligible hole contours, even though foreground has only one component.
Status build_contour_candidates_async(const ImagePlaneU8& binary, const DetectorConfig& config,
                                      ContourCandidateBuffers* buffers,
                                      DeviceCandidates* candidates, bool append,
                                      cudaStream_t stream);

}  // namespace aruco3cuda::detail
#endif  // ARUCO3CUDA_CORE_CONTOUR_CANDIDATES_HPP
