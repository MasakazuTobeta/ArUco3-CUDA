// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_DETECTIONS_HPP
#define ARUCO3CUDA_DETECTIONS_HPP

#include <cstdint>
#include <vector>

namespace aruco3cuda {

/// Device-reported failures. Any nonzero combination invalidates all detections.
/// Read only on the producing stream or after its successful completion.
enum class DeviceFailure : std::int32_t {
    kNone = 0,
    kInvalidContourCycle = 1,
    kContourPointOverflow = 2,
    kCandidateOverflow = 4,
    kContourApproximationOverflow = 8,
    kInvalidCandidateCounters = 16
};

/// Detection results that stay resident on the device.
///
/// Purpose:
///   Lets a GPU-resident pipeline read the results without a host synchronization.
///   If the stages that follow, such as pose estimation, run on the same device,
///   there is no need to bring the data back to the host.
///
/// The corners are ordered **after** the rotation found by dictionary matching has
/// been undone. rotations_ still holds the value from before that, so a later stage
/// that reads rotations_ and rotates again ends up 90 degrees off.
///
/// The coordinates are in the full resolution of the input image, not in the
/// coordinates of the downscaled segmentation image.
///
/// Ownership: the Detector owns the memory behind every pointer here. This struct
///            only holds references; it neither copies nor frees them. They become
///            invalid after destruction, reinitialization, or layout rebuilding.
///            Every later submission can overwrite the result contents. The caller
///            must finish or explicitly order downstream readers before reuse.
/// Synchronization: this is only a set of references and holds no synchronization
///                  point. The contents are not final until the kernels already
///                  issued have completed. When failure_bits_ is non-null, a
///                  consumer must check it on the same ordered stream or after
///                  completion. Any nonzero value invalidates all detections.
///
/// Example input: a Detector initialized with max_markers_ = 1024
/// Example output: capacity_ = 1024 and corner_x_ holding 4096 elements
struct DeviceDetections {
    /// Matched dictionary IDs.
    std::int32_t* ids_ = nullptr;
    /// Matched rotations, 0 through 3. The corners already have them undone.
    std::int32_t* rotations_ = nullptr;
    /// Corner x coordinates. The index is (corner * capacity_) + detection.
    float* corner_x_ = nullptr;
    /// Corner y coordinates, laid out like corner_x_.
    float* corner_y_ = nullptr;
    /// Index of the candidate each detection came from.
    std::int32_t* source_ = nullptr;
    /// Detection count after truncation. One element, on the device, so the host
    /// cannot read it directly.
    std::int32_t* count_ = nullptr;
    /// Diagnostic accepted count. With a zero failure word, a value larger than
    /// count_ indicates marker truncation. Nonzero failure bits invalidate the result
    /// regardless of either counter.
    std::int32_t* accepted_total_ = nullptr;
    int capacity_ = 0;
    /// Device diagnostic scalar containing DeviceFailure bits. Rewritten on every
    /// invocation, including blank frames. Zero means no reported failure; any
    /// nonzero value invalidates the entire invocation and count_ is zero.
    /// Marker-limit truncation is separate: accepted_total_ may exceed count_
    /// while failure_bits_ is zero. The detector owns this scalar for both routes.
    std::int32_t* failure_bits_ = nullptr;
};

/// Detection results pulled out to the host.
///
/// Purpose:
///   Copies the device-resident results into a form the host can work with. The copy
///   synchronizes exactly once, at that moment.
///
/// Ownership: holds values only and references no external resource. It may be copied
///            and kept.
/// Synchronization: this is only a set of values and holds no synchronization point.
///
/// Example input: the result of download() with two detections
/// Example output: ids_ with 2 elements and corners_ with 16 elements
struct HostDetections {
    std::vector<std::int32_t> ids_;
    /// Corners in the order x0, y0, x1, y1, x2, y2, x3, y3, so 8 elements per detection.
    std::vector<float> corners_;
    std::vector<std::int32_t> rotations_;
    /// Detection count before truncation. If it is larger than the size of ids_,
    /// detections were dropped.
    std::int32_t accepted_total_ = 0;
    /// Whether the results were truncated at the limit.
    bool marker_overflow_ = false;
};

}  // namespace aruco3cuda

#endif  // ARUCO3CUDA_DETECTIONS_HPP
