// SPDX-License-Identifier: Apache-2.0
#include "detection_emit.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/workspace.hpp"
#include "candidate_filter.hpp"
#include "candidate_tree.hpp"
#include "cuda_check.hpp"
#include "dictionary_match.hpp"
#include "quad_extract.hpp"
#include "scan.hpp"

namespace aruco3cuda::detail {
namespace {

constexpr std::size_t kPlaneAlignment = 256U;
constexpr int kEmitThreads = 256;

/// Returns the byte count of one plane. Returns 0 on overflow.
std::size_t plane_bytes(std::size_t count, std::size_t element_bytes) {
    if (count != 0U && element_bytes > std::numeric_limits<std::size_t>::max() / count) {
        return 0U;
    }
    return align_up(count * element_bytes, kPlaneAlignment);
}

/// Writes the verdict for each candidate. The candidate count lives only on the
/// device, so the whole capacity is filled.
__global__ void predicate_kernel(const std::int32_t* ids, const std::int32_t* depth,
                                 const std::int32_t* stop_depth, const std::int32_t* count,
                                 std::int32_t* offsets, int capacity) {
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (index >= capacity) {
        return;
    }
    // Write an explicit 0 outside the range too, so values from the previous frame do not linger.
    if (index >= *count) {
        offsets[index] = 0;
        return;
    }
    const bool reached = depth[index] < *stop_depth;
    const bool identified = ids[index] >= 0;
    offsets[index] = (reached && identified) ? 1 : 0;
}

/// Packs the accepted candidates and writes them out.
__global__ void emit_kernel(const DeviceCandidates grouped, const MatchBuffers matches,
                            const std::int32_t* depth, const std::int32_t* stop_depth,
                            const std::int32_t* offsets, const std::int32_t* count,
                            DeviceDetections out) {
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (index >= *count) {
        return;
    }
    const std::int32_t rotation = matches.rotations_[index];
    if (depth[index] >= *stop_depth || matches.ids_[index] < 0) {
        return;
    }
    const std::int32_t destination = offsets[index];
    if (destination >= out.capacity_) {
        return;
    }

    out.ids_[destination] = matches.ids_[index];
    out.rotations_[destination] = rotation;
    out.source_[destination] = index;
    for (int corner = 0; corner < kQuadCornerCount; ++corner) {
        // Undo the rotation found by matching. OpenCV's correctCornerPosition
        // is std::rotate(begin, begin + 4 - rot, end), which is the same as
        // new[i] = old[(i + 4 - rot) % 4].
        const int source = (corner + kQuadCornerCount - rotation) % kQuadCornerCount;
        out.corner_x_[(corner * out.capacity_) + destination] =
                static_cast<float>(grouped.corner_x_[(source * grouped.capacity_) + index]);
        out.corner_y_[(corner * out.capacity_) + destination] =
                static_cast<float>(grouped.corner_y_[(source * grouped.capacity_) + index]);
    }
}

/// Writes out the detection count.
__global__ void store_detection_count_kernel(const std::int32_t* total, std::int32_t* count,
                                             std::int32_t* accepted_total, int capacity) {
    if (threadIdx.x != 0U || blockIdx.x != 0U) {
        return;
    }
    const std::int32_t value = *total;
    *accepted_total = value;
    *count = (value > capacity) ? capacity : value;
}

}  // namespace

std::size_t detection_workspace_bytes(const DetectorConfig& config) {
    if (config.max_candidates_ <= 0 || config.max_markers_ <= 0 ||
        config.max_markers_ > config.max_candidates_) {
        return 0U;
    }
    const auto candidates = static_cast<std::size_t>(config.max_candidates_);
    const auto markers = static_cast<std::size_t>(config.max_markers_);

    const std::size_t offsets = plane_bytes(candidates, sizeof(std::int32_t));
    const std::size_t scan = scan_workspace_bytes(config.max_candidates_);
    const std::size_t int_planes = plane_bytes(markers, sizeof(std::int32_t)) * 3U;
    const std::size_t corner_planes = plane_bytes(markers * kQuadCornerCount, sizeof(float)) * 2U;
    const std::size_t scalars = plane_bytes(1U, sizeof(std::int32_t)) * 2U;
    if (offsets == 0U || scan == 0U || int_planes == 0U || corner_planes == 0U || scalars == 0U) {
        return 0U;
    }
    return offsets + scan + int_planes + corner_planes + scalars;
}

Status reserve_detections(const DetectorConfig& config, Workspace& workspace,
                          DetectionEmitBuffers* out_buffers, DeviceDetections* out) {
    if (out_buffers == nullptr || out == nullptr) {
        return Status::kInvalidArgument;
    }
    if (detection_workspace_bytes(config) == 0U) {
        return Status::kInvalidConfig;
    }
    const auto candidates = static_cast<std::size_t>(config.max_candidates_);
    const auto markers = static_cast<std::size_t>(config.max_markers_);

    DetectionEmitBuffers buffers;
    buffers.capacity_ = config.max_candidates_;
    void* pointer = nullptr;
    Status status =
            workspace.allocate(candidates * sizeof(std::int32_t), kPlaneAlignment, &pointer);
    if (status != Status::kOk) {
        return status;
    }
    buffers.offsets_ = static_cast<std::int32_t*>(pointer);
    // The scan runs in candidate space. Passing the detection limit would
    // leave the tail of the predicate unscanned.
    status = reserve_scan(config.max_candidates_, workspace, &buffers.scan_);
    if (status != Status::kOk) {
        return status;
    }

    DeviceDetections detections;
    detections.capacity_ = config.max_markers_;
    std::int32_t** int_targets[] = {&detections.ids_, &detections.rotations_, &detections.source_};
    for (std::int32_t** target : int_targets) {
        status = workspace.allocate(markers * sizeof(std::int32_t), kPlaneAlignment, &pointer);
        if (status != Status::kOk) {
            return status;
        }
        *target = static_cast<std::int32_t*>(pointer);
    }
    float** corner_targets[] = {&detections.corner_x_, &detections.corner_y_};
    for (float** target : corner_targets) {
        status = workspace.allocate(markers * kQuadCornerCount * sizeof(float), kPlaneAlignment,
                                    &pointer);
        if (status != Status::kOk) {
            return status;
        }
        *target = static_cast<float*>(pointer);
    }
    std::int32_t** scalar_targets[] = {&detections.count_, &detections.accepted_total_};
    for (std::int32_t** target : scalar_targets) {
        status = workspace.allocate(sizeof(std::int32_t), kPlaneAlignment, &pointer);
        if (status != Status::kOk) {
            return status;
        }
        *target = static_cast<std::int32_t*>(pointer);
    }

    *out_buffers = buffers;
    *out = detections;
    return Status::kOk;
}

Status emit_detections_async(const DeviceCandidates& grouped, const MatchBuffers& matches,
                             const CandidateTreeBuffers& tree, DetectionEmitBuffers* buffers,
                             DeviceDetections* out, cudaStream_t stream) {
    if (buffers == nullptr || out == nullptr || buffers->offsets_ == nullptr ||
        out->ids_ == nullptr || out->count_ == nullptr || grouped.count_ == nullptr ||
        matches.ids_ == nullptr || tree.depth_ == nullptr || tree.stop_depth_ == nullptr) {
        return Status::kInvalidArgument;
    }
    if (buffers->capacity_ < grouped.capacity_ || matches.capacity_ < grouped.capacity_ ||
        tree.capacity_ < grouped.capacity_) {
        return Status::kInvalidArgument;
    }

    const auto capacity = static_cast<unsigned int>(buffers->capacity_);
    const unsigned int blocks = (capacity + static_cast<unsigned int>(kEmitThreads) - 1U) /
                                static_cast<unsigned int>(kEmitThreads);
    predicate_kernel<<<blocks, static_cast<unsigned int>(kEmitThreads), 0, stream>>>(
            matches.ids_, tree.depth_, tree.stop_depth_, grouped.count_, buffers->offsets_,
            buffers->capacity_);
    Status status = check_kernel_launch("detection_emit.predicate_kernel", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }

    status = exclusive_scan_async(buffers->offsets_, buffers->capacity_, &buffers->scan_, stream);
    if (status != Status::kOk) {
        return status;
    }

    emit_kernel<<<blocks, static_cast<unsigned int>(kEmitThreads), 0, stream>>>(
            grouped, matches, tree.depth_, tree.stop_depth_, buffers->offsets_, grouped.count_,
            *out);
    status = check_kernel_launch("detection_emit.emit_kernel", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }

    store_detection_count_kernel<<<1U, 1U, 0, stream>>>(buffers->scan_.total_, out->count_,
                                                        out->accepted_total_, out->capacity_);
    return check_kernel_launch("detection_emit.store_detection_count_kernel", -1, false, stream);
}

Status read_detection_count(const DeviceDetections& detections, int* out_count,
                            cudaStream_t stream) {
    if (out_count == nullptr || detections.count_ == nullptr ||
        detections.accepted_total_ == nullptr) {
        return Status::kInvalidArgument;
    }
    std::int32_t values[2] = {0, 0};
    Status status = check_cuda(cudaMemcpyAsync(&values[0], detections.count_, sizeof(std::int32_t),
                                               cudaMemcpyDeviceToHost, stream),
                               "cudaMemcpyAsync", "detection.read_count", -1, stream);
    if (status != Status::kOk) {
        return status;
    }
    status = check_cuda(cudaMemcpyAsync(&values[1], detections.accepted_total_,
                                        sizeof(std::int32_t), cudaMemcpyDeviceToHost, stream),
                        "cudaMemcpyAsync", "detection.read_total", -1, stream);
    if (status != Status::kOk) {
        return status;
    }
    status = check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize",
                        "detection.read_count", -1, stream);
    if (status != Status::kOk) {
        return status;
    }
    *out_count = static_cast<int>(values[0]);
    // Always report truncation to the caller. Dropping it silently would make
    // the cause impossible to isolate.
    return (values[1] > values[0]) ? Status::kMarkerOverflow : Status::kOk;
}

}  // namespace aruco3cuda::detail
