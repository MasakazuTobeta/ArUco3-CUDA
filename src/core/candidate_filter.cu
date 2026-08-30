// SPDX-License-Identifier: Apache-2.0
#include "candidate_filter.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/workspace.hpp"
#include "cuda_check.hpp"
#include "labeling.hpp"
#include "quad_extract.hpp"
#include "scan.hpp"

namespace aruco3cuda::detail {
namespace {

constexpr std::size_t kPlaneAlignment = 256U;
constexpr int kLinearThreads = 256;
/// How far from an edge a pixel may lie and still count as supporting it.
///
/// A binarized contour wobbles by a pixel, so looking only directly on the edge
/// undercounts. Measurements on synthetic shapes confirm that including pixels
/// up to 2 away gives shapes that should pass a support of at least 2.5 times
/// the chain code length.
constexpr float kEdgeSupportBandPx = 2.0F;

/// Values the filter decision needs, bundled together to keep the argument count down.
struct FilterCriteria {
    int min_perimeter_ = 0;
    int max_perimeter_ = 0;
    float min_corner_distance_rate_ = 0.0F;
    int min_distance_to_border_px_ = 0;
    float min_inlier_ratio_ = 0.0F;
    float min_edge_support_ratio_ = 0.0F;
    int width_px_ = 0;
    int height_px_ = 0;
};

/// Chain code length of a line segment. Equals the number of steps taken with 8-connectivity.
__device__ std::int32_t chain_length(std::int32_t x0, std::int32_t y0, std::int32_t x1,
                                     std::int32_t y1) {
    const std::int32_t dx = (x1 > x0) ? (x1 - x0) : (x0 - x1);
    const std::int32_t dy = (y1 > y0) ? (y1 - y0) : (y0 - y1);
    return (dx > dy) ? dx : dy;
}

/// Whether a point lies inside a convex quadrilateral. A point on an edge counts as inside.
__device__ bool inside_quad(const std::int32_t* xs, const std::int32_t* ys, std::int32_t px,
                            std::int32_t py) {
    bool has_positive = false;
    bool has_negative = false;
    for (int i = 0; i < kQuadCornerCount; ++i) {
        const int next = (i + 1) % kQuadCornerCount;
        const long long cross =
                (static_cast<long long>(xs[next] - xs[i]) * static_cast<long long>(py - ys[i])) -
                (static_cast<long long>(ys[next] - ys[i]) * static_cast<long long>(px - xs[i]));
        if (cross > 0) {
            has_positive = true;
        } else if (cross < 0) {
            has_negative = true;
        }
    }
    return !(has_positive && has_negative);
}

/// Whether a point lies within band of a segment and projects onto the segment itself.
__device__ bool near_segment(std::int32_t x0, std::int32_t y0, std::int32_t x1, std::int32_t y1,
                             std::int32_t px, std::int32_t py, float band) {
    const float ex = static_cast<float>(x1 - x0);
    const float ey = static_cast<float>(y1 - y0);
    const float length_squared = (ex * ex) + (ey * ey);
    if (length_squared <= 0.0F) {
        return false;
    }
    const float dx = static_cast<float>(px - x0);
    const float dy = static_cast<float>(py - y0);
    const float projection = ((dx * ex) + (dy * ey)) / length_squared;
    if (projection < 0.0F || projection > 1.0F) {
        return false;
    }
    const float distance = fabsf((dx * ey) - (dy * ex)) / sqrtf(length_squared);
    return distance <= band;
}

/// Fills the verdicts and the scratch space with 0.
///
/// The range beyond the label count is zeroed too. The scan walks the whole
/// allocation, so a nonzero value in the unused range would shift the write
/// positions.
__global__ void reset_filter_kernel(std::int32_t* accepted, std::int32_t* offsets,
                                    std::int32_t* inside_count, std::int32_t* perimeter,
                                    std::int32_t* edge_support, int capacity) {
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (index >= capacity) {
        return;
    }
    accepted[index] = 0;
    offsets[index] = 0;
    inside_count[index] = 0;
    perimeter[index] = 0;
    for (int corner = 0; corner < kQuadCornerCount; ++corner) {
        edge_support[(corner * capacity) + index] = 0;
    }
}

/// Counts the component pixels inside the estimated quadrilateral and those supporting each edge.
__global__ void count_inside_kernel(const std::int32_t* labels, const std::int32_t* corner_x,
                                    const std::int32_t* corner_y, const std::uint8_t* valid,
                                    std::int32_t* inside_count, std::int32_t* edge_support,
                                    int capacity, int width, int height) {
    const int x = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const int y = static_cast<int>((blockIdx.y * blockDim.y) + threadIdx.y);
    if (x >= width || y >= height) {
        return;
    }
    const std::int32_t label = labels[(y * width) + x];
    if (label == kBackgroundLabel || valid[label] == 0U) {
        return;
    }
    std::int32_t xs[kQuadCornerCount];
    std::int32_t ys[kQuadCornerCount];
    for (int corner = 0; corner < kQuadCornerCount; ++corner) {
        xs[corner] = corner_x[(corner * capacity) + label];
        ys[corner] = corner_y[(corner * capacity) + label];
    }
    if (inside_quad(xs, ys, x, y)) {
        atomicAdd(&inside_count[label], 1);
    }
    for (int corner = 0; corner < kQuadCornerCount; ++corner) {
        const int next = (corner + 1) % kQuadCornerCount;
        if (near_segment(xs[corner], ys[corner], xs[next], ys[next], x, y, kEdgeSupportBandPx)) {
            atomicAdd(&edge_support[(corner * capacity) + label], 1);
        }
    }
}

/// Runs the filter decision for one label.
__global__ void evaluate_kernel(const std::int32_t* corner_x, const std::int32_t* corner_y,
                                const std::uint8_t* valid, const std::int32_t* pixel_count,
                                const std::int32_t* inside_count, const std::int32_t* edge_support,
                                std::int32_t* accepted, std::int32_t* offsets,
                                std::int32_t* perimeter, int capacity, FilterCriteria criteria,
                                const std::int32_t* label_count) {
    const int label = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (label >= *label_count) {
        return;
    }
    if (valid[label] == 0U) {
        return;
    }
    std::int32_t xs[kQuadCornerCount];
    std::int32_t ys[kQuadCornerCount];
    for (int corner = 0; corner < kQuadCornerCount; ++corner) {
        xs[corner] = corner_x[(corner * capacity) + label];
        ys[corner] = corner_y[(corner * capacity) + label];
    }

    std::int32_t total_length = 0;
    long long min_side_squared = -1;
    for (int i = 0; i < kQuadCornerCount; ++i) {
        const int next = (i + 1) % kQuadCornerCount;
        total_length += chain_length(xs[i], ys[i], xs[next], ys[next]);
        const long long dx = xs[next] - xs[i];
        const long long dy = ys[next] - ys[i];
        const long long squared = (dx * dx) + (dy * dy);
        if (min_side_squared < 0 || squared < min_side_squared) {
            min_side_squared = squared;
        }
    }
    perimeter[label] = total_length;

    if (total_length < criteria.min_perimeter_ || total_length > criteria.max_perimeter_) {
        return;
    }
    // As in the CPU path, side length is judged as a ratio of the perimeter.
    const float min_corner_distance =
            static_cast<float>(total_length) * criteria.min_corner_distance_rate_;
    if (static_cast<float>(min_side_squared) < (min_corner_distance * min_corner_distance)) {
        return;
    }
    for (int corner = 0; corner < kQuadCornerCount; ++corner) {
        if (xs[corner] < criteria.min_distance_to_border_px_ ||
            ys[corner] < criteria.min_distance_to_border_px_ ||
            xs[corner] > (criteria.width_px_ - 1 - criteria.min_distance_to_border_px_) ||
            ys[corner] > (criteria.height_px_ - 1 - criteria.min_distance_to_border_px_)) {
            return;
        }
    }
    const std::int32_t pixels = pixel_count[label];
    if (pixels <= 0) {
        return;
    }
    const float ratio = static_cast<float>(inside_count[label]) / static_cast<float>(pixels);
    if (ratio < criteria.min_inlier_ratio_) {
        return;
    }
    // Require all four edges to be supported by the component. If even one
    // edge runs outside the component, the shape is not a quadrilateral.
    for (int corner = 0; corner < kQuadCornerCount; ++corner) {
        const int next = (corner + 1) % kQuadCornerCount;
        const std::int32_t chain = chain_length(xs[corner], ys[corner], xs[next], ys[next]);
        if (chain <= 0) {
            return;
        }
        const float support = static_cast<float>(edge_support[(corner * capacity) + label]) /
                              static_cast<float>(chain);
        if (support < criteria.min_edge_support_ratio_) {
            return;
        }
    }
    accepted[label] = 1;
    offsets[label] = 1;
}

/// Decides where writing starts: the current count when appending, otherwise 0.
__global__ void capture_base_kernel(const std::int32_t* count, std::int32_t* base, bool append) {
    if (threadIdx.x != 0U || blockIdx.x != 0U) {
        return;
    }
    *base = append ? *count : 0;
}

/// Packs the accepted labels into a dense array.
__global__ void compact_kernel(const std::int32_t* corner_x, const std::int32_t* corner_y,
                               const std::int32_t* accepted, const std::int32_t* offsets,
                               const std::int32_t* perimeter, const std::int32_t* base,
                               int label_capacity, DeviceCandidates candidates, int capacity) {
    const int label = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (label >= label_capacity) {
        return;
    }
    if (accepted[label] == 0) {
        return;
    }
    const std::int32_t destination = *base + offsets[label];
    if (destination >= capacity) {
        // Anything past the limit is not written. accepted_total_ reveals the truncation.
        return;
    }
    for (int corner = 0; corner < kQuadCornerCount; ++corner) {
        candidates.corner_x_[(corner * capacity) + destination] =
                corner_x[(corner * label_capacity) + label];
        candidates.corner_y_[(corner * capacity) + destination] =
                corner_y[(corner * label_capacity) + label];
    }
    candidates.label_[destination] = label;
    candidates.perimeter_[destination] = perimeter[label];
}

/// Writes out the packed candidate count and the total that passed the filter.
__global__ void store_count_kernel(const std::int32_t* total, const std::int32_t* base,
                                   std::int32_t* count, std::int32_t* accepted_total, int capacity,
                                   bool append) {
    if (threadIdx.x != 0U || blockIdx.x != 0U) {
        return;
    }
    const std::int32_t previous = append ? *accepted_total : 0;
    const std::int32_t value = previous + *total;
    *accepted_total = value;
    const std::int32_t written = *base + *total;
    *count = (written < capacity) ? written : capacity;
}

/// Returns the byte count of an int32 array. Returns 0 on overflow.
std::size_t array_bytes_i32(std::size_t count) {
    if (count == 0U || count > std::numeric_limits<std::size_t>::max() / sizeof(std::int32_t)) {
        return 0U;
    }
    return align_up(count * sizeof(std::int32_t), kPlaneAlignment);
}

}  // namespace

std::size_t candidate_workspace_bytes(const DetectorConfig& config, int width_px, int height_px) {
    const int label_capacity = max_label_count(width_px, height_px);
    if (label_capacity == 0 || config.max_candidates_ <= 0) {
        return 0U;
    }
    const auto labels = static_cast<std::size_t>(label_capacity);
    const auto candidates = static_cast<std::size_t>(config.max_candidates_);
    if (candidates > std::numeric_limits<std::size_t>::max() / kQuadCornerCount) {
        return 0U;
    }
    const std::size_t label_arrays = (array_bytes_i32(labels) * 4U) +
                                     array_bytes_i32(labels * kQuadCornerCount) +
                                     array_bytes_i32(1U);
    const std::size_t scan_bytes = align_up(scan_workspace_bytes(label_capacity), kPlaneAlignment);
    const std::size_t corner_arrays = array_bytes_i32(candidates * kQuadCornerCount) * 2U;
    const std::size_t candidate_arrays = array_bytes_i32(candidates) * 2U;
    const std::size_t counters = array_bytes_i32(1U) * 2U;
    if (label_arrays == 0U || scan_bytes == 0U || corner_arrays == 0U || candidate_arrays == 0U ||
        counters == 0U) {
        return 0U;
    }
    return label_arrays + scan_bytes + corner_arrays + candidate_arrays + counters;
}

Status reserve_candidates(const DetectorConfig& config, int width_px, int height_px,
                          Workspace& workspace, CandidateFilterBuffers* out_buffers,
                          DeviceCandidates* out_candidates) {
    if (out_buffers == nullptr || out_candidates == nullptr) {
        return Status::kInvalidArgument;
    }
    if (width_px <= 0 || height_px <= 0) {
        return Status::kInvalidArgument;
    }
    const int label_capacity = max_label_count(width_px, height_px);
    if (label_capacity == 0 || config.max_candidates_ <= 0 ||
        candidate_workspace_bytes(config, width_px, height_px) == 0U) {
        return Status::kInvalidConfig;
    }

    CandidateFilterBuffers buffers;
    buffers.capacity_ = label_capacity;
    const auto labels = static_cast<std::size_t>(label_capacity);
    std::int32_t** label_targets[] = {&buffers.accepted_, &buffers.offsets_, &buffers.inside_count_,
                                      &buffers.perimeter_};
    for (std::int32_t** target : label_targets) {
        void* pointer = nullptr;
        const Status status =
                workspace.allocate(labels * sizeof(std::int32_t), kPlaneAlignment, &pointer);
        if (status != Status::kOk) {
            return status;
        }
        *target = static_cast<std::int32_t*>(pointer);
    }
    {
        void* pointer = nullptr;
        const Status status = workspace.allocate(labels * kQuadCornerCount * sizeof(std::int32_t),
                                                 kPlaneAlignment, &pointer);
        if (status != Status::kOk) {
            return status;
        }
        buffers.edge_support_ = static_cast<std::int32_t*>(pointer);
    }
    {
        void* pointer = nullptr;
        const Status status = workspace.allocate(sizeof(std::int32_t), kPlaneAlignment, &pointer);
        if (status != Status::kOk) {
            return status;
        }
        buffers.base_ = static_cast<std::int32_t*>(pointer);
    }
    Status status = reserve_scan(label_capacity, workspace, &buffers.scan_);
    if (status != Status::kOk) {
        return status;
    }

    DeviceCandidates candidates;
    candidates.capacity_ = config.max_candidates_;
    const auto capacity = static_cast<std::size_t>(config.max_candidates_);
    std::int32_t** corner_targets[] = {&candidates.corner_x_, &candidates.corner_y_};
    for (std::int32_t** target : corner_targets) {
        void* pointer = nullptr;
        status = workspace.allocate(capacity * kQuadCornerCount * sizeof(std::int32_t),
                                    kPlaneAlignment, &pointer);
        if (status != Status::kOk) {
            return status;
        }
        *target = static_cast<std::int32_t*>(pointer);
    }
    std::int32_t** plain_targets[] = {&candidates.label_, &candidates.perimeter_,
                                      &candidates.count_, &candidates.accepted_total_};
    const std::size_t plain_counts[] = {capacity, capacity, 1U, 1U};
    for (std::size_t i = 0; i < 4U; ++i) {
        void* pointer = nullptr;
        status = workspace.allocate(plain_counts[i] * sizeof(std::int32_t), kPlaneAlignment,
                                    &pointer);
        if (status != Status::kOk) {
            return status;
        }
        *plain_targets[i] = static_cast<std::int32_t*>(pointer);
    }

    *out_buffers = buffers;
    *out_candidates = candidates;
    return Status::kOk;
}

Status build_candidates_async(const LabelBuffers& labels, const LabelStatisticsBuffers& stats,
                              const QuadBuffers& quads, const DetectorConfig& config,
                              CandidateFilterBuffers* buffers, DeviceCandidates* candidates,
                              bool append, cudaStream_t stream) {
    if (buffers == nullptr || candidates == nullptr || buffers->accepted_ == nullptr ||
        buffers->edge_support_ == nullptr || buffers->base_ == nullptr ||
        candidates->corner_x_ == nullptr || labels.labels_ == nullptr ||
        stats.pixel_count_ == nullptr || quads.corner_x_ == nullptr) {
        return Status::kInvalidArgument;
    }
    if (buffers->capacity_ != quads.capacity_ || quads.capacity_ != stats.capacity_) {
        return Status::kInvalidArgument;
    }

    const int width = labels.width_px_;
    const int height = labels.height_px_;
    const int label_capacity = buffers->capacity_;
    const int longest = (width > height) ? width : height;

    FilterCriteria criteria;
    criteria.min_perimeter_ =
            static_cast<int>(config.min_marker_perimeter_rate_ * static_cast<double>(longest));
    if (config.use_aruco3_detection_ && config.min_side_length_canonical_img_px_ != 0) {
        criteria.min_perimeter_ = 4 * config.min_side_length_canonical_img_px_;
    }
    criteria.max_perimeter_ =
            static_cast<int>(config.max_marker_perimeter_rate_ * static_cast<double>(longest));
    criteria.min_corner_distance_rate_ = static_cast<float>(config.min_corner_distance_rate_);
    criteria.min_distance_to_border_px_ = config.min_distance_to_border_px_;
    criteria.min_inlier_ratio_ = static_cast<float>(config.min_quad_inlier_ratio_);
    criteria.min_edge_support_ratio_ = static_cast<float>(config.min_edge_support_ratio_);
    criteria.width_px_ = width;
    criteria.height_px_ = height;

    const int linear_blocks = (label_capacity + kLinearThreads - 1) / kLinearThreads;
    const auto linear_grid = static_cast<unsigned int>(linear_blocks);
    const auto linear_block = static_cast<unsigned int>(kLinearThreads);
    const dim3 block(16U, 16U, 1U);
    const dim3 grid((static_cast<unsigned int>(width) + block.x - 1U) / block.x,
                    (static_cast<unsigned int>(height) + block.y - 1U) / block.y, 1U);

    reset_filter_kernel<<<linear_grid, linear_block, 0, stream>>>(
            buffers->accepted_, buffers->offsets_, buffers->inside_count_, buffers->perimeter_,
            buffers->edge_support_, label_capacity);
    Status status = check_kernel_launch("candidate.reset_filter_kernel", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    count_inside_kernel<<<grid, block, 0, stream>>>(
            labels.labels_, quads.corner_x_, quads.corner_y_, quads.valid_, buffers->inside_count_,
            buffers->edge_support_, label_capacity, width, height);
    status = check_kernel_launch("candidate.count_inside_kernel", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    evaluate_kernel<<<linear_grid, linear_block, 0, stream>>>(
            quads.corner_x_, quads.corner_y_, quads.valid_, stats.pixel_count_,
            buffers->inside_count_, buffers->edge_support_, buffers->accepted_, buffers->offsets_,
            buffers->perimeter_, label_capacity, criteria, labels.label_count_);
    status = check_kernel_launch("candidate.evaluate_kernel", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    status = exclusive_scan_async(buffers->offsets_, label_capacity, &buffers->scan_, stream);
    if (status != Status::kOk) {
        return status;
    }
    capture_base_kernel<<<1U, 1U, 0, stream>>>(candidates->count_, buffers->base_, append);
    status = check_kernel_launch("candidate.capture_base_kernel", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    compact_kernel<<<linear_grid, linear_block, 0, stream>>>(
            quads.corner_x_, quads.corner_y_, buffers->accepted_, buffers->offsets_,
            buffers->perimeter_, buffers->base_, label_capacity, *candidates,
            candidates->capacity_);
    status = check_kernel_launch("candidate.compact_kernel", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    store_count_kernel<<<1U, 1U, 0, stream>>>(buffers->scan_.total_, buffers->base_,
                                              candidates->count_, candidates->accepted_total_,
                                              candidates->capacity_, append);
    return check_kernel_launch("candidate.store_count_kernel", -1, false, stream);
}

Status read_candidate_count(const DeviceCandidates& candidates, int* out_count,
                            cudaStream_t stream) {
    if (out_count == nullptr || candidates.count_ == nullptr ||
        candidates.accepted_total_ == nullptr) {
        return Status::kInvalidArgument;
    }
    std::int32_t values[2] = {0, 0};
    Status status = check_cuda(cudaMemcpyAsync(&values[0], candidates.count_, sizeof(std::int32_t),
                                               cudaMemcpyDeviceToHost, stream),
                               "cudaMemcpyAsync", "candidate.read_count", -1, stream);
    if (status != Status::kOk) {
        return status;
    }
    status = check_cuda(cudaMemcpyAsync(&values[1], candidates.accepted_total_,
                                        sizeof(std::int32_t), cudaMemcpyDeviceToHost, stream),
                        "cudaMemcpyAsync", "candidate.read_total", -1, stream);
    if (status != Status::kOk) {
        return status;
    }
    status = check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize",
                        "candidate.read_count", -1, stream);
    if (status != Status::kOk) {
        return status;
    }
    *out_count = static_cast<int>(values[0]);
    // Always report truncation to the caller. Dropping it silently would make
    // the cause impossible to isolate.
    return (values[1] > values[0]) ? Status::kCandidateOverflow : Status::kOk;
}

}  // namespace aruco3cuda::detail
