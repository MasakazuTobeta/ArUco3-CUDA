// SPDX-License-Identifier: Apache-2.0
#include "candidate_group.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/workspace.hpp"
#include "candidate_filter.hpp"
#include "cuda_check.hpp"
#include "quad_extract.hpp"
#include "scan.hpp"

namespace aruco3cuda::detail {
namespace {

constexpr std::size_t kPlaneAlignment = 256U;
constexpr int kLinearThreads = 256;
/// Number of blocks in the kernel that walks candidate pairs.
///
/// The pair count is the square of the candidate count, and the candidate count
/// lives only on the device. Walking a fixed grid in grid-stride fashion avoids
/// launching useless threads when there are few candidates.
constexpr int kPairBlocks = 1024;

/// Walks to the root of the union-find over rank space.
__device__ std::int32_t find_root(const std::int32_t* parent, std::int32_t index) {
    while (parent[index] != index) {
        index = parent[index];
    }
    return index;
}

/// Merges two ranks into the same group. The smaller rank becomes the root.
__device__ void merge_ranks(std::int32_t* parent, std::int32_t a, std::int32_t b) {
    while (a != b) {
        a = find_root(parent, a);
        b = find_root(parent, b);
        if (a < b) {
            const std::int32_t previous = atomicMin(&parent[b], a);
            if (previous == b) {
                return;
            }
            b = previous;
        } else if (b < a) {
            const std::int32_t previous = atomicMin(&parent[a], b);
            if (previous == a) {
                return;
            }
            a = previous;
        }
    }
}

/// Mean corner distance between two candidates. Takes the smallest over the
/// four starting-vertex correspondences.
///
/// Same as getAverageDistance in the CPU path. Every correspondence is tried so
/// that the same marker is still recognized when the vertex order is rotated by
/// one.
__device__ float average_quad_distance(const std::int32_t* corner_x, const std::int32_t* corner_y,
                                       int capacity, std::int32_t first, std::int32_t second) {
    float minimum = 3.402823466e+38F;
    for (int fc = 0; fc < kQuadCornerCount; ++fc) {
        float total = 0.0F;
        for (int c = 0; c < kQuadCornerCount; ++c) {
            const int mod_c = (c + fc) % kQuadCornerCount;
            const float dx = static_cast<float>(corner_x[(mod_c * capacity) + first] -
                                                corner_x[(c * capacity) + second]);
            const float dy = static_cast<float>(corner_y[(mod_c * capacity) + first] -
                                                corner_y[(c * capacity) + second]);
            total += (dx * dx) + (dy * dy);
        }
        total /= static_cast<float>(kQuadCornerCount);
        if (total < minimum) {
            minimum = total;
        }
    }
    return sqrtf(minimum);
}

/// Initializes the scratch space.
__global__ void reset_group_kernel(CandidateGroupBuffers buffers, int capacity) {
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (index >= capacity) {
        return;
    }
    buffers.parent_[index] = index;
    buffers.selected_[index] = 0;
    buffers.offsets_[index] = 0;
    buffers.rank_[index] = 0;
    buffers.order_[index] = 0;
}

/// Assigns ranks in descending perimeter order.
///
/// Ties are broken by the smaller candidate index. This matches the order of
/// the CPU path's stable_sort and does not vary from run to run.
__global__ void rank_kernel(const std::int32_t* perimeter, std::int32_t* rank, std::int32_t* order,
                            const std::int32_t* count) {
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const int total = *count;
    if (index >= total) {
        return;
    }
    const std::int32_t own = perimeter[index];
    int position = 0;
    for (int other = 0; other < total; ++other) {
        const std::int32_t value = perimeter[other];
        if (value > own || (value == own && other < index)) {
            ++position;
        }
    }
    rank[index] = position;
    order[position] = index;
}

/// Merges nearby pairs into the same group.
__global__ void merge_pairs_kernel(const std::int32_t* corner_x, const std::int32_t* corner_y,
                                   const std::int32_t* perimeter, const std::int32_t* order,
                                   std::int32_t* parent, int capacity, float distance_rate,
                                   const std::int32_t* count) {
    const long long total = *count;
    const long long pairs = total * total;
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long index = (static_cast<long long>(blockIdx.x) * blockDim.x) + threadIdx.x;
         index < pairs; index += stride) {
        const long long a = index / total;
        const long long b = index % total;
        if (a >= b) {
            // Handle each pair once. The smaller rank is a.
            continue;
        }
        const std::int32_t first = order[a];
        const std::int32_t second = order[b];
        const float distance = average_quad_distance(corner_x, corner_y, capacity, first, second);
        // Judge against the perimeter of the later-ranked side (the smaller perimeter).
        const float threshold = static_cast<float>(perimeter[second]) * distance_rate;
        if (distance < threshold) {
            merge_ranks(parent, static_cast<std::int32_t>(a), static_cast<std::int32_t>(b));
        }
    }
}

/// Resolves the roots to pick the representatives.
__global__ void select_kernel(std::int32_t* parent, std::int32_t* selected, std::int32_t* offsets,
                              const std::int32_t* count) {
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (index >= *count) {
        return;
    }
    const std::int32_t root = find_root(parent, index);
    parent[index] = root;
    const std::int32_t is_root = (root == index) ? 1 : 0;
    selected[index] = is_root;
    offsets[index] = is_root;
}

/// Packs the representatives and writes them to the output.
__global__ void emit_kernel(const DeviceCandidates input, DeviceCandidates output,
                            const std::int32_t* order, const std::int32_t* selected,
                            const std::int32_t* offsets, int capacity) {
    const int rank = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (rank >= capacity) {
        return;
    }
    if (selected[rank] == 0) {
        return;
    }
    const std::int32_t destination = offsets[rank];
    if (destination >= output.capacity_) {
        return;
    }
    const std::int32_t source = order[rank];
    for (int corner = 0; corner < kQuadCornerCount; ++corner) {
        output.corner_x_[(corner * output.capacity_) + destination] =
                input.corner_x_[(corner * input.capacity_) + source];
        output.corner_y_[(corner * output.capacity_) + destination] =
                input.corner_y_[(corner * input.capacity_) + source];
    }
    output.label_[destination] = input.label_[source];
    output.perimeter_[destination] = input.perimeter_[source];
}

/// Writes out the candidate count after merging.
__global__ void store_group_count_kernel(const std::int32_t* total, std::int32_t* count,
                                         std::int32_t* accepted_total, int capacity) {
    if (threadIdx.x != 0U || blockIdx.x != 0U) {
        return;
    }
    const std::int32_t value = *total;
    *accepted_total = value;
    *count = (value < capacity) ? value : capacity;
}

/// Returns the byte count of an int32 array. Returns 0 on overflow.
std::size_t array_bytes_i32(std::size_t count) {
    if (count == 0U || count > std::numeric_limits<std::size_t>::max() / sizeof(std::int32_t)) {
        return 0U;
    }
    return align_up(count * sizeof(std::int32_t), kPlaneAlignment);
}

}  // namespace

std::size_t candidate_group_workspace_bytes(const DetectorConfig& config) {
    if (config.max_candidates_ <= 0) {
        return 0U;
    }
    const auto capacity = static_cast<std::size_t>(config.max_candidates_);
    if (capacity > std::numeric_limits<std::size_t>::max() / kQuadCornerCount) {
        return 0U;
    }
    const std::size_t work = array_bytes_i32(capacity) * 5U;
    const std::size_t scan_bytes =
            align_up(scan_workspace_bytes(config.max_candidates_), kPlaneAlignment);
    const std::size_t corner_arrays = array_bytes_i32(capacity * kQuadCornerCount) * 2U;
    const std::size_t plain_arrays = array_bytes_i32(capacity) * 2U;
    const std::size_t counters = array_bytes_i32(1U) * 2U;
    if (work == 0U || scan_bytes == 0U || corner_arrays == 0U || plain_arrays == 0U ||
        counters == 0U) {
        return 0U;
    }
    return work + scan_bytes + corner_arrays + plain_arrays + counters;
}

Status reserve_candidate_groups(const DetectorConfig& config, Workspace& workspace,
                                CandidateGroupBuffers* out_buffers, DeviceCandidates* out_grouped) {
    if (out_buffers == nullptr || out_grouped == nullptr) {
        return Status::kInvalidArgument;
    }
    if (config.max_candidates_ <= 0 || candidate_group_workspace_bytes(config) == 0U) {
        return Status::kInvalidConfig;
    }

    CandidateGroupBuffers buffers;
    buffers.capacity_ = config.max_candidates_;
    const auto capacity = static_cast<std::size_t>(config.max_candidates_);
    std::int32_t** work_targets[] = {&buffers.rank_, &buffers.order_, &buffers.parent_,
                                     &buffers.selected_, &buffers.offsets_};
    for (std::int32_t** target : work_targets) {
        void* pointer = nullptr;
        const Status status =
                workspace.allocate(capacity * sizeof(std::int32_t), kPlaneAlignment, &pointer);
        if (status != Status::kOk) {
            return status;
        }
        *target = static_cast<std::int32_t*>(pointer);
    }
    Status status = reserve_scan(config.max_candidates_, workspace, &buffers.scan_);
    if (status != Status::kOk) {
        return status;
    }

    DeviceCandidates grouped;
    grouped.capacity_ = config.max_candidates_;
    std::int32_t** corner_targets[] = {&grouped.corner_x_, &grouped.corner_y_};
    for (std::int32_t** target : corner_targets) {
        void* pointer = nullptr;
        status = workspace.allocate(capacity * kQuadCornerCount * sizeof(std::int32_t),
                                    kPlaneAlignment, &pointer);
        if (status != Status::kOk) {
            return status;
        }
        *target = static_cast<std::int32_t*>(pointer);
    }
    std::int32_t** plain_targets[] = {&grouped.label_, &grouped.perimeter_, &grouped.count_,
                                      &grouped.accepted_total_};
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
    *out_grouped = grouped;
    return Status::kOk;
}

Status build_candidate_groups_async(const DeviceCandidates& input, const DetectorConfig& config,
                                    CandidateGroupBuffers* buffers, DeviceCandidates* grouped,
                                    cudaStream_t stream) {
    if (buffers == nullptr || grouped == nullptr || buffers->parent_ == nullptr ||
        grouped->corner_x_ == nullptr || input.corner_x_ == nullptr || input.count_ == nullptr) {
        return Status::kInvalidArgument;
    }
    if (buffers->capacity_ < input.capacity_ || grouped->capacity_ < input.capacity_) {
        return Status::kInvalidArgument;
    }

    const int capacity = buffers->capacity_;
    const int linear_blocks = (capacity + kLinearThreads - 1) / kLinearThreads;
    const auto linear_grid = static_cast<unsigned int>(linear_blocks);
    const auto linear_block = static_cast<unsigned int>(kLinearThreads);

    reset_group_kernel<<<linear_grid, linear_block, 0, stream>>>(*buffers, capacity);
    Status status = check_kernel_launch("group.reset_group_kernel", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    rank_kernel<<<linear_grid, linear_block, 0, stream>>>(input.perimeter_, buffers->rank_,
                                                          buffers->order_, input.count_);
    status = check_kernel_launch("group.rank_kernel", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    merge_pairs_kernel<<<static_cast<unsigned int>(kPairBlocks), linear_block, 0, stream>>>(
            input.corner_x_, input.corner_y_, input.perimeter_, buffers->order_, buffers->parent_,
            input.capacity_, static_cast<float>(config.min_marker_distance_rate_), input.count_);
    status = check_kernel_launch("group.merge_pairs_kernel", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    select_kernel<<<linear_grid, linear_block, 0, stream>>>(buffers->parent_, buffers->selected_,
                                                            buffers->offsets_, input.count_);
    status = check_kernel_launch("group.select_kernel", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    status = exclusive_scan_async(buffers->offsets_, capacity, &buffers->scan_, stream);
    if (status != Status::kOk) {
        return status;
    }
    emit_kernel<<<linear_grid, linear_block, 0, stream>>>(
            input, *grouped, buffers->order_, buffers->selected_, buffers->offsets_, capacity);
    status = check_kernel_launch("group.emit_kernel", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    store_group_count_kernel<<<1U, 1U, 0, stream>>>(buffers->scan_.total_, grouped->count_,
                                                    grouped->accepted_total_, grouped->capacity_);
    return check_kernel_launch("group.store_group_count_kernel", -1, false, stream);
}

}  // namespace aruco3cuda::detail
