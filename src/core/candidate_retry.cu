// SPDX-License-Identifier: Apache-2.0
#include "candidate_retry.hpp"

#include <climits>
#include <cmath>
#include <limits>

#include "cuda_check.hpp"
#include "quad_extract.hpp"

namespace aruco3cuda::detail {
namespace {
constexpr std::size_t kAlignment = 256U;
constexpr int kThreads = 256;

__device__ float distance(const DeviceCandidates raw, int first, int second) {
    float minimum = 3.402823466e+38F;
    for (int rotation = 0; rotation < 4; ++rotation) {
        float sum = 0.0F;
        for (int corner = 0; corner < 4; ++corner) {
            const int a = ((corner + rotation) % 4) * raw.capacity_ + first;
            const int b = corner * raw.capacity_ + second;
            const float dx = static_cast<float>(raw.corner_x_[a] - raw.corner_x_[b]);
            const float dy = static_cast<float>(raw.corner_y_[a] - raw.corner_y_[b]);
            sum += (dx * dx) + (dy * dy);
        }
        minimum = fminf(minimum, sum / 4.0F);
    }
    return sqrtf(minimum);
}

__global__ void reset_kernel(CandidateRetryBuffers retries) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < retries.capacity_) {
        retries.flags_[index] = 0;
        retries.offsets_[index] = 0;
        retries.chosen_[index] = INT_MAX;
    }
}

__global__ void choose_alternatives_kernel(DeviceCandidates raw, MatchBuffers primary,
                                           CandidateGroupBuffers groups,
                                           CandidateRetryBuffers retries, float minimum_distance,
                                           int module_count) {
    const int root = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = *raw.count_;
    if (root >= total || groups.selected_[root] == 0) {
        return;
    }
    const int owner = groups.offsets_[root];
    if (primary.ids_[owner] >= 0) {
        return;
    }
    int current = groups.order_[root];
    for (int rank = root + 1; rank < total; ++rank) {
        // Group compression completed in a preceding stream operation; these
        // parent values are immutable throughout preparation.
        if (groups.parent_[rank] != root) {
            continue;
        }
        const int candidate = groups.order_[rank];
        const float module =
                groups.quad_perimeter_[candidate] / (4.0F * static_cast<float>(module_count));
        if (distance(raw, candidate, current) > minimum_distance * module) {
            current = candidate;
            retries.flags_[rank] = owner + 1;
            retries.offsets_[rank] = 1;
        }
    }
}

__global__ void compact_kernel(DeviceCandidates raw, DeviceCandidates grouped,
                               CandidateGroupBuffers groups, CandidateRetryBuffers retries) {
    const int rank = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (rank >= *raw.count_ || retries.flags_[rank] == 0) {
        return;
    }
    const int destination = retries.offsets_[rank];
    const int owner = retries.flags_[rank] - 1;
    const int source = groups.order_[rank];
    for (int corner = 0; corner < 4; ++corner) {
        retries.candidates_.corner_x_[corner * retries.capacity_ + destination] =
                raw.corner_x_[corner * raw.capacity_ + source];
        retries.candidates_.corner_y_[corner * retries.capacity_ + destination] =
                raw.corner_y_[corner * raw.capacity_ + source];
    }
    retries.candidates_.label_[destination] = raw.label_[source];
    retries.candidates_.perimeter_[destination] = grouped.perimeter_[owner];
    retries.owners_[destination] = owner;
}

__global__ void count_kernel(CandidateRetryBuffers retries) {
    *retries.candidates_.count_ = *retries.scan_.total_;
    *retries.candidates_.accepted_total_ = *retries.scan_.total_;
}

__global__ void choose_success_kernel(CandidateRetryBuffers retries) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < *retries.candidates_.count_ && retries.matches_.ids_[index] >= 0) {
        atomicMin(&retries.chosen_[retries.owners_[index]], index);
    }
}

__global__ void resolve_kernel(CandidateRetryBuffers retries, DeviceCandidates grouped,
                               MatchBuffers primary) {
    const int owner = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (owner >= *grouped.count_ || primary.ids_[owner] >= 0) {
        return;
    }
    const int source = retries.chosen_[owner];
    if (source == INT_MAX) {
        return;
    }
    for (int corner = 0; corner < 4; ++corner) {
        grouped.corner_x_[corner * grouped.capacity_ + owner] =
                retries.candidates_.corner_x_[corner * retries.capacity_ + source];
        grouped.corner_y_[corner * grouped.capacity_ + owner] =
                retries.candidates_.corner_y_[corner * retries.capacity_ + source];
    }
    primary.ids_[owner] = retries.matches_.ids_[source];
    primary.rotations_[owner] = retries.matches_.rotations_[source];
    primary.distances_[owner] = retries.matches_.distances_[source];
}

std::size_t bytes(std::size_t count) {
    return align_up(count * sizeof(std::int32_t), kAlignment);
}
bool valid_candidates(const DeviceCandidates& candidates) {
    return candidates.capacity_ > 0 && candidates.capacity_ <= (INT_MAX - kThreads + 1) / 4 &&
           candidates.corner_x_ != nullptr && candidates.corner_y_ != nullptr &&
           candidates.label_ != nullptr && candidates.perimeter_ != nullptr &&
           candidates.count_ != nullptr && candidates.accepted_total_ != nullptr;
}

bool valid_matches(const MatchBuffers& matches, int capacity) {
    return matches.capacity_ >= capacity && matches.ids_ != nullptr &&
           matches.rotations_ != nullptr && matches.distances_ != nullptr;
}

bool valid_retries(const CandidateRetryBuffers& retries) {
    return retries.capacity_ > 0 && retries.capacity_ <= (INT_MAX - kThreads + 1) / 4 &&
           retries.candidates_.capacity_ == retries.capacity_ &&
           valid_candidates(retries.candidates_) &&
           valid_matches(retries.matches_, retries.capacity_) && retries.owners_ != nullptr &&
           retries.flags_ != nullptr && retries.offsets_ != nullptr && retries.chosen_ != nullptr &&
           retries.scan_.total_ != nullptr && retries.scan_.block_offsets_ != nullptr &&
           retries.scan_.capacity_ >= retries.capacity_ &&
           retries.scan_.block_count_ >= scan_block_count(retries.capacity_);
}
}  // namespace

std::size_t candidate_retry_workspace_bytes(const DetectorConfig& config) {
    if (config.max_candidates_ <= 0 || config.max_candidates_ > (INT_MAX - kThreads + 1) / 4) {
        return 0U;
    }
    const auto capacity = static_cast<std::size_t>(config.max_candidates_);
    if (capacity > std::numeric_limits<std::size_t>::max() / 128U) {
        return 0U;
    }
    return bytes(capacity) * 6U + bytes(4U * capacity) * 2U + bytes(1U) * 2U +
           align_up(scan_workspace_bytes(config.max_candidates_), kAlignment) +
           align_up(match_workspace_bytes(config), kAlignment);
}

Status reserve_candidate_retries(const DetectorConfig& config, Workspace& workspace,
                                 CandidateRetryBuffers* out) {
    if (out == nullptr) {
        return Status::kInvalidArgument;
    }
    if (candidate_retry_workspace_bytes(config) == 0U) {
        return Status::kInvalidConfig;
    }
    CandidateRetryBuffers result;
    result.capacity_ = config.max_candidates_;
    result.candidates_.capacity_ = config.max_candidates_;
    auto reserve = [&](std::int32_t** target, std::size_t count) {
        void* pointer = nullptr;
        const Status status =
                workspace.allocate(count * sizeof(std::int32_t), kAlignment, &pointer);
        *target = static_cast<std::int32_t*>(pointer);
        return status;
    };
    const auto capacity = static_cast<std::size_t>(config.max_candidates_);
    std::int32_t** targets[] = {&result.owners_,
                                &result.flags_,
                                &result.offsets_,
                                &result.chosen_,
                                &result.candidates_.label_,
                                &result.candidates_.perimeter_};
    for (auto** target : targets) {
        const Status status = reserve(target, capacity);
        if (status != Status::kOk)
            return status;
    }
    for (auto** target : {&result.candidates_.corner_x_, &result.candidates_.corner_y_}) {
        const Status status = reserve(target, capacity * 4U);
        if (status != Status::kOk)
            return status;
    }
    for (auto** target : {&result.candidates_.count_, &result.candidates_.accepted_total_}) {
        const Status status = reserve(target, 1U);
        if (status != Status::kOk)
            return status;
    }
    Status status = reserve_scan(config.max_candidates_, workspace, &result.scan_);
    if (status != Status::kOk)
        return status;
    status = reserve_matches(config, workspace, &result.matches_);
    if (status != Status::kOk)
        return status;
    *out = result;
    return Status::kOk;
}

Status prepare_candidate_retries_async(const DeviceCandidates& raw, const DeviceCandidates& grouped,
                                       const MatchBuffers& primary,
                                       const CandidateGroupBuffers& groups,
                                       const DetectorConfig& config, int marker_size,
                                       CandidateRetryBuffers* retries, cudaStream_t stream) {
    if (retries == nullptr || !valid_retries(*retries) || !valid_candidates(raw) ||
        !valid_candidates(grouped) || retries->capacity_ < raw.capacity_ ||
        grouped.capacity_ < raw.capacity_ || grouped.capacity_ > retries->capacity_ ||
        !valid_matches(primary, grouped.capacity_) || groups.capacity_ < raw.capacity_ ||
        groups.parent_ == nullptr || groups.quad_perimeter_ == nullptr || groups.rank_ == nullptr ||
        groups.order_ == nullptr || groups.selected_ == nullptr || groups.offsets_ == nullptr ||
        marker_size <= 0 || marker_size > 64 || config.marker_border_bits_ <= 0 ||
        config.marker_border_bits_ > 64 || !std::isfinite(config.min_group_distance_) ||
        config.min_group_distance_ < 0.0 ||
        config.min_group_distance_ > static_cast<double>(std::numeric_limits<float>::max())) {
        return Status::kInvalidArgument;
    }
    const unsigned int blocks =
            static_cast<unsigned int>((retries->capacity_ + kThreads - 1) / kThreads);
    reset_kernel<<<blocks, kThreads, 0, stream>>>(*retries);
    Status status = check_kernel_launch("retry.reset", -1, false, stream);
    if (status != Status::kOk)
        return status;
    choose_alternatives_kernel<<<blocks, kThreads, 0, stream>>>(
            raw, primary, groups, *retries, static_cast<float>(config.min_group_distance_),
            marker_size + 2 * config.marker_border_bits_);
    status = check_kernel_launch("retry.choose_alternatives", -1, false, stream);
    if (status != Status::kOk)
        return status;
    status = exclusive_scan_async(retries->offsets_, retries->capacity_, &retries->scan_, stream);
    if (status != Status::kOk)
        return status;
    compact_kernel<<<blocks, kThreads, 0, stream>>>(raw, grouped, groups, *retries);
    status = check_kernel_launch("retry.compact", -1, false, stream);
    if (status != Status::kOk)
        return status;
    count_kernel<<<1, 1, 0, stream>>>(*retries);
    return check_kernel_launch("retry.count", -1, false, stream);
}

Status resolve_candidate_retries_async(const CandidateRetryBuffers& retries,
                                       DeviceCandidates* grouped, MatchBuffers* primary,
                                       cudaStream_t stream) {
    if (grouped == nullptr || primary == nullptr || !valid_retries(retries) ||
        !valid_candidates(*grouped) || grouped->capacity_ > retries.capacity_ ||
        !valid_matches(*primary, grouped->capacity_)) {
        return Status::kInvalidArgument;
    }
    const unsigned int blocks =
            static_cast<unsigned int>((retries.capacity_ + kThreads - 1) / kThreads);
    choose_success_kernel<<<blocks, kThreads, 0, stream>>>(retries);
    Status status = check_kernel_launch("retry.choose_success", -1, false, stream);
    if (status != Status::kOk)
        return status;
    resolve_kernel<<<blocks, kThreads, 0, stream>>>(retries, *grouped, *primary);
    return check_kernel_launch("retry.resolve", -1, false, stream);
}
}  // namespace aruco3cuda::detail
