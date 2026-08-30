// SPDX-License-Identifier: Apache-2.0
#include "dictionary_match.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/dictionary.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/workspace.hpp"
#include "candidate_filter.hpp"
#include "cell_decode.hpp"
#include "cuda_check.hpp"

namespace aruco3cuda::detail {
namespace {

constexpr std::size_t kPlaneAlignment = 256U;
/// Thread count of the block that handles one candidate.
constexpr int kMatchThreads = 256;
/// Largest supported marker side. It follows from MarkerCode being 64 bits wide.
constexpr int kMaxMarkerSize = 7;

/// Constants passed to the kernel, bundled to keep the argument count down.
struct MatchParams {
    int cells_per_side_ = 0;
    int border_bits_ = 0;
    int marker_size_ = 0;
    int code_count_ = 0;
    int max_errors_ = 0;
    float valid_bit_threshold_ = 0.0F;
};

/// Computes the smallest distance over the four rotations of one ID.
///
/// The rotations are visited in the same order as OpenCV's hammingDistanceToId. The update uses a
/// strict less-than, so on a tie the lower rotation wins, and the loop breaks at distance 0 just
/// as the reference does.
__device__ int distance_to_id(const MarkerCode* codes, int id, MarkerCode not_black,
                              MarkerCode selector, int bit_count, int* out_rotation) {
    int smallest = bit_count + 1;
    int rotation = 0;
    for (int r = 0; r < 4; ++r) {
        const MarkerCode code =
                codes[(static_cast<std::size_t>(id) * 4U) + static_cast<std::size_t>(r)];
        const int distance = __popcll(not_black ^ (selector & code));
        if (distance < smallest) {
            smallest = distance;
            rotation = r;
            if (distance == 0) {
                break;
            }
        }
    }
    *out_rotation = rotation;
    return smallest;
}

/// Matches the cell ratios of each candidate against the dictionary. One block handles one
/// candidate.
__global__ void match_kernel(const float* ratios, const std::uint8_t* accepted,
                             const std::int32_t* count, const MarkerCode* codes, std::int32_t* ids,
                             std::int32_t* rotations, std::int32_t* distances, MatchParams params) {
    const int candidate = static_cast<int>(blockIdx.x);
    if (candidate >= *count) {
        return;
    }

    // unsigned long long and MarkerCode have the same width here. atomicOr only exists for
    // unsigned long long, so the masks are held in that type.
    __shared__ unsigned long long not_black;
    __shared__ unsigned long long not_white;
    __shared__ int best_id;
    __shared__ int smallest_distance;

    if (threadIdx.x == 0U) {
        not_black = 0ULL;
        not_white = 0ULL;
        best_id = params.code_count_;
        smallest_distance = (params.marker_size_ * params.marker_size_) + 1;
    }
    __syncthreads();

    if (accepted[candidate] == 0U) {
        // Candidates rejected by the border check are not matched. The CPU reference bails out
        // at the same point.
        if (threadIdx.x == 0U) {
            ids[candidate] = -1;
            rotations[candidate] = 0;
            distances[candidate] = smallest_distance;
        }
        return;
    }

    const int cells = params.cells_per_side_;
    const int border = params.border_bits_;
    const int marker_size = params.marker_size_;
    const int bit_count = marker_size * marker_size;
    const float* cell_ratios =
            ratios + (static_cast<std::size_t>(candidate) * static_cast<std::size_t>(cells) *
                      static_cast<std::size_t>(cells));
    // The upper bound is computed in float because OpenCV evaluates `1 - validBitIdThreshold`
    // in float. Computing it in double precision would shift the threshold itself by 1 ULP.
    const float upper = 1.0F - params.valid_bit_threshold_;

    unsigned long long local_black = 0ULL;
    unsigned long long local_white = 0ULL;
    for (int i = static_cast<int>(threadIdx.x); i < bit_count; i += static_cast<int>(blockDim.x)) {
        const int row = i / marker_size;
        const int col = i % marker_size;
        const float ratio = cell_ratios[((row + border) * cells) + col + border];
        const unsigned long long bit = 1ULL << i;
        if (ratio > params.valid_bit_threshold_) {
            local_black |= bit;
        }
        if (ratio < upper) {
            local_white |= bit;
        }
    }
    if (local_black != 0ULL) {
        atomicOr(&not_black, local_black);
    }
    if (local_white != 0ULL) {
        atomicOr(&not_white, local_white);
    }
    __syncthreads();

    const MarkerCode candidate_black = not_black;
    const MarkerCode selector = not_black ^ not_white;
    for (int id = static_cast<int>(threadIdx.x); id < params.code_count_;
         id += static_cast<int>(blockDim.x)) {
        int rotation = 0;
        const int distance =
                distance_to_id(codes, id, candidate_black, selector, bit_count, &rotation);
        atomicMin(&smallest_distance, distance);
        // Take the first ID that satisfies the condition, not the ID with the smallest
        // distance. Taking the minimum over the satisfying IDs reproduces OpenCV, which walks
        // the IDs in ascending order and stops at the first one that satisfies it.
        if (distance <= params.max_errors_) {
            atomicMin(&best_id, id);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0U) {
        if (best_id >= params.code_count_) {
            ids[candidate] = -1;
            rotations[candidate] = 0;
            distances[candidate] = smallest_distance;
            return;
        }
        // Recompute the rotation of the selected ID. That is cheaper than keeping one per ID.
        int rotation = 0;
        const int distance =
                distance_to_id(codes, best_id, candidate_black, selector, bit_count, &rotation);
        ids[candidate] = best_id;
        rotations[candidate] = rotation;
        distances[candidate] = distance;
    }
}

}  // namespace

std::size_t device_dictionary_workspace_bytes(const DictionaryTable& table) {
    if (table.codes_ == nullptr || table.code_count_ <= 0 || table.marker_size_ < 1 ||
        table.marker_size_ > kMaxMarkerSize) {
        return 0U;
    }
    const auto entries = static_cast<std::size_t>(table.code_count_) * 4U;
    return align_up(entries * sizeof(MarkerCode), kPlaneAlignment);
}

Status upload_dictionary(const DictionaryTable& table, Workspace& workspace, DeviceDictionary* out,
                         cudaStream_t stream) {
    if (out == nullptr) {
        return Status::kInvalidArgument;
    }
    const std::size_t bytes = device_dictionary_workspace_bytes(table);
    if (bytes == 0U) {
        return Status::kInvalidArgument;
    }
    void* pointer = nullptr;
    const Status status = workspace.allocate(bytes, kPlaneAlignment, &pointer);
    if (status != Status::kOk) {
        return status;
    }
    const auto entries = static_cast<std::size_t>(table.code_count_) * 4U;
    const Status copied =
            check_cuda(cudaMemcpyAsync(pointer, table.codes_, entries * sizeof(MarkerCode),
                                       cudaMemcpyHostToDevice, stream),
                       "cudaMemcpyAsync", "dictionary_match.upload_dictionary", -1, nullptr);
    if (copied != Status::kOk) {
        return copied;
    }

    DeviceDictionary dictionary;
    dictionary.codes_ = static_cast<const MarkerCode*>(pointer);
    dictionary.marker_size_ = table.marker_size_;
    dictionary.code_count_ = table.code_count_;
    dictionary.max_correction_bits_ = table.max_correction_bits_;
    *out = dictionary;
    return Status::kOk;
}

std::size_t match_workspace_bytes(const DetectorConfig& config) {
    if (config.max_candidates_ <= 0) {
        return 0U;
    }
    const auto capacity = static_cast<std::size_t>(config.max_candidates_);
    const std::size_t plane = align_up(capacity * sizeof(std::int32_t), kPlaneAlignment);
    if (plane == 0U) {
        return 0U;
    }
    return plane * 3U;
}

Status reserve_matches(const DetectorConfig& config, Workspace& workspace, MatchBuffers* out) {
    if (out == nullptr) {
        return Status::kInvalidArgument;
    }
    if (match_workspace_bytes(config) == 0U) {
        return Status::kInvalidConfig;
    }
    const auto capacity = static_cast<std::size_t>(config.max_candidates_);

    MatchBuffers buffers;
    buffers.capacity_ = config.max_candidates_;
    std::int32_t** targets[] = {&buffers.ids_, &buffers.rotations_, &buffers.distances_};
    for (std::int32_t** target : targets) {
        void* pointer = nullptr;
        const Status status =
                workspace.allocate(capacity * sizeof(std::int32_t), kPlaneAlignment, &pointer);
        if (status != Status::kOk) {
            return status;
        }
        *target = static_cast<std::int32_t*>(pointer);
    }
    *out = buffers;
    return Status::kOk;
}

Status match_candidates_async(const CellRatioBuffers& ratios, const DeviceCandidates& candidates,
                              const DeviceDictionary& dictionary, const DetectorConfig& config,
                              MatchBuffers* matches, cudaStream_t stream) {
    if (matches == nullptr || matches->ids_ == nullptr || ratios.ratios_ == nullptr ||
        ratios.accepted_ == nullptr || candidates.count_ == nullptr ||
        dictionary.codes_ == nullptr) {
        return Status::kInvalidArgument;
    }
    if (dictionary.marker_size_ < 1 || dictionary.marker_size_ > kMaxMarkerSize ||
        dictionary.code_count_ <= 0) {
        return Status::kInvalidArgument;
    }
    if (matches->capacity_ < candidates.capacity_) {
        return Status::kInvalidArgument;
    }
    // The interior can only be carved out when the cell-ratio side is marker_size + 2 * border.
    if (ratios.cells_per_side_ != dictionary.marker_size_ + (2 * config.marker_border_bits_)) {
        return Status::kInvalidConfig;
    }

    MatchParams params;
    params.cells_per_side_ = ratios.cells_per_side_;
    params.border_bits_ = config.marker_border_bits_;
    params.marker_size_ = dictionary.marker_size_;
    params.code_count_ = dictionary.code_count_;
    // Truncation toward zero, the same as OpenCV's maxCorrectionRecalculed.
    params.max_errors_ = static_cast<int>(static_cast<double>(dictionary.max_correction_bits_) *
                                          config.error_correction_rate_);
    params.valid_bit_threshold_ = static_cast<float>(config.valid_bit_threshold_);

    match_kernel<<<static_cast<unsigned int>(candidates.capacity_),
                   static_cast<unsigned int>(kMatchThreads), 0, stream>>>(
            ratios.ratios_, ratios.accepted_, candidates.count_, dictionary.codes_, matches->ids_,
            matches->rotations_, matches->distances_, params);
    return check_kernel_launch("dictionary_match.match_kernel", -1, false, stream);
}

}  // namespace aruco3cuda::detail
