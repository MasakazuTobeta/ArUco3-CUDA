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
/// 1 候補を担当する block の thread 数。
constexpr int kMatchThreads = 256;
/// 対応する最大の marker 1 辺。MarkerCode が 64 bit であることによる。
constexpr int kMaxMarkerSize = 7;

/// kernel へ渡す定数。引数の数を抑えるためまとめる。
struct MatchParams {
    int cells_per_side_ = 0;
    int border_bits_ = 0;
    int marker_size_ = 0;
    int code_count_ = 0;
    int max_errors_ = 0;
    float valid_bit_threshold_ = 0.0F;
};

/// 1 つの ID について 4 回転の最小距離を求める。
///
/// OpenCV の hammingDistanceToId と同じ順序で見る。更新は厳密な小なりであり、
/// 同値なら小さい回転が残る。距離 0 で打ち切るのも同じである。
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

/// 候補ごとにセル比を Dictionary と照合する。1 block が 1 候補を担当する。
__global__ void match_kernel(const float* ratios, const std::uint8_t* accepted,
                             const std::int32_t* count, const MarkerCode* codes, std::int32_t* ids,
                             std::int32_t* rotations, std::int32_t* distances, MatchParams params) {
    const int candidate = static_cast<int>(blockIdx.x);
    if (candidate >= *count) {
        return;
    }

    // 既定では unsigned long long と MarkerCode は同じ幅である。atomicOr は
    // unsigned long long の版しか無いため、その型で持つ。
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
        // border 検証で落ちた候補は照合しない。CPU 基準も同じ位置で打ち切る。
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
    // 上限は float で求める。OpenCV は `1 - validBitIdThreshold` を float で
    // 評価する。倍精度で求めると閾値そのものが 1 ULP 動く。
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
        // 最小距離の ID ではなく、条件を満たした最初の ID を採る。ID の昇順に
        // 見て最初に条件を満たしたところで打ち切る OpenCV と同じにするため、
        // 満たした ID の最小値を取る。
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
        // 採用した ID の回転を求め直す。全 ID 分を保持するより安い。
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
    // セル比の 1 辺は marker_size + 2 * border でなければ内側を切り出せない。
    if (ratios.cells_per_side_ != dictionary.marker_size_ + (2 * config.marker_border_bits_)) {
        return Status::kInvalidConfig;
    }

    MatchParams params;
    params.cells_per_side_ = ratios.cells_per_side_;
    params.border_bits_ = config.marker_border_bits_;
    params.marker_size_ = dictionary.marker_size_;
    params.code_count_ = dictionary.code_count_;
    // 0 方向への切り捨て。OpenCV の maxCorrectionRecalculed と同じ。
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
