// SPDX-License-Identifier: Apache-2.0
#include "cell_decode.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/workspace.hpp"
#include "candidate_filter.hpp"
#include "cell_sample.hpp"
#include "cuda_check.hpp"

namespace aruco3cuda::detail {
namespace {

constexpr std::size_t kPlaneAlignment = 256U;
/// 1 候補を担当する block の thread 数。
constexpr int kDecodeThreads = 256;
/// 階調の数。8-bit なので 256。
constexpr int kHistogramBins = 256;
constexpr int kWarpSize = 32;
/// 1 block の warp 数。畳み込みの共有領域をこの数に抑える。
constexpr int kDecodeWarps = kDecodeThreads / kWarpSize;
/// OpenCV が Otsu の除外判定に使う値。FLT_EPSILON と同じ。
constexpr double kOtsuGuard = 1.1920928955078125e-07;

/// 判定に必要な定数をまとめて渡す。引数の数を抑えるため。
struct DecodeParams {
    int canonical_side_ = 0;
    int cells_ = 0;
    int cell_size_ = 0;
    int cell_margin_ = 0;
    int border_bits_ = 0;
    int max_border_errors_ = 0;
    float valid_bit_threshold_ = 0.0F;
    double min_otsu_std_dev_ = 0.0;
};

/// Otsu の漸化式が bin ごとに残す状態。
///
/// 逐次でなければならないのは漸化式だけである。その前後は要素ごとに独立で
/// あり、並列に計算しても丸めは変わらない。
struct OtsuState {
    double q1_[kHistogramBins];
    double mu1_[kHistogramBins];
    /// guard を通り抜けた bin か。通り抜けなかった bin は候補にならない。
    bool candidate_[kHistogramBins];
};

/// 閾値の探索で使う 1 候補分の値。
struct OtsuCandidate {
    double sigma_;
    int index_;
};

/// histogram の重心を並列に求める。
///
/// 各項 i * histogram[i] とその部分和はすべて整数であり、上限は
/// 255 * pixel_count である。canonical の 1 辺は 32767 以下に制限されている
/// ため pixel_count は 1.07e9 以下、255 倍しても 2.7e11 で 2^53 に収まる。
/// **すべての部分和が倍精度で厳密に表せるため丸めが 1 度も起きず、加算の
/// 順序を変えても結果は bit 同一である。**
__device__ double otsu_center(const int* histogram, double scale, double* warp_partial) {
    double partial = 0.0;
    for (int i = static_cast<int>(threadIdx.x); i < kHistogramBins;
         i += static_cast<int>(blockDim.x)) {
        partial += static_cast<double>(i) * static_cast<double>(histogram[i]);
    }
    // warp 内は shuffle で畳む。共有 memory を thread 数だけ使わずに済む。
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        partial += __shfl_down_sync(0xffffffffU, partial, offset);
    }
    const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    if (lane == 0) {
        warp_partial[warp] = partial;
    }
    __syncthreads();
    double total = 0.0;
    for (int w = 0; w < kDecodeWarps; ++w) {
        total += warp_partial[w];
    }
    return total * scale;
}

/// Otsu の漸化式を逐次で回し、bin ごとの状態を残す。
///
/// **この部分は絶対に並列化してはならない。** p_i は丸められた倍精度であり、
/// q1 の累積は順序に依存する。さらに guard で飛ばした反復では mu1 が
/// 正規化されない状態 (前回の mu1 に前回の q1 を掛けたもの) のまま持ち越され、
/// 次の反復でさらに新しい q1 を掛けられる。つまり mu1 は「連続して何回
/// 飛ばしたか」に依存する経路依存量であり、q1 を並列 scan で作ってから
/// mu1 を復元する形では必ず壊れる。
__device__ void otsu_recurrence(const int* histogram, double scale, OtsuState* state) {
    double mu1 = 0.0;
    double q1 = 0.0;
    for (int i = 0; i < kHistogramBins; ++i) {
        const double p_i = static_cast<double>(histogram[i]) * scale;
        mu1 *= q1;
        q1 += p_i;
        const double q2 = 1.0 - q1;
        state->q1_[i] = q1;
        if (fmin(q1, q2) < kOtsuGuard || fmax(q1, q2) > 1.0 - kOtsuGuard) {
            state->candidate_[i] = false;
            continue;
        }
        mu1 = (mu1 + (static_cast<double>(i) * p_i)) / q1;
        state->mu1_[i] = mu1;
        state->candidate_[i] = true;
    }
}

/// bin ごとの分離度を並列に求め、最大を与える階調を選ぶ。
///
/// 逐次版は max_sigma を 0.0 で始め、厳密な大なりで更新する。そこから
/// 3 つの性質が出る。
///
/// 1. sigma が 0 以下の bin は絶対に選ばれない
/// 2. 同値なら小さい階調が残る
/// 3. 一度も更新されなければ 0 を返す
///
/// 畳み込みでもこの 3 つを厳密に守る。比較は (sigma が大きい, 階調が小さい)
/// の辞書式にする。
__device__ int otsu_argmax(const OtsuState& state, double mu, OtsuCandidate* warp_best) {
    double best_sigma = 0.0;
    int best_index = kHistogramBins;
    for (int i = static_cast<int>(threadIdx.x); i < kHistogramBins;
         i += static_cast<int>(blockDim.x)) {
        if (!state.candidate_[i]) {
            continue;
        }
        const double q1 = state.q1_[i];
        const double mu1 = state.mu1_[i];
        const double q2 = 1.0 - q1;
        const double mu2 = (mu - (q1 * mu1)) / q2;
        const double sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2);
        // 0 以下は候補にしない。逐次版の max_sigma の初期値が 0.0 で、
        // 更新が厳密な大なりであることに対応する。
        if (sigma <= 0.0) {
            continue;
        }
        if (sigma > best_sigma || (sigma == best_sigma && i < best_index)) {
            best_sigma = sigma;
            best_index = i;
        }
    }
    // warp 内は shuffle で畳む。比較は (sigma が大きい, 階調が小さい) の辞書式。
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        const double other_sigma = __shfl_down_sync(0xffffffffU, best_sigma, offset);
        const int other_index = __shfl_down_sync(0xffffffffU, best_index, offset);
        if (other_sigma > best_sigma || (other_sigma == best_sigma && other_index < best_index)) {
            best_sigma = other_sigma;
            best_index = other_index;
        }
    }
    const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    if (lane == 0) {
        warp_best[warp].sigma_ = best_sigma;
        warp_best[warp].index_ = best_index;
    }
    __syncthreads();
    double sigma = 0.0;
    int index = kHistogramBins;
    for (int w = 0; w < kDecodeWarps; ++w) {
        if (warp_best[w].sigma_ > sigma ||
            (warp_best[w].sigma_ == sigma && warp_best[w].index_ < index)) {
            sigma = warp_best[w].sigma_;
            index = warp_best[w].index_;
        }
    }
    // 一度も候補が無ければ 0 を返す。
    return (index >= kHistogramBins) ? 0 : index;
}

/// 候補ごとにセル比を求め、外周セルの誤りを数える。1 block が 1 候補を担当する。
__global__ void decode_cells_kernel(const std::uint8_t* canonical, const std::int32_t* count,
                                    float* ratios, std::int32_t* border_errors,
                                    std::uint8_t* accepted, std::int32_t* thresholds,
                                    DecodeParams params) {
    const int candidate = static_cast<int>(blockIdx.x);
    if (candidate >= *count) {
        return;
    }

    __shared__ int histogram[kHistogramBins];
    // Otsu の 3 相で使う共有領域。漸化式が残す状態と、畳み込みの作業領域。
    __shared__ OtsuState otsu_state;
    __shared__ double reduction[kDecodeWarps];
    __shared__ OtsuCandidate best[kDecodeWarps];
    __shared__ unsigned long long inner_sum;
    __shared__ unsigned long long inner_square_sum;
    __shared__ int threshold;
    __shared__ float uniform_ratio;
    __shared__ bool uniform;

    const int side = params.canonical_side_;
    const int pixels = side * side;
    const std::uint8_t* image =
            canonical + (static_cast<std::size_t>(candidate) * static_cast<std::size_t>(side) *
                         static_cast<std::size_t>(side));

    for (int i = static_cast<int>(threadIdx.x); i < kHistogramBins;
         i += static_cast<int>(blockDim.x)) {
        histogram[i] = 0;
    }
    if (threadIdx.x == 0U) {
        inner_sum = 0ULL;
        inner_square_sum = 0ULL;
    }
    __syncthreads();

    // 内側の範囲。CPU 基準は各辺を cell の半分だけ寄せる。整数除算である。
    const int margin = params.cell_size_ / 2;
    const int inner_begin = margin;
    const int inner_end = side - margin;

    unsigned long long local_sum = 0ULL;
    unsigned long long local_square = 0ULL;
    for (int index = static_cast<int>(threadIdx.x); index < pixels;
         index += static_cast<int>(blockDim.x)) {
        const int value = static_cast<int>(image[index]);
        // Otsu は canonical の全体に掛ける。内側ではない。
        atomicAdd(&histogram[value], 1);
        const int x = index % side;
        const int y = index / side;
        if (x >= inner_begin && x < inner_end && y >= inner_begin && y < inner_end) {
            local_sum += static_cast<unsigned long long>(value);
            local_square += static_cast<unsigned long long>(value * value);
        }
    }
    atomicAdd(&inner_sum, local_sum);
    atomicAdd(&inner_square_sum, local_square);
    __syncthreads();

    if (threadIdx.x == 0U) {
        const int inner_side = inner_end - inner_begin;
        const int inner_count = inner_side * inner_side;
        // 和と二乗和は整数のまま積んだ。丸めが入るのはここからである。
        const double scale = 1.0 / static_cast<double>(inner_count);
        const double mean = static_cast<double>(inner_sum) * scale;
        const double variance = (static_cast<double>(inner_square_sum) * scale) - (mean * mean);
        const double deviation = sqrt(fmax(variance, 0.0));
        uniform = deviation < params.min_otsu_std_dev_;
        uniform_ratio = (mean > 127.0) ? 1.0F : 0.0F;
    }
    __syncthreads();

    // Otsu の閾値を 3 相で求める。重心の集計と分離度の探索は要素ごとに独立
    // なので並列にし、漸化式だけを thread 0 の逐次で回す。
    if (!uniform) {
        const double pixel_scale = 1.0 / static_cast<double>(pixels);
        const double center = otsu_center(histogram, pixel_scale, reduction);
        if (threadIdx.x == 0U) {
            otsu_recurrence(histogram, pixel_scale, &otsu_state);
        }
        __syncthreads();
        const int found = otsu_argmax(otsu_state, center, best);
        if (threadIdx.x == 0U) {
            threshold = found;
        }
    } else if (threadIdx.x == 0U) {
        threshold = 0;
    }
    __syncthreads();
    if (threadIdx.x == 0U) {
        thresholds[candidate] = threshold;
    }

    // セルごとの比。余白を除いた範囲の白画素数を数える。
    const int cells = params.cells_;
    const int cell_size = params.cell_size_;
    const int cell_margin = params.cell_margin_;
    const int inner_cell = cell_size - (2 * cell_margin);
    const float denominator = static_cast<float>(inner_cell) * static_cast<float>(inner_cell);
    float* cell_ratios =
            ratios + (static_cast<std::size_t>(candidate) * static_cast<std::size_t>(cells) *
                      static_cast<std::size_t>(cells));
    const int cell_count = cells * cells;
    for (int cell = static_cast<int>(threadIdx.x); cell < cell_count;
         cell += static_cast<int>(blockDim.x)) {
        if (uniform) {
            cell_ratios[cell] = uniform_ratio;
            continue;
        }
        const int cell_x = cell % cells;
        const int cell_y = cell / cells;
        const int origin_x = (cell_x * cell_size) + cell_margin;
        const int origin_y = (cell_y * cell_size) + cell_margin;
        int white = 0;
        for (int y = 0; y < inner_cell; ++y) {
            for (int x = 0; x < inner_cell; ++x) {
                const int value = static_cast<int>(image[((origin_y + y) * side) + origin_x + x]);
                // 二値化は「画素 > 閾値」。等しい場合は 0 側である。
                if (value > threshold) {
                    ++white;
                }
            }
        }
        cell_ratios[cell] = static_cast<float>(white) / denominator;
    }
    __syncthreads();

    if (threadIdx.x == 0U) {
        // 外周セルの誤りを数える。左右の列を全て見てから、上下の中央列を見る。
        // CPU 基準と同じ走査であり、角のセルを二重に数えない。
        int errors = 0;
        for (int y = 0; y < cells; ++y) {
            for (int k = 0; k < params.border_bits_; ++k) {
                if (cell_ratios[(y * cells) + k] > params.valid_bit_threshold_) {
                    ++errors;
                }
                if (cell_ratios[(y * cells) + (cells - 1 - k)] > params.valid_bit_threshold_) {
                    ++errors;
                }
            }
        }
        for (int x = params.border_bits_; x < cells - params.border_bits_; ++x) {
            for (int k = 0; k < params.border_bits_; ++k) {
                if (cell_ratios[(k * cells) + x] > params.valid_bit_threshold_) {
                    ++errors;
                }
                if (cell_ratios[((cells - 1 - k) * cells) + x] > params.valid_bit_threshold_) {
                    ++errors;
                }
            }
        }
        border_errors[candidate] = errors;
        accepted[candidate] = (errors > params.max_border_errors_) ? 0U : 1U;
    }
}

}  // namespace

int cells_per_side(const DetectorConfig& config, int marker_size) {
    if (marker_size < 1 || config.marker_border_bits_ < 1) {
        return 0;
    }
    const long long cells =
            static_cast<long long>(marker_size) + (2LL * config.marker_border_bits_);
    if (cells > 4096LL) {
        return 0;
    }
    return static_cast<int>(cells);
}

std::size_t cell_ratio_workspace_bytes(const DetectorConfig& config, int marker_size) {
    const int cells = cells_per_side(config, marker_size);
    if (cells == 0 || config.max_candidates_ <= 0) {
        return 0U;
    }
    const auto capacity = static_cast<std::size_t>(config.max_candidates_);
    const auto per_candidate = static_cast<std::size_t>(cells) * static_cast<std::size_t>(cells);
    if (per_candidate > std::numeric_limits<std::size_t>::max() / capacity) {
        return 0U;
    }
    const std::size_t ratios = align_up(per_candidate * capacity * sizeof(float), kPlaneAlignment);
    const std::size_t errors = align_up(capacity * sizeof(std::int32_t), kPlaneAlignment);
    const std::size_t flags = align_up(capacity * sizeof(std::uint8_t), kPlaneAlignment);
    const std::size_t thresholds = align_up(capacity * sizeof(std::int32_t), kPlaneAlignment);
    if (ratios == 0U || errors == 0U || flags == 0U || thresholds == 0U) {
        return 0U;
    }
    return ratios + errors + flags + thresholds;
}

Status reserve_cell_ratios(const DetectorConfig& config, int marker_size, Workspace& workspace,
                           CellRatioBuffers* out) {
    if (out == nullptr) {
        return Status::kInvalidArgument;
    }
    const int cells = cells_per_side(config, marker_size);
    if (cells == 0) {
        return Status::kInvalidArgument;
    }
    if (cell_ratio_workspace_bytes(config, marker_size) == 0U) {
        return Status::kInvalidConfig;
    }
    const auto capacity = static_cast<std::size_t>(config.max_candidates_);
    const auto per_candidate = static_cast<std::size_t>(cells) * static_cast<std::size_t>(cells);

    CellRatioBuffers buffers;
    buffers.cells_per_side_ = cells;
    buffers.capacity_ = config.max_candidates_;

    void* pointer = nullptr;
    Status status =
            workspace.allocate(per_candidate * capacity * sizeof(float), kPlaneAlignment, &pointer);
    if (status != Status::kOk) {
        return status;
    }
    buffers.ratios_ = static_cast<float*>(pointer);

    status = workspace.allocate(capacity * sizeof(std::int32_t), kPlaneAlignment, &pointer);
    if (status != Status::kOk) {
        return status;
    }
    buffers.border_errors_ = static_cast<std::int32_t*>(pointer);

    status = workspace.allocate(capacity * sizeof(std::uint8_t), kPlaneAlignment, &pointer);
    if (status != Status::kOk) {
        return status;
    }
    buffers.accepted_ = static_cast<std::uint8_t*>(pointer);

    status = workspace.allocate(capacity * sizeof(std::int32_t), kPlaneAlignment, &pointer);
    if (status != Status::kOk) {
        return status;
    }
    buffers.thresholds_ = static_cast<std::int32_t*>(pointer);

    *out = buffers;
    return Status::kOk;
}

Status build_cell_ratios_async(const CanonicalBuffers& canonical,
                               const DeviceCandidates& candidates, const DetectorConfig& config,
                               int marker_size, CellRatioBuffers* ratios, cudaStream_t stream) {
    if (ratios == nullptr || ratios->ratios_ == nullptr || ratios->thresholds_ == nullptr ||
        canonical.images_ == nullptr || candidates.count_ == nullptr) {
        return Status::kInvalidArgument;
    }
    const int cells = cells_per_side(config, marker_size);
    if (cells == 0 || ratios->cells_per_side_ != cells) {
        return Status::kInvalidArgument;
    }
    if (ratios->capacity_ < candidates.capacity_) {
        return Status::kInvalidArgument;
    }

    DecodeParams params;
    params.canonical_side_ = canonical.side_px_;
    params.cells_ = cells;
    params.cell_size_ = config.perspective_remove_pixel_per_cell_;
    // 余白は 0 方向へ切り捨てる。既定設定では 0 になり、cell の全域を数える。
    params.cell_margin_ = static_cast<int>(config.perspective_remove_ignored_margin_per_cell_ *
                                           config.perspective_remove_pixel_per_cell_);
    params.border_bits_ = config.marker_border_bits_;
    // 外周セル数ではなく marker_size の 2 乗に率を掛ける。CPU 基準と同じである。
    params.max_border_errors_ = static_cast<int>(static_cast<double>(marker_size) * marker_size *
                                                 config.max_erroneous_bits_in_border_rate_);
    params.valid_bit_threshold_ = static_cast<float>(config.valid_bit_threshold_);
    params.min_otsu_std_dev_ = config.min_otsu_std_dev_;
    if (params.cell_size_ - (2 * params.cell_margin_) < 1) {
        return Status::kInvalidConfig;
    }

    decode_cells_kernel<<<static_cast<unsigned int>(candidates.capacity_),
                          static_cast<unsigned int>(kDecodeThreads), 0, stream>>>(
            canonical.images_, candidates.count_, ratios->ratios_, ratios->border_errors_,
            ratios->accepted_, ratios->thresholds_, params);
    return check_kernel_launch("cell_decode.decode_cells_kernel", -1, false, stream);
}

}  // namespace aruco3cuda::detail
