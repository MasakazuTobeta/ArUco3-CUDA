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

/// histogram から Otsu の閾値を求める。
///
/// OpenCV の getThreshVal_Otsu と同じ漸化式にする。評価の順序を変えると
/// 丸めが変わる。更新は厳密な大なりであり、同値なら小さい階調が残る。
/// 全画素が同じ値の場合は一度も更新されず 0 が返る。
__device__ int otsu_threshold(const int* histogram, int pixel_count) {
    const double scale = 1.0 / static_cast<double>(pixel_count);
    double mu = 0.0;
    for (int i = 0; i < kHistogramBins; ++i) {
        mu += static_cast<double>(i) * static_cast<double>(histogram[i]);
    }
    mu *= scale;

    double mu1 = 0.0;
    double q1 = 0.0;
    double max_sigma = 0.0;
    int max_val = 0;
    for (int i = 0; i < kHistogramBins; ++i) {
        const double p_i = static_cast<double>(histogram[i]) * scale;
        mu1 *= q1;
        q1 += p_i;
        const double q2 = 1.0 - q1;
        if (fmin(q1, q2) < kOtsuGuard || fmax(q1, q2) > 1.0 - kOtsuGuard) {
            continue;
        }
        mu1 = (mu1 + (static_cast<double>(i) * p_i)) / q1;
        const double mu2 = (mu - (q1 * mu1)) / q2;
        const double sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2);
        if (sigma > max_sigma) {
            max_sigma = sigma;
            max_val = i;
        }
    }
    return max_val;
}

/// 候補ごとにセル比を求め、外周セルの誤りを数える。1 block が 1 候補を担当する。
__global__ void decode_cells_kernel(const std::uint8_t* canonical, const std::int32_t* count,
                                    float* ratios, std::int32_t* border_errors,
                                    std::uint8_t* accepted, DecodeParams params) {
    const int candidate = static_cast<int>(blockIdx.x);
    if (candidate >= *count) {
        return;
    }

    __shared__ int histogram[kHistogramBins];
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
        threshold = uniform ? 0 : otsu_threshold(histogram, pixels);
    }
    __syncthreads();

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
    if (ratios == 0U || errors == 0U || flags == 0U) {
        return 0U;
    }
    return ratios + errors + flags;
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

    *out = buffers;
    return Status::kOk;
}

Status build_cell_ratios_async(const CanonicalBuffers& canonical,
                               const DeviceCandidates& candidates, const DetectorConfig& config,
                               int marker_size, CellRatioBuffers* ratios, cudaStream_t stream) {
    if (ratios == nullptr || ratios->ratios_ == nullptr || canonical.images_ == nullptr ||
        candidates.count_ == nullptr) {
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
            ratios->accepted_, params);
    return check_kernel_launch("cell_decode.decode_cells_kernel", -1, false, stream);
}

}  // namespace aruco3cuda::detail
