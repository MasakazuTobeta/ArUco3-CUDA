// SPDX-License-Identifier: Apache-2.0
#include "threshold.hpp"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"
#include "aruco3cuda/workspace.hpp"
#include "cuda_check.hpp"

namespace aruco3cuda::detail {
namespace {

constexpr std::size_t kPlaneAlignment = 256U;

/// BORDER_REPLICATE の index 制限。端の画素をそのまま繰り返す。
__device__ inline int replicate(int index, int size) {
    if (index < 0) {
        return 0;
    }
    if (index >= size) {
        return size - 1;
    }
    return index;
}

/// 行方向の合計を求める。
///
/// thread と data の対応:
///   thread (x, y) が row_sums[y][x] だけを書き込む。入力は同じ行の
///   x-radius から x+radius を読む。書き込み先が重複しないため競合はない。
/// 境界条件:
///   出力範囲外の thread は何もしない。行内の参照は BORDER_REPLICATE で
///   端の画素を繰り返す。合計は最大 255 * 4096 であり int に収まる。
__global__ void row_sum_kernel(const std::uint8_t* __restrict__ src, std::size_t src_pitch,
                               int width, int height, std::int32_t* __restrict__ row_sums,
                               std::size_t row_sums_pitch, int radius) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) {
        return;
    }
    const std::uint8_t* row = src + static_cast<std::size_t>(y) * src_pitch;
    int sum = 0;
    for (int offset = -radius; offset <= radius; ++offset) {
        sum += static_cast<int>(row[replicate(x + offset, width)]);
    }
    auto* destination =
            reinterpret_cast<std::int32_t*>(reinterpret_cast<std::uint8_t*>(row_sums) +
                                            static_cast<std::size_t>(y) * row_sums_pitch);
    destination[x] = sum;
}

/// 列方向の合計から平均を求め、閾値判定を行う。
///
/// thread と data の対応:
///   thread (x, y) が出力の 1 画素だけを書き込む。row_sums の同じ列の
///   y-radius から y+radius を読む。書き込み先が重複しないため競合はない。
/// 境界条件:
///   出力範囲外の thread は何もしない。列の参照は BORDER_REPLICATE で
///   端の行を繰り返す。合計は最大 255 * 4096 * 4096 であり int の範囲を
///   超え得るため、window 数の上限と画像寸法の上限で抑える。
/// 丸め:
///   OpenCV の boxFilter は正規化した値を saturate_cast<uchar> で丸める。
///   これは最近接偶数丸めであるため __double2int_rn を使う。
__global__ void column_threshold_kernel(const std::uint8_t* __restrict__ src, std::size_t src_pitch,
                                        const std::int32_t* __restrict__ row_sums,
                                        std::size_t row_sums_pitch, std::uint8_t* __restrict__ dst,
                                        std::size_t dst_pitch, int width, int height, int radius,
                                        double inverse_area, int idelta) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) {
        return;
    }
    int sum = 0;
    for (int offset = -radius; offset <= radius; ++offset) {
        const int sy = replicate(y + offset, height);
        const auto* row = reinterpret_cast<const std::int32_t*>(
                reinterpret_cast<const std::uint8_t*>(row_sums) +
                static_cast<std::size_t>(sy) * row_sums_pitch);
        sum += row[x];
    }
    int mean = __double2int_rn(static_cast<double>(sum) * inverse_area);
    mean = mean < 0 ? 0 : (mean > 255 ? 255 : mean);

    const int value = static_cast<int>(
            src[static_cast<std::size_t>(y) * src_pitch + static_cast<std::size_t>(x)]);
    // OpenCV の THRESH_BINARY_INV は (画素 - 平均) <= -idelta で maxval を返す。
    const std::uint8_t output = ((value - mean) <= -idelta) ? 255U : 0U;
    dst[static_cast<std::size_t>(y) * dst_pitch + static_cast<std::size_t>(x)] = output;
}

std::size_t plane_bytes_u8(int width_px, int height_px) {
    if (width_px <= 0 || height_px <= 0) {
        return 0U;
    }
    const std::size_t pitch = align_up(static_cast<std::size_t>(width_px), kPlaneAlignment);
    if (pitch == 0U) {
        return 0U;
    }
    const auto height = static_cast<std::size_t>(height_px);
    if (pitch > std::numeric_limits<std::size_t>::max() / height) {
        return 0U;
    }
    return align_up(pitch * height, kPlaneAlignment);
}

std::size_t plane_bytes_i32(int width_px, int height_px) {
    if (width_px <= 0 || height_px <= 0) {
        return 0U;
    }
    const std::size_t pitch =
            align_up(static_cast<std::size_t>(width_px) * sizeof(std::int32_t), kPlaneAlignment);
    if (pitch == 0U) {
        return 0U;
    }
    const auto height = static_cast<std::size_t>(height_px);
    if (pitch > std::numeric_limits<std::size_t>::max() / height) {
        return 0U;
    }
    return align_up(pitch * height, kPlaneAlignment);
}

dim3 block_dim_of(const DetectorConfig& config) {
    const auto side = static_cast<unsigned int>(config.cuda_block_dim_);
    return dim3(side, side, 1U);
}

dim3 grid_dim_for(int width, int height, dim3 block) {
    return dim3((static_cast<unsigned int>(width) + block.x - 1U) / block.x,
                (static_cast<unsigned int>(height) + block.y - 1U) / block.y, 1U);
}

}  // namespace

Status threshold_window_sizes(const DetectorConfig& config, int* out_sizes, int capacity,
                              int* out_count) {
    if (out_sizes == nullptr || out_count == nullptr || capacity < 1) {
        return Status::kInvalidArgument;
    }
    const Status config_status = config.validate(nullptr);
    if (config_status != Status::kOk) {
        return config_status;
    }
    const int count =
            (config.adaptive_thresh_win_size_max_px_ - config.adaptive_thresh_win_size_min_px_) /
                    config.adaptive_thresh_win_size_step_px_ +
            1;
    if (count > capacity) {
        return Status::kInvalidConfig;
    }
    for (int i = 0; i < count; ++i) {
        int window = config.adaptive_thresh_win_size_min_px_ +
                     i * config.adaptive_thresh_win_size_step_px_;
        // OpenCV の _threshold は偶数の window を奇数へ切り上げる。
        if (window % 2 == 0) {
            ++window;
        }
        out_sizes[i] = window;
    }
    *out_count = count;
    return Status::kOk;
}

std::size_t threshold_workspace_bytes(const DetectorConfig& config, int width_px, int height_px) {
    int sizes[kMaxAdaptiveThresholdWindows] = {};
    int count = 0;
    if (threshold_window_sizes(config, sizes, kMaxAdaptiveThresholdWindows, &count) !=
        Status::kOk) {
        return 0U;
    }
    const std::size_t binary = plane_bytes_u8(width_px, height_px);
    const std::size_t row_sums = plane_bytes_i32(width_px, height_px);
    if (binary == 0U || row_sums == 0U) {
        return 0U;
    }
    const auto window_count = static_cast<std::size_t>(count);
    if (binary > (std::numeric_limits<std::size_t>::max() - row_sums) / window_count) {
        return 0U;
    }
    return binary * window_count + row_sums;
}

Status reserve_threshold(const DetectorConfig& config, int width_px, int height_px,
                         Workspace& workspace, ThresholdBuffers* out) {
    if (out == nullptr || width_px < 1 || height_px < 1) {
        return Status::kInvalidArgument;
    }
    ThresholdBuffers buffers;
    const Status size_status = threshold_window_sizes(
            config, buffers.window_sizes_px_, kMaxAdaptiveThresholdWindows, &buffers.window_count_);
    if (size_status != Status::kOk) {
        return size_status;
    }
    buffers.width_px_ = width_px;
    buffers.height_px_ = height_px;

    const std::size_t binary_pitch = align_up(static_cast<std::size_t>(width_px), kPlaneAlignment);
    for (int i = 0; i < buffers.window_count_; ++i) {
        void* pointer = nullptr;
        const Status status = workspace.allocate(binary_pitch * static_cast<std::size_t>(height_px),
                                                 kPlaneAlignment, &pointer);
        if (status != Status::kOk) {
            return status;
        }
        ImagePlaneU8& plane = buffers.binary_[i];
        plane.data_ = static_cast<std::uint8_t*>(pointer);
        plane.width_px_ = width_px;
        plane.height_px_ = height_px;
        plane.pitch_bytes_ = binary_pitch;
    }

    const std::size_t row_sums_pitch =
            align_up(static_cast<std::size_t>(width_px) * sizeof(std::int32_t), kPlaneAlignment);
    void* row_sums_pointer = nullptr;
    const Status row_sums_status =
            workspace.allocate(row_sums_pitch * static_cast<std::size_t>(height_px),
                               kPlaneAlignment, &row_sums_pointer);
    if (row_sums_status != Status::kOk) {
        return row_sums_status;
    }
    buffers.row_sums_ = static_cast<std::int32_t*>(row_sums_pointer);
    buffers.row_sums_pitch_bytes_ = row_sums_pitch;

    *out = buffers;
    return Status::kOk;
}

Status build_threshold_async(const ImageViewU8& segmentation, ThresholdBuffers* buffers,
                             const DetectorConfig& config, cudaStream_t stream) {
    if (buffers == nullptr || buffers->row_sums_ == nullptr || buffers->window_count_ < 1) {
        return Status::kInvalidArgument;
    }
    if (segmentation.data_ == nullptr || segmentation.width_px_ != buffers->width_px_ ||
        segmentation.height_px_ != buffers->height_px_) {
        return Status::kInvalidArgument;
    }
    // OpenCV の THRESH_BINARY_INV は定数を切り捨てて整数化する。
    const int idelta = static_cast<int>(std::floor(config.adaptive_thresh_constant_));
    const dim3 block = block_dim_of(config);
    const dim3 grid = grid_dim_for(buffers->width_px_, buffers->height_px_, block);

    for (int i = 0; i < buffers->window_count_; ++i) {
        const int window = buffers->window_sizes_px_[i];
        const int radius = window / 2;
        const double inverse_area = 1.0 / (static_cast<double>(window) * window);

        row_sum_kernel<<<grid, block, 0, stream>>>(
                segmentation.data_, segmentation.pitch_bytes_, buffers->width_px_,
                buffers->height_px_, buffers->row_sums_, buffers->row_sums_pitch_bytes_, radius);
        Status status = detail::check_kernel_launch("threshold.row_sum_kernel", -1, false, stream);
        if (status != Status::kOk) {
            return status;
        }

        // 行合計の作業領域を window ごとに使い回すため、同じ stream で順に実行する。
        // stream を分けると前の window の結果を上書きする。
        column_threshold_kernel<<<grid, block, 0, stream>>>(
                segmentation.data_, segmentation.pitch_bytes_, buffers->row_sums_,
                buffers->row_sums_pitch_bytes_, buffers->binary_[i].data_,
                buffers->binary_[i].pitch_bytes_, buffers->width_px_, buffers->height_px_, radius,
                inverse_area, idelta);
        status =
                detail::check_kernel_launch("threshold.column_threshold_kernel", -1, false, stream);
        if (status != Status::kOk) {
            return status;
        }
    }
    return Status::kOk;
}

}  // namespace aruco3cuda::detail
