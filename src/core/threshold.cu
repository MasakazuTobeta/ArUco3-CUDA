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

// Experimental ablation switches. The default remains the measured separable
// implementation until end-to-end and multi-device validation is complete.
#ifndef ARUCO3CUDA_ENABLE_FUSED_THRESHOLD
#define ARUCO3CUDA_ENABLE_FUSED_THRESHOLD 0
#endif
#ifndef ARUCO3CUDA_FUSED_THRESHOLD_MAX_WINDOW_PX
#define ARUCO3CUDA_FUSED_THRESHOLD_MAX_WINDOW_PX 63
#endif

namespace aruco3cuda::detail {
namespace {

constexpr std::size_t kPlaneAlignment = 256U;

/// BORDER_REPLICATE index clamp. Repeats the edge pixel as is.
__device__ inline int replicate(int index, int size) {
    if (index < 0) {
        return 0;
    }
    if (index >= size) {
        return size - 1;
    }
    return index;
}

/// Computes the row-wise sums.
///
/// Thread-to-data mapping:
///   Thread (x, y) writes row_sums[y][x] only. It reads x-radius through
///   x+radius of the same input row. Since destinations never overlap, there is
///   no race.
/// Boundary conditions:
///   Threads outside the output extent do nothing. Reads within the row repeat
///   the edge pixel with BORDER_REPLICATE. The sum is at most 255 * 4096, which
///   fits in an int.
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

/// Derives the mean from the column-wise sums and applies the threshold test.
///
/// Thread-to-data mapping:
///   Thread (x, y) writes exactly one output pixel. It reads y-radius through
///   y+radius of the same row_sums column. Since destinations never overlap,
///   there is no race.
/// Boundary conditions:
///   Threads outside the output extent do nothing. Reads down the column repeat
///   the edge row with BORDER_REPLICATE. The sum is at most 255 * 4096 * 4096,
///   which can exceed the range of an int. The launcher chooses a 64-bit sum
///   when 255 * window * window exceeds INT_MAX; small windows keep 32-bit sums.
/// Rounding:
///   OpenCV boxFilter rounds the normalized value with saturate_cast<uchar>.
///   That rounds half to even, so __double2int_rn is used here.
template <typename Sum>
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
    Sum sum = 0;
    for (int offset = -radius; offset <= radius; ++offset) {
        const int sy = replicate(y + offset, height);
        const auto* row = reinterpret_cast<const std::int32_t*>(
                reinterpret_cast<const std::uint8_t*>(row_sums) +
                static_cast<std::size_t>(sy) * row_sums_pitch);
        sum += static_cast<Sum>(row[x]);
    }
    int mean = __double2int_rn(static_cast<double>(sum) * inverse_area);
    mean = mean < 0 ? 0 : (mean > 255 ? 255 : mean);

    const int value = static_cast<int>(
            src[static_cast<std::size_t>(y) * src_pitch + static_cast<std::size_t>(x)]);
    // OpenCV THRESH_BINARY_INV returns maxval when (pixel - mean) <= -idelta.
    const std::uint8_t output = ((value - mean) <= -idelta) ? 255U : 0U;
    dst[static_cast<std::size_t>(y) * dst_pitch + static_cast<std::size_t>(x)] = output;
}

#if ARUCO3CUDA_ENABLE_FUSED_THRESHOLD
/// Captured launch arguments for all windows; these contain no frame-derived state.
struct FusedThresholdParams {
    std::uint8_t* output_[kMaxAdaptiveThresholdWindows] = {};
    std::size_t pitch_[kMaxAdaptiveThresholdWindows] = {};
    int radius_[kMaxAdaptiveThresholdWindows] = {};
    double inverse_area_[kMaxAdaptiveThresholdWindows] = {};
    int window_count_ = 0;
    int max_radius_ = 0;
    int idelta_ = 0;
};

/// Builds a shared-memory integral tile once, then emits every threshold window.
///
/// Each block owns blockDim.x by blockDim.y output pixels. Threads cooperatively
/// load an isolated BORDER_REPLICATE halo plus a zero top row/left column. Row
/// prefix workers own disjoint rows; after a barrier, column workers own disjoint
/// columns. A final barrier publishes the completed integral image before each
/// thread reads four sums per window and writes its unique output pixel. Threads
/// outside the image still take part in every barrier, avoiding divergent waits
/// on partial tiles. No block reads another block's scratch or output.
///
/// The launcher limits the tile footprint so 255 * tile area fits in int and
/// shared memory stays below the portable 48 KiB per-block budget. Larger windows
/// use the separable route; tile sums are exact and normalization retains the
/// same double operation and nearest-even rounding as that route.
__global__ void fused_threshold_kernel(const std::uint8_t* __restrict__ src, std::size_t src_pitch,
                                       int width, int height, FusedThresholdParams params) {
    extern __shared__ int prefix[];
    const int block_width = static_cast<int>(blockDim.x);
    const int block_height = static_cast<int>(blockDim.y);
    const int thread = static_cast<int>(threadIdx.y) * block_width + static_cast<int>(threadIdx.x);
    const int threads = block_width * block_height;
    const int tile_width = block_width + 2 * params.max_radius_;
    const int tile_height = block_height + 2 * params.max_radius_;
    const int stride = tile_width + 1;
    const int origin_x = static_cast<int>(blockIdx.x) * block_width - params.max_radius_;
    const int origin_y = static_cast<int>(blockIdx.y) * block_height - params.max_radius_;

    for (int index = thread; index < stride * (tile_height + 1); index += threads) {
        const int row = index / stride;
        const int col = index - row * stride;
        int value = 0;
        if (row > 0 && col > 0) {
            const int sy = replicate(origin_y + row - 1, height);
            const int sx = replicate(origin_x + col - 1, width);
            value = src[static_cast<std::size_t>(sy) * src_pitch + static_cast<std::size_t>(sx)];
        }
        prefix[index] = value;
    }
    __syncthreads();

    for (int row = thread + 1; row <= tile_height; row += threads) {
        int sum = 0;
        for (int col = 1; col <= tile_width; ++col) {
            sum += prefix[row * stride + col];
            prefix[row * stride + col] = sum;
        }
    }
    __syncthreads();

    for (int col = thread + 1; col <= tile_width; col += threads) {
        int sum = 0;
        for (int row = 1; row <= tile_height; ++row) {
            sum += prefix[row * stride + col];
            prefix[row * stride + col] = sum;
        }
    }
    __syncthreads();

    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) {
        return;
    }
    const int local_x = static_cast<int>(threadIdx.x) + params.max_radius_;
    const int local_y = static_cast<int>(threadIdx.y) + params.max_radius_;
    const int value = src[static_cast<std::size_t>(y) * src_pitch + static_cast<std::size_t>(x)];
    for (int i = 0; i < params.window_count_; ++i) {
        const int x0 = local_x - params.radius_[i];
        const int x1 = local_x + params.radius_[i] + 1;
        const int y0 = local_y - params.radius_[i];
        const int y1 = local_y + params.radius_[i] + 1;
        const int sum = prefix[y1 * stride + x1] - prefix[y0 * stride + x1] -
                        prefix[y1 * stride + x0] + prefix[y0 * stride + x0];
        int mean = __double2int_rn(static_cast<double>(sum) * params.inverse_area_[i]);
        mean = mean < 0 ? 0 : (mean > 255 ? 255 : mean);
        params.output_[i][static_cast<std::size_t>(y) * params.pitch_[i] +
                          static_cast<std::size_t>(x)] =
                ((value - mean) <= -params.idelta_) ? 255U : 0U;
    }
}
#endif

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
        // OpenCV _threshold rounds an even window up to an odd one.
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
    // OpenCV THRESH_BINARY_INV truncates the constant to an integer.
    const int idelta = static_cast<int>(std::floor(config.adaptive_thresh_constant_));
    const dim3 block = block_dim_of(config);
    const dim3 grid = grid_dim_for(buffers->width_px_, buffers->height_px_, block);

#if ARUCO3CUDA_ENABLE_FUSED_THRESHOLD
    const int largest_window = buffers->window_sizes_px_[buffers->window_count_ - 1];
    const std::size_t shared_width = static_cast<std::size_t>(block.x) + largest_window;
    const std::size_t shared_height = static_cast<std::size_t>(block.y) + largest_window;
    const std::size_t shared_bytes = shared_width * shared_height * sizeof(int);
    if (largest_window <= ARUCO3CUDA_FUSED_THRESHOLD_MAX_WINDOW_PX && shared_bytes <= 48U * 1024U) {
        FusedThresholdParams params;
        params.window_count_ = buffers->window_count_;
        params.max_radius_ = largest_window / 2;
        params.idelta_ = idelta;
        for (int i = 0; i < buffers->window_count_; ++i) {
            const int window = buffers->window_sizes_px_[i];
            params.output_[i] = buffers->binary_[i].data_;
            params.pitch_[i] = buffers->binary_[i].pitch_bytes_;
            params.radius_[i] = window / 2;
            params.inverse_area_[i] = 1.0 / (static_cast<double>(window) * window);
        }
        fused_threshold_kernel<<<grid, block, shared_bytes, stream>>>(
                segmentation.data_, segmentation.pitch_bytes_, buffers->width_px_,
                buffers->height_px_, params);
        return detail::check_kernel_launch("threshold.fused_threshold_kernel", -1, false, stream);
    }
#endif

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

        // The row-sum scratch space is reused across windows, so the windows run in
        // order on the same stream. Splitting them across streams would overwrite
        // the previous window's results.
        const std::int64_t max_sum = 255LL * window * window;
        if (max_sum <= std::numeric_limits<std::int32_t>::max()) {
            column_threshold_kernel<std::int32_t><<<grid, block, 0, stream>>>(
                    segmentation.data_, segmentation.pitch_bytes_, buffers->row_sums_,
                    buffers->row_sums_pitch_bytes_, buffers->binary_[i].data_,
                    buffers->binary_[i].pitch_bytes_, buffers->width_px_, buffers->height_px_,
                    radius, inverse_area, idelta);
        } else {
            column_threshold_kernel<std::int64_t><<<grid, block, 0, stream>>>(
                    segmentation.data_, segmentation.pitch_bytes_, buffers->row_sums_,
                    buffers->row_sums_pitch_bytes_, buffers->binary_[i].data_,
                    buffers->binary_[i].pitch_bytes_, buffers->width_px_, buffers->height_px_,
                    radius, inverse_area, idelta);
        }
        status =
                detail::check_kernel_launch("threshold.column_threshold_kernel", -1, false, stream);
        if (status != Status::kOk) {
            return status;
        }
    }
    return Status::kOk;
}

}  // namespace aruco3cuda::detail
