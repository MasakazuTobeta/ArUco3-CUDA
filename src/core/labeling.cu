// SPDX-License-Identifier: Apache-2.0
#include "labeling.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "aruco3cuda/status.hpp"
#include "aruco3cuda/workspace.hpp"
#include "cuda_check.hpp"
#include "preprocess.hpp"

namespace aruco3cuda::detail {
namespace {

constexpr std::size_t kPlaneAlignment = 256U;
/// scan 1 block あたりの thread 数と要素数。
///
/// 共有 memory 上の走査を 1 要素 1 thread で行うため両者は等しい。
constexpr int kScanThreads = 256;
/// 走査系 kernel の block あたり thread 数。
constexpr int kLinearThreads = 256;
/// 外接矩形の初期値。atomicMin と atomicMax の単位元として使う。
///
/// std::numeric_limits は device code から呼べないため、値を直接置く。
constexpr std::int32_t kBoundsMinInitial = 2147483647;
constexpr std::int32_t kBoundsMaxInitial = -2147483647 - 1;

/// label 配列の根を辿る。
///
/// 併合中の他 thread が親を書き換えても、親の値は必ず小さくなる方向へ
/// しか動かない。そのため辿る先は有限で、無限 loop にはならない。
__device__ std::int32_t find_root(const std::int32_t* labels, std::int32_t index) {
    while (labels[index] != index) {
        index = labels[index];
    }
    return index;
}

/// 2 つの成分を併合する。
///
/// 小さい index を根とする。atomicMin の戻り値が期待どおりなら併合が
/// 成立し、そうでなければ他 thread が先に書き換えているため、その値から
/// 辿り直す。
__device__ void merge_components(std::int32_t* labels, std::int32_t a, std::int32_t b) {
    while (a != b) {
        a = find_root(labels, a);
        b = find_root(labels, b);
        if (a < b) {
            const std::int32_t previous = atomicMin(&labels[b], a);
            if (previous == b) {
                return;
            }
            b = previous;
        } else if (b < a) {
            const std::int32_t previous = atomicMin(&labels[a], b);
            if (previous == a) {
                return;
            }
            a = previous;
        }
    }
}

/// 前景画素へ自分の線形 index を、背景へ kBackgroundLabel を入れる。
__global__ void initialize_labels_kernel(const std::uint8_t* binary, std::size_t binary_pitch,
                                         std::int32_t* labels, int width, int height) {
    const int x = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const int y = static_cast<int>((blockIdx.y * blockDim.y) + threadIdx.y);
    if (x >= width || y >= height) {
        return;
    }
    const std::int32_t index = (y * width) + x;
    const std::uint8_t value =
            binary[(static_cast<std::size_t>(y) * binary_pitch) + static_cast<std::size_t>(x)];
    labels[index] = (value != 0U) ? index : kBackgroundLabel;
}

/// 走査順で手前にある 4 近傍と併合する。
///
/// 8 近傍の辺は 2 つの端点を持つ。後から現れる側だけが併合を行えば、
/// 全ての辺をちょうど 1 回ずつ処理できる。
__global__ void merge_neighbors_kernel(const std::int32_t* labels_in, std::int32_t* labels,
                                       int width, int height) {
    const int x = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const int y = static_cast<int>((blockIdx.y * blockDim.y) + threadIdx.y);
    if (x >= width || y >= height) {
        return;
    }
    const std::int32_t index = (y * width) + x;
    if (labels_in[index] == kBackgroundLabel) {
        return;
    }
    const int offsets[4][2] = {{-1, 0}, {-1, -1}, {0, -1}, {1, -1}};
    for (const auto& offset : offsets) {
        const int nx = x + offset[0];
        const int ny = y + offset[1];
        if (nx < 0 || nx >= width || ny < 0) {
            continue;
        }
        const std::int32_t neighbor = (ny * width) + nx;
        if (labels_in[neighbor] == kBackgroundLabel) {
            continue;
        }
        merge_components(labels, index, neighbor);
    }
}

/// 各画素の label を根へ置き換える。
__global__ void compress_labels_kernel(std::int32_t* labels, int count) {
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (index >= count) {
        return;
    }
    if (labels[index] == kBackgroundLabel) {
        return;
    }
    labels[index] = find_root(labels, index);
}

/// 根である画素へ 1、それ以外へ 0 を書く。
__global__ void flag_roots_kernel(const std::int32_t* labels, std::int32_t* flags, int count) {
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (index >= count) {
        return;
    }
    flags[index] = (labels[index] == index) ? 1 : 0;
}

/// block ごとに排他 scan を行い、block の合計を block_sums へ書く。
///
/// 入出力に同じ配列を渡してよい。各 thread は書き込みの前に自分の値を
/// register へ読み出し、他 thread の位置へは書かないためである。
__global__ void scan_within_block_kernel(std::int32_t* values, std::int32_t* block_sums,
                                         int count) {
    __shared__ std::int32_t shared[kScanThreads];
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const std::int32_t value = (index < count) ? values[index] : 0;
    shared[threadIdx.x] = value;
    __syncthreads();
    for (int offset = 1; offset < kScanThreads; offset <<= 1) {
        std::int32_t addend = 0;
        if (static_cast<int>(threadIdx.x) >= offset) {
            addend = shared[threadIdx.x - static_cast<unsigned int>(offset)];
        }
        __syncthreads();
        shared[threadIdx.x] += addend;
        __syncthreads();
    }
    if (index < count) {
        values[index] = shared[threadIdx.x] - value;
    }
    if (threadIdx.x == static_cast<unsigned int>(kScanThreads - 1)) {
        block_sums[blockIdx.x] = shared[threadIdx.x];
    }
}

/// block の合計を 1 block で排他 scan し、総和を out_total へ書く。
///
/// block 数が thread 数を超える場合は塊ごとに走査し、直前までの総和を
/// 繰り越す。block 数は画素数の 1/256 であり、この逐次化は問題にならない。
__global__ void scan_block_sums_kernel(std::int32_t* block_sums, int block_count,
                                       std::int32_t* out_total) {
    __shared__ std::int32_t shared[kScanThreads];
    std::int32_t running = 0;
    for (int base = 0; base < block_count; base += kScanThreads) {
        const int index = base + static_cast<int>(threadIdx.x);
        const std::int32_t value = (index < block_count) ? block_sums[index] : 0;
        shared[threadIdx.x] = value;
        __syncthreads();
        for (int offset = 1; offset < kScanThreads; offset <<= 1) {
            std::int32_t addend = 0;
            if (static_cast<int>(threadIdx.x) >= offset) {
                addend = shared[threadIdx.x - static_cast<unsigned int>(offset)];
            }
            __syncthreads();
            shared[threadIdx.x] += addend;
            __syncthreads();
        }
        if (index < block_count) {
            block_sums[index] = running + shared[threadIdx.x] - value;
        }
        const std::int32_t chunk_total = shared[kScanThreads - 1];
        __syncthreads();
        running += chunk_total;
    }
    if (threadIdx.x == 0U) {
        *out_total = running;
    }
}

/// block ごとの開始位置を各要素へ加える。
__global__ void add_block_offsets_kernel(std::int32_t* values, const std::int32_t* block_offsets,
                                         int count) {
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (index >= count) {
        return;
    }
    values[index] += block_offsets[blockIdx.x];
}

/// 根の線形 index を詰めた label へ置き換える。
__global__ void relabel_kernel(std::int32_t* labels, const std::int32_t* compact_ids, int count) {
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (index >= count) {
        return;
    }
    const std::int32_t root = labels[index];
    if (root == kBackgroundLabel) {
        return;
    }
    labels[index] = compact_ids[root];
}

/// 統計を初期集約値へ戻す。
///
/// label 数を device 側で読み、使う範囲だけ触る。上限で確保しているため
/// 全体を初期化すると無駄が大きい。
__global__ void reset_stats_kernel(LabelStatisticsBuffers stats, const std::int32_t* label_count) {
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (index >= *label_count) {
        return;
    }
    stats.min_x_[index] = kBoundsMinInitial;
    stats.min_y_[index] = kBoundsMinInitial;
    stats.max_x_[index] = kBoundsMaxInitial;
    stats.max_y_[index] = kBoundsMaxInitial;
    stats.pixel_count_[index] = 0;
    stats.sum_x_[index] = 0ULL;
    stats.sum_y_[index] = 0ULL;
    stats.centroid_x_[index] = 0.0F;
    stats.centroid_y_[index] = 0.0F;
}

/// 画素を走査し、label ごとの外接矩形と画素数と座標和を集める。
__global__ void accumulate_stats_kernel(const std::int32_t* labels, LabelStatisticsBuffers stats,
                                        int width, int height) {
    const int x = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const int y = static_cast<int>((blockIdx.y * blockDim.y) + threadIdx.y);
    if (x >= width || y >= height) {
        return;
    }
    const std::int32_t label = labels[(y * width) + x];
    if (label == kBackgroundLabel) {
        return;
    }
    atomicMin(&stats.min_x_[label], x);
    atomicMin(&stats.min_y_[label], y);
    atomicMax(&stats.max_x_[label], x);
    atomicMax(&stats.max_y_[label], y);
    atomicAdd(&stats.pixel_count_[label], 1);
    atomicAdd(&stats.sum_x_[label], static_cast<unsigned long long>(x));
    atomicAdd(&stats.sum_y_[label], static_cast<unsigned long long>(y));
}

/// 座標和と画素数から重心を求める。
///
/// 割り算を double で行ってから float へ落とす。総和は最大で
/// 画素数 x 座標であり、float の仮数では正確に表せない。
__global__ void finalize_stats_kernel(LabelStatisticsBuffers stats,
                                      const std::int32_t* label_count) {
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (index >= *label_count) {
        return;
    }
    const std::int32_t count = stats.pixel_count_[index];
    if (count <= 0) {
        return;
    }
    stats.centroid_x_[index] = static_cast<float>(static_cast<double>(stats.sum_x_[index]) /
                                                  static_cast<double>(count));
    stats.centroid_y_[index] = static_cast<float>(static_cast<double>(stats.sum_y_[index]) /
                                                  static_cast<double>(count));
}

/// 画素数から scan の block 数を求める。
int scan_block_count_of(int count) {
    return (count + kScanThreads - 1) / kScanThreads;
}

/// int32 配列の byte 数を求める。桁溢れなら 0 を返す。
std::size_t array_bytes_i32(std::size_t count) {
    if (count == 0U) {
        return 0U;
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(std::int32_t)) {
        return 0U;
    }
    return align_up(count * sizeof(std::int32_t), kPlaneAlignment);
}

}  // namespace

std::size_t labeling_workspace_bytes(int width_px, int height_px) {
    if (width_px <= 0 || height_px <= 0) {
        return 0U;
    }
    const auto width = static_cast<std::size_t>(width_px);
    const auto height = static_cast<std::size_t>(height_px);
    if (width > std::numeric_limits<std::size_t>::max() / height) {
        return 0U;
    }
    const std::size_t count = width * height;
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        // label へ線形 index を入れるため、画素数が int の範囲を超えると扱えない。
        return 0U;
    }
    const std::size_t plane = array_bytes_i32(count);
    if (plane == 0U) {
        return 0U;
    }
    const std::size_t blocks =
            array_bytes_i32(static_cast<std::size_t>(scan_block_count_of(static_cast<int>(count))));
    const std::size_t total_counter = array_bytes_i32(1U);
    if (blocks == 0U || total_counter == 0U) {
        return 0U;
    }
    return (plane * 2U) + blocks + total_counter;
}

Status reserve_labeling(int width_px, int height_px, Workspace& workspace, LabelBuffers* out) {
    if (out == nullptr) {
        return Status::kInvalidArgument;
    }
    if (width_px <= 0 || height_px <= 0) {
        return Status::kInvalidArgument;
    }
    if (labeling_workspace_bytes(width_px, height_px) == 0U) {
        return Status::kInvalidConfig;
    }

    LabelBuffers buffers;
    buffers.width_px_ = width_px;
    buffers.height_px_ = height_px;
    const std::size_t count =
            static_cast<std::size_t>(width_px) * static_cast<std::size_t>(height_px);
    const std::size_t plane_bytes = count * sizeof(std::int32_t);

    void* pointer = nullptr;
    Status status = workspace.allocate(plane_bytes, kPlaneAlignment, &pointer);
    if (status != Status::kOk) {
        return status;
    }
    buffers.labels_ = static_cast<std::int32_t*>(pointer);

    status = workspace.allocate(plane_bytes, kPlaneAlignment, &pointer);
    if (status != Status::kOk) {
        return status;
    }
    buffers.compact_ids_ = static_cast<std::int32_t*>(pointer);

    buffers.scan_block_count_ = scan_block_count_of(static_cast<int>(count));
    status = workspace.allocate(
            static_cast<std::size_t>(buffers.scan_block_count_) * sizeof(std::int32_t),
            kPlaneAlignment, &pointer);
    if (status != Status::kOk) {
        return status;
    }
    buffers.block_offsets_ = static_cast<std::int32_t*>(pointer);

    status = workspace.allocate(sizeof(std::int32_t), kPlaneAlignment, &pointer);
    if (status != Status::kOk) {
        return status;
    }
    buffers.label_count_ = static_cast<std::int32_t*>(pointer);

    *out = buffers;
    return Status::kOk;
}

Status build_labels_async(const ImagePlaneU8& binary, LabelBuffers* buffers, cudaStream_t stream) {
    if (buffers == nullptr || buffers->labels_ == nullptr || binary.data_ == nullptr) {
        return Status::kInvalidArgument;
    }
    if (binary.width_px_ != buffers->width_px_ || binary.height_px_ != buffers->height_px_) {
        return Status::kInvalidArgument;
    }

    const int width = buffers->width_px_;
    const int height = buffers->height_px_;
    const int count = width * height;
    const dim3 block(16U, 16U, 1U);
    const dim3 grid((static_cast<unsigned int>(width) + block.x - 1U) / block.x,
                    (static_cast<unsigned int>(height) + block.y - 1U) / block.y, 1U);
    const int linear_blocks = (count + kLinearThreads - 1) / kLinearThreads;
    const int scan_blocks = buffers->scan_block_count_;

    Status status = Status::kOk;
    initialize_labels_kernel<<<grid, block, 0, stream>>>(binary.data_, binary.pitch_bytes_,
                                                         buffers->labels_, width, height);
    status = check_kernel_launch("labeling.initialize_labels_kernel", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    // 初期 label を読み出しにも使う。併合で書き換わるのは根の側だけであり、
    // 前景か背景かの判定は初期値のままで正しい。
    merge_neighbors_kernel<<<grid, block, 0, stream>>>(buffers->labels_, buffers->labels_, width,
                                                       height);
    status = check_kernel_launch("labeling.merge_neighbors_kernel", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    compress_labels_kernel<<<static_cast<unsigned int>(linear_blocks),
                             static_cast<unsigned int>(kLinearThreads), 0, stream>>>(
            buffers->labels_, count);
    status = check_kernel_launch("labeling.compress_labels_kernel", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    flag_roots_kernel<<<static_cast<unsigned int>(linear_blocks),
                        static_cast<unsigned int>(kLinearThreads), 0, stream>>>(
            buffers->labels_, buffers->compact_ids_, count);
    status = check_kernel_launch("labeling.flag_roots_kernel", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    scan_within_block_kernel<<<static_cast<unsigned int>(scan_blocks),
                               static_cast<unsigned int>(kScanThreads), 0, stream>>>(
            buffers->compact_ids_, buffers->block_offsets_, count);
    status = check_kernel_launch("labeling.scan_within_block_kernel", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    scan_block_sums_kernel<<<1U, static_cast<unsigned int>(kScanThreads), 0, stream>>>(
            buffers->block_offsets_, scan_blocks, buffers->label_count_);
    status = check_kernel_launch("labeling.scan_block_sums_kernel", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    add_block_offsets_kernel<<<static_cast<unsigned int>(scan_blocks),
                               static_cast<unsigned int>(kScanThreads), 0, stream>>>(
            buffers->compact_ids_, buffers->block_offsets_, count);
    status = check_kernel_launch("labeling.add_block_offsets_kernel", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    relabel_kernel<<<static_cast<unsigned int>(linear_blocks),
                     static_cast<unsigned int>(kLinearThreads), 0, stream>>>(
            buffers->labels_, buffers->compact_ids_, count);
    status = check_kernel_launch("labeling.relabel_kernel", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    return Status::kOk;
}

Status read_label_count(const LabelBuffers& buffers, int* out_count, cudaStream_t stream) {
    if (out_count == nullptr || buffers.label_count_ == nullptr) {
        return Status::kInvalidArgument;
    }
    std::int32_t value = 0;
    Status status = check_cuda(cudaMemcpyAsync(&value, buffers.label_count_, sizeof(std::int32_t),
                                               cudaMemcpyDeviceToHost, stream),
                               "cudaMemcpyAsync", "labeling.read_label_count", -1, stream);
    if (status != Status::kOk) {
        return status;
    }
    status = check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize",
                        "labeling.read_label_count", -1, stream);
    if (status != Status::kOk) {
        return status;
    }
    *out_count = static_cast<int>(value);
    return Status::kOk;
}

int max_label_count(int width_px, int height_px) {
    if (width_px <= 0 || height_px <= 0) {
        return 0;
    }
    const long long columns = (static_cast<long long>(width_px) + 1) / 2;
    const long long rows = (static_cast<long long>(height_px) + 1) / 2;
    const long long total = columns * rows;
    if (total > static_cast<long long>(std::numeric_limits<int>::max())) {
        return 0;
    }
    return static_cast<int>(total);
}

std::size_t label_stats_workspace_bytes(int width_px, int height_px) {
    const int capacity = max_label_count(width_px, height_px);
    if (capacity == 0) {
        return 0U;
    }
    const auto count = static_cast<std::size_t>(capacity);
    // int32 が 5 本、uint64 が 2 本、float が 2 本。
    const std::size_t int32_bytes = array_bytes_i32(count);
    const std::size_t sum_bytes = align_up(count * sizeof(unsigned long long), kPlaneAlignment);
    const std::size_t float_bytes = align_up(count * sizeof(float), kPlaneAlignment);
    if (int32_bytes == 0U || sum_bytes == 0U || float_bytes == 0U) {
        return 0U;
    }
    return (int32_bytes * 5U) + (sum_bytes * 2U) + (float_bytes * 2U);
}

Status reserve_label_stats(int width_px, int height_px, Workspace& workspace,
                           LabelStatisticsBuffers* out) {
    if (out == nullptr) {
        return Status::kInvalidArgument;
    }
    if (width_px <= 0 || height_px <= 0) {
        return Status::kInvalidArgument;
    }
    const int capacity = max_label_count(width_px, height_px);
    if (capacity == 0 || label_stats_workspace_bytes(width_px, height_px) == 0U) {
        return Status::kInvalidConfig;
    }

    LabelStatisticsBuffers buffers;
    buffers.capacity_ = capacity;
    const auto count = static_cast<std::size_t>(capacity);

    std::int32_t** int32_targets[] = {&buffers.min_x_, &buffers.min_y_, &buffers.max_x_,
                                      &buffers.max_y_, &buffers.pixel_count_};
    for (std::int32_t** target : int32_targets) {
        void* pointer = nullptr;
        const Status status =
                workspace.allocate(count * sizeof(std::int32_t), kPlaneAlignment, &pointer);
        if (status != Status::kOk) {
            return status;
        }
        *target = static_cast<std::int32_t*>(pointer);
    }
    unsigned long long** sum_targets[] = {&buffers.sum_x_, &buffers.sum_y_};
    for (unsigned long long** target : sum_targets) {
        void* pointer = nullptr;
        const Status status =
                workspace.allocate(count * sizeof(unsigned long long), kPlaneAlignment, &pointer);
        if (status != Status::kOk) {
            return status;
        }
        *target = static_cast<unsigned long long*>(pointer);
    }
    float** float_targets[] = {&buffers.centroid_x_, &buffers.centroid_y_};
    for (float** target : float_targets) {
        void* pointer = nullptr;
        const Status status = workspace.allocate(count * sizeof(float), kPlaneAlignment, &pointer);
        if (status != Status::kOk) {
            return status;
        }
        *target = static_cast<float*>(pointer);
    }

    *out = buffers;
    return Status::kOk;
}

Status build_label_stats_async(const LabelBuffers& labels, LabelStatisticsBuffers* stats,
                               cudaStream_t stream) {
    if (stats == nullptr || stats->min_x_ == nullptr || labels.labels_ == nullptr ||
        labels.label_count_ == nullptr) {
        return Status::kInvalidArgument;
    }
    if (stats->capacity_ < max_label_count(labels.width_px_, labels.height_px_)) {
        // label が確保数を超えうる組み合わせ。上限で確保していれば起きない。
        return Status::kInvalidArgument;
    }

    const int width = labels.width_px_;
    const int height = labels.height_px_;
    const int capacity = stats->capacity_;
    const int linear_blocks = (capacity + kLinearThreads - 1) / kLinearThreads;
    const dim3 block(16U, 16U, 1U);
    const dim3 grid((static_cast<unsigned int>(width) + block.x - 1U) / block.x,
                    (static_cast<unsigned int>(height) + block.y - 1U) / block.y, 1U);

    reset_stats_kernel<<<static_cast<unsigned int>(linear_blocks),
                         static_cast<unsigned int>(kLinearThreads), 0, stream>>>(
            *stats, labels.label_count_);
    Status status = check_kernel_launch("labeling.reset_stats_kernel", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    accumulate_stats_kernel<<<grid, block, 0, stream>>>(labels.labels_, *stats, width, height);
    status = check_kernel_launch("labeling.accumulate_stats_kernel", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    finalize_stats_kernel<<<static_cast<unsigned int>(linear_blocks),
                            static_cast<unsigned int>(kLinearThreads), 0, stream>>>(
            *stats, labels.label_count_);
    return check_kernel_launch("labeling.finalize_stats_kernel", -1, false, stream);
}

}  // namespace aruco3cuda::detail
