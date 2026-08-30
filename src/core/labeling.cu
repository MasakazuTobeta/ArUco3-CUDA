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
#include "scan.hpp"

namespace aruco3cuda::detail {
namespace {

constexpr std::size_t kPlaneAlignment = 256U;
/// Threads per block for the linear-traversal kernels.
constexpr int kLinearThreads = 256;
/// Initial values of the bounding box. Used as the identity elements for
/// atomicMin and atomicMax.
///
/// std::numeric_limits cannot be called from device code, so the values are
/// spelled out directly.
constexpr std::int32_t kBoundsMinInitial = 2147483647;
constexpr std::int32_t kBoundsMaxInitial = -2147483647 - 1;

/// Walks the label array to the root.
///
/// Even when another thread rewrites a parent mid-merge, the parent value only
/// ever moves toward smaller values. The chain to walk is therefore finite, and
/// this cannot loop forever.
__device__ std::int32_t find_root(const std::int32_t* labels, std::int32_t index) {
    while (labels[index] != index) {
        index = labels[index];
    }
    return index;
}

/// Merges two components.
///
/// The smaller index becomes the root. When the return value of atomicMin is what
/// was expected, the merge took effect; otherwise another thread rewrote the entry
/// first, so the walk restarts from that value.
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

/// Stores its own linear index in each foreground pixel and kBackgroundLabel in
/// each background pixel.
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

/// Merges with the 4 neighbors that precede this pixel in scan order.
///
/// Every edge of the 8-neighborhood has two endpoints. Letting only the endpoint
/// that comes later perform the merge processes each edge exactly once.
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

/// Replaces each pixel's label with its root.
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

/// Writes 1 for pixels that are roots and 0 for the rest.
__global__ void flag_roots_kernel(const std::int32_t* labels, std::int32_t* flags, int count) {
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (index >= count) {
        return;
    }
    flags[index] = (labels[index] == index) ? 1 : 0;
}

/// Replaces the root's linear index with the compacted label.
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

/// Resets the statistics to their initial aggregation values.
///
/// Reads the label count on the device side and touches only the range actually in
/// use. The allocation is sized for the upper bound, so initializing all of it
/// would waste a great deal of work.
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

/// Traverses the pixels and gathers the bounding box, pixel count, and coordinate
/// sums per label.
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

/// Derives the centroid from the coordinate sums and the pixel count.
///
/// Performs the division in double and then narrows to float. The sum is at most
/// the pixel count times the coordinate, which a float mantissa cannot represent
/// exactly.
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

/// Computes the byte count of an int32 array. Returns 0 on overflow.
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
        // The linear index is stored in the label, so a pixel count beyond the range
        // of an int cannot be handled.
        return 0U;
    }
    const std::size_t plane = array_bytes_i32(count);
    if (plane == 0U) {
        return 0U;
    }
    const std::size_t blocks =
            align_up(scan_workspace_bytes(static_cast<int>(count)), kPlaneAlignment);
    const std::size_t label_counter = array_bytes_i32(1U);
    if (blocks == 0U || label_counter == 0U) {
        return 0U;
    }
    return (plane * 2U) + blocks + label_counter;
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

    status = reserve_scan(static_cast<int>(count), workspace, &buffers.scan_);
    if (status != Status::kOk) {
        return status;
    }
    // The scan's grand total is the label count as is. There is no need to count
    // them again separately.
    buffers.label_count_ = buffers.scan_.total_;

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

    Status status = Status::kOk;
    initialize_labels_kernel<<<grid, block, 0, stream>>>(binary.data_, binary.pitch_bytes_,
                                                         buffers->labels_, width, height);
    status = check_kernel_launch("labeling.initialize_labels_kernel", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    // The initial labels double as the read side. A merge rewrites only the root
    // entries, so the foreground-versus-background test remains correct against the
    // initial values.
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
    status = exclusive_scan_async(buffers->compact_ids_, count, &buffers->scan_, stream);
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
    // 5 int32 arrays, 2 uint64 arrays, and 2 float arrays.
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
        // A combination where the labels could exceed the allocated count. This
        // cannot happen when the allocation is sized for the upper bound.
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
