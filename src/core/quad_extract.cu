// SPDX-License-Identifier: Apache-2.0
#include "quad_extract.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "aruco3cuda/status.hpp"
#include "aruco3cuda/workspace.hpp"
#include "cuda_check.hpp"
#include "labeling.hpp"

namespace aruco3cuda::detail {
namespace {

constexpr std::size_t kPlaneAlignment = 256U;
constexpr int kLinearThreads = 256;

/// 距離と画素 index を 1 語へ詰める。
///
/// atomicMax 1 回で最遠点とその位置を同時に決めるための表現である。
/// 距離が等しい画素が複数ある場合は index の大きい方が残る。順序に
/// 依存しない規則であり、実行ごとに同じ点が選ばれる。
__device__ unsigned long long pack_best(std::uint32_t key, std::int32_t index) {
    return (static_cast<unsigned long long>(key) << 32) |
           static_cast<unsigned long long>(static_cast<std::uint32_t>(index));
}

/// 詰めた語から画素 index を取り出す。
__device__ std::int32_t unpack_index(unsigned long long packed) {
    return static_cast<std::int32_t>(static_cast<std::uint32_t>(packed & 0xFFFFFFFFULL));
}

/// 詰めた語から距離の key を取り出す。
__device__ std::uint32_t unpack_key(unsigned long long packed) {
    return static_cast<std::uint32_t>(packed >> 32);
}

/// 非負の float を、順序を保ったまま 32 bit 整数へ写す。
///
/// IEEE 754 の非負値は bit 列を符号なし整数として比べても大小関係が
/// 変わらない。距離の比較に atomicMax をそのまま使えるようにする。
__device__ std::uint32_t float_order_key(float value) {
    return __float_as_uint(value);
}

/// 探索用の語を 0 で埋める。
__global__ void reset_best_kernel(unsigned long long* best, unsigned long long* best_positive,
                                  unsigned long long* best_negative,
                                  const std::int32_t* label_count) {
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (index >= *label_count) {
        return;
    }
    best[index] = 0ULL;
    best_positive[index] = 0ULL;
    best_negative[index] = 0ULL;
}

/// 重心から最も遠い点を求める。
__global__ void farthest_from_centroid_kernel(const std::int32_t* labels, const float* centroid_x,
                                              const float* centroid_y, unsigned long long* best,
                                              int width, int height) {
    const int x = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const int y = static_cast<int>((blockIdx.y * blockDim.y) + threadIdx.y);
    if (x >= width || y >= height) {
        return;
    }
    const std::int32_t index = (y * width) + x;
    const std::int32_t label = labels[index];
    if (label == kBackgroundLabel) {
        return;
    }
    const float dx = static_cast<float>(x) - centroid_x[label];
    const float dy = static_cast<float>(y) - centroid_y[label];
    atomicMax(&best[label], pack_best(float_order_key((dx * dx) + (dy * dy)), index));
}

/// 指定した点から最も遠い点を求める。
///
/// 起点は画素であり座標が整数のため、二乗距離を整数のまま扱える。
/// 画像の最大寸法でも 2^23 に収まり、32 bit の key へ入る。
__global__ void farthest_from_point_kernel(const std::int32_t* labels, const std::int32_t* origin_x,
                                           const std::int32_t* origin_y, unsigned long long* best,
                                           int width, int height) {
    const int x = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const int y = static_cast<int>((blockIdx.y * blockDim.y) + threadIdx.y);
    if (x >= width || y >= height) {
        return;
    }
    const std::int32_t index = (y * width) + x;
    const std::int32_t label = labels[index];
    if (label == kBackgroundLabel) {
        return;
    }
    const std::int32_t dx = x - origin_x[label];
    const std::int32_t dy = y - origin_y[label];
    const std::int32_t distance = (dx * dx) + (dy * dy);
    atomicMax(&best[label], pack_best(static_cast<std::uint32_t>(distance), index));
}

/// 直線 c0c2 の左右それぞれで最も離れた点を求める。
///
/// 外積の絶対値は、直線からの距離に線分長を掛けたものである。線分長は
/// label ごとに一定であるため、外積の絶対値を最大にする点が最も離れた
/// 点になる。整数のまま扱えるため丸めが入らない。
__global__ void farthest_from_line_kernel(const std::int32_t* labels, const std::int32_t* corner_x,
                                          const std::int32_t* corner_y, int capacity,
                                          unsigned long long* best_positive,
                                          unsigned long long* best_negative, int width,
                                          int height) {
    const int x = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const int y = static_cast<int>((blockIdx.y * blockDim.y) + threadIdx.y);
    if (x >= width || y >= height) {
        return;
    }
    const std::int32_t index = (y * width) + x;
    const std::int32_t label = labels[index];
    if (label == kBackgroundLabel) {
        return;
    }
    const std::int32_t x0 = corner_x[label];
    const std::int32_t y0 = corner_y[label];
    const std::int32_t x2 = corner_x[(2 * capacity) + label];
    const std::int32_t y2 = corner_y[(2 * capacity) + label];
    const long long cross = (static_cast<long long>(x2 - x0) * static_cast<long long>(y - y0)) -
                            (static_cast<long long>(y2 - y0) * static_cast<long long>(x - x0));
    if (cross > 0) {
        atomicMax(&best_positive[label], pack_best(static_cast<std::uint32_t>(cross), index));
    } else if (cross < 0) {
        atomicMax(&best_negative[label], pack_best(static_cast<std::uint32_t>(-cross), index));
    }
    // 直線上の点はどちらの側でもないため、どちらの探索にも寄与しない。
}

/// 探索結果を四隅の座標として書き出す。
__global__ void store_corner_kernel(const unsigned long long* best, std::int32_t* corner_x,
                                    std::int32_t* corner_y, int capacity, int corner, int width,
                                    const std::int32_t* label_count) {
    const int label = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (label >= *label_count) {
        return;
    }
    const std::int32_t index = unpack_index(best[label]);
    corner_x[(corner * capacity) + label] = index % width;
    corner_y[(corner * capacity) + label] = index / width;
}

/// 四隅が定まったかを判定し、並びを一定の向きへ揃える。
///
/// 直線の片側に点が無い成分は四隅が定まらない。1 画素の成分や、
/// 幅 1 画素の直線状の成分がこれにあたる。
__global__ void finalize_quad_kernel(const unsigned long long* best_positive,
                                     const unsigned long long* best_negative,
                                     std::int32_t* corner_x, std::int32_t* corner_y,
                                     std::uint8_t* valid, int capacity,
                                     const std::int32_t* label_count) {
    const int label = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (label >= *label_count) {
        return;
    }
    if (unpack_key(best_positive[label]) == 0U || unpack_key(best_negative[label]) == 0U) {
        valid[label] = 0U;
        return;
    }
    std::int32_t xs[kQuadCornerCount];
    std::int32_t ys[kQuadCornerCount];
    for (int corner = 0; corner < kQuadCornerCount; ++corner) {
        xs[corner] = corner_x[(corner * capacity) + label];
        ys[corner] = corner_y[(corner * capacity) + label];
    }
    // 4 点が相異なることを確かめる。潰れた四角形は候補にしない。
    for (int a = 0; a < kQuadCornerCount; ++a) {
        for (int b = a + 1; b < kQuadCornerCount; ++b) {
            if (xs[a] == xs[b] && ys[a] == ys[b]) {
                valid[label] = 0U;
                return;
            }
        }
    }
    // OpenCV の _reorderCandidatesCorners と同じ向きへ揃える。
    const long long cross =
            (static_cast<long long>(xs[1] - xs[0]) * static_cast<long long>(ys[2] - ys[0])) -
            (static_cast<long long>(ys[1] - ys[0]) * static_cast<long long>(xs[2] - xs[0]));
    if (cross < 0) {
        const std::int32_t swap_x = xs[1];
        const std::int32_t swap_y = ys[1];
        xs[1] = xs[3];
        ys[1] = ys[3];
        xs[3] = swap_x;
        ys[3] = swap_y;
    }
    for (int corner = 0; corner < kQuadCornerCount; ++corner) {
        corner_x[(corner * capacity) + label] = xs[corner];
        corner_y[(corner * capacity) + label] = ys[corner];
    }
    valid[label] = 1U;
}

}  // namespace

std::size_t quad_workspace_bytes(int width_px, int height_px) {
    const int capacity = max_label_count(width_px, height_px);
    if (capacity == 0) {
        return 0U;
    }
    const auto count = static_cast<std::size_t>(capacity);
    if (count > std::numeric_limits<std::size_t>::max() / kQuadCornerCount) {
        return 0U;
    }
    const std::size_t corner_bytes =
            align_up(count * kQuadCornerCount * sizeof(std::int32_t), kPlaneAlignment);
    const std::size_t valid_bytes = align_up(count * sizeof(std::uint8_t), kPlaneAlignment);
    const std::size_t best_bytes = align_up(count * sizeof(unsigned long long), kPlaneAlignment);
    if (corner_bytes == 0U || valid_bytes == 0U || best_bytes == 0U) {
        return 0U;
    }
    return (corner_bytes * 2U) + valid_bytes + (best_bytes * 3U);
}

Status reserve_quads(int width_px, int height_px, Workspace& workspace, QuadBuffers* out) {
    if (out == nullptr) {
        return Status::kInvalidArgument;
    }
    if (width_px <= 0 || height_px <= 0) {
        return Status::kInvalidArgument;
    }
    const int capacity = max_label_count(width_px, height_px);
    if (capacity == 0 || quad_workspace_bytes(width_px, height_px) == 0U) {
        return Status::kInvalidConfig;
    }

    QuadBuffers buffers;
    buffers.capacity_ = capacity;
    const auto count = static_cast<std::size_t>(capacity);

    std::int32_t** corner_targets[] = {&buffers.corner_x_, &buffers.corner_y_};
    for (std::int32_t** target : corner_targets) {
        void* pointer = nullptr;
        const Status status = workspace.allocate(count * kQuadCornerCount * sizeof(std::int32_t),
                                                 kPlaneAlignment, &pointer);
        if (status != Status::kOk) {
            return status;
        }
        *target = static_cast<std::int32_t*>(pointer);
    }
    void* pointer = nullptr;
    Status status = workspace.allocate(count * sizeof(std::uint8_t), kPlaneAlignment, &pointer);
    if (status != Status::kOk) {
        return status;
    }
    buffers.valid_ = static_cast<std::uint8_t*>(pointer);

    unsigned long long** best_targets[] = {&buffers.best_, &buffers.best_positive_,
                                           &buffers.best_negative_};
    for (unsigned long long** target : best_targets) {
        status = workspace.allocate(count * sizeof(unsigned long long), kPlaneAlignment, &pointer);
        if (status != Status::kOk) {
            return status;
        }
        *target = static_cast<unsigned long long*>(pointer);
    }

    *out = buffers;
    return Status::kOk;
}

Status build_quads_async(const LabelBuffers& labels, const LabelStatisticsBuffers& stats,
                         QuadBuffers* quads, cudaStream_t stream) {
    if (quads == nullptr || quads->corner_x_ == nullptr || labels.labels_ == nullptr ||
        labels.label_count_ == nullptr || stats.centroid_x_ == nullptr) {
        return Status::kInvalidArgument;
    }
    if (quads->capacity_ != stats.capacity_) {
        return Status::kInvalidArgument;
    }

    const int width = labels.width_px_;
    const int height = labels.height_px_;
    const int capacity = quads->capacity_;
    const int linear_blocks = (capacity + kLinearThreads - 1) / kLinearThreads;
    const auto linear_grid = static_cast<unsigned int>(linear_blocks);
    const auto linear_block = static_cast<unsigned int>(kLinearThreads);
    const dim3 block(16U, 16U, 1U);
    const dim3 grid((static_cast<unsigned int>(width) + block.x - 1U) / block.x,
                    (static_cast<unsigned int>(height) + block.y - 1U) / block.y, 1U);

    reset_best_kernel<<<linear_grid, linear_block, 0, stream>>>(
            quads->best_, quads->best_positive_, quads->best_negative_, labels.label_count_);
    Status status = check_kernel_launch("quad.reset_best_kernel", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }

    // 第 1 段: 重心から最も遠い点を c0 とする。
    farthest_from_centroid_kernel<<<grid, block, 0, stream>>>(
            labels.labels_, stats.centroid_x_, stats.centroid_y_, quads->best_, width, height);
    status = check_kernel_launch("quad.farthest_from_centroid_kernel", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    store_corner_kernel<<<linear_grid, linear_block, 0, stream>>>(quads->best_, quads->corner_x_,
                                                                  quads->corner_y_, capacity, 0,
                                                                  width, labels.label_count_);
    status = check_kernel_launch("quad.store_corner_kernel.c0", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }

    // 第 2 段: c0 から最も遠い点を c2 とする。
    reset_best_kernel<<<linear_grid, linear_block, 0, stream>>>(
            quads->best_, quads->best_positive_, quads->best_negative_, labels.label_count_);
    status = check_kernel_launch("quad.reset_best_kernel.c2", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    farthest_from_point_kernel<<<grid, block, 0, stream>>>(
            labels.labels_, quads->corner_x_, quads->corner_y_, quads->best_, width, height);
    status = check_kernel_launch("quad.farthest_from_point_kernel", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    store_corner_kernel<<<linear_grid, linear_block, 0, stream>>>(quads->best_, quads->corner_x_,
                                                                  quads->corner_y_, capacity, 2,
                                                                  width, labels.label_count_);
    status = check_kernel_launch("quad.store_corner_kernel.c2", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }

    // 第 3 段: 直線 c0c2 の左右で最も離れた点を c1 と c3 とする。
    farthest_from_line_kernel<<<grid, block, 0, stream>>>(
            labels.labels_, quads->corner_x_, quads->corner_y_, capacity, quads->best_positive_,
            quads->best_negative_, width, height);
    status = check_kernel_launch("quad.farthest_from_line_kernel", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    store_corner_kernel<<<linear_grid, linear_block, 0, stream>>>(
            quads->best_positive_, quads->corner_x_, quads->corner_y_, capacity, 1, width,
            labels.label_count_);
    status = check_kernel_launch("quad.store_corner_kernel.c1", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    store_corner_kernel<<<linear_grid, linear_block, 0, stream>>>(
            quads->best_negative_, quads->corner_x_, quads->corner_y_, capacity, 3, width,
            labels.label_count_);
    status = check_kernel_launch("quad.store_corner_kernel.c3", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }

    finalize_quad_kernel<<<linear_grid, linear_block, 0, stream>>>(
            quads->best_positive_, quads->best_negative_, quads->corner_x_, quads->corner_y_,
            quads->valid_, capacity, labels.label_count_);
    return check_kernel_launch("quad.finalize_quad_kernel", -1, false, stream);
}

}  // namespace aruco3cuda::detail
