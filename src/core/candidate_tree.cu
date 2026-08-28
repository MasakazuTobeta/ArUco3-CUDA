// SPDX-License-Identifier: Apache-2.0
#include "candidate_tree.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/workspace.hpp"
#include "candidate_filter.hpp"
#include "cuda_check.hpp"
#include "dictionary_match.hpp"
#include "quad_extract.hpp"

namespace aruco3cuda::detail {
namespace {

constexpr std::size_t kPlaneAlignment = 256U;
constexpr int kResetThreads = 256;
/// 打ち切りの走査は 1 block で行う。段ごとに block 全体の同期が要る。
constexpr int kSuppressThreads = 256;

/// 1 平面の byte 数を求める。桁溢れしたら 0 を返す。
std::size_t plane_bytes(std::size_t count, std::size_t element_bytes) {
    if (count != 0U && element_bytes > std::numeric_limits<std::size_t>::max() / count) {
        return 0U;
    }
    return align_up(count * element_bytes, kPlaneAlignment);
}

/// 点が四角形の内側か境界上にあるかを判定する。
///
/// OpenCV の pointPolygonTest を measureDist = false で呼んだ場合と同じにする。
/// 走査は辺 (c3 から c0)、(c0 から c1)、(c1 から c2)、(c2 から c3) の順である。
/// 交差積は 64 bit で計算する。座標が整数であれば OpenCV が倍精度で計算した
/// 符号と厳密に一致する。凸性は仮定しない。極点探索で作る四角形は凹に
/// なりうるため、符号の一致だけで判定すると結果が変わる。
__device__ bool point_in_quad(const std::int32_t* quad_x, const std::int32_t* quad_y,
                              std::int32_t point_x, std::int32_t point_y) {
    int counter = 0;
    std::int32_t vx = quad_x[kQuadCornerCount - 1];
    std::int32_t vy = quad_y[kQuadCornerCount - 1];
    for (int i = 0; i < kQuadCornerCount; ++i) {
        const std::int32_t v0x = vx;
        const std::int32_t v0y = vy;
        vx = quad_x[i];
        vy = quad_y[i];

        if ((v0y <= point_y && vy <= point_y) || (v0y > point_y && vy > point_y) ||
            (v0x < point_x && vx < point_x)) {
            // 走査線と交わらない辺。ただし頂点や水平な辺の上に乗る場合は内側。
            if (point_y == vy &&
                (point_x == vx || (point_y == v0y && ((v0x <= point_x && point_x <= vx) ||
                                                      (vx <= point_x && point_x <= v0x))))) {
                return true;
            }
            continue;
        }
        long long distance =
                (static_cast<long long>(point_y - v0y) * static_cast<long long>(vx - v0x)) -
                (static_cast<long long>(point_x - v0x) * static_cast<long long>(vy - v0y));
        if (distance == 0) {
            // 辺の上にある。境界は内側として扱う。
            return true;
        }
        if (vy < v0y) {
            distance = -distance;
        }
        counter += (distance > 0) ? 1 : 0;
    }
    return (counter % 2) != 0;
}

/// buffer を初期化する。候補数は device 上にしかないため上限まで埋める。
__global__ void reset_tree_kernel(std::int32_t* parent, std::int32_t* depth, std::int32_t* visited,
                                  int capacity) {
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (index >= capacity) {
        return;
    }
    parent[index] = -1;
    depth[index] = 0;
    visited[index] = 0;
}

/// 候補ごとに親を決める。1 thread が 1 候補を担当する。
///
/// OpenCV は j を i-1 から降順に見て最初に見つかった時点で打ち切る。同じ
/// 順序で走ることで、原子操作も競合も無しに同じ親が決まる。候補数は統合後
/// なので多くて数十であり、最悪 O(N^2) でも問題にならない。
__global__ void parent_kernel(const DeviceCandidates grouped, std::int32_t* parent,
                              const std::int32_t* count) {
    const int inner = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const int total = *count;
    if (inner >= total) {
        return;
    }

    std::int32_t inner_x[kQuadCornerCount];
    std::int32_t inner_y[kQuadCornerCount];
    for (int corner = 0; corner < kQuadCornerCount; ++corner) {
        inner_x[corner] = grouped.corner_x_[(corner * grouped.capacity_) + inner];
        inner_y[corner] = grouped.corner_y_[(corner * grouped.capacity_) + inner];
    }

    // index が小さいほど周長が大きい。降順に見るので、最初に見つかるのは
    // 自分を囲むもののうち最も内側である。
    for (int outer = inner - 1; outer >= 0; --outer) {
        std::int32_t outer_x[kQuadCornerCount];
        std::int32_t outer_y[kQuadCornerCount];
        for (int corner = 0; corner < kQuadCornerCount; ++corner) {
            outer_x[corner] = grouped.corner_x_[(corner * grouped.capacity_) + outer];
            outer_y[corner] = grouped.corner_y_[(corner * grouped.capacity_) + outer];
        }
        bool inside = true;
        for (int corner = 0; corner < kQuadCornerCount && inside; ++corner) {
            inside = point_in_quad(outer_x, outer_y, inner_x[corner], inner_y[corner]);
        }
        if (inside) {
            parent[inner] = outer;
            return;
        }
    }
}

/// 親を辿って段数を伝える。
///
/// index の降順に 1 つずつ進める。OpenCV も同じ順序であり、順序を変えると
/// 3 段以上の入れ子で段数が変わる。
__global__ void depth_kernel(const std::int32_t* parent, std::int32_t* depth,
                             const std::int32_t* count) {
    if (threadIdx.x != 0U || blockIdx.x != 0U) {
        return;
    }
    const int total = *count;
    for (int index = total - 1; index >= 0; --index) {
        const std::int32_t up = parent[index];
        if (up < 0) {
            continue;
        }
        const std::int32_t candidate_depth = depth[index] + 1;
        if (candidate_depth > depth[up]) {
            depth[up] = candidate_depth;
        }
    }
}

/// 識別の打ち切りが起きる段数を求める。
///
/// OpenCV の while ループをそのまま辿る。到達数の数え方まで同じにする。
/// 祖先として数えた候補を自分の段でもう一度数える点も含める。
__global__ void suppress_kernel(const std::int32_t* parent, const std::int32_t* depth,
                                const std::int32_t* ids, std::int32_t* visited,
                                std::int32_t* stop_depth, std::int32_t* counter,
                                const std::int32_t* count) {
    const int total = *count;
    __shared__ int reached;
    __shared__ int level_size;
    if (threadIdx.x == 0U) {
        reached = 0;
        *stop_depth = 0;
    }
    __syncthreads();
    if (total <= 0) {
        if (threadIdx.x == 0U) {
            *counter = 0;
        }
        return;
    }

    // 段数は候補数を超えない。上限を候補数にして必ず有界にする。
    for (int level = 0; level < total; ++level) {
        __syncthreads();
        if (reached >= total) {
            break;
        }
        if (threadIdx.x == 0U) {
            level_size = 0;
            *stop_depth = level + 1;
        }
        __syncthreads();

        int local_size = 0;
        for (int v = static_cast<int>(threadIdx.x); v < total; v += static_cast<int>(blockDim.x)) {
            if (depth[v] != level) {
                continue;
            }
            ++local_size;
            if (ids[v] < 0) {
                continue;
            }
            // 識別できた候補の祖先へ印を付ける。既に印があっても辿り続ける。
            std::int32_t up = parent[v];
            while (up >= 0) {
                // 同じ祖先を 2 つの thread が同時に見ても 1 回だけ数える。
                if (atomicExch(&visited[up], 1) == 0) {
                    atomicAdd(&reached, 1);
                }
                up = parent[up];
            }
        }
        atomicAdd(&level_size, local_size);
        __syncthreads();
        if (threadIdx.x == 0U) {
            // 段の候補は識別の成否によらず数える。
            reached += level_size;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0U) {
        *counter = reached;
    }
}

}  // namespace

std::size_t candidate_tree_workspace_bytes(const DetectorConfig& config) {
    if (config.max_candidates_ <= 0) {
        return 0U;
    }
    const auto capacity = static_cast<std::size_t>(config.max_candidates_);
    const std::size_t parent = plane_bytes(capacity, sizeof(std::int32_t));
    const std::size_t depth = plane_bytes(capacity, sizeof(std::int32_t));
    const std::size_t visited = plane_bytes(capacity, sizeof(std::int32_t));
    const std::size_t scalars = plane_bytes(1U, sizeof(std::int32_t)) * 2U;
    if (parent == 0U || depth == 0U || visited == 0U || scalars == 0U) {
        return 0U;
    }
    return parent + depth + visited + scalars;
}

Status reserve_candidate_tree(const DetectorConfig& config, Workspace& workspace,
                              CandidateTreeBuffers* out) {
    if (out == nullptr) {
        return Status::kInvalidArgument;
    }
    if (candidate_tree_workspace_bytes(config) == 0U) {
        return Status::kInvalidConfig;
    }
    const auto capacity = static_cast<std::size_t>(config.max_candidates_);

    CandidateTreeBuffers buffers;
    buffers.capacity_ = config.max_candidates_;
    void* pointer = nullptr;

    Status status = workspace.allocate(capacity * sizeof(std::int32_t), kPlaneAlignment, &pointer);
    if (status != Status::kOk) {
        return status;
    }
    buffers.parent_ = static_cast<std::int32_t*>(pointer);

    status = workspace.allocate(capacity * sizeof(std::int32_t), kPlaneAlignment, &pointer);
    if (status != Status::kOk) {
        return status;
    }
    buffers.depth_ = static_cast<std::int32_t*>(pointer);

    status = workspace.allocate(capacity * sizeof(std::int32_t), kPlaneAlignment, &pointer);
    if (status != Status::kOk) {
        return status;
    }
    buffers.visited_ = static_cast<std::int32_t*>(pointer);

    status = workspace.allocate(sizeof(std::int32_t), kPlaneAlignment, &pointer);
    if (status != Status::kOk) {
        return status;
    }
    buffers.stop_depth_ = static_cast<std::int32_t*>(pointer);

    status = workspace.allocate(sizeof(std::int32_t), kPlaneAlignment, &pointer);
    if (status != Status::kOk) {
        return status;
    }
    buffers.counter_ = static_cast<std::int32_t*>(pointer);

    *out = buffers;
    return Status::kOk;
}

Status build_candidate_tree_async(const DeviceCandidates& grouped, CandidateTreeBuffers* tree,
                                  cudaStream_t stream) {
    if (tree == nullptr || tree->parent_ == nullptr || tree->depth_ == nullptr ||
        tree->visited_ == nullptr || grouped.corner_x_ == nullptr || grouped.corner_y_ == nullptr ||
        grouped.count_ == nullptr) {
        return Status::kInvalidArgument;
    }
    if (tree->capacity_ < grouped.capacity_) {
        return Status::kInvalidArgument;
    }

    const auto capacity = static_cast<unsigned int>(tree->capacity_);
    const unsigned int reset_blocks = (capacity + static_cast<unsigned int>(kResetThreads) - 1U) /
                                      static_cast<unsigned int>(kResetThreads);
    reset_tree_kernel<<<reset_blocks, static_cast<unsigned int>(kResetThreads), 0, stream>>>(
            tree->parent_, tree->depth_, tree->visited_, tree->capacity_);
    Status status = check_kernel_launch("candidate_tree.reset_tree_kernel", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }

    parent_kernel<<<reset_blocks, static_cast<unsigned int>(kResetThreads), 0, stream>>>(
            grouped, tree->parent_, grouped.count_);
    status = check_kernel_launch("candidate_tree.parent_kernel", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }

    depth_kernel<<<1U, 1U, 0, stream>>>(tree->parent_, tree->depth_, grouped.count_);
    return check_kernel_launch("candidate_tree.depth_kernel", -1, false, stream);
}

Status resolve_suppression_async(const DeviceCandidates& grouped, const MatchBuffers& matches,
                                 CandidateTreeBuffers* tree, cudaStream_t stream) {
    if (tree == nullptr || tree->parent_ == nullptr || tree->depth_ == nullptr ||
        tree->visited_ == nullptr || tree->stop_depth_ == nullptr || tree->counter_ == nullptr ||
        matches.ids_ == nullptr || grouped.count_ == nullptr) {
        return Status::kInvalidArgument;
    }
    if (tree->capacity_ < grouped.capacity_ || matches.capacity_ < grouped.capacity_) {
        return Status::kInvalidArgument;
    }

    suppress_kernel<<<1U, static_cast<unsigned int>(kSuppressThreads), 0, stream>>>(
            tree->parent_, tree->depth_, matches.ids_, tree->visited_, tree->stop_depth_,
            tree->counter_, grouped.count_);
    return check_kernel_launch("candidate_tree.suppress_kernel", -1, false, stream);
}

}  // namespace aruco3cuda::detail
