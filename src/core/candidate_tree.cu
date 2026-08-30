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
/// The suppression walk runs in a single block. Each level needs a block-wide barrier.
constexpr int kSuppressThreads = 256;

/// Returns the byte count of one plane. Returns 0 on overflow.
std::size_t plane_bytes(std::size_t count, std::size_t element_bytes) {
    if (count != 0U && element_bytes > std::numeric_limits<std::size_t>::max() / count) {
        return 0U;
    }
    return align_up(count * element_bytes, kPlaneAlignment);
}

/// Decides whether a point lies inside a quadrilateral or on its boundary.
///
/// Matches calling OpenCV's pointPolygonTest with measureDist = false. The walk
/// visits the edges (c3 to c0), (c0 to c1), (c1 to c2), (c2 to c3) in that
/// order. Cross products are computed in 64 bit; with integer coordinates the
/// signs match exactly those OpenCV computes in double precision. Convexity is
/// not assumed: a quadrilateral built by extreme point search can be concave,
/// so deciding from matching signs alone would change the result.
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
            // An edge the scanline does not cross. It still counts as inside
            // when the point lies on a vertex or on a horizontal edge.
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
            // On the edge. The boundary counts as inside.
            return true;
        }
        if (vy < v0y) {
            distance = -distance;
        }
        counter += (distance > 0) ? 1 : 0;
    }
    return (counter % 2) != 0;
}

/// Initializes the buffers. The candidate count lives only on the device, so
/// the whole capacity is filled.
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

/// Determines the parent of each candidate. One thread handles one candidate.
///
/// OpenCV scans j downward from i-1 and stops at the first hit. Running in the
/// same order settles on the same parent with no atomics and no contention.
/// After merging there are at most a few dozen candidates, so O(N^2) in the
/// worst case is not a problem.
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

    // A smaller index means a larger perimeter. Scanning downward, the first
    // hit is the innermost of the candidates enclosing this one.
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

/// Propagates the level up through the parents.
///
/// Advances one candidate at a time in descending index order. OpenCV uses the
/// same order; changing it changes the levels for nesting three or more deep.
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

/// Determines the level at which identification is cut off.
///
/// Follows OpenCV's while loop directly, down to how the reached count is
/// tallied. That includes counting a candidate marked as an ancestor again when
/// its own level comes up.
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

    // The level never exceeds the candidate count. Bounding the loop by that count keeps it finite.
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
            // Mark the ancestors of an identified candidate. Keep walking
            // even when a mark is already there.
            std::int32_t up = parent[v];
            while (up >= 0) {
                // Count an ancestor once even when two threads reach it at the same time.
                if (atomicExch(&visited[up], 1) == 0) {
                    atomicAdd(&reached, 1);
                }
                up = parent[up];
            }
        }
        atomicAdd(&level_size, local_size);
        __syncthreads();
        if (threadIdx.x == 0U) {
            // Candidates at this level are counted whether or not identification succeeded.
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
