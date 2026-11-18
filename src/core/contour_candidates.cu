// SPDX-License-Identifier: Apache-2.0
#include "contour_candidates.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <limits>

#include "contour_approx.cuh"
#include "cuda_check.hpp"

// Keep the exact serial helper available for quality and timing ablations.
#ifndef ARUCO3CUDA_EXPERIMENT_COOPERATIVE_RDP
#define ARUCO3CUDA_EXPERIMENT_COOPERATIVE_RDP 1
#endif

#ifndef ARUCO3CUDA_EXPERIMENT_PARALLEL_CONTOUR_COUNT
#define ARUCO3CUDA_EXPERIMENT_PARALLEL_CONTOUR_COUNT 1
#endif

#ifndef ARUCO3CUDA_EXPERIMENT_COOPERATIVE_MATERIALIZE
#define ARUCO3CUDA_EXPERIMENT_COOPERATIVE_MATERIALIZE 1
#endif

#ifndef ARUCO3CUDA_EXPERIMENT_COMPACT_MATERIALIZE_EDGES
#define ARUCO3CUDA_EXPERIMENT_COMPACT_MATERIALIZE_EDGES 1
#endif

namespace aruco3cuda::detail {
namespace {
constexpr int kThreads = 256;
constexpr std::size_t kAlignment = 256U;

__device__ bool foreground(const ImagePlaneU8& binary, int x, int y) {
    return x >= 0 && y >= 0 && x < binary.width_px_ && y < binary.height_px_ &&
           binary.data_[static_cast<std::size_t>(y) * binary.pitch_bytes_ + x] != 0U;
}

/// A directed crack edge keeps foreground on its right: N/E/S/W edges travel
/// east/south/west/north. Out-of-image pixels are isolated zero background.
__device__ bool boundary_edge(const ImagePlaneU8& binary, int x, int y, int direction) {
    constexpr int kDx[4] = {0, 1, 0, -1};
    constexpr int kDy[4] = {-1, 0, 1, 0};
    return foreground(binary, x, y) && !foreground(binary, x + kDx[direction], y + kDy[direction]);
}

/// At a diagonal contact choose the left continuation. This joins 8-connected
/// foreground and keeps 4-connected background holes separate. Each active edge
/// has exactly one successor and predecessor; no visited pixels are shared.
__device__ int edge_successor(const ImagePlaneU8& binary, int edge) {
    const int direction = edge & 3;
    const int pixel = edge / 4;
    const int x = pixel % binary.width_px_;
    const int y = pixel / binary.width_px_;
    const int vertex_x = x + (direction < 2 ? 1 : 0);
    const int vertex_y = y + (direction > 0 && direction < 3 ? 1 : 0);
    constexpr int kTurns[4] = {3, 0, 1, 2};
    for (int turn = 0; turn < 4; ++turn) {
        const int next_direction = (direction + kTurns[turn]) & 3;
        const int next_x = vertex_x - (next_direction == 1 || next_direction == 2 ? 1 : 0);
        const int next_y = vertex_y - (next_direction == 2 || next_direction == 3 ? 1 : 0);
        if (boundary_edge(binary, next_x, next_y, next_direction)) {
            return ((next_y * binary.width_px_ + next_x) * 4) + next_direction;
        }
    }
    return -1;
}

__global__ void reset_invocation_kernel(ContourCandidateBuffers buffers,
                                        DeviceCandidates candidates, bool append) {
    if (!append) {
        *buffers.failure_ = 0;
    }
    *buffers.base_ = append ? *candidates.count_ : 0;
    *buffers.point_total_ = 0;
    *buffers.jump_count_ = 0;
}

/// Each thread owns its parent/length/offset slot. The predecessor destination
/// is unique by the boundary permutation. Inactive predecessor slots are never
/// read; active slots are replaced on every frame before any tracing occurs.
__global__ void initialize_edges_kernel(ImagePlaneU8 binary, ContourCandidateBuffers buffers) {
    const int edge = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (edge >= buffers.edge_capacity_) {
        return;
    }
    const int pixel = edge / 4;
    const bool active =
            boundary_edge(binary, pixel % binary.width_px_, pixel / binary.width_px_, edge & 3);
    buffers.parents_[edge] = active ? edge : -1;
    if (active) {
        const int next = edge_successor(binary, edge);
        if (next < 0 || next >= buffers.edge_capacity_) {
            atomicOr(buffers.failure_, kContourInvalidCycle);
        } else {
            buffers.previous_[next] = edge;
        }
    }
}

__device__ int edge_root(std::int32_t* parents, int edge) {
    int parent = atomicAdd(&parents[edge], 0);
    while (parent != edge) {
        edge = parent;
        parent = atomicAdd(&parents[edge], 0);
    }
    return edge;
}

/// Atomic root values decrease monotonically; concurrent reads are atomic too.
/// Linking a boundary edge only to
/// its predecessor merges a closed cycle, never unrelated boundaries at a shared
/// pixel or vertex.
__global__ void unite_edges_kernel(ContourCandidateBuffers buffers) {
    const int edge = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (edge >= buffers.edge_capacity_ || atomicAdd(&buffers.parents_[edge], 0) < 0 ||
        *buffers.failure_ != 0) {
        return;
    }
    int a = edge;
    int b = buffers.previous_[edge];
    for (;;) {
        a = edge_root(buffers.parents_, a);
        b = edge_root(buffers.parents_, b);
        if (a == b) {
            return;
        }
        const int high = max(a, b);
        const int low = min(a, b);
        const int previous = atomicMin(&buffers.parents_[high], low);
        if (previous == high) {
            return;
        }
        a = previous;
        b = low;
    }
}

/// Each warp counts roots with a ballot; eight shared counts give a stable
/// block-local rank without any global atomic allocation or ordering ambiguity.
__device__ int local_root_rank(bool is_root, int* warp_counts, int* total) {
    const unsigned int lane = threadIdx.x & 31U;
    const unsigned int warp = threadIdx.x / 32U;
    const unsigned int mask = __ballot_sync(0xffffffffU, is_root);
    if (lane == 0U) {
        warp_counts[warp] = __popc(mask);
    }
    __syncthreads();
    int prefix = 0;
    int sum = 0;
    for (unsigned int index = 0; index < blockDim.x / 32U; ++index) {
        if (index < warp) {
            prefix += warp_counts[index];
        }
        sum += warp_counts[index];
    }
    *total = sum;
    return prefix + __popc(mask & ((1U << lane) - 1U));
}

__global__ void mark_roots_kernel(ContourCandidateBuffers buffers) {
    __shared__ int warp_counts[kThreads / 32];
    const int edge = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const bool is_root = edge < buffers.edge_capacity_ && buffers.parents_[edge] == edge;
    int total = 0;
    static_cast<void>(local_root_rank(is_root, warp_counts, &total));
    if (threadIdx.x == 0U) {
        buffers.scan_.block_offsets_[blockIdx.x] = total;
    }
}

__global__ void scatter_roots_kernel(ContourCandidateBuffers buffers) {
    __shared__ int warp_counts[kThreads / 32];
    const int edge = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const bool is_root = edge < buffers.edge_capacity_ && buffers.parents_[edge] == edge;
    int unused = 0;
    const int local = local_root_rank(is_root, warp_counts, &unused);
    if (is_root) {
        const int destination = buffers.scan_.block_offsets_[blockIdx.x] + local;
        if (destination >= buffers.root_capacity_) {
            atomicOr(buffers.failure_, kContourInvalidCycle);
        } else {
            buffers.roots_[destination] = edge;
        }
    }
}

__global__ void finish_roots_kernel(ContourCandidateBuffers buffers) {
    if (*buffers.failure_ != 0) {
        *buffers.root_count_ = 0;
    }
}

/// Compact scans read the device root count. Persistent blocks visit only its
/// occupied tiles; no frame-dependent count is retrieved by the host.
__global__ void compact_scan_tiles_kernel(std::int32_t* values, ContourCandidateBuffers buffers) {
    __shared__ int shared[kThreads];
    const int count = *buffers.root_count_;
    const int tiles = (count + kThreads - 1) / kThreads;
    for (int tile = static_cast<int>(blockIdx.x); tile < tiles;
         tile += static_cast<int>(gridDim.x)) {
        const int index = tile * kThreads + static_cast<int>(threadIdx.x);
        const int value = index < count ? values[index] : 0;
        shared[threadIdx.x] = value;
        __syncthreads();
        for (int offset = 1; offset < kThreads; offset <<= 1) {
            const int addend =
                    static_cast<int>(threadIdx.x) >= offset ? shared[threadIdx.x - offset] : 0;
            __syncthreads();
            shared[threadIdx.x] += addend;
            __syncthreads();
        }
        if (index < count) {
            values[index] = shared[threadIdx.x] - value;
        }
        if (threadIdx.x == kThreads - 1U) {
            buffers.scan_.block_offsets_[tile] = shared[threadIdx.x];
        }
        __syncthreads();
    }
}

__global__ void compact_scan_sums_kernel(ContourCandidateBuffers buffers) {
    __shared__ int shared[kThreads];
    const int tiles = (*buffers.root_count_ + kThreads - 1) / kThreads;
    int running = 0;
    for (int base = 0; base < tiles; base += kThreads) {
        const int index = base + static_cast<int>(threadIdx.x);
        const int value = index < tiles ? buffers.scan_.block_offsets_[index] : 0;
        shared[threadIdx.x] = value;
        __syncthreads();
        for (int offset = 1; offset < kThreads; offset <<= 1) {
            const int addend =
                    static_cast<int>(threadIdx.x) >= offset ? shared[threadIdx.x - offset] : 0;
            __syncthreads();
            shared[threadIdx.x] += addend;
            __syncthreads();
        }
        if (index < tiles) {
            buffers.scan_.block_offsets_[index] = running + shared[threadIdx.x] - value;
        }
        const int sum = shared[kThreads - 1];
        __syncthreads();
        running += sum;
    }
    if (threadIdx.x == 0U) {
        *buffers.scan_.total_ = running;
    }
}

__global__ void compact_scan_add_kernel(std::int32_t* values, ContourCandidateBuffers buffers) {
    for (int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
         index < *buffers.root_count_; index += static_cast<int>(gridDim.x * blockDim.x)) {
        values[index] += buffers.scan_.block_offsets_[index / kThreads];
    }
}

Status scan_compact_async(std::int32_t* values, ContourCandidateBuffers buffers,
                          cudaStream_t stream) {
    compact_scan_tiles_kernel<<<64U, kThreads, 0, stream>>>(values, buffers);
    Status status = check_kernel_launch("contour.scan_tiles", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    compact_scan_sums_kernel<<<1U, kThreads, 0, stream>>>(buffers);
    status = check_kernel_launch("contour.scan_sums", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    compact_scan_add_kernel<<<64U, kThreads, 0, stream>>>(values, buffers);
    return check_kernel_launch("contour.scan_add", -1, false, stream);
}

#if ARUCO3CUDA_EXPERIMENT_PARALLEL_CONTOUR_COUNT
/// Point and approximation pools are unused until counting finishes. Reservation
/// aliases them only when their configured capacity covers this exact scratch;
/// small point budgets have separately allocated fallback regions.
__global__ void initialize_count_kernel(ContourCandidateBuffers buffers) {
    for (int path = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
         path < *buffers.root_count_; path += static_cast<int>(gridDim.x * blockDim.x)) {
        buffers.lengths_[path] = 0;
        buffers.offsets_[path] = 0;
        buffers.count_edges_[path] = 0;
        buffers.count_area_[path] = 0;
        buffers.count_seed_[path] = 2147483647;
        buffers.count_map_[buffers.roots_[path]] = path;
    }
}

/// Union has completed and this phase never modifies parents. Strictly decreasing
/// links prove termination; validate each link before following it.
__device__ int immutable_root(ContourCandidateBuffers buffers, int edge) {
    for (;;) {
        const int parent = buffers.parents_[edge];
        if (parent == edge) {
            return edge;
        }
        if (parent < 0 || parent >= edge) {
            return -1;
        }
        edge = parent;
    }
}

/// A cycle's pixel transitions equal its collapsed unit-path length; a cycle
/// with no transition is one isolated pixel. Directed edge areas add exactly.
/// Warp groups aggregate equal-root contributions before the integer atomics.
__global__ void aggregate_count_kernel(ContourCandidateBuffers buffers) {
    if (*buffers.root_count_ == 0) {
        return;
    }
    const unsigned int lane = threadIdx.x & 31U;
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long base = static_cast<long long>(blockIdx.x) * blockDim.x;
         base < buffers.edge_capacity_; base += stride) {
        const int edge = static_cast<int>(base + threadIdx.x);
        bool valid = edge < buffers.edge_capacity_ && buffers.parents_[edge] >= 0;
        int path = -1;
        int previous = -1;
        if (valid) {
            const int root = immutable_root(buffers, edge);
            previous = buffers.previous_[edge];
            if (root < 0 || previous < 0 || previous >= buffers.edge_capacity_ ||
                buffers.parents_[previous] < 0) {
                valid = false;
            } else {
                path = buffers.count_map_[root];
                valid = path >= 0 && path < *buffers.root_count_;
#if ARUCO3CUDA_EXPERIMENT_COMPACT_MATERIALIZE_EDGES
                // Canonical root entries remain read-only throughout aggregation.
                // Other entries have one writer and are consumed by a later kernel.
                if (valid && edge != root) {
                    buffers.count_map_[edge] = path;
                }
#endif
            }
            if (!valid) {
                atomicOr(buffers.failure_, kContourInvalidCycle);
            }
        }
        const unsigned int active = __ballot_sync(0xffffffffU, valid);
        if (valid) {
            const unsigned int group = __match_any_sync(active, path);
            const int pixel = edge / 4;
            const int x = pixel % buffers.width_px_;
            const int y = pixel / buffers.width_px_;
            const int direction = edge & 3;
            const int area = direction == 0   ? -y
                             : direction == 1 ? x + 1
                             : direction == 2 ? y + 1
                                              : -x;
            const int seed = direction == 1 && x + 1 < buffers.width_px_ ? edge : 2147483647;
            const int transitions = __reduce_add_sync(group, pixel != previous / 4 ? 1 : 0);
            // Public dimensions are <=65536: a 32-lane area's sum fits int.
            const int area_sum = __reduce_add_sync(group, area);
            const int first_seed = __reduce_min_sync(group, seed);
            if (lane == static_cast<unsigned int>(__ffs(group) - 1)) {
                atomicAdd(&buffers.lengths_[path], transitions);
                atomicAdd(&buffers.count_edges_[path], __popc(group));
                atomicAdd(reinterpret_cast<unsigned long long*>(&buffers.count_area_[path]),
                          static_cast<unsigned long long>(static_cast<long long>(area_sum)));
                if (first_seed != 2147483647) {
                    atomicMin(&buffers.count_seed_[path], first_seed);
                }
            }
        }
    }
}

__global__ void finish_count_kernel(ContourCandidateBuffers buffers, int minimum, int maximum) {
    for (int path = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
         path < *buffers.root_count_; path += static_cast<int>(gridDim.x * blockDim.x)) {
        const int length = max(buffers.lengths_[path], 1);
        buffers.lengths_[path] = 0;
        if (atomicAdd(buffers.failure_, 0) != 0 || length < minimum || length > maximum) {
            continue;
        }
        if (buffers.count_area_[path] < 0) {
            const int seed = buffers.count_seed_[path];
            if (seed == 2147483647) {
                atomicOr(buffers.failure_, kContourInvalidCycle);
                continue;
            }
            buffers.parents_[buffers.roots_[path]] = seed;
        }
        buffers.lengths_[path] = length;
        buffers.offsets_[path] = length;
    }
}
#endif

#if !ARUCO3CUDA_EXPERIMENT_PARALLEL_CONTOUR_COUNT
/// One thread traces each canonical root. Only a perimeter-eligible contour
/// gets point-pool space. A contour longer than max+1 cannot become eligible by
/// removing the repeated closing point, so its counting pass may stop early.
__device__ void count_path(ImagePlaneU8 binary, ContourCandidateBuffers buffers, int minimum,
                           int maximum, int path) {
    buffers.lengths_[path] = 0;
    buffers.offsets_[path] = 0;
    if (atomicAdd(buffers.failure_, 0) != 0) {
        return;
    }
    const int root = buffers.roots_[path];
    int edge = root;
    int previous_pixel = -1;
    const int first_pixel = root / 4;
    int count = 0;
    int edge_steps = 0;
    long long twice_area = 0;
    int hole_key = 2147483647;
    int hole_seed = -1;
    bool closed = false;
    for (int step = 0; step < buffers.edge_capacity_; ++step) {
        if (edge < 0 || edge >= buffers.edge_capacity_) {
            atomicOr(buffers.failure_, kContourInvalidCycle);
            return;
        }
        ++edge_steps;
        const int pixel = edge / 4;
        const int x = pixel % binary.width_px_;
        const int y = pixel / binary.width_px_;
        const int direction = edge & 3;
        if (pixel != previous_pixel) {
            ++count;
            previous_pixel = pixel;
            if (static_cast<long long>(count) > static_cast<long long>(maximum) + 1LL) {
                return;
            }
        }
        const int sx = x + (direction == 1 || direction == 2 ? 1 : 0);
        const int sy = y + (direction == 2 || direction == 3 ? 1 : 0);
        const int ex = x + (direction < 2 ? 1 : 0);
        const int ey = y + (direction > 0 && direction < 3 ? 1 : 0);
        twice_area += static_cast<long long>(sx) * ey - static_cast<long long>(sy) * ex;
        const int background_pixel = y * binary.width_px_ + x + 1;
        if (direction == 1 && x + 1 < binary.width_px_ && background_pixel < hole_key) {
            hole_key = background_pixel;
            hole_seed = edge;
        }
        edge = buffers.previous_[edge];
        if (edge == root) {
            closed = true;
            break;
        }
    }
    if (!closed) {
        atomicOr(buffers.failure_, kContourInvalidCycle);
        return;
    }
    if (count > 1 && previous_pixel == first_pixel) {
        --count;
    }
    if (count < minimum || count > maximum) {
        return;
    }
    if (twice_area < 0) {
        if (hole_seed < 0) {
            atomicOr(buffers.failure_, kContourInvalidCycle);
            return;
        }
        buffers.parents_[root] = hole_seed;
    }
    buffers.count_edges_[path] = edge_steps;
    buffers.lengths_[path] = count;
    buffers.offsets_[path] = count;
}

__global__ void count_paths_kernel(ImagePlaneU8 binary, ContourCandidateBuffers buffers,
                                   int minimum, int maximum) {
    for (int path = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
         path < *buffers.root_count_; path += static_cast<int>(gridDim.x * blockDim.x)) {
        count_path(binary, buffers, minimum, maximum, path);
    }
}

#endif

__global__ void retain_point_total_kernel(ContourCandidateBuffers buffers) {
    *buffers.point_total_ = *buffers.scan_.total_;
    if (*buffers.point_total_ > buffers.point_capacity_) {
        atomicOr(buffers.failure_, kContourPointOverflow);
    }
}

/// Only root threads write disjoint point slices assigned by the scan. Reverse
/// crack-edge traversal collapses adjacent edges belonging to the same foreground
/// pixel and reproduces the oracle's unit path orientation and starting point.
__device__ void materialize_path(ContourCandidateBuffers buffers, int root) {
    if (buffers.lengths_[root] == 0 || atomicAdd(buffers.failure_, 0) != 0) {
        return;
    }
    const int length = buffers.lengths_[root];
    const int start = buffers.parents_[buffers.roots_[root]];
    int edge = start;
    int previous_pixel = -1;
    int written = 0;
    const int offset = buffers.offsets_[root];
    for (int step = 0; step < buffers.edge_capacity_; ++step) {
        const int pixel = edge / 4;
        if (pixel != previous_pixel) {
            if (written < length) {
                buffers.points_[offset + written] = {pixel % buffers.width_px_,
                                                     pixel / buffers.width_px_};
            }
            ++written;
            previous_pixel = pixel;
        }
        edge = buffers.previous_[edge];
        if (edge == start) {
            const bool repeated_close = written == length + 1 && previous_pixel == start / 4;
            if (written != length && !repeated_close) {
                atomicOr(buffers.failure_, kContourInvalidCycle);
            }
            return;
        }
    }
    atomicOr(buffers.failure_, kContourInvalidCycle);
}

#if ARUCO3CUDA_EXPERIMENT_COOPERATIVE_MATERIALIZE
#if ARUCO3CUDA_EXPERIMENT_COMPACT_MATERIALIZE_EDGES && ARUCO3CUDA_EXPERIMENT_PARALLEL_CONTOUR_COUNT
/// Only eligible cycles longer than one warp need jump32. Original-ID order is
/// irrelevant here: each jump destination still has exactly one writing edge.
__global__ void compact_jump_edges_kernel(ContourCandidateBuffers buffers) {
    if (*buffers.point_total_ == 0) {
        return;
    }
    __shared__ int initial_failure;
    if (threadIdx.x == 0U) {
        initial_failure = atomicAdd(buffers.failure_, 0);
    }
    __syncthreads();
    if (initial_failure != 0) {
        return;
    }
    const unsigned int lane = threadIdx.x & 31U;
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (long long base = static_cast<long long>(blockIdx.x) * blockDim.x;
         base < buffers.edge_capacity_; base += stride) {
        const int edge = static_cast<int>(base + threadIdx.x);
        bool selected = false;
        if (edge < buffers.edge_capacity_ && buffers.parents_[edge] >= 0) {
            const int path = buffers.count_map_[edge];
            if (path < 0 || path >= *buffers.root_count_) {
                atomicOr(buffers.failure_, kContourInvalidCycle);
            } else {
                selected = buffers.lengths_[path] > 0 && buffers.count_edges_[path] > 32;
            }
        }
        const unsigned int mask = __ballot_sync(0xffffffffU, selected);
        int destination = 0;
        if (lane == 0U && mask != 0U) {
            destination = atomicAdd(buffers.jump_count_, __popc(mask));
        }
        destination = __shfl_sync(0xffffffffU, destination, 0);
        if (selected) {
            destination += __popc(mask & ((1U << lane) - 1U));
            if (destination >= buffers.jump_capacity_) {
                atomicOr(buffers.failure_, kContourInvalidCycle);
            } else {
                buffers.jump_edges_[destination] = edge;
            }
        }
    }
}
#endif

/// The two maps use expired counting scratch. The fifth pass leaves jump32 in
/// approximation scratch, which stays read-only while points are materialized.
__global__ void jump_predecessors_kernel(ContourCandidateBuffers buffers,
                                         const std::int32_t* source, std::int32_t* destination) {
    if (*buffers.point_total_ == 0) {
        return;
    }
    __shared__ int initial_failure;
    if (threadIdx.x == 0U) {
        initial_failure = atomicAdd(buffers.failure_, 0);
    }
    __syncthreads();
    if (initial_failure != 0) {
        return;
    }
    const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
#if ARUCO3CUDA_EXPERIMENT_COMPACT_MATERIALIZE_EDGES && ARUCO3CUDA_EXPERIMENT_PARALLEL_CONTOUR_COUNT
    const int count = *buffers.jump_count_;
#else
    const int count = buffers.edge_capacity_;
#endif
    for (long long position = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         position < count; position += stride) {
#if ARUCO3CUDA_EXPERIMENT_COMPACT_MATERIALIZE_EDGES && ARUCO3CUDA_EXPERIMENT_PARALLEL_CONTOUR_COUNT
        const int edge = buffers.jump_edges_[position];
#else
        const int edge = static_cast<int>(position);
        if (buffers.parents_[edge] < 0) {
            continue;
        }
#endif
        const int middle = source[edge];
        if (middle < 0 || middle >= buffers.edge_capacity_ || buffers.parents_[middle] < 0) {
            atomicOr(buffers.failure_, kContourInvalidCycle);
            continue;
        }
        const int next = source[middle];
        if (next < 0 || next >= buffers.edge_capacity_ || buffers.parents_[next] < 0) {
            atomicOr(buffers.failure_, kContourInvalidCycle);
            continue;
        }
        destination[edge] = next;
    }
}

/// A warp owns one path. Lane order follows consecutive predecessor edges;
/// ballots retain only foreground-pixel transitions in exactly that order.
__global__ void materialize_paths_kernel(ContourCandidateBuffers buffers) {
    const unsigned int lane = threadIdx.x & 31U;
    const int first = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) / 32U);
    const int stride = static_cast<int>(gridDim.x * blockDim.x / 32U);
    for (int root = first; root < *buffers.root_count_; root += stride) {
        int failed = 0;
        if (lane == 0U) {
            failed = atomicAdd(buffers.failure_, 0);
        }
        failed = __shfl_sync(0xffffffffU, failed, 0);
        const int length = buffers.lengths_[root];
        if (failed != 0 || length == 0) {
            continue;
        }
        const int edges = buffers.count_edges_[root];
        if (edges <= 0 || edges > buffers.edge_capacity_) {
            if (lane == 0U) {
                atomicOr(buffers.failure_, kContourInvalidCycle);
            }
            continue;
        }
        if (edges <= 32) {
            if (lane == 0U) {
                materialize_path(buffers, root);
            }
            __syncwarp();
            continue;
        }
        const int start = buffers.parents_[buffers.roots_[root]];
        int edge = start;
        for (unsigned int step = 0U; step < lane; ++step) {
            edge = buffers.previous_[edge];
        }
        const int offset = buffers.offsets_[root];
        int written = 0;
        int last_pixel = -1;
        for (int base = 0; base < edges; base += 32) {
            const int pixel = edge / 4;
            int preceding = __shfl_up_sync(0xffffffffU, pixel, 1);
            if (lane == 0U) {
                preceding = last_pixel;
            }
            const bool emit = base + static_cast<int>(lane) < edges && pixel != preceding;
            const unsigned int mask = __ballot_sync(0xffffffffU, emit);
            const int local = __popc(mask & ((1U << lane) - 1U));
            if (emit && written + local < length) {
                buffers.points_[offset + written + local] = {pixel % buffers.width_px_,
                                                             pixel / buffers.width_px_};
            }
            written += __popc(mask);
            const int last_lane = min(31, edges - base - 1);
            last_pixel = __shfl_sync(0xffffffffU, pixel, last_lane);
            if (base + 32 < edges) {
                edge = buffers.jump_target_[edge];
            }
        }
        if (lane == 0U) {
            const bool repeated_close = written == length + 1 && last_pixel == start / 4;
            if (written != length && !repeated_close) {
                atomicOr(buffers.failure_, kContourInvalidCycle);
            }
        }
        __syncwarp();
    }
}
#else
__global__ void materialize_paths_kernel(ContourCandidateBuffers buffers) {
    for (int root = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
         root < *buffers.root_count_; root += static_cast<int>(gridDim.x * blockDim.x)) {
        materialize_path(buffers, root);
    }
}

#endif

__device__ bool valid_quad(const ContourPoint* points, int perimeter,
                           double minimum_corner_distance_rate) {
    int signs = 0;
    double smallest = 1.7976931348623157e+308;
    for (int corner = 0; corner < 4; ++corner) {
        const ContourPoint a = points[corner];
        const ContourPoint b = points[(corner + 1) & 3];
        const ContourPoint c = points[(corner + 2) & 3];
        const long long cross = static_cast<long long>(b.x_ - a.x_) * (c.y_ - b.y_) -
                                static_cast<long long>(b.y_ - a.y_) * (c.x_ - b.x_);
        if (cross == 0) {
            return false;
        }
        signs |= cross > 0 ? 1 : 2;
        const double dx = b.x_ - a.x_;
        const double dy = b.y_ - a.y_;
        smallest = fmin(smallest, dx * dx + dy * dy);
    }
    const double minimum = static_cast<double>(perimeter) * minimum_corner_distance_rate;
    return signs != 3 && smallest >= minimum * minimum;
}

/// Tracing is finished, so the predecessor array is reused as acceptance/scan
/// storage. One thread owns each path's points, stack and approximation slice.
#if !ARUCO3CUDA_EXPERIMENT_COOPERATIVE_RDP
__device__ void approximate_path(ContourCandidateBuffers buffers, double epsilon_rate,
                                 double corner_distance_rate, int root) {
    buffers.previous_[root] = 0;
    const int length = buffers.lengths_[root];
    if (length == 0 || atomicAdd(buffers.failure_, 0) != 0) {
        return;
    }
    const int offset = buffers.offsets_[root];
    ContourPoint* approximation = buffers.approximation_ + offset;
    const int count = contour_approximate_closed(buffers.points_ + offset, length, approximation,
                                                 buffers.stack_ + offset,
                                                 static_cast<double>(length) * epsilon_rate);
    if (count < 0) {
        atomicOr(buffers.failure_, kContourApproximationOverflow);
        return;
    }
    if (count == 4 && valid_quad(approximation, length, corner_distance_rate)) {
        buffers.lengths_[root] = -length;
        buffers.previous_[root] = 1;
    }
}

#endif

#if ARUCO3CUDA_EXPERIMENT_COOPERATIVE_RDP
__global__ void approximate_paths_kernel(ContourCandidateBuffers buffers, double epsilon_rate,
                                         double corner_distance_rate) {
    const unsigned int lane = threadIdx.x & 31U;
    const int first = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) / 32U);
    const int stride = static_cast<int>(gridDim.x * blockDim.x / 32U);
    for (int root = first; root < *buffers.root_count_; root += stride) {
        int failed = 0;
        if (lane == 0U) {
            buffers.previous_[root] = 0;
            failed = atomicAdd(buffers.failure_, 0);
        }
        failed = __shfl_sync(0xffffffffU, failed, 0);
        const int length = buffers.lengths_[root];
        if (length == 0 || failed != 0) {
            continue;
        }
        const int offset = buffers.offsets_[root];
        ContourPoint* approximation = buffers.approximation_ + offset;
        const int count = contour_approximate_closed_warp(
                buffers.points_ + offset, length, approximation, buffers.stack_ + offset,
                static_cast<double>(length) * epsilon_rate);
        if (lane == 0U) {
            if (count < 0) {
                atomicOr(buffers.failure_, kContourApproximationOverflow);
            } else if (count == 4 && valid_quad(approximation, length, corner_distance_rate)) {
                buffers.lengths_[root] = -length;
                buffers.previous_[root] = 1;
            }
        }
        __syncwarp();
    }
}

#else
__global__ void approximate_paths_kernel(ContourCandidateBuffers buffers, double epsilon_rate,
                                         double corner_distance_rate) {
    for (int root = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
         root < *buffers.root_count_; root += static_cast<int>(gridDim.x * blockDim.x)) {
        approximate_path(buffers, epsilon_rate, corner_distance_rate, root);
    }
}

#endif

__device__ void compact_contour(ContourCandidateBuffers buffers, DeviceCandidates candidates,
                                int root) {
    if (buffers.lengths_[root] >= 0 || atomicAdd(buffers.failure_, 0) != 0) {
        return;
    }
    const int destination = *buffers.base_ + *buffers.scan_.total_ - 1 - buffers.previous_[root];
    if (destination >= candidates.capacity_) {
        atomicOr(buffers.failure_, kContourCandidateOverflow);
        return;
    }
    const ContourPoint* quad = buffers.approximation_ + buffers.offsets_[root];
    const long long cross =
            static_cast<long long>(quad[1].x_ - quad[0].x_) * (quad[2].y_ - quad[0].y_) -
            static_cast<long long>(quad[1].y_ - quad[0].y_) * (quad[2].x_ - quad[0].x_);
    for (int corner = 0; corner < 4; ++corner) {
        const int source = cross < 0 && (corner == 1 || corner == 3) ? 4 - corner : corner;
        candidates.corner_x_[corner * candidates.capacity_ + destination] = quad[source].x_;
        candidates.corner_y_[corner * candidates.capacity_ + destination] = quad[source].y_;
    }
    candidates.perimeter_[destination] = -buffers.lengths_[root];
    candidates.label_[destination] = buffers.roots_[root];
}

__global__ void compact_contours_kernel(ContourCandidateBuffers buffers,
                                        DeviceCandidates candidates) {
    for (int root = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
         root < *buffers.root_count_; root += static_cast<int>(gridDim.x * blockDim.x)) {
        compact_contour(buffers, candidates, root);
    }
}

__global__ void store_contour_count_kernel(ContourCandidateBuffers buffers,
                                           DeviceCandidates candidates, bool append) {
    const int previous_total = append ? *candidates.accepted_total_ : 0;
    const int accepted = *buffers.scan_.total_;
    *candidates.accepted_total_ = previous_total + accepted;
    const int written = *buffers.base_ + accepted;
    if (written > candidates.capacity_) {
        atomicOr(buffers.failure_, kContourCandidateOverflow);
    }
    // Compaction may stop after another thread reports failure. Never expose
    // partial or stale slots to later grouping, sampling or refinement kernels.
    *candidates.count_ = *buffers.failure_ == 0 ? written : 0;
}

int edge_capacity(int width, int height) {
    if (width <= 0 || height <= 0) {
        return 0;
    }
    const std::uint64_t pixels =
            static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height);
    // The shared scan rounds a signed count up by 255 before division.
    return pixels <= static_cast<std::uint64_t>((std::numeric_limits<int>::max() - 255) / 4)
                   ? static_cast<int>(pixels * 4U)
                   : 0;
}
int bounded_point_capacity(int width, int height, int requested) {
    // Every internal undirected adjacency contributes at most one crack edge;
    // every exterior side contributes at most one. Collapsing cannot add points.
    const std::uint64_t edge_bound =
            2U * static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height) +
            static_cast<std::uint64_t>(width) + static_cast<std::uint64_t>(height);
    return static_cast<int>(std::min(static_cast<std::uint64_t>(requested), edge_bound));
}

std::size_t aligned_bytes(std::size_t elements, std::size_t size) {
    if (elements > std::numeric_limits<std::size_t>::max() / size) {
        return 0U;
    }
    return align_up(elements * size, kAlignment);
}

}  // namespace

std::size_t contour_candidate_workspace_bytes(int width, int height, int max_points) {
    const int slots = edge_capacity(width, height);
    if (slots == 0 || max_points < 1) {
        return 0U;
    }
    const std::size_t edge_bytes =
            aligned_bytes(static_cast<std::size_t>(slots), sizeof(std::int32_t));
    const int points = bounded_point_capacity(width, height, max_points);
    const std::size_t root_bytes =
            aligned_bytes(static_cast<std::size_t>(width) * height, sizeof(std::int32_t));
    const std::size_t root_scan_bytes = scan_workspace_bytes(scan_block_count(slots));
    const std::size_t point_bytes =
            aligned_bytes(static_cast<std::size_t>(points), sizeof(ContourPoint));
    const std::size_t stack_bytes =
            aligned_bytes(static_cast<std::size_t>(points), sizeof(ContourSlice));
    const std::size_t area_bytes =
            aligned_bytes(static_cast<std::size_t>(width) * height, sizeof(long long));
    const std::size_t aggregate_bytes = std::max(area_bytes + root_bytes, edge_bytes);
    const std::size_t compact_edge_bytes = aligned_bytes(
            static_cast<std::size_t>(bounded_point_capacity(width, height, 2147483647)),
            sizeof(std::int32_t));
    const std::size_t count_stack_bytes = root_bytes + compact_edge_bytes;
    const std::size_t count_extras[] = {point_bytes >= edge_bytes ? 0U : edge_bytes,
                                        point_bytes >= aggregate_bytes ? 0U : aggregate_bytes,
                                        stack_bytes >= count_stack_bytes ? 0U : count_stack_bytes};
    const std::size_t scan_bytes = scan_workspace_bytes(slots);
    const std::size_t parts[] = {edge_bytes,  edge_bytes,  root_bytes,  root_bytes, root_bytes,
                                 point_bytes, point_bytes, stack_bytes, scan_bytes, root_scan_bytes,
                                 kAlignment,  kAlignment,  kAlignment,  kAlignment};
    std::size_t total = 0U;
    for (const std::size_t part : parts) {
        if (part == 0U || total > std::numeric_limits<std::size_t>::max() - part) {
            return 0U;
        }
        total += part;
    }
    for (const std::size_t extra : count_extras) {
        if (total > std::numeric_limits<std::size_t>::max() - extra) {
            return 0U;
        }
        total += extra;
    }
    return total;
}

Status reserve_contour_candidates(int width, int height, int max_points, Workspace& workspace,
                                  ContourCandidateBuffers* out) {
    if (out == nullptr || contour_candidate_workspace_bytes(width, height, max_points) == 0U) {
        return Status::kInvalidArgument;
    }
    ContourCandidateBuffers buffers;
    buffers.width_px_ = width;
    buffers.height_px_ = height;
    buffers.edge_capacity_ = edge_capacity(width, height);
    buffers.root_capacity_ = width * height;
    buffers.jump_capacity_ = bounded_point_capacity(width, height, 2147483647);
    buffers.point_capacity_ = bounded_point_capacity(width, height, max_points);
    std::int32_t** edge_arrays[] = {&buffers.parents_, &buffers.previous_};
    for (std::int32_t** target : edge_arrays) {
        void* pointer = nullptr;
        const Status status =
                workspace.allocate(aligned_bytes(static_cast<std::size_t>(buffers.edge_capacity_),
                                                 sizeof(std::int32_t)),
                                   kAlignment, &pointer);
        if (status != Status::kOk) {
            return status;
        }
        *target = static_cast<std::int32_t*>(pointer);
    }
    std::int32_t** root_arrays[] = {&buffers.roots_, &buffers.lengths_, &buffers.offsets_};
    for (std::int32_t** target : root_arrays) {
        void* pointer = nullptr;
        const Status status =
                workspace.allocate(aligned_bytes(static_cast<std::size_t>(buffers.root_capacity_),
                                                 sizeof(std::int32_t)),
                                   kAlignment, &pointer);
        if (status != Status::kOk) {
            return status;
        }
        *target = static_cast<std::int32_t*>(pointer);
    }
    ContourPoint** point_arrays[] = {&buffers.points_, &buffers.approximation_};
    for (ContourPoint** target : point_arrays) {
        void* pointer = nullptr;
        const Status status =
                workspace.allocate(aligned_bytes(static_cast<std::size_t>(buffers.point_capacity_),
                                                 sizeof(ContourPoint)),
                                   kAlignment, &pointer);
        if (status != Status::kOk) {
            return status;
        }
        *target = static_cast<ContourPoint*>(pointer);
    }
    const std::size_t point_bytes =
            aligned_bytes(static_cast<std::size_t>(buffers.point_capacity_), sizeof(ContourPoint));
    const std::size_t map_bytes =
            aligned_bytes(static_cast<std::size_t>(buffers.edge_capacity_), sizeof(std::int32_t));
    const std::size_t area_bytes =
            aligned_bytes(static_cast<std::size_t>(buffers.root_capacity_), sizeof(long long));
    const std::size_t seed_bytes =
            aligned_bytes(static_cast<std::size_t>(buffers.root_capacity_), sizeof(std::int32_t));
    void* map = buffers.points_;
    if (point_bytes < map_bytes) {
        const Status extra = workspace.allocate(map_bytes, kAlignment, &map);
        if (extra != Status::kOk) {
            return extra;
        }
    }
    buffers.count_map_ = static_cast<std::int32_t*>(map);
    void* aggregates = buffers.approximation_;
    const std::size_t aggregate_bytes = std::max(area_bytes + seed_bytes, map_bytes);
    if (point_bytes < aggregate_bytes) {
        const Status extra = workspace.allocate(aggregate_bytes, kAlignment, &aggregates);
        if (extra != Status::kOk) {
            return extra;
        }
    }
    buffers.jump_target_ = static_cast<std::int32_t*>(aggregates);
    buffers.count_area_ = static_cast<long long*>(aggregates);
    buffers.count_seed_ =
            reinterpret_cast<std::int32_t*>(static_cast<std::uint8_t*>(aggregates) + area_bytes);
    void* stack = nullptr;
    Status status = workspace.allocate(
            aligned_bytes(static_cast<std::size_t>(buffers.point_capacity_), sizeof(ContourSlice)),
            kAlignment, &stack);
    if (status != Status::kOk) {
        return status;
    }
    buffers.stack_ = static_cast<ContourSlice*>(stack);
    const std::size_t stack_bytes =
            aligned_bytes(static_cast<std::size_t>(buffers.point_capacity_), sizeof(ContourSlice));
    const std::size_t compact_edge_bytes =
            aligned_bytes(static_cast<std::size_t>(buffers.jump_capacity_), sizeof(std::int32_t));
    const std::size_t count_stack_bytes = seed_bytes + compact_edge_bytes;
    void* edge_counts = stack;
    if (stack_bytes < count_stack_bytes) {
        const Status extra = workspace.allocate(count_stack_bytes, kAlignment, &edge_counts);
        if (extra != Status::kOk) {
            return extra;
        }
    }
    buffers.count_edges_ = static_cast<std::int32_t*>(edge_counts);
    buffers.jump_edges_ =
            reinterpret_cast<std::int32_t*>(static_cast<std::uint8_t*>(edge_counts) + seed_bytes);
    status = reserve_scan(buffers.edge_capacity_, workspace, &buffers.scan_);
    if (status != Status::kOk) {
        return status;
    }
    status = reserve_scan(scan_block_count(buffers.edge_capacity_), workspace, &buffers.root_scan_);
    if (status != Status::kOk) {
        return status;
    }
    buffers.root_count_ = buffers.root_scan_.total_;
    std::int32_t** scalars[] = {&buffers.failure_, &buffers.point_total_, &buffers.base_,
                                &buffers.jump_count_};
    for (std::int32_t** target : scalars) {
        void* pointer = nullptr;
        status = workspace.allocate(kAlignment, kAlignment, &pointer);
        if (status != Status::kOk) {
            return status;
        }
        *target = static_cast<std::int32_t*>(pointer);
    }
    *out = buffers;
    return Status::kOk;
}

Status build_contour_candidates_async(const ImagePlaneU8& binary, const DetectorConfig& config,
                                      ContourCandidateBuffers* buffers,
                                      DeviceCandidates* candidates, bool append,
                                      cudaStream_t stream) {
    if (buffers == nullptr || candidates == nullptr || binary.data_ == nullptr ||
        binary.width_px_ != buffers->width_px_ || binary.height_px_ != buffers->height_px_ ||
        binary.pitch_bytes_ < static_cast<std::size_t>(binary.width_px_) ||
        buffers->parents_ == nullptr || buffers->previous_ == nullptr ||
        buffers->roots_ == nullptr || buffers->root_count_ == nullptr ||
        buffers->lengths_ == nullptr || buffers->offsets_ == nullptr ||
        buffers->count_map_ == nullptr || buffers->count_area_ == nullptr ||
        buffers->count_seed_ == nullptr || buffers->count_edges_ == nullptr ||
        buffers->jump_target_ == nullptr || buffers->jump_edges_ == nullptr ||
        buffers->jump_count_ == nullptr || buffers->points_ == nullptr ||
        buffers->approximation_ == nullptr || buffers->stack_ == nullptr ||
        buffers->failure_ == nullptr || buffers->point_total_ == nullptr ||
        buffers->base_ == nullptr || candidates->count_ == nullptr ||
        candidates->accepted_total_ == nullptr || candidates->corner_x_ == nullptr ||
        candidates->corner_y_ == nullptr || candidates->perimeter_ == nullptr ||
        candidates->label_ == nullptr || candidates->capacity_ < 1) {
        return Status::kInvalidArgument;
    }
    const Status valid = config.validate(nullptr);
    if (valid != Status::kOk) {
        return valid;
    }
    if (binary.width_px_ > config.max_width_px_ || binary.height_px_ > config.max_height_px_) {
        return Status::kInvalidArgument;
    }
    const int longest = std::max(binary.width_px_, binary.height_px_);
    const double requested_minimum =
            config.use_aruco3_detection_ && config.min_side_length_canonical_img_px_ != 0
                    ? 4.0 * config.min_side_length_canonical_img_px_
                    : config.min_marker_perimeter_rate_ * longest;
    const double requested_maximum = config.max_marker_perimeter_rate_ * longest;
    const int minimum = static_cast<int>(
            std::min(requested_minimum, static_cast<double>(buffers->edge_capacity_) + 1.0));
    const int maximum = static_cast<int>(
            std::min(requested_maximum, static_cast<double>(buffers->edge_capacity_)));
    const unsigned int blocks =
            (static_cast<unsigned int>(buffers->edge_capacity_) + kThreads - 1U) / kThreads;
    reset_invocation_kernel<<<1U, 1U, 0, stream>>>(*buffers, *candidates, append);
    Status status = check_kernel_launch("contour.reset", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    initialize_edges_kernel<<<blocks, kThreads, 0, stream>>>(binary, *buffers);
    status = check_kernel_launch("contour.initialize_edges", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    unite_edges_kernel<<<blocks, kThreads, 0, stream>>>(*buffers);
    status = check_kernel_launch("contour.unite_edges", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    mark_roots_kernel<<<blocks, kThreads, 0, stream>>>(*buffers);
    status = check_kernel_launch("contour.mark_roots", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    status = exclusive_scan_async(buffers->scan_.block_offsets_, static_cast<int>(blocks),
                                  &buffers->root_scan_, stream);
    if (status != Status::kOk) {
        return status;
    }
    scatter_roots_kernel<<<blocks, kThreads, 0, stream>>>(*buffers);
    status = check_kernel_launch("contour.scatter_roots", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    finish_roots_kernel<<<1U, 1U, 0, stream>>>(*buffers);
    status = check_kernel_launch("contour.finish_roots", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    constexpr unsigned int compact_blocks = 64U;
#if ARUCO3CUDA_EXPERIMENT_PARALLEL_CONTOUR_COUNT
    initialize_count_kernel<<<compact_blocks, kThreads, 0, stream>>>(*buffers);
    status = check_kernel_launch("contour.initialize_count", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    aggregate_count_kernel<<<compact_blocks, kThreads, 0, stream>>>(*buffers);
    status = check_kernel_launch("contour.aggregate_count", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    finish_count_kernel<<<compact_blocks, kThreads, 0, stream>>>(*buffers, minimum, maximum);
    status = check_kernel_launch("contour.finish_count", -1, false, stream);
#else
    count_paths_kernel<<<compact_blocks, kThreads, 0, stream>>>(binary, *buffers, minimum, maximum);
    status = check_kernel_launch("contour.count_paths", -1, false, stream);
#endif
    if (status != Status::kOk) {
        return status;
    }
    status = scan_compact_async(buffers->offsets_, *buffers, stream);
    if (status != Status::kOk) {
        return status;
    }
    retain_point_total_kernel<<<1U, 1U, 0, stream>>>(*buffers);
    status = check_kernel_launch("contour.point_total", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
#if ARUCO3CUDA_EXPERIMENT_COOPERATIVE_MATERIALIZE
#if ARUCO3CUDA_EXPERIMENT_COMPACT_MATERIALIZE_EDGES && ARUCO3CUDA_EXPERIMENT_PARALLEL_CONTOUR_COUNT
    compact_jump_edges_kernel<<<compact_blocks, kThreads, 0, stream>>>(*buffers);
    status = check_kernel_launch("contour.compact_jump_edges", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
#endif
    const std::int32_t* jump_source = buffers->previous_;
    for (int span = 2; span <= 32; span *= 2) {
        std::int32_t* destination =
                span == 2 || span == 8 || span == 32 ? buffers->jump_target_ : buffers->count_map_;
        jump_predecessors_kernel<<<compact_blocks, kThreads, 0, stream>>>(*buffers, jump_source,
                                                                          destination);
        status = check_kernel_launch("contour.jump_predecessors", -1, false, stream);
        if (status != Status::kOk) {
            return status;
        }
        jump_source = destination;
    }
#endif
    materialize_paths_kernel<<<compact_blocks, kThreads, 0, stream>>>(*buffers);
    status = check_kernel_launch("contour.materialize_paths", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    approximate_paths_kernel<<<compact_blocks, kThreads, 0, stream>>>(
            *buffers, config.polygonal_approx_accuracy_rate_, config.min_corner_distance_rate_);
    status = check_kernel_launch("contour.approximate_paths", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    status = scan_compact_async(buffers->previous_, *buffers, stream);
    if (status != Status::kOk) {
        return status;
    }
    compact_contours_kernel<<<compact_blocks, kThreads, 0, stream>>>(*buffers, *candidates);
    status = check_kernel_launch("contour.compact", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    store_contour_count_kernel<<<1U, 1U, 0, stream>>>(*buffers, *candidates, append);
    return check_kernel_launch("contour.store_count", -1, false, stream);
}

}  // namespace aruco3cuda::detail
