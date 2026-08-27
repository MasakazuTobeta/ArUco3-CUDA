// SPDX-License-Identifier: Apache-2.0
#include "scan.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "aruco3cuda/status.hpp"
#include "aruco3cuda/workspace.hpp"
#include "cuda_check.hpp"

namespace aruco3cuda::detail {
namespace {

constexpr std::size_t kPlaneAlignment = 256U;
/// scan 1 block あたりの thread 数と要素数。
///
/// 共有 memory 上の走査を 1 要素 1 thread で行うため両者は等しい。
constexpr int kScanThreads = 256;

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
/// 繰り越す。block 数は要素数の 1/256 であり、この逐次化は問題にならない。
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

}  // namespace

int scan_block_count(int count) {
    if (count <= 0) {
        return 0;
    }
    return (count + kScanThreads - 1) / kScanThreads;
}

std::size_t scan_workspace_bytes(int count) {
    const int blocks = scan_block_count(count);
    if (blocks == 0) {
        return 0U;
    }
    const std::size_t block_bytes =
            align_up(static_cast<std::size_t>(blocks) * sizeof(std::int32_t), kPlaneAlignment);
    const std::size_t total_bytes = align_up(sizeof(std::int32_t), kPlaneAlignment);
    if (block_bytes == 0U || total_bytes == 0U) {
        return 0U;
    }
    return block_bytes + total_bytes;
}

Status reserve_scan(int count, Workspace& workspace, ScanBuffers* out) {
    if (out == nullptr) {
        return Status::kInvalidArgument;
    }
    if (count <= 0) {
        return Status::kInvalidArgument;
    }
    ScanBuffers buffers;
    buffers.capacity_ = count;
    buffers.block_count_ = scan_block_count(count);

    void* pointer = nullptr;
    Status status = workspace.allocate(
            static_cast<std::size_t>(buffers.block_count_) * sizeof(std::int32_t), kPlaneAlignment,
            &pointer);
    if (status != Status::kOk) {
        return status;
    }
    buffers.block_offsets_ = static_cast<std::int32_t*>(pointer);

    status = workspace.allocate(sizeof(std::int32_t), kPlaneAlignment, &pointer);
    if (status != Status::kOk) {
        return status;
    }
    buffers.total_ = static_cast<std::int32_t*>(pointer);

    *out = buffers;
    return Status::kOk;
}

Status exclusive_scan_async(std::int32_t* values, int count, ScanBuffers* buffers,
                            cudaStream_t stream) {
    if (values == nullptr || buffers == nullptr || buffers->block_offsets_ == nullptr) {
        return Status::kInvalidArgument;
    }
    if (count <= 0 || count > buffers->capacity_) {
        return Status::kInvalidArgument;
    }
    const int blocks = scan_block_count(count);
    const auto grid = static_cast<unsigned int>(blocks);
    const auto block = static_cast<unsigned int>(kScanThreads);

    scan_within_block_kernel<<<grid, block, 0, stream>>>(values, buffers->block_offsets_, count);
    Status status = check_kernel_launch("scan.scan_within_block_kernel", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    scan_block_sums_kernel<<<1U, block, 0, stream>>>(buffers->block_offsets_, blocks,
                                                     buffers->total_);
    status = check_kernel_launch("scan.scan_block_sums_kernel", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    add_block_offsets_kernel<<<grid, block, 0, stream>>>(values, buffers->block_offsets_, count);
    return check_kernel_launch("scan.add_block_offsets_kernel", -1, false, stream);
}

}  // namespace aruco3cuda::detail
