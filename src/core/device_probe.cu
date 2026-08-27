// SPDX-License-Identifier: Apache-2.0
#include "aruco3cuda/device_probe.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "cuda_check.hpp"

namespace aruco3cuda {
namespace {

constexpr int kSelfTestElementCount = 256;
constexpr int kSelfTestBlockSize = 128;

/// 自己診断用の最小 kernel。
///
/// thread とデータの対応:
///   thread i が out[i] を 1 要素だけ書き込む。thread 間で書き込み先が
///   重複しないため競合は発生しない。
/// 境界条件:
///   count が block size の倍数でない場合に備え、範囲外を書き込まない。
__global__ void fill_squares_kernel(std::int32_t* out, int count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count) {
        out[index] = static_cast<std::int32_t>(index) * static_cast<std::int32_t>(index);
    }
}

}  // namespace

Status device_count(int* out_count) {
    if (out_count == nullptr) {
        return Status::kInvalidArgument;
    }
    int count = 0;
    const Status status = detail::check_cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount",
                                             "device_count", -1);
    if (status != Status::kOk) {
        return status;
    }
    *out_count = count;
    return Status::kOk;
}

Status probe_device(int device_index, DeviceProbeResult* out) {
    if (out == nullptr || device_index < 0) {
        return Status::kInvalidArgument;
    }
    int count = 0;
    const Status count_status = device_count(&count);
    if (count_status != Status::kOk) {
        return count_status;
    }
    if (device_index >= count) {
        return Status::kInvalidArgument;
    }

    cudaDeviceProp prop{};
    const Status prop_status =
            detail::check_cuda(cudaGetDeviceProperties(&prop, device_index),
                               "cudaGetDeviceProperties", "probe_device", device_index);
    if (prop_status != Status::kOk) {
        return prop_status;
    }

    // 失敗時に out を変更しないため、全て取得できてから書き込む。
    out->device_index_ = device_index;
    out->compute_capability_major_ = prop.major;
    out->compute_capability_minor_ = prop.minor;
    out->multi_processor_count_ = prop.multiProcessorCount;
    out->l2_cache_bytes_ = static_cast<std::size_t>(prop.l2CacheSize);
    out->integrated_ = prop.integrated != 0;
    out->concurrent_managed_access_ = prop.concurrentManagedAccess != 0;
    return Status::kOk;
}

Status run_device_self_test(int device_index) {
    if (device_index < 0) {
        return Status::kInvalidArgument;
    }
    int count = 0;
    const Status count_status = device_count(&count);
    if (count_status != Status::kOk) {
        return count_status;
    }
    if (device_index >= count) {
        return Status::kInvalidArgument;
    }

    const Status set_status = detail::check_cuda(cudaSetDevice(device_index), "cudaSetDevice",
                                                 "run_device_self_test", device_index);
    if (set_status != Status::kOk) {
        return set_status;
    }

    std::int32_t* device_buffer = nullptr;
    constexpr std::size_t kBytes = sizeof(std::int32_t) * kSelfTestElementCount;
    const Status malloc_status = detail::check_cuda(
            cudaMalloc(&device_buffer, kBytes), "cudaMalloc", "run_device_self_test", device_index);
    if (malloc_status != Status::kOk) {
        return malloc_status;
    }

    Status result = Status::kOk;
    const int block_count = (kSelfTestElementCount + kSelfTestBlockSize - 1) / kSelfTestBlockSize;
    fill_squares_kernel<<<block_count, kSelfTestBlockSize>>>(device_buffer, kSelfTestElementCount);
    result = detail::check_kernel_launch("run_device_self_test.fill_squares_kernel", device_index,
                                         true);

    if (result == Status::kOk) {
        std::int32_t host_buffer[kSelfTestElementCount] = {};
        result = detail::check_cuda(
                cudaMemcpy(host_buffer, device_buffer, kBytes, cudaMemcpyDeviceToHost),
                "cudaMemcpy", "run_device_self_test", device_index);
        if (result == Status::kOk) {
            // 計算結果が期待どおりかを確認する。転送だけが成功しても意味がない。
            for (int i = 0; i < kSelfTestElementCount; ++i) {
                if (host_buffer[i] != static_cast<std::int32_t>(i) * static_cast<std::int32_t>(i)) {
                    result = Status::kCudaError;
                    break;
                }
            }
        }
    }

    // 失敗経路でも必ず解放する。free の失敗は元のエラーを上書きしない。
    const Status free_status = detail::check_cuda(cudaFree(device_buffer), "cudaFree",
                                                  "run_device_self_test", device_index);
    if (result == Status::kOk) {
        result = free_status;
    }
    return result;
}

}  // namespace aruco3cuda
