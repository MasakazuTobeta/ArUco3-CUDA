// SPDX-License-Identifier: Apache-2.0
//
// 非同期であることを test から確かめるための道具。
//
// 一定時間 device を占有する kernel を stream へ積んでおき、その後ろに
// detect_async を積む。detect_async が host 同期していなければ、呼び出しから
// 戻った時点で stream はまだ動いている。
#include "test_detector_probe.hpp"

#include <cuda_runtime_api.h>

namespace aruco3cuda::test {
namespace {

/// clock64 を読みながら指定した cycle 数だけ回る。
__global__ void spin_kernel(long long cycles, int* sink) {
    const long long start = clock64();
    long long elapsed = 0;
    while (elapsed < cycles) {
        elapsed = clock64() - start;
    }
    if (threadIdx.x == 0U && blockIdx.x == 0U) {
        *sink = static_cast<int>(elapsed & 1);
    }
}

}  // namespace

bool enqueue_spin(long long cycles, int* device_sink, cudaStream_t stream) {
    spin_kernel<<<1U, 32U, 0, stream>>>(cycles, device_sink);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace aruco3cuda::test
