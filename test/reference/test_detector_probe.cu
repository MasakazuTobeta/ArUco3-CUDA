// SPDX-License-Identifier: Apache-2.0
//
// Tooling that lets tests confirm that a call is genuinely asynchronous.
//
// A kernel that occupies the device for a while is enqueued onto the stream,
// and detect_async is enqueued behind it. If detect_async does not synchronize
// with the host, the stream is still running by the time the call returns.
#include "test_detector_probe.hpp"

#include <cuda_runtime_api.h>

namespace aruco3cuda::test {
namespace {

/// Spins for the requested number of cycles while polling clock64.
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
