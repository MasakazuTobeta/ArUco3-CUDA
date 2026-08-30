// SPDX-License-Identifier: Apache-2.0
#include "cuda_check.hpp"

#include <cuda_runtime_api.h>

#include "aruco3cuda/status.hpp"

namespace aruco3cuda::detail {

// Records into the thread_local buffer owned by status.cpp.
void store_cuda_error_message(const char* api_name, const char* stage, int device_index,
                              const void* stream, const char* cuda_error_name,
                              const char* cuda_error_string);

Status check_cuda(cudaError_t error, const char* api_name, const char* stage, int device_index,
                  cudaStream_t stream) {
    if (error == cudaSuccess) {
        return Status::kOk;
    }
    store_cuda_error_message(api_name, stage, device_index, static_cast<const void*>(stream),
                             cudaGetErrorName(error), cudaGetErrorString(error));
    return Status::kCudaError;
}

Status check_kernel_launch(const char* stage, int device_index, bool synchronize,
                           cudaStream_t stream) {
    // Check the launch itself first. Doing so clears any non-sticky error.
    const Status launch_status =
            check_cuda(cudaGetLastError(), "cudaGetLastError", stage, device_index, stream);
    if (launch_status != Status::kOk) {
        return launch_status;
    }
    if (!synchronize) {
        return Status::kOk;
    }
    // A failure during asynchronous execution can only be detected by synchronizing.
    // Wait on the given stream when there is one, otherwise on the whole device.
    if (stream != nullptr) {
        return check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize", stage,
                          device_index, stream);
    }
    return check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", stage, device_index,
                      nullptr);
}

}  // namespace aruco3cuda::detail
