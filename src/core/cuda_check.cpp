// SPDX-License-Identifier: Apache-2.0
#include "cuda_check.hpp"

#include <cuda_runtime_api.h>

#include "aruco3cuda/status.hpp"

namespace aruco3cuda::detail {

// status.cpp が保持する thread_local buffer へ記録する。
void store_cuda_error_message(const char* api_name, const char* stage, int device_index,
                              const char* cuda_error_name, const char* cuda_error_string);

Status check_cuda(cudaError_t error, const char* api_name, const char* stage, int device_index) {
    if (error == cudaSuccess) {
        return Status::kOk;
    }
    store_cuda_error_message(api_name, stage, device_index, cudaGetErrorName(error),
                             cudaGetErrorString(error));
    return Status::kCudaError;
}

Status check_kernel_launch(const char* stage, int device_index, bool synchronize) {
    // 起動自体の失敗を先に確認する。ここで sticky でないエラーが解消される。
    const Status launch_status =
            check_cuda(cudaGetLastError(), "cudaGetLastError", stage, device_index);
    if (launch_status != Status::kOk) {
        return launch_status;
    }
    if (!synchronize) {
        return Status::kOk;
    }
    // 非同期実行中の失敗は同期しなければ検出できない。
    return check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", stage, device_index);
}

}  // namespace aruco3cuda::detail
