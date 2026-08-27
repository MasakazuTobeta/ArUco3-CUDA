// SPDX-License-Identifier: Apache-2.0
#include "cuda_check.hpp"

#include <cuda_runtime_api.h>

#include "aruco3cuda/status.hpp"

namespace aruco3cuda::detail {

// status.cpp が保持する thread_local buffer へ記録する。
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
    // 起動自体の失敗を先に確認する。ここで sticky でないエラーが解消される。
    const Status launch_status =
            check_cuda(cudaGetLastError(), "cudaGetLastError", stage, device_index, stream);
    if (launch_status != Status::kOk) {
        return launch_status;
    }
    if (!synchronize) {
        return Status::kOk;
    }
    // 非同期実行中の失敗は同期しなければ検出できない。
    // stream が指定されていればその stream のみを、無ければ device 全体を待つ。
    if (stream != nullptr) {
        return check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize", stage,
                          device_index, stream);
    }
    return check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize", stage, device_index,
                      nullptr);
}

}  // namespace aruco3cuda::detail
