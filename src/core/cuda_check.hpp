// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CORE_CUDA_CHECK_HPP
#define ARUCO3CUDA_CORE_CUDA_CHECK_HPP

#include <cuda_runtime_api.h>

#include "aruco3cuda/status.hpp"

namespace aruco3cuda::detail {

/// Checks the return value of a CUDA API call and records the context on failure.
///
/// Every CUDA API call goes through this function so that no failure is silently ignored.
/// The recorded text can be retrieved with last_cuda_error_message().
///
/// MISRA C++ 2023 discourages function-like macros, so the call-site information is passed as
/// explicit arguments instead of being captured by a macro.
///
/// @param error Return value of the CUDA API call.
/// @param api_name Name of the CUDA API that was called. Pass a string with static storage.
/// @param stage Processing stage. Identifies which kernel or step failed.
/// @param device_index Target device. Pass -1 when it is unknown.
/// @param stream Target stream. Pass nullptr for the default stream or for APIs without a stream.
/// @return Status::kOk when error is cudaSuccess, otherwise Status::kCudaError.
///
/// Example input: check_cuda(cudaErrorInvalidValue, "cudaMemcpyAsync", "upload", 0, stream)
/// Example output: Status::kCudaError, with last_cuda_error_message() returning
///         "api=cudaMemcpyAsync stage=upload device=0 stream=0x... error=..."
Status check_cuda(cudaError_t error, const char* api_name, const char* stage, int device_index,
                  cudaStream_t stream = nullptr);

/// Checks whether the most recent kernel launch failed.
///
/// A kernel launch is asynchronous, so a failure of the launch itself and a failure during
/// execution are checked separately. When synchronize is true, the call also synchronizes and
/// detects failures that occur during execution.
///
/// @param stage Processing stage.
/// @param device_index Target device.
/// @param synchronize Whether to synchronize and detect runtime errors as well.
/// @param stream Target stream. Pass nullptr for the default stream.
/// @return kOk or kCudaError.
Status check_kernel_launch(const char* stage, int device_index, bool synchronize,
                           cudaStream_t stream = nullptr);

}  // namespace aruco3cuda::detail

#endif  // ARUCO3CUDA_CORE_CUDA_CHECK_HPP
