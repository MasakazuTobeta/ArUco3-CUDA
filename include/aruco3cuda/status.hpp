// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_STATUS_HPP
#define ARUCO3CUDA_STATUS_HPP

namespace aruco3cuda {

/// Result state of the public API.
///
/// The core never throws. Failures are all reported through this value so that
/// no configuration lets an exception propagate out of a destructor, a CUDA
/// callback, or device code.
enum class Status : int {
    kOk = 0,
    kInvalidArgument,        ///< A pointer, size, index, or similar is out of range
    kInvalidImage,           ///< Image view pointer, size, stride, or memory space is invalid
    kInvalidConfig,          ///< A setting is out of range or contradicts another one
    kUnsupportedDictionary,  ///< Unsupported dictionary
    kCandidateOverflow,      ///< The candidate limit was exceeded; results are truncated
    kMarkerOverflow,         ///< The detection limit was exceeded; results are truncated
    kCudaError,              ///< A CUDA API call or a kernel launch failed
    kNotInitialized,         ///< Called before initialization
};

/// Converts a Status into its identifier string. Used for logging and test assertions.
///
/// @param status Value to convert. Never returns nullptr, even for an unknown value.
/// @return A string with static storage duration.
///
/// Ownership: the return value points into static storage. The caller neither frees
///            nor modifies it.
/// Synchronization: host only, with no synchronization point. Reentrant.
///
/// Example input: Status::kCudaError
/// Example output: "kCudaError"
const char* to_string(Status status);

/// Returns a description of the most recently recorded CUDA error.
///
/// The description carries the CUDA API name, the device, the processing stage, and
/// the error string reported by CUDA. It is held per thread. If nothing has been
/// recorded yet, an empty string is returned.
///
/// @return A string pointing into thread_local storage, valid until the next CUDA
///         error is recorded.
///
/// Ownership: the return value points into storage owned by the library. The caller
///            does not free it. The contents are overwritten when the next CUDA error
///            is recorded on the same thread.
/// Synchronization: host only, with no synchronization point. Independent per thread.
///
/// Example input: called right after check_cuda recorded a cudaMalloc failure
/// Example output: "api=cudaMalloc stage=run_device_self_test device=0 stream=(nil) error=..."
const char* last_cuda_error_message();

}  // namespace aruco3cuda

#endif  // ARUCO3CUDA_STATUS_HPP
