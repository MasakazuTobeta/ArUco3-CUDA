// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_DEVICE_PROBE_HPP
#define ARUCO3CUDA_DEVICE_PROBE_HPP

#include <cstddef>
#include <string>

#include "aruco3cuda/status.hpp"

namespace aruco3cuda {

/// The device properties that affect implementation and evaluation decisions.
///
/// integrated_ directly drives how the evaluation plan treats the memory paths. On an
/// integrated GPU, host and device share the same physical memory, so the cost of an
/// explicit copy differs greatly from a discrete GPU.
struct DeviceProbeResult {
    int device_index_ = -1;
    /// Device name. Obtainable without depending on nvidia-smi.
    std::string name_;
    int compute_capability_major_ = 0;
    int compute_capability_minor_ = 0;
    int multi_processor_count_ = 0;
    std::size_t l2_cache_bytes_ = 0;
    bool integrated_ = false;                 ///< Whether this is an integrated GPU
    bool concurrent_managed_access_ = false;  ///< Whether managed memory allows concurrent access
};

/// Retrieves the number of usable CUDA devices.
///
/// @param out_count On success, receives the device count. Must not be nullptr.
///                  Ownership of the storage stays with the caller.
/// @return One of kOk, kInvalidArgument, or kCudaError.
///
/// Synchronization: calls into the CUDA runtime but performs no device synchronization.
///                  Details of a failure are available from last_cuda_error_message().
///
/// Example input: a pointer to a valid int
/// Example output: *out_count = 1
Status device_count(int* out_count);

/// Retrieves the properties of the given device.
///
/// @param device_index At least 0 and less than the device count.
/// @param out On success, receives the result; left unchanged on failure. Ownership of
///            the storage stays with the caller. The returned name_ is a copied string
///            and does not reference memory held by CUDA.
/// @return One of kOk, kInvalidArgument, or kCudaError.
///
/// Synchronization: calls into the CUDA runtime but performs no device synchronization
///                  and does not change the current device. Details of a failure are
///                  available from last_cuda_error_message().
///
/// Example input: device_index = 0
/// Example output: compute_capability_major_ = 12, integrated_ = true
Status probe_device(int device_index, DeviceProbeResult* out);

/// Runs a minimal kernel to confirm that computation works on the device.
///
/// The point is to check that the build infrastructure and the runtime environment
/// actually talk to each other.
///
/// @param device_index The device to test. At least 0 and less than the device count.
/// @return One of kOk, kInvalidArgument, or kCudaError.
///
/// Ownership: this function owns the device buffer it allocates internally and frees it
///            before returning on both the success and the failure path. Nothing is
///            handed to the caller.
///
/// Synchronization: calls cudaDeviceSynchronize() internally and waits for the kernel to
///                  finish. It temporarily switches the current device to device_index
///                  and restores the value it had on entry.
///
/// Example input: device_index = 0
/// Example output: Status::kOk
Status run_device_self_test(int device_index);

}  // namespace aruco3cuda

#endif  // ARUCO3CUDA_DEVICE_PROBE_HPP
