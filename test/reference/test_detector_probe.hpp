// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_TEST_DETECTOR_PROBE_HPP
#define ARUCO3CUDA_TEST_DETECTOR_PROBE_HPP

#include <cuda_runtime_api.h>

namespace aruco3cuda::test {

/// Enqueues a kernel onto the stream that occupies the device for the given
/// number of cycles.
///
/// @param cycles Number of cycles to spin. Because this depends on the device
///               clock, the elapsed time cannot be pinned down exactly; the
///               caller should pass a generously large value.
/// @param device_sink Write target that keeps the loop from being optimized
///                    away. Must be a device pointer.
/// @param stream Stream to enqueue onto.
/// @return true if the launch succeeded.
///
/// Ownership: does not retain the memory passed in as arguments.
/// Synchronization: only enqueues; never synchronizes.
///
/// Example input: cycles = 1000000000, a valid device pointer
/// Example output: true
bool enqueue_spin(long long cycles, int* device_sink, cudaStream_t stream);

}  // namespace aruco3cuda::test

#endif  // ARUCO3CUDA_TEST_DETECTOR_PROBE_HPP
