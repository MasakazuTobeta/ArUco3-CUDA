// SPDX-License-Identifier: Apache-2.0
//
// Verifies that check_kernel_launch detects a launch that never started.
//
// A launch and an execution fail in different ways, and only cudaGetLastError
// reports the first. check_kernel_launch calls it before it consults the
// synchronize flag, so a failed launch is caught even when the caller does not
// synchronize - which is what all 50 launch sites in the pipeline do.
//
// The tests in test_cuda_check.cpp cover only the clean path of that function.
// Deleting its cudaGetLastError call would leave every one of them green while
// the detector stopped noticing a kernel that could not run, which is the defect
// that reached the environment probe and was fixed there separately.
//
// This file is CUDA rather than C++ so that the failure comes from a real
// launch instead of a stand-in for one.
#include "cuda_check.hpp"

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <string>

#include "aruco3cuda/status.hpp"

namespace {

__global__ void noop_kernel() {}

/// Ask for a block larger than the device allows, which fails the launch before
/// it starts. Returns false when there is no device to ask.
///
/// The error is left pending on purpose: observing it is what the tests below
/// are for. It is cudaErrorInvalidConfiguration, which is not sticky, so the
/// context stays usable for whatever runs next once the error is cleared.
///
/// @param out_max_threads Receives the device's maximum, for the message.
/// @return true when a launch was attempted.
///
/// Example input: a machine with one CUDA device
/// Example output: true, with a launch error pending
bool provoke_launch_failure(int* out_max_threads) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        return false;
    }
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) {
        return false;
    }
    *out_max_threads = properties.maxThreadsPerBlock;
    (void)cudaGetLastError();  // clear any state left behind by an earlier test
    noop_kernel<<<1, properties.maxThreadsPerBlock + 1>>>();
    return true;
}

}  // namespace

// Error: a launch that never started is reported, and reported without
// synchronizing. Every launch site in the pipeline passes synchronize=false, so
// this is the path that matters; if the cudaGetLastError call were removed from
// check_kernel_launch, this is the assertion that would fail.
TEST(CudaCheckDeliberateErrorTest, launch_failure_is_detected_without_synchronizing) {
    int max_threads = 0;
    if (!provoke_launch_failure(&max_threads)) {
        GTEST_SKIP() << "skipped: no CUDA device is available in this environment";
    }
    EXPECT_EQ(aruco3cuda::detail::check_kernel_launch("deliberate_launch", 0, false),
              aruco3cuda::Status::kCudaError)
            << "a block of " << max_threads + 1 << " threads should not have launched";
    const std::string message = aruco3cuda::last_cuda_error_message();
    EXPECT_NE(message.find("api=cudaGetLastError"), std::string::npos) << message;
    EXPECT_NE(message.find("stage=deliberate_launch"), std::string::npos) << message;
    (void)cudaGetLastError();
}

// Error: the same failure with synchronize=true. The launch is checked before
// the synchronize, so the answer must be the launch error rather than whatever
// the synchronize returns.
TEST(CudaCheckDeliberateErrorTest, launch_failure_is_reported_ahead_of_the_synchronize) {
    int max_threads = 0;
    if (!provoke_launch_failure(&max_threads)) {
        GTEST_SKIP() << "skipped: no CUDA device is available in this environment";
    }
    EXPECT_EQ(aruco3cuda::detail::check_kernel_launch("deliberate_sync", 0, true),
              aruco3cuda::Status::kCudaError);
    const std::string message = aruco3cuda::last_cuda_error_message();
    EXPECT_NE(message.find("api=cudaGetLastError"), std::string::npos) << message;
    (void)cudaGetLastError();
}

// The premise the function rests on, asserted rather than assumed: a
// synchronize on its own does not see this failure. Without this, a reader has
// to take on trust that checking the synchronize alone would be insufficient,
// which is precisely the assumption that produced the defect elsewhere.
TEST(CudaCheckDeliberateErrorTest, a_synchronize_alone_does_not_see_a_failed_launch) {
    int max_threads = 0;
    if (!provoke_launch_failure(&max_threads)) {
        GTEST_SKIP() << "skipped: no CUDA device is available in this environment";
    }
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess)
            << "the launch never started, so there is nothing for the device to finish";
    // Which code comes back is not pinned down. A block that is too large is
    // reported as cudaErrorInvalidValue by CUDA 13.0 and the runtimes differ on
    // this; what the test is about is that the failure is there to be seen at
    // all, and that only this call sees it.
    const cudaError_t pending = cudaGetLastError();
    EXPECT_NE(pending, cudaSuccess)
            << "the failure was pending the whole time and only this call reports it";
}
