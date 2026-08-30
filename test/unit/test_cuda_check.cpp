// SPDX-License-Identifier: Apache-2.0
//
// Verifies that CUDA errors are recorded with enough context.
//
// CONTRIBUTING.md requires that a CUDA error carry context that lets the API
// name, device, stream, and processing stage be traced. That requirement only
// shows up on failure paths, so this test includes the internal header directly
// in order to provoke those paths deliberately.
#include "cuda_check.hpp"

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <string>

#include "aruco3cuda/status.hpp"

namespace {

// Nominal: a successful API call returns kOk.
TEST(CudaCheckTest, success_returns_ok) {
    EXPECT_EQ(aruco3cuda::detail::check_cuda(cudaSuccess, "cudaTestApi", "test_stage", 0),
              aruco3cuda::Status::kOk);
}

// Error case: a failed API call returns kCudaError and records traceable context.
TEST(CudaCheckTest, failure_records_api_stage_and_device) {
    const aruco3cuda::Status status =
            aruco3cuda::detail::check_cuda(cudaErrorInvalidValue, "cudaTestApi", "test_stage", 3);
    EXPECT_EQ(status, aruco3cuda::Status::kCudaError);

    const std::string message = aruco3cuda::last_cuda_error_message();
    EXPECT_NE(message.find("api=cudaTestApi"), std::string::npos) << message;
    EXPECT_NE(message.find("stage=test_stage"), std::string::npos) << message;
    EXPECT_NE(message.find("device=3"), std::string::npos) << message;
    // The stream belongs in the context too; it becomes necessary once several
    // streams run concurrently.
    EXPECT_NE(message.find("stream="), std::string::npos) << message;
    // The CUDA error name and description are included as well, so that the exact
    // failure can be identified after the fact.
    EXPECT_NE(message.find(cudaGetErrorName(cudaErrorInvalidValue)), std::string::npos) << message;
}

// Nominal: a success does not overwrite the previously recorded message.
// Even when a successful call immediately follows a failure, the failure context
// must not be lost.
TEST(CudaCheckTest, success_does_not_clear_previous_message) {
    ASSERT_EQ(aruco3cuda::detail::check_cuda(cudaErrorNotReady, "cudaFirstApi", "first_stage", 1),
              aruco3cuda::Status::kCudaError);
    ASSERT_EQ(aruco3cuda::detail::check_cuda(cudaSuccess, "cudaSecondApi", "second_stage", 1),
              aruco3cuda::Status::kOk);
    const std::string message = aruco3cuda::last_cuda_error_message();
    EXPECT_NE(message.find("api=cudaFirstApi"), std::string::npos) << message;
}

// Error case: a failure returned by a real CUDA API takes the same path.
//
// This test makes a CUDA API fail on purpose. Compute Sanitizer detects and
// reports every CUDA API error whether or not it was intended, so the sanitizer
// runs exclude this test by suite name. That is why it lives in its own suite.
TEST(CudaCheckDeliberateErrorTest, real_cuda_failure_is_recorded) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        GTEST_SKIP() << "skipped: no CUDA device is available in this environment";
    }
    // Ask for a device index that does not exist; an out-of-range index always fails.
    const cudaError_t error = cudaSetDevice(device_count + 100);
    ASSERT_NE(error, cudaSuccess);
    EXPECT_EQ(aruco3cuda::detail::check_cuda(error, "cudaSetDevice", "real_failure_stage", -1),
              aruco3cuda::Status::kCudaError);
    const std::string message = aruco3cuda::last_cuda_error_message();
    EXPECT_NE(message.find("api=cudaSetDevice"), std::string::npos) << message;
    EXPECT_NE(message.find("device=-1"), std::string::npos) << message;
    // Restore a valid device so that the following tests are unaffected.
    (void)cudaGetLastError();
    (void)cudaSetDevice(0);
}

// Nominal: with no launch failure pending, kOk is returned. The behavior is
// exercised both with and without synchronization.
TEST(CudaCheckTest, kernel_launch_check_passes_when_clean) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        GTEST_SKIP() << "skipped: no CUDA device is available in this environment";
    }
    (void)cudaGetLastError();  // clear any state left behind by an earlier test
    EXPECT_EQ(aruco3cuda::detail::check_kernel_launch("clean_stage", 0, false),
              aruco3cuda::Status::kOk);
    EXPECT_EQ(aruco3cuda::detail::check_kernel_launch("clean_stage", 0, true),
              aruco3cuda::Status::kOk);
}

// Boundary: a record longer than the buffer is truncated without corrupting its end.
TEST(CudaCheckTest, long_context_is_truncated_safely) {
    const std::string long_stage(2048, 'x');
    EXPECT_EQ(aruco3cuda::detail::check_cuda(cudaErrorInvalidValue, "cudaTestApi",
                                             long_stage.c_str(), 0),
              aruco3cuda::Status::kCudaError);
    const std::string message = aruco3cuda::last_cuda_error_message();
    EXPECT_LT(message.size(), 1024U);
    EXPECT_NE(message.find("api=cudaTestApi"), std::string::npos);
}

// Nominal: when a stream is given, its value appears in the context.
TEST(CudaCheckTest, stream_is_recorded_in_context) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        GTEST_SKIP() << "skipped: no CUDA device is available in this environment";
    }
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    EXPECT_EQ(aruco3cuda::detail::check_cuda(cudaErrorInvalidValue, "cudaTestApi", "stream_stage",
                                             0, stream),
              aruco3cuda::Status::kCudaError);
    const std::string message = aruco3cuda::last_cuda_error_message();
    EXPECT_NE(message.find("stream=0x"), std::string::npos) << message;
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

// Nominal: a synchronizing check with a stream waits on that stream rather than
// on the whole device.
TEST(CudaCheckTest, kernel_launch_check_accepts_stream) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        GTEST_SKIP() << "skipped: no CUDA device is available in this environment";
    }
    (void)cudaGetLastError();
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    EXPECT_EQ(aruco3cuda::detail::check_kernel_launch("stream_clean", 0, true, stream),
              aruco3cuda::Status::kOk);
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

}  // namespace
