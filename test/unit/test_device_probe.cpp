// SPDX-License-Identifier: Apache-2.0
//
// Verifies the nominal, error, and boundary behavior of the device probe.
//
// Note:
//   The test must still hold up in an environment with no CUDA device. Checks
//   that depend on a device run only when one is present; otherwise they are
//   recorded as skipped.
#include "aruco3cuda/device_probe.hpp"

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include "aruco3cuda/status.hpp"

namespace {

int available_device_count() {
    int count = 0;
    if (aruco3cuda::device_count(&count) != aruco3cuda::Status::kOk) {
        return 0;
    }
    return count;
}

// Error case: a null output pointer is rejected.
TEST(DeviceProbeTest, device_count_rejects_null_output) {
    EXPECT_EQ(aruco3cuda::device_count(nullptr), aruco3cuda::Status::kInvalidArgument);
}

// Nominal: the device count can be queried and a non-negative value is written.
// The helper swallows failures, so without this test a device_count that always
// failed would merely skip the other tests and go unnoticed.
TEST(DeviceProbeTest, device_count_succeeds_and_writes_non_negative) {
    int count = -1;
    const aruco3cuda::Status status = aruco3cuda::device_count(&count);
    ASSERT_EQ(status, aruco3cuda::Status::kOk) << aruco3cuda::last_cuda_error_message();
    EXPECT_GE(count, 0);
}

// Error case: a null output pointer is rejected.
TEST(DeviceProbeTest, probe_device_rejects_null_output) {
    EXPECT_EQ(aruco3cuda::probe_device(0, nullptr), aruco3cuda::Status::kInvalidArgument);
}

// Error case: a negative index is rejected.
TEST(DeviceProbeTest, probe_device_rejects_negative_index) {
    aruco3cuda::DeviceProbeResult result;
    EXPECT_EQ(aruco3cuda::probe_device(-1, &result), aruco3cuda::Status::kInvalidArgument);
}

// Boundary: the device count itself is not a valid index.
TEST(DeviceProbeTest, probe_device_rejects_index_equal_to_device_count) {
    const int count = available_device_count();
    aruco3cuda::DeviceProbeResult result;
    EXPECT_EQ(aruco3cuda::probe_device(count, &result), aruco3cuda::Status::kInvalidArgument);
}

// Error case: the output is left unchanged on failure.
TEST(DeviceProbeTest, probe_device_leaves_output_unchanged_on_failure) {
    aruco3cuda::DeviceProbeResult result;
    result.compute_capability_major_ = 42;
    const int count = available_device_count();
    ASSERT_EQ(aruco3cuda::probe_device(count, &result), aruco3cuda::Status::kInvalidArgument);
    EXPECT_EQ(result.compute_capability_major_, 42);
}

// Nominal: the device properties can be read when a device exists.
TEST(DeviceProbeTest, probe_device_returns_properties_when_device_exists) {
    const int count = available_device_count();
    if (count == 0) {
        GTEST_SKIP() << "skipped: no CUDA device is available in this environment";
    }
    aruco3cuda::DeviceProbeResult result;
    ASSERT_EQ(aruco3cuda::probe_device(0, &result), aruco3cuda::Status::kOk);
    EXPECT_EQ(result.device_index_, 0);
    EXPECT_GE(result.compute_capability_major_, 5);
    EXPECT_GT(result.multi_processor_count_, 0);
    EXPECT_GT(result.l2_cache_bytes_, 0U);
    EXPECT_FALSE(result.name_.empty());
}

// Nominal: the self-test kernel returns the expected computation result.
TEST(DeviceProbeTest, self_test_succeeds_when_device_exists) {
    const int count = available_device_count();
    if (count == 0) {
        GTEST_SKIP() << "skipped: no CUDA device is available in this environment";
    }
    EXPECT_EQ(aruco3cuda::run_device_self_test(0), aruco3cuda::Status::kOk)
            << aruco3cuda::last_cuda_error_message();
}

// Error and boundary cases: out-of-range and negative indices are rejected.
TEST(DeviceProbeTest, self_test_rejects_out_of_range_index) {
    const int count = available_device_count();
    EXPECT_EQ(aruco3cuda::run_device_self_test(count), aruco3cuda::Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::run_device_self_test(-1), aruco3cuda::Status::kInvalidArgument);
}

}  // namespace
