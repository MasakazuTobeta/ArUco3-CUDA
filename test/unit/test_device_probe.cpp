// SPDX-License-Identifier: Apache-2.0
//
// device probe の正常系、異常系、境界値を検証する。
//
// 備考:
//   CUDA device が存在しない環境でも test 自体は成立させる。device 依存の
//   検証は device がある場合にのみ実行し、無い場合は skip として記録する。
#include "aruco3cuda/device_probe.hpp"

#include <gtest/gtest.h>

#include "aruco3cuda/status.hpp"

namespace {

int available_device_count() {
    int count = 0;
    if (aruco3cuda::device_count(&count) != aruco3cuda::Status::kOk) {
        return 0;
    }
    return count;
}

// 異常系: nullptr を拒否する。
TEST(DeviceProbeTest, device_count_rejects_null_output) {
    EXPECT_EQ(aruco3cuda::device_count(nullptr), aruco3cuda::Status::kInvalidArgument);
}

// 異常系: nullptr を拒否する。
TEST(DeviceProbeTest, probe_device_rejects_null_output) {
    EXPECT_EQ(aruco3cuda::probe_device(0, nullptr), aruco3cuda::Status::kInvalidArgument);
}

// 異常系: 負の index を拒否する。
TEST(DeviceProbeTest, probe_device_rejects_negative_index) {
    aruco3cuda::DeviceProbeResult result;
    EXPECT_EQ(aruco3cuda::probe_device(-1, &result), aruco3cuda::Status::kInvalidArgument);
}

// 境界値: device 数そのものは有効な index ではない。
TEST(DeviceProbeTest, probe_device_rejects_index_equal_to_device_count) {
    const int count = available_device_count();
    aruco3cuda::DeviceProbeResult result;
    EXPECT_EQ(aruco3cuda::probe_device(count, &result), aruco3cuda::Status::kInvalidArgument);
}

// 異常系: 失敗時に出力先を変更しない。
TEST(DeviceProbeTest, probe_device_leaves_output_unchanged_on_failure) {
    aruco3cuda::DeviceProbeResult result;
    result.compute_capability_major_ = 42;
    const int count = available_device_count();
    ASSERT_EQ(aruco3cuda::probe_device(count, &result), aruco3cuda::Status::kInvalidArgument);
    EXPECT_EQ(result.compute_capability_major_, 42);
}

// 正常系: device が存在する場合に性質を取得できる。
TEST(DeviceProbeTest, probe_device_returns_properties_when_device_exists) {
    const int count = available_device_count();
    if (count == 0) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    aruco3cuda::DeviceProbeResult result;
    ASSERT_EQ(aruco3cuda::probe_device(0, &result), aruco3cuda::Status::kOk);
    EXPECT_EQ(result.device_index_, 0);
    EXPECT_GE(result.compute_capability_major_, 5);
    EXPECT_GT(result.multi_processor_count_, 0);
    EXPECT_GT(result.l2_cache_bytes_, 0U);
}

// 正常系: 自己診断 kernel が期待どおりの計算結果を返す。
TEST(DeviceProbeTest, self_test_succeeds_when_device_exists) {
    const int count = available_device_count();
    if (count == 0) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    EXPECT_EQ(aruco3cuda::run_device_self_test(0), aruco3cuda::Status::kOk)
            << aruco3cuda::last_cuda_error_message();
}

// 異常系と境界値: 範囲外および負の index を拒否する。
TEST(DeviceProbeTest, self_test_rejects_out_of_range_index) {
    const int count = available_device_count();
    EXPECT_EQ(aruco3cuda::run_device_self_test(count), aruco3cuda::Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::run_device_self_test(-1), aruco3cuda::Status::kInvalidArgument);
}

}  // namespace
