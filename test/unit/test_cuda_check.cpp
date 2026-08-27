// SPDX-License-Identifier: Apache-2.0
//
// CUDA エラーの文脈記録を検証する。
//
// CONTRIBUTING.md は「CUDA のエラーには API 名、device、stream、処理段階を
// 追跡できる文脈を付ける」と定める。この要件は失敗経路でしか現れないため、
// 内部 header を直接使って失敗経路を意図的に発生させる。
#include "cuda_check.hpp"

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <string>

#include "aruco3cuda/status.hpp"

namespace {

// 正常系: 成功した API 呼び出しは kOk を返す。
TEST(CudaCheckTest, success_returns_ok) {
    EXPECT_EQ(aruco3cuda::detail::check_cuda(cudaSuccess, "cudaTestApi", "test_stage", 0),
              aruco3cuda::Status::kOk);
}

// 異常系: 失敗した API 呼び出しは kCudaError を返し、追跡できる文脈を記録する。
TEST(CudaCheckTest, failure_records_api_stage_and_device) {
    const aruco3cuda::Status status =
            aruco3cuda::detail::check_cuda(cudaErrorInvalidValue, "cudaTestApi", "test_stage", 3);
    EXPECT_EQ(status, aruco3cuda::Status::kCudaError);

    const std::string message = aruco3cuda::last_cuda_error_message();
    EXPECT_NE(message.find("api=cudaTestApi"), std::string::npos) << message;
    EXPECT_NE(message.find("stage=test_stage"), std::string::npos) << message;
    EXPECT_NE(message.find("device=3"), std::string::npos) << message;
    // stream も文脈へ含める。複数 stream を並行して使う段階で必要になる。
    EXPECT_NE(message.find("stream="), std::string::npos) << message;
    // CUDA 側の error 名と説明も含める。どの失敗かを後から特定できる必要がある。
    EXPECT_NE(message.find(cudaGetErrorName(cudaErrorInvalidValue)), std::string::npos) << message;
}

// 正常系: 成功時に直前の記録を上書きしない。
// 失敗の直後に成功した呼び出しがあっても、失敗の文脈が失われてはならない。
TEST(CudaCheckTest, success_does_not_clear_previous_message) {
    ASSERT_EQ(aruco3cuda::detail::check_cuda(cudaErrorNotReady, "cudaFirstApi", "first_stage", 1),
              aruco3cuda::Status::kCudaError);
    ASSERT_EQ(aruco3cuda::detail::check_cuda(cudaSuccess, "cudaSecondApi", "second_stage", 1),
              aruco3cuda::Status::kOk);
    const std::string message = aruco3cuda::last_cuda_error_message();
    EXPECT_NE(message.find("api=cudaFirstApi"), std::string::npos) << message;
}

// 異常系: 実際の CUDA API が返す失敗でも同じ経路を通る。
//
// この test は CUDA API を意図的に失敗させる。Compute Sanitizer は
// 意図の有無に関わらず全ての CUDA API エラーを検出して報告するため、
// sanitizer 実行では suite 名で除外する。suite を分けているのはそのためである。
TEST(CudaCheckDeliberateErrorTest, real_cuda_failure_is_recorded) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    // 存在しない device 番号を指定する。範囲外は必ず失敗する。
    const cudaError_t error = cudaSetDevice(device_count + 100);
    ASSERT_NE(error, cudaSuccess);
    EXPECT_EQ(aruco3cuda::detail::check_cuda(error, "cudaSetDevice", "real_failure_stage", -1),
              aruco3cuda::Status::kCudaError);
    const std::string message = aruco3cuda::last_cuda_error_message();
    EXPECT_NE(message.find("api=cudaSetDevice"), std::string::npos) << message;
    EXPECT_NE(message.find("device=-1"), std::string::npos) << message;
    // 以降の test へ影響させないため、有効な device へ戻す。
    (void)cudaGetLastError();
    (void)cudaSetDevice(0);
}

// 正常系: 起動の失敗が無い状態では kOk を返す。同期の有無で挙動を分ける。
TEST(CudaCheckTest, kernel_launch_check_passes_when_clean) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    (void)cudaGetLastError();  // 直前の test が残した状態を消す
    EXPECT_EQ(aruco3cuda::detail::check_kernel_launch("clean_stage", 0, false),
              aruco3cuda::Status::kOk);
    EXPECT_EQ(aruco3cuda::detail::check_kernel_launch("clean_stage", 0, true),
              aruco3cuda::Status::kOk);
}

// 境界値: 記録は buffer 長を超えても切り詰められ、末尾が壊れない。
TEST(CudaCheckTest, long_context_is_truncated_safely) {
    const std::string long_stage(2048, 'x');
    EXPECT_EQ(aruco3cuda::detail::check_cuda(cudaErrorInvalidValue, "cudaTestApi",
                                             long_stage.c_str(), 0),
              aruco3cuda::Status::kCudaError);
    const std::string message = aruco3cuda::last_cuda_error_message();
    EXPECT_LT(message.size(), 1024U);
    EXPECT_NE(message.find("api=cudaTestApi"), std::string::npos);
}

// 正常系: stream を指定した場合、その値が文脈へ現れる。
TEST(CudaCheckTest, stream_is_recorded_in_context) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
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

// 正常系: stream を指定した同期確認は device 全体ではなく stream を待つ。
TEST(CudaCheckTest, kernel_launch_check_accepts_stream) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    (void)cudaGetLastError();
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    EXPECT_EQ(aruco3cuda::detail::check_kernel_launch("stream_clean", 0, true, stream),
              aruco3cuda::Status::kOk);
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

}  // namespace
