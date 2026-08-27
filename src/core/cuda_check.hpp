// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CORE_CUDA_CHECK_HPP
#define ARUCO3CUDA_CORE_CUDA_CHECK_HPP

#include <cuda_runtime_api.h>

#include "aruco3cuda/status.hpp"

namespace aruco3cuda::detail {

/// CUDA API の戻り値を検査し、失敗時に文脈付きで記録する。
///
/// 失敗を無言で継続させないため、全ての CUDA API 呼び出しをこの関数へ通す。
/// 記録内容は last_cuda_error_message() から取得できる。
///
/// MISRA C++ 2023 は function-like macro を避ける方針であるため、
/// 呼出位置の情報は macro ではなく明示的な引数として渡す。
///
/// @param error CUDA API の戻り値。
/// @param api_name 呼び出した CUDA API 名。静的記憶域を持つ文字列を渡す。
/// @param stage 処理段階。どの kernel または手順で失敗したかを示す。
/// @param device_index 対象 device。不明な場合は -1 を渡す。
/// @return error が cudaSuccess なら Status::kOk、それ以外は Status::kCudaError。
Status check_cuda(cudaError_t error, const char* api_name, const char* stage, int device_index);

/// 直近の kernel 起動の失敗を検査する。
///
/// kernel 起動は非同期であるため、起動自体の失敗と実行中の失敗を分けて確認する。
/// synchronize が true の場合は同期して実行中の失敗も検出する。
///
/// @param stage 処理段階。
/// @param device_index 対象 device。
/// @param synchronize 同期して実行時エラーまで検出するか。
/// @return kOk または kCudaError。
Status check_kernel_launch(const char* stage, int device_index, bool synchronize);

}  // namespace aruco3cuda::detail

#endif  // ARUCO3CUDA_CORE_CUDA_CHECK_HPP
