// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_DEVICE_PROBE_HPP
#define ARUCO3CUDA_DEVICE_PROBE_HPP

#include <cstddef>
#include <string>

#include "aruco3cuda/status.hpp"

namespace aruco3cuda {

/// device の性質のうち、実装と評価の判断に影響する項目。
///
/// integrated_ は評価計画の memory 経路の扱いに直結する。統合 GPU では
/// host と device が同一物理 memory を共有するため、明示的な copy の費用が
/// discrete GPU と大きく異なる。
struct DeviceProbeResult {
    int device_index_ = -1;
    /// device 名。nvidia-smi へ依存せず取得できる。
    std::string name_;
    int compute_capability_major_ = 0;
    int compute_capability_minor_ = 0;
    int multi_processor_count_ = 0;
    std::size_t l2_cache_bytes_ = 0;
    bool integrated_ = false;                 ///< 統合 GPU か
    bool concurrent_managed_access_ = false;  ///< managed memory へ同時 access できるか
};

/// 利用可能な CUDA device 数を取得する。
///
/// @param out_count 成功時に device 数を格納する。nullptr を渡してはならない。
///                  領域の所有権は呼出側にある。
/// @return kOk、kInvalidArgument、kCudaError のいずれか。
///
/// 同期動作: CUDA runtime を呼ぶが device 同期は行わない。
///           失敗時の詳細は last_cuda_error_message() から取得できる。
///
/// 入力例: 有効な int への pointer
/// 出力例: *out_count = 1
Status device_count(int* out_count);

/// 指定 device の性質を取得する。
///
/// @param device_index 0 以上、device 数未満。
/// @param out 成功時に結果を格納する。失敗時は変更しない。領域の所有権は呼出側にある。
///            戻り値の name_ は複製された文字列であり、CUDA 側の領域を参照しない。
/// @return kOk、kInvalidArgument、kCudaError のいずれか。
///
/// 同期動作: CUDA runtime を呼ぶが device 同期は行わず、current device も変更しない。
///           失敗時の詳細は last_cuda_error_message() から取得できる。
///
/// 入力例: device_index = 0
/// 出力例: compute_capability_major_ = 12, integrated_ = true
Status probe_device(int device_index, DeviceProbeResult* out);

/// 最小の kernel を実行し、device 上で計算が成立することを確認する。
///
/// build 基盤と実行環境の疎通確認を目的とする。
///
/// @param device_index 対象 device。0 以上 device 数未満。
/// @return kOk、kInvalidArgument、kCudaError のいずれか。
///
/// 所有権: 内部で確保する device buffer はこの関数が所有し、
///         成功経路と失敗経路のいずれでも呼出前に解放する。呼出側へは渡らない。
///
/// 同期動作: 呼出内で cudaDeviceSynchronize() を行い、kernel の実行完了まで待つ。
///           current device を一時的に device_index へ変更するが、呼出前の値へ戻す。
///
/// 入力例: device_index = 0
/// 出力例: Status::kOk
Status run_device_self_test(int device_index);

}  // namespace aruco3cuda

#endif  // ARUCO3CUDA_DEVICE_PROBE_HPP
