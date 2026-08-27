// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_DEVICE_PROBE_HPP
#define ARUCO3CUDA_DEVICE_PROBE_HPP

#include <cstddef>

#include "aruco3cuda/status.hpp"

namespace aruco3cuda {

/// device の性質のうち、実装と評価の判断に影響する項目。
///
/// integrated_ は評価計画の memory 経路の扱いに直結する。統合 GPU では
/// host と device が同一物理 memory を共有するため、明示的な copy の費用が
/// discrete GPU と大きく異なる。
struct DeviceProbeResult {
    int device_index_ = -1;
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
/// @return kOk、kInvalidArgument、kCudaError のいずれか。
///
/// 入力例: 有効な int への pointer
/// 出力例: *out_count = 1
Status device_count(int* out_count);

/// 指定 device の性質を取得する。
///
/// @param device_index 0 以上、device 数未満。
/// @param out 成功時に結果を格納する。失敗時は変更しない。
/// @return kOk、kInvalidArgument、kCudaError のいずれか。
///
/// 入力例: device_index = 0
/// 出力例: compute_capability_major_ = 12, integrated_ = true
Status probe_device(int device_index, DeviceProbeResult* out);

/// 最小の kernel を実行し、device 上で計算が成立することを確認する。
///
/// build 基盤と実行環境の疎通確認を目的とする。呼出内で同期する。
///
/// @param device_index 対象 device。
/// @return kOk、kInvalidArgument、kCudaError のいずれか。
Status run_device_self_test(int device_index);

}  // namespace aruco3cuda

#endif  // ARUCO3CUDA_DEVICE_PROBE_HPP
