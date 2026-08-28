// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_TEST_DETECTOR_PROBE_HPP
#define ARUCO3CUDA_TEST_DETECTOR_PROBE_HPP

#include <cuda_runtime_api.h>

namespace aruco3cuda::test {

/// 指定した cycle 数だけ device を占有する kernel を stream へ積む。
///
/// @param cycles 回る cycle 数。device の clock に依存するため、時間を厳密に
///               決めることはできない。呼出側は十分に大きい値を渡す。
/// @param device_sink 最適化で消されないための書き込み先。device pointer。
/// @param stream 積む stream。
/// @return 起動できたら true。
///
/// 所有権: 引数の領域を保持しない。
/// 同期動作: 発行するだけで同期しない。
///
/// 入力例: cycles = 1000000000、有効な device pointer
/// 出力例: true
bool enqueue_spin(long long cycles, int* device_sink, cudaStream_t stream);

}  // namespace aruco3cuda::test

#endif  // ARUCO3CUDA_TEST_DETECTOR_PROBE_HPP
