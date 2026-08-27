// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_VERSION_HPP
#define ARUCO3CUDA_VERSION_HPP

namespace aruco3cuda {

/// library の version 番号。
inline constexpr int kVersionMajor = 0;
inline constexpr int kVersionMinor = 1;
inline constexpr int kVersionPatch = 0;

/// version を "major.minor.patch" 形式で返す。
///
/// @return 静的記憶域を持つ文字列。
///
/// 所有権: 戻り値は静的記憶域を指す。呼出側は解放も変更もしない。
/// 同期動作: host 専用であり同期点を持たない。初回呼び出しで書式化した結果を
///           静的 buffer へ保持するため、複数 thread から同時に初回呼び出しを
///           行わないこと。
///
/// 入力例: 引数なし
/// 出力例: "0.1.0"
const char* version_string();

}  // namespace aruco3cuda

#endif  // ARUCO3CUDA_VERSION_HPP
