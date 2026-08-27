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
/// @return 静的記憶域を持つ文字列。呼出側は解放しない。
const char* version_string();

}  // namespace aruco3cuda

#endif  // ARUCO3CUDA_VERSION_HPP
