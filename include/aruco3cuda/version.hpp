// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_VERSION_HPP
#define ARUCO3CUDA_VERSION_HPP

namespace aruco3cuda {

/// Version numbers of the library.
inline constexpr int kVersionMajor = 0;
inline constexpr int kVersionMinor = 1;
inline constexpr int kVersionPatch = 0;

/// Returns the version in "major.minor.patch" form.
///
/// @return A string with static storage duration.
///
/// Ownership: the return value points into static storage. The caller neither frees
///            nor modifies it.
/// Synchronization: host only, with no synchronization point. The first call formats
///                  the result into a static buffer, so the first call must not be
///                  made from several threads at once.
///
/// Example input: no arguments
/// Example output: "0.1.0"
const char* version_string();

}  // namespace aruco3cuda

#endif  // ARUCO3CUDA_VERSION_HPP
