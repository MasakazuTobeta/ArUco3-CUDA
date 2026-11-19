// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_APRILTAG_DICTIONARY_HPP
#define ARUCO3CUDA_APRILTAG_DICTIONARY_HPP

#include "aruco3cuda/dictionary.hpp"

namespace aruco3cuda {

/// Look up an AprilTag family by its OpenCV identifier.
/// @param name Null-terminated OpenCV dictionary name, or nullptr.
/// @return The family table, or nullptr for an unknown name.
/// Ownership: the pointer and its codewords have static lifetime.
/// Synchronization: read-only host lookup with no CUDA calls.
/// Example input: "DICT_APRILTAG_36h11"
/// Example output: a 6x6 family table containing 587 codes.
const DictionaryTable* find_apriltag_dictionary(const char* name);

}  // namespace aruco3cuda
#endif  // ARUCO3CUDA_APRILTAG_DICTIONARY_HPP
