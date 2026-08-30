// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_DICTIONARY_HPP
#define ARUCO3CUDA_DICTIONARY_HPP

#include <cstddef>
#include <cstdint>

#include "aruco3cuda/status.hpp"

namespace aruco3cuda {

/// Type holding the bit pattern of one marker.
///
/// The largest supported marker size is 7x7 = 49 bits, which fits in 64 bits.
/// Bit 0 is (row 0, col 0) and the rest follow in row-major order. The unused high
/// bits are 0.
using MarkerCode = std::uint64_t;

/// Packed representation of a predefined dictionary.
///
/// Purpose:
///   Holds the codewords in a form that can be transferred as-is from CUDA into
///   constant memory or global memory, and that allows matching all four rotations
///   without branching.
///
/// Ownership:
///   codes_ points into static storage. This struct owns nothing.
struct DictionaryTable {
    const char* name_ = nullptr;
    int marker_size_ = 0;          ///< Bits per side; 6 for a 6x6 marker
    int max_correction_bits_ = 0;  ///< maxCorrectionBits of OpenCV
    int code_count_ = 0;           ///< Number of IDs in the table
    /// [code_count_ * 4]. The index is id * 4 + rotation.
    /// Rotation advances counterclockwise in 90 degree steps.
    const MarkerCode* codes_ = nullptr;

    /// Returns the number of bits in the table.
    ///
    /// @return marker_size_ * marker_size_, or 0 if marker_size_ is 0.
    ///
    /// Ownership: holds no resource.
    /// Synchronization: a pure computation callable from both host and device, with no
    ///                  synchronization point.
    ///
    /// Example input: a table with marker_size_ = 6
    /// Example output: 36
    int bit_count() const { return this->marker_size_ * this->marker_size_; }
};

/// Result of a dictionary match.
struct DictionaryMatch {
    int id_ = -1;
    int rotation_ = 0;  ///< 0 through 3, matching the rotation index of codes_
    int distance_ = 0;  ///< Smallest Hamming distance
};

/// The two bit masks built from the cell ratios.
///
/// Purpose:
///   A cell ratio is not necessarily 0 or 1. Rather than a single bit string, OpenCV
///   represents a candidate with two masks, "not black" and "not white". A ratio near
///   the threshold sets a bit in both masks, so it counts as an error whether the
///   expected bit is 0 or 1. Collapsing this into a single bit string would lose that
///   distinction.
///
/// Ownership:
///   Holds values only.
/// Synchronization:
///   Holds values only and has no synchronization point. It may be handed to the device
///   as-is.
///
/// Example input: threshold 0.49 with the 2x2 ratios {1.0, 0.0, 0.5, 1.0}
/// Example output: not_black_ = 0b1101, not_white_ = 0b0110
struct CellMasks {
    /// Cells whose ratio exceeded the threshold. Bit 0 is (row 0, col 0), row-major.
    MarkerCode not_black_ = 0;
    /// Cells whose ratio fell below one minus the threshold. Laid out like not_black_.
    MarkerCode not_white_ = 0;
};

/// Returns the number of predefined dictionaries included.
///
/// @return The count. Never 0.
///
/// Ownership: holds no resource.
/// Synchronization: host only, with no synchronization point.
///
/// Example input: no arguments
/// Example output: 1
std::size_t builtin_dictionary_count();

/// Returns the predefined dictionary at the given index.
///
/// @param index At least 0 and less than builtin_dictionary_count(). Out of range
///              returns nullptr.
/// @return A pointer to a table with static storage duration, or nullptr if out of range.
///
/// Ownership: the return value points into static storage. The caller neither frees nor
///            modifies it. The codes_ the table points at is likewise static storage.
/// Synchronization: host only, with no synchronization point.
///
/// Example input: 0
/// Example output: a table whose name_ is "DICT_ARUCO_MIP_36h12"
const DictionaryTable* builtin_dictionary_at(std::size_t index);

/// Looks up a predefined dictionary by name.
///
/// @param name The name to look for. May be nullptr, in which case nullptr is returned.
/// @return A pointer to a table with static storage duration, or nullptr if not found.
///
/// Ownership: the return value points into static storage. The caller neither frees nor
///            modifies it. The name argument is neither copied nor retained.
/// Synchronization: host only, with no synchronization point.
///
/// Example input: "DICT_ARUCO_MIP_36h12"
/// Example output: a table with marker_size_ = 6 and code_count_ = 250
const DictionaryTable* find_builtin_dictionary(const char* name);

/// Matches a candidate bit string against a dictionary.
///
/// Finds the smallest Hamming distance over all IDs and all four rotations, and calls it
/// a match if that distance is at most max_correction_errors.
///
/// @param table The dictionary to search.
/// @param candidate The candidate bit string. Bits beyond the table's bit_count() must
///                  be 0.
/// @param max_correction_errors The largest Hamming distance accepted.
/// @param out Receives the match result. If there is no match, id_ is set to -1. The
///            caller owns the storage.
/// @return kOk if the match could be performed. kInvalidArgument if an argument is
///         invalid.
///
/// Ownership: the codes_ the table points at is only read, never retained.
/// Synchronization: host only, with no synchronization point. The CUDA implementation
///                  reimplements the same rule as a device-side function.
///
/// Note:
///   Finding no match is not a failure, so it is not distinguished by the return value.
///   The caller decides from whether out->id_ is -1.
///
/// Example input: a 6x6 dictionary and candidate = the exact codeword of ID 42
/// Example output: out->id_ = 42, out->rotation_ = 0, out->distance_ = 0
/// Example input: a candidate far from every codeword
/// Example output: out->id_ = -1
Status match_dictionary(const DictionaryTable& table, MarkerCode candidate,
                        int max_correction_errors, DictionaryMatch* out);

/// Builds CellMasks from the cell ratios.
///
/// Sorts the ratios into the two masks by the same rule as `CellBitMasks` in OpenCV: a
/// ratio above the threshold sets a bit in not_black_, and a ratio below one minus the
/// threshold sets a bit in not_white_. With the default threshold of 0.49, a ratio
/// greater than 0.49 and less than 0.51 sets a bit in both, so it is an error whether
/// the expected bit is 0 or 1.
///
/// @param ratios The cell ratios. The length is at least marker_size * marker_size,
///               row-major. The caller owns the memory, and this function neither copies
///               nor retains it.
/// @param marker_size Cells per side. At least 1 and at most 7. The border is not
///                    included.
/// @param valid_bit_threshold The threshold above which a ratio counts as a bit.
///                            Corresponds to `validBitIdThreshold` in OpenCV.
/// @param out On success, receives the masks. The caller owns the storage.
/// @return kOk. kInvalidArgument if ratios or out is nullptr, or if marker_size is out
///         of range.
///
/// Ownership: does not retain the memory behind the arguments.
/// Synchronization: host only, with no synchronization point.
///
/// Example input: ratios = {1.0, 0.0, 0.0, 1.0}, marker_size = 2, threshold 0.49
/// Example output: out->not_black_ = 0b1001, out->not_white_ = 0b0110
Status build_cell_masks(const float* ratios, int marker_size, float valid_bit_threshold,
                        CellMasks* out);

/// Identifies a candidate by the same rule as `Dictionary::identify` in OpenCV.
///
/// There are two differences from match_dictionary. First, the ambiguity of a ratio is
/// handled with two masks. Second, instead of finding the smallest distance over all
/// IDs, it walks the IDs in ascending order and **stops at the first ID that meets the
/// accepted distance**. For a dictionary whose minimum distance between entries is more
/// than twice the accepted distance the two agree, but otherwise the results can differ.
/// This one is used on the detection path because matching OpenCV takes priority.
///
/// The accepted distance is `int(max_correction_bits_ * error_correction_rate)`. The
/// fraction is truncated toward zero.
///
/// @param table The dictionary to search.
/// @param masks The masks built from the candidate's cell ratios.
/// @param error_correction_rate `errorCorrectionRate` of OpenCV. At least 0 and at most 1.
/// @param out Receives the match result. If there is no match, id_ is set to -1 and
///            distance_ receives the smallest distance found across all IDs. The caller
///            owns the storage.
/// @return kOk if the match could be performed. kInvalidArgument if an argument is
///         invalid.
///
/// Ownership: the codes_ the table points at is only read, never retained.
/// Synchronization: host only, with no synchronization point. The CUDA implementation
///                  reimplements the same rule on the device side.
///
/// Example input: a 6x6 dictionary and masks built from the ratios of the exact codeword
///                of ID 42
/// Example output: out->id_ = 42, out->rotation_ = 0, out->distance_ = 0
/// Example input: masks far from every codeword
/// Example output: out->id_ = -1
Status identify_marker(const DictionaryTable& table, const CellMasks& masks,
                       double error_correction_rate, DictionaryMatch* out);

/// Computes the minimum Hamming distance within a dictionary.
///
/// Finds the smallest distance, rotations included, over every pair of IDs. Used to
/// confirm agreement with the nominal value. For a dictionary with a large code_count
/// this is O(n^2), so it is meant for use in tests.
///
/// @param table The dictionary to examine.
/// @param out_distance On success, receives the minimum distance. The caller owns the
///                     storage.
/// @return kOk or kInvalidArgument.
///
/// Ownership: the codes_ the table points at is only read, never retained.
/// Synchronization: host only, with no synchronization point.
///
/// Example input: the DICT_ARUCO_MIP_36h12 table
/// Example output: *out_distance = 12
Status minimum_hamming_distance(const DictionaryTable& table, int* out_distance);

/// Builds a MarkerCode from a bit array.
///
/// @param bits Each element is 0 or 1. The length is at least marker_size * marker_size.
///             The caller owns the memory, and this function neither copies nor retains it.
/// @param marker_size Bits per side. At least 1 and at most 7.
/// @param out On success, receives the packed value. The caller owns the storage.
/// @return kOk. kInvalidArgument if bits or out is nullptr, or if marker_size is out of
///         range.
///
/// Ownership: does not retain the memory behind the arguments.
/// Synchronization: host only, with no synchronization point.
///
/// Example input: bits = {1, 0, 1, 0}, marker_size = 2
/// Example output: *out = 0b0101
Status pack_marker_code(const std::uint8_t* bits, int marker_size, MarkerCode* out);

/// Expands a MarkerCode into a bit array; the inverse of pack_marker_code.
///
/// @param code The value to expand.
/// @param marker_size Bits per side. At least 1 and at most 7.
/// @param out_bits The destination. It must hold at least marker_size * marker_size
///                 elements, and allocating and freeing it is the caller's
///                 responsibility. Each element is written as 0 or 1.
/// @return kOk. kInvalidArgument if out_bits is nullptr, or if marker_size is out of
///         range.
///
/// Ownership: does not retain the memory behind the arguments.
/// Synchronization: host only, with no synchronization point.
///
/// Example input: code = 0b101, marker_size = 2
/// Example output: out_bits = {1, 0, 1, 0}
Status unpack_marker_code(MarkerCode code, int marker_size, std::uint8_t* out_bits);

/// Returns the MarkerCode of the bit array rotated 90 degrees counterclockwise.
///
/// @param code The original value.
/// @param marker_size Bits per side. At least 1 and at most 7.
/// @param out On success, receives the rotated value. The caller owns the storage.
/// @return kOk. kInvalidArgument if out is nullptr, or if marker_size is out of range.
///
/// Ownership: does not retain the memory behind the arguments.
/// Synchronization: host only, with no synchronization point.
///
/// Example input: the 2x2 pattern {1, 0, 0, 0}, marker_size = 2
/// Example output: {0, 0, 1, 0}, rotated 90 degrees counterclockwise
Status rotate_marker_code(MarkerCode code, int marker_size, MarkerCode* out);

}  // namespace aruco3cuda

#endif  // ARUCO3CUDA_DICTIONARY_HPP
