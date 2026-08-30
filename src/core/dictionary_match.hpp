// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CORE_DICTIONARY_MATCH_HPP
#define ARUCO3CUDA_CORE_DICTIONARY_MATCH_HPP

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/dictionary.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/workspace.hpp"
#include "candidate_filter.hpp"
#include "cell_decode.hpp"

namespace aruco3cuda::detail {

/// A dictionary residing on the device.
///
/// The codes_ member of DictionaryTable points into host static storage, which a kernel cannot
/// read as is, so this structure references a device-side copy of the same content.
///
/// Ownership: the workspace owns the region codes_ points to.
/// Synchronization: this is only a reference and holds no synchronization point.
///
/// Example input: the DICT_ARUCO_MIP_36h12 table
/// Example output: marker_size_ = 6, code_count_ = 250, codes_ holding 1000 elements
struct DeviceDictionary {
    /// [code_count_ * 4]. The index is id * 4 + rotation, the same layout as on the host.
    const MarkerCode* codes_ = nullptr;
    int marker_size_ = 0;
    int code_count_ = 0;
    int max_correction_bits_ = 0;
};

/// Per-candidate matching result.
///
/// Ownership: the workspace owns every region these pointers refer to.
/// Synchronization: this is only a bundle of references and holds no synchronization point.
///
/// Example input: candidate cap 4096
/// Example output: ids_ holding 4096 elements
struct MatchBuffers {
    /// The matched ID, or -1 when nothing matched.
    std::int32_t* ids_ = nullptr;
    /// The matched rotation, 0 to 3. 0 when nothing matched.
    std::int32_t* rotations_ = nullptr;
    /// Distance of the selected ID, or the smallest distance across all IDs when nothing
    /// matched. A candidate rejected by the border check is never matched, so it stores the
    /// square of marker_size plus one.
    std::int32_t* distances_ = nullptr;
    int capacity_ = 0;
};

/// Returns the workspace size required by the device-side dictionary.
///
/// @param table Dictionary to size.
/// @return Required number of bytes, or 0 when the table is invalid.
///
/// Ownership: does not retain the regions of the arguments.
/// Synchronization: host only, holds no synchronization point. Calls no CUDA API.
///
/// Example input: the DICT_ARUCO_MIP_36h12 table
/// Example output: 8192
std::size_t device_dictionary_workspace_bytes(const DictionaryTable& table);

/// Allocates the region that holds the dictionary on the device and transfers its content.
///
/// The source is host static storage and therefore pageable. This is meant to be called once
/// during initialization and never appears on the per-frame path.
///
/// @param table Dictionary to upload.
/// @param workspace Workspace in device space.
/// @param out Receives the device-side reference on success. The caller owns the region.
/// @param stream Stream the transfer is issued on.
/// @return kOk, kInvalidArgument when an argument is invalid, kInvalidConfig when the capacity is
///         insufficient, kCudaError when the transfer fails.
///
/// Ownership: codes_ of table is only read as the transfer source and is not retained.
/// Synchronization: the transfer is issued asynchronously on the stream. The content is not final
///           until the caller synchronizes.
///
/// Example input: the DICT_ARUCO_MIP_36h12 table and a workspace with room left
/// Example output: kOk, out->code_count_ = 250
Status upload_dictionary(const DictionaryTable& table, Workspace& workspace, DeviceDictionary* out,
                         cudaStream_t stream);

/// Returns the workspace size required by the matching results.
///
/// @param config Configuration containing the candidate cap.
/// @return Required number of bytes, or 0 when the configuration is invalid.
///
/// Ownership: does not retain the regions of the arguments.
/// Synchronization: host only, holds no synchronization point. Calls no CUDA API.
///
/// Example input: a configuration with max_candidates_ = 4096
/// Example output: 49152
std::size_t match_workspace_bytes(const DetectorConfig& config);

/// Allocates the regions that hold the matching results.
///
/// @param config Configuration containing the candidate cap.
/// @param workspace Workspace in device space.
/// @param out Receives the buffers on success. The caller owns the regions.
/// @return kOk, kInvalidArgument when an argument is invalid, kInvalidConfig when the capacity is
///         insufficient.
///
/// Ownership: does not retain the regions of the arguments.
/// Synchronization: host only, holds no synchronization point.
///
/// Example input: a configuration with max_candidates_ = 4096 and a workspace with room left
/// Example output: kOk, out->capacity_ = 4096
Status reserve_matches(const DetectorConfig& config, Workspace& workspace, MatchBuffers* out);

/// Matches the cell ratios of the candidates against the dictionary.
///
/// The decision follows the same rules as OpenCV's `Dictionary::identify`. The cell ratios are
/// split into two masks, "not black" and "not white", and the IDs are visited in ascending order;
/// the first ID within the allowed distance is taken. That is not the ID with the smallest
/// distance.
///
/// Candidates that failed the border check are not matched at all and receive -1 in ids_.
///
/// @param ratios Cell ratios and the result of the border check.
/// @param candidates Buffer holding the candidate count.
/// @param dictionary Device-side dictionary.
/// @param config Configuration containing the thresholds and the error-correction rate.
/// @param matches Destination of the results.
/// @param stream Stream the kernel is issued on.
/// @return kOk, kInvalidArgument when an argument is invalid, kInvalidConfig when the
///         configuration is inconsistent, kCudaError when the kernel launch fails.
///
/// Ownership: does not retain the regions of the arguments.
/// Synchronization: the kernel is issued asynchronously on the stream. The results are not final
///           until the caller synchronizes.
///
/// Example input: cell ratios for four candidates and DICT_ARUCO_MIP_36h12
/// Example output: kOk, with four IDs or -1 entries in ids_
Status match_candidates_async(const CellRatioBuffers& ratios, const DeviceCandidates& candidates,
                              const DeviceDictionary& dictionary, const DetectorConfig& config,
                              MatchBuffers* matches, cudaStream_t stream);

}  // namespace aruco3cuda::detail

#endif  // ARUCO3CUDA_CORE_DICTIONARY_MATCH_HPP
