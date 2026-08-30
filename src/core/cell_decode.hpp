// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CORE_CELL_DECODE_HPP
#define ARUCO3CUDA_CORE_CELL_DECODE_HPP

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/workspace.hpp"
#include "candidate_filter.hpp"
#include "cell_sample.hpp"

namespace aruco3cuda::detail {

/// Per-candidate cell ratios and the result of the border check.
///
/// Ownership: the workspace owns every region these pointers refer to.
/// Synchronization: this is only a bundle of references and holds no synchronization point. The
///           contents are not final until the already-issued kernels complete.
///
/// Example input: default configuration, marker_size = 6, candidate cap 4096
/// Example output: cells_per_side_ = 8, ratios_ holding 4096 * 8 * 8 elements, thresholds_ holding
///           4096 elements
struct CellRatioBuffers {
    /// White-pixel ratio per cell. The index is (candidate * cells * cells) + (row * cells) + col.
    ///
    /// Kept as a ratio rather than collapsed into a bit. The default denominator is 16, and a
    /// ratio of 0.5 matches neither bit 0 nor bit 1; collapsing into a bit matrix would change the
    /// result relative to the CPU reference.
    float* ratios_ = nullptr;
    /// Number of erroneous border cells.
    std::int32_t* border_errors_ = nullptr;
    /// Whether the border check passed. 1 means it passed.
    std::uint8_t* accepted_ = nullptr;
    /// Threshold chosen by Otsu. Candidates that took the low-variance path store 0.
    ///
    /// Comparing only the ratios would hide a threshold that is off by one level whenever no
    /// pixel sits on the boundary, so the threshold itself is exposed for comparison against the
    /// CPU reference.
    std::int32_t* thresholds_ = nullptr;
    /// Number of cells along one side. Equal to marker_size + 2 * marker_border_bits_.
    int cells_per_side_ = 0;
    int capacity_ = 0;
};

/// Returns the number of cells along one side.
///
/// @param config Detector configuration.
/// @param marker_size Cell count of the dictionary. At least 1.
/// @return Number of cells along one side, or 0 when an argument is invalid.
///
/// Ownership: holds no resources.
/// Synchronization: host only, holds no synchronization point.
///
/// Example input: default configuration and marker_size = 6
/// Example output: 8
int cells_per_side(const DetectorConfig& config, int marker_size);

/// Returns the workspace capacity required by the cell ratios.
///
/// @param config Detector configuration. The candidate cap is used.
/// @param marker_size Cell count of the dictionary.
/// @return Required number of bytes, or 0 on overflow or when an argument is invalid.
///
/// Ownership: holds no resources.
/// Synchronization: host only, holds no synchronization point.
///
/// Example input: default configuration and marker_size = 6
/// Example output: the number of bytes that hold the ratios, the error counts and the verdicts
std::size_t cell_ratio_workspace_bytes(const DetectorConfig& config, int marker_size);

/// Carves the cell-ratio regions out of the workspace.
///
/// @param config Detector configuration. The candidate cap is used.
/// @param marker_size Cell count of the dictionary. At least 1.
/// @param workspace Source of the allocation. Owned by the caller.
/// @param out Receives the buffers on success. Must not be nullptr.
/// @return kOk, kInvalidConfig when the capacity is insufficient, kInvalidArgument when an
///         argument is invalid.
///
/// Ownership: the workspace keeps ownership of the carved-out regions.
/// Synchronization: host only, holds no synchronization point.
///
/// Example input: default configuration, marker_size = 6 and a workspace with enough capacity
/// Example output: cells_per_side_ = 8, capacity_ = 4096
Status reserve_cell_ratios(const DetectorConfig& config, int marker_size, Workspace& workspace,
                           CellRatioBuffers* out);

/// Computes the cell ratios from the canonical images and counts the erroneous border cells.
///
/// Corresponds to `_extractCellPixelRatio` and `_getBorderErrors` in the CPU reference. The steps
/// below all follow the OpenCV implementation.
///
/// 1. Compute the mean and the population standard deviation over the interior of the canonical
///    image (the range obtained by pulling each side in by half a cell). The sum and the sum of
///    squares are accumulated as integers and only the final division is done in double
///    precision. The mean is `S * (1/N)`, not `S / N`.
/// 2. When the standard deviation falls below `min_otsu_std_dev_`, fill every cell ratio with 1
///    or 0 depending on whether the mean exceeds 127. The border check still runs in this case.
/// 3. Otherwise apply Otsu to the **whole** canonical image, not to its interior. Binarization is
///    `pixel > threshold`, so an equal value falls on the 0 side.
/// 4. Compute the white-pixel ratio per cell. The cell margin is `(int)(rate * cell)`, which is 0
///    under the default configuration. The denominator is the square of the side length with the
///    margin removed.
/// 5. Count the border cells whose ratio exceeds `valid_bit_threshold_`. The candidate fails once
///    that count exceeds the square of `marker_size` multiplied by the rate.
///
/// Multiply-add contraction is disabled: this translation unit is compiled with `-fmad=false`.
/// Contraction shifts the variance by 1 ULP, which flips the verdict near the standard-deviation
/// threshold.
///
/// @param canonical Canonical images filled in by build_canonical_async.
/// @param candidates Compacted candidates. The candidate count is read on the device.
/// @param config Detector configuration.
/// @param marker_size Cell count of the dictionary. At least 1.
/// @param ratios Output regions returned by reserve_cell_ratios. Must not be nullptr.
/// @param stream Stream to issue the work on. Pass nullptr to use the default stream.
/// @return kOk, kInvalidArgument or kCudaError.
///
/// Ownership: the workspace keeps ownership of the regions the arguments point to.
/// Synchronization: only issues kernels on the stream and performs no host synchronization.
///
/// Example input: canonical images for four markers
/// Example output: 8x8 ratios in ratios_ and four entries of 1 in accepted_
Status build_cell_ratios_async(const CanonicalBuffers& canonical,
                               const DeviceCandidates& candidates, const DetectorConfig& config,
                               int marker_size, CellRatioBuffers* ratios, cudaStream_t stream);

}  // namespace aruco3cuda::detail

#endif  // ARUCO3CUDA_CORE_CELL_DECODE_HPP
