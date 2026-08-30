// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CORE_LABELING_HPP
#define ARUCO3CUDA_CORE_LABELING_HPP

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "aruco3cuda/status.hpp"
#include "aruco3cuda/workspace.hpp"
#include "preprocess.hpp"
#include "scan.hpp"

namespace aruco3cuda::detail {

/// Label stored in background pixels.
///
/// Foreground labels are consecutive numbers starting at 0, so the background is
/// given a negative value. Using 0 for the background would force the foreground
/// numbering to start at 1, putting the label and the index into the statistics
/// arrays off by one. That kind of skew invites mix-ups, so it is avoided.
constexpr std::int32_t kBackgroundLabel = -1;

/// Output and scratch space of connected-component labeling.
///
/// Ownership: the regions all of the pointers refer to are owned by the
///            workspace. This struct holds references only; it neither copies nor
///            frees. Every pointer becomes invalid once the workspace is reset()
///            or destroyed.
/// Synchronization: a plain set of references, so it carries no synchronization
///                  point. The contents are not settled until the already-issued
///                  kernels complete.
///
/// Example input: a 427x240 binary image
/// Example output: labels_ holds 102480 elements, and the label count lands in
///                 label_count_
struct LabelBuffers {
    /// Label per pixel. The element count is width_px_ * height_px_.
    ///
    /// Carries no pitch; the rows are packed one after another. Union-find walks
    /// pixels by linear index, and a pitch would push the index-to-address
    /// conversion into the inner loop. In labeling, the linear traversal dominates
    /// over 2D neighbor reads, so the conversion cost outweighs the benefit of a
    /// pitch.
    std::int32_t* labels_ = nullptr;
    /// Mapping from a root's linear index to the compacted label. Same element
    /// count as labels_.
    ///
    /// Doubles as the scan's input and output. Writing a 1/0 flag marking whether
    /// a pixel is a root and then applying the exclusive scan yields the compacted
    /// label directly.
    std::int32_t* compact_ids_ = nullptr;
    /// Scratch space for the exclusive scan that derives the compacted labels.
    ScanBuffers scan_;
    /// Device-side label count. One element. Points at the same region as the
    /// scan's grand total.
    std::int32_t* label_count_ = nullptr;
    int width_px_ = 0;
    int height_px_ = 0;
};

/// Returns the required workspace capacity.
///
/// @param width_px Width of the binary image.
/// @param height_px Height of the binary image.
/// @return Required byte count. 0 on overflow or an invalid argument.
///
/// Ownership: retains no resource.
/// Synchronization: host only, so it carries no synchronization point.
///
/// Example input: 427 and 240
/// Example output: the byte count that holds 2 label arrays and the scan scratch
///                 space
std::size_t labeling_workspace_bytes(int width_px, int height_px);

/// Carves the labeling regions out of the workspace.
///
/// @param width_px Width of the binary image. At least 1.
/// @param height_px Height of the binary image. At least 1.
/// @param workspace Source of the carve-out. Owned by the caller.
/// @param out Receives the full set of buffers on success. Must not be nullptr.
/// @return kOk. kInvalidConfig when the capacity is insufficient,
///         kInvalidArgument when an argument is invalid.
///
/// Ownership: the carved-out regions stay owned by the workspace.
/// Synchronization: host only, so it carries no synchronization point.
///
/// Example input: 427, 240, and a workspace with sufficient capacity
/// Example output: labels_ and compact_ids_ receive pointers
Status reserve_labeling(int width_px, int height_px, Workspace& workspace, LabelBuffers* out);

/// Splits the foreground of a binary image into 8-connected components.
///
/// 8-connectivity is used because OpenCV `findContours` traverses the foreground
/// as 8-connected. With 4-connectivity, foreground regions that touch only
/// diagonally would fall into separate components, splitting a shape the CPU
/// reference treats as one candidate into two.
///
/// Labels are consecutive numbers starting at 0, assigned in ascending order of
/// the root's linear index. This is what makes the same input yield the same
/// labels on every run; numbering that depends on the arrival order of atomics is
/// not used.
///
/// @param binary Input binary image. 0 is treated as background and anything else
///               as foreground. Must have the same size as passed to
///               reserve_labeling.
/// @param buffers The set of buffers returned by reserve_labeling. Must not be
///                nullptr.
/// @param stream Stream to issue on. Pass nullptr to use the default stream.
/// @return kOk, or kInvalidArgument, kCudaError.
///
/// Ownership: the regions buffers points at stay owned by the workspace.
/// Synchronization: only issues kernels on the stream and performs no host
///                  synchronization.
///
/// Example input: a 427x240 binary image with a single square at its center
/// Example output: the square's pixels carry label 0 and the rest kBackgroundLabel
Status build_labels_async(const ImagePlaneU8& binary, LabelBuffers* buffers, cudaStream_t stream);

/// Reads the label count back to the host.
///
/// @param buffers The set of buffers passed to build_labels_async.
/// @param out_count Receives the label count on success. Must not be nullptr.
/// @param stream Stream to issue on. Waits for the transfer to complete.
/// @return kOk, or kInvalidArgument, kCudaError.
///
/// Ownership: retains none of the regions passed as arguments.
/// Synchronization: waits for the stream to complete. out_count is settled by the
///                  time the call returns.
///
/// Example input: the buffers after processing an image with 4 squares
/// Example output: out_count = 4
Status read_label_count(const LabelBuffers& buffers, int* out_count, cudaStream_t stream);

/// Per-label statistics, held as a structure of arrays rather than an array of
/// structures.
///
/// The aggregation uses per-label atomics, so many threads converge on the same
/// field. Keeping a separate array per field prevents one atomic from dragging in
/// the cache line of a neighboring field.
///
/// Ownership: the regions all of the pointers refer to are owned by the workspace.
/// Synchronization: a plain set of references, so it carries no synchronization
///                  point. The contents are not settled until the already-issued
///                  kernels complete.
///
/// Example input: reserve_label_stats for a 427x240 image
/// Example output: capacity_ = 214 * 120 = 25680
struct LabelStatisticsBuffers {
    /// Bounding box. Both min and max are inclusive.
    std::int32_t* min_x_ = nullptr;
    std::int32_t* min_y_ = nullptr;
    std::int32_t* max_x_ = nullptr;
    std::int32_t* max_y_ = nullptr;
    /// Number of pixels belonging to the component.
    std::int32_t* pixel_count_ = nullptr;
    /// Sum of the coordinates. Kept in order to derive the centroid.
    ///
    /// The type is unsigned long long to match the atomicAdd overload. Under LP64,
    /// std::uint64_t is unsigned long, and the pointer types would not match.
    unsigned long long* sum_x_ = nullptr;
    unsigned long long* sum_y_ = nullptr;
    /// Centroid. The sums divided by the pixel count.
    float* centroid_x_ = nullptr;
    float* centroid_y_ = nullptr;
    /// Number of labels allocated for. Equal to max_label_count().
    int capacity_ = 0;
};

/// Returns the upper bound on the label count for the given image size.
///
/// Under 8-connectivity, distinct components must not touch horizontally,
/// vertically, or diagonally. The arrangement that produces the most components
/// places single pixels every other position, so the bound is
/// ceil(W/2) * ceil(H/2). Allocating for that value makes a label overflow
/// impossible, so the statistics side needs no overflow path.
///
/// @param width_px Image width.
/// @param height_px Image height.
/// @return Upper bound on the label count. 0 when an argument is invalid.
///
/// Ownership: retains no resource.
/// Synchronization: host only, so it carries no synchronization point.
///
/// Example input: 427 and 240
/// Example output: 214 * 120 = 25680
int max_label_count(int width_px, int height_px);

/// Returns the workspace capacity the statistics require.
///
/// @param width_px Image width.
/// @param height_px Image height.
/// @return Required byte count. 0 on overflow or an invalid argument.
///
/// Ownership: retains no resource.
/// Synchronization: host only, so it carries no synchronization point.
///
/// Example input: 427 and 240
/// Example output: the byte count that holds the statistics for 25680 labels
std::size_t label_stats_workspace_bytes(int width_px, int height_px);

/// Carves the statistics regions out of the workspace.
///
/// @param width_px Image width. At least 1.
/// @param height_px Image height. At least 1.
/// @param workspace Source of the carve-out. Owned by the caller.
/// @param out Receives the full set of buffers on success. Must not be nullptr.
/// @return kOk. kInvalidConfig when the capacity is insufficient,
///         kInvalidArgument when an argument is invalid.
///
/// Ownership: the carved-out regions stay owned by the workspace.
/// Synchronization: host only, so it carries no synchronization point.
///
/// Example input: 427, 240, and a workspace with sufficient capacity
/// Example output: capacity_ = 25680 and every array receives a pointer
Status reserve_label_stats(int width_px, int height_px, Workspace& workspace,
                           LabelStatisticsBuffers* out);

/// Aggregates the bounding box, pixel count, and centroid per label.
///
/// The aggregation uses integer atomics only. Changing the order of the additions
/// does not change the result, so every run yields the same values. The centroid
/// is computed exactly once, when the sums are divided by the pixel count, so no
/// difference arises from the order of the divisions either.
///
/// @param labels The set of labels filled in by build_labels_async.
/// @param stats The set of buffers returned by reserve_label_stats. Must not be
///              nullptr.
/// @param stream Stream to issue on. Pass nullptr to use the default stream.
/// @return kOk, or kInvalidArgument, kCudaError.
///
/// Ownership: the regions the arguments refer to stay owned by the workspace.
/// Synchronization: only issues kernels on the stream and performs no host
///                  synchronization.
///
/// Example input: a label image whose central 30x30 square carries label 0
/// Example output: pixel_count_[0] = 900, the bounding box covers the square, and
///                 the centroid sits at its center
Status build_label_stats_async(const LabelBuffers& labels, LabelStatisticsBuffers* stats,
                               cudaStream_t stream);

}  // namespace aruco3cuda::detail

#endif  // ARUCO3CUDA_CORE_LABELING_HPP
