// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CORE_SCAN_HPP
#define ARUCO3CUDA_CORE_SCAN_HPP

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "aruco3cuda/status.hpp"
#include "aruco3cuda/workspace.hpp"

namespace aruco3cuda::detail {

/// Scratch space for the exclusive scan.
///
/// On the GPU, "compact only the elements that satisfy a condition" takes the
/// shape of an exclusive scan over a per-element 1/0 flag that decides the
/// destination. Claiming a slot with atomicAdd instead would order the output by
/// arrival, which makes the result order vary from run to run.
///
/// Ownership: the regions the pointers refer to are owned by the workspace.
/// Synchronization: a plain set of references, so it carries no synchronization point.
///
/// Example input: reserve_scan for an element count of 102480
/// Example output: block_count_ = 401
struct ScanBuffers {
    /// Start position per block.
    std::int32_t* block_offsets_ = nullptr;
    /// Grand total. One element.
    std::int32_t* total_ = nullptr;
    int block_count_ = 0;
    int capacity_ = 0;
};

/// Returns the scan block count for an element count.
///
/// @param count Number of elements to scan.
/// @return Block count. 0 when count is 0 or less.
///
/// Ownership: retains no resource.
/// Synchronization: host only, so it carries no synchronization point.
///
/// Example input: 102480
/// Example output: 401
int scan_block_count(int count);

/// Returns the workspace capacity the scan requires.
///
/// @param count Number of elements to scan.
/// @return Required byte count. 0 when an argument is invalid.
///
/// Ownership: retains no resource.
/// Synchronization: host only, so it carries no synchronization point.
///
/// Example input: 102480
/// Example output: the byte count that holds 401 blocks plus 1 grand total
std::size_t scan_workspace_bytes(int count);

/// Carves the scan regions out of the workspace.
///
/// @param count Number of elements to scan. At least 1.
/// @param workspace Source of the carve-out. Owned by the caller.
/// @param out Receives the full set of buffers on success. Must not be nullptr.
/// @return kOk. kInvalidConfig when the capacity is insufficient,
///         kInvalidArgument when an argument is invalid.
///
/// Ownership: the carved-out regions stay owned by the workspace.
/// Synchronization: host only, so it carries no synchronization point.
///
/// Example input: 102480 and a workspace with sufficient capacity
/// Example output: the pointers are filled in with block_count_ = 401
Status reserve_scan(int count, Workspace& workspace, ScanBuffers* out);

/// Exclusive-scans an array in place.
///
/// The addition is over integers, so splitting the work into blocks leaves the
/// result independent of how it was split. The same input always yields the same
/// output.
///
/// @param values Array to scan. The input is overwritten. Must not be nullptr.
/// @param count Element count. Must not exceed the value passed to reserve_scan.
/// @param buffers The set of buffers returned by reserve_scan. Must not be nullptr.
/// @param stream Stream to issue on. Pass nullptr to use the default stream.
/// @return kOk, or kInvalidArgument, kCudaError.
///
/// Ownership: the regions the arguments refer to stay owned by the caller and by
///            the workspace.
/// Synchronization: only issues kernels on the stream and performs no host
///                  synchronization. The grand total is written to buffers.total_,
///                  and reading it from the host requires synchronization.
///
/// Example input: {1, 0, 1, 1}
/// Example output: {0, 1, 1, 2}, total_ = 3
Status exclusive_scan_async(std::int32_t* values, int count, ScanBuffers* buffers,
                            cudaStream_t stream);

}  // namespace aruco3cuda::detail

#endif  // ARUCO3CUDA_CORE_SCAN_HPP
