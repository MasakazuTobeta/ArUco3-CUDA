// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_WORKSPACE_HPP
#define ARUCO3CUDA_WORKSPACE_HPP

#include <cstddef>
#include <string>

#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"

namespace aruco3cuda {

/// How the workspace is being used.
///
/// Purpose:
///   Lets the caller and the tests confirm that no allocation happens per frame. The
///   coding rules require avoiding `cudaMalloc` and `cudaFree` per frame, but whether
///   that is actually being honored can only be seen from the statistics.
struct WorkspaceStatistics {
    /// Number of times device memory was actually allocated. It stops growing in
    /// steady state.
    std::size_t allocation_count_ = 0;
    /// Number of times memory was reallocated because the capacity was too small.
    /// Included in allocation_count_.
    std::size_t reallocation_count_ = 0;
    /// Current capacity, in bytes.
    std::size_t capacity_bytes_ = 0;
    /// Amount currently carved out, in bytes.
    std::size_t used_bytes_ = 0;
    /// Largest amount ever carved out, in bytes.
    std::size_t peak_used_bytes_ = 0;
    /// Number of times a carve-out failed for lack of capacity.
    std::size_t exhausted_count_ = 0;
};

/// An arena holding all the intermediate buffers of the detection pipeline.
///
/// Purpose:
///   Allocating separately for each stage lines up a `cudaMalloc` and a `cudaFree` per
///   frame, and their cost leaks into the measurements. Instead, one large region is
///   allocated up front and the per-stage buffers are carved out of it.
///
/// Design:
///   Carve-outs use a bump pointer, and no individual free is offered. reset() is called
///   at the start of a frame to rewind the carve-out position. The lifetime of the
///   intermediate buffers ends within the frame, so this is sufficient.
///
///   allocate() does not grow the arena automatically when the capacity runs short.
///   Automatic growth would introduce per-frame allocation and would quietly create the
///   very state the coding rules tell us to avoid. Capacity is instead reserved with
///   ensure_capacity() when the detector is initialized.
///
/// Ownership:
///   This class owns the device memory it allocates and frees it in the destructor. The
///   pointers allocate() returns point inside the arena and are not freed by the caller.
///   They become invalid once the arena is reset() or destroyed.
///
/// Synchronization:
///   ensure_capacity() and release() call `cudaMalloc` and `cudaFree`. Those carry an
///   implicit device synchronization, so the design calls for not invoking them during
///   detection. The other member functions do host-side computation only and hold no
///   synchronization point.
///   **This ownership and synchronization applies to all public member functions.**
///   A single instance must not be used from several threads at once.
///
/// Example input:
///   Workspace workspace;
///   workspace.ensure_capacity(1 << 20, MemorySpace::kDevice);
///   void* buffer = nullptr;
///   workspace.allocate(4096, 256, &buffer);
/// Example output:
///   statistics().allocation_count_ == 1 and used_bytes_ == 4096
class Workspace {
public:
    Workspace() = default;
    ~Workspace();

    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;
    Workspace(Workspace&& other) noexcept;
    Workspace& operator=(Workspace&& other) noexcept;

    /// Reserves the requested capacity. Does nothing if it is already sufficient.
    ///
    /// If it is not, the memory is reallocated. Every existing carve-out becomes
    /// invalid, so call this at initialization time rather than during detection.
    ///
    /// @param bytes The capacity required. Passing 0 does nothing and returns kOk.
    /// @param space The memory space to allocate in. If it differs from the space of the
    ///              existing capacity, the memory is reallocated.
    /// @param out_message On failure, receives the reason. May be nullptr.
    /// @return kOk. kCudaError if the allocation failed.
    ///
    /// Example input: bytes = 1048576, space = MemorySpace::kDevice
    /// Example output: Status::kOk; statistics().capacity_bytes_ becomes at least 1048576
    Status ensure_capacity(std::size_t bytes, MemorySpace space,
                           std::string* out_message = nullptr);

    /// Carves a region out of the arena.
    ///
    /// @param bytes The number of bytes required. Passing 0 sets *out to nullptr and
    ///              returns kOk.
    /// @param alignment The alignment boundary. It must be a power of two.
    /// @param out Receives the start of the carved-out region. Must not be nullptr.
    ///            Ownership stays with the arena; the caller does not free it.
    /// @return kOk. kInvalidArgument if out is nullptr or alignment is not a power of
    ///         two. kInvalidConfig if the capacity is insufficient: the capacity is
    ///         derived from the configuration, so running short means the configuration
    ///         is wrong.
    ///
    /// Synchronization: host-side computation only, with no synchronization point.
    ///                  Calls no CUDA API.
    ///
    /// Example input: bytes = 4096, alignment = 256
    /// Example output: Status::kOk; *out becomes a pointer aligned to 256
    Status allocate(std::size_t bytes, std::size_t alignment, void** out);

    /// Rewinds the carve-out position to the start, keeping the reserved capacity.
    ///
    /// Call this at the start of a frame. Every pointer allocate() has returned so far
    /// becomes invalid. The capacity is not released, so no allocation occurs.
    ///
    /// @return Nothing.
    ///
    /// Example input: reset() after allocate(1024, 256, &p)
    /// Example output: statistics().used_bytes_ goes back to 0 and the next allocate
    ///                 returns the same position
    void reset();

    /// Releases the reserved capacity.
    ///
    /// Call this when the detector is destroyed; the destructor calls it as well. It is
    /// safe to call twice. Since it is called from the destructor, it throws no
    /// exception and does not report a failed free through a return value. A failure
    /// remains available as a recorded CUDA error.
    ///
    /// @return Nothing.
    ///
    /// Example input: release() after ensure_capacity(4096, MemorySpace::kDevice)
    /// Example output: statistics().capacity_bytes_ becomes 0 and allocate fails
    void release();

    /// Returns the usage statistics.
    ///
    /// @return A reference to the statistics. Ownership stays with the workspace, and
    ///         the reference is valid until the next operation.
    ///
    /// Example input: after repeating reset and allocate for 100 frames
    /// Example output: allocation_count_ still 1 and reallocation_count_ 0
    const WorkspaceStatistics& statistics() const { return this->statistics_; }

    /// Returns the current memory space.
    ///
    /// @return The space given to the most recent ensure_capacity. The value is
    ///         meaningless if nothing has been allocated.
    ///
    /// Example input: after ensure_capacity(4096, MemorySpace::kHostPinned)
    /// Example output: MemorySpace::kHostPinned
    MemorySpace space() const { return this->space_; }

private:
    void* base_ = nullptr;
    std::size_t capacity_bytes_ = 0;
    std::size_t offset_bytes_ = 0;
    MemorySpace space_ = MemorySpace::kDevice;
    WorkspaceStatistics statistics_;
};

/// Returns the value rounded up to an alignment boundary.
///
/// @param value The value to round up.
/// @param alignment The alignment boundary. It must be a power of two. Passing 0
///                  returns value unchanged.
/// @return The smallest multiple of alignment that is at least value. Returns 0 on
///         overflow.
///
/// Ownership: holds no resource.
/// Synchronization: host only, with no synchronization point.
///
/// Example input: value = 100, alignment = 256
/// Example output: 256
std::size_t align_up(std::size_t value, std::size_t alignment);

}  // namespace aruco3cuda

#endif  // ARUCO3CUDA_WORKSPACE_HPP
