// SPDX-License-Identifier: Apache-2.0
//
// Verifies the allocation policy of the workspace.
//
// The most important property is that no allocation happens per frame. The
// conventions require avoiding a cudaMalloc and cudaFree on every frame, but
// whether that is actually being honored can only be seen from the statistics.
// This test explicitly confirms that allocation_count_ does not grow in the
// steady state.
#include "aruco3cuda/workspace.hpp"

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"

namespace {

using aruco3cuda::MemorySpace;
using aruco3cuda::Status;
using aruco3cuda::Workspace;

bool has_cuda_device() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

/// Mimics the sub-allocations of a single frame, taking the buffer for each stage in turn.
Status simulate_frame(Workspace& workspace) {
    workspace.reset();
    const std::size_t sizes[] = {4096U, 8192U, 1024U, 65536U};
    for (const std::size_t size : sizes) {
        void* buffer = nullptr;
        const Status status = workspace.allocate(size, 256U, &buffer);
        if (status != Status::kOk) {
            return status;
        }
        if (buffer == nullptr) {
            return Status::kInvalidArgument;
        }
    }
    return Status::kOk;
}

// Nominal: align_up rounds up to the alignment boundary.
TEST(AlignUpTest, rounds_up_to_alignment) {
    EXPECT_EQ(aruco3cuda::align_up(0U, 256U), 0U);
    EXPECT_EQ(aruco3cuda::align_up(1U, 256U), 256U);
    EXPECT_EQ(aruco3cuda::align_up(256U, 256U), 256U);
    EXPECT_EQ(aruco3cuda::align_up(257U, 256U), 512U);
    // An alignment of 0 returns the value unchanged.
    EXPECT_EQ(aruco3cuda::align_up(100U, 0U), 100U);
}

// Boundary: rounding up returns 0 when it would overflow.
// Callers treat that as insufficient capacity. Continuing range arithmetic with a
// wrapped value would be dangerous.
TEST(AlignUpTest, returns_zero_on_overflow) {
    const std::size_t near_max = std::numeric_limits<std::size_t>::max() - 10U;
    EXPECT_EQ(aruco3cuda::align_up(near_max, 256U), 0U);
}

// Error case: nothing can be carved out of a workspace with no reserved capacity.
TEST(WorkspaceTest, allocate_fails_before_capacity_is_reserved) {
    Workspace workspace;
    void* buffer = nullptr;
    EXPECT_EQ(workspace.allocate(16U, 256U, &buffer), Status::kInvalidConfig);
    EXPECT_EQ(workspace.statistics().exhausted_count_, 1U);
    EXPECT_EQ(workspace.statistics().allocation_count_, 0U);
}

// Error case: invalid arguments are rejected.
TEST(WorkspaceTest, rejects_invalid_arguments) {
    Workspace workspace;
    void* buffer = nullptr;
    EXPECT_EQ(workspace.allocate(16U, 256U, nullptr), Status::kInvalidArgument);
    // The alignment must be a power of two.
    EXPECT_EQ(workspace.allocate(16U, 3U, &buffer), Status::kInvalidArgument);
    EXPECT_EQ(workspace.allocate(16U, 0U, &buffer), Status::kInvalidArgument);
    // Pageable host memory is not accepted as an arena.
    std::string message;
    EXPECT_EQ(workspace.ensure_capacity(1024U, MemorySpace::kHostPageable, &message),
              Status::kInvalidArgument);
    EXPECT_FALSE(message.empty());
}

// Boundary: a zero-byte request succeeds and yields nullptr.
TEST(WorkspaceTest, zero_sized_requests_are_no_ops) {
    Workspace workspace;
    EXPECT_EQ(workspace.ensure_capacity(0U, MemorySpace::kDevice, nullptr), Status::kOk);
    EXPECT_EQ(workspace.statistics().allocation_count_, 0U);

    void* buffer = reinterpret_cast<void*>(0x1234);
    EXPECT_EQ(workspace.allocate(0U, 256U, &buffer), Status::kOk);
    EXPECT_EQ(buffer, nullptr);
}

// Nominal: capacity can be reserved and aligned regions carved out of it.
TEST(WorkspaceTest, allocates_aligned_buffers) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipped: no CUDA device is available in this environment";
    }
    Workspace workspace;
    std::string message;
    ASSERT_EQ(workspace.ensure_capacity(1U << 20, MemorySpace::kDevice, &message), Status::kOk)
            << message;
    EXPECT_EQ(workspace.statistics().allocation_count_, 1U);
    EXPECT_GE(workspace.statistics().capacity_bytes_, 1U << 20);

    void* first = nullptr;
    void* second = nullptr;
    ASSERT_EQ(workspace.allocate(100U, 256U, &first), Status::kOk);
    ASSERT_EQ(workspace.allocate(100U, 256U, &second), Status::kOk);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(first) % 256U, 0U);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(second) % 256U, 0U);
    // The second region does not overlap the first.
    EXPECT_GE(reinterpret_cast<std::uintptr_t>(second) - reinterpret_cast<std::uintptr_t>(first),
              100U);
}

// Nominal: no per-frame allocation happens in the steady state.
// This is the single most important property the workspace has to satisfy.
TEST(WorkspaceTest, steady_state_does_not_allocate_per_frame) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipped: no CUDA device is available in this environment";
    }
    Workspace workspace;
    ASSERT_EQ(workspace.ensure_capacity(1U << 20, MemorySpace::kDevice, nullptr), Status::kOk);
    const std::size_t after_reserve = workspace.statistics().allocation_count_;
    ASSERT_EQ(after_reserve, 1U);

    for (int frame = 0; frame < 100; ++frame) {
        ASSERT_EQ(simulate_frame(workspace), Status::kOk) << "frame=" << frame;
        // Neither allocations nor reallocations increase.
        ASSERT_EQ(workspace.statistics().allocation_count_, after_reserve) << "frame=" << frame;
        ASSERT_EQ(workspace.statistics().reallocation_count_, 0U) << "frame=" << frame;
        ASSERT_EQ(workspace.statistics().exhausted_count_, 0U) << "frame=" << frame;
    }
    // Peak usage stays within a single frame's worth, which is the evidence that
    // reset() is taking effect.
    EXPECT_LT(workspace.statistics().peak_used_bytes_, 1U << 20);
    EXPECT_EQ(workspace.statistics().used_bytes_, workspace.statistics().peak_used_bytes_);
}

// Nominal: reset() rewinds the sub-allocation offset.
TEST(WorkspaceTest, reset_rewinds_offset) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipped: no CUDA device is available in this environment";
    }
    Workspace workspace;
    ASSERT_EQ(workspace.ensure_capacity(1U << 16, MemorySpace::kDevice, nullptr), Status::kOk);

    void* first = nullptr;
    ASSERT_EQ(workspace.allocate(1024U, 256U, &first), Status::kOk);
    EXPECT_EQ(workspace.statistics().used_bytes_, 1024U);

    workspace.reset();
    EXPECT_EQ(workspace.statistics().used_bytes_, 0U);

    void* again = nullptr;
    ASSERT_EQ(workspace.allocate(1024U, 256U, &again), Status::kOk);
    // After reset() the next region starts at the same address as before.
    EXPECT_EQ(first, again);
}

// Boundary: a request beyond the capacity fails and does not grow the arena.
// Growing automatically would quietly reintroduce per-frame allocation.
TEST(WorkspaceTest, allocate_does_not_grow_capacity) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipped: no CUDA device is available in this environment";
    }
    Workspace workspace;
    ASSERT_EQ(workspace.ensure_capacity(4096U, MemorySpace::kDevice, nullptr), Status::kOk);
    const std::size_t before = workspace.statistics().allocation_count_;

    void* buffer = nullptr;
    // A request for exactly the capacity succeeds.
    ASSERT_EQ(workspace.allocate(4096U, 256U, &buffer), Status::kOk);
    // Exceeding it by even one byte fails.
    EXPECT_EQ(workspace.allocate(1U, 256U, &buffer), Status::kInvalidConfig);
    EXPECT_EQ(workspace.statistics().allocation_count_, before);
    EXPECT_EQ(workspace.statistics().exhausted_count_, 1U);
}

// Nominal: no reallocation happens while the existing capacity suffices.
TEST(WorkspaceTest, ensure_capacity_is_idempotent_when_sufficient) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipped: no CUDA device is available in this environment";
    }
    Workspace workspace;
    ASSERT_EQ(workspace.ensure_capacity(1U << 16, MemorySpace::kDevice, nullptr), Status::kOk);
    ASSERT_EQ(workspace.ensure_capacity(1U << 15, MemorySpace::kDevice, nullptr), Status::kOk);
    ASSERT_EQ(workspace.ensure_capacity(1U << 16, MemorySpace::kDevice, nullptr), Status::kOk);
    EXPECT_EQ(workspace.statistics().allocation_count_, 1U);
    EXPECT_EQ(workspace.statistics().reallocation_count_, 0U);
}

// Nominal: insufficient capacity triggers a reallocation, counted as such.
TEST(WorkspaceTest, ensure_capacity_reallocates_when_insufficient) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipped: no CUDA device is available in this environment";
    }
    Workspace workspace;
    ASSERT_EQ(workspace.ensure_capacity(4096U, MemorySpace::kDevice, nullptr), Status::kOk);
    ASSERT_EQ(workspace.ensure_capacity(1U << 20, MemorySpace::kDevice, nullptr), Status::kOk);
    EXPECT_EQ(workspace.statistics().allocation_count_, 2U);
    EXPECT_EQ(workspace.statistics().reallocation_count_, 1U);
    EXPECT_GE(workspace.statistics().capacity_bytes_, 1U << 20);
    // A reallocation rewinds the sub-allocation offset to the start.
    EXPECT_EQ(workspace.statistics().used_bytes_, 0U);
}

// Nominal: changing the memory space forces a reallocation.
TEST(WorkspaceTest, changing_memory_space_reallocates) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipped: no CUDA device is available in this environment";
    }
    Workspace workspace;
    ASSERT_EQ(workspace.ensure_capacity(4096U, MemorySpace::kDevice, nullptr), Status::kOk);
    EXPECT_EQ(workspace.space(), MemorySpace::kDevice);
    ASSERT_EQ(workspace.ensure_capacity(4096U, MemorySpace::kHostPinned, nullptr), Status::kOk);
    EXPECT_EQ(workspace.space(), MemorySpace::kHostPinned);
    EXPECT_EQ(workspace.statistics().reallocation_count_, 1U);
}

// Nominal: capacity can also be reserved in the pinned and managed spaces.
// The evaluation plan treats the memory kind as an independent measurement axis.
TEST(WorkspaceTest, supports_pinned_and_managed_spaces) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipped: no CUDA device is available in this environment";
    }
    for (const MemorySpace space : {MemorySpace::kHostPinned, MemorySpace::kManaged}) {
        Workspace workspace;
        std::string message;
        ASSERT_EQ(workspace.ensure_capacity(4096U, space, &message), Status::kOk)
                << aruco3cuda::to_string(space) << ": " << message;
        void* buffer = nullptr;
        ASSERT_EQ(workspace.allocate(1024U, 256U, &buffer), Status::kOk);
        ASSERT_NE(buffer, nullptr);
        // In spaces visible from the host, confirm that the memory is writable too.
        if (space != MemorySpace::kDevice) {
            auto* bytes = static_cast<std::uint8_t*>(buffer);
            bytes[0] = 42U;
            EXPECT_EQ(bytes[0], 42U);
        }
    }
}

// Nominal: moving a workspace does not free its resources twice.
TEST(WorkspaceTest, move_transfers_ownership) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipped: no CUDA device is available in this environment";
    }
    Workspace source;
    ASSERT_EQ(source.ensure_capacity(4096U, MemorySpace::kDevice, nullptr), Status::kOk);
    void* buffer = nullptr;
    ASSERT_EQ(source.allocate(256U, 256U, &buffer), Status::kOk);

    Workspace moved(std::move(source));
    EXPECT_EQ(moved.statistics().allocation_count_, 1U);
    EXPECT_GE(moved.statistics().capacity_bytes_, 4096U);
    // The moved-from workspace has given up its resources and can allocate nothing.
    // Inspecting the moved-from state is deliberate: the move constructor explicitly
    // leaving the source empty is what rules out a double free.
    void* from_source = nullptr;
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_EQ(source.allocate(16U, 256U, &from_source), Status::kInvalidConfig);

    Workspace assigned;
    assigned = std::move(moved);
    EXPECT_GE(assigned.statistics().capacity_bytes_, 4096U);
    void* from_assigned = nullptr;
    EXPECT_EQ(assigned.allocate(256U, 256U, &from_assigned), Status::kOk);
}

// Nominal: release() frees the capacity and the statistics reflect it.
TEST(WorkspaceTest, release_frees_capacity) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipped: no CUDA device is available in this environment";
    }
    Workspace workspace;
    ASSERT_EQ(workspace.ensure_capacity(4096U, MemorySpace::kDevice, nullptr), Status::kOk);
    workspace.release();
    EXPECT_EQ(workspace.statistics().capacity_bytes_, 0U);
    void* buffer = nullptr;
    EXPECT_EQ(workspace.allocate(16U, 256U, &buffer), Status::kInvalidConfig);
    // Releasing twice is harmless.
    workspace.release();
}

}  // namespace
