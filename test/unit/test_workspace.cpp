// SPDX-License-Identifier: Apache-2.0
//
// workspace の確保方針を検証する。
//
// 最も重要なのは「フレームごとの確保が発生しない」ことである。規約は
// フレームごとの cudaMalloc と cudaFree を避けることを求めるが、守れているかは
// 統計を見なければ分からない。定常状態で allocation_count_ が増えないことを
// 明示的に確認する。
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

/// 1 フレーム分の切り出しを模した処理。段階ごとの buffer を順に取る。
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

// 正常系: align_up が境界へ切り上げる。
TEST(AlignUpTest, rounds_up_to_alignment) {
    EXPECT_EQ(aruco3cuda::align_up(0U, 256U), 0U);
    EXPECT_EQ(aruco3cuda::align_up(1U, 256U), 256U);
    EXPECT_EQ(aruco3cuda::align_up(256U, 256U), 256U);
    EXPECT_EQ(aruco3cuda::align_up(257U, 256U), 512U);
    // alignment が 0 なら値をそのまま返す。
    EXPECT_EQ(aruco3cuda::align_up(100U, 0U), 100U);
}

// 境界値: 切り上げで桁溢れする場合は 0 を返す。
// 呼出側は容量不足として扱う。wrap した値で範囲計算を続けると危険である。
TEST(AlignUpTest, returns_zero_on_overflow) {
    const std::size_t near_max = std::numeric_limits<std::size_t>::max() - 10U;
    EXPECT_EQ(aruco3cuda::align_up(near_max, 256U), 0U);
}

// 異常系: 未確保の workspace からは切り出せない。
TEST(WorkspaceTest, allocate_fails_before_capacity_is_reserved) {
    Workspace workspace;
    void* buffer = nullptr;
    EXPECT_EQ(workspace.allocate(16U, 256U, &buffer), Status::kInvalidConfig);
    EXPECT_EQ(workspace.statistics().exhausted_count_, 1U);
    EXPECT_EQ(workspace.statistics().allocation_count_, 0U);
}

// 異常系: 不正な引数を拒否する。
TEST(WorkspaceTest, rejects_invalid_arguments) {
    Workspace workspace;
    void* buffer = nullptr;
    EXPECT_EQ(workspace.allocate(16U, 256U, nullptr), Status::kInvalidArgument);
    // alignment は 2 の冪である必要がある。
    EXPECT_EQ(workspace.allocate(16U, 3U, &buffer), Status::kInvalidArgument);
    EXPECT_EQ(workspace.allocate(16U, 0U, &buffer), Status::kInvalidArgument);
    // pageable host memory は arena として扱わない。
    std::string message;
    EXPECT_EQ(workspace.ensure_capacity(1024U, MemorySpace::kHostPageable, &message),
              Status::kInvalidArgument);
    EXPECT_FALSE(message.empty());
}

// 境界値: 0 byte の要求は nullptr を返して成功する。
TEST(WorkspaceTest, zero_sized_requests_are_no_ops) {
    Workspace workspace;
    EXPECT_EQ(workspace.ensure_capacity(0U, MemorySpace::kDevice, nullptr), Status::kOk);
    EXPECT_EQ(workspace.statistics().allocation_count_, 0U);

    void* buffer = reinterpret_cast<void*>(0x1234);
    EXPECT_EQ(workspace.allocate(0U, 256U, &buffer), Status::kOk);
    EXPECT_EQ(buffer, nullptr);
}

// 正常系: 容量を確保し、境界の揃った領域を切り出せる。
TEST(WorkspaceTest, allocates_aligned_buffers) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
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
    // 2 つ目は 1 つ目と重ならない。
    EXPECT_GE(reinterpret_cast<std::uintptr_t>(second) - reinterpret_cast<std::uintptr_t>(first),
              100U);
}

// 正常系: 定常状態でフレームごとの確保が発生しない。
// WP-1.2 の完了条件そのものである。
TEST(WorkspaceTest, steady_state_does_not_allocate_per_frame) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    Workspace workspace;
    ASSERT_EQ(workspace.ensure_capacity(1U << 20, MemorySpace::kDevice, nullptr), Status::kOk);
    const std::size_t after_reserve = workspace.statistics().allocation_count_;
    ASSERT_EQ(after_reserve, 1U);

    for (int frame = 0; frame < 100; ++frame) {
        ASSERT_EQ(simulate_frame(workspace), Status::kOk) << "frame=" << frame;
        // 確保も再確保も増えない。
        ASSERT_EQ(workspace.statistics().allocation_count_, after_reserve) << "frame=" << frame;
        ASSERT_EQ(workspace.statistics().reallocation_count_, 0U) << "frame=" << frame;
        ASSERT_EQ(workspace.statistics().exhausted_count_, 0U) << "frame=" << frame;
    }
    // 使用量の最大値は 1 フレーム分に収まる。reset() が効いている証拠になる。
    EXPECT_LT(workspace.statistics().peak_used_bytes_, 1U << 20);
    EXPECT_EQ(workspace.statistics().used_bytes_, workspace.statistics().peak_used_bytes_);
}

// 正常系: reset() で切り出し位置が戻る。
TEST(WorkspaceTest, reset_rewinds_offset) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
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
    // reset() 後は同じ位置から切り出される。
    EXPECT_EQ(first, again);
}

// 境界値: 容量を超える切り出しは失敗し、自動で拡張しない。
// 自動拡張はフレームごとの確保を静かに招く。
TEST(WorkspaceTest, allocate_does_not_grow_capacity) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    Workspace workspace;
    ASSERT_EQ(workspace.ensure_capacity(4096U, MemorySpace::kDevice, nullptr), Status::kOk);
    const std::size_t before = workspace.statistics().allocation_count_;

    void* buffer = nullptr;
    // 容量ちょうどは通る。
    ASSERT_EQ(workspace.allocate(4096U, 256U, &buffer), Status::kOk);
    // 1 byte でも超えると失敗する。
    EXPECT_EQ(workspace.allocate(1U, 256U, &buffer), Status::kInvalidConfig);
    EXPECT_EQ(workspace.statistics().allocation_count_, before);
    EXPECT_EQ(workspace.statistics().exhausted_count_, 1U);
}

// 正常系: 容量が足りていれば確保し直さない。
TEST(WorkspaceTest, ensure_capacity_is_idempotent_when_sufficient) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    Workspace workspace;
    ASSERT_EQ(workspace.ensure_capacity(1U << 16, MemorySpace::kDevice, nullptr), Status::kOk);
    ASSERT_EQ(workspace.ensure_capacity(1U << 15, MemorySpace::kDevice, nullptr), Status::kOk);
    ASSERT_EQ(workspace.ensure_capacity(1U << 16, MemorySpace::kDevice, nullptr), Status::kOk);
    EXPECT_EQ(workspace.statistics().allocation_count_, 1U);
    EXPECT_EQ(workspace.statistics().reallocation_count_, 0U);
}

// 正常系: 容量が足りなければ確保し直し、再確保として数える。
TEST(WorkspaceTest, ensure_capacity_reallocates_when_insufficient) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    Workspace workspace;
    ASSERT_EQ(workspace.ensure_capacity(4096U, MemorySpace::kDevice, nullptr), Status::kOk);
    ASSERT_EQ(workspace.ensure_capacity(1U << 20, MemorySpace::kDevice, nullptr), Status::kOk);
    EXPECT_EQ(workspace.statistics().allocation_count_, 2U);
    EXPECT_EQ(workspace.statistics().reallocation_count_, 1U);
    EXPECT_GE(workspace.statistics().capacity_bytes_, 1U << 20);
    // 再確保で切り出し位置は先頭へ戻る。
    EXPECT_EQ(workspace.statistics().used_bytes_, 0U);
}

// 正常系: memory 空間を変えると確保し直す。
TEST(WorkspaceTest, changing_memory_space_reallocates) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    Workspace workspace;
    ASSERT_EQ(workspace.ensure_capacity(4096U, MemorySpace::kDevice, nullptr), Status::kOk);
    EXPECT_EQ(workspace.space(), MemorySpace::kDevice);
    ASSERT_EQ(workspace.ensure_capacity(4096U, MemorySpace::kHostPinned, nullptr), Status::kOk);
    EXPECT_EQ(workspace.space(), MemorySpace::kHostPinned);
    EXPECT_EQ(workspace.statistics().reallocation_count_, 1U);
}

// 正常系: pinned と managed の空間でも確保できる。
// 評価計画は memory 種別を独立した測定軸として扱う。
TEST(WorkspaceTest, supports_pinned_and_managed_spaces) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    for (const MemorySpace space : {MemorySpace::kHostPinned, MemorySpace::kManaged}) {
        Workspace workspace;
        std::string message;
        ASSERT_EQ(workspace.ensure_capacity(4096U, space, &message), Status::kOk)
                << aruco3cuda::to_string(space) << ": " << message;
        void* buffer = nullptr;
        ASSERT_EQ(workspace.allocate(1024U, 256U, &buffer), Status::kOk);
        ASSERT_NE(buffer, nullptr);
        // host から見える空間では書き込めることまで確認する。
        if (space != MemorySpace::kDevice) {
            auto* bytes = static_cast<std::uint8_t*>(buffer);
            bytes[0] = 42U;
            EXPECT_EQ(bytes[0], 42U);
        }
    }
}

// 正常系: 移動しても資源が二重に解放されない。
TEST(WorkspaceTest, move_transfers_ownership) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    Workspace source;
    ASSERT_EQ(source.ensure_capacity(4096U, MemorySpace::kDevice, nullptr), Status::kOk);
    void* buffer = nullptr;
    ASSERT_EQ(source.allocate(256U, 256U, &buffer), Status::kOk);

    Workspace moved(std::move(source));
    EXPECT_EQ(moved.statistics().allocation_count_, 1U);
    EXPECT_GE(moved.statistics().capacity_bytes_, 4096U);
    // 移動元は資源を手放しており、切り出せない。
    // move 後の状態を意図的に確認する。move constructor が移動元を
    // 明示的に空へ戻していることが、二重解放を防ぐ根拠になる。
    void* from_source = nullptr;
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_EQ(source.allocate(16U, 256U, &from_source), Status::kInvalidConfig);

    Workspace assigned;
    assigned = std::move(moved);
    EXPECT_GE(assigned.statistics().capacity_bytes_, 4096U);
    void* from_assigned = nullptr;
    EXPECT_EQ(assigned.allocate(256U, 256U, &from_assigned), Status::kOk);
}

// 正常系: release() で容量が解放され、統計へ反映される。
TEST(WorkspaceTest, release_frees_capacity) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    Workspace workspace;
    ASSERT_EQ(workspace.ensure_capacity(4096U, MemorySpace::kDevice, nullptr), Status::kOk);
    workspace.release();
    EXPECT_EQ(workspace.statistics().capacity_bytes_, 0U);
    void* buffer = nullptr;
    EXPECT_EQ(workspace.allocate(16U, 256U, &buffer), Status::kInvalidConfig);
    // 二重の release でも問題ない。
    workspace.release();
}

}  // namespace
