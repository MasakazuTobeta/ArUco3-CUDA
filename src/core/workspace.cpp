// SPDX-License-Identifier: Apache-2.0
#include "aruco3cuda/workspace.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"
#include "cuda_check.hpp"

namespace aruco3cuda {
namespace {

/// memory 空間に対応する確保を行う。
///
/// 統合 GPU でも空間ごとに費用が異なるため、同じ arena の実装で
/// 空間だけを切り替えられるようにする。
Status allocate_space(MemorySpace space, std::size_t bytes, void** out) {
    switch (space) {
        case MemorySpace::kDevice:
            return detail::check_cuda(cudaMalloc(out, bytes), "cudaMalloc",
                                      "workspace.ensure_capacity", -1);
        case MemorySpace::kHostPinned:
            return detail::check_cuda(cudaMallocHost(out, bytes), "cudaMallocHost",
                                      "workspace.ensure_capacity", -1);
        case MemorySpace::kManaged:
            return detail::check_cuda(cudaMallocManaged(out, bytes), "cudaMallocManaged",
                                      "workspace.ensure_capacity", -1);
        case MemorySpace::kHostPageable:
            // pageable は CUDA の確保 API を持たない。arena としては扱わない。
            return Status::kInvalidArgument;
    }
    return Status::kInvalidArgument;
}

/// allocate_space と対になる解放を行う。
Status free_space(MemorySpace space, void* pointer) {
    if (pointer == nullptr) {
        return Status::kOk;
    }
    switch (space) {
        case MemorySpace::kDevice:
        case MemorySpace::kManaged:
            return detail::check_cuda(cudaFree(pointer), "cudaFree", "workspace.release", -1);
        case MemorySpace::kHostPinned:
            return detail::check_cuda(cudaFreeHost(pointer), "cudaFreeHost", "workspace.release",
                                      -1);
        case MemorySpace::kHostPageable:
            return Status::kInvalidArgument;
    }
    return Status::kInvalidArgument;
}

bool is_power_of_two(std::size_t value) {
    return value != 0U && (value & (value - 1U)) == 0U;
}

}  // namespace

std::size_t align_up(std::size_t value, std::size_t alignment) {
    if (alignment == 0U) {
        return value;
    }
    const std::size_t remainder = value % alignment;
    if (remainder == 0U) {
        return value;
    }
    const std::size_t padding = alignment - remainder;
    // 切り上げで桁溢れする場合は 0 を返す。呼出側は容量不足として扱う。
    if (value > std::numeric_limits<std::size_t>::max() - padding) {
        return 0U;
    }
    return value + padding;
}

Workspace::~Workspace() {
    this->release();
}

Workspace::Workspace(Workspace&& other) noexcept
    : base_(other.base_),
      capacity_bytes_(other.capacity_bytes_),
      offset_bytes_(other.offset_bytes_),
      space_(other.space_),
      statistics_(other.statistics_) {
    // 移動元が destructor で解放しないよう、保持している資源を手放させる。
    other.base_ = nullptr;
    other.capacity_bytes_ = 0U;
    other.offset_bytes_ = 0U;
    other.statistics_ = WorkspaceStatistics();
}

Workspace& Workspace::operator=(Workspace&& other) noexcept {
    if (this != &other) {
        this->release();
        this->base_ = other.base_;
        this->capacity_bytes_ = other.capacity_bytes_;
        this->offset_bytes_ = other.offset_bytes_;
        this->space_ = other.space_;
        this->statistics_ = other.statistics_;
        other.base_ = nullptr;
        other.capacity_bytes_ = 0U;
        other.offset_bytes_ = 0U;
        other.statistics_ = WorkspaceStatistics();
    }
    return *this;
}

Status Workspace::ensure_capacity(std::size_t bytes, MemorySpace space, std::string* out_message) {
    if (bytes == 0U) {
        return Status::kOk;
    }
    if (space == MemorySpace::kHostPageable) {
        if (out_message != nullptr) {
            *out_message = "workspace は pageable host memory を扱わない";
        }
        return Status::kInvalidArgument;
    }
    // 同じ空間で容量が足りていれば確保し直さない。ここで確保し直すと
    // フレームごとの確保になり、規約が避けよと定める状態になる。
    if (this->base_ != nullptr && this->space_ == space && this->capacity_bytes_ >= bytes) {
        return Status::kOk;
    }

    void* next = nullptr;
    const Status allocate_status = allocate_space(space, bytes, &next);
    if (allocate_status != Status::kOk) {
        if (out_message != nullptr) {
            *out_message = std::string("workspace の確保に失敗した: ") + std::to_string(bytes) +
                           " byte, " + to_string(space) + ", " + last_cuda_error_message();
        }
        return allocate_status;
    }

    // 新しい領域の確保が成功してから古い領域を解放する。先に解放すると、
    // 確保に失敗した場合に以前の容量まで失う。
    const bool had_previous = this->base_ != nullptr;
    if (had_previous) {
        const Status free_status = free_space(this->space_, this->base_);
        if (free_status != Status::kOk && out_message != nullptr) {
            *out_message =
                    std::string("以前の workspace の解放に失敗した: ") + last_cuda_error_message();
        }
        ++this->statistics_.reallocation_count_;
    }

    this->base_ = next;
    this->capacity_bytes_ = bytes;
    this->offset_bytes_ = 0U;
    this->space_ = space;
    ++this->statistics_.allocation_count_;
    this->statistics_.capacity_bytes_ = bytes;
    this->statistics_.used_bytes_ = 0U;
    return Status::kOk;
}

Status Workspace::allocate(std::size_t bytes, std::size_t alignment, void** out) {
    if (out == nullptr || !is_power_of_two(alignment)) {
        return Status::kInvalidArgument;
    }
    if (bytes == 0U) {
        *out = nullptr;
        return Status::kOk;
    }
    const std::size_t aligned_offset = align_up(this->offset_bytes_, alignment);
    // align_up が 0 を返すのは桁溢れした場合であり、容量不足として扱う。
    if (aligned_offset == 0U && this->offset_bytes_ != 0U) {
        ++this->statistics_.exhausted_count_;
        return Status::kInvalidConfig;
    }
    if (bytes > this->capacity_bytes_ || aligned_offset > this->capacity_bytes_ - bytes) {
        ++this->statistics_.exhausted_count_;
        return Status::kInvalidConfig;
    }

    *out = static_cast<void*>(static_cast<std::uint8_t*>(this->base_) + aligned_offset);
    this->offset_bytes_ = aligned_offset + bytes;
    this->statistics_.used_bytes_ = this->offset_bytes_;
    if (this->offset_bytes_ > this->statistics_.peak_used_bytes_) {
        this->statistics_.peak_used_bytes_ = this->offset_bytes_;
    }
    return Status::kOk;
}

void Workspace::reset() {
    this->offset_bytes_ = 0U;
    this->statistics_.used_bytes_ = 0U;
}

void Workspace::release() {
    if (this->base_ != nullptr) {
        // destructor から呼ばれるため例外を送出しない。解放の失敗は
        // 記録済みの CUDA エラーとして残り、戻り値では通知しない。
        (void)free_space(this->space_, this->base_);
        this->base_ = nullptr;
    }
    this->capacity_bytes_ = 0U;
    this->offset_bytes_ = 0U;
    this->statistics_.capacity_bytes_ = 0U;
    this->statistics_.used_bytes_ = 0U;
}

}  // namespace aruco3cuda
