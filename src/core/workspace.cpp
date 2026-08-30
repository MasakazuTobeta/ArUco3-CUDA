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

/// Allocates memory in the requested memory space.
///
/// The cost differs per space even on an integrated GPU, so the same arena implementation can
/// switch the space without any other change.
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
            // Pageable memory has no CUDA allocation API, so the arena does not handle it.
            return Status::kInvalidArgument;
    }
    return Status::kInvalidArgument;
}

/// Performs the release that pairs with allocate_space.
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
    // Return 0 when rounding up would overflow. The caller treats that as insufficient capacity.
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
    // Make the moved-from object drop the resource it held so its destructor frees nothing.
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
            *out_message = "workspace does not handle pageable host memory";
        }
        return Status::kInvalidArgument;
    }
    // Do not reallocate when the space matches and the capacity already suffices. Reallocating
    // here would turn into a per-frame allocation, which the conventions require us to avoid.
    if (this->base_ != nullptr && this->space_ == space && this->capacity_bytes_ >= bytes) {
        return Status::kOk;
    }

    void* next = nullptr;
    const Status allocate_status = allocate_space(space, bytes, &next);
    if (allocate_status != Status::kOk) {
        if (out_message != nullptr) {
            *out_message = std::string("workspace allocation failed: ") + std::to_string(bytes) +
                           " bytes, " + to_string(space) + ", " + last_cuda_error_message();
        }
        return allocate_status;
    }

    // Release the old region only after the new one has been allocated. Releasing first would
    // also lose the previous capacity whenever the allocation fails.
    const bool had_previous = this->base_ != nullptr;
    if (had_previous) {
        const Status free_status = free_space(this->space_, this->base_);
        if (free_status != Status::kOk && out_message != nullptr) {
            *out_message = std::string("releasing the previous workspace failed: ") +
                           last_cuda_error_message();
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
    // align_up returns 0 only on overflow, which is treated as insufficient capacity.
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
        // Called from the destructor, so it must not throw. A release failure survives as the
        // recorded CUDA error and is not reported through a return value.
        (void)free_space(this->space_, this->base_);
        this->base_ = nullptr;
    }
    this->capacity_bytes_ = 0U;
    this->offset_bytes_ = 0U;
    this->statistics_.capacity_bytes_ = 0U;
    this->statistics_.used_bytes_ = 0U;
}

}  // namespace aruco3cuda
