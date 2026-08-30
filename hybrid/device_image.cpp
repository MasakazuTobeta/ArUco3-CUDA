// SPDX-License-Identifier: Apache-2.0
#include "device_image.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"

namespace aruco3cuda::hybrid {
namespace {

/// Stores the failure reason in out_message. Does nothing if it is nullptr.
void set_message(std::string* out_message, const std::string& text) {
    if (out_message != nullptr) {
        *out_message = text;
    }
}

}  // namespace

/// The implementation itself. Kept behind a pimpl so the public header does
/// not depend on CUDA.
class DeviceImage::Impl {
public:
    Impl() = default;
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
    ~Impl() { this->release(); }

    Status reserve(MemorySpace space, int width_px, int height_px, std::string* out_message) {
        if (width_px <= 0 || height_px <= 0) {
            set_message(out_message, "image dimensions must be at least 1");
            return Status::kInvalidArgument;
        }
        if (this->data_ != nullptr && space == this->space_ &&
            width_px <= this->capacity_width_px_ && height_px <= this->capacity_height_px_) {
            // The existing region is large enough. Avoid a per-frame allocation.
            this->view_.width_px_ = width_px;
            this->view_.height_px_ = height_px;
            return Status::kOk;
        }
        this->release();
        this->space_ = space;

        std::size_t pitch = 0;
        cudaError_t error = cudaSuccess;
        if (space == MemorySpace::kManaged) {
            // For managed memory there is no staging buffer; we write straight
            // into this region. We pick the pitch ourselves and align it to a
            // 128-byte boundary so device-side reads stay coalesced.
            pitch = ((static_cast<std::size_t>(width_px) + 127U) / 128U) * 128U;
            error = cudaMallocManaged(&this->data_,
                                      pitch * static_cast<std::size_t>(height_px));
        } else {
            error = cudaMallocPitch(&this->data_, &pitch, static_cast<std::size_t>(width_px),
                                    static_cast<std::size_t>(height_px));
        }
        if (error != cudaSuccess) {
            this->data_ = nullptr;
            set_message(out_message, std::string("cannot allocate the device buffer: ") +
                                             cudaGetErrorString(error));
            return Status::kCudaError;
        }
        this->capacity_width_px_ = width_px;
        this->capacity_height_px_ = height_px;
        this->view_.data_ = static_cast<const std::uint8_t*>(this->data_);
        this->view_.width_px_ = width_px;
        this->view_.height_px_ = height_px;
        this->view_.pitch_bytes_ = pitch;
        this->view_.space_ =
                (space == MemorySpace::kManaged) ? MemorySpace::kManaged : MemorySpace::kDevice;
        return Status::kOk;
    }

    Status reserve(int width_px, int height_px, std::string* out_message) {
        return this->reserve(MemorySpace::kDevice, width_px, height_px, out_message);
    }

    Status upload(const std::uint8_t* data, int width_px, int height_px,
                  std::size_t source_pitch_bytes, std::string* out_message) {
        if (data == nullptr) {
            set_message(out_message, "the input pointer is nullptr");
            return Status::kInvalidArgument;
        }
        if (this->data_ == nullptr) {
            set_message(out_message, "reserve() has not been called");
            return Status::kNotInitialized;
        }
        if (width_px <= 0 || height_px <= 0 || width_px > this->capacity_width_px_ ||
            height_px > this->capacity_height_px_) {
            set_message(out_message, "cannot transfer more than the reserved dimensions");
            return Status::kInvalidArgument;
        }
        if (this->space_ == MemorySpace::kManaged) {
            // With managed memory the device and the host see the same region,
            // so no transfer happens here. The migration cost shows up when the
            // device first touches the data.
            const cudaError_t copied = cudaMemcpy2D(
                    this->data_, this->view_.pitch_bytes_, data, source_pitch_bytes,
                    static_cast<std::size_t>(width_px), static_cast<std::size_t>(height_px),
                    cudaMemcpyHostToHost);
            if (copied != cudaSuccess) {
                set_message(out_message, std::string("cannot copy into the managed region: ") +
                                                 cudaGetErrorString(copied));
                return Status::kCudaError;
            }
            this->view_.width_px_ = width_px;
            this->view_.height_px_ = height_px;
            return Status::kOk;
        }

        // Whether the source is page-locked is the caller's decision. This
        // class reads the pointer it is given as is. Preparing a page-locked
        // input is the caller's responsibility: staging it through an
        // intermediate buffer here would measure the cost of that extra copy
        // rather than the effect of the input buffer's memory space.
        const cudaError_t error = cudaMemcpy2D(this->data_, this->view_.pitch_bytes_, data,
                                               source_pitch_bytes,
                                               static_cast<std::size_t>(width_px),
                                               static_cast<std::size_t>(height_px),
                                               cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            set_message(out_message, std::string("transfer to the device failed: ") +
                                             cudaGetErrorString(error));
            return Status::kCudaError;
        }
        this->view_.width_px_ = width_px;
        this->view_.height_px_ = height_px;
        return Status::kOk;
    }

    const ImageViewU8& view() const { return this->view_; }

private:
    void release() {
        if (this->data_ != nullptr) {
            static_cast<void>(cudaFree(this->data_));
            this->data_ = nullptr;
        }
        this->capacity_width_px_ = 0;
        this->capacity_height_px_ = 0;
        this->view_ = ImageViewU8{};
    }

    void* data_ = nullptr;
    MemorySpace space_ = MemorySpace::kDevice;
    int capacity_width_px_ = 0;
    int capacity_height_px_ = 0;
    ImageViewU8 view_;
};

DeviceImage::DeviceImage() : impl_(std::make_unique<Impl>()) {}
DeviceImage::~DeviceImage() = default;
DeviceImage::DeviceImage(DeviceImage&&) noexcept = default;
DeviceImage& DeviceImage::operator=(DeviceImage&&) noexcept = default;

Status DeviceImage::reserve(int width_px, int height_px, std::string* out_message) {
    return this->impl_->reserve(width_px, height_px, out_message);
}

Status DeviceImage::reserve(MemorySpace space, int width_px, int height_px,
                            std::string* out_message) {
    return this->impl_->reserve(space, width_px, height_px, out_message);
}

Status DeviceImage::upload(const std::uint8_t* data, int width_px, int height_px,
                           std::size_t source_pitch_bytes, std::string* out_message) {
    return this->impl_->upload(data, width_px, height_px, source_pitch_bytes, out_message);
}

const ImageViewU8& DeviceImage::view() const { return this->impl_->view(); }

}  // namespace aruco3cuda::hybrid
