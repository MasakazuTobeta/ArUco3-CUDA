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

/// 失敗理由を out_message へ入れる。nullptr なら何もしない。
void set_message(std::string* out_message, const std::string& text) {
    if (out_message != nullptr) {
        *out_message = text;
    }
}

}  // namespace

/// 実装本体。公開 header が CUDA へ依存しないよう pimpl とする。
class DeviceImage::Impl {
public:
    Impl() = default;
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
    ~Impl() { this->release(); }

    Status reserve(int width_px, int height_px, std::string* out_message) {
        if (width_px <= 0 || height_px <= 0) {
            set_message(out_message, "画像の寸法は 1 以上である必要がある");
            return Status::kInvalidArgument;
        }
        if (this->data_ != nullptr && width_px <= this->capacity_width_px_ &&
            height_px <= this->capacity_height_px_) {
            // 既存の領域で足りる。frame ごとの確保を避ける。
            this->view_.width_px_ = width_px;
            this->view_.height_px_ = height_px;
            return Status::kOk;
        }
        this->release();

        std::size_t pitch = 0;
        const cudaError_t error =
                cudaMallocPitch(&this->data_, &pitch, static_cast<std::size_t>(width_px),
                                static_cast<std::size_t>(height_px));
        if (error != cudaSuccess) {
            this->data_ = nullptr;
            set_message(out_message,
                        std::string("device buffer を確保できない: ") + cudaGetErrorString(error));
            return Status::kCudaError;
        }
        this->capacity_width_px_ = width_px;
        this->capacity_height_px_ = height_px;
        this->view_.data_ = static_cast<const std::uint8_t*>(this->data_);
        this->view_.width_px_ = width_px;
        this->view_.height_px_ = height_px;
        this->view_.pitch_bytes_ = pitch;
        this->view_.space_ = MemorySpace::kDevice;
        return Status::kOk;
    }

    Status upload(const std::uint8_t* data, int width_px, int height_px,
                  std::size_t source_pitch_bytes, std::string* out_message) {
        if (data == nullptr) {
            set_message(out_message, "入力 pointer が nullptr である");
            return Status::kInvalidArgument;
        }
        if (this->data_ == nullptr) {
            set_message(out_message, "reserve() が呼ばれていない");
            return Status::kNotInitialized;
        }
        if (width_px <= 0 || height_px <= 0 || width_px > this->capacity_width_px_ ||
            height_px > this->capacity_height_px_) {
            set_message(out_message, "確保済みの寸法を超える転送はできない");
            return Status::kInvalidArgument;
        }
        const cudaError_t error = cudaMemcpy2D(this->data_, this->view_.pitch_bytes_, data,
                                               source_pitch_bytes,
                                               static_cast<std::size_t>(width_px),
                                               static_cast<std::size_t>(height_px),
                                               cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            set_message(out_message,
                        std::string("device への転送に失敗した: ") + cudaGetErrorString(error));
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

Status DeviceImage::upload(const std::uint8_t* data, int width_px, int height_px,
                           std::size_t source_pitch_bytes, std::string* out_message) {
    return this->impl_->upload(data, width_px, height_px, source_pitch_bytes, out_message);
}

const ImageViewU8& DeviceImage::view() const { return this->impl_->view(); }

}  // namespace aruco3cuda::hybrid
