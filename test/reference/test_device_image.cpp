// SPDX-License-Identifier: Apache-2.0
//
// Verifies image upload to the device.
//
// The detector performs no upload of its own; it takes a view pointing at
// memory that is readable from the device. If an upload failure passes
// silently, detection still "succeeds" with zero results, and there is no way
// to tell whether the upload or the detection was at fault. These tests pin
// down the boundaries and the failure paths.
#include "device_image.hpp"

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"

namespace {

using aruco3cuda::MemorySpace;
using aruco3cuda::Status;
using aruco3cuda::hybrid::DeviceImage;

bool has_cuda_device() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

/// Test image with a fixed, deterministic sequence of values.
std::vector<std::uint8_t> make_pattern(int width, int height) {
    std::vector<std::uint8_t> data(static_cast<std::size_t>(width) *
                                   static_cast<std::size_t>(height));
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<std::uint8_t>(i % 251U);
    }
    return data;
}

/// Copies the device contents back to the host.
std::vector<std::uint8_t> download(const aruco3cuda::ImageViewU8& view) {
    std::vector<std::uint8_t> result(static_cast<std::size_t>(view.width_px_) *
                                     static_cast<std::size_t>(view.height_px_));
    const cudaError_t error =
            cudaMemcpy2D(result.data(), static_cast<std::size_t>(view.width_px_), view.data_,
                         view.pitch_bytes_, static_cast<std::size_t>(view.width_px_),
                         static_cast<std::size_t>(view.height_px_), cudaMemcpyDeviceToHost);
    EXPECT_EQ(error, cudaSuccess) << cudaGetErrorString(error);
    return result;
}

// Happy path: the uploaded content matches on the device side.
TEST(DeviceImageTest, uploads_content_unchanged) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    const int width = 133;
    const int height = 71;
    const std::vector<std::uint8_t> source = make_pattern(width, height);

    DeviceImage image;
    std::string message;
    ASSERT_EQ(image.reserve(width, height, &message), Status::kOk) << message;
    ASSERT_EQ(image.upload(source.data(), width, height, static_cast<std::size_t>(width), &message),
              Status::kOk)
            << message;

    const aruco3cuda::ImageViewU8& view = image.view();
    EXPECT_EQ(view.width_px_, width);
    EXPECT_EQ(view.height_px_, height);
    EXPECT_EQ(view.space_, MemorySpace::kDevice);
    // The pitch is at least the width; cudaMallocPitch aligns the start of
    // each row.
    EXPECT_GE(view.pitch_bytes_, static_cast<std::size_t>(width));
    EXPECT_EQ(download(view), source);
}

// Happy path: uploads correctly even when the host-side pitch differs from
// the width.
TEST(DeviceImageTest, handles_padded_source_pitch) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    const int width = 64;
    const int height = 40;
    const std::size_t source_pitch = static_cast<std::size_t>(width) + 17U;
    std::vector<std::uint8_t> padded(source_pitch * static_cast<std::size_t>(height), 0xEEU);
    const std::vector<std::uint8_t> expected = make_pattern(width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            padded[(static_cast<std::size_t>(y) * source_pitch) + static_cast<std::size_t>(x)] =
                    expected[(static_cast<std::size_t>(y) * static_cast<std::size_t>(width)) +
                             static_cast<std::size_t>(x)];
        }
    }

    DeviceImage image;
    std::string message;
    ASSERT_EQ(image.reserve(width, height, &message), Status::kOk) << message;
    ASSERT_EQ(image.upload(padded.data(), width, height, source_pitch, &message), Status::kOk)
            << message;
    EXPECT_EQ(download(image.view()), expected);
}

// Happy path: calling reserve again with the same or smaller dimensions does
// not reallocate.
TEST(DeviceImageTest, reserve_reuses_existing_buffer) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    DeviceImage image;
    ASSERT_EQ(image.reserve(200, 100), Status::kOk);
    const void* first = image.view().data_;
    const std::size_t pitch = image.view().pitch_bytes_;

    ASSERT_EQ(image.reserve(200, 100), Status::kOk);
    EXPECT_EQ(image.view().data_, first);
    ASSERT_EQ(image.reserve(120, 60), Status::kOk);
    // A smaller request does not reallocate, so that we avoid allocating once
    // per frame.
    EXPECT_EQ(image.view().data_, first);
    EXPECT_EQ(image.view().pitch_bytes_, pitch);
    EXPECT_EQ(image.view().width_px_, 120);
    EXPECT_EQ(image.view().height_px_, 60);
}

// Happy path: requesting larger dimensions reallocates.
TEST(DeviceImageTest, reserve_grows_for_larger_request) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    DeviceImage image;
    ASSERT_EQ(image.reserve(64, 64), Status::kOk);
    ASSERT_EQ(image.reserve(512, 512), Status::kOk);
    EXPECT_EQ(image.view().width_px_, 512);
    EXPECT_EQ(image.view().height_px_, 512);
    const std::vector<std::uint8_t> source = make_pattern(512, 512);
    ASSERT_EQ(image.upload(source.data(), 512, 512, 512U), Status::kOk);
    EXPECT_EQ(download(image.view()), source);
}

// Happy path: the content survives a move.
TEST(DeviceImageTest, can_be_moved) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    const std::vector<std::uint8_t> source = make_pattern(48, 32);
    DeviceImage original;
    ASSERT_EQ(original.reserve(48, 32), Status::kOk);
    ASSERT_EQ(original.upload(source.data(), 48, 32, 48U), Status::kOk);

    DeviceImage moved(std::move(original));
    EXPECT_EQ(download(moved.view()), source);

    DeviceImage assigned;
    assigned = std::move(moved);
    EXPECT_EQ(download(assigned.view()), source);
}

// Failure path: invalid arguments perform neither an allocation nor an
// upload.
TEST(DeviceImageTest, rejects_invalid_arguments) {
    DeviceImage image;
    std::string message;
    EXPECT_EQ(image.reserve(0, 10, &message), Status::kInvalidArgument);
    EXPECT_FALSE(message.empty());
    EXPECT_EQ(image.reserve(10, 0, &message), Status::kInvalidArgument);
    EXPECT_EQ(image.reserve(-1, 10, &message), Status::kInvalidArgument);
    // Omitting out_message must not crash.
    EXPECT_EQ(image.reserve(0, 10), Status::kInvalidArgument);

    const std::vector<std::uint8_t> source = make_pattern(8, 8);
    // Upload without a preceding reserve.
    EXPECT_EQ(image.upload(source.data(), 8, 8, 8U, &message), Status::kNotInitialized);
    EXPECT_FALSE(message.empty());
}

// Failure path: rejects an upload larger than the reserved dimensions.
TEST(DeviceImageTest, rejects_upload_larger_than_reserved) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    DeviceImage image;
    std::string message;
    ASSERT_EQ(image.reserve(32, 32, &message), Status::kOk) << message;
    const std::vector<std::uint8_t> source = make_pattern(64, 64);
    EXPECT_EQ(image.upload(source.data(), 64, 64, 64U, &message), Status::kInvalidArgument);
    EXPECT_EQ(image.upload(source.data(), 16, 64, 16U, &message), Status::kInvalidArgument);
    EXPECT_EQ(image.upload(nullptr, 16, 16, 16U, &message), Status::kInvalidArgument);
    EXPECT_EQ(image.upload(source.data(), 0, 16, 16U, &message), Status::kInvalidArgument);
}

// Happy path: the allocation method and the view's memory space vary per
// memory kind.
TEST(DeviceImageTest, reserves_each_memory_space) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    struct Case {
        aruco3cuda::MemorySpace requested_;
        aruco3cuda::MemorySpace expected_view_;
        const char* name_;
    };
    const std::vector<Case> cases = {
            {aruco3cuda::MemorySpace::kDevice, aruco3cuda::MemorySpace::kDevice, "device"},
            {aruco3cuda::MemorySpace::kHostPinned, aruco3cuda::MemorySpace::kDevice, "pinned"},
            {aruco3cuda::MemorySpace::kHostPageable, aruco3cuda::MemorySpace::kDevice, "pageable"},
            {aruco3cuda::MemorySpace::kManaged, aruco3cuda::MemorySpace::kManaged, "managed"},
    };
    // Fill in a known pattern so we can check that every kind delivers the
    // same content to the device side.
    std::vector<std::uint8_t> source(static_cast<std::size_t>(64) * 48U);
    for (std::size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<std::uint8_t>((i * 7U) % 251U);
    }

    for (const Case& item : cases) {
        aruco3cuda::hybrid::DeviceImage image;
        std::string message;
        ASSERT_EQ(image.reserve(item.requested_, 64, 48, &message), aruco3cuda::Status::kOk)
                << item.name_ << " " << message;
        EXPECT_EQ(image.view().space_, item.expected_view_) << item.name_;
        ASSERT_EQ(image.upload(source.data(), 64, 48, 64U, &message), aruco3cuda::Status::kOk)
                << item.name_ << " " << message;

        // Read the content back. Managed memory is directly readable from the
        // host.
        std::vector<std::uint8_t> received(source.size());
        ASSERT_EQ(cudaMemcpy2D(received.data(), 64U, image.view().data_, image.view().pitch_bytes_,
                               64U, 48U, cudaMemcpyDeviceToHost),
                  cudaSuccess)
                << item.name_;
        EXPECT_EQ(received, source) << item.name_;
    }
}

// Happy path: reserving again with a different memory kind does not reuse the
// previous allocation.
TEST(DeviceImageTest, changing_space_reallocates) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    aruco3cuda::hybrid::DeviceImage image;
    std::string message;
    ASSERT_EQ(image.reserve(aruco3cuda::MemorySpace::kDevice, 64, 48, &message),
              aruco3cuda::Status::kOk)
            << message;
    ASSERT_EQ(image.view().space_, aruco3cuda::MemorySpace::kDevice);
    ASSERT_EQ(image.reserve(aruco3cuda::MemorySpace::kManaged, 64, 48, &message),
              aruco3cuda::Status::kOk)
            << message;
    // The space changed, so the buffer was reallocated.
    //
    // **We deliberately do not assert that the pointer changes.** The allocator
    // is free to immediately reuse a freed address, and on Jetson AGX Orin it
    // in fact hands back the same address. What must be verified is that the
    // space changed and that the buffer is usable as that space.
    EXPECT_EQ(image.view().space_, aruco3cuda::MemorySpace::kManaged);
    // Managed memory is directly writable from the host; a device-only
    // allocation would fault here.
    auto* writable = const_cast<std::uint8_t*>(image.view().data_);
    writable[0] = 123U;
    EXPECT_EQ(writable[0], 123U);

    // Shrinking within the same space reuses the existing buffer.
    const aruco3cuda::ImageViewU8 managed = image.view();
    ASSERT_EQ(image.reserve(aruco3cuda::MemorySpace::kManaged, 32, 24, &message),
              aruco3cuda::Status::kOk)
            << message;
    EXPECT_EQ(image.view().data_, managed.data_);
    EXPECT_EQ(image.view().width_px_, 32);
}

}  // namespace
