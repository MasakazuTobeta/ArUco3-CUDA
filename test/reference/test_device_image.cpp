// SPDX-License-Identifier: Apache-2.0
//
// device への画像転送を検証する。
//
// 検出器は転送を行わず、device から読める memory を指す view を受け取る。
// 転送の失敗を無言で通すと、検出が「0 件」として成立してしまい、原因が
// 転送なのか検出なのか区別できない。境界と異常系を固定する。
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

/// 決まった値の並びを持つ試験画像。
std::vector<std::uint8_t> make_pattern(int width, int height) {
    std::vector<std::uint8_t> data(static_cast<std::size_t>(width) *
                                   static_cast<std::size_t>(height));
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<std::uint8_t>(i % 251U);
    }
    return data;
}

/// device の内容を host へ戻す。
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

// 正常系: 転送した内容が device 側で一致する。
TEST(DeviceImageTest, uploads_content_unchanged) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
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
    // pitch は幅以上になる。cudaMallocPitch は行頭を整列させる。
    EXPECT_GE(view.pitch_bytes_, static_cast<std::size_t>(width));
    EXPECT_EQ(download(view), source);
}

// 正常系: host 側の pitch が幅と異なっても正しく転送する。
TEST(DeviceImageTest, handles_padded_source_pitch) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
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

// 正常系: 同じか小さい寸法で reserve を呼び直しても確保し直さない。
TEST(DeviceImageTest, reserve_reuses_existing_buffer) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    DeviceImage image;
    ASSERT_EQ(image.reserve(200, 100), Status::kOk);
    const void* first = image.view().data_;
    const std::size_t pitch = image.view().pitch_bytes_;

    ASSERT_EQ(image.reserve(200, 100), Status::kOk);
    EXPECT_EQ(image.view().data_, first);
    ASSERT_EQ(image.reserve(120, 60), Status::kOk);
    // 小さい要求では確保し直さない。frame ごとの確保を避けるためである。
    EXPECT_EQ(image.view().data_, first);
    EXPECT_EQ(image.view().pitch_bytes_, pitch);
    EXPECT_EQ(image.view().width_px_, 120);
    EXPECT_EQ(image.view().height_px_, 60);
}

// 正常系: 大きい寸法を要求すると確保し直す。
TEST(DeviceImageTest, reserve_grows_for_larger_request) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
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

// 正常系: move しても内容を保つ。
TEST(DeviceImageTest, can_be_moved) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
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

// 異常系: 引数が不正なら確保も転送も行わない。
TEST(DeviceImageTest, rejects_invalid_arguments) {
    DeviceImage image;
    std::string message;
    EXPECT_EQ(image.reserve(0, 10, &message), Status::kInvalidArgument);
    EXPECT_FALSE(message.empty());
    EXPECT_EQ(image.reserve(10, 0, &message), Status::kInvalidArgument);
    EXPECT_EQ(image.reserve(-1, 10, &message), Status::kInvalidArgument);
    // out_message を渡さなくても落ちない。
    EXPECT_EQ(image.reserve(0, 10), Status::kInvalidArgument);

    const std::vector<std::uint8_t> source = make_pattern(8, 8);
    // reserve していない状態での転送。
    EXPECT_EQ(image.upload(source.data(), 8, 8, 8U, &message), Status::kNotInitialized);
    EXPECT_FALSE(message.empty());
}

// 異常系: 確保済みの寸法を超える転送を拒否する。
TEST(DeviceImageTest, rejects_upload_larger_than_reserved) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
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

}  // namespace
