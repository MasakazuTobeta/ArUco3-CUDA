// SPDX-License-Identifier: Apache-2.0
//
// 画像 view の境界検証を確認する。
//
// 不正な view をそのまま CUDA へ渡すと、範囲外 access が非同期の失敗として
// 離れた場所で現れる。境界で弾めることが検出器の前提になる。
#include "aruco3cuda/types.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include "aruco3cuda/status.hpp"

namespace {

/// 検証だけを行うため、実体のない pointer を使う。
/// validate_image_view は CUDA API を呼ばず、pointer の実在を確認しない。
std::uint8_t* dummy_pointer() {
    static std::uint8_t storage = 0;
    return &storage;
}

aruco3cuda::ImageViewU8 valid_view() {
    aruco3cuda::ImageViewU8 view;
    view.data_ = dummy_pointer();
    view.width_px_ = 1920;
    view.height_px_ = 1080;
    view.pitch_bytes_ = 1920;
    view.space_ = aruco3cuda::MemorySpace::kDevice;
    return view;
}

// 正常系: 妥当な view を受理する。
TEST(ImageViewTest, accepts_valid_view) {
    std::string message = "unchanged";
    EXPECT_EQ(aruco3cuda::validate_image_view(valid_view(), &message), aruco3cuda::Status::kOk);
    // 成功時は message を変更しない。
    EXPECT_EQ(message, "unchanged");
}

// 正常系: out_message に nullptr を渡してもよい。
TEST(ImageViewTest, accepts_null_message) {
    EXPECT_EQ(aruco3cuda::validate_image_view(valid_view(), nullptr), aruco3cuda::Status::kOk);
    aruco3cuda::ImageViewU8 invalid = valid_view();
    invalid.data_ = nullptr;
    EXPECT_EQ(aruco3cuda::validate_image_view(invalid, nullptr), aruco3cuda::Status::kInvalidImage);
}

// 異常系: data が nullptr の view を拒否する。
TEST(ImageViewTest, rejects_null_data) {
    aruco3cuda::ImageViewU8 view = valid_view();
    view.data_ = nullptr;
    std::string message;
    EXPECT_EQ(aruco3cuda::validate_image_view(view, &message), aruco3cuda::Status::kInvalidImage);
    EXPECT_NE(message.find("data"), std::string::npos) << message;
}

// 境界値: 幅と高さの下限と上限。
TEST(ImageViewTest, checks_dimension_bounds) {
    std::string message;
    for (const int width : {0, -1, aruco3cuda::kMaxImageWidthPx + 1}) {
        aruco3cuda::ImageViewU8 view = valid_view();
        view.width_px_ = width;
        view.pitch_bytes_ = static_cast<std::size_t>(width > 0 ? width : 1);
        EXPECT_EQ(aruco3cuda::validate_image_view(view, &message),
                  aruco3cuda::Status::kInvalidImage)
                << "width=" << width;
        EXPECT_NE(message.find("width_px"), std::string::npos) << message;
    }
    for (const int height : {0, -1, aruco3cuda::kMaxImageHeightPx + 1}) {
        aruco3cuda::ImageViewU8 view = valid_view();
        view.height_px_ = height;
        EXPECT_EQ(aruco3cuda::validate_image_view(view, &message),
                  aruco3cuda::Status::kInvalidImage)
                << "height=" << height;
        EXPECT_NE(message.find("height_px"), std::string::npos) << message;
    }
    // 下限と上限そのものは受理する。
    aruco3cuda::ImageViewU8 smallest = valid_view();
    smallest.width_px_ = 1;
    smallest.height_px_ = 1;
    smallest.pitch_bytes_ = 1;
    EXPECT_EQ(aruco3cuda::validate_image_view(smallest, &message), aruco3cuda::Status::kOk);
}

// 境界値: pitch は 1 行分以上である必要がある。
TEST(ImageViewTest, checks_pitch_lower_bound) {
    aruco3cuda::ImageViewU8 view = valid_view();
    std::string message;

    view.pitch_bytes_ = static_cast<std::size_t>(view.width_px_) - 1U;
    EXPECT_EQ(aruco3cuda::validate_image_view(view, &message), aruco3cuda::Status::kInvalidImage);
    EXPECT_NE(message.find("pitch_bytes"), std::string::npos) << message;

    // 幅と等しい pitch は連続配置であり有効。
    view.pitch_bytes_ = static_cast<std::size_t>(view.width_px_);
    EXPECT_EQ(aruco3cuda::validate_image_view(view, &message), aruco3cuda::Status::kOk);

    // 幅より大きい pitch は ROI や padding であり有効。
    view.pitch_bytes_ = static_cast<std::size_t>(view.width_px_) + 128U;
    EXPECT_EQ(aruco3cuda::validate_image_view(view, &message), aruco3cuda::Status::kOk);

    // pitch が 0 は 1 行分を満たさない。
    view.pitch_bytes_ = 0U;
    EXPECT_EQ(aruco3cuda::validate_image_view(view, &message), aruco3cuda::Status::kInvalidImage);
}

// 異常系: pitch と高さの積が size_t を超える view を拒否する。
// 通り抜けると範囲計算が wrap し、検証を通過したまま範囲外 access になる。
TEST(ImageViewTest, rejects_pitch_height_overflow) {
    aruco3cuda::ImageViewU8 view = valid_view();
    view.width_px_ = 1;
    view.height_px_ = 2;
    view.pitch_bytes_ = std::numeric_limits<std::size_t>::max();
    std::string message;
    EXPECT_EQ(aruco3cuda::validate_image_view(view, &message), aruco3cuda::Status::kInvalidImage);
    EXPECT_NE(message.find("size_t"), std::string::npos) << message;
}

// 異常系: 列挙に無い memory 空間を拒否する。
TEST(ImageViewTest, rejects_unknown_memory_space) {
    aruco3cuda::ImageViewU8 view = valid_view();
    view.space_ = static_cast<aruco3cuda::MemorySpace>(999);
    std::string message;
    EXPECT_EQ(aruco3cuda::validate_image_view(view, &message), aruco3cuda::Status::kInvalidImage);
    EXPECT_NE(message.find("space"), std::string::npos) << message;
}

// 正常系: 全ての memory 空間を受理する。
TEST(ImageViewTest, accepts_all_memory_spaces) {
    const aruco3cuda::MemorySpace spaces[] = {
            aruco3cuda::MemorySpace::kHostPageable, aruco3cuda::MemorySpace::kHostPinned,
            aruco3cuda::MemorySpace::kManaged, aruco3cuda::MemorySpace::kDevice};
    for (const aruco3cuda::MemorySpace space : spaces) {
        aruco3cuda::ImageViewU8 view = valid_view();
        view.space_ = space;
        EXPECT_EQ(aruco3cuda::validate_image_view(view, nullptr), aruco3cuda::Status::kOk)
                << aruco3cuda::to_string(space);
    }
}

// 正常系: memory 空間の識別子が評価計画の表記と一致する。
TEST(MemorySpaceTest, identifiers_match_evaluation_plan) {
    EXPECT_STREQ(aruco3cuda::to_string(aruco3cuda::MemorySpace::kHostPageable), "M-Pageable");
    EXPECT_STREQ(aruco3cuda::to_string(aruco3cuda::MemorySpace::kHostPinned), "M-Pinned");
    EXPECT_STREQ(aruco3cuda::to_string(aruco3cuda::MemorySpace::kManaged), "M-Managed");
    EXPECT_STREQ(aruco3cuda::to_string(aruco3cuda::MemorySpace::kDevice), "M-Device");
    EXPECT_STREQ(aruco3cuda::to_string(static_cast<aruco3cuda::MemorySpace>(999)), "Unknown");
}

}  // namespace
