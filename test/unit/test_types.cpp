// SPDX-License-Identifier: Apache-2.0
//
// Verifies boundary validation of the image view.
//
// Passing an invalid view straight to CUDA turns an out-of-bounds access into an
// asynchronous failure that surfaces far from its cause. Rejecting such views at
// the boundary is a precondition the detector relies on.
#include "aruco3cuda/types.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include "aruco3cuda/status.hpp"

namespace {

/// Only validation is exercised here, so a pointer to no real image is enough.
/// validate_image_view calls no CUDA API and does not confirm that the pointer
/// refers to real storage.
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

// Nominal: a well-formed view is accepted.
TEST(ImageViewTest, accepts_valid_view) {
    std::string message = "unchanged";
    EXPECT_EQ(aruco3cuda::validate_image_view(valid_view(), &message), aruco3cuda::Status::kOk);
    // On success the message is left untouched.
    EXPECT_EQ(message, "unchanged");
}

// Nominal: passing nullptr for out_message is allowed.
TEST(ImageViewTest, accepts_null_message) {
    EXPECT_EQ(aruco3cuda::validate_image_view(valid_view(), nullptr), aruco3cuda::Status::kOk);
    aruco3cuda::ImageViewU8 invalid = valid_view();
    invalid.data_ = nullptr;
    EXPECT_EQ(aruco3cuda::validate_image_view(invalid, nullptr), aruco3cuda::Status::kInvalidImage);
}

// Error case: a view whose data pointer is null is rejected.
TEST(ImageViewTest, rejects_null_data) {
    aruco3cuda::ImageViewU8 view = valid_view();
    view.data_ = nullptr;
    std::string message;
    EXPECT_EQ(aruco3cuda::validate_image_view(view, &message), aruco3cuda::Status::kInvalidImage);
    EXPECT_NE(message.find("data"), std::string::npos) << message;
}

// Boundary: the lower and upper limits on width and height.
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
    // The limits themselves are accepted.
    aruco3cuda::ImageViewU8 smallest = valid_view();
    smallest.width_px_ = 1;
    smallest.height_px_ = 1;
    smallest.pitch_bytes_ = 1;
    EXPECT_EQ(aruco3cuda::validate_image_view(smallest, &message), aruco3cuda::Status::kOk);
}

// Boundary: the pitch must cover at least one full row.
TEST(ImageViewTest, checks_pitch_lower_bound) {
    aruco3cuda::ImageViewU8 view = valid_view();
    std::string message;

    view.pitch_bytes_ = static_cast<std::size_t>(view.width_px_) - 1U;
    EXPECT_EQ(aruco3cuda::validate_image_view(view, &message), aruco3cuda::Status::kInvalidImage);
    EXPECT_NE(message.find("pitch_bytes"), std::string::npos) << message;

    // A pitch equal to the width means a contiguous layout, which is valid.
    view.pitch_bytes_ = static_cast<std::size_t>(view.width_px_);
    EXPECT_EQ(aruco3cuda::validate_image_view(view, &message), aruco3cuda::Status::kOk);

    // A pitch larger than the width means an ROI or padding, which is valid.
    view.pitch_bytes_ = static_cast<std::size_t>(view.width_px_) + 128U;
    EXPECT_EQ(aruco3cuda::validate_image_view(view, &message), aruco3cuda::Status::kOk);

    // A pitch of 0 does not cover a single row.
    view.pitch_bytes_ = 0U;
    EXPECT_EQ(aruco3cuda::validate_image_view(view, &message), aruco3cuda::Status::kInvalidImage);
}

// Error case: a view whose pitch times height exceeds size_t is rejected.
// If it slipped through, the range computation would wrap and the view would pass
// validation while still producing out-of-bounds accesses.
TEST(ImageViewTest, rejects_pitch_height_overflow) {
    aruco3cuda::ImageViewU8 view = valid_view();
    view.width_px_ = 1;
    view.height_px_ = 2;
    view.pitch_bytes_ = std::numeric_limits<std::size_t>::max();
    std::string message;
    EXPECT_EQ(aruco3cuda::validate_image_view(view, &message), aruco3cuda::Status::kInvalidImage);
    EXPECT_NE(message.find("size_t"), std::string::npos) << message;
}

// Error case: a memory space outside the enumeration is rejected.
TEST(ImageViewTest, rejects_unknown_memory_space) {
    aruco3cuda::ImageViewU8 view = valid_view();
    view.space_ = static_cast<aruco3cuda::MemorySpace>(999);
    std::string message;
    EXPECT_EQ(aruco3cuda::validate_image_view(view, &message), aruco3cuda::Status::kInvalidImage);
    EXPECT_NE(message.find("space"), std::string::npos) << message;
}

// Nominal: every defined memory space is accepted.
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

// Nominal: the memory space identifiers match the notation used in the evaluation plan.
TEST(MemorySpaceTest, identifiers_match_evaluation_plan) {
    EXPECT_STREQ(aruco3cuda::to_string(aruco3cuda::MemorySpace::kHostPageable), "M-Pageable");
    EXPECT_STREQ(aruco3cuda::to_string(aruco3cuda::MemorySpace::kHostPinned), "M-Pinned");
    EXPECT_STREQ(aruco3cuda::to_string(aruco3cuda::MemorySpace::kManaged), "M-Managed");
    EXPECT_STREQ(aruco3cuda::to_string(aruco3cuda::MemorySpace::kDevice), "M-Device");
    EXPECT_STREQ(aruco3cuda::to_string(static_cast<aruco3cuda::MemorySpace>(999)), "Unknown");
}

}  // namespace
