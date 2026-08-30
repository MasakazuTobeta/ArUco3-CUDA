// SPDX-License-Identifier: Apache-2.0
#include "aruco3cuda/types.hpp"

#include <cstddef>
#include <limits>
#include <string>

#include "aruco3cuda/status.hpp"

namespace aruco3cuda {
namespace {

/// Stores the reason for the failure into out_message. Does nothing when it is nullptr.
///
/// Called only on failure, so that the success path never builds a string.
void set_message(std::string* out_message, const std::string& name, long long value,
                 const std::string& reason) {
    if (out_message == nullptr) {
        return;
    }
    *out_message =
            "invalid image view: " + name + "=" + std::to_string(value) + " (" + reason + ")";
}

}  // namespace

const char* to_string(MemorySpace space) {
    switch (space) {
        case MemorySpace::kHostPageable:
            return "M-Pageable";
        case MemorySpace::kHostPinned:
            return "M-Pinned";
        case MemorySpace::kManaged:
            return "M-Managed";
        case MemorySpace::kDevice:
            return "M-Device";
    }
    // Never return nullptr, not even when a value outside the enumeration is passed in.
    return "Unknown";
}

Status validate_image_view(const ImageViewU8& image, std::string* out_message) {
    if (image.data_ == nullptr) {
        if (out_message != nullptr) {
            *out_message = "invalid image view: data is nullptr";
        }
        return Status::kInvalidImage;
    }
    if (image.width_px_ < 1 || image.width_px_ > kMaxImageWidthPx) {
        set_message(out_message, "width_px", image.width_px_,
                    "must be at least 1 and at most " + std::to_string(kMaxImageWidthPx));
        return Status::kInvalidImage;
    }
    if (image.height_px_ < 1 || image.height_px_ > kMaxImageHeightPx) {
        set_message(out_message, "height_px", image.height_px_,
                    "must be at least 1 and at most " + std::to_string(kMaxImageHeightPx));
        return Status::kInvalidImage;
    }

    // The image is 8-bit grayscale, so one row occupies width_px_ bytes.
    const auto required_pitch = static_cast<std::size_t>(image.width_px_);
    if (image.pitch_bytes_ < required_pitch) {
        set_message(out_message, "pitch_bytes", static_cast<long long>(image.pitch_bytes_),
                    "must be at least the " + std::to_string(required_pitch) +
                            " bytes taken by one row");
        return Status::kInvalidImage;
    }

    // If the product of pitch and height exceeds size_t, the extent computation wraps around and
    // an out-of-range access slips through validation.
    const auto height = static_cast<std::size_t>(image.height_px_);
    if (image.pitch_bytes_ > std::numeric_limits<std::size_t>::max() / height) {
        set_message(out_message, "pitch_bytes", static_cast<long long>(image.pitch_bytes_),
                    "the product with height_px exceeds the range of size_t");
        return Status::kInvalidImage;
    }

    switch (image.space_) {
        case MemorySpace::kHostPageable:
        case MemorySpace::kHostPinned:
        case MemorySpace::kManaged:
        case MemorySpace::kDevice:
            break;
        default:
            set_message(out_message, "space", static_cast<long long>(image.space_),
                        "is not one of the values enumerated by MemorySpace");
            return Status::kInvalidImage;
    }
    return Status::kOk;
}

}  // namespace aruco3cuda
