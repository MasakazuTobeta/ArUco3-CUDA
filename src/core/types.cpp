// SPDX-License-Identifier: Apache-2.0
#include "aruco3cuda/types.hpp"

#include <cstddef>
#include <limits>
#include <string>

#include "aruco3cuda/status.hpp"

namespace aruco3cuda {
namespace {

/// 失敗理由を out_message へ格納する。nullptr の場合は何もしない。
///
/// 成功経路で文字列を組み立てないよう、失敗時にだけ呼ぶ。
void set_message(std::string* out_message, const std::string& name, long long value,
                 const std::string& reason) {
    if (out_message == nullptr) {
        return;
    }
    *out_message = "画像 view が不正: " + name + "=" + std::to_string(value) + " (" + reason + ")";
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
    // 列挙に無い値が渡された場合も nullptr を返さない。
    return "Unknown";
}

Status validate_image_view(const ImageViewU8& image, std::string* out_message) {
    if (image.data_ == nullptr) {
        if (out_message != nullptr) {
            *out_message = "画像 view が不正: data が nullptr";
        }
        return Status::kInvalidImage;
    }
    if (image.width_px_ < 1 || image.width_px_ > kMaxImageWidthPx) {
        set_message(out_message, "width_px", image.width_px_,
                    "1 以上 " + std::to_string(kMaxImageWidthPx) + " 以下である必要がある");
        return Status::kInvalidImage;
    }
    if (image.height_px_ < 1 || image.height_px_ > kMaxImageHeightPx) {
        set_message(out_message, "height_px", image.height_px_,
                    "1 以上 " + std::to_string(kMaxImageHeightPx) + " 以下である必要がある");
        return Status::kInvalidImage;
    }

    // 8-bit grayscale であるため 1 行は width_px_ byte である。
    const auto required_pitch = static_cast<std::size_t>(image.width_px_);
    if (image.pitch_bytes_ < required_pitch) {
        set_message(out_message, "pitch_bytes", static_cast<long long>(image.pitch_bytes_),
                    "1 行分の " + std::to_string(required_pitch) + " byte 以上である必要がある");
        return Status::kInvalidImage;
    }

    // pitch と高さの積が size_t を超えると、範囲計算が wrap して
    // 検証を通り抜けたまま範囲外 access になる。
    const auto height = static_cast<std::size_t>(image.height_px_);
    if (image.pitch_bytes_ > std::numeric_limits<std::size_t>::max() / height) {
        set_message(out_message, "pitch_bytes", static_cast<long long>(image.pitch_bytes_),
                    "height_px との積が size_t の範囲を超える");
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
                        "MemorySpace の列挙に無い値である");
            return Status::kInvalidImage;
    }
    return Status::kOk;
}

}  // namespace aruco3cuda
