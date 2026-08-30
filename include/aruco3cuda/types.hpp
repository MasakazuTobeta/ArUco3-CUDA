// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_TYPES_HPP
#define ARUCO3CUDA_TYPES_HPP

#include <cstddef>
#include <cstdint>
#include <string>

#include "aruco3cuda/status.hpp"

namespace aruco3cuda {

/// The memory space an input buffer lives in.
///
/// The DGX Spark GB10 and the Jetson Orin are both integrated GPUs, where host and
/// device share the same physical memory. Even so, the cost of an explicit copy, of
/// page locking, and of managed migration all differ, so the space is a distinct
/// measurement axis captured in the type rather than in the code path.
enum class MemorySpace : int {
    kHostPageable = 0,  ///< Ordinary host memory; a transfer incurs page locking
    kHostPinned,        ///< Page-locked host memory
    kManaged,           ///< Managed memory; no explicit copy
    kDevice,            ///< Device resident
};

/// Converts a MemorySpace into the notation used by the evaluation plan.
///
/// @param space Value to convert. Never returns nullptr, even for a value outside the
///              enumeration.
/// @return A string with static storage duration.
///
/// Ownership: the return value points into static storage. The caller neither frees
///            nor modifies it.
/// Synchronization: host only, with no synchronization point.
///
/// Example input: MemorySpace::kHostPinned
/// Example output: "M-Pinned"
const char* to_string(MemorySpace space);

/// A non-owning view of an 8-bit grayscale image.
///
/// Purpose:
///   Hands an image allocated by the caller to the detector without transferring
///   ownership. The row stride is kept separate from the width so that ROIs and
///   non-contiguous row layouts are supported from the start.
///
/// Ownership:
///   The caller owns the memory behind data_. This struct neither copies nor frees it.
///   The memory must stay valid for the duration of the detection.
struct ImageViewU8 {
    /// Pointer to the start of the image. It must live in the memory space named by space_.
    const std::uint8_t* data_ = nullptr;
    int width_px_ = 0;
    int height_px_ = 0;
    /// Row stride in bytes. It is not necessarily equal to width_px_.
    std::size_t pitch_bytes_ = 0;
    MemorySpace space_ = MemorySpace::kDevice;
};

/// Upper bounds on the image dimensions that can be handled.
///
/// These bounds exist because external input is not trusted, not because of a
/// performance constraint. They leave ample headroom above the 3840x2160 maximum
/// resolution of the evaluation plan.
inline constexpr int kMaxImageWidthPx = 65536;
inline constexpr int kMaxImageHeightPx = 65536;

/// Validates an image view at the boundary.
///
/// The public detector API runs this validation first. Passing an invalid view
/// straight to CUDA turns an out-of-range access into an asynchronous failure that
/// surfaces somewhere else entirely, making the cause hard to pin down.
///
/// Checks:
///   - data_ is not nullptr
///   - width_px_ and height_px_ are at least 1 and within the upper bounds
///   - pitch_bytes_ is at least the byte count of one row
///   - the product of pitch_bytes_ and height_px_ is representable in size_t
///   - space_ is one of the enumerators
///
/// @param image The view to validate.
/// @param out_message On failure, receives a reason containing "field=value". May be
///                    nullptr. Left unchanged on success. The caller owns the storage.
/// @return kOk if everything is valid, otherwise kInvalidImage.
///
/// Ownership: does not retain the memory behind the arguments.
/// Synchronization: host only, with no synchronization point. It calls no CUDA API, so
///                  it does not verify that the device pointer actually exists.
///
/// Example input: data_ = a valid device pointer, 1920x1080, pitch_bytes_ = 1920
/// Example output: Status::kOk
/// Example input: 1920x1080 with pitch_bytes_ = 1000
/// Example output: Status::kInvalidImage, with out_message containing "pitch_bytes=1000"
Status validate_image_view(const ImageViewU8& image, std::string* out_message = nullptr);

}  // namespace aruco3cuda

#endif  // ARUCO3CUDA_TYPES_HPP
