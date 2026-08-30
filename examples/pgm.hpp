// SPDX-License-Identifier: Apache-2.0
//
// Purpose:
//   Minimal binary PGM (P5) reader and writer, so that the samples need no image
//   library. PGM stores exactly what the detector takes as input, 8-bit grayscale,
//   so no conversion step sits between the file on disk and the API being shown.
//
// Why this is not part of the library:
//   The library deliberately performs no image I/O. Keeping the decoder here makes
//   that boundary visible: everything the public API needs is under
//   include/aruco3cuda, and this file is scaffolding that belongs to the samples.
#ifndef ARUCO3CUDA_EXAMPLES_PGM_HPP
#define ARUCO3CUDA_EXAMPLES_PGM_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace aruco3cuda_examples {

/// Upper bound on the side length the samples accept, in pixels.
///
/// A PGM header is untrusted input, so a declared size is never taken at face value.
/// The bound sits far above any resolution the samples target and keeps a corrupt
/// header from turning into a huge allocation.
inline constexpr int kMaxSidePx = 20000;

/// An 8-bit grayscale image held in host memory.
///
/// Purpose:
///   Carries a decoded PGM between the file and the detector. Rows are contiguous,
///   so the stride is always width_px_. The detector takes the stride separately,
///   and detect_image.cpp sets it from the device allocation rather than from this
///   struct, because cudaMallocPitch usually pads the rows.
///
/// Ownership: owns the pixel storage and releases it with the object.
/// Synchronization: host only, with no synchronization point.
///
/// Example input: a 280x280 PGM
/// Example output: width_px_ = 280, height_px_ = 280, pixels_ with 78400 elements
struct GrayImage {
    int width_px_ = 0;
    int height_px_ = 0;
    /// Row-major pixels. The size is width_px_ * height_px_.
    std::vector<std::uint8_t> pixels_;
};

/// Reads a binary PGM (P5) file.
///
/// The header is external input and is not trusted. The magic number, the
/// dimensions, the maximum value, and the length of the payload are all checked
/// before any storage is reserved. ASCII PGM (P2) and 16-bit PGM are reported as
/// errors rather than silently misread.
///
/// @param path Path of the file to read.
/// @param out On success, receives the decoded image. The caller owns the storage.
///            Left unchanged on failure.
/// @param out_message On failure, receives the reason. May be nullptr.
/// @return true on success, false if the file cannot be opened or is not a
///         supported PGM.
///
/// Ownership: does not retain the memory behind the arguments.
/// Synchronization: host only, with no synchronization point.
///
/// Example input: the path of a 280x280 P5 file whose maximum value is 255
/// Example output: true, with out->width_px_ = 280
/// Example input: the path of a P2 (ASCII) file
/// Example output: false, with out_message naming P2 as unsupported
bool read_pgm(const std::string& path, GrayImage* out, std::string* out_message = nullptr);

/// Writes a binary PGM (P5) file.
///
/// @param path Path of the file to write. An existing file is overwritten.
/// @param image The image to write. Its dimensions and the size of pixels_ must
///              agree, and the dimensions must be within kMaxSidePx.
/// @param out_message On failure, receives the reason. May be nullptr.
/// @return true on success, false if the image is inconsistent or the file cannot
///         be written.
///
/// Ownership: does not retain the memory behind the arguments.
/// Synchronization: host only, with no synchronization point.
///
/// Example input: a 280x280 image and the path "marker.pgm"
/// Example output: true, and marker.pgm holds 78400 bytes of payload
/// Example input: an image whose pixels_ is shorter than width_px_ * height_px_
/// Example output: false, with out_message naming the mismatch
bool write_pgm(const std::string& path, const GrayImage& image, std::string* out_message = nullptr);

}  // namespace aruco3cuda_examples

#endif  // ARUCO3CUDA_EXAMPLES_PGM_HPP
