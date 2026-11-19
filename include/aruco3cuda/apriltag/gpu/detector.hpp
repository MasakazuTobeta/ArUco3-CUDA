// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "aruco3cuda/dictionary.hpp"

namespace aruco3cuda::apriltag_gpu {

/// Segmentation, decoding and workspace settings. One detector uses one fixed layout.
struct Config {
    int decimate_ = 0;  ///< Zero selects adaptive sampling, 1–4 fixes the segmentation spacing.
    int small_frame_pixels_ =
            192 * 192;  ///< Adaptive mode scans this many pixels or fewer at full resolution.
    float min_pixels_per_cell_ =
            1;  ///< Minimum projected edge altitude per tag cell; zero disables.
    float fine_samples_per_cell_ =
            2;              ///< Adaptive full-resolution trigger, in coarse samples per cell.
    int min_contrast_ = 5;  ///< Minimum local light/dark range, in 8-bit levels.
    int min_cluster_points_ = 12;  ///< Minimum interface samples considered for a quad.
    int max_hamming_ = -1;         ///< 0–2 or -1 for two in 36h11 and zero in other families.
    int cluster_slots_ = 0;  ///< Power-of-two interface capacity, or 0 to scale with image size.
    int max_cluster_probes_ =
            128;           ///< Per-interface hash probe budget. Exhaustion rejects the frame.
    int max_tags_ = 1024;  ///< Maximum raw decoded quads before duplicate removal.
    float refine_radius_cells_ =
            1.25F;  ///< Post-decode edge search radius, in tag cells; zero disables.
    float duplicate_corner_ratio_ = 0.125F;  ///< Maximum corresponding corner offset / side.
};

/// One decoded marker with corners ordered clockwise from the canonical top left.
/// Ownership: a host value containing its own coordinates.
/// Synchronization: reading a returned tag requires no CUDA work.
/// Example input: a detected marker in DICT_APRILTAG_36h11.
/// Example output: id_ 7, hamming_ 0 and four (x,y) corner pairs.
struct Tag {
    int id_;
    int hamming_;
    std::array<float, 8> corners_;
};

/// Detect square AprilTags in device-resident grayscale images of a fixed size.
/// Ownership (applies to all public member functions): owns reusable scratch and
/// a dictionary copy on the current CUDA device. The caller owns the input and
/// stream and uses the same CUDA device on subsequent calls.
/// Synchronization (applies to all public member functions): construct before
/// timing. One host thread calls detect at a time. Each call completes the
/// supplied stream before returning host results.
/// Example input: a 640x480 8-bit device image and the 36h11 table.
/// Example output: one Tag for each decoded spatial instance.
class Detector {
public:
    /// Allocate scratch for this image layout and upload the dictionary.
    /// @param table Borrowed square codeword table. Contents are copied.
    /// @param width_px Image width from 1 through 16384.
    /// @param height_px Image height from 1 through 16384.
    /// @param config Segmentation, decoding and bounded workspace settings.
    /// Example input: the 36h11 table, 640, 480 and default settings.
    /// Example output: an initialized detector or an exception on failure.
    Detector(const DictionaryTable& table, int width_px, int height_px, Config config = {});
    ~Detector();
    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;

    /// Detect and return complete host results. Overflow and asynchronous CUDA
    /// failures throw. Check that an upload on another stream has completed.
    /// @param device_gray Device-accessible address of the first 8-bit pixel.
    /// @param pitch_bytes Row stride in bytes, at least the configured width.
    /// @param stream Caller-owned CUDA stream, including the default stream.
    /// @return Tags with full-resolution image coordinates (pixel centers at integers).
    /// Example input: the device address, cudaMallocPitch stride and a stream.
    /// Example output: an empty vector for a uniform white image.
    std::vector<Tag> detect(const std::uint8_t* device_gray, std::size_t pitch_bytes,
                            cudaStream_t stream);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace aruco3cuda::apriltag_gpu
