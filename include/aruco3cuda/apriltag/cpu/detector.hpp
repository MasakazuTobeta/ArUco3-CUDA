// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace aruco3cuda::apriltag_cpu {
enum class Family { k16h5, k25h9, k36h10, k36h11 };

struct Config {
    Family family_ = Family::k36h11;
    // Number of upstream detector workers (1 through 256).
    int threads_ = 1;
    // Integer quad extraction downsampling factor (1 through 4). Decoding uses full resolution.
    int quad_decimate_ = 1;
    // -1 uses OpenCV's default budget. Overrides accept 0 through 3 corrected bits.
    int corrected_bits_ = -1;
    // A projected side shorter than this many pixels per border/payload cell is rejected.
    double min_edge_px_per_cell_ = 1.0;
    // Minimum fraction of samples agreeing with a cell. The default requires a strict majority.
    double min_cell_agreement_ = 9.0 / 16.0;
    // Permitted number of wrong black ring cells as a fraction of payload area.
    double max_border_error_rate_ = 0.35;
};

/// Corner position in the input image, in pixels (OpenCV AprilTag convention).
/// Ownership: plain coordinates. Synchronization: ready on return.
/// Example input: first decoded corner. Example output: x_=20, y_=40.
struct Corner {
    double x_;
    double y_;
};

struct Detection {
    int id_;
    // Clockwise corners, starting at the top left of a canonically oriented marker.
    std::array<Corner, 4> corners_;
};

/// Detects classic AprilTag families with the AprilTag 3 CPU algorithm.
/// Ownership (applies to all public member functions): the detector owns its configuration,
/// decoder and worker pool. Callers own images and returned results.
/// Synchronization (applies to all public member functions): calls complete synchronously.
/// Use each instance from one caller at a time.
/// Example input: Config with family k36h11. Example output: a reusable detector.
class Detector {
public:
    /// Set up a detector and build the family decoder index.
    /// @param config Detection parameters, copied at construction.
    /// Example input: Config{}. Example output: a detector for tag36h11.
    explicit Detector(const Config& config = {});
    ~Detector();
    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;

    /// Detect tags with IDs and clockwise, canonically oriented corners.
    /// @param gray Beginning of a single-channel, 8-bit grayscale image.
    /// @param size_bytes Accessible bytes from gray, including row padding before the last row.
    /// @param width Visible width, 1 through 32767 pixels.
    /// @param height Visible height, 1 through 32767 pixels.
    /// @param stride Distance between consecutive row starts, in bytes. At least width.
    /// Upstream requires height times stride to fit in a signed int.
    /// @return Owning detection list. Invalid inputs throw std::invalid_argument.
    /// Example input: detect(gray, 307200, 640, 480, 640).
    /// Example output: an empty vector on a constant image.
    std::vector<Detection> detect(const std::uint8_t* gray, std::size_t size_bytes, int width,
                                  int height, int stride);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace aruco3cuda::apriltag_cpu
