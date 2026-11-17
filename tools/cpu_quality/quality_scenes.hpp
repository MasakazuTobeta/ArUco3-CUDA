// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CPU_QUALITY_SCENES_HPP
#define ARUCO3CUDA_CPU_QUALITY_SCENES_HPP

#include <opencv2/core.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace aruco3cuda::cpu_quality {

/// One rendered marker's ID and oriented outer pixel-boundary geometry.
/// Ownership: owns values only; no external storage is retained.
/// Synchronization: plain values, no synchronization point.
/// Example input: ID 42 and four clockwise full-frame corners.
/// Example output: the same marker can be rendered into a deterministic frame.
struct MarkerSpec {
    int id_ = 0;
    std::array<cv::Point2f, 4> corners_{};
};

/// Independent frame generation settings; no preceding frame is consulted.
/// Ownership: owns strings, markers, and scalar values.
/// Synchronization: plain values, no synchronization point.
/// Example input: one marker, mild blur, noise and perspective.
/// Example output: a reproducible frame with known marker IDs and geometry.
struct FrameSpec {
    std::string name_;
    std::string family_;
    cv::Size size_{641, 479};
    std::uint64_t seed_ = 0U;
    std::vector<MarkerSpec> markers_;
    double blur_sigma_px_ = 0.0;
    double noise_sigma_levels_ = 0.0;
    double contrast_ = 1.0;
    double illumination_gradient_ = 0.0;
    double occlusion_fraction_ = 0.0;
    int background_kind_ = 0;
    bool require_detection_ = false;
};

/// Build the 96-frame adversarial synthetic preset, including detection-positive scenes.
/// @param size Full-frame dimensions; each side must be at least 320 pixels.
/// @param dictionary_count Number of dictionary IDs available, greater than zero.
/// @return Specs covering pose, projective distortion, blur/noise, lighting, size bounds,
///         border/occlusion, dense markers, and negative scenes.
/// Ownership: returns owned values; retains no argument.
/// Synchronization: synchronous host computation, no I/O and no previous-frame state.
/// Example input: cv::Size(641, 479), 250.
/// Example output: 96 reproducible specs with different IDs and placements.
std::vector<FrameSpec> make_adversarial_specs(cv::Size size = cv::Size(641, 479),
                                              int dictionary_count = 250);

/// Render one frame solely from its spec and an OpenCV predefined dictionary.
/// @param spec Frame settings; only borrowed during this call.
/// @param dictionary Marker dictionary obtained through the OpenCV public API.
/// @param border_bits Black border width in marker cells; must be positive.
/// @return A new CV_8UC1 image. Invalid settings throw std::invalid_argument.
/// Ownership: returned image owns its pixels; input values are not retained.
/// Synchronization: synchronous host operations; no CUDA or temporal reuse.
/// Example input: one mild projective spec and DICT_ARUCO_MIP_36h12.
/// Example output: a grayscale frame with the supplied oriented marker geometry.
cv::Mat render_frame(const FrameSpec& spec, const cv::aruco::Dictionary& dictionary,
                     int border_bits = 1);

}  // namespace aruco3cuda::cpu_quality
#endif
