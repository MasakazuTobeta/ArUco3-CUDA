// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "component_candidates.hpp"
#include "tag_codebook.hpp"

namespace aruco3cuda::apriltag_small_cpu::detail {

// Expands each quarter-resolution midpoint to 4 x 4 native pixels. The 16 x 16
// tile gate suppresses inactive regions. The low plane is reused for midpoints.
void dense_midrange_mask(const cv::Mat& gray, cv::Mat& low, const cv::Mat& high,
                         const cv::Mat& tiles, int minimum_contrast, cv::Mat& mask);

struct DenseBoardConfig {
    int minimum_island_contrast_ = 18;
    int minimum_tile_contrast_ = 24;
    std::size_t minimum_candidates_ = 7;
    int minimum_ring_matches_ = 23;
    float minimum_decode_contrast_ = 13;
    int maximum_corrected_bits_ = 2;
    int maximum_component_side_px_ = 320;
    std::size_t minimum_planar_tags_ = 16;
    std::size_t curvature_probe_tags_ = 24;  // Zero disables the native curved-edge check.
    float minimum_normalized_bow_ =
            .045F;  // Native sag times image span divided by edge length squared.
    float minimum_curvature_r_squared_ = .20F;
    float maximum_lattice_center_shift_px_ = .5F;  // Zero disables pitch-based center refinement.
    float maximum_planar_correction_px_ = .7F;     // Zero preserves independently fitted corners.
};

// A per-detector workspace. Each call verifies the candidates using the current Gray8 frame.
// The returned detections remain valid until the next call.
class DenseBoardSearch {
public:
    explicit DenseBoardSearch(DenseBoardConfig config = {});
    const std::vector<Detection>& detect(const cv::Mat& gray, bool verify_seeds = false);

private:
    void make_mask(const cv::Mat& gray);
    Decode read_soft(const cv::Mat& gray, const Quad& quad) const;
    Decode read_candidate(const cv::Mat& gray, Quad& quad, float side, bool verify_seeds) const;

    DenseBoardConfig config_;
    ComponentConfig components_;
    ComponentCandidates search_;
    SupplementalComponents supplemental_;
    cv::Mat sampled_, high_, low_, tiles_, mask_;
    std::vector<std::uint64_t> packed_mask_;
    std::vector<std::vector<cv::Point>> polygons_;
    std::vector<Detection> detections_, independently_decoded_;
};

}  // namespace aruco3cuda::apriltag_small_cpu::detail
