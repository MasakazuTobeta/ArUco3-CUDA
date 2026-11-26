// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "aruco3cuda/apriltag/small_cpu/detector.hpp"
#include "tag_codebook.hpp"

namespace aruco3cuda::apriltag_small_cpu::detail {

struct ObliqueBoardConfig {
    std::size_t minimum_seeds_ = 6;
    std::size_t maximum_seeds_ = 512;
    float minimum_long_edge_px_ = 20;
    // Projected area divided by the square of the longest side. Zero disables recovery.
    float maximum_area_ratio_ = .26F;
    float minimum_pitch_ = 1.025F;
    float maximum_pitch_ = 2.2F;
    float maximum_shape_error_ = .12F;  // In units of the neighboring printed marker width.
    int minimum_ring_matches_ = 27;
    float minimum_contrast_ = 22;
};

struct LargeBoardConfig {
    ObliqueBoardConfig oblique_;
    int stride_px_ = 8;
    int adaptive_window_ = 31;
    double adaptive_offset_ = 6;
    int minimum_side_samples_ = 8;
    double minimum_area_samples_ = 40;
    double polygon_tolerance_ratio_ = 0.04;
    int minimum_ring_matches_ = 19;
    float minimum_contrast_ = 10;
    int maximum_neighbor_trials_ = 220;
    int small_threshold_ = 112;
    int small_dark_threshold_ = 64;
    int small_bright_threshold_ = 176;
    int secondary_hysteresis_ = 10;
    int secondary_padding_px_ = 24;
    int small_roi_padding_px_ = 12;
    std::array<float, 3> expansion_samples_{{2.6F, 0.0F, -1.0F}};
};

// One workspace per caller. Current-frame neighbors require a decoded board pitch and source
// evidence for the payload. Results belong to this object until detect() is called again.
class LargeBoardSearch {
public:
    explicit LargeBoardSearch(LargeBoardConfig config = {});
    const std::vector<Detection>& detect(const cv::Mat& gray);
    // Extends current-frame detections only where same-plane neighbor geometry and the
    // native dark border, light surround, payload and rotation all agree.
    void recover_oblique(const cv::Mat& gray, std::vector<Detection>& tags);
    // Supplements a verified large-marker sighting with a separate native search for small tags.
    const std::vector<Detection>& detect_mixed(const cv::Mat& gray);

private:
    struct Seed {
        Projection projection;
        Point center;
        float side;
        float pitch = 0;
    };

    void append_small(const cv::Mat& gray);
    void search_tiles(const cv::Mat& gray, int bit, Detector& detector);
    void extract(const cv::Mat& gray);
    void accept_contour(const cv::Mat& gray, const Quad& quad);
    void supplement(const cv::Mat& gray);
    void try_neighbor(const cv::Mat& gray, const Quad& quad, Point center);
    void append(Quad quad, const Decode& tag);

    LargeBoardConfig config_;
    Detector small_, dark_, bright_;
    cv::Mat sampled_, mask_, bands_;
    std::vector<Detection> tags_;
    std::vector<std::vector<cv::Point>> contours_;
    std::vector<cv::Point> polygon_;
    std::vector<Seed> seeds_;
    std::vector<Point> centers_;
    std::vector<float> local_pitch_, all_pitch_;
};

}  // namespace aruco3cuda::apriltag_small_cpu::detail
