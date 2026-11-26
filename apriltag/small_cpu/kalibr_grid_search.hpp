// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "aruco3cuda/apriltag/small_cpu/video_detector.hpp"
#include "component_candidates.hpp"
#include "tag_codebook.hpp"

namespace aruco3cuda::apriltag_small_cpu::detail {

class KalibrGridSearch {
public:
    explicit KalibrGridSearch(KalibrGridConfig config = {});

    // On light surrounds, bound residual dark samples outside ordinary current-frame tags.
    // An empty rectangle means the sparse guide found no unused dark ink.
    cv::Rect uncovered_region(const cv::Mat& gray, const std::vector<Detection>& ordinary);
    // A coarse preview searches for one two-border tag, verified in the current native image.
    bool has_candidate(const cv::Mat& gray);

    // Independent search on this Gray8 frame. The result lives until the next detect call.
    const std::vector<Detection>& detect(const cv::Mat& gray, bool check_ordinary = false);

    // Exact one-cell decodes on unused seed components of a confirmed two-cell scene.
    const std::vector<Detection>& ordinary_tags() const { return ordinary_; }

private:
    Decode read(const cv::Mat& gray, const Quad& quad) const;
    bool add_candidate(const cv::Mat& gray, Quad quad, bool check_ordinary = false);
    bool occupied(Point center, float radius) const;
    bool find_seeds(const cv::Mat& gray, cv::Rect region, double level, Point target = {},
                    float maximum_distance = -1, bool check_ordinary = false);
    void complete_grid(const cv::Mat& gray);

    KalibrGridConfig config_;
    ComponentCandidates components_;
    ComponentConfig component_config_;
    cv::Mat sampled_, mask_, cores_;
    cv::Mat preview_kernel_, erosion_kernel_;
    std::vector<std::vector<cv::Point>> polygons_;
    std::vector<Detection> tags_, ordinary_;
    std::vector<cv::Rect> explained_;
};

}  // namespace aruco3cuda::apriltag_small_cpu::detail
