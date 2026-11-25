// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "aruco3cuda/apriltag/small_cpu/detector.hpp"

namespace aruco3cuda::apriltag_small_cpu::detail {

void validate_border_config(const BorderConfig& config);

// Uses the caller's validated grayscale image and appends unique edge detections.
class BorderSearch {
public:
    explicit BorderSearch(const BorderConfig& config, int correction_limit);
    ~BorderSearch();
    void append(const cv::Mat& gray, std::vector<Detection>& results);
    void align_visible(const cv::Mat& gray, std::vector<Detection>& results,
                       float margin_ratio = 0);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace aruco3cuda::apriltag_small_cpu::detail
