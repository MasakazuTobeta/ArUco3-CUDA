// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>

namespace aruco3cuda::apriltag_small_cpu::detail {

/// Estimates integer camera translation and proposes changed regions in the current Gray8 frame.
/// Owns its thumbnails and scratch buffers. A single caller supplies frames in presentation order.
class MotionRegions {
public:
    explicit MotionRegions(int quiet_threshold = 2, int change_threshold = 22, int padding_px = 23,
                           int minimum_close_percent = 0, int maximum_regions = 5);

    /// Saves the current frame as the next comparison reference. Returns true when changed
    /// contains all proposed crops (possibly none). False requests full-frame discovery.
    /// The translation maps previous-frame coordinates to current-frame coordinates.
    /// reliable_overlap distinguishes broad residuals after a reliable alignment from a new scene.
    bool update(const cv::Mat& gray, bool force_full, cv::Point& translation,
                std::vector<cv::Rect>& changed, bool* reliable_overlap = nullptr);
    void reset();

private:
    struct Alignment {
        cv::Point shift;
        int exposure = 0;
        bool quiet = true;
        bool reliable = true;
    };

    Alignment align(const cv::Mat& gray, const cv::Point* retry_shift = nullptr) const;
    bool locate(const cv::Mat& gray, const Alignment& alignment, cv::Point& phase,
                std::vector<cv::Rect>& changed, bool& has_overlap);
    bool regions(cv::Size image_size, cv::Rect overlap, cv::Point phase, int threshold,
                 const cv::Mat& aligned_previous, std::vector<cv::Rect>& changed);

    int quiet_threshold_;
    int change_threshold_;
    int padding_px_;
    int minimum_close_percent_;
    std::size_t maximum_regions_;

    cv::Size size_;
    cv::Point phase_;
    cv::Point last_shift_;
    cv::Mat previous_;
    cv::Mat current_;
    cv::Mat adjusted_;
    cv::Mat tiles_;
    std::vector<std::vector<cv::Point>> contours_;
};

}  // namespace aruco3cuda::apriltag_small_cpu::detail
