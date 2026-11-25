// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cpu_contour_scan.hpp"
#include "noisy_crop_filter.hpp"

#include "aruco3cuda/apriltag/small_cpu/read_noise_config.hpp"
#include "aruco3cuda/apriltag/small_cpu/video_detector.hpp"

namespace aruco3cuda::apriltag_small_cpu::detail {

// The ordinary faint decoder and the noisy crop use the same current-source verification.
bool verify_faint_source(const cv::Mat& gray, const std::array<cv::Point2f, 4>& seed,
                         Detection& tag, int contrast, int radius);

// Refine a decoded tag from printed transitions in its current, locally averaged source crop.
bool fit_noisy_grid(const cv::Mat& filtered, const ReadNoiseConfig& noise, Detection& tag);

class NoisyInk {
public:
    // Reuse the ordinary profile's spatial grid and raw histogram information.
    static bool plausible(const cv::Mat& gray, const VideoConfig& settings,
                          const ReadNoiseConfig& noise, int row_step, int column_step, int count,
                          int rough, int raw_span);
    static void validate(const ReadNoiseConfig& noise);

    struct Proposals {
        std::vector<Detection> ordinary_;
        std::vector<Detection> outline_retries_;

        bool empty() const { return ordinary_.empty() && outline_retries_.empty(); }
    };

    // Supplemental proposals must verify the current filtered source crop.
    // If requested, return this frame's guide when every sample averages a full 4x4 block.
    // A supplied current guide must already describe the same input view.
    Proposals find(const cv::Mat& gray, const VideoConfig& settings, const ReadNoiseConfig& noise,
                   cv::Mat* area4 = nullptr, bool supplied_current = false);

private:
    aruco3cuda::cpu::detail::ContourScanner guide_contours_;
    cv::Mat guide_, mask_, median_, local_mean_, filtered_;
    NoisyCropFilter crop_filter_;
};

}  // namespace aruco3cuda::apriltag_small_cpu::detail
