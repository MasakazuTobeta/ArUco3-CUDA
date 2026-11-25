// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "aruco3cuda/apriltag/small_cpu/detector.hpp"

namespace aruco3cuda::apriltag_small_cpu::detail {

enum class AutomaticRoute { kNative, kCoarse, kComponents };

void validate_automatic_config(const AutomaticConfig& config);
AutomaticRoute choose_automatic_route(const cv::Mat& gray, const AutomaticConfig& config);

// A video caller can take responsibility for native discovery when sparse,
// widely separated component detections do not yet establish a dense board.
struct BoardSeedPolicy {
    std::size_t minimum_count_;
    std::size_t dense_count_;
    float maximum_side_px_;
    float minimum_spacing_sides_;
    bool dispersed(const std::vector<Detection>& tags) const;
};

// The component searches own their workspaces. Decisions use only the current input.
class AutomaticSearch {
public:
    explicit AutomaticSearch(const Config& config);
    std::vector<Detection> detect(const cv::Mat& gray,
                                  const BoardSeedPolicy* board_owner = nullptr);

private:
    friend class aruco3cuda::apriltag_small_cpu::Detector;
    std::vector<Detection> detect_route(const cv::Mat& gray, AutomaticRoute route,
                                        const BoardSeedPolicy* board_owner);

    const AutomaticConfig config_;
    Detector native_;
    Detector coarse_;
    Detector components_;
    cv::Mat sharpened_region_;
    cv::Mat resized_region_;
};

}  // namespace aruco3cuda::apriltag_small_cpu::detail
