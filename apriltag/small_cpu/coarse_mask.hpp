// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "aruco3cuda/apriltag/small_cpu/detector.hpp"

#include <cstdint>
#include <vector>

namespace aruco3cuda::apriltag_small_cpu::detail {

// Threshold used by the fixed or exposure-adjusted component route.
int component_threshold(const cv::Mat& gray, const ComponentConfig& config);

class CoarseMask {
public:
    // The result belongs to this workspace and lasts until the next build call.
    // With sample_origin, fixed-threshold masks may omit zero margins.
    // Set bound_adaptive to also allow a bounded adaptive mask.
    // If packed is populated, it contains the eroded mask with bit zero at
    // the left. The returned raster provides the extent in that case.
    const cv::Mat& build(const cv::Mat& gray, const Config& config, const CoarseConfig& coarse,
                         cv::Point* sample_origin = nullptr, bool bound_adaptive = false,
                         std::vector<std::uint64_t>* packed = nullptr);

    const cv::Mat& build_components(const cv::Mat& gray, const ComponentConfig& config);

private:
    struct Range {
        std::uint8_t low_;
        std::uint8_t high_;
    };

    void threshold_tiles(const cv::Mat& gray, const CoarseConfig& config, cv::Point* sample_origin,
                         bool emit_bits);

    cv::Mat local_kernel_;
    cv::Mat local_high_;
    cv::Mat local_low_;
    cv::Mat sampled_;
    cv::Mat adaptive_binary_storage_;
    cv::Mat adaptive_eroded_storage_;
    cv::Mat binary_;
    cv::Mat eroded_;
    std::vector<std::uint8_t> row_, threshold_row_;
    std::vector<std::uint64_t> source_bits_;
    cv::Mat extrema_;
    cv::Mat neighboring_extrema_;
};

}  // namespace aruco3cuda::apriltag_small_cpu::detail
