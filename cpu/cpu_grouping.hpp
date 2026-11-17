// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CPU_GROUPING_HPP
#define ARUCO3CUDA_CPU_GROUPING_HPP

#include "cpu_candidates.hpp"

namespace aruco3cuda::cpu::detail {

/// Groups candidates with conservative spatial bounds before the exact OpenCV-compatible tests.
/// @param image_size Size of the current segmentation image.
/// @param candidates Four finite corners per candidate, moved out on return.
/// @param contours Matching contour vectors, moved out on return.
/// @param config Validated detector configuration.
/// @param marker_size Dictionary side length in cells.
/// @return Representatives and containment relations in the original stable order.
/// Ownership: consumes vector elements; the returned nodes own their storage.
/// Synchronization: synchronous and reentrant; retains no previous-frame state.
/// Example input: three threshold-window contours surrounding the same marker.
/// Example output: one representative with the same retries as the unpruned grouping.
std::vector<hybrid::CandidateNode> filter_spatial_candidates(
        const cv::Size& image_size, std::vector<std::vector<cv::Point2f>>& candidates,
        std::vector<std::vector<cv::Point>>& contours, const DetectorConfig& config,
        int marker_size);

}  // namespace aruco3cuda::cpu::detail
#endif
