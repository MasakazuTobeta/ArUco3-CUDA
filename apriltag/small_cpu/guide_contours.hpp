// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cpu_contour_scan.hpp"

#include <algorithm>
#include <vector>

namespace aruco3cuda::apriltag_small_cpu::detail {

inline constexpr int kMinimumGuideContourSide = 7;

// The serial scan skips paths that cannot meet the two readers' minimum outline size.
// Keep OpenCV path order and follow its configured parallel route on a wider thread pool.
inline void find_guide_contours(const cv::Mat& mask,
                                aruco3cuda::cpu::detail::ContourScanner& scratch,
                                std::vector<std::vector<cv::Point>>& contours) {
    if (cv::getNumThreads() > 1) {
        cv::findContours(mask, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        return;
    }
    contours.clear();
    scratch.for_each<cv::CHAIN_APPROX_SIMPLE, true, true, 128>(
            mask, true, [&](const std::vector<cv::Point>& path) {
                if (path.size() < 2)
                    return;
                cv::Point low = path.front();
                cv::Point high = low;
                for (const auto& point : path) {
                    low.x = std::min(low.x, point.x);
                    low.y = std::min(low.y, point.y);
                    high.x = std::max(high.x, point.x);
                    high.y = std::max(high.y, point.y);
                }
                if (high.x - low.x + 1 >= kMinimumGuideContourSide &&
                    high.y - low.y + 1 >= kMinimumGuideContourSide)
                    contours.push_back(path);
            });
    std::reverse(contours.begin(), contours.end());
}

}  // namespace aruco3cuda::apriltag_small_cpu::detail
