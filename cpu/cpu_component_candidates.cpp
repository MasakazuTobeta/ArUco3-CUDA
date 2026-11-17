// SPDX-License-Identifier: Apache-2.0
#include "cpu_component_candidates.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace aruco3cuda::cpu::detail {
namespace {

std::int64_t cross(const cv::Point& first, const cv::Point& second, const cv::Point& point) {
    return static_cast<std::int64_t>(second.x - first.x) * (point.y - first.y) -
           static_cast<std::int64_t>(second.y - first.y) * (point.x - first.x);
}

int chain_length(const cv::Point& first, const cv::Point& second) {
    return std::max(std::abs(second.x - first.x), std::abs(second.y - first.y));
}

bool inside(const std::array<cv::Point, 4>& corners, const cv::Point& point) {
    bool positive = false;
    bool negative = false;
    for (std::size_t c = 0; c < 4U; ++c) {
        const std::int64_t value = cross(corners[c], corners[(c + 1U) % 4U], point);
        positive = positive || value > 0;
        negative = negative || value < 0;
    }
    return !(positive && negative);
}

}  // namespace

void ComponentCandidates::append(const cv::Mat& binary, const DetectorConfig& config,
                                 double edge_band_px,
                                 std::vector<std::vector<cv::Point2f>>& candidates,
                                 std::vector<std::vector<cv::Point>>& contours) {
    const int longest = std::max(binary.cols, binary.rows);
    int minimum = static_cast<int>(config.min_marker_perimeter_rate_ * longest);
    const int maximum = static_cast<int>(config.max_marker_perimeter_rate_ * longest);
    if (config.use_aruco3_detection_ && config.min_side_length_canonical_img_px_ != 0) {
        minimum = 4 * config.min_side_length_canonical_img_px_;
    }
    const int label_count = cv::connectedComponentsWithStats(
            binary, this->labels_, this->statistics_, this->centroids_, 8, CV_32S, cv::CCL_GRANA);
    this->active_labels_.assign(static_cast<std::size_t>(label_count), -1);
    this->offsets_.clear();
    this->offsets_.push_back(0U);
    std::vector<int> active;
    for (int label = 1; label < label_count; ++label) {
        const int* stats = this->statistics_.ptr<int>(label);
        const int area = stats[cv::CC_STAT_AREA];
        const int width = stats[cv::CC_STAT_WIDTH];
        const int height = stats[cv::CC_STAT_HEIGHT];
        // Both bounds are conservative for the four-extrema method. This avoids allocating
        // and revisiting the tiny components that dominate noisy thresholded images.
        if (static_cast<std::int64_t>(area) * 8 < minimum ||
            4 * std::max(width - 1, height - 1) < minimum) {
            continue;
        }
        const int index = static_cast<int>(active.size());
        active.push_back(label);
        this->active_labels_[static_cast<std::size_t>(label)] = index;
        this->offsets_.push_back(this->offsets_.back() + static_cast<std::size_t>(area));
    }
    this->points_.resize(this->offsets_.back());
    this->cursors_ = this->offsets_;
    for (int y = 0; y < binary.rows; ++y) {
        const int* labels = this->labels_.ptr<int>(y);
        for (int x = 0; x < binary.cols; ++x) {
            const int index = this->active_labels_[static_cast<std::size_t>(labels[x])];
            if (index >= 0) {
                this->points_[this->cursors_[static_cast<std::size_t>(index)]++] = cv::Point(x, y);
            }
        }
    }
    for (std::size_t index = 0; index < active.size(); ++index) {
        const std::size_t begin = this->offsets_[index];
        const std::size_t end = this->offsets_[index + 1U];
        const int label = active[index];
        const float center_x = static_cast<float>(this->centroids_.at<double>(label, 0));
        const float center_y = static_cast<float>(this->centroids_.at<double>(label, 1));
        std::array<cv::Point, 4> corners{};
        float farthest = -1.0F;
        for (std::size_t p = begin; p < end; ++p) {
            const cv::Point point = this->points_[p];
            const float dx = static_cast<float>(point.x) - center_x;
            const float dy = static_cast<float>(point.y) - center_y;
            const float distance = dx * dx + dy * dy;
            if (distance >= farthest) {
                farthest = distance;
                corners[0] = point;
            }
        }
        std::int64_t opposite_distance = -1;
        for (std::size_t p = begin; p < end; ++p) {
            const cv::Point point = this->points_[p];
            const std::int64_t dx = point.x - corners[0].x;
            const std::int64_t dy = point.y - corners[0].y;
            const std::int64_t distance = dx * dx + dy * dy;
            if (distance >= opposite_distance) {
                opposite_distance = distance;
                corners[2] = point;
            }
        }
        std::int64_t positive = 0;
        std::int64_t negative = 0;
        for (std::size_t p = begin; p < end; ++p) {
            const cv::Point point = this->points_[p];
            const std::int64_t distance = cross(corners[0], corners[2], point);
            if (distance > 0 && distance >= positive) {
                positive = distance;
                corners[1] = point;
            } else if (distance < 0 && -distance >= negative) {
                negative = -distance;
                corners[3] = point;
            }
        }
        if (positive == 0 || negative == 0) {
            continue;
        }
        if (cross(corners[0], corners[1], corners[2]) < 0) {
            std::swap(corners[1], corners[3]);
        }
        int perimeter = 0;
        std::int64_t shortest_squared = std::numeric_limits<std::int64_t>::max();
        bool near_border = false;
        for (std::size_t c = 0; c < 4U; ++c) {
            const cv::Point point = corners[c];
            const cv::Point next = corners[(c + 1U) % 4U];
            perimeter += chain_length(point, next);
            const std::int64_t dx = next.x - point.x;
            const std::int64_t dy = next.y - point.y;
            shortest_squared = std::min(shortest_squared, dx * dx + dy * dy);
            const int margin = config.min_distance_to_border_px_;
            near_border = near_border || point.x < margin || point.y < margin ||
                          point.x > binary.cols - 1 - margin || point.y > binary.rows - 1 - margin;
        }
        const double corner_distance = perimeter * config.min_corner_distance_rate_;
        if (near_border || perimeter < minimum || perimeter > maximum ||
            static_cast<double>(shortest_squared) < corner_distance * corner_distance) {
            continue;
        }
        std::array<double, 4> edge_length_squared{};
        for (std::size_t c = 0; c < 4U; ++c) {
            const cv::Point edge = corners[(c + 1U) % 4U] - corners[c];
            edge_length_squared[c] =
                    static_cast<double>(edge.x) * edge.x + static_cast<double>(edge.y) * edge.y;
        }
        std::size_t inliers = 0U;
        std::array<std::size_t, 4> support{};
        for (std::size_t p = begin; p < end; ++p) {
            const cv::Point point = this->points_[p];
            inliers += inside(corners, point) ? 1U : 0U;
            for (std::size_t c = 0; c < 4U; ++c) {
                const cv::Point edge = corners[(c + 1U) % 4U] - corners[c];
                const cv::Point delta = point - corners[c];
                const double projection = static_cast<double>(delta.x) * edge.x +
                                          static_cast<double>(delta.y) * edge.y;
                const double perpendicular =
                        static_cast<double>(cross(corners[c], corners[(c + 1U) % 4U], point));
                if (projection >= 0.0 && projection <= edge_length_squared[c] &&
                    perpendicular * perpendicular <=
                            edge_band_px * edge_band_px * edge_length_squared[c]) {
                    ++support[c];
                }
            }
        }
        if (static_cast<double>(inliers) / static_cast<double>(end - begin) <
            config.min_quad_inlier_ratio_) {
            continue;
        }
        bool supported = true;
        for (std::size_t c = 0; c < 4U; ++c) {
            const int edge_chain = chain_length(corners[c], corners[(c + 1U) % 4U]);
            if (edge_chain == 0 ||
                static_cast<double>(support[c]) / static_cast<double>(edge_chain) <
                        config.min_edge_support_ratio_) {
                supported = false;
                break;
            }
        }
        if (!supported) {
            continue;
        }
        std::vector<cv::Point2f> quad(4);
        for (std::size_t c = 0; c < 4U; ++c) {
            quad[c] =
                    cv::Point2f(static_cast<float>(corners[c].x), static_cast<float>(corners[c].y));
        }
        candidates.push_back(std::move(quad));
        // Host grouping and level selection consume the contour's length but never its
        // samples. A placeholder avoids reconstructing an ordered contour that was not found.
        contours.emplace_back(static_cast<std::size_t>(perimeter));
    }
}

}  // namespace aruco3cuda::cpu::detail
