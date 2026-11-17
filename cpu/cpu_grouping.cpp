// SPDX-License-Identifier: Apache-2.0
#include "cpu_grouping.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace aruco3cuda::cpu::detail {
namespace {
using hybrid::average_module_size;
using hybrid::average_quad_distance;
using hybrid::CandidateNode;
using hybrid::quad_inside_quad;
using hybrid::quad_perimeter;

struct Bounds {
    float min_x_;
    float min_y_;
    float max_x_;
    float max_y_;
    int sum_x_ = 0;
    int sum_y_ = 0;
    bool integer_image_points_ = true;
};

Bounds make_bounds(const std::vector<cv::Point2f>& corners) {
    Bounds bounds{corners[0].x, corners[0].y, corners[0].x, corners[0].y};
    for (const cv::Point2f& point : corners) {
        bounds.min_x_ = std::min(bounds.min_x_, point.x);
        bounds.min_y_ = std::min(bounds.min_y_, point.y);
        bounds.max_x_ = std::max(bounds.max_x_, point.x);
        bounds.max_y_ = std::max(bounds.max_y_, point.y);
        if (point.x >= 0.0F && point.x <= 65536.0F && point.y >= 0.0F && point.y <= 65536.0F) {
            const int x = static_cast<int>(point.x);
            const int y = static_cast<int>(point.y);
            bounds.integer_image_points_ = bounds.integer_image_points_ &&
                                           static_cast<float>(x) == point.x &&
                                           static_cast<float>(y) == point.y;
            bounds.sum_x_ += x;
            bounds.sum_y_ += y;
        } else {
            bounds.integer_image_points_ = false;
        }
    }
    return bounds;
}

bool separated(const Bounds& first, const Bounds& second, float threshold) {
    float gap = std::max({0.0F, first.min_x_ - second.max_x_, second.min_x_ - first.max_x_,
                          first.min_y_ - second.max_y_, second.min_y_ - first.max_y_});
    if (first.integer_image_points_ && second.integer_image_points_) {
        // RMS corner distance bounds the centroid distance for every rotation.
        // Integer image coordinates make these quarter-pixel centroids exact.
        // At the supported coordinate bound, float RMS rounding error is less
        // than 0.1 pixel. A full-pixel guard keeps boundary decisions conservative.
        const int difference = std::max(std::abs(first.sum_x_ - second.sum_x_),
                                        std::abs(first.sum_y_ - second.sum_y_));
        gap = std::max(gap, static_cast<float>(difference) * 0.25F - 1.0F);
    }
    if (gap == 0.0F) {
        return false;
    }
    // Every paired corner has at least this separation in one axis, regardless
    // of rotation. Use the same positive float accumulation and square root as
    // the exact metric so monotonic rounding preserves the lower bound. Do not
    // replace it with a squared-threshold test at the acceptance boundary.
    const float square = gap * gap;
    float sum = 0.0F;
    for (int corner = 0; corner < 4; ++corner) {
        sum += square;
    }
    return std::sqrt(sum / 4.0F) >= threshold;
}

bool contains(const Bounds& outer, const Bounds& inner) {
    return inner.min_x_ >= outer.min_x_ && inner.min_y_ >= outer.min_y_ &&
           inner.max_x_ <= outer.max_x_ && inner.max_y_ <= outer.max_y_;
}
}  // namespace

std::vector<CandidateNode> filter_spatial_candidates(
        const cv::Size& image_size, std::vector<std::vector<cv::Point2f>>& candidates,
        std::vector<std::vector<cv::Point>>& contours, const DetectorConfig& config,
        int marker_size) {
    std::vector<CandidateNode> nodes(candidates.size());
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        nodes[i].corners_ = std::move(candidates[i]);
        nodes[i].contour_ = std::move(contours[i]);
        nodes[i].perimeter_ = quad_perimeter(nodes[i].corners_);
    }
    // Sort by decreasing perimeter. Ties keep the order of the windows.
    std::stable_sort(nodes.begin(), nodes.end(),
                     [](const CandidateNode& lhs, const CandidateNode& rhs) {
                         return lhs.perimeter_ > rhs.perimeter_;
                     });

    std::vector<Bounds> bounds;
    bounds.reserve(nodes.size());
    for (const CandidateNode& node : nodes) {
        bounds.push_back(make_bounds(node.corners_));
    }
    std::vector<int> group_id(nodes.size(), -1);
    std::vector<std::vector<std::size_t>> groups;
    std::vector<bool> selected(nodes.size(), true);
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        for (std::size_t j = i + 1U; j < nodes.size(); ++j) {
            const float threshold =
                    nodes[j].perimeter_ * static_cast<float>(config.min_marker_distance_rate_);
            if (separated(bounds[i], bounds[j], threshold)) {
                continue;
            }
            const float distance = average_quad_distance(nodes[i].corners_, nodes[j].corners_);
            if (distance >=
                nodes[j].perimeter_ * static_cast<float>(config.min_marker_distance_rate_)) {
                continue;
            }
            selected[i] = false;
            selected[j] = false;
            if (group_id[i] < 0 && group_id[j] < 0) {
                group_id[i] = static_cast<int>(groups.size());
                group_id[j] = group_id[i];
                groups.push_back({i, j});
            } else if (group_id[i] > -1 && group_id[j] == -1) {
                group_id[j] = group_id[i];
                groups[static_cast<std::size_t>(group_id[i])].push_back(j);
            } else if (group_id[j] > -1 && group_id[i] == -1) {
                group_id[i] = group_id[j];
                groups[static_cast<std::size_t>(group_id[j])].push_back(i);
            }
            // When both already belong to different groups, they are not
            // merged. This matches OpenCV's behavior.
        }
        if (selected[i]) {
            // A candidate close to nothing else forms a group of its own.
            selected[i] = false;
            group_id[i] = static_cast<int>(groups.size());
            groups.push_back({i});
        }
    }

    for (std::vector<std::size_t>& group : groups) {
        // The indices follow decreasing perimeter, so sorting them ascending
        // puts the largest perimeter first.
        std::stable_sort(group.begin(), group.end());
        std::size_t current = group[0];
        bool too_near_border = false;
        for (const cv::Point2f& corner : nodes[current].corners_) {
            const auto margin = static_cast<float>(config.min_distance_to_border_px_);
            if (corner.x < margin || corner.y < margin ||
                corner.x > static_cast<float>(image_size.width - 1) - margin ||
                corner.y > static_cast<float>(image_size.height - 1) - margin) {
                too_near_border = true;
                break;
            }
        }
        if (too_near_border) {
            continue;
        }
        selected[current] = true;
        for (std::size_t k = 1U; k < group.size(); ++k) {
            const std::size_t id = group[k];
            const float distance =
                    average_quad_distance(nodes[id].corners_, nodes[current].corners_);
            const float module_size = average_module_size(nodes[id].corners_, marker_size,
                                                          config.marker_border_bits_);
            if (distance > static_cast<float>(config.min_group_distance_) * module_size) {
                current = id;
                nodes[group[0]].close_contours_.push_back(nodes[id]);
            }
        }
    }

    std::vector<Bounds> result_bounds;
    result_bounds.reserve(groups.size());
    std::vector<CandidateNode> result;
    result.reserve(groups.size());
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        if (selected[i]) {
            result.push_back(std::move(nodes[i]));
            result_bounds.push_back(bounds[i]);
        }
    }

    // Walk the containment relations to assign each parent and depth. result is
    // ordered by decreasing perimeter, so the more deeply nested a candidate is,
    // the later it appears.
    for (int i = static_cast<int>(result.size()) - 1; i >= 0; --i) {
        for (int j = i - 1; j >= 0; --j) {
            const auto outer = static_cast<std::size_t>(j);
            const auto inner = static_cast<std::size_t>(i);
            if (contains(result_bounds[outer], result_bounds[inner]) &&
                quad_inside_quad(result[inner].corners_, result[outer].corners_)) {
                result[inner].parent_ = j;
                result[outer].depth_ = std::max(result[outer].depth_, result[inner].depth_ + 1);
                break;
            }
        }
    }
    return result;
}
}  // namespace aruco3cuda::cpu::detail
