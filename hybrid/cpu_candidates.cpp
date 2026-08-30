// SPDX-License-Identifier: Apache-2.0
#include "cpu_candidates.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "aruco3cuda/config.hpp"

namespace aruco3cuda::hybrid {


float quad_perimeter(const std::vector<cv::Point2f>& corners) {
    float perimeter = 0.0F;
    for (std::size_t i = 0; i < 4U; ++i) {
        const cv::Point2f edge = corners[i] - corners[(i + 1U) % 4U];
        perimeter += std::sqrt((edge.x * edge.x) + (edge.y * edge.y));
    }
    return perimeter;
}

float average_quad_distance(const std::vector<cv::Point2f>& first,
                            const std::vector<cv::Point2f>& second) {
    float min_distance_squared = std::numeric_limits<float>::max();
    for (int fc = 0; fc < 4; ++fc) {
        float distance_squared = 0.0F;
        for (int c = 0; c < 4; ++c) {
            const int mod_c = (c + fc) % 4;
            const cv::Point2f delta = first[static_cast<std::size_t>(mod_c)] -
                                      second[static_cast<std::size_t>(c)];
            distance_squared += (delta.x * delta.x) + (delta.y * delta.y);
        }
        distance_squared /= 4.0F;
        min_distance_squared = std::min(min_distance_squared, distance_squared);
    }
    return std::sqrt(min_distance_squared);
}

float average_module_size(const std::vector<cv::Point2f>& corners, int marker_size,
                          int border_bits) {
    const int module_count = marker_size + (border_bits * 2);
    return quad_perimeter(corners) / (4.0F * static_cast<float>(module_count));
}

bool quad_inside_quad(const std::vector<cv::Point2f>& first,
                      const std::vector<cv::Point2f>& second) {
    for (std::size_t i = 0; i < 4U; ++i) {
        if (cv::pointPolygonTest(second, first[i], false) < 0.0) {
            return false;
        }
    }
    return true;
}

std::vector<CandidateNode> filter_too_close_candidates(
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

    std::vector<int> group_id(nodes.size(), -1);
    std::vector<std::vector<std::size_t>> groups;
    std::vector<bool> selected(nodes.size(), true);
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        for (std::size_t j = i + 1U; j < nodes.size(); ++j) {
            const float distance = average_quad_distance(nodes[i].corners_, nodes[j].corners_);
            if (distance >= nodes[j].perimeter_ *
                                    static_cast<float>(config.min_marker_distance_rate_)) {
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

    std::vector<CandidateNode> result;
    result.reserve(groups.size());
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        if (selected[i]) {
            result.push_back(std::move(nodes[i]));
        }
    }

    // Walk the containment relations to assign each parent and depth. result is
    // ordered by decreasing perimeter, so the more deeply nested a candidate is,
    // the later it appears.
    for (int i = static_cast<int>(result.size()) - 1; i >= 0; --i) {
        for (int j = i - 1; j >= 0; --j) {
            const auto outer = static_cast<std::size_t>(j);
            const auto inner = static_cast<std::size_t>(i);
            if (quad_inside_quad(result[inner].corners_, result[outer].corners_)) {
                result[inner].parent_ = j;
                result[outer].depth_ = std::max(result[outer].depth_, result[inner].depth_ + 1);
                break;
            }
        }
    }
    return result;
}
void find_quad_candidates(const cv::Mat& binary, const DetectorConfig& config,
                          std::vector<std::vector<cv::Point2f>>& candidates,
                          std::vector<std::vector<cv::Point>>& contours_out) {
    const int longest = std::max(binary.cols, binary.rows);
    auto min_perimeter =
            static_cast<unsigned int>(config.min_marker_perimeter_rate_ * longest);
    const auto max_perimeter =
            static_cast<unsigned int>(config.max_marker_perimeter_rate_ * longest);
    if (config.use_aruco3_detection_ && config.min_side_length_canonical_img_px_ != 0) {
        min_perimeter = static_cast<unsigned int>(4 * config.min_side_length_canonical_img_px_);
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    for (const auto& contour : contours) {
        if (contour.size() < min_perimeter || contour.size() > max_perimeter) {
            continue;
        }
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx,
                         static_cast<double>(contour.size()) *
                                 config.polygonal_approx_accuracy_rate_,
                         true);
        if (approx.size() != 4U || !cv::isContourConvex(approx)) {
            continue;
        }
        double min_distance_squared = static_cast<double>(longest) * longest;
        for (int j = 0; j < 4; ++j) {
            const double dx = approx[j].x - approx[(j + 1) % 4].x;
            const double dy = approx[j].y - approx[(j + 1) % 4].y;
            min_distance_squared = std::min(min_distance_squared, dx * dx + dy * dy);
        }
        const double min_corner_distance =
                static_cast<double>(contour.size()) * config.min_corner_distance_rate_;
        if (min_distance_squared < min_corner_distance * min_corner_distance) {
            continue;
        }
        std::vector<cv::Point2f> candidate(4);
        for (int j = 0; j < 4; ++j) {
            candidate[j] = cv::Point2f(static_cast<float>(approx[j].x),
                                       static_cast<float>(approx[j].y));
        }
        candidates.push_back(candidate);
        contours_out.push_back(contour);
    }
}

void reorder_corners(std::vector<cv::Point2f>& candidate) {
    // Take the differences in float and only then widen to double, so the
    // rounding matches OpenCV.
    const double dx1 = static_cast<double>(candidate[1].x - candidate[0].x);
    const double dy1 = static_cast<double>(candidate[1].y - candidate[0].y);
    const double dx2 = static_cast<double>(candidate[2].x - candidate[0].x);
    const double dy2 = static_cast<double>(candidate[2].y - candidate[0].y);
    if ((dx1 * dy2) - (dy1 * dx2) < 0.0) {
        std::swap(candidate[1], candidate[3]);
    }
}
}  // namespace aruco3cuda::hybrid
