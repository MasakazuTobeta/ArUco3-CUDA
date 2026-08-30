// SPDX-License-Identifier: Apache-2.0
#include "report_diff.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "aruco3cuda/util/json_writer.hpp"
#include "geometry.hpp"

namespace aruco3cuda::report {
namespace {

/// Centroid of the four corners.
std::pair<double, double> centroid(const Detection& detection) {
    return quad_centroid(detection.corners_);
}

/// Mean side length of the quadrilateral formed by the four corners. Used to
/// derive the matching radius.
double average_side(const Detection& detection) {
    return quad_average_side(detection.corners_);
}

/// Largest corner-to-corner distance after rotating the target corners by steps positions.
double corner_error(const Detection& baseline, const Detection& target, int steps) {
    return quad_corner_error(baseline.corners_, target.corners_, steps);
}

/// One candidate pairing.
struct Pair {
    std::size_t baseline_index_ = 0;
    std::size_t target_index_ = 0;
    double distance_ = 0.0;
};

}  // namespace

const char* diff_kind_name(DiffKind kind) {
    switch (kind) {
        case DiffKind::kMissed:
            return "missed";
        case DiffKind::kExtra:
            return "extra";
        case DiffKind::kIdMismatch:
            return "id mismatch";
        case DiffKind::kRotationMismatch:
            return "rotation mismatch";
        case DiffKind::kCornerShift:
            return "corner shift";
    }
    // A value outside the enumeration. Keeps a future added kind visible instead of silent.
    return "unknown";
}

ImageComparison compare_detections(const std::string& image_path,
                                   const std::vector<Detection>& baseline,
                                   const std::vector<Detection>& target,
                                   const CompareConfig& config) {
    ImageComparison comparison;
    comparison.image_path_ = image_path;
    comparison.baseline_count_ = baseline.size();
    comparison.target_count_ = target.size();

    // Commit the pairs with nearby centroids in order of increasing distance. Because
    // the pairing is by position rather than by ID, a misread ID still counts as a
    // single difference.
    std::vector<Pair> pairs;
    for (std::size_t i = 0; i < baseline.size(); ++i) {
        const auto base_center = centroid(baseline[i]);
        const double radius = average_side(baseline[i]) * config.match_radius_ratio_;
        for (std::size_t j = 0; j < target.size(); ++j) {
            const auto target_center = centroid(target[j]);
            const double dx = base_center.first - target_center.first;
            const double dy = base_center.second - target_center.second;
            const double distance = std::sqrt((dx * dx) + (dy * dy));
            if (distance <= radius) {
                pairs.push_back({i, j, distance});
            }
        }
    }
    // Break ties on distance by input order, so the result does not change from run to run.
    std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) {
        if (a.distance_ != b.distance_) {
            return a.distance_ < b.distance_;
        }
        if (a.baseline_index_ != b.baseline_index_) {
            return a.baseline_index_ < b.baseline_index_;
        }
        return a.target_index_ < b.target_index_;
    });

    std::vector<bool> baseline_matched(baseline.size(), false);
    std::vector<bool> target_matched(target.size(), false);
    for (const Pair& pair : pairs) {
        if (baseline_matched[pair.baseline_index_] || target_matched[pair.target_index_]) {
            continue;
        }
        baseline_matched[pair.baseline_index_] = true;
        target_matched[pair.target_index_] = true;

        const Detection& base = baseline[pair.baseline_index_];
        const Detection& other = target[pair.target_index_];
        const auto base_center = centroid(base);

        Diff diff;
        diff.baseline_id_ = base.id_;
        diff.target_id_ = other.id_;
        diff.center_x_px_ = base_center.first;
        diff.center_y_px_ = base_center.second;

        if (base.id_ != other.id_) {
            diff.kind_ = DiffKind::kIdMismatch;
            diff.corner_error_px_ = corner_error(base, other, 0);
            comparison.diffs_.push_back(diff);
            continue;
        }

        const double aligned_error = corner_error(base, other, 0);
        comparison.worst_corner_error_px_ =
                std::max(comparison.worst_corner_error_px_, aligned_error);
        if (aligned_error <= config.corner_tolerance_px_) {
            ++comparison.agreed_count_;
            continue;
        }

        // If the corners agree after rotating, the problem is the corner ordering rather
        // than the position.
        int best_steps = 0;
        double best_error = aligned_error;
        for (int steps = 1; steps < 4; ++steps) {
            const double error = corner_error(base, other, steps);
            if (error < best_error) {
                best_error = error;
                best_steps = steps;
            }
        }
        if (best_steps != 0 && best_error <= config.corner_tolerance_px_) {
            diff.kind_ = DiffKind::kRotationMismatch;
            diff.rotation_steps_ = best_steps;
            diff.corner_error_px_ = best_error;
        } else {
            diff.kind_ = DiffKind::kCornerShift;
            diff.corner_error_px_ = aligned_error;
        }
        comparison.diffs_.push_back(diff);
    }

    for (std::size_t i = 0; i < baseline.size(); ++i) {
        if (baseline_matched[i]) {
            continue;
        }
        const auto center = centroid(baseline[i]);
        Diff diff;
        diff.kind_ = DiffKind::kMissed;
        diff.baseline_id_ = baseline[i].id_;
        diff.center_x_px_ = center.first;
        diff.center_y_px_ = center.second;
        comparison.diffs_.push_back(diff);
    }
    for (std::size_t j = 0; j < target.size(); ++j) {
        if (target_matched[j]) {
            continue;
        }
        const auto center = centroid(target[j]);
        Diff diff;
        diff.kind_ = DiffKind::kExtra;
        diff.target_id_ = target[j].id_;
        diff.center_x_px_ = center.first;
        diff.center_y_px_ = center.second;
        comparison.diffs_.push_back(diff);
    }
    return comparison;
}

Summary summarize(const std::vector<ImageComparison>& comparisons) {
    Summary summary;
    summary.image_count_ = comparisons.size();
    for (const ImageComparison& comparison : comparisons) {
        if (comparison.agrees()) {
            ++summary.agreed_image_count_;
        }
        summary.baseline_detection_count_ += comparison.baseline_count_;
        summary.target_detection_count_ += comparison.target_count_;
        summary.agreed_detection_count_ += comparison.agreed_count_;
        summary.worst_corner_error_px_ =
                std::max(summary.worst_corner_error_px_, comparison.worst_corner_error_px_);
        for (const Diff& diff : comparison.diffs_) {
            summary.kind_counts_[static_cast<std::size_t>(diff.kind_)] += 1U;
        }
    }
    return summary;
}

void write_text_report(std::ostream& out, const std::vector<ImageComparison>& comparisons,
                       const Summary& summary) {
    out << "=== Difference report ===\n";
    out << "images " << summary.image_count_ << " (agreeing " << summary.agreed_image_count_
        << ")\n";
    out << "detections baseline " << summary.baseline_detection_count_ << " / target "
        << summary.target_detection_count_ << " / agreed " << summary.agreed_detection_count_
        << "\n";
    out << "worst corner difference " << summary.worst_corner_error_px_ << " px\n";
    out << "\nCounts by kind\n";
    const DiffKind kinds[] = {DiffKind::kMissed, DiffKind::kExtra, DiffKind::kIdMismatch,
                              DiffKind::kRotationMismatch, DiffKind::kCornerShift};
    for (const DiffKind kind : kinds) {
        out << "  " << diff_kind_name(kind) << " "
            << summary.kind_counts_[static_cast<std::size_t>(kind)] << "\n";
    }

    // List every image that has a difference. Showing only the results with few
    // differences would misrepresent the overall trend.
    bool any = false;
    for (const ImageComparison& comparison : comparisons) {
        if (comparison.agrees()) {
            continue;
        }
        if (!any) {
            out << "\nImages with differences\n";
            any = true;
        }
        out << "  " << comparison.image_path_ << " (baseline " << comparison.baseline_count_
            << " target " << comparison.target_count_ << ")\n";
        for (const Diff& diff : comparison.diffs_) {
            out << "    " << diff_kind_name(diff.kind_) << " at (" << diff.center_x_px_ << ", "
                << diff.center_y_px_ << ")";
            if (diff.kind_ == DiffKind::kIdMismatch) {
                out << " baseline id=" << diff.baseline_id_ << " target id=" << diff.target_id_;
            } else if (diff.kind_ == DiffKind::kMissed) {
                out << " id=" << diff.baseline_id_;
            } else if (diff.kind_ == DiffKind::kExtra) {
                out << " id=" << diff.target_id_;
            } else {
                out << " id=" << diff.baseline_id_ << " difference " << diff.corner_error_px_
                    << " px";
                if (diff.kind_ == DiffKind::kRotationMismatch) {
                    out << " (rotated " << diff.rotation_steps_ << " steps)";
                }
            }
            out << '\n';
        }
    }
    if (!any) {
        out << "\nNo differences.\n";
    }
}

void write_json_report(std::ostream& out, const std::vector<ImageComparison>& comparisons,
                       const Summary& summary, const CompareConfig& config) {
    aruco3cuda::util::JsonWriter writer(out);
    writer.begin_object();
    writer.member_int("schema_version", 1);
    writer.key("config");
    writer.begin_object();
    writer.member_double("cornerTolerancePx", config.corner_tolerance_px_, 6);
    writer.member_double("matchRadiusRatio", config.match_radius_ratio_, 6);
    writer.end_object();

    writer.key("summary");
    writer.begin_object();
    writer.member_int("imageCount", static_cast<long long>(summary.image_count_));
    writer.member_int("agreedImageCount", static_cast<long long>(summary.agreed_image_count_));
    writer.member_int("baselineDetectionCount",
                      static_cast<long long>(summary.baseline_detection_count_));
    writer.member_int("targetDetectionCount",
                      static_cast<long long>(summary.target_detection_count_));
    writer.member_int("agreedDetectionCount",
                      static_cast<long long>(summary.agreed_detection_count_));
    writer.member_double("worstCornerErrorPx", summary.worst_corner_error_px_, 6);
    writer.key("kindCounts");
    writer.begin_object();
    const DiffKind kinds[] = {DiffKind::kMissed, DiffKind::kExtra, DiffKind::kIdMismatch,
                              DiffKind::kRotationMismatch, DiffKind::kCornerShift};
    const char* keys[] = {"missed", "extra", "idMismatch", "rotationMismatch", "cornerShift"};
    for (std::size_t k = 0; k < 5U; ++k) {
        writer.member_int(
                keys[k],
                static_cast<long long>(summary.kind_counts_[static_cast<std::size_t>(kinds[k])]));
    }
    writer.end_object();
    writer.end_object();

    writer.key("images");
    writer.begin_array();
    for (const ImageComparison& comparison : comparisons) {
        writer.begin_object();
        writer.member_string("path", comparison.image_path_);
        writer.member_int("baselineCount", static_cast<long long>(comparison.baseline_count_));
        writer.member_int("targetCount", static_cast<long long>(comparison.target_count_));
        writer.member_int("agreedCount", static_cast<long long>(comparison.agreed_count_));
        writer.member_double("worstCornerErrorPx", comparison.worst_corner_error_px_, 6);
        writer.key("diffs");
        writer.begin_array();
        for (const Diff& diff : comparison.diffs_) {
            writer.begin_object();
            writer.member_string("kind", diff_kind_name(diff.kind_));
            writer.member_int("baselineId", diff.baseline_id_);
            writer.member_int("targetId", diff.target_id_);
            writer.member_int("rotationSteps", diff.rotation_steps_);
            writer.member_double("cornerErrorPx", diff.corner_error_px_, 6);
            writer.member_double("centerXPx", diff.center_x_px_, 6);
            writer.member_double("centerYPx", diff.center_y_px_, 6);
            writer.end_object();
        }
        writer.end_array();
        writer.end_object();
    }
    writer.end_array();
    writer.end_object();
    out << '\n';
}

}  // namespace aruco3cuda::report
