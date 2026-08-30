// SPDX-License-Identifier: Apache-2.0
#include "accuracy.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "geometry.hpp"

namespace aruco3cuda::evaluate {
namespace {

using aruco3cuda::report::quad_centroid;
using aruco3cuda::report::quad_squared_error;

/// One candidate pairing.
struct Pair {
    std::size_t truth_index_ = 0;
    std::size_t observed_index_ = 0;
    double distance_ = 0.0;
};

/// Determines by how many positions the corner ordering is rotated.
///
/// The corners of a true value are ordered in marker coordinates. A detection whose
/// ID was read correctly follows the same order, so a non-zero return means the
/// rotation was determined incorrectly.
int best_rotation_steps(const Quad& truth, const Quad& observed, double* out_squared) {
    int best_steps = 0;
    double best_squared = quad_squared_error(truth, observed, 0);
    for (int steps = 1; steps < 4; ++steps) {
        const double squared = quad_squared_error(truth, observed, steps);
        if (squared < best_squared) {
            best_squared = squared;
            best_steps = steps;
        }
    }
    *out_squared = best_squared;
    return best_steps;
}

}  // namespace

ImageAccuracy compare_to_truth(const std::string& image_name, const std::vector<TruthMarker>& truth,
                               const std::vector<Observation>& observed,
                               const MatchConfig& config) {
    ImageAccuracy accuracy;
    accuracy.image_name_ = image_name;
    accuracy.truth_count_ = truth.size();
    accuracy.observed_count_ = observed.size();

    std::vector<Pair> pairs;
    for (std::size_t i = 0; i < truth.size(); ++i) {
        const auto truth_center = quad_centroid(truth[i].corners_);
        const double radius = truth[i].side_px_ * config.match_radius_ratio_;
        for (std::size_t j = 0; j < observed.size(); ++j) {
            const auto observed_center = quad_centroid(observed[j].corners_);
            const double dx = truth_center.first - observed_center.first;
            const double dy = truth_center.second - observed_center.second;
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
        if (a.truth_index_ != b.truth_index_) {
            return a.truth_index_ < b.truth_index_;
        }
        return a.observed_index_ < b.observed_index_;
    });

    accuracy.truth_outcomes_.assign(truth.size(), TruthOutcome::kMissed);
    accuracy.truth_squared_error_px2_.assign(truth.size(), 0.0);
    accuracy.truth_corner_max_px_.assign(truth.size(), 0.0);
    accuracy.truth_rotation_agreed_.assign(truth.size(), false);

    std::vector<bool> truth_matched(truth.size(), false);
    std::vector<bool> observed_matched(observed.size(), false);
    for (const Pair& pair : pairs) {
        if (truth_matched[pair.truth_index_] || observed_matched[pair.observed_index_]) {
            continue;
        }
        truth_matched[pair.truth_index_] = true;
        observed_matched[pair.observed_index_] = true;

        const TruthMarker& expected = truth[pair.truth_index_];
        const Observation& actual = observed[pair.observed_index_];
        if (expected.id_ != actual.id_) {
            // The ID of the same marker was read incorrectly. Count it both as a
            // missed true value and as a wrong detection. Counting it on one side only
            // would let either precision or recall overlook the error.
            ++accuracy.false_positive_;
            ++accuracy.false_negative_;
            accuracy.truth_outcomes_[pair.truth_index_] = TruthOutcome::kIdMismatched;
            continue;
        }

        ++accuracy.true_positive_;
        double squared = 0.0;
        const int steps = best_rotation_steps(expected.corners_, actual.corners_, &squared);
        if (steps == 0) {
            ++accuracy.rotation_agreed_;
        }
        const double worst =
                aruco3cuda::report::quad_corner_error(expected.corners_, actual.corners_, steps);
        accuracy.corner_squared_sum_px2_ += squared;
        accuracy.corner_sample_count_ += aruco3cuda::report::kQuadCorners;
        accuracy.corner_max_px_ = std::max(accuracy.corner_max_px_, worst);
        accuracy.truth_outcomes_[pair.truth_index_] = TruthOutcome::kDetected;
        accuracy.truth_squared_error_px2_[pair.truth_index_] = squared;
        accuracy.truth_corner_max_px_[pair.truth_index_] = worst;
        accuracy.truth_rotation_agreed_[pair.truth_index_] = (steps == 0);
    }

    for (std::size_t i = 0; i < truth.size(); ++i) {
        if (!truth_matched[i]) {
            ++accuracy.false_negative_;
        }
    }
    for (std::size_t j = 0; j < observed.size(); ++j) {
        if (!observed_matched[j]) {
            ++accuracy.false_positive_;
        }
    }
    return accuracy;
}

void accumulate(const ImageAccuracy& image, AccuracySummary* out_summary) {
    if (out_summary == nullptr) {
        return;
    }
    ++out_summary->image_count_;
    out_summary->truth_count_ += image.truth_count_;
    out_summary->observed_count_ += image.observed_count_;
    out_summary->true_positive_ += image.true_positive_;
    out_summary->false_positive_ += image.false_positive_;
    out_summary->false_negative_ += image.false_negative_;
    out_summary->rotation_agreed_ += image.rotation_agreed_;
    out_summary->corner_squared_sum_px2_ += image.corner_squared_sum_px2_;
    out_summary->corner_sample_count_ += image.corner_sample_count_;
    out_summary->corner_max_px_ = std::max(out_summary->corner_max_px_, image.corner_max_px_);
}

void accumulate_selected(const ImageAccuracy& image, const std::vector<bool>& selected,
                         AccuracySummary* out_summary) {
    if (out_summary == nullptr || selected.size() != image.truth_outcomes_.size()) {
        return;
    }
    ++out_summary->image_count_;
    for (std::size_t i = 0; i < selected.size(); ++i) {
        if (!selected[i]) {
            continue;
        }
        ++out_summary->truth_count_;
        switch (image.truth_outcomes_[i]) {
            case TruthOutcome::kDetected:
                ++out_summary->true_positive_;
                if (image.truth_rotation_agreed_[i]) {
                    ++out_summary->rotation_agreed_;
                }
                out_summary->corner_squared_sum_px2_ += image.truth_squared_error_px2_[i];
                out_summary->corner_sample_count_ += aruco3cuda::report::kQuadCorners;
                out_summary->corner_max_px_ =
                        std::max(out_summary->corner_max_px_, image.truth_corner_max_px_[i]);
                break;
            case TruthOutcome::kIdMismatched:
                // A true value whose ID was misread lowers recall just as a miss does.
                // The corresponding false positive belongs to no true value and is not
                // accumulated.
                ++out_summary->false_negative_;
                break;
            case TruthOutcome::kMissed:
                ++out_summary->false_negative_;
                break;
        }
    }
}

bool precision(const AccuracySummary& summary, double* out_value) {
    if (out_value == nullptr) {
        return false;
    }
    const std::size_t denominator = summary.true_positive_ + summary.false_positive_;
    if (denominator == 0U) {
        return false;
    }
    *out_value = static_cast<double>(summary.true_positive_) / static_cast<double>(denominator);
    return true;
}

bool recall(const AccuracySummary& summary, double* out_value) {
    if (out_value == nullptr) {
        return false;
    }
    const std::size_t denominator = summary.true_positive_ + summary.false_negative_;
    if (denominator == 0U) {
        return false;
    }
    *out_value = static_cast<double>(summary.true_positive_) / static_cast<double>(denominator);
    return true;
}

bool corner_rmse_px(const AccuracySummary& summary, double* out_value) {
    if (out_value == nullptr || summary.corner_sample_count_ == 0U) {
        return false;
    }
    *out_value = std::sqrt(summary.corner_squared_sum_px2_ /
                           static_cast<double>(summary.corner_sample_count_));
    return true;
}

}  // namespace aruco3cuda::evaluate
