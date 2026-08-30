// SPDX-License-Identifier: Apache-2.0
#include "aruco3cuda/config.hpp"

#include <string>

#include "aruco3cuda/status.hpp"

namespace aruco3cuda {
namespace {

/// Valid range of an integer configuration item.
struct IntRange {
    const char* name;
    int value;
    int minimum;
    int maximum;
};

/// Valid range of a floating-point configuration item.
struct DoubleRange {
    const char* name;
    double value;
    double minimum;
    double maximum;
};

void set_range_message(std::string* out_message, const char* name, const std::string& value,
                       const std::string& minimum, const std::string& maximum) {
    if (out_message == nullptr) {
        return;
    }
    *out_message = std::string("configuration value out of range: ") + name + "=" + value +
                   " (valid range " + minimum + " to " + maximum + ")";
}

void set_conflict_message(std::string* out_message, const std::string& detail) {
    if (out_message != nullptr) {
        *out_message = "configuration values conflict: " + detail;
    }
}

}  // namespace

const char* to_string(CornerRefineMethod method) {
    switch (method) {
        case CornerRefineMethod::kNone:
            return "kNone";
        case CornerRefineMethod::kSubpix:
            return "kSubpix";
    }
    return "kUnknown";
}

DetectorConfig DetectorConfig::opencv_defaults() {
    DetectorConfig config;
    config.use_aruco3_detection_ = false;
    config.min_marker_length_ratio_original_img_ = 0.0F;
    // OpenCV defaults to performing no corner refinement when ArUco3 is disabled.
    config.corner_refine_method_ = CornerRefineMethod::kNone;
    return config;
}

Status DetectorConfig::validate(std::string* out_message) const {
    // The ranges come from what OpenCV asserts and from the values that still make sense as
    // CUDA launch settings. The upper bounds exist because external input is not trusted; they
    // are not performance limits.
    const IntRange int_ranges[] = {
            {"adaptive_thresh_win_size_min_px", this->adaptive_thresh_win_size_min_px_, 3, 4096},
            {"adaptive_thresh_win_size_max_px", this->adaptive_thresh_win_size_max_px_, 3, 4096},
            {"adaptive_thresh_win_size_step_px", this->adaptive_thresh_win_size_step_px_, 1, 4096},
            {"min_distance_to_border_px", this->min_distance_to_border_px_, 0, 4096},
            {"marker_border_bits", this->marker_border_bits_, 1, 16},
            {"perspective_remove_pixel_per_cell", this->perspective_remove_pixel_per_cell_, 1, 256},
            {"corner_refinement_win_size_px", this->corner_refinement_win_size_px_, 1, 256},
            {"corner_refinement_max_iterations", this->corner_refinement_max_iterations_, 1, 10000},
            {"min_side_length_canonical_img_px", this->min_side_length_canonical_img_px_, 0, 4096},
            {"max_candidates", this->max_candidates_, 1, 1 << 22},
            {"max_markers", this->max_markers_, 1, 1 << 22},
            {"max_width_px", this->max_width_px_, 1, 65536},
            {"max_height_px", this->max_height_px_, 1, 65536},
            {"cuda_block_dim", this->cuda_block_dim_, 4, 32},
    };
    for (const IntRange& range : int_ranges) {
        if (range.value < range.minimum || range.value > range.maximum) {
            set_range_message(out_message, range.name, std::to_string(range.value),
                              std::to_string(range.minimum), std::to_string(range.maximum));
            return Status::kInvalidConfig;
        }
    }

    const DoubleRange double_ranges[] = {
            {"adaptive_thresh_constant", this->adaptive_thresh_constant_, -255.0, 255.0},
            {"min_marker_perimeter_rate", this->min_marker_perimeter_rate_, 0.0, 8.0},
            {"max_marker_perimeter_rate", this->max_marker_perimeter_rate_, 0.0, 64.0},
            {"polygonal_approx_accuracy_rate", this->polygonal_approx_accuracy_rate_, 0.0, 1.0},
            {"min_corner_distance_rate", this->min_corner_distance_rate_, 0.0, 1.0},
            {"min_marker_distance_rate", this->min_marker_distance_rate_, 0.0, 4.0},
            {"min_group_distance", this->min_group_distance_, 0.0, 4.0},
            {"min_quad_inlier_ratio", this->min_quad_inlier_ratio_, 0.0, 1.0},
            {"min_edge_support_ratio", this->min_edge_support_ratio_, 0.0, 64.0},
            {"relative_corner_refinement_win_size", this->relative_corner_refinement_win_size_, 0.0,
             4.0},
            {"perspective_remove_ignored_margin_per_cell",
             this->perspective_remove_ignored_margin_per_cell_, 0.0, 0.5},
            {"max_erroneous_bits_in_border_rate", this->max_erroneous_bits_in_border_rate_, 0.0,
             1.0},
            {"min_otsu_std_dev", this->min_otsu_std_dev_, 0.0, 255.0},
            {"error_correction_rate", this->error_correction_rate_, 0.0, 1.0},
            {"valid_bit_threshold", this->valid_bit_threshold_, 0.0, 1.0},
            {"corner_refinement_min_accuracy_px", this->corner_refinement_min_accuracy_px_, 0.0,
             16.0},
            {"min_marker_length_ratio_original_img",
             static_cast<double>(this->min_marker_length_ratio_original_img_), 0.0, 1.0},
    };
    for (const DoubleRange& range : double_ranges) {
        // Every comparison against NaN is false, so written this way the check rejects NaN too.
        if (!(range.value >= range.minimum) || !(range.value <= range.maximum)) {
            set_range_message(out_message, range.name, std::to_string(range.value),
                              std::to_string(range.minimum), std::to_string(range.maximum));
            return Status::kInvalidConfig;
        }
    }

    // A condition OpenCV asserts. With the order reversed the window sweep cannot run.
    if (this->adaptive_thresh_win_size_max_px_ < this->adaptive_thresh_win_size_min_px_) {
        set_conflict_message(out_message,
                             "adaptive_thresh_win_size_max_px=" +
                                     std::to_string(this->adaptive_thresh_win_size_max_px_) +
                                     " is smaller than adaptive_thresh_win_size_min_px=" +
                                     std::to_string(this->adaptive_thresh_win_size_min_px_));
        return Status::kInvalidConfig;
    }
    // OpenCV asserts minPerimeterRate > 0. A value of 0 cannot narrow the candidates at all.
    if (!(this->min_marker_perimeter_rate_ > 0.0)) {
        set_conflict_message(out_message, "min_marker_perimeter_rate must be greater than 0");
        return Status::kInvalidConfig;
    }
    if (this->max_marker_perimeter_rate_ <= this->min_marker_perimeter_rate_) {
        set_conflict_message(out_message,
                             "max_marker_perimeter_rate is not greater than "
                             "min_marker_perimeter_rate");
        return Status::kInvalidConfig;
    }
    if (!(this->polygonal_approx_accuracy_rate_ > 0.0)) {
        set_conflict_message(out_message, "polygonal_approx_accuracy_rate must be greater than 0");
        return Status::kInvalidConfig;
    }
    // OpenCV rejects this combination with an assert when ArUco3 is enabled.
    if (this->use_aruco3_detection_ && this->min_side_length_canonical_img_px_ == 0 &&
        this->min_marker_length_ratio_original_img_ == 0.0F) {
        set_conflict_message(out_message,
                             "use_aruco3_detection is enabled while "
                             "min_side_length_canonical_img_px and "
                             "min_marker_length_ratio_original_img are both 0");
        return Status::kInvalidConfig;
    }
    // Once the window count passes the cap, the amount of binarized image kept in memory grows
    // without bound.
    const int window_count =
            (this->adaptive_thresh_win_size_max_px_ - this->adaptive_thresh_win_size_min_px_) /
                    this->adaptive_thresh_win_size_step_px_ +
            1;
    if (window_count > kMaxAdaptiveThresholdWindows) {
        set_conflict_message(out_message, "adaptive threshold window count " +
                                                  std::to_string(window_count) +
                                                  " exceeds the limit of " +
                                                  std::to_string(kMaxAdaptiveThresholdWindows));
        return Status::kInvalidConfig;
    }

    // The thread count along one block side must be a power of two, otherwise the warp
    // assignment falls apart.
    if ((this->cuda_block_dim_ & (this->cuda_block_dim_ - 1)) != 0) {
        set_conflict_message(out_message,
                             "cuda_block_dim=" + std::to_string(this->cuda_block_dim_) +
                                     " must be a power of two");
        return Status::kInvalidConfig;
    }

    // The marker count cannot exceed the candidate count. Reversed, the caps lose their meaning.
    if (this->max_markers_ > this->max_candidates_) {
        set_conflict_message(out_message, "max_markers=" + std::to_string(this->max_markers_) +
                                                  " exceeds max_candidates=" +
                                                  std::to_string(this->max_candidates_));
        return Status::kInvalidConfig;
    }
    return Status::kOk;
}

}  // namespace aruco3cuda
