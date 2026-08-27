// SPDX-License-Identifier: Apache-2.0
#include "aruco3cuda/config.hpp"

#include <string>

#include "aruco3cuda/status.hpp"

namespace aruco3cuda {
namespace {

/// 整数項目の有効範囲。
struct IntRange {
    const char* name;
    int value;
    int minimum;
    int maximum;
};

/// 実数項目の有効範囲。
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
    *out_message = std::string("設定値が範囲外: ") + name + "=" + value + " (有効範囲 " + minimum +
                   " から " + maximum + ")";
}

void set_conflict_message(std::string* out_message, const std::string& detail) {
    if (out_message != nullptr) {
        *out_message = "設定値が矛盾: " + detail;
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
    // OpenCV は ArUco3 が無効なら補正しないことを既定とする。
    config.corner_refine_method_ = CornerRefineMethod::kNone;
    return config;
}

Status DetectorConfig::validate(std::string* out_message) const {
    // 範囲は OpenCV の assert と、CUDA の起動設定として意味を成す値から決める。
    // 上限は外部入力を信頼しないためのものであり、性能上の制約ではない。
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
            {"relative_corner_refinement_win_size", this->relative_corner_refinement_win_size_, 0.0,
             4.0},
            {"perspective_remove_ignored_margin_per_cell",
             this->perspective_remove_ignored_margin_per_cell_, 0.0, 0.5},
            {"max_erroneous_bits_in_border_rate", this->max_erroneous_bits_in_border_rate_, 0.0,
             1.0},
            {"min_otsu_std_dev", this->min_otsu_std_dev_, 0.0, 255.0},
            {"error_correction_rate", this->error_correction_rate_, 0.0, 1.0},
            {"corner_refinement_min_accuracy_px", this->corner_refinement_min_accuracy_px_, 0.0,
             16.0},
            {"min_marker_length_ratio_original_img",
             static_cast<double>(this->min_marker_length_ratio_original_img_), 0.0, 1.0},
    };
    for (const DoubleRange& range : double_ranges) {
        // NaN は比較が全て false になるため、この書き方で同時に弾ける。
        if (!(range.value >= range.minimum) || !(range.value <= range.maximum)) {
            set_range_message(out_message, range.name, std::to_string(range.value),
                              std::to_string(range.minimum), std::to_string(range.maximum));
            return Status::kInvalidConfig;
        }
    }

    // OpenCV が assert する条件。順序が逆だと window の走査が成立しない。
    if (this->adaptive_thresh_win_size_max_px_ < this->adaptive_thresh_win_size_min_px_) {
        set_conflict_message(out_message,
                             "adaptive_thresh_win_size_max_px=" +
                                     std::to_string(this->adaptive_thresh_win_size_max_px_) +
                                     " が adaptive_thresh_win_size_min_px=" +
                                     std::to_string(this->adaptive_thresh_win_size_min_px_) +
                                     " より小さい");
        return Status::kInvalidConfig;
    }
    // OpenCV は minPerimeterRate > 0 を assert する。0 は候補を絞れない。
    if (!(this->min_marker_perimeter_rate_ > 0.0)) {
        set_conflict_message(out_message, "min_marker_perimeter_rate は 0 より大きい必要がある");
        return Status::kInvalidConfig;
    }
    if (this->max_marker_perimeter_rate_ <= this->min_marker_perimeter_rate_) {
        set_conflict_message(out_message,
                             "max_marker_perimeter_rate が min_marker_perimeter_rate 以下");
        return Status::kInvalidConfig;
    }
    if (!(this->polygonal_approx_accuracy_rate_ > 0.0)) {
        set_conflict_message(out_message,
                             "polygonal_approx_accuracy_rate は 0 より大きい必要がある");
        return Status::kInvalidConfig;
    }
    // OpenCV は ArUco3 有効時にこの組み合わせを assert で拒否する。
    if (this->use_aruco3_detection_ && this->min_side_length_canonical_img_px_ == 0 &&
        this->min_marker_length_ratio_original_img_ == 0.0F) {
        set_conflict_message(out_message,
                             "use_aruco3_detection が有効で min_side_length_canonical_img_px と "
                             "min_marker_length_ratio_original_img が共に 0");
        return Status::kInvalidConfig;
    }
    // window 数が上限を超えると二値化画像の保持量が際限なく増える。
    const int window_count =
            (this->adaptive_thresh_win_size_max_px_ - this->adaptive_thresh_win_size_min_px_) /
                    this->adaptive_thresh_win_size_step_px_ +
            1;
    if (window_count > kMaxAdaptiveThresholdWindows) {
        set_conflict_message(out_message, "適応的二値化の window 数 " +
                                                  std::to_string(window_count) + " が上限 " +
                                                  std::to_string(kMaxAdaptiveThresholdWindows) +
                                                  " を超える");
        return Status::kInvalidConfig;
    }

    // block 1 辺の thread 数は 2 の冪でなければ warp の割り当てが崩れる。
    if ((this->cuda_block_dim_ & (this->cuda_block_dim_ - 1)) != 0) {
        set_conflict_message(out_message,
                             "cuda_block_dim=" + std::to_string(this->cuda_block_dim_) +
                                     " は 2 の冪である必要がある");
        return Status::kInvalidConfig;
    }

    // 検出数は候補数を超えられない。逆転していると上限の意味が失われる。
    if (this->max_markers_ > this->max_candidates_) {
        set_conflict_message(out_message,
                             "max_markers=" + std::to_string(this->max_markers_) +
                                     " が max_candidates=" + std::to_string(this->max_candidates_) +
                                     " を超えている");
        return Status::kInvalidConfig;
    }
    return Status::kOk;
}

}  // namespace aruco3cuda
