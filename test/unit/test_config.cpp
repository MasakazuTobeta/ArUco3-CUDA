// SPDX-License-Identifier: Apache-2.0
//
// 検出設定の境界検証を確認する。
//
// 範囲外の値をそのまま CUDA kernel の起動設定へ渡すと、無効な block 数や
// 負の反復回数として現れ、原因が設定であることが分かりにくくなる。
#include "aruco3cuda/config.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>

#include "aruco3cuda/status.hpp"

namespace {

using aruco3cuda::DetectorConfig;
using aruco3cuda::Status;

// 正常系: 既定値は有効である。
TEST(DetectorConfigTest, default_config_is_valid) {
    std::string message = "unchanged";
    EXPECT_EQ(DetectorConfig().validate(&message), Status::kOk) << message;
    // 成功時は message を変更しない。
    EXPECT_EQ(message, "unchanged");
    EXPECT_EQ(DetectorConfig().validate(nullptr), Status::kOk);
}

// 正常系: OpenCV 互換の既定値も有効であり、ArUco3 の既定が異なる。
TEST(DetectorConfigTest, opencv_defaults_are_valid_and_differ) {
    const DetectorConfig opencv = DetectorConfig::opencv_defaults();
    EXPECT_EQ(opencv.validate(nullptr), Status::kOk);
    EXPECT_FALSE(opencv.use_aruco3_detection_);
    EXPECT_FLOAT_EQ(opencv.min_marker_length_ratio_original_img_, 0.0F);
    EXPECT_EQ(opencv.corner_refine_method_, aruco3cuda::CornerRefineMethod::kNone);

    // 本 project の既定は ArUco3 検出戦略を有効にする。
    const DetectorConfig project;
    EXPECT_TRUE(project.use_aruco3_detection_);
    EXPECT_GT(project.min_marker_length_ratio_original_img_, 0.0F);
}

// 境界値: 適応的二値化の window は OpenCV の assert と同じ条件で弾く。
TEST(DetectorConfigTest, rejects_invalid_threshold_windows) {
    std::string message;

    DetectorConfig too_small;
    too_small.adaptive_thresh_win_size_min_px_ = 2;
    EXPECT_EQ(too_small.validate(&message), Status::kInvalidConfig);
    EXPECT_NE(message.find("adaptive_thresh_win_size_min_px"), std::string::npos) << message;

    // 下限そのものは受理する。
    DetectorConfig at_bound;
    at_bound.adaptive_thresh_win_size_min_px_ = 3;
    EXPECT_EQ(at_bound.validate(nullptr), Status::kOk);

    DetectorConfig zero_step;
    zero_step.adaptive_thresh_win_size_step_px_ = 0;
    EXPECT_EQ(zero_step.validate(&message), Status::kInvalidConfig);
    EXPECT_NE(message.find("adaptive_thresh_win_size_step_px"), std::string::npos) << message;

    DetectorConfig inverted;
    inverted.adaptive_thresh_win_size_min_px_ = 21;
    inverted.adaptive_thresh_win_size_max_px_ = 5;
    EXPECT_EQ(inverted.validate(&message), Status::kInvalidConfig);
    EXPECT_NE(message.find("矛盾"), std::string::npos) << message;

    // min と max が等しいのは window 1 種類であり有効。
    DetectorConfig single;
    single.adaptive_thresh_win_size_min_px_ = 11;
    single.adaptive_thresh_win_size_max_px_ = 11;
    EXPECT_EQ(single.validate(nullptr), Status::kOk);
}

// 境界値: OpenCV が assert する候補フィルタの条件。
TEST(DetectorConfigTest, rejects_invalid_candidate_filters) {
    std::string message;

    DetectorConfig zero_perimeter;
    zero_perimeter.min_marker_perimeter_rate_ = 0.0;
    EXPECT_EQ(zero_perimeter.validate(&message), Status::kInvalidConfig);
    EXPECT_NE(message.find("min_marker_perimeter_rate"), std::string::npos) << message;

    DetectorConfig inverted;
    inverted.min_marker_perimeter_rate_ = 1.0;
    inverted.max_marker_perimeter_rate_ = 0.5;
    EXPECT_EQ(inverted.validate(&message), Status::kInvalidConfig);

    DetectorConfig zero_accuracy;
    zero_accuracy.polygonal_approx_accuracy_rate_ = 0.0;
    EXPECT_EQ(zero_accuracy.validate(&message), Status::kInvalidConfig);
    EXPECT_NE(message.find("polygonal_approx_accuracy_rate"), std::string::npos) << message;

    DetectorConfig zero_border;
    zero_border.marker_border_bits_ = 0;
    EXPECT_EQ(zero_border.validate(&message), Status::kInvalidConfig);

    // OpenCV は cellMarginRate <= 0.5 を assert する。
    DetectorConfig wide_margin;
    wide_margin.perspective_remove_ignored_margin_per_cell_ = 0.6;
    EXPECT_EQ(wide_margin.validate(&message), Status::kInvalidConfig);
    DetectorConfig at_margin_bound;
    at_margin_bound.perspective_remove_ignored_margin_per_cell_ = 0.5;
    EXPECT_EQ(at_margin_bound.validate(nullptr), Status::kOk);
}

// 境界値: ArUco3 有効で縮小の指定が両方 0 は OpenCV が拒否する組み合わせ。
TEST(DetectorConfigTest, rejects_aruco3_with_both_zero) {
    DetectorConfig config;
    config.use_aruco3_detection_ = true;
    config.min_side_length_canonical_img_px_ = 0;
    config.min_marker_length_ratio_original_img_ = 0.0F;
    std::string message;
    EXPECT_EQ(config.validate(&message), Status::kInvalidConfig);
    EXPECT_NE(message.find("use_aruco3_detection"), std::string::npos) << message;

    // ArUco3 が無効なら同じ組み合わせでも問題ない。
    config.use_aruco3_detection_ = false;
    EXPECT_EQ(config.validate(nullptr), Status::kOk);
}

// 境界値: CUDA 固有の上限。検出数が候補数を超える設定は矛盾している。
TEST(DetectorConfigTest, rejects_invalid_cuda_limits) {
    std::string message;

    DetectorConfig zero_candidates;
    zero_candidates.max_candidates_ = 0;
    EXPECT_EQ(zero_candidates.validate(&message), Status::kInvalidConfig);
    EXPECT_NE(message.find("max_candidates"), std::string::npos) << message;

    DetectorConfig inverted;
    inverted.max_candidates_ = 100;
    inverted.max_markers_ = 200;
    EXPECT_EQ(inverted.validate(&message), Status::kInvalidConfig);
    EXPECT_NE(message.find("max_markers"), std::string::npos) << message;

    // 等しい場合は全候補が検出になり得るという意味であり有効。
    DetectorConfig equal;
    equal.max_candidates_ = 100;
    equal.max_markers_ = 100;
    EXPECT_EQ(equal.validate(nullptr), Status::kOk);

    DetectorConfig zero_size;
    zero_size.max_width_px_ = 0;
    EXPECT_EQ(zero_size.validate(&message), Status::kInvalidConfig);
}

// 異常系: NaN を弾く。比較が全て false になるため見落としやすい。
TEST(DetectorConfigTest, rejects_nan_values) {
    std::string message;
    DetectorConfig nan_otsu;
    nan_otsu.min_otsu_std_dev_ = std::nan("");
    EXPECT_EQ(nan_otsu.validate(&message), Status::kInvalidConfig);
    EXPECT_NE(message.find("min_otsu_std_dev"), std::string::npos) << message;

    DetectorConfig nan_ratio;
    nan_ratio.min_marker_length_ratio_original_img_ = std::nanf("");
    EXPECT_EQ(nan_ratio.validate(&message), Status::kInvalidConfig);
}

// 異常系: 無限大も弾く。
TEST(DetectorConfigTest, rejects_infinite_values) {
    DetectorConfig config;
    config.error_correction_rate_ = std::numeric_limits<double>::infinity();
    EXPECT_EQ(config.validate(nullptr), Status::kInvalidConfig);
}

// 正常系: 四隅補正方式の識別子。
TEST(CornerRefineMethodTest, identifiers_are_stable) {
    EXPECT_STREQ(aruco3cuda::to_string(aruco3cuda::CornerRefineMethod::kNone), "kNone");
    EXPECT_STREQ(aruco3cuda::to_string(aruco3cuda::CornerRefineMethod::kSubpix), "kSubpix");
    EXPECT_STREQ(aruco3cuda::to_string(static_cast<aruco3cuda::CornerRefineMethod>(999)),
                 "kUnknown");
}

}  // namespace
