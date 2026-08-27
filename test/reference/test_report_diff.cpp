// SPDX-License-Identifier: Apache-2.0
//
// 差分レポートの分類を検証する。
//
// 目的:
//   差異の種類ごとに、意図した分類になることを確かめる。分類を誤ると、
//   四隅のずれと取りこぼしが同じ数値へ丸められ、原因の切り分けができない。
#include "report_diff.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

namespace {

using aruco3cuda::report::CompareConfig;
using aruco3cuda::report::Detection;
using aruco3cuda::report::DiffKind;

/// 軸に平行な正方形の検出を作る。
Detection make_square(int id, double x, double y, double side) {
    Detection detection;
    detection.id_ = id;
    const std::array<double, 8> corners = {x, y, x + side, y, x + side, y + side, x, y + side};
    detection.corners_ = corners;
    return detection;
}

/// 四隅を steps 段巡回させた検出を作る。
Detection rotate_corners(const Detection& source, int steps) {
    Detection detection = source;
    for (std::size_t c = 0; c < 4U; ++c) {
        const std::size_t shifted = (c + static_cast<std::size_t>(steps)) % 4U;
        detection.corners_[shifted * 2U] = source.corners_[c * 2U];
        detection.corners_[(shifted * 2U) + 1U] = source.corners_[(c * 2U) + 1U];
    }
    return detection;
}

// 正常系: 同じ検出どうしでは差異が出ない。
TEST(ReportDiffTest, identical_results_produce_no_diff) {
    const std::vector<Detection> detections = {make_square(7, 100.0, 100.0, 80.0),
                                               make_square(9, 400.0, 300.0, 80.0)};
    const auto comparison =
            aruco3cuda::report::compare_detections("a.png", detections, detections, {});
    EXPECT_TRUE(comparison.agrees());
    EXPECT_EQ(comparison.agreed_count_, 2U);
    EXPECT_EQ(comparison.worst_corner_error_px_, 0.0);
}

// 正常系: 基準にあり対象に無い検出は未検出になる。
TEST(ReportDiffTest, classifies_missed_detection) {
    const std::vector<Detection> baseline = {make_square(7, 100.0, 100.0, 80.0),
                                             make_square(9, 400.0, 300.0, 80.0)};
    const std::vector<Detection> target = {baseline[0]};
    const auto comparison = aruco3cuda::report::compare_detections("a.png", baseline, target, {});
    ASSERT_EQ(comparison.diffs_.size(), 1U);
    EXPECT_EQ(comparison.diffs_[0].kind_, DiffKind::kMissed);
    EXPECT_EQ(comparison.diffs_[0].baseline_id_, 9);
    EXPECT_EQ(comparison.agreed_count_, 1U);
}

// 正常系: 対象にあり基準に無い検出は過検出になる。
TEST(ReportDiffTest, classifies_extra_detection) {
    const std::vector<Detection> baseline = {make_square(7, 100.0, 100.0, 80.0)};
    const std::vector<Detection> target = {baseline[0], make_square(9, 400.0, 300.0, 80.0)};
    const auto comparison = aruco3cuda::report::compare_detections("a.png", baseline, target, {});
    ASSERT_EQ(comparison.diffs_.size(), 1U);
    EXPECT_EQ(comparison.diffs_[0].kind_, DiffKind::kExtra);
    EXPECT_EQ(comparison.diffs_[0].target_id_, 9);
}

// 正常系: 同じ位置で ID が違う場合は ID 不一致になる。未検出と過検出には分けない。
TEST(ReportDiffTest, classifies_id_mismatch) {
    const std::vector<Detection> baseline = {make_square(7, 100.0, 100.0, 80.0)};
    const std::vector<Detection> target = {make_square(8, 100.0, 100.0, 80.0)};
    const auto comparison = aruco3cuda::report::compare_detections("a.png", baseline, target, {});
    ASSERT_EQ(comparison.diffs_.size(), 1U);
    EXPECT_EQ(comparison.diffs_[0].kind_, DiffKind::kIdMismatch);
    EXPECT_EQ(comparison.diffs_[0].baseline_id_, 7);
    EXPECT_EQ(comparison.diffs_[0].target_id_, 8);
}

// 正常系: 四隅の並びだけが巡回している場合は rotation 不一致になる。
TEST(ReportDiffTest, classifies_rotation_mismatch) {
    const Detection base = make_square(7, 100.0, 100.0, 80.0);
    const std::vector<Detection> baseline = {base};
    const std::vector<Detection> target = {rotate_corners(base, 1)};
    const auto comparison = aruco3cuda::report::compare_detections("a.png", baseline, target, {});
    ASSERT_EQ(comparison.diffs_.size(), 1U);
    EXPECT_EQ(comparison.diffs_[0].kind_, DiffKind::kRotationMismatch);
    // 巡回の向きは、対象を何段ずらすと基準に戻るかで表す。
    EXPECT_EQ(comparison.diffs_[0].rotation_steps_, 1);
    EXPECT_EQ(comparison.diffs_[0].corner_error_px_, 0.0);
}

// 正常系: 許容差を超える四隅のずれは四隅ずれになる。
TEST(ReportDiffTest, classifies_corner_shift) {
    const std::vector<Detection> baseline = {make_square(7, 100.0, 100.0, 80.0)};
    const std::vector<Detection> target = {make_square(7, 103.0, 100.0, 80.0)};
    const auto comparison = aruco3cuda::report::compare_detections("a.png", baseline, target, {});
    ASSERT_EQ(comparison.diffs_.size(), 1U);
    EXPECT_EQ(comparison.diffs_[0].kind_, DiffKind::kCornerShift);
    EXPECT_NEAR(comparison.diffs_[0].corner_error_px_, 3.0, 1e-9);
}

// 境界値: 許容差ちょうどのずれは一致として扱う。
TEST(ReportDiffTest, tolerance_boundary_is_inclusive) {
    const std::vector<Detection> baseline = {make_square(7, 100.0, 100.0, 80.0)};
    const std::vector<Detection> target = {make_square(7, 101.0, 100.0, 80.0)};
    CompareConfig config;
    config.corner_tolerance_px_ = 1.0;
    const auto comparison =
            aruco3cuda::report::compare_detections("a.png", baseline, target, config);
    EXPECT_TRUE(comparison.agrees());
    EXPECT_NEAR(comparison.worst_corner_error_px_, 1.0, 1e-9);
}

// 境界値: 対応付けの半径を超えて離れた検出は同じマーカーとみなさない。
TEST(ReportDiffTest, distant_detections_are_not_matched) {
    const std::vector<Detection> baseline = {make_square(7, 100.0, 100.0, 80.0)};
    const std::vector<Detection> target = {make_square(7, 200.0, 100.0, 80.0)};
    const auto comparison = aruco3cuda::report::compare_detections("a.png", baseline, target, {});
    ASSERT_EQ(comparison.diffs_.size(), 2U);
    EXPECT_EQ(comparison.diffs_[0].kind_, DiffKind::kMissed);
    EXPECT_EQ(comparison.diffs_[1].kind_, DiffKind::kExtra);
}

// 正常系: 近い候補が複数ある場合、より近い組から対応が確定する。
TEST(ReportDiffTest, closest_pair_is_matched_first) {
    const std::vector<Detection> baseline = {make_square(1, 100.0, 100.0, 80.0),
                                             make_square(2, 120.0, 100.0, 80.0)};
    // 対象は基準 2 の位置に近い 1 件のみ。基準 1 は未検出になる。
    const std::vector<Detection> target = {make_square(2, 121.0, 100.0, 80.0)};
    const auto comparison = aruco3cuda::report::compare_detections("a.png", baseline, target, {});
    ASSERT_EQ(comparison.diffs_.size(), 1U);
    EXPECT_EQ(comparison.diffs_[0].kind_, DiffKind::kMissed);
    EXPECT_EQ(comparison.diffs_[0].baseline_id_, 1);
    EXPECT_EQ(comparison.agreed_count_, 1U);
}

// 正常系: 集計は画像をまたいで種類ごとに数える。
TEST(ReportDiffTest, summary_counts_each_kind) {
    const std::vector<Detection> baseline = {make_square(7, 100.0, 100.0, 80.0)};
    std::vector<aruco3cuda::report::ImageComparison> comparisons;
    comparisons.push_back(aruco3cuda::report::compare_detections("a.png", baseline, baseline, {}));
    comparisons.push_back(aruco3cuda::report::compare_detections("b.png", baseline, {}, {}));
    comparisons.push_back(aruco3cuda::report::compare_detections("c.png", {}, baseline, {}));

    const auto summary = aruco3cuda::report::summarize(comparisons);
    EXPECT_EQ(summary.image_count_, 3U);
    EXPECT_EQ(summary.agreed_image_count_, 1U);
    EXPECT_EQ(summary.baseline_detection_count_, 2U);
    EXPECT_EQ(summary.target_detection_count_, 2U);
    EXPECT_EQ(summary.agreed_detection_count_, 1U);
    EXPECT_EQ(summary.kind_counts_[static_cast<std::size_t>(DiffKind::kMissed)], 1U);
    EXPECT_EQ(summary.kind_counts_[static_cast<std::size_t>(DiffKind::kExtra)], 1U);
}

// 正常系: 報告は差異のある画像を全て列挙する。
TEST(ReportDiffTest, text_report_lists_every_differing_image) {
    const std::vector<Detection> baseline = {make_square(7, 100.0, 100.0, 80.0)};
    std::vector<aruco3cuda::report::ImageComparison> comparisons;
    comparisons.push_back(aruco3cuda::report::compare_detections("a.png", baseline, {}, {}));
    comparisons.push_back(aruco3cuda::report::compare_detections("b.png", baseline, baseline, {}));
    comparisons.push_back(aruco3cuda::report::compare_detections("c.png", {}, baseline, {}));

    std::ostringstream out;
    aruco3cuda::report::write_text_report(out, comparisons,
                                          aruco3cuda::report::summarize(comparisons));
    const std::string text = out.str();
    EXPECT_NE(text.find("a.png"), std::string::npos);
    EXPECT_NE(text.find("c.png"), std::string::npos);
    // 一致した画像は列挙しない。
    EXPECT_EQ(text.find("b.png"), std::string::npos);
    EXPECT_NE(text.find("未検出"), std::string::npos);
    EXPECT_NE(text.find("過検出"), std::string::npos);
}

// 正常系: 差異が無い場合はその旨を示す。
TEST(ReportDiffTest, text_report_states_when_there_is_no_diff) {
    const std::vector<Detection> baseline = {make_square(7, 100.0, 100.0, 80.0)};
    const std::vector<aruco3cuda::report::ImageComparison> comparisons = {
            aruco3cuda::report::compare_detections("a.png", baseline, baseline, {})};
    std::ostringstream out;
    aruco3cuda::report::write_text_report(out, comparisons,
                                          aruco3cuda::report::summarize(comparisons));
    EXPECT_NE(out.str().find("差異は無い"), std::string::npos);
}

// 正常系: JSON 出力に集計と画像ごとの内訳が含まれる。
TEST(ReportDiffTest, json_report_contains_summary_and_images) {
    const std::vector<Detection> baseline = {make_square(7, 100.0, 100.0, 80.0)};
    const std::vector<aruco3cuda::report::ImageComparison> comparisons = {
            aruco3cuda::report::compare_detections("a.png", baseline, {}, {})};
    std::ostringstream out;
    aruco3cuda::report::write_json_report(out, comparisons,
                                          aruco3cuda::report::summarize(comparisons), {});
    const std::string json = out.str();
    EXPECT_NE(json.find("\"schema_version\": 1"), std::string::npos);
    EXPECT_NE(json.find("\"missed\": 1"), std::string::npos);
    EXPECT_NE(json.find("\"a.png\""), std::string::npos);
    EXPECT_NE(json.find("\"cornerTolerancePx\""), std::string::npos);
}

// 境界値: 列挙に無い値でも名前解決が破綻しない。
TEST(ReportDiffTest, unknown_kind_has_a_name) {
    EXPECT_STREQ(aruco3cuda::report::diff_kind_name(static_cast<DiffKind>(99)), "不明");
}

}  // namespace
