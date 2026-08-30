// SPDX-License-Identifier: Apache-2.0
//
// Verifies how the diff report classifies differences.
//
// Purpose:
//   Confirm that every kind of difference falls into the intended class. A wrong
//   classification collapses corner shifts and missed detections into the same
//   number, which makes it impossible to isolate the cause.
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

/// Builds a detection whose corners form an axis-aligned square.
Detection make_square(int id, double x, double y, double side) {
    Detection detection;
    detection.id_ = id;
    const std::array<double, 8> corners = {x, y, x + side, y, x + side, y + side, x, y + side};
    detection.corners_ = corners;
    return detection;
}

/// Builds a detection whose four corners are rotated by `steps` positions.
Detection rotate_corners(const Detection& source, int steps) {
    Detection detection = source;
    for (std::size_t c = 0; c < 4U; ++c) {
        const std::size_t shifted = (c + static_cast<std::size_t>(steps)) % 4U;
        detection.corners_[shifted * 2U] = source.corners_[c * 2U];
        detection.corners_[(shifted * 2U) + 1U] = source.corners_[(c * 2U) + 1U];
    }
    return detection;
}

// Nominal: identical detections produce no difference.
TEST(ReportDiffTest, identical_results_produce_no_diff) {
    const std::vector<Detection> detections = {make_square(7, 100.0, 100.0, 80.0),
                                               make_square(9, 400.0, 300.0, 80.0)};
    const auto comparison =
            aruco3cuda::report::compare_detections("a.png", detections, detections, {});
    EXPECT_TRUE(comparison.agrees());
    EXPECT_EQ(comparison.agreed_count_, 2U);
    EXPECT_EQ(comparison.worst_corner_error_px_, 0.0);
}

// Nominal: a detection present in the baseline but absent from the target counts as missed.
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

// Nominal: a detection present in the target but absent from the baseline counts as extra.
TEST(ReportDiffTest, classifies_extra_detection) {
    const std::vector<Detection> baseline = {make_square(7, 100.0, 100.0, 80.0)};
    const std::vector<Detection> target = {baseline[0], make_square(9, 400.0, 300.0, 80.0)};
    const auto comparison = aruco3cuda::report::compare_detections("a.png", baseline, target, {});
    ASSERT_EQ(comparison.diffs_.size(), 1U);
    EXPECT_EQ(comparison.diffs_[0].kind_, DiffKind::kExtra);
    EXPECT_EQ(comparison.diffs_[0].target_id_, 9);
}

// Nominal: same position with a different ID is an ID mismatch. It is not split
// into a missed plus an extra detection.
TEST(ReportDiffTest, classifies_id_mismatch) {
    const std::vector<Detection> baseline = {make_square(7, 100.0, 100.0, 80.0)};
    const std::vector<Detection> target = {make_square(8, 100.0, 100.0, 80.0)};
    const auto comparison = aruco3cuda::report::compare_detections("a.png", baseline, target, {});
    ASSERT_EQ(comparison.diffs_.size(), 1U);
    EXPECT_EQ(comparison.diffs_[0].kind_, DiffKind::kIdMismatch);
    EXPECT_EQ(comparison.diffs_[0].baseline_id_, 7);
    EXPECT_EQ(comparison.diffs_[0].target_id_, 8);
}

// Nominal: when only the order of the four corners is rotated, the result is a rotation mismatch.
TEST(ReportDiffTest, classifies_rotation_mismatch) {
    const Detection base = make_square(7, 100.0, 100.0, 80.0);
    const std::vector<Detection> baseline = {base};
    const std::vector<Detection> target = {rotate_corners(base, 1)};
    const auto comparison = aruco3cuda::report::compare_detections("a.png", baseline, target, {});
    ASSERT_EQ(comparison.diffs_.size(), 1U);
    EXPECT_EQ(comparison.diffs_[0].kind_, DiffKind::kRotationMismatch);
    // The direction of the rotation is expressed as the number of steps the target
    // must be shifted by to return to the baseline.
    EXPECT_EQ(comparison.diffs_[0].rotation_steps_, 1);
    EXPECT_EQ(comparison.diffs_[0].corner_error_px_, 0.0);
}

// Nominal: a corner displacement beyond the tolerance is reported as a corner shift.
TEST(ReportDiffTest, classifies_corner_shift) {
    const std::vector<Detection> baseline = {make_square(7, 100.0, 100.0, 80.0)};
    const std::vector<Detection> target = {make_square(7, 103.0, 100.0, 80.0)};
    const auto comparison = aruco3cuda::report::compare_detections("a.png", baseline, target, {});
    ASSERT_EQ(comparison.diffs_.size(), 1U);
    EXPECT_EQ(comparison.diffs_[0].kind_, DiffKind::kCornerShift);
    EXPECT_NEAR(comparison.diffs_[0].corner_error_px_, 3.0, 1e-9);
}

// Boundary: a displacement exactly at the tolerance is treated as agreement.
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

// Boundary: detections farther apart than the matching radius are not taken to be the same marker.
TEST(ReportDiffTest, distant_detections_are_not_matched) {
    const std::vector<Detection> baseline = {make_square(7, 100.0, 100.0, 80.0)};
    const std::vector<Detection> target = {make_square(7, 200.0, 100.0, 80.0)};
    const auto comparison = aruco3cuda::report::compare_detections("a.png", baseline, target, {});
    ASSERT_EQ(comparison.diffs_.size(), 2U);
    EXPECT_EQ(comparison.diffs_[0].kind_, DiffKind::kMissed);
    EXPECT_EQ(comparison.diffs_[1].kind_, DiffKind::kExtra);
}

// Nominal: when several candidates are close, the nearest pair is matched first.
TEST(ReportDiffTest, closest_pair_is_matched_first) {
    const std::vector<Detection> baseline = {make_square(1, 100.0, 100.0, 80.0),
                                             make_square(2, 120.0, 100.0, 80.0)};
    // The target holds a single detection, near the position of baseline 2.
    // Baseline 1 is therefore reported as missed.
    const std::vector<Detection> target = {make_square(2, 121.0, 100.0, 80.0)};
    const auto comparison = aruco3cuda::report::compare_detections("a.png", baseline, target, {});
    ASSERT_EQ(comparison.diffs_.size(), 1U);
    EXPECT_EQ(comparison.diffs_[0].kind_, DiffKind::kMissed);
    EXPECT_EQ(comparison.diffs_[0].baseline_id_, 1);
    EXPECT_EQ(comparison.agreed_count_, 1U);
}

// Nominal: the summary counts each kind across all images.
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

// Nominal: the report lists every image that shows a difference.
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
    // Images that agree are not listed.
    EXPECT_EQ(text.find("b.png"), std::string::npos);
    EXPECT_NE(text.find("missed"), std::string::npos);
    EXPECT_NE(text.find("extra"), std::string::npos);
}

// Nominal: when there is no difference, the report says so.
TEST(ReportDiffTest, text_report_states_when_there_is_no_diff) {
    const std::vector<Detection> baseline = {make_square(7, 100.0, 100.0, 80.0)};
    const std::vector<aruco3cuda::report::ImageComparison> comparisons = {
            aruco3cuda::report::compare_detections("a.png", baseline, baseline, {})};
    std::ostringstream out;
    aruco3cuda::report::write_text_report(out, comparisons,
                                          aruco3cuda::report::summarize(comparisons));
    EXPECT_NE(out.str().find("No differences"), std::string::npos);
}

// Nominal: the JSON output carries the summary and the per-image breakdown.
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

// Boundary: a value outside the enumeration still resolves to a name.
TEST(ReportDiffTest, unknown_kind_has_a_name) {
    EXPECT_STREQ(aruco3cuda::report::diff_kind_name(static_cast<DiffKind>(99)), "unknown");
}

}  // namespace
