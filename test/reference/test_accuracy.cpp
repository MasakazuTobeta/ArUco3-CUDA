// SPDX-License-Identifier: Apache-2.0
//
// Verification of the comparison against ground truth.
//
// Pins down the pairing rules, where a wrong ID is counted, and how metrics that
// cannot be defined are handled.
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "accuracy.hpp"

namespace {

using aruco3cuda::evaluate::AccuracySummary;
using aruco3cuda::evaluate::ImageAccuracy;
using aruco3cuda::evaluate::MatchConfig;
using aruco3cuda::evaluate::Observation;
using aruco3cuda::evaluate::TruthMarker;
using aruco3cuda::report::Quad;

/// Axis-aligned square centered at (cx, cy) with edge length side.
///
/// The order corresponds to (0,0), (S,0), (S,S), (0,S) in marker coordinates.
Quad square(double cx, double cy, double side) {
    const double half = side / 2.0;
    return Quad{cx - half, cy - half, cx + half, cy - half,
                cx + half, cy + half, cx - half, cy + half};
}

TruthMarker truth_at(int id, double cx, double cy, double side) {
    TruthMarker marker;
    marker.id_ = id;
    marker.corners_ = square(cx, cy, side);
    marker.side_px_ = side;
    return marker;
}

Observation observed_at(int id, double cx, double cy, double side) {
    Observation observation;
    observation.id_ = id;
    observation.corners_ = square(cx, cy, side);
    return observation;
}

}  // namespace

TEST(Accuracy, CountsExactMatchAsTruePositive) {
    const std::vector<TruthMarker> truth = {truth_at(7, 100.0, 100.0, 40.0)};
    const std::vector<Observation> observed = {observed_at(7, 100.0, 100.0, 40.0)};

    const ImageAccuracy result =
            aruco3cuda::evaluate::compare_to_truth("scene", truth, observed, MatchConfig{});

    EXPECT_EQ(result.truth_count_, 1U);
    EXPECT_EQ(result.observed_count_, 1U);
    EXPECT_EQ(result.true_positive_, 1U);
    EXPECT_EQ(result.false_positive_, 0U);
    EXPECT_EQ(result.false_negative_, 0U);
    EXPECT_EQ(result.rotation_agreed_, 1U);
    EXPECT_DOUBLE_EQ(result.corner_max_px_, 0.0);
    EXPECT_EQ(result.corner_sample_count_, 4U);
}

TEST(Accuracy, CountsWrongIdAsBothFalsePositiveAndFalseNegative) {
    // When the ID is wrong at the correct position, both precision and recall must
    // drop. Counting it on only one side would let that metric miss the error.
    const std::vector<TruthMarker> truth = {truth_at(7, 100.0, 100.0, 40.0)};
    const std::vector<Observation> observed = {observed_at(8, 100.0, 100.0, 40.0)};

    const ImageAccuracy result =
            aruco3cuda::evaluate::compare_to_truth("scene", truth, observed, MatchConfig{});

    EXPECT_EQ(result.true_positive_, 0U);
    EXPECT_EQ(result.false_positive_, 1U);
    EXPECT_EQ(result.false_negative_, 1U);
    EXPECT_EQ(result.corner_sample_count_, 0U);
}

TEST(Accuracy, CountsUnmatchedTruthAsFalseNegative) {
    const std::vector<TruthMarker> truth = {truth_at(7, 100.0, 100.0, 40.0)};
    const std::vector<Observation> observed;

    const ImageAccuracy result =
            aruco3cuda::evaluate::compare_to_truth("scene", truth, observed, MatchConfig{});

    EXPECT_EQ(result.false_negative_, 1U);
    EXPECT_EQ(result.false_positive_, 0U);
}

TEST(Accuracy, CountsUnmatchedObservationAsFalsePositive) {
    const std::vector<TruthMarker> truth;
    const std::vector<Observation> observed = {observed_at(7, 100.0, 100.0, 40.0)};

    const ImageAccuracy result =
            aruco3cuda::evaluate::compare_to_truth("scene", truth, observed, MatchConfig{});

    EXPECT_EQ(result.false_positive_, 1U);
    EXPECT_EQ(result.false_negative_, 0U);
}

TEST(Accuracy, RejectsMatchBeyondRadius) {
    // The radius is a ratio of the truth edge length. With edge 40 and the default
    // 0.5, centroid distances up to 20 pair up. This checks just outside that bound.
    const std::vector<TruthMarker> truth = {truth_at(7, 100.0, 100.0, 40.0)};
    const std::vector<Observation> observed = {observed_at(7, 121.0, 100.0, 40.0)};

    const ImageAccuracy result =
            aruco3cuda::evaluate::compare_to_truth("scene", truth, observed, MatchConfig{});

    EXPECT_EQ(result.true_positive_, 0U);
    EXPECT_EQ(result.false_positive_, 1U);
    EXPECT_EQ(result.false_negative_, 1U);
}

TEST(Accuracy, AcceptsMatchAtRadiusBoundary) {
    const std::vector<TruthMarker> truth = {truth_at(7, 100.0, 100.0, 40.0)};
    const std::vector<Observation> observed = {observed_at(7, 120.0, 100.0, 40.0)};

    const ImageAccuracy result =
            aruco3cuda::evaluate::compare_to_truth("scene", truth, observed, MatchConfig{});

    EXPECT_EQ(result.true_positive_, 1U);
    EXPECT_DOUBLE_EQ(result.corner_max_px_, 20.0);
}

TEST(Accuracy, DetectsRotatedCornerOrder) {
    // The ID is correct but the corner order is cyclically shifted by one. The
    // position still matches, so this is a true positive, but it does not count as
    // a rotation agreement.
    const Quad rotated{120.0, 80.0, 120.0, 120.0, 80.0, 120.0, 80.0, 80.0};
    const std::vector<TruthMarker> truth = {truth_at(7, 100.0, 100.0, 40.0)};
    std::vector<Observation> observed(1);
    observed[0].id_ = 7;
    observed[0].corners_ = rotated;

    const ImageAccuracy result =
            aruco3cuda::evaluate::compare_to_truth("scene", truth, observed, MatchConfig{});

    EXPECT_EQ(result.true_positive_, 1U);
    EXPECT_EQ(result.rotation_agreed_, 0U);
}

TEST(Accuracy, PairsNearestFirstWhenSeveralAreInRange) {
    // When two detections fall within the radius of the same truth marker, the
    // nearer one is the pair.
    const std::vector<TruthMarker> truth = {truth_at(7, 100.0, 100.0, 40.0)};
    const std::vector<Observation> observed = {observed_at(7, 118.0, 100.0, 40.0),
                                               observed_at(7, 102.0, 100.0, 40.0)};

    const ImageAccuracy result =
            aruco3cuda::evaluate::compare_to_truth("scene", truth, observed, MatchConfig{});

    EXPECT_EQ(result.true_positive_, 1U);
    EXPECT_EQ(result.false_positive_, 1U);
    EXPECT_DOUBLE_EQ(result.corner_max_px_, 2.0);
}

TEST(Accuracy, AccumulatesAcrossImages) {
    AccuracySummary summary;
    ImageAccuracy first;
    first.truth_count_ = 2U;
    first.observed_count_ = 2U;
    first.true_positive_ = 2U;
    first.rotation_agreed_ = 2U;
    first.corner_squared_sum_px2_ = 4.0;
    first.corner_sample_count_ = 8U;
    first.corner_max_px_ = 1.0;
    ImageAccuracy second;
    second.truth_count_ = 1U;
    second.false_negative_ = 1U;
    second.corner_max_px_ = 0.5;

    aruco3cuda::evaluate::accumulate(first, &summary);
    aruco3cuda::evaluate::accumulate(second, &summary);

    EXPECT_EQ(summary.image_count_, 2U);
    EXPECT_EQ(summary.truth_count_, 3U);
    EXPECT_EQ(summary.true_positive_, 2U);
    EXPECT_EQ(summary.false_negative_, 1U);
    EXPECT_DOUBLE_EQ(summary.corner_max_px_, 1.0);
}

TEST(Accuracy, IgnoresNullSummaryOnAccumulate) {
    const ImageAccuracy image;
    aruco3cuda::evaluate::accumulate(image, nullptr);
    SUCCEED();
}

TEST(Accuracy, RecordsPerTruthOutcomes) {
    const std::vector<TruthMarker> truth = {truth_at(7, 100.0, 100.0, 40.0),
                                            truth_at(8, 300.0, 100.0, 40.0),
                                            truth_at(9, 500.0, 100.0, 40.0)};
    const std::vector<Observation> observed = {observed_at(7, 100.0, 100.0, 40.0),
                                               observed_at(99, 300.0, 100.0, 40.0)};

    const ImageAccuracy result =
            aruco3cuda::evaluate::compare_to_truth("scene", truth, observed, MatchConfig{});

    ASSERT_EQ(result.truth_outcomes_.size(), 3U);
    EXPECT_EQ(result.truth_outcomes_[0], aruco3cuda::evaluate::TruthOutcome::kDetected);
    EXPECT_EQ(result.truth_outcomes_[1], aruco3cuda::evaluate::TruthOutcome::kIdMismatched);
    EXPECT_EQ(result.truth_outcomes_[2], aruco3cuda::evaluate::TruthOutcome::kMissed);
    EXPECT_TRUE(result.truth_rotation_agreed_[0]);
    EXPECT_FALSE(result.truth_rotation_agreed_[1]);
}

TEST(Accuracy, AccumulatesOnlySelectedTruth) {
    // Used to report recall broken down by marker size. Truth markers that were not
    // selected are left out of the counts.
    const std::vector<TruthMarker> truth = {truth_at(7, 100.0, 100.0, 40.0),
                                            truth_at(8, 300.0, 100.0, 40.0)};
    const std::vector<Observation> observed = {observed_at(7, 100.0, 100.0, 40.0)};
    const ImageAccuracy result =
            aruco3cuda::evaluate::compare_to_truth("scene", truth, observed, MatchConfig{});

    AccuracySummary detected_only;
    aruco3cuda::evaluate::accumulate_selected(result, {true, false}, &detected_only);
    EXPECT_EQ(detected_only.truth_count_, 1U);
    EXPECT_EQ(detected_only.true_positive_, 1U);
    EXPECT_EQ(detected_only.false_negative_, 0U);
    EXPECT_EQ(detected_only.rotation_agreed_, 1U);

    AccuracySummary missed_only;
    aruco3cuda::evaluate::accumulate_selected(result, {false, true}, &missed_only);
    EXPECT_EQ(missed_only.truth_count_, 1U);
    EXPECT_EQ(missed_only.true_positive_, 0U);
    EXPECT_EQ(missed_only.false_negative_, 1U);
}

TEST(Accuracy, LeavesPrecisionUndefinedAfterSelectedAccumulation) {
    // An unpaired detection belongs to no truth marker, so it is not accumulated.
    // As a consequence precision cannot be derived from this summary. This test pins
    // that down so a reader does not misread the numbers.
    const std::vector<TruthMarker> truth = {truth_at(7, 100.0, 100.0, 40.0)};
    const std::vector<Observation> observed = {observed_at(7, 100.0, 100.0, 40.0),
                                               observed_at(8, 400.0, 400.0, 40.0)};
    const ImageAccuracy result =
            aruco3cuda::evaluate::compare_to_truth("scene", truth, observed, MatchConfig{});
    ASSERT_EQ(result.false_positive_, 1U);

    AccuracySummary summary;
    aruco3cuda::evaluate::accumulate_selected(result, {true}, &summary);

    EXPECT_EQ(summary.false_positive_, 0U);
    EXPECT_EQ(summary.observed_count_, 0U);
}

TEST(Accuracy, CountsIdMismatchAsMissedInSelectedAccumulation) {
    const std::vector<TruthMarker> truth = {truth_at(7, 100.0, 100.0, 40.0)};
    const std::vector<Observation> observed = {observed_at(8, 100.0, 100.0, 40.0)};
    const ImageAccuracy result =
            aruco3cuda::evaluate::compare_to_truth("scene", truth, observed, MatchConfig{});

    AccuracySummary summary;
    aruco3cuda::evaluate::accumulate_selected(result, {true}, &summary);

    EXPECT_EQ(summary.true_positive_, 0U);
    EXPECT_EQ(summary.false_negative_, 1U);
}

TEST(Accuracy, IgnoresSelectionOfWrongLength) {
    const std::vector<TruthMarker> truth = {truth_at(7, 100.0, 100.0, 40.0)};
    const std::vector<Observation> observed = {observed_at(7, 100.0, 100.0, 40.0)};
    const ImageAccuracy result =
            aruco3cuda::evaluate::compare_to_truth("scene", truth, observed, MatchConfig{});

    AccuracySummary summary;
    aruco3cuda::evaluate::accumulate_selected(result, {true, true}, &summary);
    EXPECT_EQ(summary.image_count_, 0U);

    aruco3cuda::evaluate::accumulate_selected(result, {true}, nullptr);
    SUCCEED();
}

TEST(Accuracy, ComputesPrecisionAndRecall) {
    AccuracySummary summary;
    summary.true_positive_ = 3U;
    summary.false_positive_ = 1U;
    summary.false_negative_ = 1U;

    double value = 0.0;
    ASSERT_TRUE(aruco3cuda::evaluate::precision(summary, &value));
    EXPECT_DOUBLE_EQ(value, 0.75);
    ASSERT_TRUE(aruco3cuda::evaluate::recall(summary, &value));
    EXPECT_DOUBLE_EQ(value, 0.75);
}

TEST(Accuracy, ReportsPrecisionUndefinedWithoutDetections) {
    // With no detections at all, precision is undefined. Returning 1.0 would mean
    // "the less you detect, the higher your precision".
    const AccuracySummary summary;
    double value = -1.0;
    EXPECT_FALSE(aruco3cuda::evaluate::precision(summary, &value));
    EXPECT_DOUBLE_EQ(value, -1.0);
}

TEST(Accuracy, ReportsRecallUndefinedWithoutTruth) {
    // The corpus includes scenes that contain zero markers.
    AccuracySummary summary;
    summary.false_positive_ = 1U;
    double value = -1.0;
    EXPECT_FALSE(aruco3cuda::evaluate::recall(summary, &value));
    EXPECT_DOUBLE_EQ(value, -1.0);
}

TEST(Accuracy, ComputesCornerRmse) {
    AccuracySummary summary;
    summary.corner_squared_sum_px2_ = 16.0;
    summary.corner_sample_count_ = 4U;

    double value = 0.0;
    ASSERT_TRUE(aruco3cuda::evaluate::corner_rmse_px(summary, &value));
    EXPECT_DOUBLE_EQ(value, 2.0);
}

TEST(Accuracy, ReportsRmseUndefinedWithoutSamples) {
    const AccuracySummary summary;
    double value = -1.0;
    EXPECT_FALSE(aruco3cuda::evaluate::corner_rmse_px(summary, &value));
}

TEST(Accuracy, RejectsNullOutputPointers) {
    const AccuracySummary summary;
    EXPECT_FALSE(aruco3cuda::evaluate::precision(summary, nullptr));
    EXPECT_FALSE(aruco3cuda::evaluate::recall(summary, nullptr));
    EXPECT_FALSE(aruco3cuda::evaluate::corner_rmse_px(summary, nullptr));
}
