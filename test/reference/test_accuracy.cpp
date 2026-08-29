// SPDX-License-Identifier: Apache-2.0
//
// ground truth との突き合わせの検証。
//
// 対応付けの規則、ID を誤った場合の計上先、定義できない指標の扱いを固定する。
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

/// 中心 (cx, cy)、1 辺 side の軸並行正方形。
///
/// 並びはマーカー座標系の (0,0)、(S,0)、(S,S)、(0,S) に対応させる。
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
    // 同じ位置で ID を誤った場合、precision と recall の両方が下がるべきである。
    // 片方だけに数えると、一方の指標がこの誤りを見落とす。
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
    // 半径は真値の 1 辺に対する比で決まる。1 辺 40 に対し既定の 0.5 なので
    // 重心距離 20 までが対応する。境界の外側を確かめる。
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
    // ID は合っているが四隅の並びが 1 段巡回している場合。位置は一致するので
    // true positive だが、rotation の一致には数えない。
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
    // 2 つの検出が同じ真値の半径内にある場合、近い方が対応する。
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
    // 大きさで区分した recall を出す用途。選ばなかった真値は数に入らない。
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
    // 対応の付かない検出はどの真値にも属さないため足し込まない。結果として
    // この集計から precision は求まらない。読み手が誤らないことを固定する。
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
    // 検出が 1 件も無いとき precision は定義できない。1.0 を返すと
    // 「検出しないほど precision が高い」ことになる。
    const AccuracySummary summary;
    double value = -1.0;
    EXPECT_FALSE(aruco3cuda::evaluate::precision(summary, &value));
    EXPECT_DOUBLE_EQ(value, -1.0);
}

TEST(Accuracy, ReportsRecallUndefinedWithoutTruth) {
    // corpus にはマーカー 0 個の場面が含まれる。
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
