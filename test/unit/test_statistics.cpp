// SPDX-License-Identifier: Apache-2.0
//
// 測定結果の集計を検証する。分位点の定義は評価結果の比較可能性に直結するため、
// nearest-rank 法の挙動を境界値まで固定する。
#include "aruco3cuda/util/statistics.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

// 正常系: nearest-rank 法が定義どおりの順位を返す。
TEST(StatisticsTest, percentile_uses_nearest_rank) {
    const std::vector<double> samples = {1.0, 2.0, 3.0, 4.0, 5.0};
    // rank = ceil(50/100 * 5) = 3 -> index 2
    EXPECT_DOUBLE_EQ(aruco3cuda::util::percentile_nearest_rank(samples, 50.0), 3.0);
    // rank = ceil(95/100 * 5) = 5 -> index 4
    EXPECT_DOUBLE_EQ(aruco3cuda::util::percentile_nearest_rank(samples, 95.0), 5.0);
    EXPECT_DOUBLE_EQ(aruco3cuda::util::percentile_nearest_rank(samples, 100.0), 5.0);
    // 補間しないため、返る値は必ず実測値のいずれかになる。
    EXPECT_DOUBLE_EQ(aruco3cuda::util::percentile_nearest_rank(samples, 21.0), 2.0);
}

// 境界値: 標本が 1 個でも定義どおりに動く。
TEST(StatisticsTest, percentile_handles_single_sample) {
    const std::vector<double> samples = {7.5};
    EXPECT_DOUBLE_EQ(aruco3cuda::util::percentile_nearest_rank(samples, 50.0), 7.5);
    EXPECT_DOUBLE_EQ(aruco3cuda::util::percentile_nearest_rank(samples, 99.0), 7.5);
}

// 異常系: 空の標本と範囲外の分位点を拒否する。
TEST(StatisticsTest, percentile_rejects_invalid_input) {
    const std::vector<double> empty;
    const std::vector<double> samples = {1.0, 2.0};
    EXPECT_DOUBLE_EQ(aruco3cuda::util::percentile_nearest_rank(empty, 50.0), 0.0);
    EXPECT_DOUBLE_EQ(aruco3cuda::util::percentile_nearest_rank(samples, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(aruco3cuda::util::percentile_nearest_rank(samples, 101.0), 0.0);
}

// 正常系: 要約統計が既知の値と一致する。
TEST(StatisticsTest, computes_summary_statistics) {
    const std::vector<double> samples = {4.0, 1.0, 3.0, 2.0, 5.0};
    aruco3cuda::util::SampleStatistics stats;
    ASSERT_TRUE(aruco3cuda::util::compute_statistics(samples, &stats));
    EXPECT_EQ(stats.count_, 5U);
    EXPECT_DOUBLE_EQ(stats.min_, 1.0);
    EXPECT_DOUBLE_EQ(stats.max_, 5.0);
    EXPECT_DOUBLE_EQ(stats.mean_, 3.0);
    // 標本標準偏差: sqrt(10 / 4) = 1.5811...
    EXPECT_NEAR(stats.stddev_, std::sqrt(2.5), 1e-12);
    EXPECT_DOUBLE_EQ(stats.p50_, 3.0);
}

// 境界値: 標本 1 個では標準偏差を 0 とする。
TEST(StatisticsTest, single_sample_has_zero_stddev) {
    const std::vector<double> samples = {42.0};
    aruco3cuda::util::SampleStatistics stats;
    ASSERT_TRUE(aruco3cuda::util::compute_statistics(samples, &stats));
    EXPECT_EQ(stats.count_, 1U);
    EXPECT_DOUBLE_EQ(stats.stddev_, 0.0);
    EXPECT_DOUBLE_EQ(stats.mean_, 42.0);
}

// 正常系: 外れ値を除去しない。裾の値が結果へ反映される。
TEST(StatisticsTest, does_not_remove_outliers) {
    std::vector<double> samples(99, 1.0);
    samples.push_back(1000.0);
    aruco3cuda::util::SampleStatistics stats;
    ASSERT_TRUE(aruco3cuda::util::compute_statistics(samples, &stats));
    EXPECT_EQ(stats.count_, 100U);
    EXPECT_DOUBLE_EQ(stats.max_, 1000.0);
    // rank = ceil(99/100 * 100) = 99 -> index 98 は 1.0
    EXPECT_DOUBLE_EQ(stats.p99_, 1.0);
    EXPECT_GT(stats.mean_, 1.0);
}

// 異常系: 空の標本と nullptr を拒否する。
TEST(StatisticsTest, rejects_invalid_arguments) {
    aruco3cuda::util::SampleStatistics stats;
    EXPECT_FALSE(aruco3cuda::util::compute_statistics({}, &stats));
    EXPECT_FALSE(aruco3cuda::util::compute_statistics({1.0}, nullptr));
}

}  // namespace
