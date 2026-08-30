// SPDX-License-Identifier: Apache-2.0
//
// Verifies the aggregation of measurement results. The definition of a quantile
// directly determines whether evaluation results can be compared, so the behavior
// of the nearest-rank method is pinned down all the way to its boundaries.
#include "aruco3cuda/util/statistics.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

// Nominal: the nearest-rank method returns the rank its definition prescribes.
TEST(StatisticsTest, percentile_uses_nearest_rank) {
    const std::vector<double> samples = {1.0, 2.0, 3.0, 4.0, 5.0};
    // rank = ceil(50/100 * 5) = 3 -> index 2
    EXPECT_DOUBLE_EQ(aruco3cuda::util::percentile_nearest_rank(samples, 50.0), 3.0);
    // rank = ceil(95/100 * 5) = 5 -> index 4
    EXPECT_DOUBLE_EQ(aruco3cuda::util::percentile_nearest_rank(samples, 95.0), 5.0);
    EXPECT_DOUBLE_EQ(aruco3cuda::util::percentile_nearest_rank(samples, 100.0), 5.0);
    // No interpolation is done, so the returned value is always one of the samples.
    EXPECT_DOUBLE_EQ(aruco3cuda::util::percentile_nearest_rank(samples, 21.0), 2.0);
}

// Boundary: the definition still holds with a single sample.
TEST(StatisticsTest, percentile_handles_single_sample) {
    const std::vector<double> samples = {7.5};
    EXPECT_DOUBLE_EQ(aruco3cuda::util::percentile_nearest_rank(samples, 50.0), 7.5);
    EXPECT_DOUBLE_EQ(aruco3cuda::util::percentile_nearest_rank(samples, 99.0), 7.5);
}

// Error case: an empty sample set and an out-of-range percentile are rejected.
TEST(StatisticsTest, percentile_rejects_invalid_input) {
    const std::vector<double> empty;
    const std::vector<double> samples = {1.0, 2.0};
    EXPECT_DOUBLE_EQ(aruco3cuda::util::percentile_nearest_rank(empty, 50.0), 0.0);
    EXPECT_DOUBLE_EQ(aruco3cuda::util::percentile_nearest_rank(samples, 0.0), 0.0);
    EXPECT_DOUBLE_EQ(aruco3cuda::util::percentile_nearest_rank(samples, 101.0), 0.0);
}

// Nominal: the summary statistics match known values.
TEST(StatisticsTest, computes_summary_statistics) {
    const std::vector<double> samples = {4.0, 1.0, 3.0, 2.0, 5.0};
    aruco3cuda::util::SampleStatistics stats;
    ASSERT_TRUE(aruco3cuda::util::compute_statistics(samples, &stats));
    EXPECT_EQ(stats.count_, 5U);
    EXPECT_DOUBLE_EQ(stats.min_, 1.0);
    EXPECT_DOUBLE_EQ(stats.max_, 5.0);
    EXPECT_DOUBLE_EQ(stats.mean_, 3.0);
    // Sample standard deviation: sqrt(10 / 4) = 1.5811...
    EXPECT_NEAR(stats.stddev_, std::sqrt(2.5), 1e-12);
    EXPECT_DOUBLE_EQ(stats.p50_, 3.0);
}

// Boundary: with a single sample the standard deviation is defined as 0.
TEST(StatisticsTest, single_sample_has_zero_stddev) {
    const std::vector<double> samples = {42.0};
    aruco3cuda::util::SampleStatistics stats;
    ASSERT_TRUE(aruco3cuda::util::compute_statistics(samples, &stats));
    EXPECT_EQ(stats.count_, 1U);
    EXPECT_DOUBLE_EQ(stats.stddev_, 0.0);
    EXPECT_DOUBLE_EQ(stats.mean_, 42.0);
}

// Nominal: outliers are not removed. Values in the tail are reflected in the result.
TEST(StatisticsTest, does_not_remove_outliers) {
    std::vector<double> samples(99, 1.0);
    samples.push_back(1000.0);
    aruco3cuda::util::SampleStatistics stats;
    ASSERT_TRUE(aruco3cuda::util::compute_statistics(samples, &stats));
    EXPECT_EQ(stats.count_, 100U);
    EXPECT_DOUBLE_EQ(stats.max_, 1000.0);
    // rank = ceil(99/100 * 100) = 99 -> index 98, which is 1.0
    EXPECT_DOUBLE_EQ(stats.p99_, 1.0);
    EXPECT_GT(stats.mean_, 1.0);
}

// Error case: an empty sample set and a null output pointer are rejected.
TEST(StatisticsTest, rejects_invalid_arguments) {
    aruco3cuda::util::SampleStatistics stats;
    EXPECT_FALSE(aruco3cuda::util::compute_statistics({}, &stats));
    EXPECT_FALSE(aruco3cuda::util::compute_statistics({1.0}, nullptr));
}

}  // namespace
