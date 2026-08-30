// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_UTIL_STATISTICS_HPP
#define ARUCO3CUDA_UTIL_STATISTICS_HPP

#include <cstddef>
#include <vector>

namespace aruco3cuda::util {

/// Summary statistics of a measurement sample.
///
/// The evaluation plan asks for the median and the tail percentiles, not just the mean.
/// Outliers are not removed; every statistic is computed over all samples.
struct SampleStatistics {
    std::size_t count_ = 0;
    double min_ = 0.0;
    double max_ = 0.0;
    double mean_ = 0.0;
    /// Sample standard deviation. It is 0 when count_ is 1.
    double stddev_ = 0.0;
    double p50_ = 0.0;
    double p95_ = 0.0;
    double p99_ = 0.0;
};

/// Computes a percentile by the nearest-rank method.
///
/// Returns the value at rank ceil(percentile / 100 * count) of the samples sorted in
/// ascending order. No interpolation is performed, so the value returned is always one
/// of the measured values. This method is the project-wide standard so that the way
/// results are aggregated does not depend on the implementation.
///
/// @param sorted_samples The samples in ascending order. Must not be empty. They are
///                       only read, never copied or retained.
/// @param percentile Greater than 0 and at most 100. Out of range returns 0.
/// @return The corresponding sample value, or 0 if an argument is invalid.
///
/// Ownership: does not retain the memory behind the arguments. The return value is a value.
/// Synchronization: host only, with no synchronization point. Reentrant.
///
/// Example input: {1, 2, 3, 4, 5}, percentile = 50
/// Example output: 3
double percentile_nearest_rank(const std::vector<double>& sorted_samples, double percentile);

/// Computes the summary statistics of a sample.
///
/// Outliers are not removed. Aggregation always covers every sample, so that a favorable
/// subset cannot be what remains.
///
/// @param samples The samples, in any order. Returns false if empty. They are copied and
///                sorted internally, so the argument is left unchanged.
/// @param out On success, receives the result; left unchanged on failure. The caller owns
///            the storage.
/// @return true on success.
///
/// Ownership: does not retain the memory behind the arguments.
/// Synchronization: host only, with no synchronization point. Reentrant.
///
/// Example input: {4.0, 1.0, 3.0, 2.0, 5.0}
/// Example output: count_ = 5, min_ = 1.0, max_ = 5.0, mean_ = 3.0, p50_ = 3.0
bool compute_statistics(const std::vector<double>& samples, SampleStatistics* out);

}  // namespace aruco3cuda::util

#endif  // ARUCO3CUDA_UTIL_STATISTICS_HPP
