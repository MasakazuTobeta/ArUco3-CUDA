// SPDX-License-Identifier: Apache-2.0
#include "aruco3cuda/util/statistics.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace aruco3cuda::util {

double percentile_nearest_rank(const std::vector<double>& sorted_samples, double percentile) {
    if (sorted_samples.empty() || percentile <= 0.0 || percentile > 100.0) {
        return 0.0;
    }
    const double count = static_cast<double>(sorted_samples.size());
    // nearest-rank 法。順位は 1 起点であるため index へ変換する際に 1 を引く。
    const auto rank = static_cast<std::size_t>(std::ceil(percentile / 100.0 * count));
    const std::size_t index = std::min(rank, sorted_samples.size()) - 1U;
    return sorted_samples[index];
}

bool compute_statistics(const std::vector<double>& samples, SampleStatistics* out) {
    if (out == nullptr || samples.empty()) {
        return false;
    }
    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());

    SampleStatistics result;
    result.count_ = sorted.size();
    result.min_ = sorted.front();
    result.max_ = sorted.back();

    double sum = 0.0;
    for (const double value : sorted) {
        sum += value;
    }
    result.mean_ = sum / static_cast<double>(sorted.size());

    if (sorted.size() > 1U) {
        double squared_error = 0.0;
        for (const double value : sorted) {
            const double difference = value - result.mean_;
            squared_error += difference * difference;
        }
        result.stddev_ = std::sqrt(squared_error / static_cast<double>(sorted.size() - 1U));
    }

    result.p50_ = percentile_nearest_rank(sorted, 50.0);
    result.p95_ = percentile_nearest_rank(sorted, 95.0);
    result.p99_ = percentile_nearest_rank(sorted, 99.0);

    *out = result;
    return true;
}

}  // namespace aruco3cuda::util
