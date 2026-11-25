// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>

namespace aruco3cuda::apriltag_small_cpu::detail {

// cumulative_[v] counts the sampled Gray8 values strictly below v.
// The extra endpoint gives the total at 256, including fully saturated pixels.
class LumaDistribution {
public:
    using Bins = std::array<int, 256>;

    explicit LumaDistribution(const Bins& counts) {
        std::partial_sum(counts.begin(), counts.end(), this->cumulative_.begin() + 1);
    }

    int below(int level) const { return this->cumulative_[std::clamp(level, 0, 256)]; }
    int total() const { return this->cumulative_.back(); }

    // One-based sample rank. Ties select the first level to reach the rank.
    int level_at_rank(int rank) const {
        const auto first = this->cumulative_.begin() + 1;
        return static_cast<int>(std::lower_bound(first, this->cumulative_.end(), rank) - first);
    }

    int percentile(int percent) const {
        return this->level_at_rank(
                static_cast<int>((std::int64_t{this->total()} * percent + 99) / 100));
    }

private:
    std::array<int, 257> cumulative_{};
};

}  // namespace aruco3cuda::apriltag_small_cpu::detail
