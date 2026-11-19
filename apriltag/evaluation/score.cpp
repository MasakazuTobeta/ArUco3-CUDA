// SPDX-License-Identifier: Apache-2.0
// Batch bridge from validated manifest rows to the project's accuracy evaluator.
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "accuracy.hpp"

namespace {
template <typename Marker>
std::vector<Marker> read_markers(std::size_t count) {
    if (count > 65536) {
        throw std::invalid_argument("excessive marker count");
    }
    std::vector<Marker> markers(count);
    for (auto& marker : markers) {
        std::cin >> marker.id_;
        for (double& coordinate : marker.corners_) {
            std::cin >> coordinate;
        }
        if (!std::cin) {
            throw std::invalid_argument("incomplete marker");
        }
    }
    return markers;
}
}  // namespace

int main() {
    using namespace aruco3cuda::evaluate;
    std::size_t truth_count = 0;
    std::size_t observation_count = 0;
    std::cout << std::setprecision(17);
    while (std::cin >> truth_count >> observation_count) {
        auto truth = read_markers<TruthMarker>(truth_count);
        for (auto& marker : truth) {
            marker.side_px_ = aruco3cuda::report::quad_average_side(marker.corners_);
        }
        const auto observed = read_markers<Observation>(observation_count);
        const auto result = compare_to_truth("", truth, observed, {});
        std::cout << result.false_positive_;
        for (std::size_t i = 0; i < truth_count; ++i) {
            std::cout << ' ' << (result.truth_outcomes_[i] == TruthOutcome::kDetected) << ' '
                      << result.truth_rotation_agreed_[i] << ' '
                      << result.truth_squared_error_px2_[i] << ' ' << result.truth_corner_max_px_[i]
                      << ' ' << (result.truth_outcomes_[i] == TruthOutcome::kIdMismatched);
        }
        std::cout << '\n';
    }
    return std::cin.eof() ? 0 : 1;
}
