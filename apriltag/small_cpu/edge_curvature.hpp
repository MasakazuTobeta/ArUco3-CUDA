// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "geometry.hpp"

#include <vector>

namespace aruco3cuda::apriltag_small_cpu::detail {

// Under radial distortion the departure of an edge midpoint from its chord
// correlates with the radius normal to the edge. The two intercept terms allow
// for an unknown optical center, including when the input is a cropped image.
class EdgeCurvatureEvidence {
public:
    void add(double radial, Point normal, double normalized_sag, double weight) {
        samples_.push_back({{radial, normal.x, normal.y}, normalized_sag, weight});
    }

    bool coherent(double minimum_amplitude, double minimum_r_squared) const {
        if (samples_.size() < 24)
            return false;
        cv::Matx33d normal = cv::Matx33d::zeros();
        cv::Vec3d right{};
        for (const auto& sample : samples_) {
            normal += sample.weight * sample.basis * sample.basis.t();
            right += sample.weight * sample.basis * sample.sag;
        }
        cv::Vec3d model;
        if (!cv::solve(normal, right, model, cv::DECOMP_CHOLESKY))
            return false;
        double residual = 0, signal = 0, weights = 0;
        std::size_t agreement = 0;
        for (const auto& sample : samples_) {
            const double predicted = sample.basis.dot(model);
            signal += sample.weight * predicted * predicted;
            const double error = sample.sag - predicted;
            residual += sample.weight * error * error;
            weights += sample.weight;
            agreement += predicted * sample.sag > 0;
        }
        return signal > weights * minimum_amplitude * minimum_amplitude &&
               signal > minimum_r_squared * (signal + residual) &&
               agreement > .63 * samples_.size();
    }

private:
    struct Sample {
        cv::Vec3d basis;
        double sag;
        double weight;
    };
    std::vector<Sample> samples_;
};
}  // namespace aruco3cuda::apriltag_small_cpu::detail
