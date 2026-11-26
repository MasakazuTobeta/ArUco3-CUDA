// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "aruco3cuda/apriltag/small_cpu/detector.hpp"
#include "geometry.hpp"

#include <algorithm>
#include <limits>
#include <vector>

namespace aruco3cuda::apriltag_small_cpu::detail {
namespace planar_shapes {

using Basis = cv::Vec<double, 6>;
using Model = cv::Matx<double, 6, 8>;
using Normal = cv::Matx<double, 6, 6>;
struct Sample {
    Point center;
    Quad shape;
    Basis basis;
    int rotation;
    bool included;
};

inline Quad predict(const Basis& basis, const Model& model) {
    const auto v = model.t() * basis;
    Quad shape;
    for (int j = 0; j < 4; ++j)
        shape[j] = {float(v[2 * j]), float(v[2 * j + 1])};
    return shape;
}

inline bool fit(const std::vector<Sample>& samples, Model& model, Normal& inverse) {
    Normal normal = Normal::zeros();
    Model right = Model::zeros();
    for (const auto& sample : samples) {
        if (!sample.included)
            continue;
        for (int j = 0; j < 6; ++j) {
            for (int i = 0; i <= j; ++i)
                normal(j, i) += sample.basis[j] * sample.basis[i];
            for (int i = 0; i < 4; ++i) {
                right(j, 2 * i) += sample.basis[j] * sample.shape[i].x;
                right(j, 2 * i + 1) += sample.basis[j] * sample.shape[i].y;
            }
        }
    }
    for (int j = 0; j < 6; ++j)
        for (int i = 0; i < j; ++i)
            normal(i, j) = normal(j, i);
    bool valid = false;
    inverse = normal.inv(cv::DECOMP_CHOLESKY, &valid);
    if (!valid || cv::norm(normal, cv::NORM_INF) * cv::norm(inverse, cv::NORM_INF) > 2e5)
        return false;
    model = inverse * right;
    return true;
}

}  // namespace planar_shapes

// Equal-size coplanar squares have a smoothly varying image-space shape. The
// local Jacobian of a plane homography is quadratic in image coordinates.
// Estimate this shape from independently decoded current-frame neighbours;
// keep each measured center, and never change an ID or its canonical rotation.
inline void refine_planar_tag_shapes(std::vector<Detection>& tags, std::size_t minimum,
                                     float maximum_change) {
    using namespace planar_shapes;
    if (tags.size() < minimum || maximum_change <= 0)
        return;
    const auto center_of = [](const Quad& q) { return (q[0] + q[1] + q[2] + q[3]) * .25F; };
    const auto reference = tags[tags.size() / 2].corners_px_;
    const Point origin = center_of(reference);
    Point low = origin, high = origin;
    std::vector<Sample> samples;
    samples.reserve(tags.size());
    for (const auto& tag : tags) {
        const auto& q = tag.corners_px_;
        const Point center = center_of(q);
        Sample sample{};
        sample.center = center;
        float best = std::numeric_limits<float>::max();
        for (int rotation = 0; rotation < 4; ++rotation) {
            float cost = 0;
            for (int j = 0; j < 4; ++j) {
                const Point d = (q[(j + rotation) % 4] - center) - (reference[j] - origin);
                cost += d.dot(d);
            }
            if (cost < best) {
                best = cost;
                sample.rotation = rotation;
            }
        }
        for (int j = 0; j < 4; ++j)
            sample.shape[j] = q[(j + sample.rotation) % 4] - center;
        const float radius = std::max(2.F, .35F * length(reference[0] - origin));
        sample.included = best < 8 * radius * radius;
        samples.push_back(sample);
        low.x = std::min(low.x, center.x);
        low.y = std::min(low.y, center.y);
        high.x = std::max(high.x, center.x);
        high.y = std::max(high.y, center.y);
    }
    if (high.x <= low.x || high.y <= low.y)
        return;
    const Point mid = (low + high) * .5F;
    for (auto& sample : samples) {
        const double x = (sample.center.x - mid.x) / (high.x - low.x);
        const double y = (sample.center.y - mid.y) / (high.y - low.y);
        sample.basis = {1, x, y, x * x, x * y, y * y};
    }
    Normal inverse;
    Model model;
    const auto enough = [&] {
        const std::size_t count = std::count_if(samples.begin(), samples.end(),
                                                [](const Sample& s) { return s.included; });
        return count >= minimum && count * 4 >= tags.size() * 3;
    };
    for (double limit : {1.7 * maximum_change, .7 * maximum_change}) {
        if (!enough() || !fit(samples, model, inverse))
            return;
        for (auto& sample : samples) {
            if (!sample.included)
                continue;
            const auto predicted = predict(sample.basis, model);
            float sum = 0;
            for (int j = 0; j < 4; ++j) {
                const Point residual = predicted[j] - sample.shape[j];
                sum += residual.dot(residual);
            }
            sample.included = sum < 8 * limit * limit;
        }
    }
    if (!enough() || !fit(samples, model, inverse))
        return;
    // Require several independent, mutually consistent neighbours near this tag.
    // This prevents a good fit on one sheet from changing isolated markers on another.
    constexpr int bins = 8;
    std::array<int, bins * bins> support{};
    const auto bin_of = [](double coord) {
        return std::clamp(int((coord + .5) * bins), 0, bins - 1);
    };
    for (const auto& sample : samples) {
        if (!sample.included)
            continue;
        const auto predicted = predict(sample.basis, model);
        float worst = 0;
        for (int j = 0; j < 4; ++j)
            worst = std::max(worst, length(predicted[j] - sample.shape[j]));
        if (worst < maximum_change * .5F)
            ++support[bin_of(sample.basis[2]) * bins + bin_of(sample.basis[1])];
    }
    for (std::size_t i = 0; i < samples.size(); ++i) {
        const auto& sample = samples[i];
        if (!sample.included)
            continue;
        const auto predicted = predict(sample.basis, model);
        // Leverage measures dependence on this tag, including extrapolation at the board edges.
        // Dividing by (1-h) gives the error if its own observations had been omitted from the fit.
        const double leverage = sample.basis.dot(inverse * sample.basis);
        if (!(leverage >= 0 && leverage < .5))
            continue;
        float worst = 0;
        for (int j = 0; j < 4; ++j)
            worst = std::max(worst, length(predicted[j] - sample.shape[j]));
        if (worst > maximum_change * (1 - leverage))
            continue;
        const int x = bin_of(sample.basis[1]), y = bin_of(sample.basis[2]);
        int nearby = 0;
        for (int row = std::max(0, y - 1); row <= std::min(bins - 1, y + 1); ++row)
            for (int col = std::max(0, x - 1); col <= std::min(bins - 1, x + 1); ++col)
                nearby += support[row * bins + col];
        if (nearby < 4)
            continue;
        auto& corners = tags[i].corners_px_;
        for (int j = 0; j < 4; ++j)
            corners[(j + sample.rotation) % 4] = sample.center + predicted[j];
    }
}
}  // namespace aruco3cuda::apriltag_small_cpu::detail
