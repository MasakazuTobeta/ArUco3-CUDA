// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace aruco3cuda::apriltag_small_cpu::detail {

// A nine-pixel optical transition has three nuisance parameters: brightness,
// illumination slope and contrast. Remove those components from the displacement
// derivative before estimating the translation of the edge.
struct DiffuseEdgeProfile {
    std::array<float, 9> step, derivative;
    float step_energy = 0, derivative_energy = 0;

    DiffuseEdgeProfile(float phase, float normal) {
        constexpr float scale = 1.7F / .78F;
        float step_mean = 0, step_slope = 0, gradient_mean = 0, gradient_slope = 0;
        for (int j = 0; j < 9; ++j) {
            const int z = j - 4;
            const float value = 1.F / (1.F + std::exp(-scale * normal * (z + phase)));
            step[j] = value;
            derivative[j] = -scale * normal * value * (1 - value);
            step_mean += value;
            step_slope += z * value;
            gradient_mean += derivative[j];
            gradient_slope += z * derivative[j];
        }
        float correlation = 0;
        for (int j = 0; j < 9; ++j) {
            const int z = j - 4;
            step[j] -= step_mean / 9 + z * step_slope / 60;
            derivative[j] -= gradient_mean / 9 + z * gradient_slope / 60;
            correlation += derivative[j] * step[j];
            step_energy += step[j] * step[j];
        }
        for (int j = 0; j < 9; ++j) {
            derivative[j] -= correlation / step_energy * step[j];
            derivative_energy += derivative[j] * derivative[j];
        }
    }
};

inline const DiffuseEdgeProfile& diffuse_profile(float phase, float normal) {
    constexpr int phases = 65, inclinations = 17;
    static const auto profiles = [] {
        std::vector<DiffuseEdgeProfile> table;
        table.reserve(phases * inclinations);
        for (int n = 0; n < inclinations; ++n)
            for (int f = 0; f < phases; ++f)
                table.emplace_back(f / 64.F - .5F, .7F + .3F * n / 16);
        return table;
    }();
    const int n = std::clamp(cvRound((std::abs(normal) - .7F) * 16 / .3F), 0, inclinations - 1);
    const int f = std::clamp(cvRound((phase + .5F) * 64), 0, phases - 1);
    return profiles[n * phases + f];
}

inline bool diffuse_crossing(const std::uint8_t* pixels, std::ptrdiff_t stride, float phase,
                             float normal, int minimum_contrast, float& offset, float& weight) {
    const auto& kernel = diffuse_profile(phase, normal);
    // Subtract the center intensity first to avoid cancellation on bright backgrounds.
    float sum = 0, slope = 0, energy = 0, edge = 0, shift = 0;
    for (int j = 0; j < 9; ++j) {
        const float v = int(pixels[(j - 4) * stride]) - int(*pixels);
        sum += v;
        slope += (j - 4) * v;
        energy += v * v;
        edge += v * kernel.step[j];
        shift += v * kernel.derivative[j];
    }
    const float contrast = edge / kernel.step_energy;
    if (contrast * normal < .55F * minimum_contrast)
        return false;
    offset = shift / (contrast * kernel.derivative_energy);
    if (!(std::abs(offset) < .9F))
        return false;
    const float residual =
            std::max(0.F, energy - sum * sum / 9 - slope * slope / 60 - edge * contrast);
    weight = contrast * contrast * kernel.derivative_energy / (1 + residual / 9);
    return true;
}
}  // namespace aruco3cuda::apriltag_small_cpu::detail
