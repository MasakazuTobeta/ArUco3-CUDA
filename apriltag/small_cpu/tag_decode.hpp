// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "aruco3cuda/dictionary.hpp"
#include "geometry.hpp"

namespace aruco3cuda::apriltag_small_cpu::detail {
inline int bit_count(MarkerCode word) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(word);
#else
    int count = 0;
    while (word != 0) {
        word &= word - 1;
        ++count;
    }
    return count;
#endif
}

constexpr int kSourceRingSamples = 32;

struct LumaSample {
    float u;
    float v;
    float intensity;
};

struct Plane {
    float x = 0;
    float y = 0;
    float center = 0;

    float at(float u, float v) const { return center + x * u + y * v; }
};

struct Projection {
    Point x;
    Point y;
    Point initial;
    float z_x;
    float z_y;

    float sample(const cv::Mat& gray, float u, float v) const {
        const float z = 1 + z_x * u + z_y * v;
        return z < 1e-5F ? -1.0F : bilinear(gray, (initial + u * x + v * y) / z);
    }
};

inline bool project(const Quad& points, Projection& projection) {
    const Point a = points[1] - points[2];
    const Point b = points[3] - points[2];
    const Point diff = points[0] - points[1] - b;
    const float determinant = cross(a, b);
    if (std::abs(determinant) < 1e-5F) {
        return false;
    }
    projection.z_x = cross(diff, b) / determinant;
    projection.z_y = cross(a, diff) / determinant;
    projection.initial = points[0];
    projection.x = (1 + projection.z_x) * points[1] - points[0];
    projection.y = (1 + projection.z_y) * points[3] - points[0];
    return true;
}

inline bool fit_plane(const std::array<LumaSample, kSourceRingSamples>& points, Plane& fit) {
    float u = 0, v = 0, uu = 0, vv = 0, uv = 0, g = 0, gu = 0, gv = 0;
    int count = 0;
    for (LumaSample point : points) {
        if (point.intensity < 0) {
            continue;
        }
        const float x = point.u;
        const float y = point.v;
        const float level = point.intensity;
        u += x;
        v += y;
        uu += x * x;
        vv += y * y;
        uv += x * y;
        g += level;
        gu += x * level;
        gv += y * level;
        ++count;
    }
    if (count < kSourceRingSamples / 2) {
        return false;
    }
    u /= count;
    v /= count;
    g /= count;
    gu -= count * u * g;
    gv -= count * v * g;
    uu -= count * u * u;
    vv -= count * v * v;
    uv -= count * u * v;
    const float determinant = uu * vv - uv * uv;
    if (determinant < 1e-5F) {
        return false;
    }
    fit.x = (gu * vv - gv * uv) / determinant;
    fit.y = (gv * uu - gu * uv) / determinant;
    fit.center = g - u * fit.x - v * fit.y;
    return true;
}

}  // namespace aruco3cuda::apriltag_small_cpu::detail
