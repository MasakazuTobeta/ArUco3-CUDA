// SPDX-License-Identifier: Apache-2.0
#include "geometry.hpp"

#include <algorithm>

namespace aruco3cuda::apriltag_small_cpu::detail {
namespace {

struct Line {
    Point point;
    Point direction;
};

struct Sample {
    float t;
    float displacement;
    float strength;
};

struct Fit {
    float weight = 0;
    float t1 = 0;
    float t2 = 0;
    float v0 = 0;
    float v1 = 0;

    void add(Sample sample, float w) {
        weight += w;
        t1 += w * sample.t;
        t2 += w * sample.t * sample.t;
        v0 += w * sample.displacement;
        v1 += w * sample.displacement * sample.t;
    }

    bool line(Point initial, Point delta, Point normal, float minimum_weight, Line& out) const {
        const float determinant = weight * t2 - t1 * t1;
        if (weight < minimum_weight || determinant < 0.018F * weight * weight) {
            return false;
        }
        const float slope = (weight * v1 - t1 * v0) / determinant;
        const float offset = (v0 - slope * t1) / weight;
        const Point direction = delta + slope * normal;
        out = {initial + offset * normal, direction / length(direction)};
        return true;
    }
};

float border_gradient(const cv::Mat& image, Point point, Point normal, float displacement) {
    const float dark = bilinear(image, point + (displacement - 1) * normal);
    const float light = bilinear(image, point + (displacement + 1) * normal);
    return dark < 0 || light < 0 ? 0.0F : std::max(0.0F, light - dark);
}

float visible_bilinear(const cv::Mat& image, Point point) {
    const int x = static_cast<int>(point.x);
    const int y = static_cast<int>(point.y);
    const float u = point.x - x;
    const float v = point.y - y;
    const auto* top = image.ptr<std::uint8_t>(y) + x;
    const auto* bottom = top + image.step[0];
    return (top[0] + u * (top[1] - top[0])) * (1 - v) +
           (bottom[0] + u * (bottom[1] - bottom[0])) * v;
}

// Adapt the project CUDA AprilTag edge estimator to serial CPU execution.
// Opposite-half samples propose lines, and supported offsets refine the best fit.
constexpr int kMaxRobustRadius = 32;

template <int kSamples, bool check_bounds>
Line sample_robust_edge(const cv::Mat& gray, Point initial, Point delta, float side_px,
                        int radius) {
    constexpr float kSupportPx = 2.5F;
    const Point normal(delta.y / side_px, -delta.x / side_px);
    std::array<float, 2 * kMaxRobustRadius + 1> penalties;
    for (int offset = -radius; offset <= radius; ++offset) {
        penalties[kMaxRobustRadius + offset] = 1.0F - 0.32F * std::abs(offset) / radius;
    }
    std::array<Sample, kSamples> samples{};
    for (int i = 0; i < kSamples; ++i) {
        const float t = 0.07F + (0.86F / (kSamples - 1)) * i;
        const Point point = initial + t * delta;
        // A bright-side intensity is read again as the dark side two steps later.
        constexpr int kCenter = kMaxRobustRadius + 1;
        std::array<float, 2 * kCenter + 1> luma;
        for (int offset = -radius - 1; offset <= radius + 1; ++offset) {
            const Point sample = point + static_cast<float>(offset) * normal;
            luma[kCenter + offset] =
                    check_bounds ? bilinear(gray, sample) : visible_bilinear(gray, sample);
        }
        std::array<float, 2 * kMaxRobustRadius + 1> gradients;
        float strength = 0;
        float displacement = 0;
        for (int offset = -radius; offset <= radius; ++offset) {
            const float dark = luma[kCenter + offset - 1];
            const float light = luma[kCenter + offset + 1];
            const float gradient =
                    check_bounds && (dark < 0 || light < 0) ? 0.0F : std::max(0.0F, light - dark);
            gradients[kMaxRobustRadius + offset] = gradient;
            const float candidate = gradient * penalties[kMaxRobustRadius + offset];
            if (candidate > strength) {
                strength = candidate;
                displacement = static_cast<float>(offset);
            }
        }
        float weight_sum = 0;
        float weighted_position = 0;
        for (int i = -4; i <= 4; ++i) {
            const float v = displacement + 0.5F * i;
            // Even half-steps inside the scanned interval have the same two native samples.
            const float g = i % 2 == 0 && std::abs(v) <= radius
                                    ? gradients[kMaxRobustRadius + static_cast<int>(v)]
                                    : border_gradient(gray, point, normal, v);
            weight_sum += g * g;
            weighted_position += v * g * g;
        }
        samples[i] = {t, weight_sum > 0 ? weighted_position / weight_sum : displacement, strength};
    }

    float best_support = 0;
    float offset = 0;
    float slope = 0;
    for (int i = 0; i < kSamples / 2; ++i) {
        const Sample a = samples[i];
        const Sample b = samples[i + kSamples / 2];
        if (a.strength < 1 || b.strength < 1) {
            continue;
        }
        const float candidate_slope = (b.displacement - a.displacement) / (b.t - a.t);
        const float candidate_offset = a.displacement - a.t * candidate_slope;
        float support = 0;
        for (const Sample sample : samples) {
            const float residual =
                    sample.displacement - candidate_offset - candidate_slope * sample.t;
            support += sample.strength * std::max(0.0F, 1.0F - std::abs(residual) / kSupportPx);
        }
        if (support > best_support) {
            best_support = support;
            offset = candidate_offset;
            slope = candidate_slope;
        }
    }

    Fit fit;
    for (const Sample sample : samples) {
        const float residual = sample.displacement - offset - slope * sample.t;
        fit.add(sample,
                sample.strength *
                        std::max(0.0F, 1.0F - residual * residual / (kSupportPx * kSupportPx)));
    }
    Line line{initial, delta / side_px};
    fit.line(initial, delta, normal, 32.0F, line);
    return line;
}

template <int kSamples>
Line robust_edge(const cv::Mat& gray, Point initial, Point delta, float side_px, int limit) {
    const int radius =
            static_cast<int>(std::clamp(std::ceil(side_px / 16), 2.F, static_cast<float>(limit)));
    // The segment endpoints bound the tangential samples. The extra pixel
    // accommodates rounding after the largest normal offset (radius + 1).
    const float margin = static_cast<float>(radius + 2);
    const auto inside = [&](Point point) {
        return point.x >= margin && point.y >= margin && point.x < gray.cols - 1 - margin &&
               point.y < gray.rows - 1 - margin;
    };
    if (inside(initial) && inside(initial + delta)) {
        return sample_robust_edge<kSamples, false>(gray, initial, delta, side_px, radius);
    }
    return sample_robust_edge<kSamples, true>(gray, initial, delta, side_px, radius);
}

constexpr float kNarrowFitRadiusPx = 2.F;

// The two endpoints bound all tangential samples. Leave another pixel for
// rounding when adding a normal offset of at most kNarrowFitRadiusPx.
bool inside_for_narrow_fit(const cv::Mat& gray, Point point) {
    constexpr float inset = kNarrowFitRadiusPx + 1.F;
    return point.x >= inset && point.y >= inset && point.x < gray.cols - 1 - inset &&
           point.y < gray.rows - 1 - inset;
}

// A crop inside a flat surround only needs the narrow signed transition fit.
template <bool in_image>
bool narrow_edge(const cv::Mat& gray, Point initial, Point delta, float side_px, Line& line) {
    constexpr int kSamples = 8;
    constexpr int kBins = 9;
    const Point unit = delta / side_px;
    const Point normal(unit.y, -unit.x);
    Fit fit;
    for (int i = 0; i < kSamples; ++i) {
        const float t = (i + 1.0F) / (kSamples + 1.0F);
        const Point point = initial + t * delta;
        std::array<float, kBins> luma{};
        for (int k = 0; k < kBins; ++k) {
            const Point at = point + ((k - kBins / 2) * 0.5F) * normal;
            luma[k] = in_image ? visible_bilinear(gray, at) : bilinear(gray, at);
        }
        int best = -1;
        float strength = 0;
        std::array<float, kBins> gradient{};
        for (int k = 1; k < kBins - 1; ++k) {
            if (!in_image && (luma[k - 1] < 0 || luma[k + 1] < 0)) {
                continue;
            }
            gradient[k] = std::max(0.0F, luma[k + 1] - luma[k - 1]);
            const float candidate = gradient[k] * (1.0F - 0.06F * std::abs(k - 4));
            if (candidate > strength) {
                strength = candidate;
                best = k;
            }
        }
        if (strength < 7.0F) {
            continue;
        }
        const float sum = gradient[best - 1] + gradient[best] + gradient[best + 1];
        const float position = 0.5F * (best - 4 + (gradient[best + 1] - gradient[best - 1]) / sum);
        fit.add({t, position, strength}, strength);
    }
    return fit.line(initial, delta, normal, 24.0F, line);
}

}  // namespace

bool refine_quad(const cv::Mat& gray, Quad& corners, bool uniform_surround, int coarse_stride) {
    std::array<Line, 4> lines{};
    std::array<Point, 4> deltas;
    std::array<float, 4> lengths;
    for (int side = 0; side < 4; ++side) {
        deltas[side] = corners[(side + 1) % 4] - corners[side];
        lengths[side] = length(deltas[side]);
        if (!std::isfinite(lengths[side]) || lengths[side] < (uniform_surround ? 6.F : 8.F))
            return false;
    }
    Quad fitted;
    // A failing adjacent intersection disqualifies this entire quad.
    const auto intersect = [&](int corner) {
        const Line& a = lines[(corner + 3) % 4];
        const Line& b = lines[corner];
        const float denominator = cross(a.direction, b.direction);
        if (std::abs(denominator) < (uniform_surround ? 0.2F : 0.12F)) {
            return false;
        }
        const float t = cross(b.point - a.point, b.direction) / denominator;
        fitted[corner] = a.point + t * a.direction;
        const float max_shift_px = uniform_surround ? 3.0F : 0.2F * lengths[corner];
        if (length(fitted[corner] - corners[corner]) > max_shift_px) {
            return false;
        }
        return true;
    };
    for (int side = 0; side < 4; ++side) {
        const Point initial = corners[side], delta = deltas[side];
        const auto side_px = lengths[side];
        if (uniform_surround) {
            const bool in_image = inside_for_narrow_fit(gray, initial) &&
                                  inside_for_narrow_fit(gray, corners[(side + 1) % 4]);
            const bool fitted =
                    in_image ? narrow_edge<true>(gray, initial, delta, side_px, lines[side])
                             : narrow_edge<false>(gray, initial, delta, side_px, lines[side]);
            if (!fitted) {
                return false;
            }
        } else {
            lines[side] =
                    coarse_stride
                            ? robust_edge<8>(gray, initial, delta, side_px, 2 * coarse_stride)
                            : robust_edge<24>(gray, initial, delta, side_px, kMaxRobustRadius);
        }
        if (side && !intersect(side))
            return false;
    }
    if (!intersect(0))
        return false;
    corners = fitted;
    return true;
}

}  // namespace aruco3cuda::apriltag_small_cpu::detail
