// SPDX-License-Identifier: Apache-2.0
#include "small_tag_refinement.hpp"

#include "geometry.hpp"
#include "tag_codebook.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace aruco3cuda::apriltag_small_cpu::detail {
namespace {

constexpr int kAngles = 49;
constexpr int kWidths = 26;
constexpr int kDistances = 193;
constexpr float kMinWidth = .16F;
constexpr float kWidthStep = .044F;
constexpr float kDistanceScale = 24.F;
constexpr float kDistanceOrigin = (kDistances - 1) * .5F;

class ApertureTable {
public:
    ApertureTable() {
        for (int t = 0; t < kAngles; ++t) {
            const double angle = t * (CV_PI * .25) / (kAngles - 1);
            const double a = .5 * std::cos(angle), b = .5 * std::sin(angle);
            for (int bindex = 0; bindex < kWidths; ++bindex) {
                const double width = kMinWidth + kWidthStep * bindex;
                // Integrate a continuous Gaussian edge response over the square
                // pixel aperture by finite differencing its second primitive.
                const auto cdf = [](double coordinate) {
                    return .5 * std::erfc(-coordinate / std::sqrt(2.0));
                };
                const auto density = [](double coordinate) {
                    return .398942280401432678 * std::exp(-.5 * coordinate * coordinate);
                };
                const auto first = [&, width](double x) {
                    const double coordinate = x / width;
                    return x * cdf(coordinate) + width * density(coordinate);
                };
                const auto second = [&, width](double x) {
                    const double coordinate = x / width;
                    return .5 * ((x * x + width * width) * cdf(coordinate) +
                                 x * width * density(coordinate));
                };
                for (int z = 0; z < kDistances; ++z) {
                    const double d = std::abs((z - kDistanceOrigin) / kDistanceScale);
                    const double positive =
                            std::clamp(b <= 1.e-8 ? (first(d + a) - first(d - a)) / (2 * a)
                                                  : (second(d + a + b) - second(d - a + b) -
                                                     second(d + a - b) + second(d - a - b)) /
                                                            (4 * a * b),
                                       .5, 1.);
                    const double response = z < kDistanceOrigin ? 1. - positive : positive;
                    this->values_[t][bindex][z] = static_cast<float>(response);
                }
            }
        }
    }
    struct Level {
        float value, distance, width;
    };
    Level measure(int angle, float width, float d) const {
        const float a = std::clamp((width - kMinWidth) / kWidthStep, 0.F, kWidths - 1.0001F);
        const float b = std::clamp(d * kDistanceScale + kDistanceOrigin, 0.F, kDistances - 1.0001F);
        const int y = static_cast<int>(a), x = static_cast<int>(b);
        const auto& row = this->values_[angle][y];
        const auto& next = this->values_[angle][y + 1];
        const float sx = b - x, sy = a - y;
        const float lo = row[x] + sx * (row[x + 1] - row[x]);
        const float hi = next[x] + sx * (next[x + 1] - next[x]);
        return {lo + sy * (hi - lo),
                kDistanceScale * ((1 - sy) * (row[x + 1] - row[x]) + sy * (next[x + 1] - next[x])),
                (hi - lo) / kWidthStep};
    }

private:
    std::array<std::array<std::array<float, kDistances>, kWidths>, kAngles> values_;
};

struct Sample {
    float tangent, offset, intensity;
};
using Samples = std::array<Sample, 512>;
using Photo = std::array<ApertureTable::Level, 512>;
using Parameters = cv::Vec4f;
constexpr float kCurvaturePrecision = 1000.F;

struct Evaluation {
    float cost = std::numeric_limits<float>::infinity();
    float rms = std::numeric_limits<float>::infinity();
    float contrast = 0;
    cv::Matx44f gram = cv::Matx44f::zeros();
    cv::Vec4f rhs{};
};

Evaluation evaluate(const Samples& samples, int count, const ApertureTable& lut, int angle,
                    Parameters p, bool deriv) {
    Photo predicted;
    for (int i = 0; i < count; ++i) {
        const auto& item = samples[i];
        predicted[i] = lut.measure(
                angle, p[2],
                item.offset - p[0] - p[1] * item.tangent - p[3] * item.tangent * item.tangent);
    }
    cv::Vec4f solution{};
    for (int iteration = 0; iteration < 2; ++iteration) {
        cv::Matx44f h = cv::Matx44f::zeros();
        cv::Vec4f b{};
        for (int i = 0; i < count; ++i) {
            const auto& item = samples[i];
            const auto level = predicted[i].value;
            const cv::Vec4f features(1, item.tangent, level, item.tangent * level);
            const float e = iteration ? solution.dot(features) - item.intensity : 0.F;
            const float w = 1.F / std::sqrt(1 + e * e / 9);
            for (int row = 0; row < 4; ++row) {
                const float v = w * features[row];
                b[row] += v * item.intensity;
                for (int col = 0; col <= row; ++col)
                    h(row, col) += v * features[col];
            }
        }
        h(1, 1) += .05F;
        h(3, 3) += .12F;
        for (int row = 0; row < 4; ++row)
            for (int col = 0; col < row; ++col)
                h(col, row) = h(row, col);
        if (!cv::solve(h, b, solution, cv::DECOMP_CHOLESKY))
            return {};
    }
    if (solution[2] < 12 || solution[2] > 255)
        return {};
    Evaluation cost;
    cost.contrast = solution[2];
    cost.cost = .5F * kCurvaturePrecision * p[3] * p[3] / (5.5F * 5.5F);
    cost.rms = 0;
    auto projection = cv::Matx44f::zeros();
    cv::Matx44f gram_photo = cv::Matx44f::zeros();
    for (int i = 0; i < count; ++i) {
        const auto& item = samples[i];
        const auto f = predicted[i];
        const cv::Vec4f features{1, item.tangent, f.value, item.tangent * f.value};
        const float error = solution.dot(features) - item.intensity;
        const float robust = std::sqrt(1 + error * error / (5.5F * 5.5F));
        cost.cost += robust - 1;
        cost.rms += error * error;
        if (!deriv)
            continue;
        const float contrast = solution[2] + item.tangent * solution[3];
        const cv::Vec4f g{-contrast * f.distance, -contrast * f.distance * item.tangent,
                          contrast * f.width, -contrast * f.distance * item.tangent * item.tangent};
        const float w = 1.F / robust;
        for (int a = 0; a < 4; ++a) {
            const float weighted = g[a] * w;
            cost.rhs[a] -= weighted * error;
            for (int b = 0; b <= a; ++b)
                cost.gram(a, b) += weighted * g[b];
            for (int b = 0; b < 4; ++b)
                projection(a, b) += weighted * features[b];
        }
        for (int a = 0; a < 4; ++a)
            for (int b = 0; b <= a; ++b)
                gram_photo(a, b) += w * features[a] * features[b];
    }
    cost.rms = std::sqrt(cost.rms / count);
    if (deriv) {
        for (int a = 0; a < 4; ++a)
            for (int b = 0; b < a; ++b)
                cost.gram(b, a) = cost.gram(a, b);
        gram_photo(1, 1) += .05F;
        gram_photo(3, 3) += .12F;
        for (int a = 0; a < 4; ++a)
            for (int b = 0; b < a; ++b)
                gram_photo(b, a) = gram_photo(a, b);
        const auto inv = gram_photo.inv(cv::DECOMP_CHOLESKY);
        const auto removed = projection * inv * projection.t();
        for (int a = 0; a < 4; ++a)
            cost.gram(a, a) = cost.gram(a, a) * 1.05F + (a == 2 ? .15F : 1.F);
        cost.gram -= removed;
        cost.gram(3, 3) += kCurvaturePrecision;
        cost.rhs[3] -= kCurvaturePrecision * p[3];
    }
    return cost;
}

struct EdgeLine {
    cv::Point2f normal{}, center{}, tangent{};
    float offset = 0, curvature = 0, half = 0;

    float displacement(cv::Point2f p) const {
        const float along = (p - center).dot(tangent) / half;
        return curvature * along * along;
    }
};

bool fit_edge(const cv::Mat& im, const Quad& q, int side, std::uint64_t word, EdgeLine& line,
              float& residual, float& relative) {
    const auto start = q[side], end = q[(side + 1) & 3];
    const auto center = .5F * (start + end);
    const auto delta = end - start;
    const float size = length(delta), half = .5F * size;
    const auto tangent = delta * (1.F / size);
    const cv::Point2f normal(tangent.y, -tangent.x);
    const float lateral = std::min(-(q[(side + 3) & 3] - start).dot(normal),
                                   -(q[(side + 2) & 3] - end).dot(normal));
    if (lateral < 10)
        return false;
    const float radius = std::clamp(.108F * lateral, 1.45F, 2.65F);
    const float white_radius = std::min(radius, std::max(.95F, .075F * lateral));
    const int x0 = cvFloor(std::min(start.x, end.x) - 4), x1 = cvCeil(std::max(start.x, end.x) + 4);
    const int y0 = cvFloor(std::min(start.y, end.y) - 4), y1 = cvCeil(std::max(start.y, end.y) + 4);
    Samples samples;
    int count = 0;
    for (int y = y0; y <= y1; ++y) {
        const auto* row = im.ptr<std::uint8_t>(y);
        for (int x = x0; x <= x1; ++x) {
            const cv::Point2f r(x - center.x, y - center.y);
            const float along = r.dot(tangent) / half;
            if (std::abs(along) > .75F)
                continue;
            const float distance = r.dot(normal);
            if (std::abs(distance) >= radius)
                continue;
            if (distance < -white_radius) {
                const int cell = std::clamp(static_cast<int>((along + 1) * 4), 1, 6);
                const int offset = side == 0   ? cell - 1
                                   : side == 1 ? (cell - 1) * 6 + 5
                                   : side == 2 ? 36 - cell
                                               : (6 - cell) * 6;
                if ((word >> offset) & 1U)
                    continue;
            }
            if (count == static_cast<int>(samples.size()))
                return false;
            samples[count++] = {along, distance, float(row[x])};
        }
    }
    if (count < 18)
        return false;
    const float ax = std::abs(normal.x), ay = std::abs(normal.y);
    const int angle = cvRound(std::atan2(std::min(ax, ay), std::max(ax, ay)) *
                              ((kAngles - 1) / (float(CV_PI) * .25F)));
    static const ApertureTable model;
    Parameters best{0, 0, .58F, 0};
    Evaluation accepted = evaluate(samples, count, model, angle, best, true);
    if (!std::isfinite(accepted.cost))
        return false;
    for (int i = 0; i < 3; ++i) {
        cv::Vec4f shift;
        if (!cv::solve(accepted.gram, accepted.rhs, shift, cv::DECOMP_CHOLESKY))
            break;
        // A linear normal shift over the visible strip is bounded by the
        // sum of intercept movement and the supported fraction of slope.
        const float motion =
                std::abs(shift[0]) + .75F * std::abs(shift[1]) + .5625F * std::abs(shift[3]);
        if (motion < .006F && std::abs(shift[2]) < .012F)
            break;
        bool found = false;
        for (float fraction : {1.F, .5F}) {
            auto p = best + fraction * shift;
            p[0] = std::clamp(p[0], -.75F, .75F);
            p[1] = std::clamp(p[1], -.6F, .6F);
            p[2] = std::clamp(p[2], .17F, 1.2F);
            p[3] = std::clamp(p[3], -.25F, .25F);
            const auto next = evaluate(samples, count, model, angle, p, true);
            if (next.cost < accepted.cost) {
                accepted = next;
                best = p;
                found = true;
                break;
            }
        }
        if (!found || (i >= 1 && motion < .04F && std::abs(shift[2]) < .07F))
            break;
    }
    if (std::abs(best[0]) > .73F || std::abs(best[1]) > .58F || !std::isfinite(accepted.rms))
        return false;
    line.normal = normal - tangent * (best[1] / half);
    line.offset = center.dot(line.normal) + best[0];
    line.center = center;
    line.tangent = tangent;
    line.curvature = best[3];
    line.half = half;
    residual = accepted.rms;
    relative = accepted.rms / accepted.contrast;
    return true;
}

bool refine_aperture(const cv::Mat& gray, Detection& tag, float ceiling) {
    const auto prev = tag.corners_px_;
    std::array<EdgeLine, 4> edges;
    for (int j = 0; j < 4; ++j) {
        const auto pt = prev[j];
        if (!(pt.x >= 4 && pt.y >= 4 && pt.x <= gray.cols - 6 && pt.y <= gray.rows - 6))
            return false;
    }
    auto word = generated::kDictApriltag36h11Table.codes_[4 * tag.id_];
    for (int j = 0; j < 4; ++j) {
        float residual, relative;
        if (!fit_edge(gray, prev, j, word, edges[j], residual, relative) || !(relative <= ceiling))
            return false;
    }
    Quad next;
    const auto base = prev[0];
    for (int j = 0; j < 4; ++j) {
        const auto& a = edges[(j + 3) & 3];
        const auto& b = edges[j];
        const float d = a.normal.cross(b.normal);
        if (std::abs(d) < .35F)
            return false;
        const float a0 = a.offset - a.normal.dot(base), b0 = b.offset - b.normal.dot(base);
        next[j] = prev[j];
        for (int refine = 0; refine < 2; ++refine) {
            const float u = a0 + a.displacement(next[j]), v = b0 + b.displacement(next[j]);
            next[j] = base + cv::Point2f((u * b.normal.y - v * a.normal.y) / d,
                                         (a.normal.x * v - b.normal.x * u) / d);
        }
        if (length(next[j] - prev[j]) > 1.05F)
            return false;
    }
    if (!verify_payload<true>(gray, next, tag.id_, 25, 7.5F, 0))
        return false;
    tag.corners_px_ = next;
    return true;
}
bool search_signed_edge(const cv::Mat& gray, Point p, Point q, Point& normal, float& offset) {
    constexpr int kStations = 7;
    constexpr int kRadius = 16;
    constexpr float kStepPx = .25F;
    const Point along = q - p;
    const float length_px = length(along);
    const Point outward = Point(along.y, -along.x) / length_px;
    std::array<std::array<float, 2 * kRadius + 1>, kStations> response;
    for (int i = 0; i < kStations; ++i) {
        const Point at = p + ((i + 1) * .125F) * along;
        for (int j = -kRadius; j <= kRadius; ++j) {
            const float outer = bilinear(gray, at + (j * kStepPx + .65F) * outward);
            const float inner = bilinear(gray, at + (j * kStepPx - .65F) * outward);
            response[i][j + kRadius] = std::clamp(outer - inner, -30.F, 90.F);
        }
    }

    int best_a = 0, best_b = 0;
    float best_score = 0.F;
    int best_support = 0;
    const auto check = [&](int a, int b) {
        float score = 0.F;
        int support = 0;
        for (int i = 0; i < kStations; ++i) {
            const float x = ((7 - i) * a + (i + 1) * b) * .125F + kRadius;
            const int left = static_cast<int>(x);
            const float frac = x - left;
            const auto& ray = response[i];
            const float value = ray[left] + frac * (ray[left + 1] - ray[left]);
            score += value;
            support += value > 22.F;
        }
        score -= .2F * (std::abs(a) + std::abs(b));
        if (score > best_score) {
            best_score = score;
            best_a = a;
            best_b = b;
            best_support = support;
        }
    };
    for (int a = -12; a <= 12; a += 2)
        for (int b = -12; b <= 12; b += 2)
            check(a, b);
    const int a = best_a, b = best_b;
    for (int i = std::max(-12, a - 1); i <= std::min(12, a + 1); ++i)
        for (int j = std::max(-12, b - 1); j <= std::min(12, b + 1); ++j)
            check(i, j);
    if (best_support < 5)
        return false;
    normal = outward - ((best_b - best_a) * kStepPx / (length_px * length_px)) * along;
    offset = normal.dot(p) + best_a * kStepPx;
    return true;
}

bool recover_signed_boundary(const cv::Mat& gray, const Quad& q, Quad& out) {
    std::array<Point, 4> normals;
    std::array<float, 4> offsets;
    for (int j = 0; j < 4; ++j) {
        const auto p = q[j];
        if (!(p.x >= 6.F && p.y >= 6.F && p.x < gray.cols - 7.F && p.y < gray.rows - 7.F))
            return false;
    }
    for (int j = 0; j < 4; ++j)
        if (!search_signed_edge(gray, q[j], q[(j + 1) % 4], normals[j], offsets[j]))
            return false;
    for (int j = 0; j < 4; ++j) {
        const int i = (j + 3) % 4;
        const auto& a = normals[i];
        const auto& b = normals[j];
        const float det = a.cross(b);
        if (det < .40F)
            return false;
        const float u = offsets[i] - a.dot(q[j]), v = offsets[j] - b.dot(q[j]);
        const Point move((u * b.y - a.y * v) / det, (a.x * v - u * b.x) / det);
        if (length(move) > 3.5F)
            return false;
        out[j] = q[j] + move;
    }
    return true;
}

}  // namespace

void refine_small_tag_corners(const cv::Mat& gray, std::vector<Detection>& tags,
                              const SmallTagRefinementConfig& config) {
    if (!config.enabled_ || gray.total() < config.minimum_frame_pixels_ || tags.empty())
        return;

    constexpr int kMinimumRingMatches = 25;
    constexpr float kMinimumContrast = 7.5F;
    for (auto& tag : tags) {
        const auto& input = tag.corners_px_;
        float shortest_px = std::numeric_limits<float>::infinity();
        float longest_px = 0.F;
        for (std::size_t side = 0; side < input.size(); ++side) {
            const float side_px = length(input[(side + 1) % input.size()] - input[side]);
            shortest_px = std::min(shortest_px, side_px);
            longest_px = std::max(longest_px, side_px);
        }
        if (!(shortest_px >= config.minimum_side_px_ && longest_px <= config.maximum_side_px_))
            continue;

        const auto read = decode<true>(gray, input, kMinimumRingMatches, kMinimumContrast, 0);
        if (read.id == tag.id_ && read.rotation == 0 && tag.corrected_bits_ == 0) {
            refine_aperture(gray, tag, config.maximum_relative_residual_);
        } else if (read.id < 0 && tag.corrected_bits_ != 0) {
            // A coarse corrected word can seed a native fit only if the current
            // pixels confirm the same oriented word both before and after fitting.
            auto measured = tag;
            if (!recover_signed_boundary(gray, input, measured.corners_px_))
                continue;
            if (!verify_payload<true>(gray, measured.corners_px_, tag.id_, kMinimumRingMatches,
                                      kMinimumContrast, 0) ||
                !refine_aperture(gray, measured, config.maximum_relative_residual_))
                continue;
            measured.corrected_bits_ = 0;
            tag = measured;
        }
    }
}

}  // namespace aruco3cuda::apriltag_small_cpu::detail
