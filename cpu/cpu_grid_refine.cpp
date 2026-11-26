// SPDX-License-Identifier: Apache-2.0
#include "cpu_grid_refine.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>

namespace aruco3cuda::cpu::detail {
namespace {
using Point = cv::Point2f;
constexpr int kCells = 8;
constexpr int kMaximumRadius = 32;
const std::array<Point, 4> kUnitSquare{{{0, 0}, {1, 0}, {1, 1}, {0, 1}}};

Point project(const cv::Mat& homography, double u, double v) {
    const double* m = homography.ptr<double>();
    const double inverse = 1.0 / (m[6] * u + m[7] * v + m[8]);
    return Point(static_cast<float>((m[0] * u + m[1] * v + m[2]) * inverse),
                 static_cast<float>((m[3] * u + m[4] * v + m[5]) * inverse));
}

float length(Point delta) {
    return std::sqrt(delta.dot(delta));
}

float sample_inside(const cv::Mat& image, Point p) {
    const int x = static_cast<int>(p.x);
    const int y = static_cast<int>(p.y);
    const float u = p.x - x;
    const float v = p.y - y;
    const auto* upper = image.ptr<unsigned char>(y) + x;
    const auto* lower = image.ptr<unsigned char>(y + 1) + x;
    return (upper[0] + u * (upper[1] - upper[0])) * (1 - v) +
           (lower[0] + u * (lower[1] - lower[0])) * v;
}

bool inside(const cv::Mat& image, Point p) {
    return p.x >= 0 && p.y >= 0 && p.x < image.cols - 1 && p.y < image.rows - 1;
}

float sample(const cv::Mat& image, Point p) {
    return inside(image, p) ? sample_inside(image, p) : -1.F;
}

int cell(const cv::Mat& cells, int x, int y) {
    if (x < 0 || x >= kCells || y < 0 || y >= kCells)
        return 255;
    return cells.at<unsigned char>(y, x);
}

struct NormalEquations {
    cv::Matx<double, 8, 8> matrix_ = cv::Matx<double, 8, 8>::zeros();
    cv::Vec<double, 8> rhs_ = cv::Vec<double, 8>::all(0);
    int count_ = 0;

    void add(Point relative, Point normal, double weight, double displacement) {
        const double x = relative.x, y = relative.y;
        const double nx = normal.x, ny = normal.y;
        const double projection = nx * x + ny * y;
        // First-order displacement along the edge normal under an eight-parameter homography.
        const std::array<double, 8> jacobian{
                nx * x, nx * y, nx, ny * x, ny * y, ny, -x * projection, -y * projection};
        for (int row = 0; row < 8; ++row) {
            this->rhs_[row] += weight * jacobian[row] * displacement;
            for (int col = row; col < 8; ++col)
                this->matrix_(row, col) += weight * jacobian[row] * jacobian[col];
        }
        ++this->count_;
    }
};

void measure_transition(const cv::Mat& image, Point p, Point normal, int radius, Point center,
                        float span, NormalEquations& system) {
    // Omit a complete normal scan when any sample could reach outside the visible image.
    if (!(p.x >= radius + 2 && p.y >= radius + 2 && p.x < image.cols - radius - 2 &&
          p.y < image.rows - radius - 2))
        return;
    float mass = 0, moment = 0, peak = 0;
    // Dark and light probes are four half-pixel steps apart. The visibility check
    // above covers this whole interval. Buffer the scalar profile once so sampling
    // does not depend on the preceding weights or on a circular-buffer update.
    constexpr int kLag = 4;
    std::array<float, 4 * kMaximumRadius + kLag + 1> luminance;
    const int intervals = 4 * radius;
    for (int i = 0; i <= intervals + kLag; ++i) {
        const float offset = (i - 2 * radius - 2) * .5F;
        luminance[i] = sample_inside(image, p + offset * normal);
    }
    for (int i = 0; i <= intervals; ++i) {
        const float offset = (i - 2 * radius) * .5F;
        const float dark = luminance[i], light = luminance[i + kLag];
        const float gradient = std::max(0.0F, light - dark);
        const float weight = gradient * gradient;
        mass += weight;
        moment += weight * offset;
        peak = std::max(peak, gradient);
    }
    if (peak >= 8.0F)
        system.add((p - center) * (1.0F / span), normal, std::min(peak, 100.0F), moment / mass);
}

struct Line {
    Point anchor_;
    Point direction_;
};

struct EdgeSample {
    float position_;
    float offset_px_;
    float strength_;
};

float signed_gradient(const cv::Mat& gray, Point location, Point normal, float offset_px) {
    const float inside = sample(gray, location + (offset_px - 1) * normal);
    const float outside = sample(gray, location + (offset_px + 1) * normal);
    return inside < 0 || outside < 0 ? 0.0F : std::max(0.0F, outside - inside);
}

Line locate_outer_edge(const cv::Mat& gray, Point start, Point end, int limit_px) {
    const Point delta = end - start;
    const float span = length(delta);
    const Line initial{start, delta * (1.0F / span)};
    const Point normal(initial.direction_.y, -initial.direction_.x);
    const int radius = std::clamp(static_cast<int>(std::ceil(span / 16)), 2, limit_px);
    constexpr int kCount = 24;
    std::array<EdgeSample, kCount> samples{};
    for (int i = 0; i < kCount; ++i) {
        const float t = .07F + i * (.86F / (kCount - 1));
        const Point p = start + t * delta;
        float best = 0, offset = 0;
        // The two-pixel gradient reuses its outside sample two steps later.
        constexpr int kSampleLag = 2;
        std::array<float, kSampleLag> previous{
                sample(gray, p + static_cast<float>(-radius - 1) * normal),
                sample(gray, p + static_cast<float>(-radius) * normal)};
        for (int k = -radius; k <= radius; ++k) {
            const int index = (k + radius) % kSampleLag;
            const float inside = previous[index];
            const float outside = sample(gray, p + (static_cast<float>(k) + 1) * normal);
            previous[index] = outside;
            const float gradient =
                    inside < 0 || outside < 0 ? 0.0F : std::max(0.0F, outside - inside);
            const float value = gradient * (1 - .32F * std::abs(k) / radius);
            if (value > best) {
                best = value;
                offset = static_cast<float>(k);
            }
        }
        float mass = 0, moment = 0;
        for (int k = -4; k <= 4; ++k) {
            const float d = offset + .5F * k;
            const float gradient = signed_gradient(gray, p, normal, d);
            mass += gradient * gradient;
            moment += d * gradient * gradient;
        }
        samples[i] = {t, mass > 0 ? moment / mass : offset, best};
    }
    // Opposite-half sample pairs seed a robust line. Weight each hypothesis by
    // support across the whole edge before refining its intercept and slope.
    float best_support = 0, intercept = 0, slope = 0;
    constexpr float kResidualPx = 2.5F;
    for (int i = 0; i < kCount / 2; ++i) {
        const auto first = samples[i], second = samples[i + kCount / 2];
        if (first.strength_ < 1 || second.strength_ < 1)
            continue;
        const float m =
                (second.offset_px_ - first.offset_px_) / (second.position_ - first.position_);
        const float b = first.offset_px_ - first.position_ * m;
        float support = 0;
        for (const auto& value : samples)
            support += value.strength_ *
                       std::max(0.0F, 1 - std::abs(value.offset_px_ - b - m * value.position_) /
                                                      kResidualPx);
        if (support > best_support) {
            best_support = support;
            intercept = b;
            slope = m;
        }
    }
    float mass = 0, t_sum = 0, tt_sum = 0, offset_sum = 0, product_sum = 0;
    for (const auto& value : samples) {
        const float residual = value.offset_px_ - intercept - slope * value.position_;
        const float weight = value.strength_ *
                             std::max(0.0F, 1 - residual * residual / (kResidualPx * kResidualPx));
        mass += weight;
        t_sum += weight * value.position_;
        tt_sum += weight * value.position_ * value.position_;
        offset_sum += weight * value.offset_px_;
        product_sum += weight * value.offset_px_ * value.position_;
    }
    const float determinant = mass * tt_sum - t_sum * t_sum;
    if (mass < 32 || determinant < .018F * mass * mass)
        return initial;
    slope = (mass * product_sum - t_sum * offset_sum) / determinant;
    intercept = (offset_sum - slope * t_sum) / mass;
    const auto direction = delta + slope * normal;
    return {start + intercept * normal, direction * (1.0F / length(direction))};
}

float cross(Point a, Point b) {
    return a.x * b.y - a.y * b.x;
}

void align_outer_edges(const cv::Mat& gray, int limit_px, std::vector<Point>& quad) {
    std::array<Line, 4> lines{};
    std::array<Point, 4> next{};
    for (int i = 0; i < 4; ++i) {
        const float span = length(quad[(i + 1) % 4] - quad[i]);
        if (!(span >= 8 && span <= 4.0F * std::max(gray.cols, gray.rows)))
            return;
        lines[i] = locate_outer_edge(gray, quad[i], quad[(i + 1) % 4], limit_px);
    }
    for (int i = 0; i < 4; ++i) {
        const auto& a = lines[(i + 3) % 4];
        const auto& b = lines[i];
        const float denominator = cross(a.direction_, b.direction_);
        if (std::abs(denominator) < .12F)
            return;
        next[i] = a.anchor_ +
                  (cross(b.anchor_ - a.anchor_, b.direction_) / denominator) * a.direction_;
        if (!(length(next[i] - quad[i]) <= .2F * length(quad[(i + 1) % 4] - quad[i])))
            return;
    }
    std::copy(next.begin(), next.end(), quad.begin());
}

enum class TransitionSamples { kTwoPerCell, kMidpoint };

bool fit_payload(const cv::Mat& image, const cv::Mat& cells, int radius_limit_px,
                 TransitionSamples sampling, std::vector<Point>& quad) {
    const cv::Mat transform = cv::getPerspectiveTransform(kUnitSquare.data(), quad.data());
    const Point center = project(transform, .5F, .5F);
    float span = 0;
    for (int i = 0; i < 4; ++i)
        span += .25F * length(quad[(i + 1) % 4] - quad[i]);
    if (!(span >= 18 && span <= 4.0F * std::max(image.cols, image.rows)))
        return false;
    const int radius = std::clamp(static_cast<int>(span / 32), 2, radius_limit_px);
    // Horizontal and vertical transitions share their projected endpoints.
    std::array<std::array<Point, kCells + 1>, kCells + 1> vertices;
    for (int y = 0; y <= kCells; ++y) {
        for (int x = 0; x <= kCells; ++x)
            vertices[y][x] = project(transform, double(x) / kCells, double(y) / kCells);
    }
    NormalEquations system;
    for (int axis = 0; axis < 2; ++axis) {
        for (int j = 0; j < kCells; ++j) {
            for (int i = 0; i <= kCells; ++i) {
                const int first = axis ? cell(cells, j, i - 1) : cell(cells, i - 1, j);
                const int second = axis ? cell(cells, j, i) : cell(cells, i, j);
                if (first == second)
                    continue;
                const Point start = axis ? vertices[i][j] : vertices[j][i];
                const Point end = axis ? vertices[i][j + 1] : vertices[j + 1][i];
                const Point edge = end - start;
                const float cell_width = length(edge);
                if (!(cell_width >= 2.25F && std::isfinite(cell_width)))
                    continue;
                Point normal = Point(edge.y, -edge.x) * (1.0F / cell_width);
                if (axis)
                    normal = -normal;
                if (second < first)
                    normal = -normal;
                const int local_radius = std::max(
                        1,
                        static_cast<int>(std::min(static_cast<float>(radius), cell_width * .3F)));
                if (sampling == TransitionSamples::kMidpoint) {
                    measure_transition(image, start + .50F * edge, normal, local_radius, center,
                                       span, system);
                } else {
                    for (float fraction : {.30F, .70F})
                        measure_transition(image, start + fraction * edge, normal, local_radius,
                                           center, span, system);
                }
            }
        }
    }
    if (system.count_ < 16)
        return false;
    for (int row = 0; row < 8; ++row)
        for (int col = row + 1; col < 8; ++col)
            system.matrix_(col, row) = system.matrix_(row, col);
    cv::Vec<double, 8> solution;
    if (!cv::solve(system.matrix_, system.rhs_, solution, cv::DECOMP_CHOLESKY))
        return false;
    for (Point& point : quad) {
        const Point relative = (point - center) * (1.0F / span);
        const double x = relative.x, y = relative.y;
        const double denominator_delta = solution[6] * x + solution[7] * y;
        const Point shift(static_cast<float>(solution[0] * x + solution[1] * y + solution[2] -
                                             x * denominator_delta),
                          static_cast<float>(solution[3] * x + solution[4] * y + solution[5] -
                                             y * denominator_delta));
        if (!(length(shift) < std::min(span * .13F, 9.0F)))
            return false;
        point += shift;
    }
    return cv::isContourConvex(quad) && cv::contourArea(quad, true) > 0;
}
bool validate_fit(const cv::Mat& gray, const cv::Mat& cells, int radius_limit_px,
                  const std::vector<Point>& corners) {
    return gray.type() == CV_8UC1 && cells.type() == CV_8UC1 &&
           cells.size() == cv::Size(kCells, kCells) && corners.size() == 4 &&
           radius_limit_px >= 2 && radius_limit_px <= kMaximumRadius &&
           cv::isContourConvex(corners) && cv::contourArea(corners, true) > 0.0;
}

bool has_visible_cells(const cv::Mat& gray, const std::vector<Point>& quad) {
    // A decoded patch may come from a warp that pads the frame. Check the source pixels.
    const auto transform = cv::getPerspectiveTransform(kUnitSquare.data(), quad.data());
    for (int y = 0; y < kCells; ++y)
        for (int x = 0; x < kCells; ++x)
            if (!inside(gray, project(transform, (x + .5F) / kCells, (y + .5F) / kCells)))
                return false;
    return true;
}
}  // namespace

bool fit_decoded_grid(const cv::Mat& gray, const cv::Mat& cells, int radius_limit_px,
                      std::vector<cv::Point2f>& corners) {
    if (!validate_fit(gray, cells, radius_limit_px, corners))
        return false;
    auto fitted = corners;
    align_outer_edges(gray, std::min(32, 6 * radius_limit_px), fitted);
    if (!fit_payload(gray, cells, radius_limit_px, TransitionSamples::kTwoPerCell, fitted) ||
        !has_visible_cells(gray, fitted))
        return false;
    corners.swap(fitted);
    return true;
}

bool fit_decoded_payload_grid(const cv::Mat& gray, const cv::Mat& cells, int radius_limit_px,
                              std::vector<cv::Point2f>& corners) {
    if (!validate_fit(gray, cells, radius_limit_px, corners))
        return false;
    auto fitted = corners;
    if (!fit_payload(gray, cells, radius_limit_px, TransitionSamples::kMidpoint, fitted) ||
        !has_visible_cells(gray, fitted))
        return false;
    corners.swap(fitted);
    return true;
}

bool fit_visible_payload_grid(const cv::Mat& gray, const cv::Mat& cells, int radius_limit_px,
                              std::vector<cv::Point2f>& corners) {
    if (!validate_fit(gray, cells, radius_limit_px, corners))
        return false;
    auto fitted = corners;
    if (!fit_payload(gray, cells, radius_limit_px, TransitionSamples::kMidpoint, fitted))
        return false;
    corners.swap(fitted);
    return true;
}

bool visible_grid_matches(const cv::Mat& gray, const cv::Mat& cells,
                          const std::vector<cv::Point2f>& corners, int correction_budget,
                          int border_budget) {
    const auto transform = cv::getPerspectiveTransform(kUnitSquare.data(), corners.data());
    std::array<float, kCells * kCells> samples{};
    std::array<float, 2> sums{};
    std::array<int, 2> counts{};
    for (int y = 0; y < kCells; ++y) {
        for (int x = 0; x < kCells; ++x) {
            const auto at = project(transform, (x + .5F) / kCells, (y + .5F) / kCells);
            const auto luminance = sample(gray, at);
            if (luminance < 0)
                return false;
            const int color = cell(cells, x, y) != 0;
            sums[color] += luminance;
            ++counts[color];
            samples[y * kCells + x] = luminance;
        }
    }
    if (counts[0] == 0 || counts[1] == 0)
        return false;
    const float black = sums[0] / counts[0], white = sums[1] / counts[1];
    if (white - black < 8)
        return false;
    const float threshold = (black + white) * .5F;
    int border_errors = 0, payload_errors = 0;
    for (int y = 0; y < kCells; ++y)
        for (int x = 0; x < kCells; ++x) {
            if ((samples[y * kCells + x] > threshold) == (cell(cells, x, y) != 0))
                continue;
            if (x == 0 || y == 0 || x == kCells - 1 || y == kCells - 1)
                ++border_errors;
            else
                ++payload_errors;
        }
    return border_errors <= border_budget && payload_errors <= correction_budget;
}
}  // namespace aruco3cuda::cpu::detail
