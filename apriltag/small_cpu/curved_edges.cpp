// SPDX-License-Identifier: Apache-2.0
#include "curved_edges.hpp"
#include "aruco3cuda/apriltag/small_cpu/video_detector.hpp"
#include "tag_decode.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace aruco3cuda::apriltag_small_cpu::detail {
namespace {
using Grid = cv::Matx<std::uint8_t, 8, 8>;
constexpr int kCapacity = 640;

struct Crossing {
    float t, y;
};
using Samples = std::array<Crossing, kCapacity>;
struct Curve {
    bool horizontal;
    float mid, inv_radius, height, existing_slope;
    std::array<double, 4> c{};
    std::pair<double, double> value_and_slope(double x) const {
        const double t = (x - mid) * inv_radius;
        return {height + (existing_slope * (x - mid)) + ((c[3] * t + c[2]) * t + c[1]) * t + c[0],
                existing_slope + ((3 * c[3] * t + 2 * c[2]) * t + c[1]) * inv_radius};
    }
};

Point image_point(const Projection& p, float u, float v) {
    return (p.initial + u * p.x + v * p.y) / (1 + u * p.z_x + v * p.z_y);
}

Point normal_displacement(const Curve& edge, Point flat) {
    const auto [value, slope] = edge.value_and_slope(edge.horizontal ? flat.x : flat.y);
    const double delta = (value - (edge.horizontal ? flat.y : flat.x)) / (1 + slope * slope);
    return edge.horizontal ? Point(-slope * delta, delta) : Point(delta, -slope * delta);
}

bool read_levels(const cv::Mat& gray, const Grid& cells, const Quad& corners,
                 std::array<int, 2>& levels, bool final,
                 const std::array<Curve, 4>* edges = nullptr) {
    Projection p;
    if (!project(corners, p))
        return false;
    std::uint8_t groups[2][64];
    int count[2]{};
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            const float u = (x + .5F) / 8, v = (y + .5F) / 8;
            Point center = image_point(p, u, v);
            if (edges) {
                center += (1 - v) * normal_displacement((*edges)[0], image_point(p, u, 0)) +
                          u * normal_displacement((*edges)[1], image_point(p, 1, v)) +
                          v * normal_displacement((*edges)[2], image_point(p, u, 1)) +
                          (1 - u) * normal_displacement((*edges)[3], image_point(p, 0, v));
            }
            if (!(center.x >= 0 && center.x <= gray.cols - 1 && center.y >= 0 &&
                  center.y <= gray.rows - 1))
                return false;
            const int bit = cells(y, x);
            groups[bit][count[bit]++] =
                    gray.ptr<std::uint8_t>(cvRound(center.y))[cvRound(center.x)];
        }
    }
    if (!count[0] || !count[1])
        return false;
    if (!final) {
        for (int bit = 0; bit < 2; ++bit) {
            auto* group = groups[bit];
            std::nth_element(group, group + count[bit] / 2, group + count[bit]);
            levels[bit] = group[count[bit] / 2];
        }
        if (levels[1] - levels[0] < 28)
            return false;
    }
    int ambiguous = 0, errors = 0;
    const int mid = (levels[0] + levels[1]) / 2;
    for (int bit = 0; bit < 2; ++bit) {
        for (int i = 0; i < count[bit]; ++i) {
            const int v = groups[bit][i];
            ambiguous += std::min(std::abs(v - levels[0]), std::abs(v - levels[1])) > 6;
            errors += (v > mid) != bit;
        }
    }
    return ambiguous <= (final ? 6 : 12) && errors <= (final ? 2 : 8);
}

bool least_squares(const Samples& data, int n, const std::array<double, 4>* estimate,
                   std::array<double, 4>& solution) {
    double moments[7]{}, rhs[4]{}, sum_squared = 0;
    for (int i = 0; i < n; ++i) {
        const double t = data[i].t, y = data[i].y;
        double weight = 1;
        if (estimate) {
            const auto& c = *estimate;
            const double residual = y - (((c[3] * t + c[2]) * t + c[1]) * t + c[0]);
            const double ratio = .4 / std::max(.4, std::abs(residual));
            weight = ratio * ratio;
        }
        sum_squared += weight * y * y;
        for (int j = 0; j < 7; ++j) {
            moments[j] += weight;
            if (j < 4)
                rhs[j] += weight * y;
            weight *= t;
        }
    }
    // Leading blocks of the same Cholesky factor solve nested linear, quadratic and cubic fits.
    double factor[4][4]{}, z[4]{};
    int terms = 4;
    double best = std::numeric_limits<double>::infinity();
    const double penalty = 3 * std::log(n);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j <= i; ++j) {
            double value = moments[i + j];
            for (int k = 0; k < j; ++k)
                value -= factor[i][k] * factor[j][k];
            if (i == j) {
                if (!(value > 1e-10))
                    return false;
                factor[i][i] = std::sqrt(value);
            } else {
                factor[i][j] = value / factor[j][j];
            }
        }
        double residual = rhs[i];
        for (int k = 0; k < i; ++k)
            residual -= factor[i][k] * z[k];
        z[i] = residual / factor[i][i];
        sum_squared -= z[i] * z[i];
        if (estimate && i >= 1) {
            const double criterion =
                    n * std::log(std::max(.00015, sum_squared / n)) + (i + 1) * penalty;
            if (criterion < best) {
                best = criterion;
                terms = i + 1;
            }
        }
    }
    solution = {};
    for (int i = terms - 1; i >= 0; --i) {
        double value = z[i];
        for (int j = i + 1; j < terms; ++j)
            value -= factor[j][i] * solution[j];
        solution[i] = value / factor[i][i];
    }
    return true;
}

bool interval_line(const Samples& data, int n, bool binary, std::array<double, 4>& c) {
    // Offsets at the first and last samples parameterize all admissible lines.
    // Each additional pixel clips this convex parameter polygon by its measured interval.
    const double half = binary ? .50001 : .18;
    using Polygon = std::array<cv::Point2d, 40>;
    Polygon slots[2];
    slots[0][0] = {-half, -half};
    slots[0][1] = {half, -half};
    slots[0][2] = {half, half};
    slots[0][3] = {-half, half};
    const auto first = data[0], last = data[n - 1];
    const double span = last.t - first.t, rise = last.y - first.y;
    int selected = 0, count = 4;
    for (int i = 1; i + 1 < n; ++i) {
        const double t = (data[i].t - first.t) / span;
        const double delta = (data[i].y - first.y) - rise * t;
        if (std::abs(delta) < 1e-7)
            continue;
        if (std::abs(delta) > 2 * half)
            return false;
        const double sign = delta > 0 ? 1 : -1;
        auto margin = [&](cv::Point2d q) { return half + sign * (q.x + t * (q.y - q.x) - delta); };
        const auto& polygon = slots[selected];
        auto& output = slots[1 - selected];
        int used = 0;
        auto a = polygon[count - 1];
        double before = margin(a);
        for (int j = 0; j < count; ++j) {
            if (used + 2 > static_cast<int>(output.size()))
                return false;
            const auto b = polygon[j];
            const double now = margin(b);
            if ((now < 0) != (before < 0))
                output[used++] = a + (before / (before - now)) * (b - a);
            if (now >= 0)
                output[used++] = b;
            a = b;
            before = now;
        }
        if (used < 3)
            return false;
        count = used;
        selected = 1 - selected;
    }
    if (binary) {
        const auto& polygon = slots[selected];
        cv::Point2d centroid{};
        double area2 = 0;
        for (int i = 0, prev = count - 1; i < count; prev = i++) {
            const double weight = polygon[prev].cross(polygon[i]);
            area2 += weight;
            centroid += weight * (polygon[prev] + polygon[i]);
        }
        if (!(area2 > 1e-10))
            return false;
        centroid *= 1 / (3 * area2);
        c[1] = (rise + centroid.y - centroid.x) / span;
        c[0] = first.y + centroid.x - c[1] * first.t;
    } else {
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (int i = 0; i < n; ++i) {
            const double t = data[i].t, y = data[i].y;
            sx += t;
            sy += y;
            sxx += t * t;
            sxy += t * y;
        }
        c[1] = (sxy - sx * sy / n) / (sxx - sx * sx / n);
        c[0] = (sy - c[1] * sx) / n;
    }
    c[2] = c[3] = 0;
    return true;
}

enum class EdgePass { proposal, measurement, initial, refinement };

bool fit_edge(const cv::Mat& gray, const Quad& quad, int side, const std::array<int, 2>& levels,
              EdgePass pass, Curve& curve) {
    Point a = quad[side], b = quad[(side + 1) % 4];
    curve.horizontal = std::abs(b.x - a.x) >= std::abs(b.y - a.y);
    if (!curve.horizontal) {
        std::swap(a.x, a.y);
        std::swap(b.x, b.y);
    }
    const float dx = b.x - a.x, span = std::abs(dx);
    if (!(span >= 26 && span < 3000))
        return false;
    curve.mid = (a.x + b.x) * .5F;
    curve.height = (a.y + b.y) * .5F;
    curve.existing_slope = (b.y - a.y) / dx;
    curve.inv_radius = 2 / span;
    const auto inset = [&](Point endpoint, Point neighbour) {
        if (!curve.horizontal)
            std::swap(neighbour.x, neighbour.y);
        const Point tangent = neighbour - endpoint;
        const float determinant = std::abs(dx * tangent.y - (b.y - a.y) * tangent.x);
        // Keep the six-pixel normal profile clear of the adjacent edge at this corner.
        const float clearance = 3.F * std::abs(dx * tangent.x) / std::max(.0001F, determinant);
        return std::min(.12F * span, .25F + clearance);
    };
    int begin, last;
    if (pass == EdgePass::measurement) {
        const float enter = inset(a, quad[(side + 3) % 4]);
        const float leave = inset(b, quad[(side + 2) % 4]);
        begin = cvCeil(dx > 0 ? a.x + enter : b.x + leave);
        last = cvFloor(dx > 0 ? b.x - leave : a.x - enter);
    } else {
        const float inner =
                pass == EdgePass::refinement ? std::max(1.75F, .012F * span) : .0425F * span;
        begin = cvCeil(curve.mid - span * .5F + inner);
        last = cvFloor(curve.mid + span * .5F - inner);
    }
    const int cols = curve.horizontal ? gray.cols : gray.rows,
              rows = curve.horizontal ? gray.rows : gray.cols;
    if (begin < 0 || last >= cols || last < begin)
        return false;
    const int stride = std::max(1, (last - begin + kCapacity) / kCapacity);
    const auto major = curve.horizontal ? 1U : gray.step[0],
               minor = curve.horizontal ? gray.step[0] : 1U;
    const int direction = ((dx > 0) == curve.horizontal) ? 1 : -1;
    const int left = levels[direction > 0], right = levels[direction < 0],
              mid = levels[0] + levels[1];
    const int search = std::clamp(cvRound(.09F * span), 4, 72);
    Samples data;
    int n = 0, intermediate = 0;
    const bool proposal = pass == EdgePass::proposal;
    const int possible = (last - begin) / stride + 1;
    const int count = proposal ? std::min(possible, 16) : possible;
    for (int i = 0; i < count; ++i) {
        const int x = begin + (proposal ? i * (last - begin) / std::max(1, count - 1) : i * stride);
        const float predicted = curve.height + (x - curve.mid) * curve.existing_slope;
        const int center = cvFloor(predicted);
        for (int step = 0; step <= 2 * search; ++step) {
            const int dy = (step & 1) ? -(step + 1) / 2 : step / 2;
            const int y = center + dy;
            if (y < 2 || y + 3 >= rows)
                continue;
            const auto* z = gray.data + x * major + (y - 2) * minor;
            const int v0 = z[2 * minor], v1 = z[3 * minor];
            if ((2 * v0 - mid) * direction < 0 || (2 * v1 - mid) * direction > 0 ||
                (v0 - v1) * direction <= 0 || std::abs(z[0] - left) > 8 ||
                std::abs(z[5 * minor] - right) > 8)
                continue;
            int sum = 0;
            for (int k = 0; k < 6; ++k) {
                const int v = z[k * minor];
                sum += v;
                intermediate += std::min(std::abs(v - levels[0]), std::abs(v - levels[1])) > 2;
            }
            const float edge = y - 2.5F + (sum - 6 * right) / static_cast<float>(left - right);
            data[n++] = {(x - curve.mid) * curve.inv_radius, edge - predicted};
            break;
        }
    }
    if (n < (proposal ? 7 : 14) || n * 3 < count * 2)
        return false;
    if (proposal) {
        if (!least_squares(data, n, nullptr, curve.c))
            return false;
    } else if (!interval_line(data, n, intermediate == 0, curve.c)) {
        std::array<double, 4> seed;
        if (!least_squares(data, n, nullptr, seed) || !least_squares(data, n, &seed, curve.c))
            return false;
    }
    int good = 0;
    for (int i = 0; i < n; ++i) {
        const auto& c = curve.c;
        const float t = data[i].t;
        good += std::abs((((c[3] * t + c[2]) * t + c[1]) * t + c[0]) - data[i].y) < .62;
    }
    return good * 4 >= 3 * n;
}

bool intersect(const Curve& a, const Curve& b, Point start, float limit, Point& q) {
    q = start;
    for (int i = 0; i < 5; ++i) {
        const auto [ya, sa] = a.value_and_slope(a.horizontal ? q.x : q.y);
        const auto [yb, sb] = b.value_and_slope(b.horizontal ? q.x : q.y);
        const double ax = a.horizontal ? -sa : 1, ay = a.horizontal ? 1 : -sa,
                     bx = b.horizontal ? -sb : 1, by = b.horizontal ? 1 : -sb;
        const double da = ya - (a.horizontal ? q.y : q.x), db = yb - (b.horizontal ? q.y : q.x);
        const double det = ax * by - ay * bx;
        if (std::abs(det) < .2)
            return false;
        Point move((da * by - db * ay) / det, (ax * db - bx * da) / det);
        q += move;
        if (!(length(q - start) < limit))
            return false;
        if (length(move) < .001F)
            return true;
    }
    return false;
}
bool fit_pass(const cv::Mat& gray, const Grid& cells, std::array<int, 2>& levels, float span,
              EdgePass pass, Quad& corners, bool& repeat) {
    std::array<Curve, 4> edges;
    bool nonlinear = false;
    for (int i = 0; i < 4; ++i) {
        if (!fit_edge(gray, corners, i, levels, pass, edges[i]))
            return false;
        nonlinear |= edges[i].c[2] != 0 || edges[i].c[3] != 0;
    }
    Quad trial;
    float move = 0;
    for (int i = 0; i < 4; ++i) {
        if (!intersect(edges[(i + 3) % 4], edges[i], corners[i], std::max(4.F, .12F * span),
                       trial[i]))
            return false;
        move = std::max(move, length(trial[i] - corners[i]));
    }
    // Strong distortion can move straight-homography sample centers across a bit transition.
    // Check all 64 oriented cells, using the measured edge offsets if needed.
    if (pass != EdgePass::proposal && !read_levels(gray, cells, trial, levels, true) &&
        !read_levels(gray, cells, trial, levels, true, &edges))
        return false;
    corners = trial;
    repeat = move >= (nonlinear ? .5F : 1.5F);
    return true;
}
}  // namespace

bool fit_curved_edges(const cv::Mat& gray, const Grid& cells, Quad& corners) {
    const float span = length(corners[1] - corners[0]);
    std::array<int, 2> levels{};
    if (!(span >= 58 && span < 2800) || !read_levels(gray, cells, corners, levels, false))
        return false;

    // A sparse cubic estimates the search location only. Publish the corners from the
    // full native profile fit after the oriented cell check succeeds.
    Quad trial = corners;
    bool repeat = false;
    if (fit_pass(gray, cells, levels, span, EdgePass::proposal, trial, repeat) &&
        fit_pass(gray, cells, levels, span, EdgePass::measurement, trial, repeat)) {
        corners = trial;
        return true;
    }
    if (!fit_pass(gray, cells, levels, span, EdgePass::initial, corners, repeat))
        return false;
    if (repeat)
        fit_pass(gray, cells, levels, span, EdgePass::refinement, corners, repeat);
    return true;
}

namespace {
constexpr int kRailSamples = 24;
constexpr int kMaximumSearchRadiusPx = 16;

struct RailSample {
    float fraction;
    float offset_px;
};

struct RailTangents {
    Point start;
    Point start_tangent;
    Point end;
    Point end_tangent;
};

bool measure_rail(const cv::Mat& gray, Point start, Point delta, Point normal, int radius_px,
                  std::array<RailSample, kRailSamples>& samples, int& count, float& spread,
                  int& spread_count) {
    constexpr float kSampleSpan = .92F;
    const int bins = 2 * radius_px + 3;
    const int middle = bins / 2;
    count = 0;
    for (int i = 0; i < kRailSamples; ++i) {
        const float fraction = .5F + kSampleSpan * ((i + .5F) / kRailSamples - .5F);
        const Point center = start + fraction * delta;
        // This margin covers every bilinear tap along the rotated search line, including on ROIs.
        if (center.x < radius_px + 3 || center.y < radius_px + 3 ||
            center.x >= gray.cols - radius_px - 4 || center.y >= gray.rows - radius_px - 4)
            continue;
        std::array<float, 2 * kMaximumSearchRadiusPx + 3> profile;
        for (int j = 0; j < bins; ++j)
            profile[j] = bilinear(gray, center + static_cast<float>(j - middle) * normal);

        float energy = 0, weighted_offset = 0, peak = 0;
        for (int j = 1; j < bins - 1; ++j) {
            const float gradient = std::max(0.F, profile[j + 1] - profile[j - 1]);
            peak = std::max(peak, gradient);
            const float weight = gradient * gradient;
            energy += weight;
            weighted_offset += weight * (j - middle);
        }
        if (peak < 9 || energy < 50)
            continue;
        const float amplitude =
                .5F * (profile[bins - 1] + profile[bins - 2] - profile[0] - profile[1]);
        if (amplitude > 15) {
            spread += amplitude / peak;
            ++spread_count;
        }
        samples[count++] = {fraction, weighted_offset / energy};
    }
    return count >= kRailSamples / 3;
}

bool fit_affine_offsets(const std::array<RailSample, kRailSamples>& samples, int count,
                        float& intercept, float& slope) {
    constexpr float kRobustResidualPx = 2.25F;
    intercept = slope = 0;
    for (int pass = 0; pass < 2; ++pass) {
        float weight_sum = 0, t_sum = 0, t_squared_sum = 0, position_sum = 0, cross_sum = 0;
        for (int i = 0; i < count; ++i) {
            const auto [t, position] = samples[i];
            float weight = 1;
            if (pass != 0) {
                const float error = std::abs(position - intercept - slope * t);
                if (error > kRobustResidualPx)
                    weight *= kRobustResidualPx / error;
            }
            weight_sum += weight;
            t_sum += weight * t;
            t_squared_sum += weight * t * t;
            position_sum += weight * position;
            cross_sum += weight * position * t;
        }
        const float determinant = weight_sum * t_squared_sum - t_sum * t_sum;
        if (determinant < 1.e-4F)
            return false;
        slope = (weight_sum * cross_sum - t_sum * position_sum) / determinant;
        intercept = (position_sum - slope * t_sum) / weight_sum;
    }
    return true;
}

bool fit_quadratic_offsets(const std::array<RailSample, kRailSamples>& samples, int count,
                           float intercept, float slope, float maximum_quadratic_px,
                           std::array<float, 3>& coefficients) {
    float system[3][4]{};
    // Fit in centered coordinates. The affine estimate limits the effect of off-rail samples.
    for (int i = 0; i < count; ++i) {
        const auto [fraction, position] = samples[i];
        const float t = fraction - .5F;
        const float weight =
                1.F / std::max(1.F, std::abs(position - intercept - slope * fraction) * .35F);
        const std::array<float, 3> row{1, t, t * t};
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k)
                system[j][k] += weight * row[j] * row[k];
            system[j][3] += weight * position * row[j];
        }
    }
    for (int i = 0; i < 3; ++i) {
        if (std::abs(system[i][i]) < 1.e-8F)
            return false;
        const float inverse = 1 / system[i][i];
        for (int j = i; j < 4; ++j)
            system[i][j] *= inverse;
        for (int k = 0; k < 3; ++k) {
            if (k == i)
                continue;
            const float multiplier = system[k][i];
            for (int j = i; j < 4; ++j)
                system[k][j] -= system[i][j] * multiplier;
        }
    }
    const float offset = system[0][3], drift = system[1][3], bend = system[2][3];
    if (std::abs(bend) > maximum_quadratic_px)
        return false;

    float affine_error = 0, quadratic_error = 0;
    for (int i = 0; i < count; ++i) {
        const auto [fraction, position] = samples[i];
        const float t = fraction - .5F;
        const float fitted_error = position - (offset + drift * t + bend * t * t);
        const float initial_error = position - (intercept + slope * fraction);
        quadratic_error += std::min(25.F, fitted_error * fitted_error);
        affine_error += std::min(25.F, initial_error * initial_error);
    }
    coefficients = quadratic_error <= .85F * affine_error
                           ? std::array<float, 3>{offset, drift, bend}
                           : std::array<float, 3>{intercept + .5F * slope, slope, 0};
    return true;
}
}  // namespace

bool fit_long_tag_edges(const cv::Mat& gray, const Quad& corners, Quad& fitted,
                        const LargeTagRefinementConfig& config, bool straight_rails) {
    std::array<RailTangents, 4> rails;
    float spread = 0;
    int spread_count = 0;
    for (int side = 0; side < 4; ++side) {
        const Point start = corners[side];
        const Point delta = corners[(side + 1) % 4] - start;
        const float length_px = length(delta);
        if (!std::isfinite(length_px) || length_px < config.minimum_side_px_)
            return false;
        const Point normal(delta.y / length_px, -delta.x / length_px);
        std::array<RailSample, kRailSamples> samples;
        int count;
        if (!measure_rail(gray, start, delta, normal, config.search_radius_px_, samples, count,
                          spread, spread_count))
            return false;
        float intercept, slope;
        if (!fit_affine_offsets(samples, count, intercept, slope))
            return false;
        if (straight_rails) {
            rails[side] = {start + intercept * normal, delta + slope * normal,
                           start + delta + (intercept + slope) * normal, delta + slope * normal};
        } else {
            std::array<float, 3> coefficients;
            if (!fit_quadratic_offsets(samples, count, intercept, slope,
                                       config.maximum_quadratic_px_, coefficients))
                return false;
            const auto [offset, drift, bend] = coefficients;
            rails[side] = {start + normal * (offset - .5F * drift + .25F * bend),
                           delta + normal * (drift - bend),
                           start + delta + normal * (offset + .5F * drift + .25F * bend),
                           delta + normal * (drift + bend)};
        }
    }
    // Blurred outer transitions benefit from longer intensity profiles. The ordinary
    // geometry already fits sharp prints with a narrow edge.
    if (!straight_rails && (!spread_count || spread < spread_count * config.minimum_blur_ratio_))
        return false;
    Quad trial;
    for (int side = 0; side < 4; ++side) {
        const auto& previous = rails[(side + 3) % 4];
        const auto& current = rails[side];
        const float determinant = cross(previous.end_tangent, current.start_tangent);
        if (std::abs(determinant) < 10)
            return false;
        trial[side] = previous.end + cross(current.start - previous.end, current.start_tangent) /
                                             determinant * previous.end_tangent;
        if (!(length(trial[side] - corners[side]) <= config.maximum_corner_motion_px_))
            return false;
    }
    fitted = trial;
    return true;
}
}  // namespace aruco3cuda::apriltag_small_cpu::detail
