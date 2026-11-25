// SPDX-License-Identifier: Apache-2.0
#include "grid_alignment.hpp"
#include "curved_edges.hpp"
#include "diffuse_edge_profile.hpp"

#include "cpu_grid_refine.hpp"
#include "tag_decode.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace aruco3cuda::generated {
extern const DictionaryTable kDictApriltag36h11Table;
}

namespace aruco3cuda::apriltag_small_cpu::detail {
namespace {

constexpr int kGridSide = 8;
constexpr float kCellWidth = 1.0F / kGridSide;
constexpr int kSearchLimit = 18;

// Each transition supplies a scalar displacement along its normal in source pixels.
// Positions are centered and scaled by the tag side for the projective least-squares solve.
struct EdgeOffset {
    Point position;
    Point normal;
    float offset;
    float weight;
};
using Offsets = std::array<EdgeOffset, 2 * kGridSide*(kGridSide + 1)>;
using Grid = cv::Matx<std::uint8_t, kGridSide, kGridSide>;
using Parameters = cv::Vec<double, 8>;

Point coordinate(const Projection& p, float u, float v) {
    return (p.initial + u * p.x + v * p.y) / (1 + u * p.z_x + v * p.z_y);
}

int color(const Grid& cells, int x, int y) {
    // A tag36h11 outer black border meets a white quiet zone.
    return x < 0 || x >= kGridSide || y < 0 || y >= kGridSide ? 1 : cells(y, x);
}

bool measure_offset(const cv::Mat& gray, Point center, Point normal, int radius, float& offset,
                    float& strength) {
    if (!(center.x > radius + 3 && center.x < gray.cols - radius - 3 && center.y > radius + 3 &&
          center.y < gray.rows - radius - 3))
        return false;
    std::array<float, 2 * kSearchLimit + 3> gradients;
    int best = 0;
    strength = 0;
    for (int i = -radius - 1; i <= radius + 1; ++i) {
        const float gradient = bilinear(gray, center + (i + 0.65F) * normal) -
                               bilinear(gray, center + (i - 0.65F) * normal);
        gradients[i + radius + 1] = std::max(gradient, 0.0F);
        if (std::abs(i) <= radius && gradient > strength) {
            strength = gradient;
            best = i;
        }
    }
    if (strength < 5.0F)
        return false;
    const int index = best + radius + 1;
    const float left = gradients[index - 1], right = gradients[index + 1];
    const float curvature = std::max(0.001F, 2 * strength - left - right);
    offset = best + std::clamp(0.5F * (right - left) / curvature, -0.65F, 0.65F);
    return true;
}

int observe(const cv::Mat& gray, const Grid& cells, const Projection& p, Point center, float side,
            float radius_ratio, Offsets& offsets) {
    int count = 0;
    for (int axis = 0; axis < 2; ++axis) {
        for (int i = 0; i <= kGridSide; ++i) {
            for (int j = 0; j < kGridSide; ++j) {
                const int before = axis ? color(cells, j, i - 1) : color(cells, i - 1, j);
                const int after = axis ? color(cells, j, i) : color(cells, i, j);
                if (before == after)
                    continue;
                const float u = i * kCellWidth, v = j * kCellWidth;
                const Point start = axis ? coordinate(p, v, u) : coordinate(p, u, v);
                const Point end =
                        axis ? coordinate(p, v + kCellWidth, u) : coordinate(p, u, v + kCellWidth);
                const Point delta = end - start;
                const float span = length(delta);
                if (!(span >= 1.25F && std::isfinite(span)))
                    continue;
                const Point midpoint = (start + end) * 0.5F;
                Point normal = Point(delta.y, -delta.x) * (1 / span);
                if ((before > after) != static_cast<bool>(axis))
                    normal = -normal;
                const int radius = std::clamp(cvRound(span * radius_ratio), 1, kSearchLimit);
                float offset, strength;
                if (!measure_offset(gray, midpoint, normal, radius, offset, strength))
                    continue;
                // Outside edges also respond to texture around the tag. Favor interior evidence.
                const float weight = i == 0 || i == kGridSide ? 0.32F : 1.0F;
                offsets[count++] = {(midpoint - center) * (1 / side), normal, offset,
                                    std::min(strength, 90.0F) * weight};
            }
        }
    }
    return count;
}

bool solve_offsets(const Offsets& offsets, int count, Parameters& solution, int passes) {
    // Reweight once to limit the influence of gradients belonging to a neighboring edge.
    for (int pass = 0; pass < passes; ++pass) {
        auto system = cv::Matx<double, 8, 8>::zeros();
        auto rhs = Parameters::all(0);
        for (int i = 0; i < count; ++i) {
            const auto& m = offsets[i];
            const Point p = m.position, n = m.normal;
            const double projection = n.dot(p);
            const Parameters jacobian(n.x * p.x, n.x * p.y, n.x, n.y * p.x, n.y * p.y, n.y,
                                      -p.x * projection, -p.y * projection);
            const double error = std::abs(m.offset - jacobian.dot(solution));
            const double weight = m.weight / (pass ? std::max(1.0, error / 0.65) : 1.0);
            for (int j = 0; j < 8; ++j) {
                rhs[j] += weight * jacobian[j] * m.offset;
                for (int k = j; k < 8; ++k)
                    system(j, k) += weight * jacobian[j] * jacobian[k];
            }
        }
        for (int j = 0; j < 8; ++j)
            for (int k = 0; k < j; ++k)
                system(j, k) = system(k, j);
        if (!cv::solve(system, rhs, solution, cv::DECOMP_CHOLESKY))
            return false;
    }
    return true;
}

bool apply_solution(Quad& quad, Point center, float side, const Parameters& solution, float limit) {
    auto trial = quad;
    for (Point& corner : trial) {
        const Point p = (corner - center) * (1 / side);
        const double scale = solution[6] * p.x + solution[7] * p.y;
        const Point move(solution[0] * p.x + solution[1] * p.y + solution[2] - p.x * scale,
                         solution[3] * p.x + solution[4] * p.y + solution[5] - p.y * scale);
        if (!(length(move) < limit))
            return false;
        corner += move;
    }
    quad = trial;
    return true;
}

bool fit(const cv::Mat& gray, const Grid& cells, float radius_ratio, Quad& quad) {
    Projection projection;
    const float side = length(quad[1] - quad[0]);
    if (!(side >= 13 && std::isfinite(side)) || !project(quad, projection))
        return false;
    const Point center = coordinate(projection, 0.5F, 0.5F);
    Offsets measurements;
    const int count = observe(gray, cells, projection, center, side, radius_ratio, measurements);
    auto solution = Parameters::all(0);
    if (count < 16 || !solve_offsets(measurements, count, solution, 2))
        return false;
    return apply_solution(quad, center, side, solution, std::max(2.0F, 0.14F * side));
}

bool verify(const cv::Mat& gray, const Grid& cells, const Quad& quad, int payload_budget,
            int border_budget) {
    return cpu::detail::visible_grid_matches(
            gray, cv::Mat(cells, false), {quad.begin(), quad.end()}, payload_budget, border_budget);
}

float border_midpoint_error(const cv::Mat& gray, const Projection& projection, int axis, int side,
                            float position, float minimum_contrast) {
    const auto sample = [&](float inset_cells) {
        const float inset = inset_cells * kCellWidth;
        const float coordinate = side == 0 ? inset : 1.0F - inset;
        return axis == 0 ? projection.sample(gray, coordinate, position)
                         : projection.sample(gray, position, coordinate);
    };
    const float dark = sample(0.41F), light = sample(-0.41F), edge = sample(0.0F);
    if (dark < 0 || edge < 0 || light - dark < minimum_contrast)
        return -1.0F;
    const float residual = (edge - 0.5F * (dark + light)) / (light - dark);
    return std::min(1.0F, residual * residual);
}

bool original_border_fits_better(const cv::Mat& gray, const Quad& original, const Quad& fitted,
                                 const GridBorderConfig& config) {
    float max_move_squared = 0;
    for (int k = 0; k < 4; ++k) {
        const Point move = fitted[k] - original[k];
        max_move_squared = std::max(max_move_squared, move.dot(move));
    }
    if (max_move_squared < config.minimum_displacement_px_ * config.minimum_displacement_px_ ||
        length(original[0] - original[1]) < config.minimum_side_px_)
        return false;

    Projection before, after;
    if (!project(original, before) || !project(fitted, after))
        return false;
    int support = 0;
    float original_error = 0, fitted_error = 0;
    // Sample the same visible quiet-zone transitions in both hypotheses. Internal edges
    // can be displaced by toner, blur, or a nonuniform print without moving the outer border.
    for (int axis = 0; axis < 2; ++axis) {
        for (int side = 0; side < 2; ++side) {
            for (float position : {0.21F, 0.47F, 0.79F}) {
                const float a = border_midpoint_error(gray, before, axis, side, position,
                                                      config.minimum_contrast_);
                const float b = border_midpoint_error(gray, after, axis, side, position,
                                                      config.minimum_contrast_);
                if (a >= 0 && b >= 0) {
                    ++support;
                    original_error += a;
                    fitted_error += b;
                }
            }
        }
    }
    return support >= config.minimum_shared_samples_ &&
           original_error < config.maximum_original_mean_square_ * support &&
           fitted_error > original_error * config.minimum_error_ratio_ +
                                  config.minimum_mean_square_increase_ * support;
}

bool fit_verified(const cv::Mat& gray, const Grid& cells, int budget, Quad& seed) {
    if (!fit(gray, cells, 0.47F, seed))
        return false;
    fit(gray, cells, 0.32F, seed);
    return verify(gray, cells, seed, budget, 4);
}

struct StripCounts {
    int accepted = 0;
    int quiet = 0;
    int broad = 0;
};

// Integrate the central pixels of an already localized step. The two
// local plateaus provide the intensity scale even on a tilted light field.
bool measure_faint_crossing(const std::uint8_t* at, std::ptrdiff_t stride, float normal_minor,
                            int minimum_contrast, float& displacement, StripCounts& counts,
                            bool room_for_wide) {
    const auto pixel = [&](int i) { return int(at[i * stride]); };
    float before = (pixel(-4) + pixel(-3) + pixel(-2)) / 3.F;
    float after = (pixel(2) + pixel(3) + pixel(4)) / 3.F;
    // Include the shoulders of a visibly blurred edge in the area integral.
    // Keep the narrow interval for a sharp step and for cells close to the frame edge.
    const bool broad =
            room_for_wide && std::abs(pixel(1) - pixel(-1)) < .8F * std::abs(before - after);
    int total = pixel(-1) + pixel(0) + pixel(1);
    if (broad) {
        before = (pixel(-6) + pixel(-5) + pixel(-4)) / 3.F;
        after = (pixel(4) + pixel(5) + pixel(6)) / 3.F;
        total += pixel(-2) + pixel(2);
    }
    const float contrast = before - after;
    if (-normal_minor * contrast < .55F * minimum_contrast)
        return false;
    const int width = broad ? 5 : 3;
    displacement += (total - width * after) / contrast - .5F * width;
    if (std::abs(displacement) > .65F)
        return false;
    ++counts.accepted;
    const int inner = broad ? 4 : 2;
    const int outer = inner + 2;
    counts.quiet +=
            std::abs(pixel(-outer) - pixel(-inner)) + std::abs(pixel(inner) - pixel(outer)) <=
            std::max(3.F, .05F * std::abs(contrast));
    return true;
}

// Across a sharp transition, the sum of normalized pixel intensities measures the
// area of the light side. Subtract the local plateau slope to account for lighting.
bool measure_strip(const cv::Mat& gray, Point start, Point end, Point normal, Point center,
                   float side, EdgeOffset& measurement, StripCounts& counts, int faint_contrast) {
    const Point delta = end - start;
    const bool horizontal = std::abs(delta.x) >= std::abs(delta.y);
    if (!horizontal) {
        std::swap(start.x, start.y);
        std::swap(end.x, end.y);
    }
    const float extent = end.x - start.x;
    if (!(std::abs(extent) >= 1.0F))
        return false;
    const float slope = (end.y - start.y) / extent;
    const float normal_minor = horizontal ? normal.y : normal.x;
    const int first = cvCeil(std::min(start.x, end.x) + 0.45F);
    const int last = cvFloor(std::max(start.x, end.x) - 0.45F);
    if (first < 0 || last >= (horizontal ? gray.cols : gray.rows))
        return false;
    const int radius = length(delta) < 6.0F ? 2 : 3;
    const int width = 2 * radius + 1;
    const auto major = horizontal ? 1U : gray.step[0];
    const auto minor = horizontal ? gray.step[0] : 1U;
    int samples = 0;
    float sum_position = 0, sum_displacement = 0;
    const int sample_count = faint_contrast ? 18 : last - first + 1;
    int previous = first - 2;
    for (int index = 0; index < sample_count; ++index) {
        const int pos =
                faint_contrast
                        ? cvRound(start.x + extent * (.14F + .72F * index / (sample_count - 1)))
                        : first + index;
        if (pos == previous || pos < first || pos > last)
            continue;
        previous = pos;
        const float predicted = start.y + (pos - start.x) * slope;
        const int middle = cvRound(predicted);
        if (middle < radius || middle >= (horizontal ? gray.rows : gray.cols) - radius)
            continue;
        const auto* pixels = gray.data + pos * major + (middle - radius) * minor;
        const int left = pixels[0], right = pixels[(width - 1) * minor];
        if (faint_contrast) {
            if (middle < 4 || middle >= (horizontal ? gray.rows : gray.cols) - 4)
                continue;
            float displacement = middle - predicted;
            if (measure_faint_crossing(
                        gray.data + pos * major + middle * minor, minor, normal_minor,
                        faint_contrast, displacement, counts,
                        middle >= 6 && middle < (horizontal ? gray.rows : gray.cols) - 6)) {
                sum_position += pos;
                sum_displacement += displacement;
                ++samples;
            }
            continue;
        }
        if (-normal_minor * (left - right) < 36)
            continue;
        const int left_slope = pixels[minor] - left;
        const int right_slope = right - pixels[(width - 2) * minor];
        const int roughness = std::abs(left_slope) + std::abs(right_slope);
        if (roughness > 5)
            continue;
        ++counts.accepted;
        counts.quiet += roughness <= 2;
        int total = 0;
        for (int j = 0; j < width; ++j)
            total += pixels[j * minor];
        const float drift = 0.5F * (left_slope + right_slope);
        const float right_level = right - drift * 2 * radius;
        const float area =
                (total - drift * width * radius - width * right_level) / (left - right_level);
        const float displacement = middle - predicted - radius - 0.5F + area;
        if (std::abs(displacement) > 1.15F)
            continue;
        sum_position += pos;
        sum_displacement += displacement;
        ++samples;
    }
    if (!samples)
        return false;
    const float mid = sum_position / samples;
    Point position(mid, start.y + (mid - start.x) * slope);
    if (!horizontal)
        std::swap(position.x, position.y);
    measurement = {(position - center) * (1 / side), normal,
                   sum_displacement / samples * normal_minor, static_cast<float>(samples)};
    return true;
}

// Average independent native transition measurements along each cell edge.
bool measure_diffuse_strip(const cv::Mat& gray, Point start, Point end, Point normal, Point center,
                           float side, int contrast, EdgeOffset& result, StripCounts& counts) {
    const Point delta = end - start;
    const bool horizontal = std::abs(delta.x) >= std::abs(delta.y);
    const float extent = horizontal ? delta.x : delta.y, origin = horizontal ? start.x : start.y;
    if (std::abs(extent) < 5)
        return false;
    const float sign = extent > 0 ? 1.F : -1.F;
    const auto n = horizontal ? normal.y : normal.x;
    const int major_limit = horizontal ? gray.cols : gray.rows,
              minor_limit = horizontal ? gray.rows : gray.cols;
    const auto major = horizontal ? 1UL : gray.step[0], minor = horizontal ? gray.step[0] : 1UL;
    const int first = cvCeil(sign * origin + 1.45F),
              last = cvFloor(sign * (origin + .85F * extent));
    float weight = 0, position = 0, displacement = 0;
    const int stride = std::max(2, (last - first + 17) / 16);
    for (int j = first; j <= last; j += stride) {
        const int pos = sign * j;
        const float t = (pos - origin) / extent;
        if (t < .15F || pos < 0 || pos >= major_limit)
            continue;
        const float predicted = horizontal ? start.y + t * delta.y : start.x + t * delta.x;
        const int mid = cvRound(predicted);
        if (mid < 4 || mid >= minor_limit - 4)
            continue;
        float offset, w;
        const auto* pixels = gray.data + pos * major + mid * minor;
        if (!diffuse_crossing(pixels, minor, mid - predicted, n, contrast, offset, w))
            continue;
        const float step = std::abs(int(pixels[minor]) - int(*(pixels - minor)));
        const float plateaus = std::abs(int(pixels[3 * minor]) - int(*(pixels - 3 * minor)));
        ++counts.accepted;
        // Count transitions whose shoulders extend beyond the central two pixels.
        counts.broad += step < .9F * plateaus;
        weight += w;
        position += t * w;
        displacement += offset * w;
    }
    if (!(weight > 0))
        return false;
    result = {(start + (position / weight) * delta - center) * (1 / side), normal,
              n * displacement / weight, weight};
    return true;
}

// A hard binary raster brackets the geometric edge between adjacent pixel centers.
// Fit this interval directly. Area integration assumes a pixel averages intensities
// over its footprint and discards these subpixel constraints on a point-sampled grid.
int binary_levels(const cv::Mat& gray, const Projection& p, const Grid& cells,
                  std::array<int, 2>& levels, int maximum_errors) {
    levels = {-1, -1};
    int votes[2][2]{};
    for (int y = 0; y < kGridSide; ++y) {
        for (int x = 0; x < kGridSide; ++x) {
            const Point center = coordinate(p, (x + 0.5F) * kCellWidth, (y + 0.5F) * kCellWidth);
            if (!(center.x >= 0 && center.x <= gray.cols - 1 && center.y >= 0 &&
                  center.y <= gray.rows - 1))
                return -1;
            const int value = gray.ptr<std::uint8_t>(cvRound(center.y))[cvRound(center.x)];
            if (levels[0] < 0)
                levels[0] = value;
            const int second = value != levels[0];
            if (second && levels[1] < 0)
                levels[1] = value;
            if (value != levels[second])
                return -1;
            ++votes[second][cells(y, x)];
        }
    }
    const bool reversed = levels[0] > levels[1];
    if (reversed)
        std::swap(levels[0], levels[1]);
    const int errors = votes[reversed][1] + votes[!reversed][0];
    return levels[0] >= 0 && levels[1] - levels[0] >= 15 && errors <= maximum_errors ? errors : -1;
}

struct BorderLine {
    Point point;
    Point direction;
};

bool intersect_borders(const std::array<BorderLine, 4>& edges, const Quad& corners, float limit,
                       Quad& trial) {
    for (int i = 0; i < 4; ++i) {
        const auto& a = edges[(i + 3) % 4];
        const auto& b = edges[i];
        const float det = cross(a.direction, b.direction);
        if (std::abs(det) < 0.3F)
            return false;
        const float t = cross(b.point - a.point, b.direction) / det;
        trial[i] = a.point + t * a.direction;
        if (!(length(trial[i] - corners[i]) < limit))
            return false;
    }
    return true;
}

// For an edge spanning at most 96 scanlines, the feasible polygon starts with
// four vertices and gains at most one vertex per intermediate sample.
constexpr int kBinaryMaxSpan = 96;
using FeasiblePolygon = std::array<Point, kBinaryMaxSpan + 8>;

int clip_offset_polygon(const FeasiblePolygon& input, int count, float t, float residual,
                        FeasiblePolygon& output) {
    // Endpoint offsets lie in [-0.5,0.5]. Only one side of the interval at an
    // intermediate position can cut the polygon inside this box.
    const float sign = residual > 0 ? 1.0F : -1.0F;
    auto margin = [&](Point p) { return 0.5F + sign * ((1 - t) * p.x + t * p.y - residual); };
    int written = 0;
    Point previous = input[count - 1];
    float last = margin(previous);
    for (int i = 0; i < count; ++i) {
        if (written + 2 > static_cast<int>(output.size()))
            return 0;
        const Point next = input[i];
        const float current = margin(next);
        if ((current < 0) != (last < 0))
            output[written++] = previous + (last / (last - current)) * (next - previous);
        if (current >= 0)
            output[written++] = next;
        previous = next;
        last = current;
    }
    return written;
}

bool binary_line(const cv::Mat& gray, Point start, Point end, const std::array<int, 2>& levels,
                 bool rough_seed, BorderLine& line) {
    const Point span = end - start;
    const bool horizontal = std::abs(span.x) >= std::abs(span.y);
    if (!horizontal) {
        std::swap(start.x, start.y);
        std::swap(end.x, end.y);
    }
    const float dx = end.x - start.x, length = std::abs(dx);
    // Flat steps provide no subpixel phase information for this fit.
    if (!(length >= 9 && length <= kBinaryMaxSpan && std::abs(start.y - end.y) >= 2))
        return false;
    const float slope = (end.y - start.y) / dx;
    const float inset = (rough_seed ? 0.10F : 0.035F) * length;
    const int begin = std::max(0, cvCeil(std::min(start.x, end.x) + inset));
    const int last = std::min((horizontal ? gray.cols : gray.rows) - 1,
                              cvFloor(std::max(start.x, end.x) - inset));
    const int limit = horizontal ? gray.rows : gray.cols;
    const auto major = horizontal ? 1U : gray.step[0];
    const auto minor = horizontal ? gray.step[0] : 1U;
    const bool bright_first = (dx > 0) == horizontal;
    std::array<Point, kBinaryMaxSpan> samples;
    int count = 0;
    for (int x = begin; x <= last; ++x) {
        const int guess = cvRound(start.y + (x - start.x) * slope);
        constexpr int offsets[] = {-1, 0, -2, 1, -3, 2, -4, 3, -5, 4};
        for (int i = 0; i < (rough_seed ? 10 : 6); ++i) {
            const int offset = offsets[i];
            const int y = guess + offset;
            if (y < 0 || y + 1 >= limit)
                continue;
            const auto* pixel = gray.data + x * major + y * minor;
            if (*pixel == levels[bright_first] && pixel[minor] == levels[!bright_first]) {
                samples[count++] = Point(x, y + 0.5F);
                break;
            }
        }
    }
    if (count < 6 || count * 5 < (last - begin + 1) * 4)
        return false;
    const Point first = samples[0], delta = samples[count - 1] - first;
    const float inv_x = 1 / delta.x;
    FeasiblePolygon slots[2];
    slots[0][0] = {-0.5F, -0.5F};
    slots[0][1] = {0.5F, -0.5F};
    slots[0][2] = {0.5F, 0.5F};
    slots[0][3] = {-0.5F, 0.5F};
    int selected = 0, vertices = 4;
    for (int i = 1; i < count - 1; ++i) {
        const float t = (samples[i].x - first.x) * inv_x;
        const float residual = (samples[i].y - first.y) - t * delta.y;
        if (std::abs(residual) < 1e-6F)
            continue;
        vertices = clip_offset_polygon(slots[selected], vertices, t, residual, slots[1 - selected]);
        if (vertices < 3)
            return false;
        selected = 1 - selected;
    }
    const auto& polygon = slots[selected];
    Point centroid{};
    Point low = polygon[0], high = polygon[0];
    float area2 = 0;
    for (int i = 0, prev = vertices - 1; i < vertices; prev = i++) {
        const Point v = polygon[i];
        low = {std::min(low.x, v.x), std::min(low.y, v.y)};
        high = {std::max(high.x, v.x), std::max(high.y, v.y)};
        const float weight = cross(polygon[prev], v);
        centroid += weight * (polygon[prev] + polygon[i]);
        area2 += weight;
    }
    // Broad intervals give little localization information. Use the decoded payload fit
    // when the available border samples leave either endpoint uncertain by over half a pixel.
    if (!(area2 > 1e-7F) || (!rough_seed && (high.x - low.x > 0.5F || high.y - low.y > 0.5F)))
        return false;
    centroid *= 1 / (3 * area2);
    const float offset = first.y + centroid.x;
    const float fitted_slope = (delta.y + centroid.y - centroid.x) * inv_x;
    line = horizontal ? BorderLine{{first.x, offset}, {1, fitted_slope}}
                      : BorderLine{{offset, first.x}, {fitted_slope, 1}};
    return true;
}

bool fit_binary_raster(const cv::Mat& gray, const Grid& cells, const Projection& p, Quad& corners) {
    std::array<int, 2> levels;
    // A coarse seed may displace a few cell centers. The final fit must reproduce
    // every bit, including the black border, at the original two intensities.
    const int errors = binary_levels(gray, p, cells, levels, kGridSide);
    if (errors < 0)
        return false;
    Quad trial;
    auto fit_edges = [&](bool rough) {
        std::array<BorderLine, 4> edges;
        for (int i = 0; i < 4; ++i)
            if (!binary_line(gray, corners[i], corners[(i + 1) % 4], levels, rough, edges[i]))
                return false;
        return intersect_borders(edges, corners, rough ? 5.0F : 2.5F, trial);
    };
    if (errors || !fit_edges(false)) {
        if (!fit_edges(true))
            return false;
        // Ignore a weakly constrained outer edge when the code is aligned already.
        float movement = 0;
        for (int i = 0; i < 4; ++i)
            movement = std::max(movement, length(trial[i] - corners[i]));
        if (!errors && movement < 1.25F)
            return false;
    }
    Projection current;
    std::array<int, 2> check;
    if (!project(trial, current) || binary_levels(gray, current, cells, check, 0) != 0 ||
        check != levels)
        return false;
    corners = trial;
    return true;
}

bool fit_sharp_grid(const cv::Mat& gray, const Grid& cells, int budget, Quad& quad,
                    bool verify_complete = true, int faint_contrast = 0, bool diffuse = false) {
    const float side = length(quad[1] - quad[0]);
    Projection projection;
    if (!(side >= 13 && (faint_contrast || side <= 240)) || !project(quad, projection))
        return false;
    if (verify_complete && side <= kBinaryMaxSpan + 1 &&
        fit_binary_raster(gray, cells, projection, quad))
        return true;
    std::array<std::array<Point, kGridSide + 1>, kGridSide + 1> nodes;
    for (int y = 0; y <= kGridSide; ++y)
        for (int x = 0; x <= kGridSide; ++x)
            nodes[y][x] = coordinate(projection, x * kCellWidth, y * kCellWidth);
    const Point center = nodes[kGridSide / 2][kGridSide / 2];
    StripCounts counts;
    Offsets offsets;
    int count = 0;
    for (int axis = 0; axis < 2; ++axis) {
        for (int i = 0; i <= kGridSide; ++i) {
            for (int j = 0; j < kGridSide; ++j) {
                const int before = axis ? color(cells, j, i - 1) : color(cells, i - 1, j);
                const int after = axis ? color(cells, j, i) : color(cells, i, j);
                if (before == after)
                    continue;
                const Point start = axis ? nodes[i][j] : nodes[j][i];
                const Point end = axis ? nodes[i][j + 1] : nodes[j + 1][i];
                const Point delta = end - start;
                const float span = length(delta);
                if (!(span >= 1.0F && (faint_contrast || span <= 64.0F)))
                    continue;
                Point normal(delta.y / span, -delta.x / span);
                if ((before > after) != static_cast<bool>(axis))
                    normal = -normal;
                if (diffuse ? measure_diffuse_strip(gray, start, end, normal, center, side,
                                                    faint_contrast, offsets[count], counts)
                            : measure_strip(gray, start, end, normal, center, side, offsets[count],
                                            counts, faint_contrast))
                    ++count;
            }
        }
    }
    auto solution = Parameters::all(0);
    const bool supported_profile =
            diffuse ? counts.accepted > 0 && 3 * counts.broad >= counts.accepted
                    : 4 * counts.quiet >= 3 * counts.accepted;
    if (count < 14 || !supported_profile || !solve_offsets(offsets, count, solution, 1))
        return false;
    Quad trial = quad;
    if (!apply_solution(trial, center, side, solution, faint_contrast ? .5F : 1.5F) ||
        (verify_complete && !verify(gray, cells, trial, budget, 4)))
        return false;
    if (faint_contrast) {
        // Nearly axis-aligned rails can span one pixel phase along their full
        // length. Their fractional intensity does not constrain a new projective shear.
        int limited_edges = 0;
        Point shift{};
        for (int i = 0; i < 4; ++i) {
            const auto edge = quad[(i + 1) % 4] - quad[i];
            limited_edges += std::min(std::abs(edge.x), std::abs(edge.y)) < .5F;
            shift += trial[i] - quad[i];
        }
        if (limited_edges >= 2) {
            for (int i = 0; i < 4; ++i)
                trial[i] = quad[i] + .25F * shift;
        }
    }
    quad = trial;
    return true;
}

bool sharp_border_line(const cv::Mat& gray, Point start, Point end, BorderLine& line) {
    const Point delta = end - start;
    const bool horizontal = std::abs(delta.x) >= std::abs(delta.y);
    if (!horizontal) {
        std::swap(start.x, start.y);
        std::swap(end.x, end.y);
    }
    const float extent = end.x - start.x;
    const float span = std::abs(extent);
    if (!(span > 9 && span < 240))
        return false;
    const float slope = (end.y - start.y) / extent;
    const float mid = 0.5F * (start.x + end.x);
    const int begin = std::max(0, cvCeil(std::min(start.x, end.x) + 0.09F * span));
    const int stop = std::min(horizontal ? gray.cols : gray.rows,
                              cvFloor(std::max(start.x, end.x) - 0.09F * span) + 1);
    const int stride = std::max(1, (stop - begin) / 60);
    const int possible = (stop - begin + stride - 1) / stride;
    if (possible < 7)
        return false;
    const auto minor = horizontal ? gray.step[0] : 1U;
    const auto major = horizontal ? 1U : gray.step[0];
    const int sign = (horizontal ? extent > 0 : extent < 0) ? 1 : -1;
    int count = 0;
    float x_sum = 0, d_sum = 0, xx_sum = 0, xd_sum = 0, dd_sum = 0;
    for (int pos = begin; pos < stop; pos += stride) {
        const float expected = start.y + (pos - start.x) * slope;
        const int guess = cvRound(expected);
        if (guess < 6 || guess >= (horizontal ? gray.rows : gray.cols) - 6)
            continue;
        const auto* window = gray.data + pos * major + (guess - 6) * minor;
        int peak = 0, shift = 0;
        for (int k = -4; k <= 4; ++k) {
            const int gradient = window[(k + 5) * minor] - window[(k + 7) * minor];
            const int strength = sign * gradient - 3 * std::abs(k);
            if (strength > peak) {
                peak = strength;
                shift = k;
            }
        }
        if (peak < 50)
            continue;
        const auto* pixels = window + (shift + 4) * minor;
        const int first = pixels[0], last = pixels[4 * minor];
        const int contrast = first - last;
        if (sign * contrast < 50 || std::abs(first - pixels[minor]) > 3 ||
            std::abs(last - pixels[3 * minor]) > 3)
            continue;

        int sum = first;
        bool monotone = true;
        for (int j = 1; j <= 4; ++j) {
            const int sample = pixels[j * minor];
            sum += sample;
            monotone &= sign * (sample - pixels[(j - 1) * minor]) <= 7;
        }
        if (!monotone)
            continue;
        const float displacement =
                guess + shift - expected - 2.5F + (sum - 5 * last) / static_cast<float>(contrast);
        const float x = pos - mid;
        x_sum += x;
        d_sum += displacement;
        xx_sum += x * x;
        xd_sum += x * displacement;
        dd_sum += displacement * displacement;
        ++count;
    }
    if (count < 7 || count * 2 < possible)
        return false;
    const float variance = xx_sum - x_sum * x_sum / count;
    if (variance < 0.022F * span * span * count)
        return false;
    const float correction = (xd_sum - x_sum * d_sum / count) / variance;
    const float offset = (d_sum - correction * x_sum) / count;
    const float error = (dd_sum - correction * xd_sum - offset * d_sum) / count;
    if (!(error < 0.032F))
        return false;
    const float height = start.y + (mid - start.x) * slope + offset;
    line = horizontal ? BorderLine{{mid, height}, {1, slope + correction}}
                      : BorderLine{{height, mid}, {slope + correction, 1}};
    return true;
}

bool canonical_grid(int id, Grid& grid) {
    const auto& table = generated::kDictApriltag36h11Table;
    if (id < 0 || id >= table.code_count_)
        return false;
    const auto code = table.codes_[4 * id];
    for (int y = 0; y < 6; ++y)
        for (int x = 0; x < 6; ++x)
            grid(y + 1, x + 1) = (code >> (6 * y + x)) & 1U;
    return true;
}

// Compare signed contrast across the visible internal transitions. A silhouette
// cut off by the image border can still have three well-localized corners.
float transition_contrast(const cv::Mat& gray, const Grid& cells, const Quad& quad) {
    Projection p;
    if (!project(quad, p))
        return -1;
    float sum = 0;
    int count = 0;
    for (int y = 1; y < 7; ++y) {
        for (int x = 1; x < 7; ++x) {
            const float u = (x + 0.5F) / 8, v = (y + 0.5F) / 8;
            for (int axis = 0; axis < 2; ++axis) {
                const int next = axis ? cells(y + 1, x) : cells(y, x + 1);
                const int sign = next - cells(y, x);
                if (sign == 0)
                    continue;
                const auto a = axis ? p.sample(gray, u, v + 0.042F) : p.sample(gray, u + 0.042F, v);
                const auto b = axis ? p.sample(gray, u, v + 0.083F) : p.sample(gray, u + 0.083F, v);
                if (a >= 0 && b >= 0) {
                    sum += (b - a) * sign;
                    ++count;
                }
            }
        }
    }
    return count >= 12 ? sum / count : -1;
}

}  // namespace

bool try_fit_sharp_border(const cv::Mat& gray, Quad& corners) {
    std::array<BorderLine, 4> edges;
    for (int i = 0; i < 4; ++i)
        if (!sharp_border_line(gray, corners[i], corners[(i + 1) % 4], edges[i]))
            return false;
    Quad trial;
    if (!intersect_borders(edges, corners, 4.5F, trial))
        return false;
    corners = trial;
    return true;
}

bool propose_diffuse_grid(const cv::Mat& gray, int id, int contrast, Quad& quad) {
    Grid cells = Grid::zeros();
    return canonical_grid(id, cells) && fit_sharp_grid(gray, cells, 0, quad, false, contrast, true);
}

bool propose_faint_grid(const cv::Mat& gray, int id, int contrast, Quad& quad) {
    Grid cells{};
    return canonical_grid(id, cells) && fit_sharp_grid(gray, cells, 0, quad, false, contrast);
}

bool propose_visible_grid(const cv::Mat& gray, const Detection& tag, Quad& hypothesis) {
    Grid cells{};
    if (!canonical_grid(tag.id_, cells))
        return false;
    hypothesis = tag.corners_px_;
    if (fit_sharp_grid(gray, cells, tag.corrected_bits_, hypothesis, false))
        return true;
    if (fit(gray, cells, 0.47F, hypothesis)) {
        fit(gray, cells, 0.32F, hypothesis);
        fit_sharp_grid(gray, cells, tag.corrected_bits_, hypothesis, false);
        return true;
    }

    // When one corner is clipped, the contour can end on the image boundary.
    // Hold the other three corners fixed and look for the missing corner using
    // payload transitions that are still visible inside the image.
    const auto& q = tag.corners_px_;
    const float side = std::min(length(q[1] - q[0]), length(q[2] - q[1]));
    if (side < 16)
        return false;
    float best_score = transition_contrast(gray, cells, q);
    for (int i = 0; i < 4; ++i) {
        const Point pt = q[i];
        const bool horizontal = std::min(std::abs(pt.y), std::abs(pt.y - gray.rows + 1)) < 2.5F;
        const bool vertical = std::min(std::abs(pt.x), std::abs(pt.x - gray.cols + 1)) < 2.5F;
        if (horizontal == vertical)
            continue;
        const Point normal = horizontal ? Point(0, pt.y < gray.rows * .5F ? -1 : 1)
                                        : Point(pt.x < gray.cols * .5F ? -1 : 1, 0);
        const Point tangent(-normal.y, normal.x);
        const float step = std::max(1.F, std::min(4.F, side / 20));
        for (int outward = 0; outward <= 7; ++outward) {
            for (int lateral = -4; lateral <= 4; ++lateral) {
                auto trial = q;
                trial[i] += step * (outward * normal + lateral * tangent);
                const float score = transition_contrast(gray, cells, trial);
                if (score > best_score) {
                    best_score = score;
                    hypothesis = trial;
                }
            }
        }
    }
    if (hypothesis == q || !fit(gray, cells, 0.47F, hypothesis))
        return false;
    fit(gray, cells, 0.32F, hypothesis);
    fit_sharp_grid(gray, cells, tag.corrected_bits_, hypothesis, false);
    return true;
}

bool try_align_curved_grid(const cv::Mat& gray, Detection& detection) {
    Grid cells{};
    return canonical_grid(detection.id_, cells) &&
           fit_curved_edges(gray, cells, detection.corners_px_);
}

bool try_align_sharp_grid(const cv::Mat& gray, Detection& detection) {
    Grid cells{};
    return canonical_grid(detection.id_, cells) &&
           fit_sharp_grid(gray, cells, std::max(2, detection.corrected_bits_),
                          detection.corners_px_);
}

bool align_decoded_grid(const cv::Mat& gray, Detection& detection, const GridBorderConfig& border) {
    Grid cells{};
    if (!canonical_grid(detection.id_, cells))
        return false;
    const Quad& initial = detection.corners_px_;
    const int budget = std::max(2, detection.corrected_bits_);
    Quad refined = initial;
    if (fit_sharp_grid(gray, cells, budget, refined) ||
        fit_verified(gray, cells, budget, refined)) {
        if (original_border_fits_better(gray, initial, refined, border) &&
            verify(gray, cells, initial, budget, 1))
            return true;
        detection.corners_px_ = refined;
        return true;
    }

    // With a truncated edge the remaining three corners can still locate the decoded grid.
    // Gate the affine seed by its cell-scale displacement and the observed, oriented code;
    // the final fit can represent perspective again.
    const float discrepancy = length(initial[0] + initial[2] - initial[1] - initial[3]);
    const float cell =
            std::min(length(initial[1] - initial[0]), length(initial[2] - initial[1])) * kCellWidth;
    if (!(discrepancy > 0.5F * cell && discrepancy < 3 * cell))
        return false;
    for (int i = 0; i < 4; ++i) {
        refined = initial;
        refined[i] = initial[(i + 1) % 4] + initial[(i + 3) % 4] - initial[(i + 2) % 4];
        if (verify(gray, cells, refined, detection.corrected_bits_, 1) &&
            fit_verified(gray, cells, budget, refined)) {
            detection.corners_px_ = refined;
            return true;
        }
    }
    return false;
}

bool align_isolated_grid(const cv::Mat& gray, Detection& detection) {
    if (align_decoded_grid(gray, detection))
        return true;
    const auto& q = detection.corners_px_;
    const float cell = std::min(length(q[1] - q[0]), length(q[2] - q[1])) * kCellWidth;
    const float asymmetry = length(q[0] + q[2] - q[1] - q[3]);
    if (cell < 1.5F || asymmetry < 0.5F * cell || asymmetry > 3 * cell)
        return false;

    // Two truncated edges can move adjacent corners independently. A fitted
    // rectangle supplies a fresh seed; the accepted result remains projective.
    Quad rectangle;
    cv::minAreaRect(cv::Mat(q, false)).points(rectangle.data());
    Detection candidate = detection;
    float best = std::numeric_limits<float>::infinity();
    for (int start = 0; start < 4; ++start) {
        float distance = 0.F;
        for (int i = 0; i < 4; ++i) {
            const auto delta = rectangle[(start + i) % 4] - q[i];
            distance += delta.dot(delta);
        }
        if (distance < best) {
            best = distance;
            for (int i = 0; i < 4; ++i)
                candidate.corners_px_[i] = rectangle[(start + i) % 4];
        }
    }
    Grid cells{};
    if (!align_decoded_grid(gray, candidate) || !canonical_grid(detection.id_, cells) ||
        !verify(gray, cells, candidate.corners_px_, detection.corrected_bits_, 1))
        return false;
    detection.corners_px_ = candidate.corners_px_;
    return true;
}

}  // namespace aruco3cuda::apriltag_small_cpu::detail
