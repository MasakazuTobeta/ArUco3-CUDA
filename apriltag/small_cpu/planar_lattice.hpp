// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "aruco3cuda/apriltag/small_cpu/detector.hpp"
#include "geometry.hpp"

#include <algorithm>
#include <limits>
#include <vector>

namespace aruco3cuda::apriltag_small_cpu::detail {
namespace planar_lattice {
using XY = cv::Point2d;

struct Marker {
    XY center;
    cv::Vec3d location;
    XY local, index, predicted;
    double area_scale;
    bool included = true;
};

inline double median(std::vector<double>& values) {
    const auto middle = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), middle, values.end());
    return *middle;
}

// Fit an image homography to the inferred integer lattice sites using normalized
// coordinates. Input membership is based on independently decoded marker positions.
inline bool fit_grid(std::vector<Marker>& markers, XY origin, XY span) {
    XY mean{}, extent{};
    for (const auto& m : markers)
        mean += m.index;
    mean *= 1. / markers.size();
    for (const auto& m : markers) {
        extent.x = std::max(extent.x, std::abs(m.index.x - mean.x));
        extent.y = std::max(extent.y, std::abs(m.index.y - mean.y));
    }
    if (extent.x < 1 || extent.y < 1)
        return false;
    cv::Mat design(int(2 * markers.size()), 8, CV_64F), target(design.rows, 1, CV_64F);
    int rows = 0;
    for (const auto& m : markers) {
        if (!m.included)
            continue;
        const double x = (m.index.x - mean.x) / extent.x, y = (m.index.y - mean.y) / extent.y;
        const double u = m.location[1], v = m.location[2];
        const std::array<double, 8> xrow{x, y, 1, 0, 0, 0, -u * x, -u * y};
        const std::array<double, 8> yrow{0, 0, 0, x, y, 1, -v * x, -v * y};
        std::copy(xrow.begin(), xrow.end(), design.ptr<double>(rows));
        target.at<double>(rows++) = u;
        std::copy(yrow.begin(), yrow.end(), design.ptr<double>(rows));
        target.at<double>(rows++) = v;
    }
    cv::Vec<double, 8> c;
    if (rows < 8 || !cv::solve(design.rowRange(0, rows), target.rowRange(0, rows), c,
                               cv::DECOMP_NORMAL | cv::DECOMP_CHOLESKY))
        return false;
    for (auto& m : markers) {
        const double x = (m.index.x - mean.x) / extent.x, y = (m.index.y - mean.y) / extent.y;
        const double w = 1 + c[6] * x + c[7] * y;
        if (!(w > .02))
            return false;
        m.predicted = origin + XY(span.x * (c[0] * x + c[1] * y + c[2]) / w,
                                  span.y * (c[3] * x + c[4] * y + c[5]) / w);
    }
    return true;
}
}  // namespace planar_lattice

// A regular array supplies an independent center estimate, including markers at
// the edge. Derive the integer sites from current image geometry without IDs,
// board dimensions, commanded pitch or history.
inline bool refine_regular_tag_centers(std::vector<Detection>& tags, std::size_t minimum,
                                       double max_shift) {
    using namespace planar_lattice;
    if (tags.size() < minimum || max_shift <= 0)
        return false;
    std::vector<Marker> markers(tags.size());
    XY origin{}, low(tags[0].corners_px_[0]), high = low;
    for (std::size_t i = 0; i < tags.size(); ++i) {
        const auto& q = tags[i].corners_px_;
        const auto d = q[2] - q[0], e = q[3] - q[1];
        const float determinant = cross(d, e);
        if (!(std::abs(determinant) > 1))
            return false;
        auto& m = markers[i];
        m.center = XY(q[0]) + XY(d) * (double(cross(q[1] - q[0], e)) / determinant);
        m.area_scale = std::cbrt(std::abs(determinant));
        origin += m.center;
        low.x = std::min(low.x, m.center.x);
        low.y = std::min(low.y, m.center.y);
        high.x = std::max(high.x, m.center.x);
        high.y = std::max(high.y, m.center.y);
    }
    const XY span = high - low;
    if (span.x < 1 || span.y < 1)
        return false;
    origin *= 1. / markers.size();
    for (auto& m : markers)
        m.location = {1, (m.center.x - origin.x) / span.x, (m.center.y - origin.y) / span.y};
    const auto supported = [&] {
        const std::size_t count = std::count_if(markers.begin(), markers.end(),
                                                [](const Marker& m) { return m.included; });
        return count >= minimum && count * 4 >= markers.size() * 3;
    };
    // On one plane the cube root of projected area is linear in image
    // coordinates (the homography's differential area is proportional to 1/w^3).
    cv::Vec3d area_plane;
    const auto fit_area = [&] {
        cv::Matx33d normal = cv::Matx33d::zeros();
        cv::Vec3d right{};
        for (const auto& m : markers)
            if (m.included) {
                normal += m.location * m.location.t();
                right += m.area_scale * m.location;
            }
        return cv::solve(normal, right, area_plane, cv::DECOMP_CHOLESKY);
    };
    for (double tolerance : {.06, .022}) {
        if (!fit_area())
            return false;
        for (auto& m : markers)
            m.included &=
                    std::abs(m.location.dot(area_plane) - m.area_scale) <= tolerance * m.area_scale;
        if (!supported())
            return false;
    }
    if (!fit_area())
        return false;
    const auto rectify = [&](XY p) {
        const XY d = p - origin;
        const double w =
                area_plane[0] + d.x / span.x * area_plane[1] + d.y / span.y * area_plane[2];
        return d / std::max(1e-6, w);
    };

    std::vector<std::size_t> indices;
    for (std::size_t i = 0; i < markers.size(); ++i)
        if (markers[i].included)
            indices.push_back(i);
    using Shape = std::array<XY, 4>;
    const auto shape_of = [&](std::size_t i) {
        Shape shape;
        for (int j = 0; j < 4; ++j)
            shape[j] = rectify(tags[i].corners_px_[j]);
        const auto center = .25 * (shape[0] + shape[1] + shape[2] + shape[3]);
        for (auto& p : shape)
            p -= center;
        return shape;
    };
    const auto anchor = shape_of(indices[indices.size() / 2]);
    std::array<std::vector<double>, 4> axes;
    std::vector<Shape> shapes(markers.size());
    for (auto i : indices) {
        const auto shape = shape_of(i);
        double best = std::numeric_limits<double>::max();
        int rotation = 0;
        for (int r = 0; r < 4; ++r) {
            double sum = 0;
            for (int j = 0; j < 4; ++j) {
                const XY e = shape[(j + r) % 4] - anchor[j];
                sum += e.dot(e);
            }
            if (sum < best) {
                best = sum;
                rotation = r;
            }
        }
        auto& aligned = shapes[i];
        for (int j = 0; j < 4; ++j)
            aligned[j] = shape[(j + rotation) % 4];
        const XY x = .5 * (aligned[1] - aligned[0] + aligned[2] - aligned[3]);
        const XY y = .5 * (aligned[2] - aligned[1] + aligned[3] - aligned[0]);
        axes[0].push_back(x.x);
        axes[1].push_back(x.y);
        axes[2].push_back(y.x);
        axes[3].push_back(y.y);
    }
    const XY xaxis{median(axes[0]), median(axes[1])}, yaxis{median(axes[2]), median(axes[3])};
    const double det = xaxis.cross(yaxis);
    if (!(std::abs(det) > 1e-5))
        return false;
    for (auto i : indices) {
        auto& m = markers[i];
        const auto p = rectify(m.center);
        m.local = {p.cross(yaxis) / det, xaxis.cross(p) / det};
        const auto& shape = shapes[i];
        const auto x = .5 * (shape[1] - shape[0] + shape[2] - shape[3]),
                   y = .5 * (shape[2] - shape[1] + shape[3] - shape[0]);
        m.included = cv::norm(x - xaxis) < .09 * cv::norm(xaxis) &&
                     cv::norm(y - yaxis) < .09 * cv::norm(yaxis);
    }
    if (!supported())
        return false;
    // Independent axis pitches allow rectangular layouts, missing cells and arbitrary code
    // rotations.
    std::vector<double> xsteps, ysteps;
    for (std::size_t n = 0; n < indices.size();
         n += std::max<std::size_t>(1, indices.size() / 20)) {
        const auto& m = markers[indices[n]];
        if (!m.included)
            continue;
        double dx = 5, dy = 5;
        for (const auto& other : markers)
            if (other.included) {
                const XY d = other.local - m.local;
                if (std::abs(d.y) < .2 && std::abs(d.x) > 1.03)
                    dx = std::min(dx, std::abs(d.x));
                if (std::abs(d.x) < .2 && std::abs(d.y) > 1.03)
                    dy = std::min(dy, std::abs(d.y));
            }
        if (dx < 5)
            xsteps.push_back(dx);
        if (dy < 5)
            ysteps.push_back(dy);
    }
    if (xsteps.size() < 3 || ysteps.size() < 3)
        return false;
    const XY pitch{median(xsteps), median(ysteps)};
    XY cosines{}, sines{};
    for (auto& m : markers)
        if (m.included) {
            m.local.x /= pitch.x;
            m.local.y /= pitch.y;
            cosines += XY(std::cos(2 * CV_PI * m.local.x), std::cos(2 * CV_PI * m.local.y));
            sines += XY(std::sin(2 * CV_PI * m.local.x), std::sin(2 * CV_PI * m.local.y));
        }
    const XY phase =
            XY(std::atan2(sines.x, cosines.x), std::atan2(sines.y, cosines.y)) * (.5 / CV_PI);
    for (auto& m : markers)
        if (m.included) {
            const XY relative = m.local - phase;
            m.index = {std::round(relative.x), std::round(relative.y)};
            m.included = std::max(std::abs(relative.x - m.index.x),
                                  std::abs(relative.y - m.index.y)) < .19;
        }
    for (double tol : {3.2 * max_shift, 1.5 * max_shift}) {
        if (!supported() || !fit_grid(markers, origin, span))
            return false;
        for (auto& m : markers)
            m.included &= cv::norm(m.predicted - m.center) < tol;
    }
    if (!supported() || !fit_grid(markers, origin, span))
        return false;
    for (std::size_t i = 0; i < tags.size(); ++i) {
        const auto& m = markers[i];
        const XY delta = m.predicted - m.center;
        if (m.included && cv::norm(delta) < max_shift)
            for (auto& corner : tags[i].corners_px_)
                corner += Point(delta);
    }
    return true;
}
}  // namespace aruco3cuda::apriltag_small_cpu::detail
