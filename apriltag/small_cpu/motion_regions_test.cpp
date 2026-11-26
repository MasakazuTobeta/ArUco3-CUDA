// SPDX-License-Identifier: Apache-2.0
#include "motion_regions.hpp"
#include "entering_strip_search.hpp"
#include "grid_alignment.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

namespace {
using aruco3cuda::apriltag_small_cpu::detail::MotionRegions;

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

void strided_views() {
    cv::RNG rng(12345);
    for (int h : {1, 2, 3, 4, 5, 15, 31, 63, 120, 171}) {
        for (int w : {1, 3, 4, 15, 31, 64, 65, 169}) {
            cv::Mat parent(h + 5, w + 11, CV_8UC1);
            rng.fill(parent, cv::RNG::UNIFORM, 10, 240);
            cv::Mat frame = parent(cv::Rect(4, 1, w, h));
            MotionRegions view, owned;
            for (int i = 0; i < 6; ++i) {
                if (i == 4) {
                    view.reset();
                    owned.reset();
                }
                if (i == 2) {
                    frame(cv::Rect(w / 3, h / 3, std::max(1, w / 3), std::max(1, h / 3))).setTo(17);
                }
                std::vector<cv::Rect> changed{{20, 20, 1, 1}}, expected{{30, 30, 1, 1}};
                cv::Point shift(13, 13), expected_shift(12, 12);
                bool overlap = true, expected_overlap = true;
                bool local = view.update(frame, i == 3, shift, changed, &overlap);
                bool expected_local = owned.update(frame.clone(), i == 3, expected_shift, expected,
                                                   &expected_overlap);
                require(local == expected_local && shift == expected_shift && changed == expected &&
                                overlap == expected_overlap,
                        "pitched video history differs from a clone");
                require(local || changed.empty(), "full-frame request retained a stale crop");
                if (i == 0 || i == 3 || i == 4 || std::min(h, w) < 4)
                    require(!local && !overlap,
                            "first, forced or tiny frame must request discovery");
                for (const auto& rect : changed)
                    require((rect & cv::Rect({}, frame.size())) == rect, "crop outside frame");
            }
        }
    }
}

void translated_regions() {
    cv::RNG rng(54321);
    cv::Mat noise(480, 640, CV_8UC1), original;
    rng.fill(noise, cv::RNG::UNIFORM, 0, 256);
    cv::GaussianBlur(noise, original, {5, 5}, 1.4);
    original.convertTo(original, CV_8U, 2, -105);
    for (const cv::Point expected :
         {cv::Point(), cv::Point(-2, -1), cv::Point(3, 2), cv::Point(8, -8), cv::Point(-6, 8)}) {
        MotionRegions regions;
        cv::Point shift;
        std::vector<cv::Rect> changes;
        bool overlap = true;
        require(!regions.update(original, false, shift, changes, &overlap) && !overlap,
                "first frame already has a motion reference");
        cv::Mat next;
        cv::Matx23d affine(1, 0, expected.x, 0, 1, expected.y);
        cv::warpAffine(original, next, affine, original.size(), cv::INTER_NEAREST,
                       cv::BORDER_REPLICATE);
        require(regions.update(next, false, shift, changes, &overlap) && overlap &&
                        shift == expected && changes.empty(),
                "translation caused a spurious crop");
        const cv::Rect target(300, 220, 34, 30);
        next(target).setTo(0);
        require(regions.update(next, false, shift, changes) &&
                        std::any_of(changes.begin(), changes.end(),
                                    [&](auto rect) { return (rect & target) == target; }),
                "localized change did not cover the affected pixels");
        require(!regions.update(next(cv::Rect(0, 0, 128, 128)), false, shift, changes, &overlap) &&
                        !overlap && shift == cv::Point(),
                "size change reused previous coordinates");
    }
}

void low_contrast_flicker() {
    cv::Mat previous(360, 640, CV_8UC1), current;
    cv::RNG noise(988);
    noise.fill(previous, cv::RNG::UNIFORM, 10, 230);
    previous.copyTo(current);
    // Exactly three sampled pixels change in each 32x32 tile but miss the sparse anchors.
    for (int y = 0; y < current.rows; y += 32)
        for (int x = 0; x + 8 < current.cols; x += 32)
            for (int dx : {0, 4, 8})
                current.at<uchar>(y, x + dx) += 5;
    const cv::Rect target(290, 190, 20, 24);
    current(target).setTo(0);
    MotionRegions regions;
    cv::Point shift;
    bool overlap = false;
    std::vector<cv::Rect> changes;
    require(!regions.update(previous, false, shift, changes), "first flicker frame");
    require(regions.update(current, false, shift, changes, &overlap) && overlap &&
                    changes.size() == 1 && (changes.front() & target) == target,
            "widespread flicker hid the isolated high-contrast change");
}
void tile_cutoffs() {
    cv::Mat original(475, 637, CV_8UC1, cv::Scalar(130));
    for (int difference : {2, 3}) {
        for (int n : {2, 3}) {
            cv::Mat current = original.clone();
            // A partial final tile exercises the scalar tail of the same detector.
            for (int i = 0; i < n; ++i)
                current.at<uchar>(96 + 4 * i, 632) += difference;
            MotionRegions motion;
            cv::Point shift;
            std::vector<cv::Rect> changed;
            require(!motion.update(original, false, shift, changed), "first cutoff frame");
            require(motion.update(current, false, shift, changed) &&
                            changed.size() == std::size_t(difference > 2 && n >= 3),
                    "strict pixel or minimum tile count cutoff changed");
        }
    }
}

void crop_budget() {
    cv::Mat before(480, 640, CV_8UC1, cv::Scalar(150)), after = before.clone();
    for (int x : {80, 280, 480}) {
        for (int y : {80, 280})
            after(cv::Rect(x, y, 24, 24)).setTo(30);
    }
    cv::Point shift;
    std::vector<cv::Rect> changed;
    for (int limit : {0, 5, 6}) {
        MotionRegions motion(2, 22, 8, 0, limit);
        require(!motion.update(before, false, shift, changed), "budget initial frame");
        bool overlap = false;
        const bool local = motion.update(after, false, shift, changed, &overlap);
        require(overlap && (limit == 6 ? local && changed.size() == 6 : !local && changed.empty()),
                "motion ignored the configured region budget");
    }
}
// Canonicalize cyclic order and direction so the oracle compares boundary geometry.
std::vector<int> canonical_chain(const std::vector<cv::Point>& curve, int width) {
    std::vector<cv::Point> polygon;
    cv::approxPolyDP(curve, polygon, 0, true);
    if (polygon.size() < 4)
        return {};
    std::vector<int> points, best;
    for (const auto p : polygon)
        points.push_back(p.y * width + p.x);
    for (int direction = 0; direction < 2; ++direction) {
        for (std::size_t i = 0; i < points.size(); ++i) {
            if (best.empty() || points < best)
                best = points;
            std::rotate(points.begin(), points.begin() + 1, points.end());
        }
        std::reverse(points.begin(), points.end());
    }
    return best;
}

void entering_contour_oracle() {
    using aruco3cuda::apriltag_small_cpu::detail::EnteringContours;
    EnteringContours tracer;
    cv::RNG rng(984);
    for (int i = 0; i < 1100; ++i) {
        const int width = i < 512 ? 3 : 17, height = i < 512 ? 3 : 9;
        // A bright parent checks that padding treats this input as an isolated view.
        cv::Mat parent(height + 3, width + 7, CV_8UC1, cv::Scalar(255));
        cv::Mat mask = parent(cv::Rect(3, 1, width, height));
        if (i < 512) {
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x)
                    mask.at<uchar>(y, x) = (i >> (y * width + x)) & 1 ? 255 : 0;
        } else {
            rng.fill(mask, cv::RNG::UNIFORM, 0, 255);
            cv::threshold(mask, mask, 70 + i % 100, 255, cv::THRESH_BINARY);
        }
        // The production tracer reads stride-two native pixels directly. Poison
        // neighbours and row padding, and vary even/odd extents independently.
        const int level = i % 255;
        cv::Mat source_parent(2 * height + 5, 2 * width + 9, CV_8UC1);
        rng.fill(source_parent, cv::RNG::UNIFORM, 0, 256);
        const cv::Rect crop(3, 2, 2 * width - (i % 2), 2 * height - ((i / 2) % 2));
        cv::Mat source = source_parent(crop);
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                source.at<uchar>(2 * y, 2 * x) = mask.at<uchar>(y, x) ? level : level + 1;
        std::vector<std::vector<cv::Point>> all;
        cv::findContours(mask.clone(), all, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        int k = i < 512 ? 1 : 3;
        const std::array<cv::Rect, 7> sides{{{0, 0, k, height},
                                             {width - k, 0, k, height},
                                             {0, 0, width, k},
                                             {0, height - k, width, k},
                                             {-1, -1, 3, 3},
                                             {width - 1, height - 1, 3, 3},
                                             {width, height, 3, 3}}};
        for (auto seed : sides) {
            std::set<std::vector<int>> expected, actual;
            for (const auto& curve : all) {
                for (std::size_t j = 0; j < curve.size(); ++j) {
                    cv::Point a = curve[j], b = curve[(j + 1) % curve.size()];
                    if (cv::clipLine(seed, a, b)) {
                        expected.insert(canonical_chain(curve, width));
                        break;
                    }
                }
            }
            for (const auto& curve : tracer.find(source, level, seed))
                actual.insert(canonical_chain(curve, width));
            // The detector requires at least four corners; single-pixel rings can yield
            // identical foreground chains for the outer and inner boundary.
            expected.erase(std::vector<int>{});
            actual.erase(std::vector<int>{});
            require(actual == expected,
                    "entering contour differs from the one-thread OpenCV oracle");
        }
    }
    for (cv::Size size : {cv::Size(1, 1), cv::Size(1, 13), cv::Size(13, 1)}) {
        cv::Mat line(size, CV_8UC1, cv::Scalar(191));
        require(tracer.find(line, 255, {-2, -2, 20, 20}).empty(),
                "Degenerate strip acquired a polygon");
    }
    cv::Mat full(7, 7, CV_8UC1, cv::Scalar(255));
    require(tracer.find(full, 255, {0, 0, 1, 4}).size() == 1,
            "Maximum threshold lost the white edge");
    require(tracer.find(full, 254, {0, 0, 1, 4}).empty(), "Threshold ignored the current frame");
}

void entering_tags() {
    using namespace aruco3cuda::apriltag_small_cpu;
    detail::EnteringStripSearch entering(128, 120);
    detail::EnteringStripSearch disabled(0, 120);
    cv::Mat storage(187, 283, CV_8UC1, cv::Scalar(29));
    cv::Mat frame = storage(cv::Rect(4, 5, 273, 179));
    constexpr std::uint64_t id_zero = 0xd5d628584ULL;
    constexpr int cell = 5, side = 8 * cell;
    cv::Mat pattern(side, side, CV_8UC1, cv::Scalar(20));
    for (int bit = 0; bit < 36; ++bit)
        if ((id_zero >> bit) & 1)
            pattern(cv::Rect((1 + bit % 6) * cell, (1 + bit / 6) * cell, cell, cell)).setTo(230);
    Config config;
    config.components_.emplace();
    config.components_->threshold_ = 120;
    config.components_->sampled_contours_ = true;
    Detector reference(config);
    for (int direction = 0; direction < 4; ++direction) {
        const bool vertical = direction < 2, far = direction % 2;
        cv::Point shift = vertical ? cv::Point(far ? -5 : 5, 0) : cv::Point(0, far ? -5 : 5);
        const int extent = vertical ? frame.cols : frame.rows;
        const int depth = 128;
        const int pos = far ? extent - 4 - side : 4;
        const cv::Point origin = vertical ? cv::Point(pos, 65) : cv::Point(123, pos);
        frame.setTo(230);
        pattern.copyTo(frame(cv::Rect(origin, pattern.size())));
        const cv::Rect strip = vertical ? cv::Rect(far ? extent - depth : 0, 0, depth, frame.rows)
                                        : cv::Rect(0, far ? extent - depth : 0, frame.cols, depth);
        const auto oracle = reference.detect(frame(strip));
        const auto observed = entering.detect(frame, shift);
        require(observed.size() == 1 && oracle.size() == 1 && observed.front().id_ == 0 &&
                        observed.front().corrected_bits_ == oracle.front().corrected_bits_,
                "entering marker decode differs from the complete strip");
        for (int j = 0; j < 4; j++)
            require(cv::norm(observed[0].corners_px_[j] -
                             (oracle[0].corners_px_[j] + cv::Point2f(strip.tl()))) < 1.0,
                    "entering marker corner differs from the complete strip");
        require(entering.detect(frame, -shift).empty() && entering.detect(frame, {}).empty() &&
                        disabled.detect(frame, shift).empty(),
                "entry search ran on an unexposed edge");
    }
}
void pan_exposure_and_size() {
    using aruco3cuda::apriltag_small_cpu::detail::EnteringStripSearch;
    EnteringStripSearch search;
    EnteringStripSearch nominal_only(288, 120, 0);
    constexpr std::uint64_t id_zero = 0xd5d628584ULL;
    for (int cell : {5, 23}) {
        const int side = 8 * cell;
        for (const auto levels : {cv::Point(39, 101), cv::Point(137, 212)}) {
            cv::Mat storage(389, 651, CV_8UC1, cv::Scalar(150));
            cv::Mat frame = storage(cv::Rect(3, 3, 640, 384));
            const cv::Point at(5, 103);
            const cv::Rect square(at, cv::Size(side, side));
            const cv::Rect halo = (square - cv::Point(cell, cell) + cv::Size(2 * cell, 2 * cell)) &
                                  cv::Rect({}, frame.size());
            frame(halo).setTo(levels.y);
            frame(square).setTo(levels.x);
            for (int bit = 0; bit < 36; ++bit) {
                if ((id_zero >> bit) & 1) {
                    frame(cv::Rect(at.x + (1 + bit % 6) * cell, at.y + (1 + bit / 6) * cell, cell,
                                   cell))
                            .setTo(levels.y);
                }
            }
            const auto found = search.detect(frame, {7, 0});
            require(std::any_of(found.begin(), found.end(),
                                [](const auto& t) { return t.id_ == 0; }),
                    "incoming tag missed outside the nominal threshold or beyond 128 pixels");
            require(nominal_only.detect(frame, {7, 0}).empty(),
                    "disabled supplemental threshold ran");
        }
    }
    cv::Mat tiny(2, 3, CV_8UC1, cv::Scalar(127));
    require(search.detect(tiny, {std::numeric_limits<int>::min(), 1}).empty(),
            "tiny image produced an entering tag");
}
// Point-sample a digital tag at pixel centers. An area-averaging renderer exercises
// a different edge model and cannot test the quantization interval at each crossing.
void binary_corner_accuracy() {
    using aruco3cuda::apriltag_small_cpu::Detection;
    using aruco3cuda::apriltag_small_cpu::detail::try_align_sharp_grid;
    const std::array<cv::Point2f, 4> unit{{{0, 0}, {1, 0}, {1, 1}, {0, 1}}};
    constexpr std::uint64_t code = 0xd5d628584ULL;
    double sum_squared = 0, seed_squared = 0;
    int count = 0;
    for (const float size : {29.F, 48.F, 70.F, 90.F}) {
        for (const float angle : {-.34F, .27F, .65F}) {
            cv::Mat parent(171, 193, CV_8UC1, cv::Scalar(101));
            cv::Mat frame = parent(cv::Rect(3, 7, 183, 150));
            const cv::Point2f a(size * std::cos(angle), size * std::sin(angle));
            const cv::Point2f b(-.94F * a.y, .94F * a.x);
            const auto start = cv::Point2f(90.25F, 74.61F) - .5F * (a + b);
            const std::array<cv::Point2f, 4> truth{
                    start, start + a, start + a + b + cv::Point2f(1.2F, -.7F), start + b};
            const cv::Matx33d inverse(cv::getPerspectiveTransform(truth.data(), unit.data()));
            for (int y = 0; y < frame.rows; ++y) {
                auto* row = frame.ptr<std::uint8_t>(y);
                for (int x = 0; x < frame.cols; ++x) {
                    const cv::Vec3d mapped = inverse * cv::Vec3d(x, y, 1);
                    const int u = cvFloor(8 * mapped[0] / mapped[2]);
                    const int v = cvFloor(8 * mapped[1] / mapped[2]);
                    if (u < 0 || v < 0 || u > 7 || v > 7)
                        continue;
                    const bool white = u > 0 && u < 7 && v > 0 && v < 7 &&
                                       ((code >> (6 * (v - 1) + u - 1)) & 1);
                    row[x] = white ? 101 : 39;
                }
            }
            Detection fit{};
            fit.id_ = 0;
            for (int i = 0; i < 4; ++i)
                fit.corners_px_[i] = truth[i] + cv::Point2f(i % 2 ? .43F : -.32F, .37F);
            auto rough = fit;
            rough.corners_px_[2] += cv::Point2f(2.7F, -2.1F);
            require(try_align_sharp_grid(frame, rough),
                    "coarse oblique corners failed re-alignment");
            for (int i = 0; i < 4; ++i) {
                const auto error = rough.corners_px_[i] - truth[i];
                seed_squared += error.dot(error);
            }
            auto other_id = fit;
            ++other_id.id_;
            require(!try_align_sharp_grid(frame, other_id) &&
                            other_id.corners_px_ == fit.corners_px_,
                    "binary corner update accepted a different payload");
            auto owned = fit;
            require(try_align_sharp_grid(frame, fit) &&
                            try_align_sharp_grid(frame.clone(), owned) &&
                            fit.corners_px_ == owned.corners_px_,
                    "two-level corner fit differs between contiguous and pitched frames");
            for (int i = 0; i < 4; ++i) {
                const auto error = fit.corners_px_[i] - truth[i];
                sum_squared += error.dot(error);
                ++count;
            }
        }
    }
    require(std::sqrt(seed_squared / count) < .15, "coarse corner seed kept a multi-pixel error");
    require(std::sqrt(sum_squared / count) < .075,
            "point-sampled oblique tag corners lost subpixel accuracy");
}
void curved_corner_accuracy() {
    using aruco3cuda::apriltag_small_cpu::Detection;
    using aruco3cuda::apriltag_small_cpu::detail::try_align_curved_grid;
    const cv::Point2d principal(239.5, 149.5);
    constexpr double focal = 186, side = .42, tilt = .85, roll = .27;
    const cv::Vec3d center(.68, .08, .60);
    const cv::Vec3d horizontal(std::cos(tilt), 0, -std::sin(tilt)), down(0, 1, 0);
    const auto u = std::cos(roll) * horizontal + std::sin(roll) * down;
    const auto v = -std::sin(roll) * horizontal + std::cos(roll) * down;
    const auto normal = u.cross(v);
    auto project = [&](const cv::Vec3d& world) {
        const double radius = std::hypot(world[0], world[1]);
        return principal +
               cv::Point2d(world[0], world[1]) * (focal * std::atan2(radius, world[2]) / radius);
    };
    std::array<cv::Point2f, 4> truth;
    for (int i = 0; i < 4; ++i)
        truth[i] = project(center +
                           side * ((i == 1 || i == 2 ? .5 : -.5) * u + (i >= 2 ? .5 : -.5) * v));
    constexpr std::uint64_t code = 0xd5d628584ULL;
    for (int samples : {1, 3}) {
        cv::Mat storage(307, 491, CV_8UC1, cv::Scalar(155));
        cv::Mat gray = storage(cv::Rect(5, 3, 480, 300));
        for (int y = 0; y < gray.rows; ++y) {
            auto* row = gray.ptr<std::uint8_t>(y);
            for (int x = 0; x < gray.cols; ++x) {
                int sum = 0;
                for (int j = 0; j < samples; ++j) {
                    for (int i = 0; i < samples; ++i) {
                        const double a = (x + (i + .5) / samples - .5 - principal.x) / focal;
                        const double b = (y + (j + .5) / samples - .5 - principal.y) / focal;
                        const double theta = std::hypot(a, b);
                        const double scale = theta ? std::sin(theta) / theta : 1.;
                        const cv::Vec3d ray(scale * a, scale * b, std::cos(theta));
                        const double denominator = normal.dot(ray);
                        if (denominator <= 0) {
                            sum += 155;
                            continue;
                        }
                        const auto point =
                                ((normal.dot(center) / denominator) * ray - center) / side;
                        const int col = cvFloor(8 * (point.dot(u) + .5));
                        const int r = cvFloor(8 * (point.dot(v) + .5));
                        const bool paper = col >= -1 && col <= 8 && r >= -1 && r <= 8;
                        const bool inside = col >= 0 && col < 8 && r >= 0 && r < 8;
                        const bool payload = col >= 1 && col <= 6 && r >= 1 && r <= 6;
                        const bool white =
                                !inside || (payload && (code >> (6 * (r - 1) + col - 1) & 1));
                        sum += !paper ? 155 : white ? 227 : 24;
                    }
                }
                row[x] = static_cast<std::uint8_t>((sum + samples * samples / 2) /
                                                   (samples * samples));
            }
        }
        Detection view{};
        view.id_ = 0;
        for (int k = 0; k < 4; ++k)
            view.corners_px_[k] = truth[k] + cv::Point2f(k % 2 ? 1.5F : -1.F, k < 2 ? 1.F : -1.F);
        auto clone = view, wrong = view;
        const auto seed = view.corners_px_;
        wrong.id_ = 1;
        require(try_align_curved_grid(gray, view) && try_align_curved_grid(gray.clone(), clone) &&
                        view.corners_px_ == clone.corners_px_,
                "curved native edge fit failed or depends on row stride");
        require(!try_align_curved_grid(gray, wrong) && wrong.corners_px_ == seed,
                "curved edge fit accepted a different oriented payload");
        double squared = 0;
        for (int k = 0; k < 4; ++k) {
            const auto error = view.corners_px_[k] - truth[k];
            squared += error.dot(error);
        }
        require(std::sqrt(squared / 4) < (samples == 1 ? .25 : .12),
                "curved native edge fit kept a straight-chord corner bias");
    }
}
}  // namespace

int main() {
    cv::setNumThreads(1);
    strided_views();
    translated_regions();
    low_contrast_flicker();
    crop_budget();
    tile_cutoffs();
    entering_contour_oracle();
    entering_tags();
    pan_exposure_and_size();
    binary_corner_accuracy();
    curved_corner_accuracy();
}
