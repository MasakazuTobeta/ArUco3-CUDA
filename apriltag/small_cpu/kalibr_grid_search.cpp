// SPDX-License-Identifier: Apache-2.0
#include "kalibr_grid_search.hpp"

#include "gray_view.hpp"
#include "grid_alignment.hpp"

#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace aruco3cuda::apriltag_small_cpu::detail {
namespace {

#if CV_SIMD128 && !(defined(__aarch64__) && defined(__AARCH64EL__))
// Inverse threshold followed by a binary cross erosion is the threshold of
// the largest of five source intensities. Repeating the boundary pixel
// reproduces the neutral exterior of the standalone binary mask.
void dark_cross_core(const cv::Mat& gray, cv::Mat& core, int threshold) {
    core.create(gray.size(), CV_8U);
    for (int y = 0; y < gray.rows; ++y) {
        const auto* above = gray.ptr<std::uint8_t>(std::max(0, y - 1));
        const auto* row = gray.ptr<std::uint8_t>(y);
        const auto* below = gray.ptr<std::uint8_t>(std::min(gray.rows - 1, y + 1));
        auto* output = core.ptr<std::uint8_t>(y);
        int x = 0;
        constexpr int lanes = 16;
        if (gray.cols >= lanes) {
            const auto level = cv::v_setall_u8(static_cast<std::uint8_t>(threshold));
            auto prev = cv::v_setall_u8(row[0]), current = cv::v_load(row);
            for (; x + lanes <= gray.cols; x += lanes) {
                const int next_x = x + lanes;
                const auto next = next_x + lanes <= gray.cols
                                          ? cv::v_load(row + next_x)
                                          : cv::v_setall_u8(row[std::min(next_x, gray.cols - 1)]);
                // Successive blocks supply both horizontal neighbours without shifted loads.
                const auto horizontal =
                        cv::v_max(current, cv::v_max(cv::v_extract<lanes - 1>(prev, current),
                                                     cv::v_extract<1>(current, next)));
                const auto vertical = cv::v_max(cv::v_load(above + x), cv::v_load(below + x));
                cv::v_store(output + x, cv::v_le(cv::v_max(horizontal, vertical), level));
                prev = current;
                current = next;
            }
        }
        for (; x < gray.cols; ++x) {
            const auto maximum = std::max({above[x], below[x], row[x], row[std::max(0, x - 1)],
                                           row[std::min(gray.cols - 1, x + 1)]});
            output[x] = maximum <= threshold ? 255 : 0;
        }
    }
}
#endif

Point center_of(const Quad& quad) {
    return (quad[0] + quad[1] + quad[2] + quad[3]) * .25F;
}

Quad oriented_quad(const std::vector<cv::Point>& polygon, float scale_x, float scale_y) {
    Quad quad;
    for (int i = 0; i < 4; ++i)
        quad[i] = {std::floor(polygon[i].x * scale_x), std::floor(polygon[i].y * scale_y)};
    if (cross(quad[1] - quad[0], quad[2] - quad[1]) < 0)
        std::swap(quad[1], quad[3]);
    return quad;
}

Detection ordered_detection(const Quad& quad, const Decode& vote) {
    Quad corners;
    for (int j = 0; j < 4; ++j)
        corners[j] = quad[(j + 4 - vote.rotation) % 4];
    return {vote.id, corners, vote.distance};
}

void expand(Quad& quad, float growth, Point offset = {}) {
    const auto center = center_of(quad);
    for (auto& p : quad)
        p = center + (p - center) * growth + offset;
}

float border_sample(const cv::Mat& gray, const Projection& projection, int side, float position,
                    float inset) {
    const float coordinate = side < 2 ? inset : 1.F - inset;
    return side % 2 ? projection.sample(gray, position, coordinate)
                    : projection.sample(gray, coordinate, position);
}

// Only refine weak preview proposals when an exterior-to-interior transition is visible.
bool visible_outline(const cv::Mat& gray, const Quad& quad) {
    const auto center = center_of(quad);
    int positive = 0, sum = 0;
    for (int side = 0; side < 4; ++side) {
        const Point a = quad[side], b = quad[(side + 1) % 4];
        for (float t : {.25F, .5F, .75F}) {
            const Point edge = a + t * (b - a), ray = edge - center;
            const cv::Point outside(edge + .13F * ray), inside(edge - .19F * ray);
            const cv::Rect image(0, 0, gray.cols, gray.rows);
            if (!image.contains(outside) || !image.contains(inside))
                continue;
            const int delta = int(gray.ptr<std::uint8_t>(outside.y)[outside.x]) -
                              gray.ptr<std::uint8_t>(inside.y)[inside.x];
            positive += delta >= 14;
            sum += delta;
        }
    }
    return positive >= 8 && sum >= 220;
}

bool project_neighbour(const Projection& p, Point displacement, Quad& result, cv::Size size) {
    const std::array<Point, 4> unit = {Point{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    for (int i = 0; i < 4; ++i) {
        const Point uv = unit[i] + displacement;
        const float z = 1 + p.z_x * uv.x + p.z_y * uv.y;
        if (!(z > .001F))
            return false;
        const Point at = (p.initial + uv.x * p.x + uv.y * p.y) / z;
        if (!(at.x >= 0 && at.y >= 0 && at.x < size.width - 1 && at.y < size.height - 1))
            return false;
        result[i] = at;
    }
    return true;
}

// Fit the local image mapping from physical lattice coordinates to native pixels.
// The printed ID and its rotation are independent of the candidate's lattice site.
bool fit_neighbour(const std::vector<Detection>& tags, const Quad& base_quad, Point direction,
                   float pitch, Quad& prediction) {
    const Point origin = center_of(base_quad);
    const Point ax = .5F * pitch * (base_quad[1] - base_quad[0] + base_quad[2] - base_quad[3]);
    const Point ay = .5F * pitch * (base_quad[3] - base_quad[0] + base_quad[2] - base_quad[1]);
    const float det = cross(ax, ay);
    if (!(std::abs(det) > 1))
        return false;
    const std::array<Point, 4> corners = {Point{-.5F, -.5F}, {.5F, -.5F}, {.5F, .5F}, {-.5F, .5F}};
    auto basis = [](Point pos) {
        const double x = pos.x, y = pos.y;
        return cv::Vec<double, 6>{1, x, y, x * y, x * x, y * y};
    };
    cv::Matx<double, 6, 6> normal = cv::Matx<double, 6, 6>::zeros();
    cv::Matx<double, 6, 2> right = cv::Matx<double, 6, 2>::zeros();
    int supported = 0;
    for (const auto& tag : tags) {
        const auto& measured = tag.corners_px_;
        const Point center = center_of(measured), delta = center - origin;
        const Point approx{cross(delta, ay) / det, cross(ax, delta) / det};
        if (!(std::abs(approx.x) <= 2.4F && std::abs(approx.y) <= 2.4F))
            continue;
        const Point site{float(cvRound(approx.x)), float(cvRound(approx.y))};
        if (std::abs(approx.x - site.x) > .4F || std::abs(approx.y - site.y) > .4F)
            continue;
        float best = std::numeric_limits<float>::max();
        int rotation = 0;
        for (int r = 0; r < 4; ++r) {
            float error = 0;
            for (int j = 0; j < 4; ++j) {
                const Point d = (measured[(j + r) % 4] - center) - (base_quad[j] - origin);
                error += d.dot(d);
            }
            if (error < best) {
                best = error;
                rotation = r;
            }
        }
        for (int j = 0; j < 4; ++j) {
            const auto row = basis(pitch * site + corners[j]);
            const auto observed = measured[(j + rotation) % 4] - origin;
            normal += row * row.t();
            right += row * cv::Vec2d{observed.x, observed.y}.t();
        }
        ++supported;
    }
    cv::Matx<double, 6, 2> model;
    if (supported < 3 || !cv::solve(normal, right, model, cv::DECOMP_CHOLESKY))
        return false;
    for (int j = 0; j < 4; ++j) {
        const auto delta = model.t() * basis(pitch * direction + corners[j]);
        prediction[j] = origin + Point{float(delta[0]), float(delta[1])};
    }
    return true;
}

}  // namespace

KalibrGridSearch::KalibrGridSearch(KalibrGridConfig config) : config_(config) {
    if (config.proposal_dark_level_ < 0 || config.proposal_dark_level_ > 255 ||
        config.preview_dark_level_ < 0 || config.preview_dark_level_ > 255 ||
        config.preview_short_side_ < 1 || config.maximum_component_side_px_ < 4 ||
        config.maximum_component_side_px_ >= 32768 ||
        !(config.grid_pitch_ > 1 && config.grid_pitch_ < 4 && config.pitch_tolerance_ >= 0 &&
          config.pitch_tolerance_ < config.grid_pitch_ - 1) ||
        !(config.minimum_decode_contrast_ > 0 && config.minimum_decode_contrast_ <= 255) ||
        config.maximum_corrected_bits_ < 0 || config.maximum_corrected_bits_ > 2) {
        throw std::invalid_argument("Invalid two-border AprilGrid search configuration");
    }
    component_config_.maximum_quad_side_samples_ = config.maximum_component_side_px_;
    component_config_.minimum_fill_ratio_ = .27;
    component_config_.maximum_fill_ratio_ = 1;
    component_config_.minimum_quad_hull_ratio_ = .54;
    component_config_.maximum_aspect_ratio_ = 6;
    component_config_.hull_tolerance_ratio_ = .048;
    preview_kernel_ = cv::getStructuringElement(cv::MORPH_RECT, {2, 2});
    erosion_kernel_ = cv::getStructuringElement(cv::MORPH_CROSS, {3, 3});
}

Decode KalibrGridSearch::read(const cv::Mat& gray, const Quad& quad) const {
    Projection projection;
    if (!project(quad, projection))
        return {};
    float black_sum = 0, white_sum = 0;
    int visible_pairs = 0, inner_matches = 0, outer_matches = 0;
    for (int k = 0; k < 4; ++k) {
        const float t = (k + .5F) / 4;
        for (int side = 0; side < 4; ++side) {
            const float inner = border_sample(gray, projection, side, t, .12F);
            const float outer = border_sample(gray, projection, side, t, .055F);
            const float white = border_sample(gray, projection, side, t, -.075F);
            if (inner < 0 || outer < 0)
                return {};
            if (white < 0)
                continue;
            ++visible_pairs;
            black_sum += inner;
            white_sum += white;
            inner_matches += white > inner + 7;
            outer_matches += white > outer + 7;
        }
        // At most 16 pairs can become visible. Already observed negative
        // votes cannot be removed by the rest of the ring.
        const int maximum_visible = 16 - 4 * (k + 1) + visible_pairs;
        if (maximum_visible < 12 || 4 * (visible_pairs - inner_matches) > maximum_visible ||
            8 * (visible_pairs - outer_matches) > 3 * maximum_visible)
            return {};
    }
    // A frame boundary may clip the white surround while leaving the entire print visible.
    if (visible_pairs < 12)
        return {};
    const float contrast = (white_sum - black_sum) / visible_pairs;
    if (4 * inner_matches < 3 * visible_pairs || 8 * outer_matches < 5 * visible_pairs ||
        contrast < config_.minimum_decode_contrast_)
        return {};
    const float threshold = (white_sum + black_sum) / (2 * visible_pairs);
    MarkerCode code = 0;
    int confident_bits = 0;
    for (int y = 0; y < kPayloadSide; ++y) {
        for (int x = 0; x < kPayloadSide; ++x) {
            const float intensity = projection.sample(gray, (x + 2.5F) / 10, (y + 2.5F) / 10);
            if (intensity < 0)
                return {};
            code |= MarkerCode(intensity > threshold) << (kPayloadSide * y + x);
            confident_bits += std::abs(intensity - threshold) > .15F * contrast;
        }
        if (kPayloadSide * (y + 1) - confident_bits > 7)
            return {};
    }
    const auto [matched, distance] = CodeIndex::match(code, config_.maximum_corrected_bits_);
    if (matched < 0 || distance > config_.maximum_corrected_bits_)
        return {};

    // A one-cell print can yield the same word when both grids sample a broad payload cell.
    // Check the competing geometry and confirm that the second dark band really exists.
    MarkerCode one_cell = 0;
    for (int y = 0; y < kPayloadSide; ++y) {
        for (int x = 0; x < kPayloadSide; ++x) {
            const float intensity = projection.sample(gray, (x + 1.5F) / 8, (y + 1.5F) / 8);
            if (intensity < 0)
                return {};
            one_cell |= MarkerCode(intensity > threshold) << (kPayloadSide * y + x);
        }
    }
    if (CodeIndex::match(one_cell, 2).first >= 0)
        return {};
    const float second_band_threshold = black_sum / visible_pairs + .51F * contrast;
    int bright = 0;
    for (int side = 0; side < 4; ++side) {
        for (int k = 0; k < kPayloadSide; ++k) {
            const float value = border_sample(gray, projection, side, (k + 1.5F) / 8, .165F);
            if (value < 0 || ((value > second_band_threshold) && ++bright > 4))
                return {};
        }
    }
    return {matched / 4, matched % 4, distance, true};
}

cv::Rect KalibrGridSearch::uncovered_region(const cv::Mat& gray,
                                            const std::vector<Detection>& ordinary) {
    validate_image(gray);
    const int dx = std::max(6, gray.cols / 96), dy = std::max(6, gray.rows / 72);
    explained_.clear();
    for (const auto& tag : ordinary) {
        auto bounds = cv::boundingRect(tag.corners_px_);
        bounds += cv::Size(4, 4);
        bounds -= cv::Point(2, 2);
        explained_.push_back(bounds);
    }
    int left = gray.cols, top = gray.rows, right = -1, bottom = -1;
    for (int y = dy / 2; y < gray.rows; y += dy) {
        const auto* row = gray.ptr<std::uint8_t>(y);
        for (int x = dx / 2; x < gray.cols; x += dx) {
            if (row[x] > config_.preview_dark_level_)
                continue;
            bool covered = false;
            for (const auto& rect : explained_) {
                if (rect.contains(cv::Point{x, y})) {
                    covered = true;
                    break;
                }
            }
            if (!covered) {
                left = std::min(left, x);
                right = std::max(right, x);
                top = std::min(top, y);
                bottom = y;
            }
        }
    }
    if (right < left)
        return {};
    const int border = 2 * std::max(dx, dy) + std::max(right - left, bottom - top) / 6;
    return cv::Rect(cv::Point(left - border, top - border),
                    cv::Point(right + border + 1, bottom + border + 1)) &
           cv::Rect({}, gray.size());
}

bool KalibrGridSearch::has_candidate(const cv::Mat& gray) {
    validate_image(gray);
    const int step = std::clamp(
            1 + (std::min(gray.cols, gray.rows) - 1) / config_.preview_short_side_, 2, 8);
    cv::resize(gray, sampled_, {(gray.cols + step - 1) / step, (gray.rows + step - 1) / step}, 0, 0,
               cv::INTER_NEAREST);
    cv::threshold(sampled_, mask_, config_.preview_dark_level_, 255, cv::THRESH_BINARY_INV);
    cv::erode(mask_, cores_, preview_kernel_);
    component_config_.minimum_quad_side_samples_ = 3;
    components_.extract(cores_, component_config_, polygons_);
    const float sx = float(gray.cols) / sampled_.cols, sy = float(gray.rows) / sampled_.rows;
    for (const auto& polygon : polygons_) {
        const float area = cv::contourArea(polygon);
        if (area < 5)
            continue;
        const Quad original = oriented_quad(polygon, sx, sy);
        for (float margin : {2.5F, 0.F}) {
            Quad quad = original;
            expand(quad, 1 + margin / std::sqrt(area), {-.4F * sx, -.4F * sy});
            if (read(gray, quad).id >= 0 ||
                (visible_outline(gray, quad) && refine_quad(gray, quad, true) &&
                 read(gray, quad).id >= 0))
                return true;
        }
    }
    return false;
}

bool KalibrGridSearch::occupied(Point center, float radius) const {
    return std::any_of(tags_.begin(), tags_.end(), [&](const Detection& tag) {
        const Point delta = center - center_of(tag.corners_px_);
        return delta.dot(delta) < radius * radius;
    });
}

bool KalibrGridSearch::add_candidate(const cv::Mat& gray, Quad quad, bool check_ordinary) {
    auto refined = quad;
    Decode vote;
    const bool fitted = refine_quad(gray, refined, true);
    if (fitted)
        vote = read(gray, refined);
    if (vote.id >= 0)
        quad = refined;
    else
        vote = read(gray, quad);
    if (vote.id < 0) {
        // The same black component can belong to a different print convention.
        // Keep only exact current-frame words. A confirmed two-cell position takes precedence.
        if (check_ordinary) {
            auto standard = fitted ? decode(gray, refined, 22, 14, 0) : Decode{};
            if (standard.id >= 0) {
                ordinary_.push_back(ordered_detection(refined, standard));
            } else {
                standard = decode(gray, quad, 22, 14, 0);
                if (standard.id >= 0)
                    ordinary_.push_back(ordered_detection(quad, standard));
            }
        }
        return false;
    }
    const float side = std::sqrt(std::abs(cross(quad[1] - quad[0], quad[3] - quad[0])));
    if (!(side > 1) || occupied(center_of(quad), .3F * side))
        return false;
    tags_.push_back(ordered_detection(quad, vote));
    return true;
}

void KalibrGridSearch::complete_grid(const cv::Mat& gray) {
    if (tags_.size() < 2)
        return;
    for (std::size_t i = 0; i < tags_.size(); ++i) {
        const auto quad = tags_[i].corners_px_;
        Projection p;
        if (!project(quad, p))
            continue;
        float side = 0;
        for (int j = 0; j < 4; ++j)
            side += .25F * length(quad[(j + 1) % 4] - quad[j]);
        for (Point direction : {Point{1, 0}, {-1, 0}, {0, 1}, {0, -1}}) {
            const Point uv = Point{.5F, .5F} + config_.grid_pitch_ * direction;
            const float z = 1 + p.z_x * uv.x + p.z_y * uv.y;
            if (!(z > .001F))
                continue;
            const Point expected = (p.initial + uv.x * p.x + uv.y * p.y) / z;
            if (expected.x < 0 || expected.y < 0 || expected.x >= gray.cols ||
                expected.y >= gray.rows || occupied(expected, .47F * side))
                continue;
            bool found = false;
            for (float adjustment : {0.F, -config_.pitch_tolerance_, config_.pitch_tolerance_}) {
                Quad trial;
                if (project_neighbour(p, (config_.grid_pitch_ + adjustment) * direction, trial,
                                      gray.size()) &&
                    add_candidate(gray, trial)) {
                    found = true;
                    break;
                }
            }
            if (found || !config_.curved_grid_completion_)
                continue;

            // The single-marker projective estimate is sensitive to noise and peripheral bow.
            const Point axis = direction.x * .5F * (quad[1] - quad[0] + quad[2] - quad[3]) +
                               direction.y * .5F * (quad[3] - quad[0] + quad[2] - quad[1]);
            Quad translated;
            for (float adjustment : {0.F, -config_.pitch_tolerance_, config_.pitch_tolerance_}) {
                translated = quad;
                for (auto& point : translated)
                    point += (config_.grid_pitch_ + adjustment) * axis;
                if (visible_outline(gray, translated) && add_candidate(gray, translated)) {
                    found = true;
                    break;
                }
            }
            if (found)
                continue;

            // An interior gap needs a second decoded neighbour. At the frame boundary
            // that neighbour may be outside the image while the target is still visible.
            bool supported = false;
            const Point normal{-direction.y, direction.x};
            for (Point neighbour : {direction, normal, -normal}) {
                const Point site = uv + config_.grid_pitch_ * neighbour;
                const float depth = 1 + p.z_x * site.x + p.z_y * site.y;
                if (depth > .001F &&
                    occupied((p.initial + site.x * p.x + site.y * p.y) / depth, .65F * side)) {
                    supported = true;
                    break;
                }
            }
            const float border_distance = std::min(
                    {expected.x, expected.y, gray.cols - expected.x, gray.rows - expected.y});
            if (!supported && border_distance > .75F * side)
                continue;
            Quad fitted;
            if (fit_neighbour(tags_, quad, direction, config_.grid_pitch_, fitted) &&
                length(center_of(fitted) - expected) < .7F * side && add_candidate(gray, fitted))
                continue;
            Quad projected;
            if (!project_neighbour(p, config_.grid_pitch_ * direction, projected, gray.size())) {
                projected = quad;
                for (auto& point : projected)
                    point += config_.grid_pitch_ * axis;
            }
            const int margin = std::max(6, cvRound(.4F * side));
            auto region = cv::boundingRect(projected);
            region = (region + cv::Size(2 * margin, 2 * margin) - cv::Point(margin, margin)) &
                     cv::Rect({}, gray.size());
            if (region.width < 8 || region.height < 8)
                continue;
            double low, high;
            cv::minMaxLoc(gray(region), &low, &high);
            if (high - low < 18)
                continue;
            for (float fraction : {.375F, .22F, .5F}) {
                if (find_seeds(gray, region, low + fraction * (high - low), expected, .62F * side))
                    break;
            }
        }
    }
}

bool KalibrGridSearch::find_seeds(const cv::Mat& gray, cv::Rect region, double level, Point target,
                                  float maximum_distance, bool check_ordinary) {
    // Disconnect the printed corner patches from the black border before fitting its hull.
    component_config_.minimum_quad_side_samples_ = 4;
#if defined(__aarch64__) && defined(__AARCH64EL__)
    components_.extract_dark_cross(gray(region), cvFloor(level), component_config_, polygons_);
#else
#if CV_SIMD128
    dark_cross_core(gray(region), cores_, cvFloor(level));
#else
    cv::threshold(gray(region), mask_, level, 255, cv::THRESH_BINARY_INV);
    cv::erode(mask_, cores_, erosion_kernel_);
#endif
    components_.extract(cores_, component_config_, polygons_);
#endif
    const auto previous = tags_.size();
    for (const auto& polygon : polygons_) {
        const float area = cv::contourArea(polygon);
        if (area < 14)
            continue;
        Quad quad = oriented_quad(polygon, 1, 1);
        for (auto& point : quad)
            point += Point(region.tl());
        const auto center = center_of(quad);
        if (maximum_distance > 0 && length(center - target) > maximum_distance)
            continue;
        const float side = std::sqrt(area);
        if (!occupied(center, .3F * side)) {
            expand(quad, 1 + 3.4F / side);
            if (add_candidate(gray, quad, check_ordinary) && maximum_distance > 0)
                return true;
        }
    }
    return tags_.size() > previous;
}

const std::vector<Detection>& KalibrGridSearch::detect(const cv::Mat& gray, bool check_ordinary) {
    validate_image(gray);
    tags_.clear();
    ordinary_.clear();
    find_seeds(gray, cv::Rect({}, gray.size()), config_.proposal_dark_level_, {}, -1,
               check_ordinary);
    if (tags_.size() < 2 && config_.preview_dark_level_ != config_.proposal_dark_level_)
        find_seeds(gray, cv::Rect({}, gray.size()), config_.preview_dark_level_, {}, -1,
                   check_ordinary);
    complete_grid(gray);
    if (tags_.empty()) {
        ordinary_.clear();
    } else {
        auto kept = ordinary_.begin();
        for (auto it = ordinary_.begin(); it != ordinary_.end(); ++it) {
            const auto center = center_of(it->corners_px_);
            const float radius = .27F * length(it->corners_px_[2] - it->corners_px_[0]);
            if (!occupied(center, radius) &&
                std::none_of(ordinary_.begin(), kept, [&](const auto& other) {
                    return length(center - center_of(other.corners_px_)) < radius;
                })) {
                *kept = *it;
                try_align_sharp_grid(gray, *kept);
                ++kept;
            }
        }
        ordinary_.erase(kept, ordinary_.end());
    }
    return tags_;
}
}  // namespace aruco3cuda::apriltag_small_cpu::detail
