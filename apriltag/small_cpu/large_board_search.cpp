// SPDX-License-Identifier: Apache-2.0
#include "large_board_search.hpp"

#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace aruco3cuda::apriltag_small_cpu::detail {
namespace {
Point project_point(const Projection& p, Point uv) {
    return (p.initial + uv.x * p.x + uv.y * p.y) / (1 + uv.x * p.z_x + uv.y * p.z_y);
}

Point inverse_point(const Projection& p, Point pixel) {
    const Point x = p.x - p.z_x * pixel, y = p.y - p.z_y * pixel, delta = pixel - p.initial;
    const float determinant = cross(x, y);
    if (std::abs(determinant) < 1e-6F)
        return {-100, -100};
    return {cross(delta, y) / determinant, cross(x, delta) / determinant};
}

Point center(const Quad& q) {
    return (q[0] + q[1] + q[2] + q[3]) * 0.25F;
}

float median(std::vector<float>& values) {
    const auto middle = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), middle, values.end());
    return *middle;
}

float pitch_to(const Projection& a, const Projection& b, Point b_center) {
    const Point d = inverse_point(a, b_center) - Point(0.5F, 0.5F);
    const float pitch = std::max(std::abs(d.x), std::abs(d.y));
    if (std::min(std::abs(d.x), std::abs(d.y)) > 0.16F || pitch < 1.025F || pitch > 2.2F)
        return 0;
    const Point axis = inverse_point(a, project_point(b, {1, 0.5F})) -
                       inverse_point(a, project_point(b, {0, 0.5F}));
    if (std::min(std::abs(axis.x), std::abs(axis.y)) > 0.16F || length(axis) < 0.8F ||
        length(axis) > 1.2F)
        return 0;
    return pitch;
}
bool is_oblique(const Quad& corners, const ObliqueBoardConfig& config) {
    float longest_squared = 0;
    for (int k = 0; k < 4; ++k) {
        const auto edge = corners[(k + 1) % 4] - corners[k];
        longest_squared = std::max(longest_squared, edge.dot(edge));
    }
    const float double_area = std::abs(cross(corners[2] - corners[0], corners[3] - corners[1]));
    return longest_squared > config.minimum_long_edge_px_ * config.minimum_long_edge_px_ &&
           double_area < 2 * config.maximum_area_ratio_ * longest_squared;
}

// A similarly sized print on the same plane maps to a unit square centered at its own
// location in the seed's coordinates. Its four corners may be in any cyclic order.
bool same_square(const Projection& seed, const Projection& neighbor, Point local_center,
                 float tolerance) {
    int quadrants = 0;
    for (const auto vertex : {Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)}) {
        const auto delta = inverse_point(seed, project_point(neighbor, vertex)) - local_center;
        if (!(std::abs(std::abs(delta.x) - .5F) < tolerance &&
              std::abs(std::abs(delta.y) - .5F) < tolerance))
            return false;
        quadrants |= 1 << (int(delta.x > 0) + 2 * int(delta.y > 0));
    }
    return quadrants == 15;
}

// The wider decoder ring may span an appreciable native distance along a foreshortened
// edge. Check the near side of that edge separately on each of the four borders.
bool supports_oblique_edges(const cv::Mat& gray, const Quad& quad, float minimum_contrast) {
    Projection projection;
    if (!project(quad, projection))
        return false;
    std::array<int, 4> supported{};
    constexpr float kInset = .035F;
    for (int i = 0; i < kBorderSide; ++i) {
        const float t = (i + .5F) / kBorderSide;
        const std::array<std::pair<Point, Point>, 4> locations{
                {{{-kInset, t}, {kInset, t}},
                 {{1 + kInset, t}, {1 - kInset, t}},
                 {{t, -kInset}, {t, kInset}},
                 {{t, 1 + kInset}, {t, 1 - kInset}}}};
        for (std::size_t k = 0; k < locations.size(); ++k) {
            const auto [outside, inside] = locations[k];
            const float light = projection.sample(gray, outside.x, outside.y);
            const float dark = projection.sample(gray, inside.x, inside.y);
            supported[k] += dark >= 0 && light - dark >= .68F * minimum_contrast;
        }
    }
    return std::all_of(supported.begin(), supported.end(), [](int n) { return n >= 6; });
}

Config small_config(int threshold) {
    Config config;
    config.dark_threshold_ = config.bright_threshold_ = threshold;
    config.bright_sample_floor_ = std::max(config.bright_sample_floor_, threshold);
    return config;
}
}  // namespace

LargeBoardSearch::LargeBoardSearch(LargeBoardConfig config)
    : config_(config),
      small_(small_config(config.small_threshold_)),
      dark_(small_config(config.small_dark_threshold_)),
      bright_(small_config(config.small_bright_threshold_)) {
    const auto& oblique = config.oblique_;
    if (oblique.minimum_seeds_ < 3 || oblique.maximum_seeds_ < oblique.minimum_seeds_ ||
        oblique.maximum_seeds_ > 512 ||
        !(oblique.minimum_long_edge_px_ >= 6 && oblique.minimum_long_edge_px_ < 32768) ||
        !(oblique.maximum_area_ratio_ >= 0 && oblique.maximum_area_ratio_ <= 1) ||
        !(oblique.minimum_pitch_ > 1 && oblique.maximum_pitch_ >= oblique.minimum_pitch_ &&
          oblique.maximum_pitch_ <= 4) ||
        !(oblique.maximum_shape_error_ > 0 && oblique.maximum_shape_error_ < .5F) ||
        oblique.minimum_ring_matches_ < 24 || oblique.minimum_ring_matches_ > kRingSamples ||
        !(oblique.minimum_contrast_ > 0 && oblique.minimum_contrast_ <= 255))
        throw std::invalid_argument("Invalid oblique-board neighbor parameters");
    if (config.stride_px_ < 2 || config.stride_px_ > 8 || config.adaptive_window_ < 3 ||
        config.adaptive_window_ > 127 || config.adaptive_window_ % 2 == 0 ||
        !std::isfinite(config.adaptive_offset_) || config.minimum_side_samples_ < 4 ||
        config.minimum_side_samples_ > 1024 ||
        !(config.minimum_area_samples_ >= 1 && config.minimum_area_samples_ <= 1048576) ||
        !(config.polygon_tolerance_ratio_ > 0 && config.polygon_tolerance_ratio_ <= 0.5) ||
        config.minimum_ring_matches_ < 1 || config.minimum_ring_matches_ > 32 ||
        !(config.minimum_contrast_ > 0 && config.minimum_contrast_ <= 255) ||
        config.maximum_neighbor_trials_ < 0 || config.maximum_neighbor_trials_ > 4096 ||
        config.small_dark_threshold_ < 0 ||
        config.small_dark_threshold_ >= config.small_threshold_ ||
        config.small_threshold_ >= config.small_bright_threshold_ ||
        config.small_bright_threshold_ > 255 || config.secondary_hysteresis_ < 1 ||
        config.secondary_hysteresis_ > 127 || config.secondary_padding_px_ < 0 ||
        config.secondary_padding_px_ > 256 || config.small_roi_padding_px_ < 0 ||
        config.small_roi_padding_px_ > 256 ||
        std::any_of(config.expansion_samples_.begin(), config.expansion_samples_.end(),
                    [](float x) { return !(x >= -4 && x <= 4); }))
        throw std::invalid_argument("Invalid large-board candidate parameters");
}

void LargeBoardSearch::append(Quad quad, const Decode& tag) {
    std::rotate(quad.begin(), quad.begin() + (4 - tag.rotation) % 4, quad.end());
    tags_.push_back({tag.id, quad, tag.distance});
}

void LargeBoardSearch::accept_contour(const cv::Mat& gray, const Quad& quad) {
    const Point middle = center(quad);
    const float side = 0.25F * (length(quad[1] - quad[0]) + length(quad[2] - quad[1]) +
                                length(quad[3] - quad[2]) + length(quad[0] - quad[3]));
    if (side < config_.minimum_side_samples_ * config_.stride_px_ ||
        std::any_of(tags_.begin(), tags_.end(), [&](const Detection& t) {
            return length(center(t.corners_px_) - middle) < 0.15F * side;
        }))
        return;
    for (const float margin : config_.expansion_samples_) {
        Quad trial = quad;
        for (Point& p : trial)
            p = middle + (p - middle) * (1 + margin * config_.stride_px_ / side);
        auto tag = decode(gray, trial, config_.minimum_ring_matches_, config_.minimum_contrast_, 3);
        if (tag.id < 0)
            continue;
        Quad refined = trial;
        refine_quad(gray, refined, false);
        if (auto errors = verify_payload(gray, refined, tag.id, config_.minimum_ring_matches_,
                                         config_.minimum_contrast_, tag.distance, tag.rotation)) {
            tag.distance = *errors;
            trial = refined;
        }
        append(trial, tag);
        return;
    }
}

void LargeBoardSearch::extract(const cv::Mat& gray) {
    const int stride = config_.stride_px_;
    const int offset = stride / 2;
    sampled_.create((gray.rows + stride - 1) / stride, (gray.cols + stride - 1) / stride, CV_8U);
    for (int y = 0; y < sampled_.rows; ++y) {
        const auto* source = gray.ptr<std::uint8_t>(std::min(y * stride + offset, gray.rows - 1));
        auto* row = sampled_.ptr<std::uint8_t>(y);
        for (int x = 0; x < sampled_.cols; ++x)
            row[x] = source[std::min(stride * x + offset, gray.cols - 1)];
    }
    cv::adaptiveThreshold(sampled_, mask_, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV,
                          config_.adaptive_window_, config_.adaptive_offset_);
    cv::erode(mask_, mask_, {});
    cv::findContours(mask_, contours_, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    for (const auto& contour : contours_) {
        if (contour.size() < 4)
            continue;
        const auto bounds = cv::boundingRect(contour);
        if (std::min(bounds.width, bounds.height) < config_.minimum_side_samples_ ||
            cv::contourArea(contour) < config_.minimum_area_samples_)
            continue;
        cv::approxPolyDP(contour, polygon_,
                         config_.polygon_tolerance_ratio_ * cv::arcLength(contour, true), true);
        if (polygon_.size() != 4 || !cv::isContourConvex(polygon_))
            continue;
        if (cv::contourArea(polygon_, true) < 0)
            std::reverse(polygon_.begin(), polygon_.end());
        Quad quad;
        for (int i = 0; i < 4; ++i) {
            const auto& p = polygon_[i];
            quad[i] = {static_cast<float>(std::min(stride * p.x + offset, gray.cols - 1)),
                       static_cast<float>(std::min(stride * p.y + offset, gray.rows - 1))};
        }
        accept_contour(gray, quad);
    }
}

void LargeBoardSearch::try_neighbor(const cv::Mat& gray, const Quad& quad, Point middle) {
    auto result = decode(gray, quad, 23, 12, 2);
    if (result.id < 0 && (!result.supported || result.distance > 5))
        return;
    Quad refined = quad;
    refine_quad(gray, refined, false);
    const auto checked = decode(gray, refined, 23, 12, 2);
    if (checked.id >= 0 &&
        (result.id < 0 || (checked.id == result.id && checked.rotation == result.rotation &&
                           checked.distance <= result.distance))) {
        append(refined, checked);
    } else if (result.id >= 0) {
        append(quad, result);
    } else {
        return;
    }
    centers_.push_back(middle);
}

void LargeBoardSearch::supplement(const cv::Mat& gray) {
    if (tags_.size() < 2 || tags_.size() > 512)
        return;
    seeds_.clear();
    centers_.clear();
    all_pitch_.clear();
    for (const auto& t : tags_) {
        Projection p;
        if (project(t.corners_px_, p)) {
            const auto& q = t.corners_px_;
            const float side = 0.5F * (length(q[1] - q[0]) + length(q[2] - q[1]));
            const Point mid = project_point(p, {0.5F, 0.5F});
            seeds_.push_back({p, mid, side});
            centers_.push_back(mid);
        }
    }
    for (auto& a : seeds_) {
        local_pitch_.clear();
        for (const auto& b : seeds_) {
            if (&a == &b || b.side < 0.55F * a.side || b.side > 1.7F * a.side ||
                length(a.center - b.center) > 2.7F * a.side)
                continue;
            const float pitch = pitch_to(a.projection, b.projection, b.center);
            if (pitch > 0) {
                local_pitch_.push_back(pitch);
                all_pitch_.push_back(pitch);
            }
        }
        if (!local_pitch_.empty())
            a.pitch = median(local_pitch_);
    }
    if (all_pitch_.size() < 2)
        return;
    const float global = median(all_pitch_);
    int trials = 0;
    for (const auto& a : seeds_) {
        const float step = a.pitch > 0 ? a.pitch : global;
        for (const Point direction : {Point(-1, 0), Point(1, 0), Point(0, -1), Point(0, 1)}) {
            const Point move = step * direction;
            const auto at = [&](float x, float y) {
                return project_point(a.projection, move + Point(x, y));
            };
            const Point mid = at(0.5F, 0.5F);
            const auto in_image = [&](Point p) {
                return p.x >= 0 && p.x < gray.cols - 1 && p.y >= 0 && p.y < gray.rows - 1;
            };
            if (!in_image(mid) || std::any_of(centers_.begin(), centers_.end(), [&](Point p) {
                    return length(p - mid) < 0.30F * a.side;
                }))
                continue;
            if (++trials > config_.maximum_neighbor_trials_)
                return;
            const Quad trial{at(0, 0), at(1, 0), at(1, 1), at(0, 1)};
            if (std::all_of(trial.begin(), trial.end(), in_image))
                try_neighbor(gray, trial, mid);
        }
    }
}

void LargeBoardSearch::recover_oblique(const cv::Mat& gray, std::vector<Detection>& tags) {
    const auto& config = this->config_.oblique_;
    if (tags.size() < config.minimum_seeds_ || tags.size() > config.maximum_seeds_ ||
        config.maximum_area_ratio_ == 0)
        return;
    const auto thin = [&](const Detection& tag) { return is_oblique(tag.corners_px_, config); };
    if (static_cast<std::size_t>(std::count_if(tags.begin(), tags.end(), thin)) <
        config.minimum_seeds_)
        return;

    this->seeds_.clear();
    this->centers_.clear();
    for (const auto& tag : tags) {
        Projection p;
        if (!project(tag.corners_px_, p))
            continue;
        const Point middle = project_point(p, {.5F, .5F});
        this->centers_.push_back(middle);
        if (thin(tag))
            this->seeds_.push_back({p, middle, 0});
    }
    int trials = 0;
    const std::array<Point, 4> directions{{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}};
    const std::array<Point, 4> vertices{{{0, 0}, {1, 0}, {1, 1}, {0, 1}}};
    for (const auto& seed : this->seeds_) {
        std::array<float, 4> pitches{};
        for (const auto& neighbor : this->seeds_) {
            if (&seed == &neighbor)
                continue;
            const Point local = inverse_point(seed.projection, neighbor.center);
            const Point delta = local - Point(.5F, .5F);
            const int axis = std::abs(delta.x) > std::abs(delta.y) ? 0 : 2;
            const int side = axis + int((axis == 0 ? delta.x : delta.y) > 0);
            const float pitch = delta.dot(directions[side]);
            if (pitch < config.minimum_pitch_ || pitch > config.maximum_pitch_ ||
                std::abs(cross(delta, directions[side])) > config.maximum_shape_error_ ||
                !same_square(seed.projection, neighbor.projection, local,
                             config.maximum_shape_error_))
                continue;
            // Two different prints claiming the same adjacent lattice site are ambiguous.
            pitches[side] = pitches[side] == 0 ? pitch : -1.F;
        }
        for (int side = 0; side < 4; ++side) {
            const float pitch = pitches[side ^ 1];
            // Reflect the opposite neighbor through the seed only when an orthogonal
            // neighbor also agrees with the seed's pose and projected scale.
            if (pitches[side] != 0 || pitch <= 0 ||
                (pitches[side ^ 2] <= 0 && pitches[side ^ 3] <= 0))
                continue;
            const Point shift = pitch * directions[side];
            const Point local_center = shift + Point(.5F, .5F);
            if (std::any_of(this->centers_.begin(), this->centers_.end(), [&](Point center_px) {
                    const Point delta = inverse_point(seed.projection, center_px) - local_center;
                    return std::abs(delta.x) < .5F && std::abs(delta.y) < .5F;
                }))
                continue;
            Quad quad;
            std::transform(vertices.begin(), vertices.end(), quad.begin(),
                           [&](Point uv) { return project_point(seed.projection, shift + uv); });
            if (!std::all_of(quad.begin(), quad.end(), [&](Point p) {
                    return p.x >= 1 && p.y >= 1 && p.x < gray.cols - 2 && p.y < gray.rows - 2;
                }))
                continue;
            if (++trials > this->config_.maximum_neighbor_trials_)
                return;
            auto match =
                    decode(gray, quad, config.minimum_ring_matches_, config.minimum_contrast_, 2);
            if (match.id < 0)
                continue;
            Quad refined = quad;
            if (refine_quad(gray, refined, true)) {
                if (auto errors = verify_payload(
                            gray, refined, match.id, config.minimum_ring_matches_,
                            config.minimum_contrast_, match.distance, match.rotation)) {
                    quad = refined;
                    match.distance = *errors;
                }
            }
            if (!supports_oblique_edges(gray, quad, config.minimum_contrast_))
                continue;
            this->centers_.push_back(project_point(seed.projection, local_center));
            std::rotate(quad.begin(), quad.begin() + (4 - match.rotation) % 4, quad.end());
            tags.push_back({match.id, quad, match.distance});
        }
    }
}

const std::vector<Detection>& LargeBoardSearch::detect(const cv::Mat& gray) {
    tags_.clear();
    extract(gray);
    supplement(gray);
    return tags_;
}
namespace {

constexpr int kSmallStride = 4;
constexpr int kTile = 8;

constexpr int kDarkBand = 1, kMidBand = 2, kBrightBand = 4;
constexpr int kAllBands = kDarkBand | kMidBand | kBrightBand;

void sample_small_mask(const cv::Mat& gray, cv::Mat& mask, const LargeBoardConfig& config) {
    mask.create((gray.rows + kSmallStride - 1) / kSmallStride,
                (gray.cols + kSmallStride - 1) / kSmallStride, CV_8U);
    for (int y = 0; y < mask.rows; ++y) {
        const auto* source = gray.ptr<std::uint8_t>(kSmallStride * y);
        auto* out = mask.ptr<std::uint8_t>(y);
        int x = 0;
#if CV_SIMD128
        for (; (x + 16) * kSmallStride <= gray.cols; x += 16) {
            cv::v_uint8x16 a, b, c, d;
            cv::v_load_deinterleave(source + x * kSmallStride, a, b, c, d);
            const auto band = [&](int threshold, int bit) {
                return cv::v_and(cv::v_le(a, cv::v_setall_u8(static_cast<std::uint8_t>(threshold))),
                                 cv::v_setall_u8(static_cast<std::uint8_t>(bit)));
            };
            const auto dark = band(config.small_dark_threshold_, kDarkBand);
            const auto middle = band(config.small_threshold_, kMidBand);
            const auto bright = band(config.small_bright_threshold_, kBrightBand);
            cv::v_store(out + x, cv::v_or(dark, cv::v_or(middle, bright)));
        }
#endif
        for (; x < mask.cols; ++x) {
            const int value = source[x * kSmallStride];
            out[x] = (value <= config.small_dark_threshold_ ? kDarkBand : 0) |
                     (value <= config.small_threshold_ ? kMidBand : 0) |
                     (value <= config.small_bright_threshold_ ? kBrightBand : 0);
        }
    }
}

bool dense_transitions(int across, int down) {
    return across >= 2 && down >= 2 && across + down >= 7;
}

bool small_tile(const cv::Mat& mask, int x, int y0, int y1, int bands = kAllBands) {
    const int left = std::max(2, x);
    const int right = std::min(mask.cols - 2, x + kTile);
    int across = 0, down = 0;
    for (int y = y0; y < y1; ++y) {
        const auto* a = mask.ptr<std::uint8_t>(y - 2);
        const auto* b = mask.ptr<std::uint8_t>(y - 1);
        const auto* c = mask.ptr<std::uint8_t>(y);
        const auto* d = mask.ptr<std::uint8_t>(y + 1);
        const auto* e = mask.ptr<std::uint8_t>(y + 2);
        for (int i = left; i < right; ++i) {
            // XOR identifies a crossing for each threshold. Both neighbors must
            // disagree with the center at the same threshold and spatial scale.
            const auto peaks = [&](int a1, int b1, int a2, int b2) {
                return (((c[i] ^ a1) & (c[i] ^ b1)) | ((c[i] ^ a2) & (c[i] ^ b2))) & bands;
            };
            across += peaks(c[i - 1], c[i + 1], c[i - 2], c[i + 2]) != 0;
            down += peaks(b[i], d[i], a[i], e[i]) != 0;
        }
        if (dense_transitions(across, down))
            return true;
    }
    return false;
}

#if CV_SIMD128
int extrema_bits(cv::v_uint8x16 mid, const std::uint8_t* left1, const std::uint8_t* right1,
                 const std::uint8_t* left2, const std::uint8_t* right2) {
    const auto a = cv::v_and(cv::v_xor(mid, cv::v_load(left1)), cv::v_xor(mid, cv::v_load(right1)));
    const auto b = cv::v_and(cv::v_xor(mid, cv::v_load(left2)), cv::v_xor(mid, cv::v_load(right2)));
    return cv::v_signmask(cv::v_ne(cv::v_or(a, b), cv::v_setall_u8(0)));
}

void tile_pair(const cv::Mat& mask, int x, int y0, int y1, std::uint8_t* tile) {
    int across0 = 0, across1 = 0, down0 = 0, down1 = 0;
    for (int y = y0; y < y1; ++y) {
        const auto* row = mask.ptr<std::uint8_t>(y) + x;
        const auto mid = cv::v_load(row);
        const int across = extrema_bits(mid, row - 1, row + 1, row - 2, row + 2);
        const int down = extrema_bits(
                mid, mask.ptr<std::uint8_t>(y - 1) + x, mask.ptr<std::uint8_t>(y + 1) + x,
                mask.ptr<std::uint8_t>(y - 2) + x, mask.ptr<std::uint8_t>(y + 2) + x);
        across0 += bit_count(static_cast<unsigned>(across & 255));
        across1 += bit_count(static_cast<unsigned>(across >> 8));
        down0 += bit_count(static_cast<unsigned>(down & 255));
        down1 += bit_count(static_cast<unsigned>(down >> 8));
        if (dense_transitions(across0, down0) && dense_transitions(across1, down1))
            break;
    }
    tile[0] = dense_transitions(across0, down0) ? 255 : 0;
    tile[1] = dense_transitions(across1, down1) ? 255 : 0;
}
#endif

// A narrow intensity range can straddle only the dark or bright threshold. Verify
// contrast on both sides of that threshold before scheduling another native search.
bool contrast_tile(const cv::Mat& gray, int tile_x, int tile_y, int threshold, int margin) {
    const int y0 = std::max(2, tile_y * kTile);
    const int y1 =
            std::min((gray.rows + kSmallStride - 1) / kSmallStride - 2, (tile_y + 1) * kTile);
    const int x0 = std::max(2, tile_x * kTile);
    const int x1 =
            std::min((gray.cols + kSmallStride - 1) / kSmallStride - 2, (tile_x + 1) * kTile);
    const auto peak = [&](int value, int a, int b) {
        return (value < threshold - margin && std::min(a, b) > threshold + margin) ||
               (value > threshold + margin && std::max(a, b) < threshold - margin);
    };
    int across = 0, down = 0;
    for (int y = y0; y < y1; ++y) {
        const auto* row = gray.ptr<std::uint8_t>(y * kSmallStride);
        for (int x = x0; x < x1; ++x) {
            const int i = x * kSmallStride;
            const int value = row[i];
            across += peak(value, row[i - kSmallStride], row[i + kSmallStride]) ||
                      peak(value, row[i - 2 * kSmallStride], row[i + 2 * kSmallStride]);
            const auto at = [&](int offset) {
                return gray.ptr<std::uint8_t>((y + offset) * kSmallStride)[i];
            };
            down += peak(value, at(-1), at(1)) || peak(value, at(-2), at(2));
        }
        if (across >= 3 && down >= 3 && across + down >= 10)
            return true;
    }
    return false;
}

}  // namespace

void LargeBoardSearch::append_small(const cv::Mat& gray) {
    sample_small_mask(gray, sampled_, config_);
    mask_.create((sampled_.rows + kTile - 1) / kTile, (sampled_.cols + kTile - 1) / kTile, CV_8U);
    for (int j = 0; j < mask_.rows; ++j) {
        const int y0 = std::max(2, j * kTile);
        const int y1 = std::min(sampled_.rows - 2, (j + 1) * kTile);
        auto* row = mask_.ptr<std::uint8_t>(j);
        row[0] = small_tile(sampled_, 0, y0, y1) ? 255 : 0;
        int i = 1;
#if CV_SIMD128
        for (; (i + 2) * kTile <= sampled_.cols - 2; i += 2)
            tile_pair(sampled_, i * kTile, y0, y1, row + i);
#endif
        for (; i < mask_.cols; ++i)
            row[i] = small_tile(sampled_, i * kTile, y0, y1) ? 255 : 0;
    }
    bands_.create(mask_.size(), CV_8U);
    for (int j = 0; j < mask_.rows; ++j) {
        auto* bands = bands_.ptr<std::uint8_t>(j);
        const auto* active = mask_.ptr<std::uint8_t>(j);
        const int y0 = std::max(2, j * kTile), y1 = std::min(sampled_.rows - 2, (j + 1) * kTile);
        for (int i = 0; i < mask_.cols; ++i) {
            bands[i] = 0;
            if (!active[i])
                continue;
            if (small_tile(sampled_, i * kTile, y0, y1, kMidBand)) {
                bands[i] = kMidBand;
            } else {
                if (contrast_tile(gray, i, j, config_.small_dark_threshold_,
                                  config_.secondary_hysteresis_))
                    bands[i] |= kDarkBand;
                if (contrast_tile(gray, i, j, config_.small_bright_threshold_,
                                  config_.secondary_hysteresis_))
                    bands[i] |= kBrightBand;
            }
        }
    }
    search_tiles(gray, kMidBand, small_);
    search_tiles(gray, kDarkBand, dark_);
    search_tiles(gray, kBrightBand, bright_);
}

void LargeBoardSearch::search_tiles(const cv::Mat& gray, int bit, Detector& detector) {
    cv::bitwise_and(bands_, cv::Scalar(bit), mask_);
    if (bit == kMidBand)
        cv::dilate(mask_, mask_, {});
    cv::findContours(mask_, contours_, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    constexpr int kTilePx = kTile * kSmallStride;
    for (const auto& contour : contours_) {
        const cv::Rect tile = cv::boundingRect(contour);
        const int margin =
                bit == kMidBand ? config_.small_roi_padding_px_ : config_.secondary_padding_px_;
        const cv::Rect roi =
                cv::Rect(tile.x * kTilePx - margin, tile.y * kTilePx - margin,
                         tile.width * kTilePx + 2 * margin, tile.height * kTilePx + 2 * margin) &
                cv::Rect({}, gray.size());
        for (auto tag : detector.detect(gray(roi))) {
            for (auto& p : tag.corners_px_)
                p += Point(roi.tl());
            const Point here = center(tag.corners_px_);
            const float diagonal = length(tag.corners_px_[0] - tag.corners_px_[2]);
            if (std::none_of(tags_.begin(), tags_.end(), [&](const Detection& existing) {
                    const auto& q = existing.corners_px_;
                    return length(here - center(q)) <
                           0.25F * std::min(diagonal, length(q[0] - q[2]));
                }))
                tags_.push_back(tag);
        }
    }
}

const std::vector<Detection>& LargeBoardSearch::detect_mixed(const cv::Mat& gray) {
    detect(gray);
    if (!tags_.empty())
        append_small(gray);
    return tags_;
}

}  // namespace aruco3cuda::apriltag_small_cpu::detail
