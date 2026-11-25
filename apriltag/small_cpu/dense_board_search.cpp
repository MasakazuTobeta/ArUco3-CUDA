// SPDX-License-Identifier: Apache-2.0
#include "dense_board_search.hpp"
#include "edge_curvature.hpp"
#include "gray_view.hpp"
#include "packed_gray_bits.hpp"
#include "planar_lattice.hpp"
#include "planar_tag_shapes.hpp"

#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/imgproc.hpp>

#if defined(__aarch64__) && defined(__AARCH64EL__)
#include <arm_neon.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace aruco3cuda::apriltag_small_cpu::detail {
namespace {

Point center(const Quad& q) {
    return 0.25F * (q[0] + q[1] + q[2] + q[3]);
}

bool has_candidate_rim(const cv::Mat& gray, const Quad& quad) {
    Projection p;
    if (!project(quad, p))
        return false;
    int positive = 0;
    float contrast = 0;
    for (int j = 0; j < 4; ++j) {
        const float t = (j + .5F) / 4;
        for (auto [outside, inside] :
             {std::pair(p.sample(gray, -.125F, t), p.sample(gray, .07F, t)),
              std::pair(p.sample(gray, 1.125F, t), p.sample(gray, .93F, t)),
              std::pair(p.sample(gray, t, -.125F), p.sample(gray, t, .07F)),
              std::pair(p.sample(gray, t, 1.125F), p.sample(gray, t, .93F))}) {
            if (outside < 0 || inside < 0)
                continue;
            positive += outside > inside + 7;
            contrast += outside - inside;
        }
    }
    return positive >= 8 && contrast >= 128;
}

struct EdgeLine {
    Point point;
    Point direction;
};

constexpr float kMinimumEdgeLengthPx = 6.5F;
constexpr int kMinimumEdgeSupport = 3;

// Find the half-intensity crossing between two adjacent *native* pixel centers.
// Scanning along the major axis of the line keeps each measurement on original pixels.
template <bool Curved>
bool fit_edge(const cv::Mat& image, Point a, Point b, float dark, float white, EdgeLine& line,
              int reach, EdgeCurvatureEvidence* evidence) {
    const bool transpose = std::abs(b.y - a.y) > std::abs(b.x - a.x);
    const auto axes = [transpose](Point p) { return transpose ? Point(p.y, p.x) : p; };
    a = axes(a);
    b = axes(b);
    const Point delta = b - a;
    if (length(delta) < kMinimumEdgeLengthPx)
        return false;
    const float exterior = std::copysign(1.F, transpose ? delta.x : -delta.x);
    const float range = white - dark;
    const float midpoint = .5F * (white + dark);
    const float inset = std::min(2.F, .16F * std::abs(delta.x));
    const int begin = cvCeil(std::min(a.x, b.x) + inset);
    const int end = cvFloor(std::max(a.x, b.x) - inset);
    const int stride = std::max(1, (end - begin) / 30);
    float weight = 0, time = 0, time2 = 0, offset = 0, time_offset = 0;
    float bend_sum = 0, bend_squared = 0, time_bend = 0, offset_bend = 0, offset_squared = 0;
    // A Gray8 edge crosses the midpoint exactly when its integer endpoints
    // bracket the rounded-down and rounded-up midpoint.
    const int minimum_step = cvCeil(.07F * range);
    const int below_midpoint = cvFloor(midpoint), above_midpoint = cvCeil(midpoint);
    const auto line_step = transpose ? std::size_t(1) : image.step[0];
    const auto row_step = transpose ? image.step[0] : std::size_t(1);
    const int major_limit = transpose ? image.rows : image.cols,
              minor_limit = transpose ? image.cols : image.rows;
    for (int major = begin; major <= end; major += stride) {
        const float t = (major - a.x) / delta.x;
        const float expected = a.y + t * delta.y;
        const int floor = cvFloor(expected);
        float best = 0, crossing = 0;
        const int lower = std::max(0, floor - reach),
                  upper = std::min(minor_limit - 2, floor + reach);
        // Preserve the accumulation at positions whose search interval is empty.
        if (major >= 0 && major < major_limit && lower <= upper) {
            const auto* pixel =
                    image.data + std::size_t(major) * row_step + std::size_t(lower) * line_step;
            int preceding = *pixel;
            for (int minor = lower; minor <= upper; ++minor) {
                pixel += line_step;
                const int first = preceding, second = *pixel;
                preceding = second;
                if (exterior > 0 ? (first > below_midpoint || second < above_midpoint ||
                                    second - first < minimum_step)
                                 : (second > below_midpoint || first < above_midpoint ||
                                    first - second < minimum_step))
                    continue;
                const float step = (second - first) * exterior;
                const float position = minor + (midpoint - first) / (second - first) - expected;
                const float strength = step * (1.F - .30F * std::abs(position) / reach);
                if (strength > best) {
                    best = strength;
                    crossing = position;
                }
            }
        }
        const float w = std::min(1.F, best / (.37F * range));
        weight += w;
        time += w * t;
        time2 += w * t * t;
        offset += w * crossing;
        time_offset += w * t * crossing;
        if constexpr (Curved) {
            const float bend = t * (1 - t);
            bend_sum += w * bend;
            bend_squared += w * bend * bend;
            time_bend += w * t * bend;
            offset_bend += w * crossing * bend;
            offset_squared += w * crossing * crossing;
        }
    }
    const float determinant = weight * time2 - time * time;
    if (weight < kMinimumEdgeSupport || determinant < .25F)
        return false;
    float slope = (weight * time_offset - time * offset) / determinant;
    float intercept = (offset - slope * time) / weight;
    if constexpr (Curved) {
        // The quadratic basis t*(1-t) vanishes at both endpoints. Orthogonalize
        // it against the straight fit before using native crossings to estimate sag.
        const float beta = (weight * time_bend - time * bend_sum) / determinant;
        const float alpha = (bend_sum - beta * time) / weight;
        const float spread = bend_squared - alpha * bend_sum - beta * time_bend;
        if (weight >= 8 && spread > .0006F * weight) {
            float bend = (offset_bend - alpha * offset - beta * time_offset) / spread;
            const float rss = std::max(.0001F, offset_squared - intercept * offset -
                                                       slope * time_offset - bend * bend * spread);
            const float variance = rss / (weight - 3);
            const float span = float(std::max(image.cols, image.rows));
            const float squared_length = delta.dot(delta);
            if (evidence && squared_length >= 14 * 14 && weight >= 10 && std::abs(bend) < 5) {
                Point normal = axes({-delta.y / delta.x, 1});
                normal *= 1.F / length(normal);
                const Point midpoint =
                        axes(.5F * (a + b)) - Point(.5F * image.cols, .5F * image.rows);
                const float sag = .25F * bend * normal.dot(axes({0, 1}));
                evidence->add(midpoint.dot(normal) / span, normal, sag * span / squared_length,
                              std::min(4.F, squared_length / 900));
            }
            // Suppress noisy fits on short edges and limit extrapolation from a clipped rim.
            const float gain = std::max(0.F, 1 - 3 * variance / (1e-7F + bend * bend * spread));
            const float limit = std::min(4.F, 1.6F * squared_length / span);
            bend = std::clamp(gain * bend, -limit, limit);
            slope -= bend * beta;
            intercept -= bend * alpha;
        }
    }
    line = {axes(a + Point(0, intercept)), axes(delta + Point(0, slope))};
    return true;
}

float interior_dark_level(const std::array<float, 64>& interior) {
    // The rim fit uses the sixth-darkest sample as its ink level.
    std::array<float, 6> lowest;
    std::copy_n(interior.begin(), lowest.size(), lowest.begin());
    std::sort(lowest.begin(), lowest.end());
    for (std::size_t i = lowest.size(); i < interior.size(); ++i) {
        const float level = interior[i];
        if (level >= lowest.back())
            continue;
        std::size_t position = lowest.size() - 1;
        while (position > 0 && level < lowest[position - 1]) {
            lowest[position] = lowest[position - 1];
            --position;
        }
        lowest[position] = level;
    }
    return lowest.back();
}

template <bool Curved = false>
bool fit_outer_border(const cv::Mat& gray, Quad& quad, int reach = 2,
                      EdgeCurvatureEvidence* evidence = nullptr) {
    // The straight fit rejects a short edge regardless of its luminance.
    // Curved fits can record evidence on earlier edges even when a later one fails.
    if constexpr (!Curved) {
        for (int j = 0; j < 4; ++j) {
            Point a = quad[j], b = quad[(j + 1) % 4];
            const bool transpose = std::abs(b.y - a.y) > std::abs(b.x - a.x);
            if (transpose) {
                std::swap(a.x, a.y);
                std::swap(b.x, b.y);
            }
            const auto delta = b - a;
            if (length(delta) < kMinimumEdgeLengthPx)
                return false;
            // Each visible native position contributes at most one unit of
            // weight. The fitted line needs at least three supported positions.
            const float inset = std::min(2.F, .16F * std::abs(delta.x));
            const int begin = std::max(0, cvCeil(std::min(a.x, b.x) + inset));
            const int end = std::min((transpose ? gray.rows : gray.cols) - 1,
                                     cvFloor(std::max(a.x, b.x) - inset));
            if (end - begin < kMinimumEdgeSupport - 1)
                return false;
        }
    }
    Projection homography;
    if (!project(quad, homography))
        return false;
    std::array<float, 64> interior;
    std::array<float, 32> exterior;
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x)
            interior[y * 8 + x] = homography.sample(gray, (x + .5F) / 8, (y + .5F) / 8);
        const float t = (y + .5F) / 8;
        exterior[4 * y] = homography.sample(gray, t, -.085F);
        exterior[4 * y + 1] = homography.sample(gray, t, 1.085F);
        exterior[4 * y + 2] = homography.sample(gray, -.085F, t);
        exterior[4 * y + 3] = homography.sample(gray, 1.085F, t);
    }
    const float dark = interior_dark_level(interior);
    std::nth_element(exterior.begin(), exterior.begin() + 17, exterior.end());
    const float white = exterior[17];
    if (dark < 0 || white - dark < 18)
        return false;
    std::array<EdgeLine, 4> lines;
    for (int j = 0; j < 4; ++j)
        if (!fit_edge<Curved>(gray, quad[j], quad[(j + 1) % 4], dark, white, lines[j], reach,
                              evidence))
            return false;
    Quad fitted;
    for (int j = 0; j < 4; ++j) {
        const auto& a = lines[(j + 3) % 4];
        const auto& b = lines[j];
        const float determinant = cross(a.direction, b.direction);
        if (std::abs(determinant) < 1e-3F)
            return false;
        fitted[j] = a.point + a.direction * (cross(b.point - a.point, b.direction) / determinant);
        if (length(fitted[j] - quad[j]) > (reach == 2 ? 1.8F : reach + 1.F))
            return false;
    }
    quad = fitted;
    return true;
}
}  // namespace

DenseBoardSearch::DenseBoardSearch(DenseBoardConfig config) : config_(config) {
    if (config.minimum_island_contrast_ < 1 || config.minimum_island_contrast_ > 255 ||
        config.minimum_tile_contrast_ < 1 || config.minimum_tile_contrast_ > 255 ||
        config.minimum_candidates_ == 0 || config.minimum_ring_matches_ < 16 ||
        config.minimum_ring_matches_ > 32 ||
        !(config.minimum_decode_contrast_ > 0 && config.minimum_decode_contrast_ <= 255) ||
        config.maximum_corrected_bits_ < 0 || config.maximum_corrected_bits_ > 2 ||
        config.maximum_component_side_px_ < 5 || config.maximum_component_side_px_ >= 32768 ||
        config.minimum_planar_tags_ < 8 ||
        !(config.minimum_normalized_bow_ >= 0 && config.minimum_normalized_bow_ <= 1) ||
        !(config.minimum_curvature_r_squared_ >= 0 && config.minimum_curvature_r_squared_ <= 1) ||
        !(config.maximum_lattice_center_shift_px_ >= 0 &&
          config.maximum_lattice_center_shift_px_ <= 2) ||
        !(config.maximum_planar_correction_px_ >= 0 && config.maximum_planar_correction_px_ <= 2))
        throw std::invalid_argument("Invalid dense AprilTag board settings");
    components_.minimum_quad_side_samples_ = 5;
    components_.maximum_quad_side_samples_ = config.maximum_component_side_px_;
    components_.maximum_aspect_ratio_ = 10;
    components_.minimum_fill_ratio_ = .28;
    components_.maximum_fill_ratio_ = 1;
    components_.minimum_quad_hull_ratio_ = .5;
}

namespace {

// 2 * pixel < low + high exactly when pixel < ceil((low + high) / 2).
void make_dense_threshold(cv::Mat& low, const cv::Mat& high, const cv::Mat& tiles,
                          int minimum_contrast) {
    for (int row = 0; row < low.rows; ++row) {
        auto* threshold = low.ptr<std::uint8_t>(row);
        const auto* upper = high.ptr<std::uint8_t>(row);
        const auto* enabled = tiles.ptr<std::uint8_t>(row / 4);
        int column = 0;
#if defined(__aarch64__) && defined(__AARCH64EL__)
        if (minimum_contrast >= 0 && minimum_contrast <= 255) {
            static constexpr std::uint8_t kExpandTile[]{0, 0, 0, 0, 1, 1, 1, 1,
                                                        2, 2, 2, 2, 3, 3, 3, 3};
            const auto pattern = vld1q_u8(kExpandTile);
            const auto margin = vdupq_n_u8(static_cast<std::uint8_t>(minimum_contrast));
            for (; column + 16 <= low.cols; column += 16) {
                std::uint32_t tile_flags;
                std::memcpy(&tile_flags, enabled + column / 4, sizeof(tile_flags));
                const auto flags =
                        vqtbl1q_u8(vreinterpretq_u8_u32(vdupq_n_u32(tile_flags)), pattern);
                const auto lo = vld1q_u8(threshold + column);
                const auto hi = vld1q_u8(upper + column);
                const auto eligible = vandq_u8(vcgtq_u8(vqsubq_u8(hi, lo), margin),
                                               vcgtq_u8(flags, vdupq_n_u8(0)));
                vst1q_u8(threshold + column, vandq_u8(vrhaddq_u8(lo, hi), eligible));
            }
        }
#endif
        for (; column < low.cols; ++column) {
            const int lo = threshold[column], hi = upper[column];
            threshold[column] =
                    enabled[column / 4] && hi - lo > minimum_contrast ? (lo + hi + 1) / 2 : 0;
        }
    }
}
#if defined(__aarch64__) && defined(__AARCH64EL__)
void pack_dense_mask(const cv::Mat& gray, const cv::Mat& thresholds,
                     std::vector<std::uint64_t>& packed) {
    const int words = (gray.cols + 63) / 64;
    packed.resize(static_cast<std::size_t>(words) * gray.rows);
    const GrayComparisonBits bits;
    for (int tile_y = 0; tile_y < thresholds.rows; ++tile_y) {
        const auto* row_levels = thresholds.ptr<std::uint8_t>(tile_y);
        const int first = 4 * tile_y;
        const int last = std::min(first + 4, gray.rows);
        int x = 0;
        for (; x + 64 <= gray.cols; x += 64) {
            const auto level = vld1q_u8(row_levels + x / 4);
            const auto low = vzip1q_u8(level, level);
            const auto high = vzip2q_u8(level, level);
            const auto a = vzip1q_u8(low, low), b = vzip2q_u8(low, low);
            const auto c = vzip1q_u8(high, high), d = vzip2q_u8(high, high);
            if (vmaxvq_u8(level) == 0) {
                for (int y = first; y < last; ++y)
                    packed[std::size_t(y) * words + x / 64] = 0;
                continue;
            }
            for (int y = first; y < last; ++y) {
                const auto* row = gray.ptr<std::uint8_t>(y) + x;
                packed[std::size_t(y) * words + x / 64] =
                        bits.pack(vcltq_u8(vld1q_u8(row), a), vcltq_u8(vld1q_u8(row + 16), b),
                                  vcltq_u8(vld1q_u8(row + 32), c), vcltq_u8(vld1q_u8(row + 48), d));
            }
        }
        if (x < gray.cols) {
            const int count = gray.cols - x;
            for (int y = first; y < last; ++y) {
                const auto* row = gray.ptr<std::uint8_t>(y) + x;
                std::uint64_t tail = 0;
                for (int i = 0; i < count; ++i)
                    tail |= std::uint64_t{row[i] < row_levels[(x + i) / 4]} << i;
                packed[std::size_t(y) * words + x / 64] = tail;
            }
        }
    }
}
#endif

}  // namespace

void dense_midrange_mask(const cv::Mat& gray, cv::Mat& low, const cv::Mat& high,
                         const cv::Mat& tiles, int minimum_contrast, cv::Mat& mask) {
    make_dense_threshold(low, high, tiles, minimum_contrast);
    mask.create(gray.size(), CV_8U);
    for (int row = 0; row < low.rows; ++row) {
        const auto* threshold = low.ptr<std::uint8_t>(row);
        const int first_row = 4 * row;
        const int end_row = std::min(first_row + 4, gray.rows);
        int x = 0;
#if CV_SIMD128
        for (; x + 64 <= gray.cols; x += 64) {
            const cv::v_uint8x16 values = cv::v_load(threshold + x / 4);
            cv::v_uint8x16 lower, upper_half, a, b, c, d;
            // Duplicate twice along columns, then reuse these 64 thresholds for up to four rows.
            cv::v_zip(values, values, lower, upper_half);
            cv::v_zip(lower, lower, a, b);
            cv::v_zip(upper_half, upper_half, c, d);
            for (int y = first_row; y < end_row; ++y) {
                const auto* input = gray.ptr<std::uint8_t>(y) + x;
                auto* output = mask.ptr<std::uint8_t>(y) + x;
                cv::v_store(output, cv::v_lt(cv::v_load(input), a));
                cv::v_store(output + 16, cv::v_lt(cv::v_load(input + 16), b));
                cv::v_store(output + 32, cv::v_lt(cv::v_load(input + 32), c));
                cv::v_store(output + 48, cv::v_lt(cv::v_load(input + 48), d));
            }
        }
#endif
        for (int y = first_row; y < end_row; ++y) {
            const auto* input = gray.ptr<std::uint8_t>(y);
            auto* output = mask.ptr<std::uint8_t>(y);
            for (int j = x; j < gray.cols; ++j)
                output[j] = input[j] < threshold[j / 4] ? 255 : 0;
        }
    }
}

void DenseBoardSearch::make_mask(const cv::Mat& gray) {
    cv::resize(gray, sampled_, {(gray.cols + 3) / 4, (gray.rows + 3) / 4}, 0, 0, cv::INTER_NEAREST);
    const auto window = cv::getStructuringElement(cv::MORPH_RECT, {9, 9});
    cv::dilate(sampled_, high_, window);
    cv::erode(sampled_, low_, window);
    tiles_.create((gray.rows + 15) / 16, (gray.cols + 15) / 16, CV_8U);
    tiles_.setTo(0);
    // A two-sided intensity change is evidence for print or payload on the sheet.
    // Dilate at tile resolution so the candidate's outer ring is retained.
    for (int y = 2; y < sampled_.rows - 2; y += 2) {
        const auto* row = sampled_.ptr<std::uint8_t>(y);
        const auto* up = sampled_.ptr<std::uint8_t>(y - 2);
        const auto* down = sampled_.ptr<std::uint8_t>(y + 2);
        auto* keep = tiles_.ptr<std::uint8_t>(y / 4);
        const auto check_sample = [&](int x) {
            const int v = row[x];
            const int across = std::max(std::min(row[x - 2], row[x + 2]) - v,
                                        v - std::max(row[x - 2], row[x + 2]));
            const int along = std::max(std::min(up[x], down[x]) - v, v - std::max(up[x], down[x]));
            if (std::max(across, along) > config_.minimum_island_contrast_)
                keep[x / 4] = 255;
        };
        int x = 2;
#if defined(__aarch64__) && defined(__AARCH64EL__)
        if (x < sampled_.cols - 2) {
            check_sample(x);
            x += 2;
        }
        const auto limit = vdupq_n_u8(static_cast<std::uint8_t>(config_.minimum_island_contrast_));
        for (; x + 34 <= sampled_.cols; x += 32) {
            const auto center = vld2q_u8(row + x).val[0];
            const auto left = vld2q_u8(row + x - 2).val[0];
            const auto right = vld2q_u8(row + x + 2).val[0];
            const auto above = vld2q_u8(up + x).val[0];
            const auto below = vld2q_u8(down + x).val[0];
            const auto across = vmaxq_u8(vqsubq_u8(vminq_u8(left, right), center),
                                         vqsubq_u8(center, vmaxq_u8(left, right)));
            const auto along = vmaxq_u8(vqsubq_u8(vminq_u8(above, below), center),
                                        vqsubq_u8(center, vmaxq_u8(above, below)));
            const auto marks = vcgtq_u8(vmaxq_u8(across, along), limit);
            auto* output = keep + x / 4;
            const auto pairs = vget_low_u8(vpmaxq_u8(marks, marks));
            vst1_u8(output, vorr_u8(vld1_u8(output), pairs));
        }
#endif
        for (; x < sampled_.cols - 2; x += 2)
            check_sample(x);
    }
    cv::dilate(tiles_, tiles_, cv::getStructuringElement(cv::MORPH_RECT, {9, 9}));
#if defined(__aarch64__) && defined(__AARCH64EL__)
    make_dense_threshold(low_, high_, tiles_, config_.minimum_tile_contrast_);
    pack_dense_mask(gray, low_, packed_mask_);
#else
    dense_midrange_mask(gray, low_, high_, tiles_, config_.minimum_tile_contrast_, mask_);
#endif
}

Decode DenseBoardSearch::read_soft(const cv::Mat& gray, const Quad& quad) const {
    Projection p;
    if (!project(quad, p))
        return {};
    std::array<std::array<float, 10>, 10> samples;
    const auto sample = [&](int x, int y) {
        const float value = p.sample(gray, (x - .5F) / 8, (y - .5F) / 8);
        samples[y][x] = value;
        return value;
    };
    const int allowed_misses = 32 - std::max(24, config_.minimum_ring_matches_);
    int missed = 0;
    float white = 0, black = 0;
    const auto ring_pair = [&](int outer_x, int outer_y, int inner_x, int inner_y) {
        const float outside = sample(outer_x, outer_y);
        if (outside < 0)
            return false;
        const float inside = sample(inner_x, inner_y);
        if (inside < 0)
            return false;
        missed += outside - inside < 9;
        white += outside;
        black += inside;
        return missed <= allowed_misses;
    };
    for (int j = 1; j <= 8; ++j) {
        if (!ring_pair(j, 0, j, 1) || !ring_pair(j, 9, j, 8) || !ring_pair(0, j, 1, j) ||
            !ring_pair(9, j, 8, j))
            return {};
    }
    white /= 32;
    black /= 32;
    if (white - black < std::max(20.F, config_.minimum_decode_contrast_))
        return {};
    // The four unused outer corners still belong to the required sampling footprint.
    for (int y : {0, 9})
        for (int x : {0, 9})
            if (sample(x, y) < 0)
                return {};
    for (int y = 2; y < 8; ++y)
        for (int x = 2; x < 8; ++x)
            if (sample(x, y) < 0)
                return {};
    Decode best;
    // Correct sensor blur in cell coordinates; competing assignments across settings disqualify the
    // quad.
    for (float sharpen : {.15F, .40F})
        for (float fraction : {.46F, .34F}) {
            MarkerCode code = 0;
            for (int y = 2; y < 8; ++y)
                for (int x = 2; x < 8; ++x) {
                    const float v = samples[y][x];
                    const float filtered =
                            v + sharpen * (4 * v - samples[y - 1][x] - samples[y + 1][x] -
                                           samples[y][x - 1] - samples[y][x + 1]);
                    code |= MarkerCode(filtered > black + fraction * (white - black))
                            << ((y - 2) * 6 + x - 2);
                }
            const auto [word, distance] = CodeIndex::match(code);
            if (word < 0 || distance > config_.maximum_corrected_bits_)
                continue;
            if (best.id >= 0 && word != 4 * best.id + best.rotation)
                return {};
            if (distance < best.distance)
                best = {word / 4, word % 4, distance, true};
        }
    return best;
}

Decode DenseBoardSearch::read_candidate(const cv::Mat& gray, Quad& quad, float side,
                                        bool verify_seeds) const {
    const Quad initial = quad;
    if (!refine_quad(gray, quad, true)) {
        const auto mid = center(quad);
        for (auto& p : quad)
            p = mid + (p - mid) * (1.F + 1.F / side);
    }
    auto match = decode(gray, quad, config_.minimum_ring_matches_, config_.minimum_decode_contrast_,
                        config_.maximum_corrected_bits_);
    if (match.id < 0)
        match = read_soft(gray, quad);
    if (match.id >= 0 || side >= 100)
        return match;
    auto outer = quad;
    if (fit_outer_border(gray, outer)) {
        match = read_soft(gray, outer);
        if (match.id >= 0) {
            quad = outer;
            return match;
        }
    }
    // At sub-two-pixel cells the signed-gradient fit can follow interior ink.
    // Regularize the original silhouette to a local affine plane before refitting.
    if (side < 18) {
        const auto mid = center(initial);
        const auto perspective = .25F * (initial[0] + initial[2] - initial[1] - initial[3]);
        auto affine = initial;
        for (int j = 0; j < 4; ++j) {
            const Point point = initial[j] + (j % 2 ? perspective : -perspective);
            affine[j] = mid + (point - mid) * (1.F + 1.F / side);
        }
        auto polished = affine;
        if (fit_outer_border(gray, polished) && polished != affine) {
            match = read_soft(gray, polished);
            if (match.id >= 0) {
                quad = polished;
                return match;
            }
        }
        match = read_soft(gray, affine);
        if (match.id >= 0) {
            quad = affine;
            return match;
        }
    }
    if (verify_seeds)
        return {};
    // One truncated hull vertex can tilt a complete side on a subpixel payload.
    // Recover the four possible local affine completions and require an unambiguous code.
    const float discrepancy = length(initial[0] + initial[2] - initial[1] - initial[3]);
    if (side < 20 && discrepancy > 1 && discrepancy < .75F * side) {
        Decode best;
        Quad best_quad;
        for (int j = 0; j < 4; ++j) {
            auto completed = initial;
            completed[j] = initial[(j + 1) % 4] + initial[(j + 3) % 4] - initial[(j + 2) % 4];
            const auto mid = center(completed);
            for (auto& point : completed)
                point = mid + (point - mid) * (1.F + 1.F / side);
            fit_outer_border(gray, completed);
            const auto verified = read_soft(gray, completed);
            if (verified.id < 0)
                continue;
            if (best.id >= 0 && (verified.id != best.id || verified.rotation != best.rotation))
                return {};
            if (verified.distance < best.distance) {
                best = verified;
                best_quad = completed;
            }
        }
        if (best.id >= 0) {
            quad = best_quad;
            return best;
        }
    }
    float min_side = std::numeric_limits<float>::max(), max_side = 0;
    for (int j = 0; j < 4; ++j) {
        float l = length(initial[(j + 1) % 4] - initial[j]);
        min_side = std::min(min_side, l);
        max_side = std::max(max_side, l);
    }
    if (min_side < 42 && max_side > 1.75F * min_side) {
        auto stretched = initial;
        if (fit_outer_border(gray, stretched, std::clamp(cvCeil(.55F * min_side), 3, 9))) {
            match = read_soft(gray, stretched);
            if (match.id >= 0) {
                quad = stretched;
                return match;
            }
        }
    }
    auto adjusted = initial;
    if (refine_quad(gray, adjusted, false)) {
        match = read_soft(gray, adjusted);
        if (match.id >= 0) {
            quad = adjusted;
            return match;
        }
    }
    Quad best_quad;
    Decode best;
    int votes = 0;
    constexpr std::array<std::array<float, 2>, 8> offsets = {{{-.32F, 0},
                                                              {.32F, 0},
                                                              {0, -.32F},
                                                              {0, .32F},
                                                              {-.27F, -.27F},
                                                              {.27F, .27F},
                                                              {-.27F, .27F},
                                                              {.27F, -.27F}}};
    for (std::size_t i = 0; i < offsets.size(); ++i) {
        // One remaining read cannot reach the two-vote acceptance threshold.
        if (votes == 0 && i + 1 == offsets.size())
            return {};
        auto moved = quad;
        for (auto& p : moved)
            p += Point(offsets[i][0], offsets[i][1]);
        const auto vote = read_soft(gray, moved);
        if (vote.id < 0)
            continue;
        if (votes && (best.id != vote.id || best.rotation != vote.rotation))
            return {};
        ++votes;
        if (vote.distance < best.distance) {
            best = vote;
            best_quad = moved;
        }
    }
    if (votes < 2)
        return {};
    quad = best_quad;
    return best;
}

const std::vector<Detection>& DenseBoardSearch::detect(const cv::Mat& gray, bool verify_seeds) {
    validate_image(gray);
    detections_.clear();
    if (gray.cols < 8 || gray.rows < 8)
        return detections_;
    make_mask(gray);
#if defined(__aarch64__) && defined(__AARCH64EL__)
    search_.extract_packed(gray, packed_mask_, components_, polygons_, &supplemental_);
#else
    search_.extract(mask_, components_, polygons_, &supplemental_);
#endif
    const std::size_t ordinary = polygons_.size();
    for (auto& quad : supplemental_.polygons_)
        polygons_.push_back(std::move(quad));
    if (polygons_.size() < config_.minimum_candidates_)
        return detections_;
    for (std::size_t i = 0; i < polygons_.size(); ++i) {
        const auto& polygon = polygons_[i];
        const float area = static_cast<float>(cv::contourArea(polygon));
        if (area < 38)
            continue;
        Quad quad;
        for (int j = 0; j < 4; ++j)
            quad[j] = polygon[j];
        if (cross(quad[1] - quad[0], quad[2] - quad[1]) < 0)
            std::swap(quad[1], quad[3]);
        float shortest = std::numeric_limits<float>::max(), longest = 0;
        for (int j = 0; j < 4; ++j) {
            const float side = length(quad[(j + 1) % 4] - quad[j]);
            shortest = std::min(shortest, side);
            longest = std::max(longest, side);
        }
        if (shortest < 4.8F || longest > config_.maximum_component_side_px_ ||
            shortest * 12 < longest)
            continue;
        const float side = std::sqrt(area);
        if ((verify_seeds || i >= ordinary) && !has_candidate_rim(gray, quad))
            continue;
        auto match = read_candidate(gray, quad, side, verify_seeds);
        if (match.id < 0)
            continue;
        auto fitted = quad;
        if (fit_outer_border(gray, fitted)) {
            auto verified =
                    decode(gray, fitted, config_.minimum_ring_matches_,
                           config_.minimum_decode_contrast_, config_.maximum_corrected_bits_);
            if (verified.id < 0)
                verified = read_soft(gray, fitted);
            if (verified.id == match.id && verified.rotation == match.rotation) {
                quad = fitted;
                match = verified;
            }
        }
        const auto mid = center(quad);
        if (std::any_of(detections_.begin(), detections_.end(), [&](const auto& t) {
                return length(center(t.corners_px_) - mid) < .36F * side;
            }))
            continue;
        Quad canonical;
        for (int j = 0; j < 4; ++j)
            canonical[j] = quad[(j + 4 - match.rotation) % 4];
        detections_.push_back({match.id, canonical, match.distance});
    }
    if (detections_.size() < config_.minimum_planar_tags_ ||
        (config_.maximum_planar_correction_px_ == 0 &&
         config_.maximum_lattice_center_shift_px_ == 0))
        return detections_;
    independently_decoded_ = detections_;
    refine_planar_tag_shapes(detections_, config_.minimum_planar_tags_,
                             config_.maximum_planar_correction_px_);
    const bool regular = refine_regular_tag_centers(detections_, config_.minimum_planar_tags_,
                                                    config_.maximum_lattice_center_shift_px_);
    if (!regular && config_.curvature_probe_tags_ > 0) {
        EdgeCurvatureEvidence evidence;
        const auto stride = std::max<std::size_t>(
                1, independently_decoded_.size() / config_.curvature_probe_tags_);
        for (std::size_t i = 0; i < independently_decoded_.size(); i += stride) {
            auto quad = independently_decoded_[i].corners_px_;
            fit_outer_border<true>(gray, quad, 2, &evidence);
        }
        if (evidence.coherent(config_.minimum_normalized_bow_,
                              config_.minimum_curvature_r_squared_)) {
            detections_ = independently_decoded_;
            for (auto& tag : detections_)
                fit_outer_border<true>(gray, tag.corners_px_);
        }
    }
    // Neighbour agreement can still move a sampling point across an ambiguous cell.
    // Keep the independent measurement unless the adjusted canonical grid reads identically.
    for (std::size_t i = 0; i < detections_.size(); ++i) {
        auto& tag = detections_[i];
        const auto& original = independently_decoded_[i];
        if (tag.corners_px_ == original.corners_px_)
            continue;
        auto verified = decode(gray, tag.corners_px_, config_.minimum_ring_matches_,
                               config_.minimum_decode_contrast_, config_.maximum_corrected_bits_);
        if (verified.id < 0)
            verified = read_soft(gray, tag.corners_px_);
        if (verified.id == tag.id_ && verified.rotation == 0)
            tag.corrected_bits_ = verified.distance;
        else
            tag = original;
    }
    return detections_;
}
}  // namespace aruco3cuda::apriltag_small_cpu::detail
