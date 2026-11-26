// SPDX-License-Identifier: Apache-2.0
#include "isolated_ink.hpp"
#include "noisy_ink.hpp"

#include "faint_guide.hpp"
#include "grid_alignment.hpp"
#include "guide_average.hpp"
#include "guide_contours.hpp"
#include "local_contour_proof.hpp"
#include "tag_codebook.hpp"

#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/imgproc.hpp>

#include <cstring>

#if CV_SIMD128 && defined(__aarch64__) && defined(__AARCH64EL__) && defined(__linux__)
#include <arm_neon.h>
#include <asm/hwcap.h>
#include <sys/auxv.h>
#if __has_include(<arm_sve.h>)
#include <arm_sve.h>
#define ARUCO3_FAINT_SVE_AVAILABLE 1
#endif
#endif

namespace aruco3cuda::apriltag_small_cpu::detail {
namespace {
// For integer reductions, sample at the center of each source block and generate
// the two thumbnail planes in one pass over the source. Only the hint uses the thumbnail.
template <int stride>
void sample_integer(const cv::Mat& gray, int cutoff, cv::Mat& guide, cv::Mat& mask) {
    const cv::Size size(gray.cols / stride, gray.rows / stride);
    guide.create(size, CV_8U);
    mask.create(size, CV_8U);
    for (int y = 0; y < size.height; ++y) {
        const auto* row = gray.ptr<std::uint8_t>(stride * y + stride / 2);
        auto* out = guide.ptr<std::uint8_t>(y);
        auto* dark = mask.ptr<std::uint8_t>(y);
        int x = 0;
#if CV_SIMD128
        if constexpr (stride == 4) {
            const auto limit = cv::v_setall_u8(static_cast<unsigned char>(cutoff));
            for (; x + 16 <= size.width; x += 16) {
                const auto values = load_nearest4<2>(row + 4 * x);
                cv::v_store(out + x, values);
                cv::v_store(dark + x, cv::v_le(values, limit));
            }
        }
#endif
        for (; x < size.width; ++x) {
            out[x] = row[stride * x + stride / 2];
            dark[x] = out[x] <= cutoff ? 255 : 0;
        }
    }
}

// Luma changes along both native axes indicate that one-pixel sampling can
// amplify read noise or fine texture. Straight rails are sparse on this grid.
bool noisy_guide(const cv::Mat& gray, int delta, int per_mille) {
    const int dx = std::max(29, gray.cols / 50);
    const int dy = std::max(23, gray.rows / 44);
    const int x_end = gray.cols - 2, y_end = gray.rows - 2;
    const int columns = 1 + (x_end - 1 - 3) / dx;
    const int rows = 1 + (y_end - 1 - 3) / dy;
    const int required = (columns * rows * per_mille + 999) / 1000;
    int votes = 0;
    for (int y = 3; y < y_end; y += dy) {
        const auto* row = gray.ptr<std::uint8_t>(y);
        const auto* above = gray.ptr<std::uint8_t>(y - 1);
        const auto* below = gray.ptr<std::uint8_t>(y + 1);
        for (int x = 3; x < x_end; x += dx) {
            const int twice = 2 * row[x];
            if (std::abs(row[x - 1] + row[x + 1] - twice) >= delta &&
                std::abs(above[x] + below[x] - twice) >= delta && ++votes >= required)
                return true;
        }
    }
    return false;
}

cv::Rect expand(cv::Rect bounds, int pad, cv::Size size) {
    return (bounds + cv::Size(2 * pad, 2 * pad) - cv::Point(pad, pad)) & cv::Rect({}, size);
}

bool refine(const cv::Mat& gray, Detection& tag) {
    if (try_align_curved_grid(gray, tag))
        return true;
    if (align_decoded_grid(gray, tag)) {
        try_align_curved_grid(gray, tag);
        return true;
    }
    return verify_payload(gray, tag.corners_px_, tag.id_, 20, 12, Config{}.maximum_corrected_bits_)
            .has_value();
}

struct FaintLine {
    Point anchor, direction;
};

#ifdef ARUCO3_FAINT_SVE_AVAILABLE
__attribute__((target("arch=armv8.2-a+sve"))) std::pair<int, float> score_strided_line(
        const std::uint8_t* column, std::uint32_t pitch, int center, int radius, int spread,
        int sign, float expected, int rows) {
    const int lo = center - radius, end = center + radius + 1;
    const auto target = svdup_n_f32(expected);
    const auto den = svdup_n_f32(static_cast<float>(radius + 1));
    svfloat32_t scores = svdup_n_f32(0.F);
    svint32_t selected = svdup_n_s32(rows);
    for (int i = lo; i < end; i += svcntw()) {
        const auto lanes = svwhilelt_b32(i, end);
        const auto indices = svindex_s32(i, 1);
        const auto a = svmul_n_u32_x(
                lanes, svreinterpret_u32_s32(svsub_n_s32_x(lanes, indices, sign * spread)), pitch);
        const auto b = svmul_n_u32_x(
                lanes, svreinterpret_u32_s32(svadd_n_s32_x(lanes, indices, sign * spread)), pitch);
        const auto low = svld1ub_gather_u32offset_u32(lanes, column, a);
        const auto high = svld1ub_gather_u32offset_u32(lanes, column, b);
        const auto contrast =
                svcvt_f32_s32_x(lanes, svreinterpret_s32_u32(svsub_u32_x(lanes, low, high)));
        const auto distance =
                svabs_f32_x(lanes, svsub_f32_x(lanes, svcvt_f32_s32_x(lanes, indices), target));
        const auto damping = svdiv_f32_x(lanes, svmul_n_f32_x(lanes, distance, .20F), den);
        const auto weight =
                svmul_f32_x(lanes, contrast, svsub_f32_x(lanes, svdup_n_f32(1.F), damping));
        const auto better = svcmpgt_f32(lanes, weight, scores);
        scores = svsel_f32(better, weight, scores);
        selected = svsel_s32(better, indices, selected);
    }
    const auto all = svptrue_b32();
    const float best = svmaxv_f32(all, scores);
    return {best > 0 ? svminv_s32(svcmpeq_n_f32(all, scores, best), selected) : center, best};
}
#endif

bool faint_line(const cv::Mat& gray, Point a, Point b, int radius, int minimum_contrast,
                FaintLine& line, int spread) {
    const bool horizontal = std::abs(b.x - a.x) >= std::abs(b.y - a.y);
    const auto transpose = [&](Point p) { return horizontal ? p : Point(p.y, p.x); };
    a = transpose(a);
    b = transpose(b);
    const Point delta = b - a;
    if (std::abs(delta.x) < 14)
        return false;
    const int sign = ((delta.x > 0) == horizontal) ? 1 : -1;
    const int cols = horizontal ? gray.cols : gray.rows, rows = horizontal ? gray.rows : gray.cols;
    const int margin = std::max(3, cvRound(.07F * std::abs(delta.x)));
    const int first = std::max(0, cvCeil(std::min(a.x, b.x)) + margin);
    const int last = std::min(cols - 1, cvFloor(std::max(a.x, b.x)) - margin);
    const int stride = std::max(1, (last - first + 300) / 300);
    const auto major = horizontal ? 1UL : gray.step[0], minor = horizontal ? gray.step[0] : 1UL;
#ifdef ARUCO3_FAINT_SVE_AVAILABLE
    static const bool sve = (getauxval(AT_HWCAP) & HWCAP_SVE) != 0;
    const bool use_sve = horizontal && sve && radius >= 3 &&
                         gray.step[0] <= std::numeric_limits<std::uint32_t>::max() / rows;
#endif
    // Regress native half-intensity crossings along the major image axis.
    double count = 0, sum_t = 0, sum_t_squared = 0, sum_offset = 0, sum_t_offset = 0;
    for (int x = first; x <= last; x += stride) {
        const float t = (x - a.x) / delta.x, expected = a.y + t * delta.y;
        const int center = cvFloor(expected);
        if (center - radius < 5 || center + radius >= rows - 6)
            continue;
        const auto* column = gray.data + x * major;
        auto pixel = [&](int y) { return column[y * minor]; };
        int chosen = center;
        float best = 0;
        int candidate = center - radius;
#ifdef ARUCO3_FAINT_SVE_AVAILABLE
        if (use_sve) {
            const auto [found, strength] =
                    score_strided_line(column, static_cast<std::uint32_t>(minor), center, radius,
                                       spread, sign, expected, rows);
            chosen = found;
            best = strength;
            candidate = center + radius + 1;
        }
#endif
#if CV_SIMD128 && defined(__aarch64__) && defined(__AARCH64EL__) && defined(__linux__)
        // These candidates lie in one native row for a vertical-dominant side.
        // Use the original FP32 arithmetic and retain the earliest maximum.
        if (!horizontal) {
            constexpr int kLanes = 4;
            const int32x4_t offsets = {0, 1, 2, 3};
            auto y = vaddq_s32(vdupq_n_s32(candidate), offsets);
            auto values = vdupq_n_f32(0.F);
            auto indices = vdupq_n_s32(center);
            const auto target = vdupq_n_f32(expected);
            const auto denominator = vdupq_n_f32(static_cast<float>(radius + 1));
            for (; candidate + (kLanes - 1) <= center + radius; candidate += kLanes) {
                const auto a = vld1_u8(column + candidate - sign * spread);
                const auto b = vld1_u8(column + candidate + sign * spread);
                const auto signed_difference = vget_low_s16(vreinterpretq_s16_u16(vsubl_u8(a, b)));
                const auto diff = vcvtq_f32_s32(vmovl_s16(signed_difference));
                const auto penalty =
                        vdivq_f32(vmulq_n_f32(vabsq_f32(vsubq_f32(vcvtq_f32_s32(y), target)), .20F),
                                  denominator);
                const auto weight = vmulq_f32(diff, vsubq_f32(vdupq_n_f32(1.F), penalty));
                const auto better = vcgtq_f32(weight, values);
                values = vbslq_f32(better, weight, values);
                indices = vbslq_s32(better, y, indices);
                y = vaddq_s32(y, vdupq_n_s32(kLanes));
            }
            best = vmaxvq_f32(values);
            if (best > 0) {
                const auto matches = vceqq_f32(values, vdupq_n_f32(best));
                chosen = vminvq_s32(vbslq_s32(matches, indices, vdupq_n_s32(rows)));
            }
        }
#endif
        for (int y = candidate; y <= center + radius; ++y) {
            const int difference = sign * (int(pixel(y - spread)) - int(pixel(y + spread)));
            const float weight = difference * (1 - .20F * std::abs(y - expected) / (radius + 1));
            if (weight > best) {
                chosen = y;
                best = weight;
            }
        }
        if (best < .55F * minimum_contrast)
            continue;
        const int y = chosen;
        const float left = (pixel(y - 4) + pixel(y - 3) + pixel(y - 2)) / 3.F;
        const float right = (pixel(y + 2) + pixel(y + 3) + pixel(y + 4)) / 3.F;
        if (sign * (left - right) < .7F * minimum_contrast)
            continue;
        float sum = 0;
        for (int j = -2; j <= 3; ++j)
            sum += pixel(y + j);
        // Integrating six native pixels estimates a fractional boundary even at low contrast.
        const float offset_sample = y - 2.5F + (sum - 6 * right) / (left - right) - expected;
        count += 1;
        sum_t += t;
        sum_t_squared += t * t;
        sum_offset += offset_sample;
        sum_t_offset += t * offset_sample;
    }
    if (count < 12 || count * 2 * stride < last - first ||
        count * sum_t_squared - sum_t * sum_t < 1)
        return false;
    const float slope =
            (count * sum_t_offset - sum_t * sum_offset) / (count * sum_t_squared - sum_t * sum_t);
    const float offset = (sum_offset - slope * sum_t) / count;
    line = {transpose(a + Point(0, offset)), transpose(delta + Point(0, slope))};
    return true;
}

bool fit_faint_border(const cv::Mat& gray, const Quad& q, int radius, int contrast, Quad& fit,
                      int spread = 1) {
    std::array<FaintLine, 4> lines;
    for (int i = 0; i < 4; ++i)
        if (!faint_line(gray, q[i], q[(i + 1) % 4], radius, contrast, lines[i], spread))
            return false;
    for (int i = 0; i < 4; ++i) {
        const auto& a = lines[(i + 3) % 4];
        const auto& b = lines[i];
        const float d = cross(a.direction, b.direction);
        if (std::abs(d) < 1)
            return false;
        fit[i] = a.anchor + a.direction * (cross(b.anchor - a.anchor, b.direction) / d);
        if (length(fit[i] - q[i]) > 2.5F * radius)
            return false;
    }
    return true;
}

bool refine_faint(const cv::Mat& gray, Detection& tag, int radius, int contrast) {
    Quad fit;
    // A diffuse transition can clear the same native contrast over four pixels.
    if (!fit_faint_border(gray, tag.corners_px_, radius, contrast, fit) &&
        !fit_faint_border(gray, tag.corners_px_, radius, contrast, fit, 2))
        return false;
    // A projective fit to the internal print transitions can reduce phase bias
    // when the four long outer rails have very little subpixel variation.
    Quad payload_fit = fit;
    if (propose_faint_grid(gray, tag.id_, contrast, payload_fit)) {
        if (auto errors = verify_payload(gray, payload_fit, tag.id_, 26, contrast, 1)) {
            tag.corners_px_ = payload_fit;
            tag.corrected_bits_ = *errors;
            return true;
        }
    }
    const auto errors = verify_payload(gray, fit, tag.id_, 26, contrast, 1);
    if (!errors)
        return false;
    tag.corners_px_ = fit;
    tag.corrected_bits_ = *errors;
    return true;
}

bool read_source_quad(const cv::Mat& gray, Quad q, Detection& tag, int ring, float contrast,
                      int corrections, int faint_radius, bool require_outline, int spread) {
    auto code = decode(gray, q, ring, contrast, corrections);
    // A noisy thumbnail can shift the seed enough to cross a native payload cell.
    if (code.id < 0 && faint_radius &&
        (!require_outline || code.supported ||
         sample_payload(gray, q, 18, contrast * .65F).ring_supported)) {
        Quad fit;
        if (fit_faint_border(gray, q, faint_radius, contrast, fit, spread)) {
            q = fit;
            code = decode(gray, q, ring, contrast, corrections);
        }
    }

    if (code.id < 0)
        return false;
    std::rotate(q.begin(), q.begin() + (4 - code.rotation) % 4, q.end());
    tag = {code.id, q, code.distance};
    return true;
}
bool read_quad(const cv::Mat& gray, const std::vector<cv::Point>& contour, const cv::Rect& box,
               OutlineHull& builder, cv::Point2f scale, Detection& tag, int ring = 20,
               float contrast = 12, int corrections = 3, int faint_radius = 0,
               bool require_outline = false) {
    std::vector<cv::Point> hull, polygon;
    builder.build(contour, box, hull);
    cv::approxPolyDP(hull, polygon, .025 * cv::arcLength(hull, true), true);
    if (polygon.size() != 4)
        return false;
    if (cv::contourArea(polygon, true) < 0)
        std::reverse(polygon.begin(), polygon.end());
    Quad q;
    for (int i = 0; i < 4; ++i)
        q[i] = {(polygon[i].x + .5F) * scale.x - .5F, (polygon[i].y + .5F) * scale.y - .5F};
    return read_source_quad(gray, q, tag, ring, contrast, corrections, faint_radius,
                            require_outline, 1);
}

bool read_faint_rectangle(const cv::Mat& gray, const std::vector<cv::Point>& contour,
                          const cv::Rect& bounds, cv::Point2f scale, const VideoConfig& settings,
                          int radius, Detection& tag) {
    const auto& limits = settings.sparse_faint_;
    if (std::max(bounds.width, bounds.height) >
        limits.maximum_box_aspect_ratio_ * std::min(bounds.width, bounds.height))
        return false;
    auto rect = cv::minAreaRect(contour);
    if (std::min(rect.size.width, rect.size.height) < 4)
        return false;
    std::vector<cv::Point> hull;
    cv::convexHull(contour, hull);
    if (cv::contourArea(hull) < limits.minimum_hull_fill_ * rect.size.area())
        return false;
    // Keep the fractional support lines. Rounding in this guide can shift a seed
    // by several native pixels, enough to sample the adjacent payload cell.
    Quad q;
    rect.points(q.data());
    for (auto& point : q)
        point = {(point.x + .5F) * scale.x - .5F, (point.y + .5F) * scale.y - .5F};
    if (cross(q[1] - q[0], q[2] - q[0]) < 0)
        std::reverse(q.begin(), q.end());
    return read_source_quad(gray, q, tag, 26, settings.faint_decode_contrast_, 1, radius, true, 2);
}

Point quad_center(const Quad& quad) {
    return .25F * (quad[0] + quad[1] + quad[2] + quad[3]);
}

// Sample dark local minima that the current ordinary detections do not explain.
bool has_unexplained_ink(const cv::Mat& gray, int difference, const std::vector<Detection>& tags) {
    const int step = std::max(3, std::min(gray.rows, gray.cols) / 64);
    std::vector<cv::Rect> covered;
    for (const auto& tag : tags) {
        auto box = cv::boundingRect(tag.corners_px_);
        box -= cv::Point(step, step);
        box += cv::Size(2 * step, 2 * step);
        covered.push_back(box);
    }
    int evidence = 0;
    for (int y = step; y + step < gray.rows; y += step) {
        const auto* above = gray.ptr<std::uint8_t>(y - step);
        const auto* row = gray.ptr<std::uint8_t>(y);
        const auto* below = gray.ptr<std::uint8_t>(y + step);
        for (int x = step; x + step < gray.cols; x += step) {
            const int surround = row[x - step] + row[x + step] + above[x] + below[x];
            if (surround - 4 * row[x] >= 4 * difference &&
                std::none_of(covered.begin(), covered.end(),
                             [&](auto box) { return box.contains(cv::Point(x, y)); }) &&
                ++evidence == 3)
                return true;
        }
    }
    return false;
}

// Under symmetric blur the physical boundary is halfway between the two plateaus.
// Measure both halves of every rail so opposite corner displacements cannot cancel.
float rail_error(const cv::Mat& gray, const Quad& quad, int contrast) {
    float error = 0;
    for (int side = 0; side < 4; ++side) {
        const Point start = quad[side], edge = quad[(side + 1) % 4] - start;
        const float extent = length(edge);
        if (extent < 32)
            return -1;
        const Point normal{edge.y / extent, -edge.x / extent};
        for (int half = 0; half < 2; ++half) {
            float residual = 0;
            int count = 0;
            for (int k = 0; k < 4; ++k) {
                const Point p = start + (.15F + .1F * (4 * half + k)) * edge;
                const float inner = bilinear(gray, p - 3.F * normal);
                const float outer = bilinear(gray, p + 3.F * normal);
                if (inner < 0 || outer - inner < .7F * contrast)
                    continue;
                residual += std::clamp((2 * bilinear(gray, p) - outer - inner) / (outer - inner),
                                       -1.5F, 1.5F);
                ++count;
            }
            if (count < 3)
                return -1;
            const float mean = residual / count;
            error += mean * mean;
        }
    }
    return error / 8;
}

bool recover_diffuse_seed(const cv::Mat& gray, const Detection& tag, int contrast, float before,
                          Quad& quad) {
    // Three-point profiles have to disagree by at least 15 percent RMS.
    if (before <= .15F * .15F)
        return false;
    Quad fit;
    if (!fit_faint_border(gray, quad, 4, contrast, fit))
        return false;
    Quad trial = fit;
    if (propose_diffuse_grid(gray, tag.id_, contrast, trial))
        fit = trial;
    float correction = 0;
    for (int i = 0; i < 4; ++i) {
        if (length(fit[i] - tag.corners_px_[i]) > 3.5F)
            return false;
        correction = std::max(correction, length(fit[i] - quad[i]));
    }
    // Ordinary diffuse alignment already fits within a half-pixel trust region.
    // Reseeding should only address displacements beyond that fit.
    if (correction <= .5F)
        return false;
    const float after = rail_error(gray, fit, contrast);
    if (after < 0 || after >= .55F * before)
        return false;
    if (!verify_payload(gray, fit, tag.id_, 26, contrast, std::min(1, tag.corrected_bits_)))
        return false;
    quad = fit;
    return true;
}

// The outer rails supply the fit. Independent payload transitions must confirm that
// the same projective warp improves each adequately sampled quadrant of the print.
struct PayloadResidual {
    std::array<float, 4> quadrant{};
    int count = 0;
    float mean = 0;
};

bool payload_residual(const cv::Mat& gray, MarkerCode code, const Quad& quad, int contrast,
                      PayloadResidual& residual) {
    Projection projection;
    if (!project(quad, projection))
        return false;
    const auto color = [code](int x, int y) {
        return x < 1 || x > 6 || y < 1 || y > 6
                       ? 0
                       : static_cast<int>((code >> (6 * (y - 1) + x - 1)) & 1);
    };
    std::array<int, 4> counts{};
    constexpr float shoulder = .39F / kBorderSide;
    for (int j = 0; j < kBorderSide; ++j) {
        for (int i = 1; i < kBorderSide; ++i) {
            for (int axis = 0; axis < 2; ++axis) {
                const int sign =
                        axis ? color(j, i) - color(j, i - 1) : color(i, j) - color(i - 1, j);
                if (!sign)
                    continue;
                const float u = (axis ? j + .5F : i) / kBorderSide;
                const float v = (axis ? i : j + .5F) / kBorderSide;
                const float du = axis ? 0.F : shoulder;
                const float dv = axis ? shoulder : 0.F;
                const float before = projection.sample(gray, u - du, v - dv);
                const float after = projection.sample(gray, u + du, v + dv);
                if (before < 0 || after < 0 || (after - before) * sign < 2 * contrast)
                    continue;
                const float middle = projection.sample(gray, u, v);
                if (middle < 0)
                    continue;
                const float offset =
                        std::clamp((2 * middle - before - after) / (after - before), -1.5F, 1.5F);
                const int sector = (u > .5F) + 2 * (v > .5F);
                residual.quadrant[sector] += offset * offset;
                residual.mean += offset * offset;
                ++counts[sector];
                ++residual.count;
            }
        }
    }
    if (*std::min_element(counts.begin(), counts.end()) < 3)
        return false;
    for (int i = 0; i < 4; ++i)
        residual.quadrant[i] /= counts[i];
    residual.mean /= residual.count;
    return true;
}

void recover_sheared_seed(const cv::Mat& gray, Detection& tag, int contrast, float before,
                          float max_cells) {
    if (before <= .003F || max_cells == 0)
        return;
    const auto& table = generated::kDictApriltag36h11Table;
    if (tag.id_ < 0 || tag.id_ >= table.code_count_)
        return;
    const auto& quad = tag.corners_px_;
    float side = length(quad[3] - quad[0]);
    for (int i = 1; i < 4; ++i)
        side = std::min(side, length(quad[i] - quad[i - 1]));
    const float cell = side / kBorderSide;
    if (cell < 5)
        return;
    const MarkerCode code = table.codes_[4 * tag.id_];
    PayloadResidual original;
    if (!payload_residual(gray, code, quad, contrast, original) || original.mean <= .01F)
        return;
    Quad trial;
    const int radius = std::min(8, cvCeil(.60F * cell));
    if (!fit_faint_border(gray, quad, radius, contrast, trial))
        return;
    Quad local = trial;
    if (propose_diffuse_grid(gray, tag.id_, contrast, local))
        trial = local;
    float movement = 0;
    for (int i = 0; i < 4; ++i)
        movement = std::max(movement, length(trial[i] - quad[i]));
    if (movement <= .5F || movement > std::min(8.F, max_cells * cell))
        return;

    const float after = rail_error(gray, trial, contrast);
    PayloadResidual candidate;
    if (after < 0 || after >= .65F * before ||
        !payload_residual(gray, code, trial, contrast, candidate) ||
        candidate.count + 2 < original.count || candidate.mean >= .50F * original.mean)
        return;
    for (int i = 0; i < 4; ++i)
        if (candidate.quadrant[i] > original.quadrant[i] + .0016F)
            return;
    if (verify_payload(gray, trial, tag.id_, 26, contrast, std::min(1, tag.corrected_bits_)))
        tag.corners_px_ = trial;
}

}  // namespace

void refine_diffuse_tags(const cv::Mat& gray, int minimum_contrast, std::vector<Detection>& tags,
                         float maximum_reseed_cells) {
    for (auto& tag : tags) {
        auto quad = tag.corners_px_;
        const bool fitted = propose_diffuse_grid(gray, tag.id_, minimum_contrast, quad);
        const float before = rail_error(gray, quad, minimum_contrast);
        if (recover_diffuse_seed(gray, tag, minimum_contrast, before, quad)) {
            tag.corners_px_ = quad;
            continue;
        }
        if (!fitted) {
            recover_sheared_seed(gray, tag, minimum_contrast, before, maximum_reseed_cells);
            continue;
        }
        if (verify_payload(gray, quad, tag.id_, 26, minimum_contrast, 1))
            tag.corners_px_ = quad;
    }
}

#if CV_SIMD128 && defined(__aarch64__) && defined(__AARCH64EL__) && defined(__linux__)
namespace {
// Keep the original access pattern on the tested pre-SVE AArch64 hardware.
bool use_fused_faint_guide() {
    static const bool supported = (getauxval(AT_HWCAP) & HWCAP_SVE) != 0;
    return supported;
}

void sample_faint_stride_twelve(const cv::Mat& gray, cv::Mat& guide, cv::Mat& low, cv::Mat& high) {
    constexpr int stride = 12;
    const int w = gray.cols / stride, h = gray.rows / stride;
    guide.create(h, w, CV_8U);
    low.create((h + 3) / 4, (w + 3) / 4, CV_8U);
    high.create(low.size(), CV_8U);
    const uint8x16_t index = {6,   18,  30,  42,  54,  66,  78,  90,
                              102, 114, 126, 138, 150, 162, 174, 186};
    for (int y = 0; y < h; y += 4) {
        std::array<const std::uint8_t*, 4> a;
        std::array<std::uint8_t*, 4> d;
        for (int j = 0; j < 4; ++j) {
            const int i = std::min(h - 1, y + j);
            a[j] = gray.ptr<std::uint8_t>(stride * i + stride / 2);
            d[j] = guide.ptr<std::uint8_t>(i);
        }
        auto* l = low.ptr<std::uint8_t>(y / 4);
        auto* u = high.ptr<std::uint8_t>(y / 4);
        int x = 0;
        for (; x + 16 <= w; x += 16) {
            uint8x16_t z[4];
            for (int j = 0; j < 4; ++j) {
                const auto* addr = a[j] + stride * x;
                if (x + 32 < w)
                    __builtin_prefetch(addr + 384, 0, 2);
                auto q = vqtbl4q_u8(vld1q_u8_x4(addr), index);
                q = vqtbx4q_u8(q, vld1q_u8_x4(addr + 64), vsubq_u8(index, vdupq_n_u8(64)));
                q = vqtbx4q_u8(q, vld1q_u8_x4(addr + 128), vsubq_u8(index, vdupq_n_u8(128)));
                z[j] = q;
                vst1q_u8(d[j] + x, q);
            }
            const auto mn = vminq_u8(vminq_u8(z[0], z[1]), vminq_u8(z[2], z[3]));
            const auto mx = vmaxq_u8(vmaxq_u8(z[0], z[1]), vmaxq_u8(z[2], z[3]));
            const auto tl = vpmin_u8(vget_low_u8(mn), vget_high_u8(mn)),
                       tu = vpmax_u8(vget_low_u8(mx), vget_high_u8(mx));
            const auto lb = vget_lane_u32(vreinterpret_u32_u8(vpmin_u8(tl, tl)), 0),
                       ub = vget_lane_u32(vreinterpret_u32_u8(vpmax_u8(tu, tu)), 0);
            std::memcpy(l + x / 4, &lb, 4);
            std::memcpy(u + x / 4, &ub, 4);
        }
        for (; x < w; x += 4) {
            int mn = 255, mx = 0;
            for (int j = 0; j < 4; ++j)
                for (int i = x; i < std::min(x + 4, w); ++i) {
                    const auto q = a[j][stride * i + stride / 2];
                    d[j][i] = q;
                    mn = std::min(mn, int(q));
                    mx = std::max(mx, int(q));
                }
            l[x / 4] = mn;
            u[x / 4] = mx;
        }
    }
}
cv::Rect blackhat_possible_region(const cv::Mat& src, int difference, const cv::Mat& lows,
                                  const cv::Mat& highs) {
    const int w = lows.cols, h = lows.rows;
    cv::Mat peaks;
    // Closing takes a minimum of dilated values. A corner dilation window
    // provides an upper bound for every corresponding position in the tile.
    cv::dilate(highs, peaks, cv::Mat::ones(5, 5, CV_8U));
    int l = w, r = -1, t = h, b = -1;
    for (int y = 0; y < h; ++y) {
        const auto* a = peaks.ptr<std::uint8_t>(std::max(0, y - 2));
        const auto* z = peaks.ptr<std::uint8_t>(std::min(h - 1, y + 2));
        const auto* low = lows.ptr<std::uint8_t>(y);
        for (int x = 0; x < w; ++x) {
            const int x0 = std::max(0, x - 2), x1 = std::min(w - 1, x + 2);
            if (std::min({a[x0], a[x1], z[x0], z[x1]}) <= int(low[x]) + difference)
                continue;
            l = std::min(l, x);
            r = std::max(r, x);
            t = std::min(t, y);
            b = y;
        }
    }
    return r < l ? cv::Rect()
                 : cv::Rect(4 * l, 4 * t, std::min(src.cols, 4 * r + 4) - 4 * l,
                            std::min(src.rows, 4 * b + 4) - 4 * t);
}
}  // namespace
#endif

std::vector<Detection> IsolatedInk::find_faint(const cv::Mat& gray, const VideoConfig& settings,
                                               FaintGuide mode,
                                               const std::vector<Detection>* explained,
                                               bool* used_local_mean, cv::Mat* shared_area) {
    std::vector<Detection> found;
    if (used_local_mean)
        *used_local_mean = false;
    if (std::min(gray.cols, gray.rows) < 16 || !settings.maximum_isolated_regions_)
        return found;
    if (explained && !has_unexplained_ink(gray, settings.faint_background_difference_, *explained))
        return found;
    const bool sparse = mode == FaintGuide::sparse;
    const int side = sparse ? settings.sparse_faint_.guide_side_px_ : settings.faint_guide_side_px_;
    if (side == 0)
        return found;
    const double scale = std::min(1., double(side) / std::min(gray.rows, gray.cols));
    const cv::Size target(cvRound(gray.cols * scale), cvRound(gray.rows * scale));
    if (std::min(target.width, target.height) < 2)
        return found;
    // Closing fills narrow black rails from their local light surround; subtracting
    // the original removes smooth illumination without needing a calibrated exposure.
    const int window = std::max(3, 2 * cvRound(.5 * settings.faint_background_window_ *
                                               settings.faint_guide_side_px_ / 180.) +
                                           1);
    std::vector<std::vector<cv::Point>> contours;
    const auto kernel = cv::getStructuringElement(cv::MORPH_RECT, {window, window});
    const auto limit = static_cast<std::size_t>(settings.maximum_isolated_regions_);
    const bool predetermined = sparse || explained;
#if CV_SIMD128 && defined(__aarch64__) && defined(__AARCH64EL__) && defined(__linux__)
    cv::Mat low_tiles, high_tiles;
#endif
    const auto outlines = [&](const cv::Mat& samples) {
        cv::Mat result;
        cv::Point origin{};
#if CV_SIMD128 && defined(__aarch64__) && defined(__AARCH64EL__) && defined(__linux__)
        if (!low_tiles.empty() && samples.data == guide_.data) {
            const auto possible = blackhat_possible_region(
                    samples, settings.faint_background_difference_, low_tiles, high_tiles);
            if (possible.empty()) {
                contours.clear();
                return true;
            }
            // Dilation followed by erosion needs two radii of original samples.
            const int halo = window - 1;
            const auto expanded =
                    (possible + cv::Size(2 * halo, 2 * halo) - cv::Point(halo, halo)) &
                    cv::Rect({}, samples.size());
            cv::morphologyEx(samples(expanded), mask_, cv::MORPH_BLACKHAT, kernel, {-1, -1}, 1,
                             cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);
            result = mask_(cv::Rect(possible.tl() - expanded.tl(), possible.size()));
            origin = possible.tl();
        }
#endif
        if (result.empty()) {
            cv::morphologyEx(samples, mask_, cv::MORPH_BLACKHAT, kernel);
            result = mask_;
        }
        cv::threshold(result, result, settings.faint_background_difference_, 255,
                      cv::THRESH_BINARY);
        // Contour padding must see the certified zero exterior of a cropped mask.
        const cv::Mat view(result.rows, result.cols, result.type(), result.data, result.step[0]);
        if (exceeds_external_limit(view, limit, obscured_columns_))
            return false;
        cv::findContours(view, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, origin);
        return contours.size() <= limit;
    };
    const bool full_quarter_blocks =
            gray.cols == 4 * target.width && gray.rows == 4 * target.height;
    const auto use_center_average = [&] {
        return full_quarter_blocks && settings.adaptive_average_minimum_noise_per_mille_ > 0 &&
               noisy_guide(gray, 4 * settings.faint_background_difference_,
                           settings.adaptive_average_minimum_noise_per_mille_);
    };
    const bool direct_average =
            predetermined && mode != FaintGuide::exterior && use_center_average();
    const auto sample =
            direct_average ? QuarterGuideSample::central_area : QuarterGuideSample::nearest;
    const bool shared = shared_area && shared_area->empty() && scale == .25 &&
                        try_sample_gray4x4(gray, *shared_area, guide_, sample);
    if (!shared) {
        if (direct_average) {
            guide_.create(target, CV_8U);
            sample_area4(gray, guide_);
        } else if (scale == .25 && full_quarter_blocks) {
            resize_nearest_guide(gray, guide_, 4, cv::INTER_NEAREST_EXACT);
#if CV_SIMD128 && defined(__aarch64__) && defined(__AARCH64EL__) && defined(__linux__)
        } else if (!predetermined && window == 17 && gray.cols == 12 * target.width &&
                   gray.rows == 12 * target.height && use_fused_faint_guide()) {
            sample_faint_stride_twelve(gray, guide_, low_tiles, high_tiles);
#endif
        } else {
            cv::resize(gray, guide_, {}, scale, scale, cv::INTER_NEAREST_EXACT);
        }
    }
    const bool needs_fallback = predetermined || !outlines(guide_);
    const bool local_search = needs_fallback && mode != FaintGuide::exterior;
    if (used_local_mean)
        *used_local_mean = local_search;
    if (local_search) {
        if (!predetermined && use_center_average())
            sample_area4(gray, guide_);
            // The inner edge of the rail can survive even if structured texture joins its exterior.
#if CV_SIMD128
        if (settings.faint_background_window_ == 15 &&
            !may_have_local_contour15(guide_,
                                      settings.faint_background_difference_ + (sparse ? 0 : 1)))
            return found;
#endif
        local_ink_mask(guide_, settings.faint_background_window_,
                       settings.faint_background_difference_ + (sparse ? 0 : 1), variation_, mask_);
        find_guide_contours(mask_, this->guide_contours_, contours);
    } else if (needs_fallback) {
        cv::medianBlur(guide_, variation_, 3);
        if (!outlines(variation_))
            return found;
    }
    const cv::Point2f factor(float(gray.cols) / guide_.cols, float(gray.rows) / guide_.rows);
    for (const auto& contour : contours) {
        const auto box = cv::boundingRect(contour);
        if (std::min(box.height, box.width) < kMinimumGuideContourSide)
            continue;
        if (explained) {
            const Point middle{(box.x + .5F * box.width) * factor.x,
                               (box.y + .5F * box.height) * factor.y};
            const float size = std::min(box.width * factor.x, box.height * factor.y);
            if (std::any_of(explained->begin(), explained->end(), [&](const Detection& prior) {
                    const auto& q = prior.corners_px_;
                    return length(middle - quad_center(q)) <
                           .15F * std::min(size, length(q[2] - q[0]));
                }))
                continue;
        }
        Detection tag;
        const int radius = std::max(3, cvCeil(1.5F * std::max(factor.x, factor.y)));
        if (!read_quad(gray, contour, box, this->outline_hull_, factor, tag, 26,
                       settings.faint_decode_contrast_, 1, radius, local_search) &&
            !(sparse && read_faint_rectangle(gray, contour, box, factor, settings, radius, tag)))
            continue;
        if (!refine_faint(gray, tag, radius, settings.faint_decode_contrast_))
            continue;
        const auto& quad = tag.corners_px_;
        if (!local_search || std::none_of(found.begin(), found.end(), [&](const Detection& prior) {
                return length(quad_center(prior.corners_px_) - quad_center(quad)) <
                       .2F * length(quad[2] - quad[0]);
            }))
            found.push_back(tag);
    }
    return found;
}

bool verify_faint_source(const cv::Mat& gray, const Quad& seed, Detection& tag, int contrast,
                         int radius) {
    return read_source_quad(gray, seed, tag, 26, contrast, 1, radius, true, 1) &&
           refine_faint(gray, tag, radius, contrast);
}

namespace {
using Span = std::pair<int, int>;

#if CV_SIMD128
// Every pixel used by a one- or four-pixel difference lies in this band plus
// four rows. Its column's minimum and maximum bound the actual intensity.
// Only an inconclusive bound needs the original per-pixel and footprint tests.
template <int vectors>
void uncertain_ranges(const cv::Mat& gray, int first_row, int last_row, int maximum_step,
                      std::uint8_t* low, std::uint8_t* high, std::vector<Span>& ranges) {
    constexpr int lanes = 16;
    const int width = gray.cols;
    int x = 0;
    for (; x + lanes * vectors <= width; x += lanes * vectors) {
        std::array<cv::v_uint8x16, vectors> lower, upper;
        const auto* first = gray.ptr<std::uint8_t>(first_row) + x;
        for (int j = 0; j < vectors; ++j)
            lower[j] = upper[j] = cv::v_load(first + j * lanes);
        int y = first_row + 1;
        if constexpr (vectors == 8) {
            for (; y + 1 < last_row; y += 2) {
                const auto* row = gray.ptr<std::uint8_t>(y) + x;
                const auto* next = gray.ptr<std::uint8_t>(y + 1) + x;
                for (int j = 0; j < vectors; ++j) {
                    const auto a = cv::v_load(row + j * lanes);
                    const auto b = cv::v_load(next + j * lanes);
                    lower[j] = cv::v_min(lower[j], cv::v_min(a, b));
                    upper[j] = cv::v_max(upper[j], cv::v_max(a, b));
                }
            }
        }
        for (; y < last_row; ++y) {
            const auto* row = gray.ptr<std::uint8_t>(y) + x;
            for (int j = 0; j < vectors; ++j) {
                const auto value = cv::v_load(row + j * lanes);
                lower[j] = cv::v_min(lower[j], value);
                upper[j] = cv::v_max(upper[j], value);
            }
        }
        for (int j = 0; j < vectors; ++j) {
            cv::v_store(low + x + lanes * j, lower[j]);
            cv::v_store(high + x + lanes * j, upper[j]);
        }
    }
    for (; x < width; ++x) {
        std::uint8_t lo = 255, hi = 0;
        for (int y = first_row; y < last_row; ++y) {
            const auto value = gray.ptr<std::uint8_t>(y)[x];
            lo = std::min(lo, value);
            hi = std::max(hi, value);
        }
        low[x] = lo;
        high[x] = hi;
    }
    const auto append = [&](int first, int end) {
        if (!ranges.empty() && ranges.back().second == first)
            ranges.back().second = end;
        else
            ranges.emplace_back(first, end);
    };
    x = 0;
    for (; x + lanes + 4 <= width; x += lanes) {
        const auto lo = cv::v_min(cv::v_load(low + x),
                                  cv::v_min(cv::v_load(low + x + 1), cv::v_load(low + x + 4)));
        const auto hi = cv::v_max(cv::v_load(high + x),
                                  cv::v_max(cv::v_load(high + x + 1), cv::v_load(high + x + 4)));
        if (cv::v_reduce_max(cv::v_sub(hi, lo)) > maximum_step)
            append(x, x + lanes);
    }
    const int tail_start = x;
    for (; x < width; ++x) {
        const int direction = x + 4 < width ? 1 : -1;
        const int near = x + direction, far = x + 4 * direction;
        if (std::max({high[x], high[near], high[far]}) - std::min({low[x], low[near], low[far]}) >
            maximum_step) {
            append(tail_start, width);
            break;
        }
    }
}
#endif
}  // namespace

bool IsolatedInk::quiet_outside(const cv::Mat& gray, const std::vector<Detection>& tags,
                                int maximum_step) {
    // Check native contrast outside the decoded footprints, including sampling phases
    // absent from the guide. Short and four-pixel steps catch sharp and gradual edges.
    if (tags.size() > 3 || std::min(gray.rows, gray.cols) < 20)
        return false;
    struct Edge {
        Point first;
        Point last;
        Point delta;
    };
    struct Footprint {
        std::array<Edge, 4> edges;
        float top, bottom;
    };
    std::array<Footprint, 3> footprints;
    for (std::size_t i = 0; i < tags.size(); ++i) {
        Quad expanded;
        const Quad& q = tags[i].corners_px_;
        Point c = (q[0] + q[1] + q[2] + q[3]) * .25F;
        for (int j = 0; j < 4; ++j) {
            Point delta = q[j] - c;
            expanded[j] = q[j] + delta * (9.F / std::max(1.F, length(delta)));
        }
        auto& footprint = footprints[i];
        footprint.top = footprint.bottom = expanded[0].y;
        for (int j = 0; j < 4; ++j) {
            auto first = expanded[j], last = expanded[(j + 1) % 4];
            if (first.y > last.y)
                std::swap(first, last);
            footprint.edges[j] = {first, last, last - first};
            footprint.top = std::min(footprint.top, first.y);
            footprint.bottom = std::max(footprint.bottom, last.y);
        }
    }
    // Inspect the first row directly so a textured frame can exit immediately.
    std::vector<Span> ranges{{0, gray.cols}};
#if CV_SIMD128
#if CV_SIMD128 && defined(__aarch64__) && defined(__AARCH64EL__) && defined(__linux__)
    const bool wide_bands = gray.total() >= 6000000 && use_fused_faint_guide();
    const int band_rows = wide_bands ? 32 : 16;
#else
    constexpr int band_rows = 16;
#endif
    cv::AutoBuffer<std::uint8_t> limits(2 * gray.cols);
#endif
    for (int y = 0; y < gray.rows; ++y) {
#if CV_SIMD128
        if (y > 0 && (y - 1) % band_rows == 0) {
            const int end = std::min(y + band_rows, gray.rows);
            ranges.clear();
#if CV_SIMD128 && defined(__aarch64__) && defined(__AARCH64EL__) && defined(__linux__)
            if (wide_bands)
                uncertain_ranges<8>(gray, y, std::min(end + 4, gray.rows), maximum_step,
                                    limits.data(), limits.data() + gray.cols, ranges);
            else
#endif
                uncertain_ranges<4>(gray, y, std::min(end + 4, gray.rows), maximum_step,
                                    limits.data(), limits.data() + gray.cols, ranges);
            if (ranges.empty()) {
                y = end - 1;
                continue;
            }
        }
#endif
        const auto* row = gray.ptr<std::uint8_t>(y);
        const auto* lower = gray.ptr<std::uint8_t>(std::min(y + 4, gray.rows - 1));
        const auto* next = gray.ptr<std::uint8_t>(std::min(y + 1, gray.rows - 1));
        std::array<std::pair<int, int>, 3> skip;
        int spans = 0;
        for (std::size_t i = 0; i < tags.size(); ++i) {
            const auto& footprint = footprints[i];
            if (y < footprint.top || y >= footprint.bottom)
                continue;
            float left = gray.cols, right = 0;
            for (const auto& edge : footprint.edges) {
                if (y < edge.first.y || y >= edge.last.y)
                    continue;
                const float x = edge.first.x + edge.delta.x * ((y - edge.first.y) / edge.delta.y);
                left = std::min(left, x);
                right = std::max(right, x);
            }
            if (right > left)
                skip[spans++] = {std::max(0, cvFloor(left) - 4),
                                 std::min(gray.cols, cvCeil(right) + 4)};
        }
        std::sort(skip.begin(), skip.begin() + spans);
        for (auto [first, last] : ranges) {
            int x = first;
            const int limit = std::min(last, gray.cols - 4);
            for (int j = 0; j <= spans; ++j) {
                const int end = j == spans ? limit : std::min(limit, skip[j].first);
#if CV_SIMD128
                cv::v_uint8x16 high = cv::v_setzero_u8();
                for (; x + 16 <= end; x += 16) {
                    const auto v = cv::v_load(row + x);
                    high = cv::v_max(high, cv::v_max(cv::v_absdiff(v, cv::v_load(row + x + 1)),
                                                     cv::v_absdiff(v, cv::v_load(next + x))));
                    high = cv::v_max(high, cv::v_max(cv::v_absdiff(v, cv::v_load(row + x + 4)),
                                                     cv::v_absdiff(v, cv::v_load(lower + x))));
                }
                if (cv::v_reduce_max(high) > maximum_step)
                    return false;
#endif
                for (; x < end; ++x)
                    if (std::max({std::abs(row[x] - row[x + 4]), std::abs(row[x] - lower[x]),
                                  std::abs(row[x] - row[x + 1]), std::abs(row[x] - next[x])}) >
                        maximum_step)
                        return false;
                if (j < spans)
                    x = std::max(x, skip[j].second);
            }
        }
        if (ranges.back().second < gray.cols)
            continue;
        for (int x = gray.cols - 4; x < gray.cols; ++x)
            if (std::max({std::abs(row[x] - row[x - 4]), std::abs(row[x] - lower[x]),
                          std::abs(row[x] - row[x - 1]), std::abs(row[x] - next[x])}) >
                maximum_step)
                return false;
    }
    return true;
}
bool IsolatedInk::find(const cv::Mat& gray, int low, const VideoConfig& settings, Detector& native,
                       Detector& patch, std::vector<Detection>& tags) {
    if (!settings.maximum_isolated_regions_)
        return false;
    const int stride = std::max(
            3, cvRound(double(std::min(gray.rows, gray.cols)) / settings.isolated_guide_side_px_));
    const int cutoff = low - settings.isolated_luma_margin_;
    if (cutoff < 0)
        return false;
    if (stride == 4 && gray.rows % 4 == 0 && gray.cols % 4 == 0)
        sample_integer<4>(gray, cutoff, guide_, mask_);
    else if (stride == 12 && gray.rows % 12 == 0 && gray.cols % 12 == 0)
        sample_integer<12>(gray, cutoff, guide_, mask_);
    else {
        cv::resize(gray, guide_,
                   {(gray.cols + stride - 1) / stride, (gray.rows + stride - 1) / stride}, 0, 0,
                   cv::INTER_NEAREST_EXACT);
        cv::threshold(guide_, mask_, cutoff, 255, cv::THRESH_BINARY_INV);
    }
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask_, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty() ||
        contours.size() > static_cast<std::size_t>(settings.maximum_isolated_regions_))
        return false;
    const cv::Point2f scale{float(gray.cols) / guide_.cols, float(gray.rows) / guide_.rows};
    // Look for a second, low-contrast target that might never cross the global dark level.
    cv::morphologyEx(guide_, variation_, cv::MORPH_GRADIENT, cv::Mat());
    std::vector<Detection> confirmed;
    for (const auto& contour : contours) {
        const auto box = cv::boundingRect(contour);
        if (std::min(box.width, box.height) < 4 || box.x == 0 || box.y == 0 ||
            box.br().x >= guide_.cols || box.br().y >= guide_.rows ||
            box.width > 2.4 * box.height || box.height > 2.4 * box.width ||
            static_cast<std::size_t>(box.area()) > guide_.total() / 5)
            return false;
        variation_(expand(box, 2 + std::max(box.width, box.height) / 5, guide_.size())).setTo(0);
        Detection candidate;
        if (read_quad(gray, contour, box, this->outline_hull_, scale, candidate)) {
            // read_quad already verified this oriented payload on the native image.
            refine(gray, candidate);
            confirmed.push_back(candidate);
            continue;
        }
        const auto region = expand(box, 6, guide_.size());
        std::vector<Detection> local;
        if (std::min(region.width, region.height) >= 24) {
            local = native.detect(guide_(region));
            for (auto& tag : local)
                for (auto& q : tag.corners_px_)
                    q = {(q.x + region.x + .5F) * scale.x - .5F,
                         (q.y + region.y + .5F) * scale.y - .5F};
        }
        if (local.empty()) {
            const cv::Rect source =
                    cv::Rect(cv::Point(cvFloor(region.x * scale.x), cvFloor(region.y * scale.y)),
                             cv::Point(cvCeil(region.br().x * scale.x),
                                       cvCeil(region.br().y * scale.y))) &
                    cv::Rect({}, gray.size());
            local = patch.detect(gray(source));
            for (auto& tag : local)
                for (auto& q : tag.corners_px_)
                    q += cv::Point2f(source.tl());
        }
        if (local.size() != 1 || !refine(gray, local.front()))
            return false;
        confirmed.push_back(local.front());
    }
    double strongest;
    cv::minMaxLoc(variation_, nullptr, &strongest);
    if (strongest >= settings.isolated_residual_contrast_)
        return false;
    tags = std::move(confirmed);
    return true;
}
}  // namespace aruco3cuda::apriltag_small_cpu::detail
