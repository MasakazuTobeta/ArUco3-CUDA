// SPDX-License-Identifier: Apache-2.0
#include "noisy_ink.hpp"

#include "binary_median.hpp"
#include "faint_guide.hpp"
#include "guide_average.hpp"
#include "guide_contours.hpp"
#include "isolated_ink.hpp"
#include "luma_distribution.hpp"
#include "tag_codebook.hpp"

#include <opencv2/imgproc.hpp>

#include <optional>

namespace aruco3cuda::apriltag_small_cpu::detail {
namespace {
cv::Rect expand(cv::Rect box, int pad, cv::Size size) {
    return (box + cv::Size(2 * pad, 2 * pad) - cv::Point(pad, pad)) & cv::Rect({}, size);
}
Point quad_center(const Quad& quad) {
    return .25F * (quad[0] + quad[1] + quad[2] + quad[3]);
}
Decode noisy_seed(const cv::Mat& guide, const std::vector<cv::Point>& contour,
                  const ReadNoiseConfig& noise, Quad& seed) {
    std::vector<cv::Point> hull, polygon;
    cv::convexHull(contour, hull);
    cv::approxPolyDP(hull, polygon, .025 * cv::arcLength(hull, true), true);
    if (polygon.size() < 4 || polygon.size() > 8)
        return {};
    // A binary majority filter can chamfer the original printed corner. Drop the
    // smallest convex triangles only when four vertices still explain the area.
    const double initial_area = cv::contourArea(polygon);
    while (polygon.size() > 4) {
        std::size_t smallest = 0;
        double minimum = std::numeric_limits<double>::max();
        for (std::size_t j = 0; j < polygon.size(); ++j) {
            const auto a = polygon[(j + polygon.size() - 1) % polygon.size()];
            const auto b = polygon[j], c = polygon[(j + 1) % polygon.size()];
            const double twice_area = std::abs((b - a).cross(c - a));
            if (twice_area < minimum) {
                smallest = j;
                minimum = twice_area;
            }
        }
        polygon.erase(polygon.begin() + smallest);
    }
    const double area = cv::contourArea(polygon, true);
    if (std::abs(area) < noise.minimum_hull_retention_ * initial_area)
        return {};
    if (area < 0)
        std::reverse(polygon.begin(), polygon.end());
    std::copy(polygon.begin(), polygon.end(), seed.begin());
    return decode(guide, seed, 16, noise.guide_difference_, 1);
}

bool read_noisy_crop(const cv::Mat& gray, Quad seed, cv::Rect box, Point scale, int radius,
                     const Decode& guide_code, float guide_step, const ReadNoiseConfig& noise,
                     NoisyCropFilter& crop_filter, cv::Mat& filtered, Detection& tag,
                     std::optional<Quad>& retry_guide) {
    const int pad =
            2 * radius + 4 + cvCeil(.1F * std::max(box.width * scale.x, box.height * scale.y));
    cv::Rect roi = expand({cvFloor(box.x * scale.x), cvFloor(box.y * scale.y),
                           cvCeil(box.width * scale.x), cvCeil(box.height * scale.y)},
                          pad, gray.size());
    roi.width -= roi.width % 2;
    roi.height -= roi.height % 2;
    if (roi.empty())
        return false;
    // Each half-size sample covers two native pixels per axis. Transform both
    // directions at pixel centers, including when the crop touches a frame edge.
    const auto crop_coordinates = [&](Quad corners) {
        for (auto& p : corners) {
            p.x = .5F * ((p.x + .5F) * scale.x - 1.F - roi.x);
            p.y = .5F * ((p.y + .5F) * scale.y - 1.F - roi.y);
        }
        return corners;
    };
    crop_filter.filter(gray(roi), noise.crop_blur_sigma_, filtered);
    const int half_radius = std::max(3, (radius + 1) / 2);
    if (!verify_faint_source(filtered, crop_coordinates(seed), tag, noise.minimum_contrast_,
                             half_radius)) {
        if (guide_code.id < 0)
            return false;
        // The reduced mask can place the outline inside the printed edge. Try
        // moving the seed half a guide pixel outwards using the same source crop.
        const auto center = quad_center(seed);
        const float radius_samples = std::max(1.F, .5F * length(seed[2] - seed[0]));
        for (auto& corner : seed)
            corner = center + (1.F + .5F / radius_samples) * (corner - center);
        if (!verify_faint_source(filtered, crop_coordinates(seed), tag, noise.minimum_contrast_,
                                 half_radius) ||
            tag.id_ != guide_code.id)
            return false;
        for (auto& p : seed) {
            p.x = (p.x + .5F) * guide_step - .5F;
            p.y = (p.y + .5F) * guide_step - .5F;
        }
        std::rotate(seed.begin(), seed.begin() + (4 - guide_code.rotation) % 4, seed.end());
        retry_guide = seed;
    }
    for (auto& p : tag.corners_px_)
        p = 2.F * p + Point(.5F + roi.x, .5F + roi.y);
    return true;
}

struct PrintedCrossing {
    float u, v;
    int axis, polarity;
};

constexpr int kMaximumCrossings = 2 * 8 * 9;

int printed_crossings(int id, std::array<PrintedCrossing, kMaximumCrossings>& crossings) {
    const auto& dictionary = generated::kDictApriltag36h11Table;
    if (id < 0 || id >= dictionary.code_count_)
        return 0;
    const auto word = dictionary.codes_[id * 4];
    const auto white = [&](int x, int y) {
        if (x < 0 || y < 0 || x >= 8 || y >= 8)
            return true;
        if (x == 0 || y == 0 || x == 7 || y == 7)
            return false;
        return bool((word >> ((y - 1) * 6 + x - 1)) & 1);
    };
    int count = 0;
    for (int axis = 0; axis < 2; ++axis)
        for (int cell = 0; cell < 8; ++cell)
            for (int border = 0; border <= 8; ++border) {
                const int sign = axis ? int(white(cell, border)) - int(white(cell, border - 1))
                                      : int(white(border, cell)) - int(white(border - 1, cell));
                if (sign)
                    crossings[count++] = {.125F * (axis ? cell + .5F : border),
                                          .125F * (axis ? border : cell + .5F), axis, sign};
            }
    return count;
}

struct MidpointSample {
    cv::Vec<float, 8> jacobian;
    float residual, slope, contrast;
};

bool noisy_grid_step(const cv::Mat& image,
                     const std::array<PrintedCrossing, kMaximumCrossings>& sites, int site_count,
                     int minimum_contrast, Point center, float side, Quad& quad, float& max_step) {
    Projection projection;
    if (!project(quad, projection))
        return false;
    std::array<MidpointSample, kMaximumCrossings> samples;
    std::array<float, kMaximumCrossings> contrasts, slopes;
    int count = 0;
    for (int i = 0; i < site_count; ++i) {
        const auto site = sites[i];
        const float z = 1 + projection.z_x * site.u + projection.z_y * site.v;
        if (!(z > 0))
            return false;
        const Point p = (projection.initial + projection.x * site.u + projection.y * site.v) / z;
        const Point dx = (projection.x - p * projection.z_x) / z,
                    dy = (projection.y - p * projection.z_y) / z;
        const Point tangent = site.axis ? dx : dy;
        const float extent = length(tangent);
        if (!(extent >= 1))
            return false;
        Point normal{tangent.y / extent, -tangent.x / extent};
        const float pitch = .125F * normal.dot(site.axis ? dy : dx);
        if (pitch < 0)
            normal = -normal;
        // Stay inside the adjacent cells, even on an oblique projection.
        const float gap = std::min(3.F, .31F * std::abs(pitch));
        if (!(gap >= 1))
            return false;
        const auto span = normal * gap;
        const float low = bilinear(image, p - span), high = bilinear(image, p + span);
        const float mid = bilinear(image, p), inner = bilinear(image, p - .4F * span),
                    outer = bilinear(image, p + .4F * span);
        if (std::min({low, high, mid, inner, outer}) < 0)
            continue;
        const auto local = (p - center) / side;
        const float projective = -local.dot(normal);
        auto& sample = samples[count];
        // Project the eight homography derivatives onto the observed edge normal.
        sample.jacobian = {local.x * normal.x,   local.y * normal.x,  normal.x,
                           local.x * normal.y,   local.y * normal.y,  normal.y,
                           projective * local.x, projective * local.y};
        sample.residual = (.5F * (low + high) - mid) * site.polarity;
        sample.slope = (outer - inner) * site.polarity / (.8F * gap);
        sample.contrast = (high - low) * site.polarity;
        contrasts[count] = sample.contrast;
        slopes[count] = sample.slope;
        ++count;
    }
    if (count * 5 < site_count * 4)
        return false;
    const auto median = [count](auto& values) {
        const auto middle = values.begin() + count / 2;
        std::nth_element(values.begin(), middle, values.begin() + count);
        return *middle;
    };
    const float typical_contrast = median(contrasts), typical_slope = median(slopes);
    if (typical_contrast < .45F * minimum_contrast || typical_slope <= .5F)
        return false;
    // Local reflectance and illumination cancel at a symmetric blurred print transition.
    // Limit individual slopes and offsets so a noisy cell cannot determine a corner.
    cv::Matx<float, 8, 8> system = cv::Matx<float, 8, 8>::zeros();
    cv::Vec<float, 8> rhs = cv::Vec<float, 8>::all(0), update;
    for (int i = 0; i < count; ++i) {
        const auto& sample = samples[i];
        const float weight = std::clamp(sample.contrast, 0.F, 1.8F * typical_contrast);
        if (!weight)
            continue;
        const float slope = std::clamp(sample.slope, .5F * typical_slope, 2 * typical_slope);
        const float offset = std::clamp(sample.residual / slope, -2.F, 2.F);
        for (int v = 0; v < 8; ++v) {
            const float weighted = weight * sample.jacobian[v];
            rhs[v] += weighted * offset;
            for (int j = 0; j <= v; ++j)
                system(v, j) += weighted * sample.jacobian[j];
        }
    }
    for (int v = 0; v < 8; ++v)
        for (int j = 0; j < v; ++j)
            system(j, v) = system(v, j);
    if (!cv::solve(system, rhs, update, cv::DECOMP_CHOLESKY))
        return false;
    max_step = 0;
    for (auto& p : quad) {
        const auto t = (p - center) / side;
        const float z = update[6] * t.x + update[7] * t.y;
        Point step{update[0] * t.x + update[1] * t.y + update[2] - t.x * z,
                   update[3] * t.x + update[4] * t.y + update[5] - t.y * z};
        if (!(std::isfinite(step.x) && std::isfinite(step.y)))
            return false;
        p += step;
        max_step = std::max(max_step, length(step));
    }
    return true;
}

void refine_noisy_geometry(const cv::Mat& gray, Detection& tag, const ReadNoiseConfig& noise,
                           float maximum_reseed, cv::Mat& filtered, const Quad* retry_guide) {
    const auto box = cv::boundingRect(tag.corners_px_);
    const auto roi = expand(box, 12 + std::max(box.height, box.width) / 8, gray.size());
    cv::blur(gray(roi), filtered, {5, 5}, {-1, -1}, cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);
    std::vector<Detection> local{tag};
    for (auto& p : local.front().corners_px_)
        p -= Point(roi.tl());
    if (!fit_noisy_grid(filtered, noise, local.front())) {
        const auto primary = local.front().corners_px_;
        refine_diffuse_tags(filtered, noise.minimum_contrast_, local, maximum_reseed);
        if (retry_guide && local.front().corners_px_ == primary) {
            Detection alternative{tag.id_, *retry_guide, 0};
            for (auto& p : alternative.corners_px_)
                p -= Point(roi.tl());
            // The guide can still localize a print when its half-size crop fit
            // slips onto a false rail. Both ends of the native fit must read
            // the same oriented word without correcting any payload bit.
            const auto exact = [&](const Quad& corners) {
                return verify_payload(filtered, corners, tag.id_, 26, noise.minimum_contrast_, 0);
            };
            if (!exact(primary) && exact(alternative.corners_px_) &&
                fit_noisy_grid(filtered, noise, alternative))
                local.front() = alternative;
        }
    }
    tag = local.front();
    for (auto& p : tag.corners_px_)
        p += Point(roi.tl());
}

}  // namespace

bool fit_noisy_grid(const cv::Mat& filtered, const ReadNoiseConfig& noise, Detection& tag) {
    if (noise.maximum_grid_shift_cells_ <= 0)
        return false;
    std::array<PrintedCrossing, kMaximumCrossings> sites;
    const int count = printed_crossings(tag.id_, sites);
    if (!count)
        return false;
    auto quad = tag.corners_px_;
    const auto center = quad_center(quad);
    float side = 0;
    for (int j = 0; j < 4; ++j)
        side += .25F * length(quad[j] - quad[(j + 1) % 4]);
    if (!(side >= 24 && std::isfinite(side)))
        return false;
    for (int iteration = 0; iteration < 2; ++iteration) {
        float step;
        if (!noisy_grid_step(filtered, sites, count, noise.minimum_contrast_, center, side, quad,
                             step))
            return false;
        if (step < .08F)
            break;
    }
    const float limit = noise.maximum_grid_shift_cells_ * .125F * side;
    for (int j = 0; j < 4; ++j)
        if (length(quad[j] - tag.corners_px_[j]) > limit)
            return false;
    const auto errors = verify_payload(filtered, quad, tag.id_, 26, noise.minimum_contrast_,
                                       std::min(1, tag.corrected_bits_));
    if (!errors)
        return false;
    tag.corners_px_ = quad;
    tag.corrected_bits_ = *errors;
    return true;
}

void NoisyInk::validate(const ReadNoiseConfig& noise) {
    if (noise.minimum_residual_per_mille_ < 1 || noise.minimum_residual_per_mille_ > 1000 ||
        noise.extra_scene_span_ < 0 || noise.extra_scene_span_ > 255 ||
        noise.guide_difference_ < 1 || noise.guide_difference_ > 255 ||
        noise.minimum_contrast_ < 1 || noise.minimum_contrast_ > 255 ||
        !(noise.minimum_hull_retention_ > 0 && noise.minimum_hull_retention_ <= 1) ||
        !(noise.maximum_grid_shift_cells_ >= 0 && noise.maximum_grid_shift_cells_ <= 1) ||
        !(noise.crop_blur_sigma_ > 0 && std::isfinite(noise.crop_blur_sigma_)))
        throw std::invalid_argument("Invalid read-noise ink search settings");
}

bool NoisyInk::plausible(const cv::Mat& gray, const VideoConfig& settings,
                         const ReadNoiseConfig& noise, int row_step, int column_step, int count,
                         int rough, int raw_span) {
    if (!noise.enabled_ || !settings.maximum_isolated_regions_ || !settings.faint_guide_side_px_ ||
        !count ||
        std::int64_t{rough} * 1000 <
                std::int64_t{count} * 3 * settings.smooth_max_rough_per_mille_ ||
        raw_span > settings.border_surround_maximum_span_ + noise.extra_scene_span_)
        return false;
    LumaDistribution::Bins histogram{};
    std::array<int, 4> population{}, high_pass{};
    std::int64_t residual = 0;
    for (int y = 4; y + 3 < gray.rows; y += row_step) {
        const auto* a = gray.ptr<std::uint8_t>(y - 1);
        const auto* b = gray.ptr<std::uint8_t>(y);
        const auto* c = gray.ptr<std::uint8_t>(y + 1);
        for (int x = 4; x + 3 < gray.cols; x += column_step) {
            const int mean = (a[x - 1] + a[x] + a[x + 1] + b[x - 1] + b[x] + b[x + 1] + c[x - 1] +
                              c[x] + c[x + 1] + 4) /
                             9;
            ++histogram[mean];
            residual += std::abs(b[x] - mean);
            const int quadrant = 2 * (y >= gray.rows / 2) + (x >= gray.cols / 2);
            ++population[quadrant];
            high_pass[quadrant] += std::abs(4 * b[x] - b[x - 1] - b[x + 1] - a[x] - c[x]) >=
                                   4 * settings.faint_decode_contrast_;
        }
    }
    for (int k = 0; k < 4; ++k)
        if (!population[k] ||
            std::int64_t{high_pass[k]} * 1000 <
                    std::int64_t{population[k]} * noise.minimum_residual_per_mille_)
            return false;
    // Uniform illumination with only camera noise has little span after averaging.
    const LumaDistribution luma(histogram);
    const int span = luma.percentile(90) - luma.percentile(10);
    return span >= settings.faint_decode_contrast_ &&
           span <= settings.border_surround_maximum_span_ &&
           std::int64_t{span} * count > 2 * residual;
}

NoisyInk::Proposals NoisyInk::find(const cv::Mat& gray, const VideoConfig& settings,
                                   const ReadNoiseConfig& noise, cv::Mat* area4,
                                   bool supplied_current) {
    if (area4 && !supplied_current)
        area4->release();
    Proposals result;
    auto& tags = result.ordinary_;
    if (!noise.enabled_ || !settings.maximum_isolated_regions_ || !settings.faint_guide_side_px_ ||
        std::min(gray.cols, gray.rows) < 16)
        return result;
    const double reduction =
            std::min(1., double(settings.faint_guide_side_px_) / std::min(gray.cols, gray.rows));
    const cv::Mat* current = &guide_;
    if (reduction == .25 && gray.rows % 4 == 0 && gray.cols % 4 == 0) {
        if (area4 && !area4->empty())
            current = area4;
        else {
            average_gray4x4(gray, guide_);
            if (area4)
                *area4 = guide_;
        }
    } else {
        cv::resize(gray, guide_, {}, reduction, reduction, cv::INTER_AREA);
    }
    if (std::min(current->cols, current->rows) < 7)
        return result;
    local_ink_mask(*current, settings.faint_background_window_, noise.guide_difference_,
                   local_mean_, mask_);
    median_binary3(this->mask_, this->median_);
    std::vector<std::vector<cv::Point>> contours;
    find_guide_contours(this->median_, this->guide_contours_, contours);
    const Point scale{float(gray.cols) / current->cols, float(gray.rows) / current->rows};
    const int radius = std::max(3, cvCeil(1.5F * std::max(scale.x, scale.y)));
    struct OutlineRetry {
        Detection tag;
        Quad seed;
    };
    std::vector<OutlineRetry> retries;
    for (const auto& contour : contours) {
        const auto box = cv::boundingRect(contour);
        if (std::min(box.width, box.height) < kMinimumGuideContourSide)
            continue;
        Quad seed;
        Detection tag;
        std::optional<Quad> retry;
        const auto guide_code = noisy_seed(*current, contour, noise, seed);
        if (!guide_code.supported ||
            !read_noisy_crop(gray, seed, box, scale, radius, guide_code, float(1. / reduction),
                             noise, this->crop_filter_, filtered_, tag, retry))
            continue;
        if (retry) {
            retries.push_back({tag, *retry});
            continue;
        }
        const auto& quad = tag.corners_px_;
        if (std::none_of(tags.begin(), tags.end(), [&](const Detection& other) {
                return length(quad_center(other.corners_px_) - quad_center(quad)) <
                       .2F * length(quad[2] - quad[0]);
            }))
            tags.push_back(tag);
    }
    if (settings.refine_diffuse_edges_)
        for (auto& tag : tags)
            refine_noisy_geometry(gray, tag, noise, settings.maximum_diffuse_reseed_cells_,
                                  filtered_, nullptr);

    // Retry proposals are additive. A noisy outline must not displace a source
    // observation that survived the original search.
    for (auto& entry : retries) {
        if (settings.refine_diffuse_edges_)
            refine_noisy_geometry(gray, entry.tag, noise, settings.maximum_diffuse_reseed_cells_,
                                  filtered_, &entry.seed);
        result.outline_retries_.push_back(entry.tag);
    }
    return result;
}

}  // namespace aruco3cuda::apriltag_small_cpu::detail
