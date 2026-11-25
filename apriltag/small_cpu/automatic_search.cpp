// SPDX-License-Identifier: Apache-2.0
#include "automatic_search.hpp"

#include "luma_distribution.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#if defined(__aarch64__) && defined(__linux__) && __has_include(<arm_sve.h>)
#include <arm_sve.h>
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace aruco3cuda::apriltag_small_cpu::detail {
namespace {

constexpr int kRowStep = 13;
constexpr int kColumnStep = 17;
constexpr int kImageSideLimit = 32768;

struct DenseHint {
    int count_ = 0;
    int left_ = kImageSideLimit;
    int right_ = 0;
    int top_ = kImageSideLimit;
    int bottom_ = 0;
    std::int64_t low_sum_ = 0;
    std::int64_t high_sum_ = 0;

#if defined(__aarch64__) && defined(__linux__) && __has_include(<arm_sve.h>)
    __attribute__((target("arch=armv8.2-a+sve"))) void sample_sve(const cv::Mat& gray, int y,
                                                                  int minimum_contrast) {
        constexpr int kFirstColumn = 10, kDistance = 7;
        auto* row = gray.ptr<uchar>(y);
        auto* before = gray.ptr<uchar>(y - kDistance);
        auto* after = gray.ptr<uchar>(y + kDistance);
        const auto ix = svindex_u32(0, kColumnStep);
        const int group = svcntw() * kColumnStep;
        for (int x = kFirstColumn; x + kDistance < gray.cols; x += group) {
            const auto pg = svcmplt_n_u32(svptrue_b32(), ix, gray.cols - kDistance - x);
            auto cen = svld1ub_gather_u32offset_u32(pg, row + x, ix);
            auto l = svld1ub_gather_u32offset_u32(pg, row + x - kDistance, ix),
                 r = svld1ub_gather_u32offset_u32(pg, row + x + kDistance, ix);
            auto a = svld1ub_gather_u32offset_u32(pg, before + x, ix),
                 b = svld1ub_gather_u32offset_u32(pg, after + x, ix);
            auto min1 = svmin_u32_x(pg, l, r), max1 = svmax_u32_x(pg, l, r),
                 min2 = svmin_u32_x(pg, a, b), max2 = svmax_u32_x(pg, a, b);
            auto below = svmax_u32_x(pg, min1, min2), above = svmin_u32_x(pg, max1, max2);
            auto check = svmax_u32_x(pg, svqsub_u32(below, cen), svqsub_u32(cen, above));
            auto active = svcmpge_n_u32(pg, check, minimum_contrast);
            if (!svptest_any(pg, active))
                continue;
            count_ += svcntp_b32(pg, active);
            low_sum_ += svaddv_u32(active, svmin_u32_x(pg, cen, svmin_u32_x(pg, min1, min2)));
            high_sum_ += svaddv_u32(active, svmax_u32_x(pg, cen, svmax_u32_x(pg, max1, max2)));
            left_ = std::min(left_, x + int(svminv_u32(active, ix)));
            right_ = std::max(right_, x + int(svmaxv_u32(active, ix)));
            top_ = std::min(top_, y);
            bottom_ = y;
        }
    }
#endif
    void sample_row(const cv::Mat& gray, int y, int minimum_contrast) {
        constexpr int kFirstColumn = 10;
        constexpr int kDistance = 7;
        const auto* row = gray.ptr<std::uint8_t>(y);
        const auto* before = gray.ptr<std::uint8_t>(y - kDistance);
        const auto* after = gray.ptr<std::uint8_t>(y + kDistance);

#if defined(__aarch64__) && defined(__linux__) && __has_include(<arm_sve.h>)
        static const bool sve = (getauxval(AT_HWCAP) & HWCAP_SVE) != 0;
        if (sve && gray.cols >= 1200) {
            sample_sve(gray, y, minimum_contrast);
            return;
        }
#endif
        for (int x = kFirstColumn; x + kDistance < gray.cols; x += kColumnStep) {
            const int center = row[x];
            const int left = row[x - kDistance], right = row[x + kDistance];
            const int above = before[x], below = after[x];
            const int horizontal =
                    std::max(std::min(left, right) - center, center - std::max(left, right));
            const int vertical =
                    std::max(std::min(above, below) - center, center - std::max(above, below));
            if (std::max(horizontal, vertical) < minimum_contrast)
                continue;
            ++this->count_;
            this->low_sum_ += std::min({center, left, right, above, below});
            this->high_sum_ += std::max({center, left, right, above, below});
            this->left_ = std::min(this->left_, x);
            this->right_ = std::max(this->right_, x);
            this->top_ = std::min(this->top_, y);
            this->bottom_ = y;
        }
    }

    cv::Rect region(const cv::Mat& gray, const DenseRegionConfig& config) const {
        if (this->count_ < config.minimum_sample_count_)
            return {};
        const std::int64_t extent = std::int64_t{this->right_ - this->left_ + kColumnStep} *
                                    (this->bottom_ - this->top_ + kRowStep);
        const std::int64_t frame_area = std::int64_t{gray.cols} * gray.rows;
        if (extent * config.minimum_density_per_mille_ >
                    std::int64_t{this->count_} * kColumnStep * kRowStep * 1000 ||
            extent * 1000 > frame_area * config.maximum_extent_per_mille_)
            return {};
        const int pad = config.padding_px_;
        const cv::Rect bounds(this->left_ - pad, this->top_ - pad,
                              this->right_ - this->left_ + 2 * pad + 1,
                              this->bottom_ - this->top_ + 2 * pad + 1);
        const auto crop = bounds & cv::Rect(0, 0, gray.cols, gray.rows);
        if (static_cast<std::size_t>(crop.width) * crop.height > config.maximum_roi_pixels_)
            return {};
        return crop;
    }

    int low() const { return static_cast<int>(this->low_sum_ / this->count_); }
    int high() const { return static_cast<int>(this->high_sum_ / this->count_); }
};

struct PitchSummary {
    std::int64_t transitions_ = 0;
    std::int64_t short_runs_ = 0;
    int rows_ = 0;
};

PitchSummary count_transitions(const cv::Mat& gray, const cv::Rect& roi, int low, int high) {
    constexpr int kRowOffset = 2;
    constexpr int kPitchRowStep = 11;
    constexpr int kLongestCellRun = 24;
    constexpr int kShortCellRun = 2;
    constexpr int kRowSupport = 14;
    const int level = (high + low) / 2;
    const int contrast = (high - low) * 45 / 100;
    PitchSummary result;
    for (int y = roi.y + kRowOffset; y < roi.y + roi.height; y += kPitchRowStep) {
        const auto* row = gray.ptr<std::uint8_t>(y);
        int start = roi.x, peak = 0, previous_peak = 0, count = 0;
        bool light = row[start] > level;
        for (int x = roi.x; x < roi.x + roi.width; ++x) {
            const bool next_light = row[x] > level;
            if (next_light != light) {
                if (x - start <= kLongestCellRun && peak + previous_peak >= contrast) {
                    ++result.transitions_;
                    result.short_runs_ += x - start <= kShortCellRun;
                    ++count;
                }
                previous_peak = peak;
                peak = 0;
                start = x;
                light = next_light;
            }
            peak = std::max(peak, std::abs(row[x] - level));
        }
        result.rows_ += count >= kRowSupport;
    }
    return result;
}

struct LumaSummary {
    LumaDistribution luma_;
    int rough_;
    int flat_;

    int per_mille(int count) const {
        return static_cast<int>(std::int64_t{count} * 1000 / this->luma_.total());
    }
};

using HistogramLanes = std::array<LumaDistribution::Bins, 4>;

LumaSummary summarize_lanes(HistogramLanes& lanes, int rough, int flat) {
    auto& counts = lanes.front();
    for (std::size_t i = 0; i < counts.size(); ++i)
        counts[i] += lanes[1][i] + lanes[2][i] + lanes[3][i];
    return {LumaDistribution(counts), rough, flat};
}

#if defined(__aarch64__) && defined(__linux__) && __has_include(<arm_sve.h>)
__attribute__((target("arch=armv8.2-a+sve"))) LumaSummary summarize_sve(const cv::Mat& gray,
                                                                        DenseHint* hint,
                                                                        int minimum_contrast) {
    constexpr int kOffset = 4, kProbeDistance = 3, kRoughDifference = 16, kFlatDifference = 2;
    HistogramLanes histogram{};
    const int w = (gray.cols - 8) / kColumnStep + 1;
    std::vector<uchar> samples(w);
    const auto ix = svindex_u32(0, kColumnStep);
    auto rough = svdup_u32(0), flat = svdup_u32(0);
    for (int y = kOffset; y + kProbeDistance < gray.rows; y += kRowStep) {
        const auto *r = gray.ptr<uchar>(y) + kOffset,
                   *a = gray.ptr<uchar>(y + kProbeDistance) + kOffset;
        for (int i = 0; i < w; i += svcntw()) {
            const auto pg = svwhilelt_b32(i, w);
            const auto z = svld1ub_gather_u32offset_u32(pg, r + i * kColumnStep, ix),
                       u = svld1ub_gather_u32offset_u32(pg, r + i * kColumnStep + kProbeDistance,
                                                        ix),
                       v = svld1ub_gather_u32offset_u32(pg, a + i * kColumnStep, ix);
            auto diff = svmax_u32_x(pg, svabd_u32_x(pg, z, u), svabd_u32_x(pg, z, v));
            rough = svadd_n_u32_m(svcmpge_n_u32(pg, diff, kRoughDifference), rough, 1);
            flat = svadd_n_u32_m(svcmple_n_u32(pg, diff, kFlatDifference), flat, 1);
            svst1b_u32(pg, samples.data() + i, z);
        }
        int i = 0;
        for (; i + 3 < w; i += 4) {
            ++histogram[0][samples[i]];
            ++histogram[1][samples[i + 1]];
            ++histogram[2][samples[i + 2]];
            ++histogram[3][samples[i + 3]];
        }
        for (; i < w; ++i)
            ++histogram[0][samples[i]];
        if (hint && y + 13 < gray.rows)
            hint->sample_row(gray, y + 6, minimum_contrast);
    }
    return summarize_lanes(histogram, static_cast<int>(svaddv_u32(svptrue_b32(), rough)),
                           static_cast<int>(svaddv_u32(svptrue_b32(), flat)));
}
#endif
LumaSummary summarize(const cv::Mat& gray, DenseHint* hint = nullptr, int minimum_contrast = 0) {
    constexpr int kOffset = 4;
    constexpr int kProbeDistance = 3;
    constexpr int kRoughDifference = 16;
    constexpr int kFlatDifference = 2;
    int rough = 0, flat = 0;
    HistogramLanes histogram{};

#if defined(__aarch64__) && defined(__linux__) && __has_include(<arm_sve.h>)
    static const bool sve = (getauxval(AT_HWCAP) & HWCAP_SVE) != 0;
    if (gray.cols >= 1200 && gray.rows >= 600 && sve)
        return summarize_sve(gray, hint, minimum_contrast);
#endif
    for (int y = kOffset; y + kProbeDistance < gray.rows; y += kRowStep) {
        const auto* row = gray.ptr<std::uint8_t>(y);
        const auto* next = gray.ptr<std::uint8_t>(y + kProbeDistance);
        int x = kOffset;
        auto sample = [&](int xx, int k) {
            const int value = row[xx];
            const int difference = std::max(std::abs(value - row[xx + kProbeDistance]),
                                            std::abs(value - next[xx]));
            ++histogram[k][value];
            rough += difference >= kRoughDifference;
            flat += difference <= kFlatDifference;
        };
        for (; x + 3 * kColumnStep + kProbeDistance < gray.cols; x += 4 * kColumnStep) {
            sample(x, 0);
            sample(x + kColumnStep, 1);
            sample(x + 2 * kColumnStep, 2);
            sample(x + 3 * kColumnStep, 3);
        }
        for (; x + kProbeDistance < gray.cols; x += kColumnStep)
            sample(x, 0);
        constexpr int kDenseOffset = 6;
        constexpr int kDenseRadius = 7;
        if (hint && y + kDenseOffset + kDenseRadius < gray.rows)
            hint->sample_row(gray, y + kDenseOffset, minimum_contrast);
    }
    return summarize_lanes(histogram, rough, flat);
}

Config child_config(const Config& config, AutomaticRoute route) {
    Config child = config;
    child.automatic_.reset();
    child.border_.reset();
    child.recovery_.reset();
    switch (route) {
        case AutomaticRoute::kNative:
            break;
        case AutomaticRoute::kCoarse:
            child.coarse_ = config.automatic_->coarse_;
            break;
        case AutomaticRoute::kComponents:
            child.components_ = config.automatic_->components_;
            break;
    }
    return child;
}

float squared_side(const Detection& tag) {
    const auto side = tag.corners_px_[1] - tag.corners_px_[0];
    return side.dot(side);
}

// Ratios apply to the squared distance between twice the diagonal midpoints.
bool is_duplicate(const Detection& tag, const Detection& other, float distance_ratio_squared) {
    const auto difference =
            tag.corners_px_[0] + tag.corners_px_[2] - other.corners_px_[0] - other.corners_px_[2];
    return tag.id_ == other.id_ &&
           difference.dot(difference) < distance_ratio_squared * squared_side(tag);
}

void merge_unique(std::vector<Detection>& base, const std::vector<Detection>& extra,
                  float distance_ratio_squared) {
    for (const auto& tag : extra) {
        const auto duplicate = std::find_if(
                base.begin(), base.end(), [&tag, distance_ratio_squared](const auto& other) {
                    return is_duplicate(tag, other, distance_ratio_squared);
                });
        if (duplicate == base.end())
            base.push_back(tag);
    }
}

}  // namespace

void validate_automatic_config(const AutomaticConfig& config) {
    const auto fraction_valid = [](int fraction) { return fraction >= 0 && fraction <= 1000; };
    const auto& dense = config.dense_region_;
    constexpr std::size_t kMaximumRegionPixels = 16U * 1024U * 1024U;
    if (dense.minimum_local_contrast_ < 1 || dense.minimum_local_contrast_ > 255 ||
        dense.minimum_sample_count_ < 1 || !fraction_valid(dense.minimum_density_per_mille_) ||
        dense.maximum_extent_per_mille_ < 1 || !fraction_valid(dense.maximum_extent_per_mille_) ||
        dense.padding_px_ < 0 || dense.padding_px_ >= kImageSideLimit ||
        dense.maximum_roi_pixels_ == 0 || dense.maximum_roi_pixels_ > kMaximumRegionPixels ||
        dense.maximum_upsample_source_pixels_ > kMaximumRegionPixels / 4 ||
        dense.upsample_sharpen_percent_ < 0 || dense.upsample_sharpen_percent_ > 25 ||
        dense.minimum_verified_tags_ < 1 ||
        dense.replace_minimum_tags_ < dense.minimum_verified_tags_) {
        throw std::invalid_argument("Invalid dense AprilTag region parameters");
    }
    if (config.native_low_percentile_floor_ < 0 ||
        config.native_high_percentile_ceiling_ < config.native_low_percentile_floor_ ||
        config.native_high_percentile_ceiling_ > 255 || config.component_min_luma_span_ < 0 ||
        config.component_min_luma_span_ > 255 ||
        !fraction_valid(config.native_max_dark_per_mille_) ||
        !fraction_valid(config.component_min_rough_per_mille_) ||
        !fraction_valid(config.component_max_flat_per_mille_) ||
        !fraction_valid(config.component_max_bright_per_mille_) ||
        config.dense_small_tag_count_ < 0 ||
        !(config.dense_small_tag_side_px_ > 0 && config.dense_small_tag_side_px_ < 32768)) {
        throw std::invalid_argument("Invalid automatic AprilTag search parameters");
    }
}

static AutomaticRoute choose_route(const LumaSummary& stats, const AutomaticConfig& config) {
    const auto& luma = stats.luma_;
    if (!luma.total())
        return AutomaticRoute::kNative;

    constexpr int kDarkLuma = 115, kBrightLuma = 218;
    const int low = luma.percentile(10);
    const int high = luma.percentile(90);
    const int dark = stats.per_mille(luma.below(kDarkLuma + 1));
    const int flat = stats.per_mille(stats.flat_);

    // Sparse foreground on a bright or almost uniform surround favors native sampling.
    if ((low >= config.native_low_percentile_floor_ &&
         high <= config.native_high_percentile_ceiling_ &&
         dark <= config.native_max_dark_per_mille_) ||
        (low == high && flat >= 990 && dark <= 3)) {
        return AutomaticRoute::kNative;
    }
    // Broad, locally varying midtones favor the component search. A dominant
    // bright region also occurs around crowded contrasting tags in the coarse search.
    if (high - low >= config.component_min_luma_span_ &&
        stats.per_mille(stats.rough_) >= config.component_min_rough_per_mille_ &&
        flat <= config.component_max_flat_per_mille_ &&
        stats.per_mille(luma.total() - luma.below(kBrightLuma)) <=
                config.component_max_bright_per_mille_) {
        return AutomaticRoute::kComponents;
    }
    return AutomaticRoute::kCoarse;
}

AutomaticRoute choose_automatic_route(const cv::Mat& gray, const AutomaticConfig& config) {
    return choose_route(summarize(gray), config);
}

bool BoardSeedPolicy::dispersed(const std::vector<Detection>& tags) const {
    if (minimum_spacing_sides_ <= 0 || tags.size() < minimum_count_ || tags.size() >= dense_count_)
        return false;
    struct Footprint {
        cv::Point2f center;
        float side;
    };
    std::vector<Footprint> footprints;
    footprints.reserve(tags.size());
    std::size_t small = 0;
    for (std::size_t i = 0; i < tags.size(); ++i) {
        if (std::any_of(tags.begin(), tags.begin() + i,
                        [&](const Detection& previous) { return previous.id_ == tags[i].id_; }))
            return false;
        const auto& q = tags[i].corners_px_;
        const float side =
                std::sqrt(std::max((q[1] - q[0]).dot(q[1] - q[0]), (q[2] - q[1]).dot(q[2] - q[1])));
        footprints.push_back({.25F * (q[0] + q[1] + q[2] + q[3]), side});
        small += side <= maximum_side_px_;
    }
    if (small < minimum_count_)
        return false;
    const float spacing2 = minimum_spacing_sides_ * minimum_spacing_sides_;
    return std::any_of(footprints.begin(), footprints.end(), [&](const Footprint& anchor) {
        return std::none_of(footprints.begin(), footprints.end(), [&](const Footprint& other) {
            if (&anchor == &other)
                return false;
            const auto delta = anchor.center - other.center;
            return delta.dot(delta) <= spacing2 * anchor.side * other.side;
        });
    });
}

AutomaticSearch::AutomaticSearch(const Config& config)
    : config_(*config.automatic_),
      native_(child_config(config, AutomaticRoute::kNative)),
      coarse_(child_config(config, AutomaticRoute::kCoarse)),
      components_(child_config(config, AutomaticRoute::kComponents)) {}

std::vector<Detection> AutomaticSearch::detect(const cv::Mat& gray,
                                               const BoardSeedPolicy* board_owner) {
    const auto& dense = this->config_.dense_region_;
    DenseHint hint;
    const auto luma =
            summarize(gray, dense.enabled_ ? &hint : nullptr, dense.minimum_local_contrast_);
    const auto route = choose_route(luma, this->config_);
    const auto roi = hint.region(gray, dense);
    if (roi.empty())
        return this->detect_route(gray, route, board_owner);

    const int low = hint.low(), high = hint.high();
    const auto pitch = count_transitions(gray, roi, low, high);
    if (pitch.rows_ < 3 || pitch.transitions_ < 95)
        return this->detect_route(gray, route, board_owner);

    const bool upsample = static_cast<std::size_t>(roi.width) * roi.height <=
                                  dense.maximum_upsample_source_pixels_ &&
                          roi.width < kImageSideLimit / 2 && roi.height < kImageSideLimit / 2 &&
                          pitch.short_runs_ * 4 >= pitch.transitions_;
    cv::Mat region = gray(roi);
    const float scale = upsample ? 2.0F : 1.0F;
    if (upsample) {
        if (dense.upsample_sharpen_percent_ != 0) {
            const float weight = dense.upsample_sharpen_percent_ / 100.0F;
            const cv::Matx33f kernel(0, -weight, 0, -weight, 1 + 4 * weight, -weight, 0, -weight,
                                     0);
            cv::filter2D(region, this->sharpened_region_, CV_8U, kernel, {-1, -1}, 0,
                         cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);
            region = this->sharpened_region_;
        }
        cv::resize(region, this->resized_region_, {}, scale, scale, cv::INTER_CUBIC);
        region = this->resized_region_;
    }
    auto result = this->native_.detect_native_with_threshold(region, low + (high - low) * 2 / 3);
    if (result.size() < static_cast<std::size_t>(dense.minimum_verified_tags_))
        return this->detect_route(gray, route, board_owner);
    for (auto& tag : result) {
        for (auto& point : tag.corners_px_) {
            point.x = (point.x + .5F) / scale - .5F + roi.x;
            point.y = (point.y + .5F) / scale - .5F + roi.y;
        }
    }
    if (result.size() >= static_cast<std::size_t>(dense.replace_minimum_tags_))
        return result;
    auto base = this->detect_route(gray, route, board_owner);
    merge_unique(base, result, 0.20F);
    return base;
}

std::vector<Detection> AutomaticSearch::detect_route(const cv::Mat& gray, AutomaticRoute route,
                                                     const BoardSeedPolicy* board_owner) {
    switch (route) {
        case AutomaticRoute::kNative:
            return this->native_.detect(gray);
        case AutomaticRoute::kCoarse:
            return this->coarse_.detect(gray);
        case AutomaticRoute::kComponents:
            break;
    }

    auto components = this->components_.detect(gray);
    if (this->config_.dense_small_tag_count_ == 0)
        return components;

    const float side = this->config_.dense_small_tag_side_px_;
    const auto small = std::count_if(components.begin(), components.end(), [side](const auto& tag) {
        return squared_side(tag) < side * side;
    });
    if (small < this->config_.dense_small_tag_count_ ||
        (board_owner && board_owner->dispersed(components)))
        return components;

    // A frame with several confirmed small tags can justify inspecting other
    // sub-sample phases at native resolution. Keep unique component detections.
    auto native = this->native_.detect(gray);
    merge_unique(native, components, 0.16F);
    return native;
}

}  // namespace aruco3cuda::apriltag_small_cpu::detail
