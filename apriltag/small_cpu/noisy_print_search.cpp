// SPDX-License-Identifier: Apache-2.0
#include "noisy_print_search.hpp"

#include "guide_average.hpp"
#include "isolated_ink.hpp"
#include "tag_codebook.hpp"

#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/imgproc.hpp>

#if defined(__aarch64__) && defined(__AARCH64EL__)
#include <arm_neon.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace aruco3cuda::apriltag_small_cpu::detail {
namespace {
constexpr int kGuideStride = 4;
constexpr int kNativeDecoderStride = 2;

int decoder_coordinate(int block, int native_extent) {
    const int start = block * kGuideStride;
    const int width = std::min(kGuideStride, native_extent - start);
    return cvRound((start + .5F * (width - 1)) / kNativeDecoderStride);
}

struct ConfirmedFootprint {
    Point center;
    float diagonal;
    Quad corners;
};

bool already_measured(const std::vector<cv::Point>& outline,
                      const std::vector<ConfirmedFootprint>& known) {
    if (known.empty())
        return false;
    Quad corners;
    for (int i = 0; i < 4; ++i)
        corners[i] = Point(outline[i] * kNativeDecoderStride);
    if (cross(corners[1] - corners[0], corners[2] - corners[0]) < 0)
        std::reverse(corners.begin(), corners.end());
    const Point center = .25F * (corners[0] + corners[1] + corners[2] + corners[3]);
    const float diagonal =
            std::max(length(corners[0] - corners[2]), length(corners[1] - corners[3]));
    for (const auto& tag : known) {
        if (diagonal < .72F * tag.diagonal || diagonal > 1.35F * tag.diagonal)
            continue;
        const float shorter = std::min(diagonal, tag.diagonal);
        if (length(center - tag.center) >= .14F * shorter)
            continue;
        // Ignore canonical corner rotation, but protect distinct crossing or
        // nested prints that happen to have a similar center and bounding box.
        for (int rotation = 0; rotation < 4; ++rotation) {
            bool aligned = true;
            for (int i = 0; i < 4 && aligned; ++i) {
                const Point delta = corners[i] - tag.corners[(i + rotation) % 4];
                aligned = delta.dot(delta) < .04F * shorter * shorter;
            }
            if (aligned)
                return true;
        }
    }
    return false;
}

std::optional<Detection> read_boundary(const cv::Mat& gray, const std::vector<cv::Point>& outline,
                                       int contrast) {
    Quad q;
    for (int i = 0; i < 4; ++i)
        q[i] = Point(outline[i] * kNativeDecoderStride);
    if (cross(q[1] - q[0], q[2] - q[0]) < 0)
        std::reverse(q.begin(), q.end());
    if (!refine_quad(gray, q, false))
        return {};
    for (const auto p : q)
        if (p.x < 0 || p.y < 0 || p.x >= gray.cols - 1 || p.y >= gray.rows - 1)
            return {};
    const auto word = decode<true>(gray, q, 30, contrast, 0);
    if (word.id < 0)
        return {};
    std::rotate(q.begin(), q.begin() + (4 - word.rotation) % 4, q.end());
    // Verify the canonical order on the measured frame before merging the result.
    if (!verify_payload<true>(gray, q, word.id, 30, contrast, 0))
        return {};
    return Detection{word.id, q, 0};
}

Config decoder_config() {
    Config config;
    config.components_.emplace();
    config.components_->grid_radius_px_ = 4;
    return config;
}

bool has_distributed_noise(const cv::Mat& gray, const NoisyPrintConfig& config) {
    if (std::min(gray.rows, gray.cols) < 128)
        return false;
    // Within white read noise, a 2x2 neighborhood carries most of the variance
    // of the enclosing 6x6 patch. Slow shading and long rails do not have this property.
    constexpr int kRows = 14, kColumns = 20;
    constexpr int kCount = kRows * kColumns;
    std::array<int, kCount> energies{};
    int visited = 0;
    const int required = (kCount * config.minimum_noisy_patches_per_mille_ + 999) / 1000;
    int votes = 0, dark_patches = 0;
    for (int row = 0; row < kRows; ++row) {
        const int y = 3 + row * (gray.rows - 13) / (kRows - 1);
        for (int column = 0; column < kColumns; ++column) {
            const int x = 3 + column * (gray.cols - 13) / (kColumns - 1);
            int sum = 0, squares = 0, within = 0;
#if defined(__aarch64__) && defined(__AARCH64EL__)
            auto sums_by_pair = vdup_n_u16(0);
            auto squares_by_pair = vdupq_n_u32(0);
            auto within_by_pair = vdupq_n_u32(0);
            for (int dy = 0; dy < 6; dy += 2) {
                // The sampling grid leaves at least ten columns to the right.
                // The fourth two-pixel group is read with the other groups but
                // excluded from every result below.
                const auto a = vld1_u8(gray.ptr<std::uint8_t>(y + dy) + x);
                const auto b = vld1_u8(gray.ptr<std::uint8_t>(y + dy + 1) + x);
                const auto pair_sum = vadd_u16(vpaddl_u8(a), vpaddl_u8(b));
                const auto pair_squares =
                        vaddq_u32(vpaddlq_u16(vmull_u8(a, a)), vpaddlq_u16(vmull_u8(b, b)));
                sums_by_pair = vadd_u16(sums_by_pair, pair_sum);
                squares_by_pair = vaddq_u32(squares_by_pair, pair_squares);
                within_by_pair = vaddq_u32(within_by_pair, vmlsl_u16(vshlq_n_u32(pair_squares, 2),
                                                                     pair_sum, pair_sum));
            }
            sum = static_cast<int>(vaddv_u16(sums_by_pair) - vget_lane_u16(sums_by_pair, 3));
            squares = static_cast<int>(vaddvq_u32(squares_by_pair) -
                                       vgetq_lane_u32(squares_by_pair, 3));
            within = static_cast<int>(vaddvq_u32(within_by_pair) -
                                      vgetq_lane_u32(within_by_pair, 3));
#else
            for (int dy = 0; dy < 6; dy += 2) {
                const auto* top = gray.ptr<std::uint8_t>(y + dy) + x;
                const auto* bottom = gray.ptr<std::uint8_t>(y + dy + 1) + x;
                for (int dx = 0; dx < 6; dx += 2) {
                    const int a = top[dx], b = top[dx + 1];
                    const int c = bottom[dx], d = bottom[dx + 1];
                    const int local_sum = a + b + c + d;
                    const int local_squares = a * a + b * b + c * c + d * d;
                    within += 4 * local_squares - local_sum * local_sum;
                    sum += local_sum;
                    squares += local_squares;
                }
            }
#endif
            const int total = 36 * squares - sum * sum;
            votes += within >= 70 && 9 * within > .55 * total;
            dark_patches += sum < 80 * 36;
            energies[visited++] = within;
        }
        if (votes + kCount - visited < required ||
            dark_patches * 1000 > kCount * config.maximum_dark_patches_per_mille_)
            return false;
    }
    // Nine four-pixel groups per patch contribute three independent differences each.
    // The histogram uses half-gray-level bins for the estimated noise standard deviation.
    std::array<int, 64> noise_bins{};
    for (int energy : energies) {
        const auto bin = static_cast<int>(2 * std::sqrt(float(energy) / 108));
        ++noise_bins[std::min(63, bin)];
    }
    const auto quantile = [&](int percent) {
        int count = 0;
        for (std::size_t i = 0; i < noise_bins.size(); ++i) {
            count += noise_bins[i];
            if (count * 100 >= percent * kCount)
                return static_cast<int>(i);
        }
        return 63;
    };
    const int median = quantile(50);
    return median >= 2 * config.minimum_noise_sigma_ && 3 * quantile(10) >= median &&
           quantile(75) < 2 * median + 2;
}

struct TransitionCost {
    double incumbent = 0;
    double proposed = 0;
    int visible = 0;
    int supported = 0;
};

double crossing_error(float dark, float middle, float light, float minimum_contrast) {
    if (light - dark < minimum_contrast)
        return .7;
    return std::min(.7, std::abs(middle - (dark + light) * .5) / (light - dark));
}

// Both hypotheses use precisely the same visible, signed ink-to-paper transitions.
// A hypothesis with insufficient local contrast pays the maximum residual.
void compare_crossing(const cv::Mat& gray, const Projection& current, const Projection& proposed,
                      float u, float v, float du, float dv, float minimum_contrast,
                      TransitionCost& cost) {
    const float dark = current.sample(gray, u - du, v - dv),
                light = current.sample(gray, u + du, v + dv);
    const float middle = current.sample(gray, u, v);
    const float new_dark = proposed.sample(gray, u - du, v - dv),
                new_light = proposed.sample(gray, u + du, v + dv);
    const float new_middle = proposed.sample(gray, u, v);
    if (std::min({dark, light, middle, new_dark, new_light, new_middle}) < 0)
        return;
    cost.incumbent += crossing_error(dark, middle, light, minimum_contrast);
    cost.proposed += crossing_error(new_dark, new_middle, new_light, minimum_contrast);
    cost.supported += new_light - new_dark >= minimum_contrast;
    ++cost.visible;
}

bool prefer_refined(const cv::Mat& gray, const Detection& current, const Detection& refined,
                    float minimum_contrast) {
    double displacement = 0, margin = std::numeric_limits<double>::max();
    for (int i = 0; i < 4; ++i) {
        const Point p = refined.corners_px_[i];
        displacement = std::max(displacement, cv::norm(p - current.corners_px_[i]));
        margin = std::min(
                margin, double(std::min({p.x, p.y, gray.cols - 1.F - p.x, gray.rows - 1.F - p.y})));
    }
    const double side = cv::norm(refined.corners_px_[0] - refined.corners_px_[1]);
    if (displacement < .05 || side < 30 || margin < std::max(2., .07 * side))
        return false;
    Projection original, candidate;
    if (!project(current.corners_px_, original) || !project(refined.corners_px_, candidate))
        return false;

    TransitionCost outer;
    const float offset = std::clamp(float(3.5 / side), .012F, .045F);
    for (int i = 0; i < 24; ++i) {
        const float position = .07F + .86F * (i + .5F) / 24;
        compare_crossing(gray, original, candidate, 0, position, -offset, 0, minimum_contrast,
                         outer);
        compare_crossing(gray, original, candidate, 1, position, offset, 0, minimum_contrast,
                         outer);
        compare_crossing(gray, original, candidate, position, 0, 0, -offset, minimum_contrast,
                         outer);
        compare_crossing(gray, original, candidate, position, 1, 0, offset, minimum_contrast,
                         outer);
    }
    if (outer.visible < 84 || outer.supported < .85 * outer.visible)
        return false;
    const auto word = generated::kDictApriltag36h11Table.codes_[4 * refined.id_];
    const auto white = [&](int x, int y) {
        return x >= 1 && x <= 6 && y >= 1 && y <= 6 && ((word >> (6 * (y - 1) + x - 1)) & 1U);
    };
    TransitionCost payload;
    for (int y = 0; y < kBorderSide; ++y) {
        for (int x = 0; x < kBorderSide; ++x) {
            const bool first = white(x, y);
            const float direction = first ? -.035F : .035F;
            for (float t : {.3F, .5F, .7F}) {
                if (x + 1 < kBorderSide && first != white(x + 1, y))
                    compare_crossing(gray, original, candidate, float(x + 1) / kBorderSide,
                                     (y + t) / kBorderSide, direction, 0, minimum_contrast,
                                     payload);
                if (y + 1 < kBorderSide && first != white(x, y + 1))
                    compare_crossing(gray, original, candidate, (x + t) / kBorderSide,
                                     float(y + 1) / kBorderSide, 0, direction, minimum_contrast,
                                     payload);
            }
        }
    }
    if (payload.visible < 50 || payload.supported < .85 * payload.visible)
        return false;
    const double before = outer.incumbent / outer.visible + payload.incumbent / payload.visible;
    const double after = outer.proposed / outer.visible + payload.proposed / payload.visible;
    return after < before &&
           (displacement < .35 * side / kBorderSide || payload.proposed < .8 * payload.incumbent);
}
// Every 5x5 dilated value is bounded by extrema of an enclosing guide tile.
bool can_contain_dark_transition([[maybe_unused]] const cv::Mat& guide, [[maybe_unused]] int cutoff,
                                 [[maybe_unused]] int contrast) {
#if CV_SIMD128
    if (guide.isSubmatrix())
        return true;
    constexpr int kTileWidth = 96, kTileHeight = 48, kRadius = 2;
    for (int y = 0; y < guide.rows; y += kTileHeight) {
        const int top = std::max(0, y - kRadius),
                  bottom = std::min(guide.rows, y + kTileHeight + kRadius);
        for (int x = 0; x < guide.cols; x += kTileWidth) {
            const int left = std::max(0, x - kRadius),
                      right = std::min(guide.cols, x + kTileWidth + kRadius);
            auto lo = cv::v_setall_u8(255), hi = cv::v_setzero_u8();
            int minimum = 255, maximum = 0;
            for (int j = top; j < bottom; ++j) {
                const auto* row = guide.ptr<std::uint8_t>(j);
                if (right - left >= 16) {
                    for (int k = left; k + 16 <= right; k += 16) {
                        auto a = cv::v_load(row + k);
                        lo = cv::v_min(lo, a);
                        hi = cv::v_max(hi, a);
                    }
                    auto a = cv::v_load(row + right - 16);
                    lo = cv::v_min(lo, a);
                    hi = cv::v_max(hi, a);
                } else {
                    for (int k = left; k < right; ++k) {
                        minimum = std::min(minimum, int(row[k]));
                        maximum = std::max(maximum, int(row[k]));
                    }
                }
            }
            minimum = std::min(minimum, int(cv::v_reduce_min(lo)));
            maximum = std::max(maximum, int(cv::v_reduce_max(hi)));
            if (minimum <= cutoff && maximum - minimum > contrast)
                return true;
        }
    }
    return false;
#else
    return true;
#endif
}
void build_noisy_mask(const cv::Mat& guide, const cv::Mat& maximum, cv::Mat& mask, int cutoff,
                      int contrast) {
    mask.create(guide.size(), CV_8UC1);
#if CV_SIMD128
    const auto low = cv::v_setall_u8(static_cast<std::uint8_t>(cutoff));
    const auto difference = cv::v_setall_u8(static_cast<std::uint8_t>(contrast));
#endif
    for (int y = 0; y < guide.rows; ++y) {
        const auto* source = guide.ptr<std::uint8_t>(y);
        const auto* peak = maximum.ptr<std::uint8_t>(y);
        auto* destination = mask.ptr<std::uint8_t>(y);
        int x = 0;
#if CV_SIMD128
        for (; x + 16 <= guide.cols; x += 16) {
            const auto a = cv::v_load(source + x);
            const auto b = cv::v_load(peak + x);
            const auto enabled = cv::v_and(cv::v_le(a, low), cv::v_lt(difference, cv::v_sub(b, a)));
            cv::v_store(destination + x, enabled);
        }
#endif
        for (; x < guide.cols; ++x)
            destination[x] = source[x] <= cutoff && peak[x] - source[x] > contrast ? 255 : 0;
    }
}

}  // namespace

NoisyPrintSearch::NoisyPrintSearch(const NoisyPrintConfig& config)
    : config_(config), decoder_(decoder_config()) {
    if (config.minimum_noisy_patches_per_mille_ < 0 ||
        config.minimum_noisy_patches_per_mille_ > 1000 ||
        config.maximum_dark_patches_per_mille_ < 0 ||
        config.maximum_dark_patches_per_mille_ > 1000 ||
        !(config.minimum_noise_sigma_ >= 0.F && config.minimum_noise_sigma_ <= 31.5F) ||
        config.maximum_ink_luma_ < 0 || config.maximum_ink_luma_ > 255 ||
        config.minimum_local_contrast_ < 1 || config.minimum_local_contrast_ > 255)
        throw std::invalid_argument("Invalid noisy print search configuration");
    this->kernel_ = cv::getStructuringElement(cv::MORPH_RECT, {5, 5});
}

bool NoisyPrintSearch::has_native_noise(const cv::Mat& gray) const {
    return has_distributed_noise(gray, this->config_);
}

void NoisyPrintSearch::append(const cv::Mat& gray, const VideoConfig& video,
                              std::vector<Detection>& tags, const cv::Mat& area4,
                              std::optional<bool> known_noise) {
    const auto may_have_candidate = [&](const cv::Mat& guide) {
        return can_contain_dark_transition(guide, this->config_.maximum_ink_luma_,
                                           this->config_.minimum_local_contrast_);
    };
    if (!this->config_.enabled_ || (!area4.empty() && !may_have_candidate(area4)))
        return;
#if defined(__aarch64__) && defined(__linux__) && defined(__AARCH64EL__)
    if (!(known_noise ? *known_noise : this->has_native_noise(gray)))
        return;
#else
    (void)known_noise;
    if (!has_distributed_noise(gray, this->config_))
        return;
#endif
    if (area4.empty()) {
        average_gray4x4(gray, this->guide_);
        if (!may_have_candidate(this->guide_))
            return;
    }
    const cv::Mat& guide = area4.empty() ? this->guide_ : area4;
    cv::dilate(guide, this->local_maximum_, this->kernel_);
    build_noisy_mask(guide, this->local_maximum_, this->mask_, this->config_.maximum_ink_luma_,
                     this->config_.minimum_local_contrast_);
    if (cv::getNumThreads() == 1) {
        this->contours_.clear();
        this->contour_scanner_.for_each<cv::CHAIN_APPROX_SIMPLE>(
                this->mask_, true, [&](const std::vector<cv::Point>& contour) {
                    if (contour.size() < 4)
                        return;
                    const auto bounds = cv::boundingRect(contour);
                    const int shortest = std::min(bounds.width, bounds.height);
                    if (shortest >= 4 && std::max(bounds.width, bounds.height) <= 2.3 * shortest)
                        this->contours_.push_back(contour);
                });
        std::reverse(this->contours_.begin(), this->contours_.end());
    } else {
        cv::findContours(this->mask_, this->contours_, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    }
    this->candidates_.clear();
    std::vector<Detection> boundary;
    std::vector<ConfirmedFootprint> occupied;
    // Dense source detections keep their native geometry. Skip only an outline
    // matching a confirmed footprint, leaving smaller and displaced prints eligible.
    if (tags.size() > 5) {
        occupied.reserve(tags.size());
        for (const auto& tag : tags) {
            const auto& q = tag.corners_px_;
            occupied.push_back({.25F * (q[0] + q[1] + q[2] + q[3]),
                                std::max(length(q[0] - q[2]), length(q[1] - q[3])), q});
        }
    }
    for (const auto& contour : this->contours_) {
        if (contour.size() < 4)
            continue;
        const auto bounds = cv::boundingRect(contour);
        const int shortest = std::min(bounds.width, bounds.height);
        if (shortest < 4 || std::max(bounds.width, bounds.height) > 2.3 * shortest)
            continue;
        cv::approxPolyDP(contour, this->polygon_, .035 * cv::arcLength(contour, true), true);
        if (this->polygon_.size() != 4 || !cv::isContourConvex(this->polygon_))
            continue;
        for (auto& point : this->polygon_)
            point = {decoder_coordinate(point.x, gray.cols),
                     decoder_coordinate(point.y, gray.rows)};
        if (already_measured(this->polygon_, occupied))
            continue;
        const int margin = std::min(
                {bounds.x, bounds.y, guide.cols - bounds.br().x, guide.rows - bounds.br().y});
        // Include two guide-sample widths for contour quantization and a half
        // tag cell for an ordinary white-ring sample that may fall off-image.
        if (margin <= 2 + shortest / (2 * kBorderSide)) {
            if (auto tag = read_boundary(gray, this->polygon_, video.faint_decode_contrast_)) {
                boundary.push_back(*tag);
                continue;
            }
        }
        this->candidates_.push_back(this->polygon_);
    }

    auto found = this->decoder_.decode_sampled_contours(gray, this->candidates_);
    refine_diffuse_tags(gray, video.faint_decode_contrast_, found,
                        video.maximum_diffuse_reseed_cells_);
    const auto append_verified = [&](const Detection& tag) {
        const auto center_of = [](const Quad& quad) {
            return .25F * (quad[0] + quad[1] + quad[2] + quad[3]);
        };
        const Point center = center_of(tag.corners_px_);
        const float diagonal = length(tag.corners_px_[0] - tag.corners_px_[2]);
        auto same = std::find_if(tags.begin(), tags.end(), [&](const Detection& prior) {
            const auto& corners = prior.corners_px_;
            const float limit = .25F * std::min(diagonal, length(corners[0] - corners[2]));
            return length(center - center_of(corners)) < limit ||
                   (prior.id_ == tag.id_ &&
                    length(corners[0] - tag.corners_px_[0]) < .30F * diagonal);
        });
        if (same == tags.end()) {
            tags.push_back(tag);
        } else if (same->id_ == tag.id_ && tags.size() <= 5 &&
                   tag.corrected_bits_ <= same->corrected_bits_ &&
                   length(same->corners_px_[0] - tag.corners_px_[0]) < .30F * diagonal &&
                   prefer_refined(gray, *same, tag, video.faint_decode_contrast_)) {
            *same = tag;
        }
    };
    for (auto& tag : found) {
        if (auto errors = verify_payload(gray, tag.corners_px_, tag.id_, 26,
                                         video.faint_decode_contrast_, 1)) {
            tag.corrected_bits_ = *errors;
            append_verified(tag);
        }
    }
    for (const auto& tag : boundary)
        append_verified(tag);
}
}  // namespace aruco3cuda::apriltag_small_cpu::detail
