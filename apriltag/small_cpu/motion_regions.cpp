// SPDX-License-Identifier: Apache-2.0
#include "motion_regions.hpp"

#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace aruco3cuda::apriltag_small_cpu::detail {
namespace {

constexpr int kStride = 4;
constexpr int kTile = 8;
constexpr int kMinimumChangedSamples = 3;
constexpr int kSearchRadius = 14;

void sample(const cv::Mat& frame, cv::Mat& thumbnail, cv::Point phase) {
    thumbnail.create(frame.rows / kStride, frame.cols / kStride, CV_8UC1);
    for (int y = 0; y < thumbnail.rows; ++y) {
        const auto* src = frame.ptr<std::uint8_t>(kStride * y + phase.y);
#if defined(__aarch64__) && defined(__GNUC__)
        // Wide source rows are separated by three rows that this sampler never visits.
        // Prefetch only within a visible source row.
        constexpr int kWideThumbnailColumns = 768;
        constexpr int kPrefetchStep = 32;
        if (thumbnail.cols >= kWideThumbnailColumns && y + 1 < thumbnail.rows) {
            const auto* next = frame.ptr<std::uint8_t>(kStride * (y + 1) + phase.y);
            for (int x = 0; x < thumbnail.cols; x += kPrefetchStep)
                __builtin_prefetch(next + kStride * x, 0, 2);
        }
#endif
        auto* dst = thumbnail.ptr<std::uint8_t>(y);
        int x = 0;
#if CV_SIMD128
        for (; x + 16 <= thumbnail.cols; x += 16) {
            cv::v_uint8x16 lanes[4];
            cv::v_load_deinterleave(src + kStride * x, lanes[0], lanes[1], lanes[2], lanes[3]);
            cv::v_store(dst + x, lanes[phase.x]);
        }
#endif
        for (; x < thumbnail.cols; ++x)
            dst[x] = src[kStride * x + phase.x];
    }
}

// Each tile is active once at least three thumbnail samples differ.
int count_active_tiles(const cv::Mat& current, const cv::Mat& previous, cv::Rect overlap,
                       int threshold, cv::Mat& tiles) {
    tiles.create((current.rows + kTile - 1) / kTile, (current.cols + kTile - 1) / kTile, CV_8UC1);
    const int right = overlap.x + overlap.width, bottom = overlap.y + overlap.height;
    int active = 0;
    for (int ty = 0; ty < tiles.rows; ty++) {
        auto* dst = tiles.ptr<std::uint8_t>(ty);
        const int y0 = std::max(ty * kTile, overlap.y), y1 = std::min((ty + 1) * kTile, bottom);
        for (int tx = 0; tx < tiles.cols; tx++) {
            const int x = tx * kTile;
#if CV_SIMD128
            if (x >= overlap.x && x + 16 <= right) {
                auto counts = cv::v_setzero_u8();
                for (int y = y0; y < y1; y++) {
                    const auto now = cv::v_load(current.ptr<std::uint8_t>(y) + x);
                    const auto before =
                            cv::v_load(previous.ptr<std::uint8_t>(y - overlap.y) + (x - overlap.x));
                    const auto diff =
                            cv::v_gt(cv::v_absdiff(now, before), cv::v_setall_u8(threshold));
                    counts = cv::v_add(counts, cv::v_and(diff, cv::v_setall_u8(1)));
                }
                cv::v_uint16x8 low, high;
                cv::v_expand(counts, low, high);
                dst[tx] = cv::v_reduce_sum(low) >= kMinimumChangedSamples ? 255 : 0;
                dst[tx + 1] = cv::v_reduce_sum(high) >= kMinimumChangedSamples ? 255 : 0;
                active += (dst[tx] != 0) + (dst[tx + 1] != 0);
                ++tx;
                continue;
            }
#endif
            int count = 0;
            int x0 = std::max(x, overlap.x), x1 = std::min(x + kTile, right);
            for (int y = y0; y < y1 && count < kMinimumChangedSamples; y++) {
                const auto* current_row = current.ptr<std::uint8_t>(y);
                const auto* previous_row = previous.ptr<std::uint8_t>(y - overlap.y);
                for (int xx = x0; xx < x1; xx++)
                    count += std::abs(current_row[xx] - previous_row[xx - overlap.x]) > threshold;
            }
            dst[tx] = count >= kMinimumChangedSamples ? 255 : 0;
            active += dst[tx] != 0;
        }
    }
    return active;
}

}  // namespace

MotionRegions::MotionRegions(int quiet_threshold, int change_threshold, int padding_px,
                             int minimum_close_percent, int maximum_regions)
    : quiet_threshold_(quiet_threshold),
      change_threshold_(change_threshold),
      padding_px_(padding_px),
      minimum_close_percent_(minimum_close_percent),
      maximum_regions_(maximum_regions) {
    if (quiet_threshold < 0 || change_threshold < quiet_threshold || change_threshold > 255 ||
        padding_px < 0 || minimum_close_percent < 0 || minimum_close_percent > 100 ||
        maximum_regions < 0) {
        throw std::invalid_argument("invalid video change-region configuration");
    }
}

void MotionRegions::reset() {
    previous_.release();
    size_ = {};
    phase_ = {};
    last_shift_ = {};
}

MotionRegions::Alignment MotionRegions::align(const cv::Mat& gray,
                                              const cv::Point* retry_shift) const {
    struct Anchor {
        int x;
        int y;
        int intensity;
    };
    std::array<Anchor, 768> anchors;
    const int rows = retry_shift ? 24 : 10;
    const int columns = retry_shift ? 32 : 12;
    const int limit = retry_shift ? static_cast<int>(anchors.size()) : 160;
    int count = 0;
    // A four-thumbnail-pixel margin permits every trial shift in [-14, 14] at every phase.
    for (int y = 4; y < previous_.rows - 4; y += std::max(5, previous_.rows / rows)) {
        const auto* row = previous_.ptr<std::uint8_t>(y);
        const auto* up = previous_.ptr<std::uint8_t>(y - 1);
        const auto* down = previous_.ptr<std::uint8_t>(y + 1);
        for (int x = 4; x < previous_.cols - 4; x += std::max(5, previous_.cols / columns)) {
            int contrast = std::abs(row[x + 1] - row[x - 1]) + std::abs(up[x] - down[x]);
            if (contrast >= 6 && count < limit) {
                anchors[count++] = {kStride * x + phase_.x, kStride * y + phase_.y, row[x]};
            }
        }
    }
    Alignment result;
    if (count < 8) {
        result.reliable = !retry_shift;
        return result;
    }

    const auto cost = [&](cv::Point shift) {
        int sum = 0;
        for (int i = 0; i < count; ++i) {
            const auto& a = anchors[i];
            int difference = gray.ptr<std::uint8_t>(a.y + shift.y)[a.x + shift.x] - a.intensity;
            sum += std::min(26, std::abs(difference));
        }
        return sum;
    };

    result.shift = retry_shift ? *retry_shift : last_shift_;
    int best = cost(result.shift);
    const auto check = [&](cv::Point shift) {
        if (std::abs(shift.x) > kSearchRadius || std::abs(shift.y) > kSearchRadius)
            return;
        int trial = cost(shift);
        if (trial < best) {
            best = trial;
            result.shift = shift;
        }
    };
    const auto refine = [&](cv::Point origin) {
        for (int y = -2; y <= 2; ++y) {
            for (int x = -2; x <= 2; ++x)
                check(origin + cv::Point(x, y));
        }
    };
    if (!retry_shift && best > count * 2)
        check({});
    if (retry_shift || best > count * 3)
        refine(result.shift);
    if (!retry_shift && best > count * 5) {
        for (int y = -12; y <= 12; y += 4) {
            for (int x = -12; x <= 12; x += 4)
                check({x, y});
        }
        refine(result.shift);
    }
    std::array<int, 768> differences;
    for (int i = 0; i < count; ++i) {
        const auto& a = anchors[i];
        differences[i] =
                gray.ptr<std::uint8_t>(a.y + result.shift.y)[a.x + result.shift.x] - a.intensity;
    }
    auto middle = differences.begin() + count / 2;
    const auto end = differences.begin() + count;
    std::nth_element(differences.begin(), middle, end);
    result.exposure = *middle;
    const auto close = std::count_if(differences.begin(), end,
                                     [&](int d) { return std::abs(d - result.exposure) <= 3; });
    result.quiet = close * 10 >= count * 9;
    result.reliable = best <= count * 9 && close * 100 >= count * minimum_close_percent_;
    return result;
}

bool MotionRegions::regions(cv::Size image_size, cv::Rect overlap, cv::Point phase, int threshold,
                            const cv::Mat& aligned_previous, std::vector<cv::Rect>& changed) {
    // Ignore newly exposed pixels outside the overlap; those are searched at the leading edge.
    const int active = count_active_tiles(current_, aligned_previous, overlap, threshold, tiles_);
    if (!active)
        return true;
    if (static_cast<std::size_t>(active) * 5 > tiles_.total())
        return false;
    cv::findContours(tiles_, contours_, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours_.size() > maximum_regions_)
        return false;
    const auto clamp = [](std::int64_t v, int upper) {
        return static_cast<int>(std::clamp<std::int64_t>(v, 0, upper));
    };
    std::int64_t area = 0;
    for (const auto& contour : contours_) {
        cv::Rect r = cv::boundingRect(contour);
        constexpr std::int64_t tile_pixels = kStride * kTile;
        const int x0 = clamp(tile_pixels * r.x + phase.x - padding_px_, image_size.width);
        const int y0 = clamp(tile_pixels * r.y + phase.y - padding_px_, image_size.height);
        const int x1 =
                clamp(tile_pixels * (r.x + r.width) + phase.x + padding_px_, image_size.width);
        const int y1 =
                clamp(tile_pixels * (r.y + r.height) + phase.y + padding_px_, image_size.height);
        changed.emplace_back(x0, y0, x1 - x0, y1 - y0);
        area += std::int64_t{x1 - x0} * (y1 - y0);
    }
    return static_cast<double>(area) < .23 * image_size.width * image_size.height;
}

bool MotionRegions::locate(const cv::Mat& gray, const Alignment& alignment, cv::Point& phase,
                           std::vector<cv::Rect>& changed, bool& has_overlap) {
    bool use_regions = false;
    has_overlap = false;
    cv::Point offset = phase_ + alignment.shift;
    phase = {(offset.x % kStride + kStride) % kStride, (offset.y % kStride + kStride) % kStride};
    cv::Point shift{(offset.x - phase.x) / kStride, (offset.y - phase.y) / kStride};
    sample(gray, current_, phase);
    const cv::Rect whole({}, current_.size());
    const cv::Rect overlap = whole & (whole + shift);
    if (!overlap.empty()) {
        has_overlap = true;
        const cv::Mat previous_overlap = previous_(overlap - shift);
        const bool adjust = std::abs(alignment.exposure) > 2;
        if (adjust)
            cv::add(previous_overlap, cv::Scalar(alignment.exposure), adjusted_);
        const auto& reference = adjust ? adjusted_ : previous_overlap;
        const int threshold = alignment.quiet ? quiet_threshold_ : change_threshold_;
        use_regions = regions(gray.size(), overlap, phase, threshold, reference, changed);
        if (!use_regions && threshold < change_threshold_) {
            changed.clear();
            use_regions =
                    regions(gray.size(), overlap, phase, change_threshold_, reference, changed);
        }
    }
    return use_regions;
}

bool MotionRegions::update(const cv::Mat& gray, bool force_full, cv::Point& translation,
                           std::vector<cv::Rect>& changed, bool* reliable_overlap) {
    CV_Assert(gray.dims == 2 && !gray.empty() && gray.type() == CV_8UC1);
    if (reliable_overlap)
        *reliable_overlap = false;
    translation = {};
    changed.clear();
    if (gray.rows < kStride || gray.cols < kStride) {
        reset();
        return false;
    }

    Alignment alignment;
    cv::Point phase;
    bool use_regions = false;
    const bool compare = !force_full && !previous_.empty() && size_ == gray.size();
    if (compare)
        alignment = align(gray);
    if (compare && alignment.reliable) {
        bool has_overlap = false;
        use_regions = locate(gray, alignment, phase, changed, has_overlap);
        // Uniform patches can make the previous shift cheap despite a one- or two-pixel
        // error. Only expand the translation search if the residual would force discovery.
        if (!use_regions && has_overlap) {
            const Alignment candidate = align(gray, &alignment.shift);
            if (candidate.reliable && candidate.shift != alignment.shift) {
                alignment = candidate;
                changed.clear();
                use_regions = locate(gray, alignment, phase, changed, has_overlap);
            }
        }
        if (reliable_overlap)
            *reliable_overlap = has_overlap;
        translation = alignment.shift;
    } else {
        sample(gray, current_, phase);
    }
    if (!use_regions)
        changed.clear();
    // Periodic discovery keeps the translation estimate, while changing image size starts a new
    // sequence.
    if (compare)
        last_shift_ = alignment.shift;
    else if (size_ != gray.size())
        last_shift_ = {};
    phase_ = phase;
    size_ = gray.size();
    std::swap(previous_, current_);
    return use_regions;
}

}  // namespace aruco3cuda::apriltag_small_cpu::detail
