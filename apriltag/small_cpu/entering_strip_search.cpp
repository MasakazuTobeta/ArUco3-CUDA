// SPDX-License-Identifier: Apache-2.0
#include "entering_strip_search.hpp"
#include "gray_view.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>

namespace aruco3cuda::apriltag_small_cpu::detail {
namespace {

constexpr int kEdgeSlackPx = 4;
constexpr int kExposureTile = 40;
constexpr int kExposureStride = 4;
constexpr int kExposureMargin = 80;
constexpr int kExposureVotes = 4;
constexpr int kExposureBin = 16;

constexpr int kSampleStride = 2;
cv::Size sampled_size(const cv::Mat& source) {
    return {(source.cols + kSampleStride - 1) / kSampleStride,
            (source.rows + kSampleStride - 1) / kSampleStride};
}

Config strip_config(int threshold) {
    Config config;
    config.components_.emplace();
    config.components_->sample_stride_px_ = kSampleStride;
    config.components_->sampled_contours_ = true;
    config.components_->threshold_ = threshold;
    return config;
}

// Clockwise in image coordinates. After a step, the previous exterior neighbour
// is one of the four cardinal neighbours of the new foreground pixel.
const std::array<cv::Point, 8> kDirection{
        {{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}}};
constexpr std::array<int, 8> kBacktrack{{6, 6, 0, 0, 2, 2, 4, 4}};

}  // namespace

struct EnteringContours::Grid {
    explicit Grid(const cv::Mat& band, int threshold)
        : image(band), level(threshold), size(sampled_size(band)) {
        const auto pitch = static_cast<std::ptrdiff_t>(image.step[0]);
        for (int i = 0; i < 8; ++i) {
            sample_offset[i] = kDirection[i].y * size.width + kDirection[i].x;
            source_offset[i] = kSampleStride * (kDirection[i].y * pitch + kDirection[i].x);
        }
    }
    const cv::Mat& image;
    int level;
    cv::Size size;
    std::array<int, 8> sample_offset;
    std::array<std::ptrdiff_t, 8> source_offset;
};

void EnteringContours::trace(const Grid& grid, cv::Point position, int background) {
    const int first = position.y * grid.size.width + position.x;
    int current = first, first_neighbor = -1, previous_direction = -1;
    const auto* pixel =
            grid.image.ptr<std::uint8_t>(kSampleStride * position.y) + kSampleStride * position.x;
    chain_.clear();
    for (;;) {
        const bool interior = position.x > 0 && position.y > 0 &&
                              position.x + 1 < grid.size.width && position.y + 1 < grid.size.height;
        int next_direction;
        std::uint8_t exterior;
        // The first neighbour is known background. A fixed start permits unrolling
        // seven remaining tests with constant directions and exterior-bit masks.
        const auto step = [&](auto first_background) {
            constexpr int start = decltype(first_background)::value;
            unsigned int seen = 1U << start;
#if defined(__GNUC__)
#pragma GCC unroll 7
#endif
            for (int k = 1; k < 8; ++k) {
                const int direction = (start + k) % 8;
                if ((interior ||
                     cv::Rect({}, grid.size).contains(position + kDirection[direction])) &&
                    pixel[grid.source_offset[direction]] <= grid.level) {
                    next_direction = direction;
                    exterior = static_cast<std::uint8_t>(seen);
                    return;
                }
                seen |= 1U << direction;
            }
            next_direction = -1;
            exterior = 255;
        };
        switch (background) {
            case 0:
                step(std::integral_constant<int, 0>{});
                break;
            case 2:
                step(std::integral_constant<int, 2>{});
                break;
            case 4:
                step(std::integral_constant<int, 4>{});
                break;
            default:
                step(std::integral_constant<int, 6>{});
                break;
        }
        visited_[current] |= exterior;
        if (next_direction < 0)
            return;
        const int next = current + grid.sample_offset[next_direction];
        // A chain may touch itself. Returning to a coordinate alone is insufficient.
        if (current == first && next == first_neighbor)
            break;
        if (first_neighbor < 0)
            first_neighbor = next;
        if (previous_direction != next_direction)
            chain_.push_back(position);
        previous_direction = next_direction;
        current = next;
        position += kDirection[next_direction];
        pixel += grid.source_offset[next_direction];
        background = kBacktrack[next_direction];
    }
    if (chain_.size() >= 4)
        contours_.push_back(chain_);
}

const std::vector<std::vector<cv::Point>>& EnteringContours::find(const cv::Mat& band,
                                                                  int threshold, cv::Rect seed) {
    contours_.clear();
    const Grid grid(band, threshold);
    seed &= cv::Rect({}, grid.size);
    if (seed.empty())
        return contours_;
    visited_.assign(static_cast<std::size_t>(grid.size.width) * grid.size.height, 0);
    for (int y = seed.y; y < seed.y + seed.height; ++y) {
        const auto* row = band.ptr<std::uint8_t>(kSampleStride * y);
        for (int x = seed.x; x < seed.x + seed.width; ++x) {
            const auto* pixel = row + kSampleStride * x;
            if (*pixel > threshold)
                continue;
            const int cell = y * grid.size.width + x;
            // Different exterior sides of the same foreground pixel may trace a hole.
            const auto begin = [&](int side, bool outside) {
                if (!(visited_[cell] & (1U << side)) &&
                    (outside || pixel[grid.source_offset[side]] > threshold))
                    trace(grid, {x, y}, side);
            };
            begin(0, x + 1 == grid.size.width);
            begin(2, y + 1 == grid.size.height);
            begin(4, x == 0);
            begin(6, y == 0);
        }
    }
    return contours_;
}

EnteringStripSearch::EnteringStripSearch(int width_px, int threshold, int minimum_contrast)
    : width_px_(width_px),
      minimum_contrast_(minimum_contrast),
      config_(strip_config(threshold)),
      decoder_(config_) {
    if (width_px < 0)
        throw std::invalid_argument("Negative video pan strip width");
    if (minimum_contrast < 0 || minimum_contrast > 255)
        throw std::invalid_argument("Invalid video pan photometric contrast");
}

void EnteringStripSearch::exposure_ranges(const cv::Mat& gray, int shift, bool vertical) {
    ranges_.clear();
    const int along_size = vertical ? gray.rows : gray.cols;
    const int across_size = vertical ? gray.cols : gray.rows;
    const int depth = std::min(width_px_, across_size);
    const int reach =
            static_cast<int>(std::min<std::int64_t>(depth - 5, std::abs(std::int64_t{shift}) + 14));
    const int nominal = config_.components_->threshold_;
    const auto pixel = [&](int along, int inward) {
        const int across = shift > 0 ? inward : across_size - 1 - inward;
        return vertical ? int(gray.ptr<std::uint8_t>(along)[across])
                        : int(gray.ptr<std::uint8_t>(across)[along]);
    };

    for (int tile = 0; tile < along_size; tile += kExposureTile) {
        std::array<int, 256 / kExposureBin> votes{};
        // A sharp transition between two plateaus can reveal both printed levels even when
        // neither crosses the nominal threshold. Reject gradual ramps and isolated noise.
        const auto vote = [&](int along, int inward, int da, int di, int first) {
            const int last = pixel(along + 3 * da, inward + 3 * di);
            const int step = std::abs(last - first);
            if (step < minimum_contrast_ || (first <= nominal) != (last <= nominal))
                return;
            const int outer = pixel(along - da, inward - di);
            const int inner = pixel(along + 4 * da, inward + 4 * di);
            if (std::abs(first - outer) + std::abs(last - inner) > step / 6 + 2)
                return;
            const auto disagreement = [&](int side) {
                const int a = along + side * di, b = inward + side * da;
                return std::abs(first - pixel(a, b)) +
                       std::abs(last - pixel(a + 3 * da, b + 3 * di));
            };
            if (std::min(disagreement(2), disagreement(-2)) > step / 3 + 3)
                return;
            const int bin = (first + last) / (2 * kExposureBin);
            const int threshold = bin * kExposureBin + kExposureBin / 2;
            if ((first <= threshold) != (last <= threshold))
                ++votes[bin];
        };
        for (int along = std::max(4, tile); along < std::min(along_size - 5, tile + kExposureTile);
             along += kExposureStride) {
            for (int inward = 2; inward < reach; inward += kExposureStride) {
                const int value = pixel(along, inward);
                vote(along, inward, 1, 0, value);
                vote(along, inward, 0, 1, value);
            }
        }
        for (int bin = 0; bin < static_cast<int>(votes.size()); ++bin) {
            if (votes[bin] < kExposureVotes)
                continue;
            int first = std::max(0, tile - kExposureMargin);
            first -= first % config_.components_->sample_stride_px_;
            const int end = std::min(along_size, tile + kExposureTile + kExposureMargin);
            const int threshold = bin * kExposureBin + kExposureBin / 2;
            auto previous =
                    std::find_if(ranges_.begin(), ranges_.end(), [&](const ExposureRange& r) {
                        return r.threshold == threshold && first <= r.end;
                    });
            if (previous == ranges_.end())
                ranges_.push_back({first, end, threshold});
            else
                previous->end = end;
        }
    }
}

void EnteringStripSearch::append(const cv::Mat& gray, int shift, bool vertical, int threshold,
                                 cv::Point offset, std::vector<Detection>& result) {
    if (!shift)
        return;
    const int extent = vertical ? gray.cols : gray.rows;
    const int depth = std::min(width_px_, extent);
    const int origin = shift > 0 ? 0 : extent - depth;
    const cv::Rect strip = vertical ? cv::Rect(origin, 0, depth, gray.rows)
                                    : cv::Rect(0, origin, gray.cols, depth);
    const cv::Mat band = gray(strip);
    const auto samples_xy = sampled_size(band);
    const int reach = static_cast<int>(
            std::min<std::int64_t>(depth, std::abs(std::int64_t{shift}) + kEdgeSlackPx));
    constexpr int stride = kSampleStride;
    // Map bounds measured on the sampled foreground back to native coordinates. Far-edge
    // extents use exclusive ends, including the shortened last sample on odd-sized strips.
    const int samples = vertical ? samples_xy.width : samples_xy.height;
    const int begin = shift > 0 ? 0 : std::max(0, (depth - reach + stride - 1) / stride - 1);
    const int end = shift > 0 ? std::min(samples, reach / stride + 1) : samples;
    const cv::Rect seed = vertical ? cv::Rect(begin, 0, end - begin, samples_xy.height)
                                   : cv::Rect(0, begin, samples_xy.width, end - begin);
    const auto& curves = contours_.find(band, threshold, seed);
    for (auto tag : decoder_.decode_sampled_contours(band, curves)) {
        for (auto& corner : tag.corners_px_)
            corner += cv::Point2f(strip.tl() + offset);
        result.push_back(tag);
    }
}

std::vector<Detection> EnteringStripSearch::detect(const cv::Mat& gray, cv::Point translation) {
    validate_image(gray);
    std::vector<Detection> result;
    if (width_px_ > 0) {
        for (const bool vertical : {true, false}) {
            const int shift = vertical ? translation.x : translation.y;
            if (!shift)
                continue;
            append(gray, shift, vertical, config_.components_->threshold_, {}, result);
            if (!minimum_contrast_)
                continue;
            exposure_ranges(gray, shift, vertical);
            for (const auto& range : ranges_) {
                const int length = range.end - range.first;
                const cv::Rect crop = vertical ? cv::Rect(0, range.first, gray.cols, length)
                                               : cv::Rect(range.first, 0, length, gray.rows);
                append(gray(crop), shift, vertical, range.threshold, crop.tl(), result);
            }
        }
    }
    return result;
}
}  // namespace aruco3cuda::apriltag_small_cpu::detail
