// SPDX-License-Identifier: Apache-2.0
#include "coarse_mask.hpp"
#include "coarse_bit_erosion.hpp"
#include "cpu_gray_bits.hpp"

#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <type_traits>

namespace aruco3cuda::apriltag_small_cpu::detail {
namespace {

using Byte = std::uint8_t;

struct SampleGrid {
    explicit SampleGrid(const cv::Mat& source, int stride)
        : source_(source),
          stride_(stride),
          center_((stride - 1) / 2),
          size_((source.cols + stride - 1) / stride, (source.rows + stride - 1) / stride) {}

    const Byte* row(int y) const {
        return this->source_.ptr<Byte>(
                std::min(y * this->stride_ + this->center_, this->source_.rows - 1));
    }

    int column(int x) const {
        return std::min(x * this->stride_ + this->center_, this->source_.cols - 1);
    }

    const cv::Mat& source_;
    const int stride_;
    const int center_;
    const cv::Size size_;
};

#if CV_SIMD128
using ByteVector = cv::v_uint8x16;

template <int Stride>
ByteVector sample_block(const Byte* row) {
    ByteVector a, b, c, d;
    if constexpr (Stride == 2) {
        cv::v_load_deinterleave(row, a, b);
        return a;
    } else if constexpr (Stride == 3) {
        cv::v_load_deinterleave(row, a, b, c);
        return b;
    } else {
        cv::v_load_deinterleave(row, a, b, c, d);
        return b;
    }
}
#endif

template <bool DarkMask = false>
void sample_gray(const SampleGrid& grid, cv::Mat& sampled, int level = 0) {
    sampled.create(grid.size_, CV_8UC1);
    for (int y = 0; y < grid.size_.height; ++y) {
        const Byte* source = grid.row(y);
        Byte* output = sampled.ptr<Byte>(y);
        int x = 0;
#if CV_SIMD128
        for (; x + 16 <= grid.size_.width && (x + 16) * grid.stride_ <= grid.source_.cols;
             x += 16) {
            const Byte* block = source + x * grid.stride_;
            ByteVector values;
            switch (grid.stride_) {
                case 2:
                    values = sample_block<2>(block);
                    break;
                case 3:
                    values = sample_block<3>(block);
                    break;
                default:
                    values = sample_block<4>(block);
                    break;
            }
            if constexpr (DarkMask)
                values = cv::v_le(values, cv::v_setall_u8(static_cast<Byte>(level)));
            cv::v_store(output + x, values);
        }
#endif
        for (; x < grid.size_.width; ++x) {
            const Byte value = source[grid.column(x)];
            output[x] = DarkMask ? (value <= level ? 255 : 0) : value;
        }
    }
}

// Inverted threshold followed by 3x3 erosion is equivalent to comparing the
// neighborhood maximum to the threshold. Retain just one row of vertical maxima.
void fixed_dark_mask(const SampleGrid& grid, int level, std::vector<Byte>& row,
                     cv::Mat& output_mask) {
    const int width = grid.size_.width, height = grid.size_.height;
    row.resize(width + 2);
    output_mask.create(grid.size_, CV_8UC1);
    for (int y = 0; y < height; ++y) {
        const Byte* above = grid.row(std::max(0, y - 1));
        const Byte* center = grid.row(y);
        const Byte* below = grid.row(std::min(height - 1, y + 1));
        int x = 0;
#if CV_SIMD128
        const auto reduce = [&](auto load) {
            for (; x + 16 <= width && (x + 16) * grid.stride_ <= grid.source_.cols; x += 16) {
                const int offset = x * grid.stride_;
                const auto low = cv::v_max(load(above + offset), load(center + offset));
                cv::v_store(row.data() + x + 1, cv::v_max(low, load(below + offset)));
            }
        };
        switch (grid.stride_) {
            case 2:
                reduce(sample_block<2>);
                break;
            case 3:
                reduce(sample_block<3>);
                break;
            default:
                reduce(sample_block<4>);
                break;
        }
#endif
        for (; x < width; ++x) {
            const int source_x = grid.column(x);
            row[x + 1] = std::max({above[source_x], center[source_x], below[source_x]});
        }
        row.front() = row[1];
        row.back() = row[width];
        Byte* output = output_mask.ptr<Byte>(y);
        x = 0;
#if CV_SIMD128
        const auto threshold = cv::v_setall_u8(static_cast<Byte>(level));
        for (; x + 16 <= width; x += 16) {
            const auto left = cv::v_load(row.data() + x);
            const auto mid = cv::v_load(row.data() + x + 1);
            const auto right = cv::v_load(row.data() + x + 2);
            const auto maximum = cv::v_max(cv::v_max(left, mid), right);
            cv::v_store(output + x, cv::v_le(maximum, threshold));
        }
#endif
        for (; x < width; ++x)
            output[x] = std::max({row[x], row[x + 1], row[x + 2]}) <= level ? 255 : 0;
    }
}

// Resize the logical byte vector before reshaping. This retains its capacity
// and reference count while exposing an independent, tightly packed image.
void create_packed_raster(cv::Mat& raster, cv::Mat& storage, cv::Size size) {
    const auto count = size.area();
    if (storage.empty())
        storage.create(count, 1, CV_8UC1);
    else
        storage.resize(count);
    raster = storage.reshape(1, size.height);
}

// A tile below the contrast threshold contributes only zeros. One additional
// tile around the nonzero support preserves the original erosion neighborhood
// and the isolated zero border for subsequent contour tracing.
cv::Rect active_tile_region(const cv::Mat& ranges, int contrast) {
    const cv::Rect full(0, 0, ranges.cols, ranges.rows);
    int left = ranges.cols, right = 0, top = ranges.rows, bottom = 0;
    for (int k = 0; k < ranges.rows; ++k) {
        const int y = k % 2 ? ranges.rows - 1 - k / 2 : k / 2;
        const auto* row = ranges.ptr<cv::Vec2b>(y);
        const auto active = [&](int x) { return 255 - row[x][1] - row[x][0] >= contrast; };
        int x = 0;
        while (x < ranges.cols && !active(x))
            ++x;
        if (x == ranges.cols)
            continue;
        int end = ranges.cols;
        while (!active(end - 1))
            --end;
        left = std::min(left, x);
        right = std::max(right, end);
        top = std::min(top, y);
        bottom = std::max(bottom, y + 1);
        if (left <= 1 && right >= ranges.cols - 1 && top <= 1 && bottom >= ranges.rows - 1)
            return full;
    }
    if (left == ranges.cols)
        return {};
    const auto region = cv::Rect(left - 1, top - 1, right - left + 2, bottom - top + 2) & full;
    // A near-full crop saves little scanning and can resize both contour workspaces
    // on successive frames. Reuse the full raster beyond the fixed-mask crossover.
    return std::int64_t{region.area()} * 5 >= std::int64_t{full.area()} * 4 ? full : region;
}

// A nonzero eroded pixel has dark samples in every row of its 3x3 neighborhood.
// Sample one row in three, including the clamped bottom edge. The dark support
// on those complete source rows conservatively bounds all surviving pixels.
cv::Rect fixed_region(const SampleGrid& grid, int level) {
    const int width = grid.size_.width, height = grid.size_.height;
    const cv::Rect full({}, grid.size_);
    if (width < 32 || height < 32)
        return full;
    int left = width, right = 0, top = height, bottom = 0;
    const int probes = (height + 2) / 3;
    for (int i = 0; i < probes; ++i) {
        // Alternating ends abandons the crop early for broad, textured fields.
        const int index = i % 2 ? probes - 1 - i / 2 : i / 2;
        const int y = std::min(1 + 3 * index, height - 1);
        const auto* row = grid.row(y);
        int first = 0, end = grid.source_.cols;
#if CV_SIMD128
        const auto cutoff = cv::v_setall_u8(static_cast<Byte>(level));
        const auto clear = [&](int x) {
            const auto a = cv::v_min(cv::v_load(row + x), cv::v_load(row + x + 16));
            const auto b = cv::v_min(cv::v_load(row + x + 32), cv::v_load(row + x + 48));
            return !cv::v_check_any(cv::v_le(cv::v_min(a, b), cutoff));
        };
        while (first + 64 <= end && clear(first))
            first += 64;
        while (end - 64 >= first && clear(end - 64))
            end -= 64;
#endif
        while (first < end && row[first] > level)
            ++first;
        if (first == end)
            continue;
        while (row[end - 1] > level)
            --end;
        left = std::min(left, first / grid.stride_);
        right = std::max(right, (end - 1) / grid.stride_ + 1);
        top = std::min(top, y);
        bottom = std::max(bottom, y + 1);
        if (std::int64_t{right - left} * (bottom - top) * 5 >= std::int64_t{width} * height * 4)
            return full;
    }
    if (left >= right)
        return {};
    // One sample covers the radius to the nearest probe, one covers the
    // neighborhood, and one remains as an isolated zero boundary for tracing.
    constexpr int kPadding = 3;
    return (cv::Rect(left, top, right - left, bottom - top) + cv::Size(2 * kPadding, 2 * kPadding) -
            cv::Point(kPadding, kPadding)) &
           full;
}

}  // namespace

int component_threshold(const cv::Mat& gray, const ComponentConfig& config) {
    if (!config.exposure_)
        return config.threshold_;

    const auto& exposure = *config.exposure_;
    std::array<int, 256> histogram{};
    std::int64_t samples = 0;
    for (int y = 7; y < gray.rows; y += 29) {
        const auto* row = gray.ptr<Byte>(y);
        for (int x = 11; x < gray.cols; x += 31) {
            ++histogram[row[x]];
            ++samples;
        }
    }
    if (samples == 0)
        return config.threshold_;

    const auto low_rank = (20 * samples + 99) / 100;
    const auto high_rank = (90 * samples + 99) / 100;
    if (std::accumulate(histogram.begin(), histogram.begin() + exposure.minimum_low_percentile_,
                        std::int64_t{0}) >= low_rank)
        return config.threshold_;

    std::int64_t cumulative = 0;
    int highlight = 0;
    for (; highlight < static_cast<int>(histogram.size()); ++highlight) {
        cumulative += histogram[highlight];
        if (cumulative >= high_rank)
            break;
    }
    const int extra = std::clamp(
            (highlight - exposure.highlight_reference_) * exposure.adjustment_percent_ / 100, 0,
            exposure.maximum_adjustment_);
    return std::min(255, config.threshold_ + extra);
}

const cv::Mat& CoarseMask::build(const cv::Mat& gray, const Config& config,
                                 const CoarseConfig& coarse, cv::Point* sample_origin,
                                 bool bound_adaptive, std::vector<std::uint64_t>* packed) {
    if (packed)
        packed->clear();
    if (sample_origin)
        *sample_origin = {};
    const SampleGrid grid(gray, coarse.sample_stride_px_);
    int dim = 0, bright = 0, samples = 0;
    for (int y = 5; y < grid.size_.height; y += 13) {
        const Byte* row = grid.row(y);
        for (int x = 5; x < grid.size_.width; x += 17) {
            const int level = row[grid.column(x)];
            dim += level < coarse.dim_sample_floor_;
            bright += level >= config.bright_sample_floor_;
            ++samples;
        }
    }
    if (dim * 100 > samples * coarse.dim_sample_percentage_) {
        const bool emit_bits =
                packed && coarse.tile_side_samples_ == 13 && coarse.tile_minimum_contrast_ > 0;
        this->threshold_tiles(gray, coarse, bound_adaptive ? sample_origin : nullptr, emit_bits);
        if (this->binary_.empty()) {
            this->eroded_.release();
            return this->eroded_;
        }
        if (packed) {
            if (emit_bits)
                erode_coarse_packed(this->binary_.size(), this->source_bits_, *packed);
            else
                erode_coarse_bits(this->binary_, this->source_bits_, *packed);
            return this->binary_;
        }
        create_packed_raster(this->eroded_, this->adaptive_eroded_storage_, this->binary_.size());
        cv::erode(this->binary_, this->eroded_, cv::Mat());
    } else {
        const int level = bright * 100 > samples * config.bright_sample_percentage_
                                  ? config.bright_threshold_
                                  : config.dark_threshold_;
        const cv::Rect full({}, grid.size_);
        const cv::Rect area = sample_origin ? fixed_region(grid, level) : full;
        if (area.empty()) {
            this->eroded_.release();
        } else if (area == full) {
            fixed_dark_mask(grid, level, this->row_, this->eroded_);
        } else {
            const int stride = grid.stride_;
            const cv::Rect source(area.x * stride, area.y * stride,
                                  std::min(gray.cols, area.br().x * stride) - area.x * stride,
                                  std::min(gray.rows, area.br().y * stride) - area.y * stride);
            const cv::Mat patch = gray(source);
            fixed_dark_mask(SampleGrid(patch, stride), level, this->row_, this->eroded_);
            if (area.x > 0)
                this->eroded_.col(0).setTo(0);
            if (area.br().x < full.width)
                this->eroded_.col(area.width - 1).setTo(0);
            if (area.y > 0)
                this->eroded_.row(0).setTo(0);
            if (area.br().y < full.height)
                this->eroded_.row(area.height - 1).setTo(0);
            *sample_origin = area.tl();
        }
    }
    return this->eroded_;
}

void CoarseMask::threshold_tiles(const cv::Mat& gray, const CoarseConfig& config,
                                 cv::Point* sample_origin, bool emit_bits) {
    const SampleGrid grid(gray, config.sample_stride_px_);
    // Adjacent tile extrema support a stable threshold along the tile boundaries.
    const int tile = config.tile_side_samples_;
    bool fused_sampling = false;
#if defined(__aarch64__) && defined(__AARCH64EL__)
    constexpr int kWideGuideSamples = 512;
    fused_sampling = tile == 13 && grid.stride_ == 3 && grid.size_.width >= kWideGuideSamples;
#endif
    if (fused_sampling)
        this->sampled_.create(grid.size_, CV_8UC1);
    else
        sample_gray(grid, this->sampled_);
    const int cols = (this->sampled_.cols + tile - 1) / tile;
    const int rows = (this->sampled_.rows + tile - 1) / tile;
    this->extrema_.create(rows, cols, CV_8UC2);
#if CV_SIMD128
    const auto active =
            cv::v_gt(cv::v_setall_u8(static_cast<Byte>(tile)),
                     cv::v_uint8x16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));
#endif
    for (int j = 0; j < rows; ++j) {
        const int bottom = std::min(this->sampled_.rows, (j + 1) * tile);
        auto* extrema = this->extrema_.ptr<cv::Vec2b>(j);
        for (int i = 0; i < cols; ++i) {
            const int left = i * tile;
            const int right = std::min(this->sampled_.cols, (i + 1) * tile);
            Range range{255, 0};
#if CV_SIMD128
            if (tile <= 16 && left + 16 <= this->sampled_.cols &&
                (!fused_sampling || 3 * (left + 16) <= gray.cols)) {
                auto minimum = cv::v_setall_u8(255);
                auto maximum = cv::v_setzero_u8();
                for (int y = j * tile; y < bottom; ++y) {
                    auto* sample_row = this->sampled_.ptr<Byte>(y) + left;
                    const auto pixels = fused_sampling ? sample_block<3>(grid.row(y) + 3 * left)
                                                       : cv::v_load(sample_row);
                    // Overlapping lanes contain the same grid samples. The extrema use
                    // only the requested tile columns.
                    if (fused_sampling)
                        cv::v_store(sample_row, pixels);
                    minimum = cv::v_min(minimum, pixels);
                    maximum = cv::v_max(maximum, pixels);
                }
                range.low_ = cv::v_reduce_min(cv::v_or(minimum, cv::v_not(active)));
                range.high_ = cv::v_reduce_max(cv::v_and(maximum, active));
                extrema[i] = {range.low_, static_cast<Byte>(255 - range.high_)};
                continue;
            }
#endif
            for (int y = j * tile; y < bottom; ++y) {
                const auto* source = fused_sampling ? grid.row(y) : nullptr;
                auto* samples = this->sampled_.ptr<Byte>(y);
                for (int x = left; x < right; ++x) {
                    const Byte value = fused_sampling ? source[grid.column(x)] : samples[x];
                    if (fused_sampling)
                        samples[x] = value;
                    range.low_ = std::min(range.low_, value);
                    range.high_ = std::max(range.high_, value);
                }
            }
            extrema[i] = {range.low_, static_cast<Byte>(255 - range.high_)};
        }
    }
    // Complement the maxima so both neighboring extrema use a minimum filter.
    // The default erosion border is neutral for both unsigned channels.
    cv::erode(this->extrema_, this->neighboring_extrema_, cv::Mat());
    const cv::Rect all_tiles(0, 0, cols, rows);
    const cv::Rect retained = sample_origin ? active_tile_region(this->neighboring_extrema_,
                                                                 config.tile_minimum_contrast_)
                                            : all_tiles;
    if (retained.empty()) {
        this->binary_.release();
        return;
    }
    const auto emit = [&](auto is_full) {
        constexpr bool whole = decltype(is_full)::value;
        const int first_col = whole ? 0 : retained.x, first_row = whole ? 0 : retained.y;
        const int last_col = whole ? cols : retained.br().x,
                  last_row = whole ? rows : retained.br().y;
        if (sample_origin)
            *sample_origin = {first_col * tile, first_row * tile};
        const int x0 = first_col * tile, y0 = first_row * tile;
        const int x1 = whole ? this->sampled_.cols : std::min(this->sampled_.cols, last_col * tile);
        const int y1 = whole ? this->sampled_.rows : std::min(this->sampled_.rows, last_row * tile);
        create_packed_raster(this->binary_, this->adaptive_binary_storage_, {x1 - x0, y1 - y0});
        if (emit_bits) {
            using Word = std::uint64_t;
            const int width = x1 - x0;
            const int words = (width + 63) / 64;
            const auto count = static_cast<std::size_t>(words) * (y1 - y0);
            const int tail = width % 64;
            const Word neutral = tail == 0 ? 0 : ~((Word{1} << tail) - 1);
            this->source_bits_.resize(count);
            this->threshold_row_.resize(width);
#if defined(__aarch64__) && defined(__AARCH64EL__)
            const aruco3cuda::cpu::detail::GrayComparisonBits pack;
#endif
            for (int j = first_row; j < last_row; ++j) {
                const int bottom = std::min(y1, (j + 1) * tile);
                const auto* ranges = this->neighboring_extrema_.ptr<cv::Vec2b>(j);
                for (int i = first_col; i < last_col; ++i) {
                    const int low = ranges[i][0], high = 255 - ranges[i][1];
                    const int left = i * tile, count = std::min(x1 - left, tile);
                    // An active tile has high > low, making its midpoint
                    // at most 254. Use an exclusive cutoff one higher;
                    // zero rejects all samples on inactive tiles.
                    const int cutoff =
                            high - low >= config.tile_minimum_contrast_ ? (high + low) / 2 + 1 : 0;
                    std::memset(this->threshold_row_.data() + left - x0, cutoff, count);
                }
                for (int y = j * tile; y < bottom; ++y) {
                    auto* row =
                            this->source_bits_.data() + static_cast<std::size_t>(y - y0) * words;
                    const auto* gray_row = this->sampled_.ptr<Byte>(y) + x0;
                    const auto* cutoffs = this->threshold_row_.data();
                    int x = 0;
#if defined(__aarch64__) && defined(__AARCH64EL__)
                    for (; x + 64 <= width; x += 64) {
                        const auto compare = [&](int pos) {
                            return vcltq_u8(vld1q_u8(gray_row + pos), vld1q_u8(cutoffs + pos));
                        };
                        row[x / 64] = pack.pack(compare(x), compare(x + 16), compare(x + 32),
                                                compare(x + 48));
                    }
#endif
                    for (; x < width; x += 64) {
                        Word bits = 0;
                        const int end = std::min(width - x, 64);
                        int k = 0;
#if CV_SIMD128
                        for (; k + 16 <= end; k += 16)
                            bits |= Word(cv::v_signmask(cv::v_lt(cv::v_load(gray_row + x + k),
                                                                 cv::v_load(cutoffs + x + k))))
                                    << k;
#endif
                        for (; k < end; ++k)
                            bits |= Word(gray_row[x + k] < cutoffs[x + k]) << k;
                        row[x / 64] = bits;
                    }
                    row[words - 1] |= neutral;
                }
            }
            return;
        }
        for (int j = first_row; j < last_row; ++j) {
            const int bottom = std::min(this->sampled_.rows, (j + 1) * tile);
            for (int i = first_col; i < last_col; ++i) {
                const int right = std::min(this->sampled_.cols, (i + 1) * tile);
                const auto range = this->neighboring_extrema_.ptr<cv::Vec2b>(j)[i];
                const int low = range[0], high = 255 - range[1];
                const int threshold = (low + high) / 2;
                const bool enough_contrast = high - low >= config.tile_minimum_contrast_;
                const int left = i * tile;
#if CV_SIMD128
                if (tile <= 16 && left + 16 <= x1) {
                    const auto cutoff = cv::v_setall_u8(static_cast<Byte>(threshold));
                    for (int y = j * tile; y < bottom; ++y) {
                        const auto* source = this->sampled_.ptr<Byte>(y);
                        auto* output = this->binary_.ptr<Byte>(y - y0);
                        const auto flags = enough_contrast
                                                   ? cv::v_le(cv::v_load(source + left), cutoff)
                                                   : cv::v_setzero_u8();
                        // The following tile always replaces the overlapping columns.
                        cv::v_store(output + (left - x0), flags);
                    }
                    continue;
                }
#endif
                for (int y = j * tile; y < bottom; ++y) {
                    const Byte* source = this->sampled_.ptr<Byte>(y);
                    Byte* output = this->binary_.ptr<Byte>(y - y0);
                    for (int x = left; x < right; ++x)
                        output[x - x0] = enough_contrast && source[x] <= threshold ? 255 : 0;
                }
            }
        }
    };
    if (retained == all_tiles)
        emit(std::true_type{});
    else
        emit(std::false_type{});
}

const cv::Mat& CoarseMask::build_components(const cv::Mat& gray, const ComponentConfig& config) {
    const SampleGrid grid(gray, config.sample_stride_px_);
    const int side = config.local_midrange_side_samples_;
    if (side != 0) {
        sample_gray(grid, this->sampled_);
        if (this->local_kernel_.cols != side)
            this->local_kernel_ = cv::getStructuringElement(cv::MORPH_RECT, {side, side});
        cv::erode(this->sampled_, this->local_low_, this->local_kernel_);
        cv::dilate(this->sampled_, this->local_high_, this->local_kernel_);
        cv::addWeighted(this->local_low_, 0.5, this->local_high_, 0.5,
                        -config.local_midrange_offset_, this->local_low_);
        cv::compare(this->sampled_, this->local_low_, this->binary_, cv::CMP_LT);
        return this->binary_;
    }
    const int level = component_threshold(gray, config);
    sample_gray<true>(grid, this->binary_, level);
    return this->binary_;
}

}  // namespace aruco3cuda::apriltag_small_cpu::detail
