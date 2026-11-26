// SPDX-License-Identifier: Apache-2.0
#include "component_candidates.hpp"
#include "bounded_quad_approximation.hpp"

#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#endif

#if defined(__aarch64__) && defined(__AARCH64EL__) && defined(__linux__) && __has_include(<arm_sve.h>)
#include <arm_sve.h>
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif
#include "packed_gray_bits.hpp"

#include <opencv2/imgproc.hpp>

#if defined(__aarch64__) && defined(__AARCH64EL__)
#include <arm_neon.h>
#endif

#include <array>

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace aruco3cuda::apriltag_small_cpu::detail {
namespace {

int skip_equal(const std::uint8_t* row, int position, int width, std::uint8_t value) {
#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && \
        (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__) && (defined(__GNUC__) || defined(__clang__))
    constexpr int kWordBytes = sizeof(std::uint64_t);
    const std::uint64_t repeated = std::uint64_t{0x0101010101010101} * value;
    for (; position <= width - kWordBytes; position += kWordBytes) {
        std::uint64_t word;
        std::memcpy(&word, row + position, sizeof(word));
        const std::uint64_t differences = word ^ repeated;
        if (differences != 0U)
            return position + __builtin_ctzll(differences) / 8;
    }
#endif
    for (; position < width && row[position] == value; ++position) {
    }
    return position;
}

#if defined(__aarch64__) && defined(__AARCH64EL__)
class DarkWord {
public:
    explicit DarkWord(int threshold) : limit_(vdupq_n_u8(static_cast<std::uint8_t>(threshold))) {}

    template <int kStride>
    std::uint64_t full(const std::uint8_t* row) const {
        const auto selected = [&](const std::uint8_t* source) {
            uint8x16_t intensities;
            if constexpr (kStride == 1)
                intensities = vld1q_u8(source);
            else
                intensities = vld2q_u8(source).val[0];
            return vcleq_u8(intensities, this->limit_);
        };
        return bits_.pack(selected(row), selected(row + 16 * kStride), selected(row + 32 * kStride),
                          selected(row + 48 * kStride));
    }

private:
    GrayComparisonBits bits_;
    uint8x16_t limit_;
};
#endif

#if defined(__aarch64__) && defined(__AARCH64EL__) && defined(__linux__) && __has_include(<arm_sve.h>)
// A byte predicate has one bit per native pixel. Keep its even-numbered bits
// from two consecutive vectors, yielding one packed sample per two pixels.
__attribute__((target("arch=armv8.2-a+sve"))) void pack_native_stride_two(
        const cv::Mat& gray, int threshold, std::vector<std::uint64_t>& packed) {
    constexpr int samples_per_word = 64;
    constexpr int input_per_word = 2 * samples_per_word;
    const int words = (gray.cols + input_per_word - 1) / input_per_word;
    const int height = (gray.rows + 1) / 2;
    // Keep the runtime vector length scalar for baseline-target compiler builds.
    std::uint64_t vector_bytes;
    asm("cntb %0" : "=r"(vector_bytes));
    const int lanes = static_cast<int>(vector_bytes);
    const auto active = svptrue_b8();
    const auto level = static_cast<std::uint8_t>(threshold);
    packed.resize(static_cast<std::size_t>(words) * height);
    for (int y = 0; y < height; ++y) {
        const auto* input = gray.ptr<std::uint8_t>(2 * y);
        auto* output = reinterpret_cast<std::uint8_t*>(packed.data() + std::size_t(y) * words);
        int x = 0;
        for (; gray.cols - x >= 2 * lanes; x += 2 * lanes) {
            const auto a = svcmple_n_u8(active, svld1_u8(active, input + x), level);
            const auto b = svcmple_n_u8(active, svld1_u8(active, input + x + lanes), level);
            const auto even = svuzp1_b8(a, b);
            // ACLE has no byte-addressed store for a predicate register. The
            // architectural store writes exactly lanes/8 bytes, in sample order.
            asm volatile("str %0, [%1]" ::"Upa"(even), "r"(output + x / 16) : "memory");
        }
        // Each complete vector contains a multiple of eight samples. Clear
        // the final partial word, including unused bits on odd or strided ROIs.
        if (x / 16 < words * 8) {
            std::fill(output + x / 16, output + words * 8, 0);
            for (; x < gray.cols; x += 2)
                output[x / 16] |= static_cast<std::uint8_t>((input[x] <= level) << ((x / 2) % 8));
        }
    }
}
#endif

#if defined(__x86_64__) && defined(__GNUC__)
__attribute__((target("avx2,bmi2"))) void pack_stride_two_x86(const cv::Mat& gray, int threshold,
                                                              std::vector<std::uint64_t>& bits) {
    const int height = (gray.rows + 1) / 2;
    const int words = (gray.cols + 127) / 128;
    bits.resize(std::size_t(words) * height);
    const auto level = _mm256_set1_epi8(static_cast<char>(threshold));
    for (int y = 0; y < height; ++y) {
        const auto* row = gray.ptr<std::uint8_t>(2 * y);
        auto* output = bits.data() + std::size_t(words) * y;
        const auto mask = [&](const std::uint8_t* p) __attribute__((target("avx2,bmi2"))) {
            const auto samples = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
            // The unsigned minimum equals the sample exactly when it meets the cutoff.
            return static_cast<std::uint32_t>(_mm256_movemask_epi8(
                    _mm256_cmpeq_epi8(_mm256_min_epu8(samples, level), samples)));
        };
        const auto even = [&](const std::uint8_t* p) __attribute__((target("avx2,bmi2"))) {
            const std::uint64_t both = mask(p) | (std::uint64_t(mask(p + 32)) << 32);
            return _pext_u64(both, UINT64_C(0x5555555555555555));
        };
        int x = 0;
        for (; gray.cols - x >= 128; x += 128)
            *output++ = even(row + x) | (even(row + x + 64) << 32);
        if (x < gray.cols) {
            std::uint64_t tail = 0;
            for (int i = 0; i < gray.cols - x; i += 2)
                tail |= std::uint64_t(row[x + i] <= threshold) << (i / 2);
            *output = tail;
        }
    }
}
#endif

int first_set_bit(std::uint64_t bits) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(bits);
#else
    int index = 0;
    for (; (bits & 1U) == 0U; bits >>= 1)
        ++index;
    return index;
#endif
}

template <typename Submit>
void emit_word(std::uint64_t dark, int offset, int& start, std::uint64_t& carry,
               const Submit& submit) {
    std::uint64_t boundaries = dark ^ ((dark << 1) | carry);
    carry = dark >> 63;
    if (!boundaries)
        return;
    if (start >= 0) {
        submit(start, offset + first_set_bit(boundaries));
        boundaries &= boundaries - 1;
        start = -1;
    }
    while (boundaries) {
        start = offset + first_set_bit(boundaries);
        boundaries &= boundaries - 1;
        if (!boundaries)
            return;
        submit(start, offset + first_set_bit(boundaries));
        boundaries &= boundaries - 1;
        start = -1;
    }
}

template <typename Submit>
void native_runs(const std::uint8_t* row, int source_width, int threshold, const Submit& submit) {
    const int width = (source_width + 1) / 2;
    int x = 0;
    int start = -1;
#if defined(__aarch64__) && defined(__AARCH64EL__)
    const DarkWord bins(threshold);
    std::uint64_t carry = 0;
    for (; 2 * x + 128 <= source_width; x += 64)
        emit_word(bins.full<2>(row + 2 * x), x, start, carry, submit);
#endif
    for (; x < width; ++x) {
        if (row[2 * x] <= threshold) {
            if (start < 0)
                start = x;
        } else if (start >= 0) {
            submit(start, x);
            start = -1;
        }
    }
    if (start >= 0)
        submit(start, width);
}

// Only three source rows are needed by the cross. Outside neighbors are
// neutral, matching erosion of a threshold plane owned by this ROI.
class DarkCrossRows {
public:
    DarkCrossRows(const cv::Mat& gray, int threshold, std::vector<std::uint64_t>& scratch)
        : gray_(gray), threshold_(threshold), words_((gray.cols + 63) / 64), packed_(scratch) {}

    void prepare() {
        this->packed_.resize(3 * this->words_);
        this->pack(0);
    }

    template <typename Submit>
    void runs(int y, const Submit& submit) {
        const int below_y = std::min(y + 1, this->gray_.rows - 1);
        if (below_y > y)
            this->pack(below_y);
        const auto* above = this->row(std::max(y - 1, 0));
        const auto* current = this->row(y);
        const auto* below = this->row(below_y);
        const int last_word = this->words_ - 1;
        const std::uint64_t right_edge = std::uint64_t{1} << ((this->gray_.cols - 1) % 64);
        int start = -1;
        std::uint64_t carry = 0;
        for (int word = 0; word < this->words_; ++word) {
            const auto center = current[word];
            const auto left = (center << 1) | (word == 0 ? 1U : current[word - 1] >> 63);
            const auto right =
                    (center >> 1) | (word == last_word ? right_edge : current[word + 1] << 63);
            const auto dark = above[word] & below[word] & center & left & right;
            emit_word(dark, 64 * word, start, carry, submit);
        }
        if (start >= 0)
            submit(start, this->gray_.cols);
    }

private:
    void pack(int y) {
        auto* dst = this->row(y);
        const auto* src = this->gray_.ptr<std::uint8_t>(y);
        int x = 0;
#if defined(__aarch64__) && defined(__AARCH64EL__)
        const DarkWord bins(this->threshold_);
        for (; x + 64 <= this->gray_.cols; x += 64)
            dst[x / 64] = bins.full<1>(src + x);
#endif
        for (; x < this->gray_.cols; x += 64) {
            const int count = std::min(64, this->gray_.cols - x);
            std::uint64_t dark = 0;
            for (int i = 0; i < count; ++i)
                dark |= std::uint64_t{src[x + i] <= this->threshold_} << i;
            dst[x / 64] = dark;
        }
    }

    std::uint64_t* row(int y) { return this->packed_.data() + (y % 3) * this->words_; }

    const cv::Mat& gray_;
    const int threshold_, words_;
    std::vector<std::uint64_t>& packed_;
};

void append_extreme(std::vector<cv::Point>& chain, cv::Point point, int turn) {
    while (chain.size() >= 2) {
        const cv::Point last = chain.back();
        const cv::Point previous = chain[chain.size() - 2];
        const auto first = last - previous;
        const auto second = point - last;
        const auto cross = static_cast<std::int64_t>(first.x) * second.y -
                           static_cast<std::int64_t>(first.y) * second.x;
        if (turn * cross > 0)
            break;
        chain.pop_back();
    }
    chain.push_back(point);
}

// A shadow can cut one corner off the binary silhouette. Intersect the two
// long edges adjacent to the shortest side of that convex pentagon.
bool complete_bevel(std::vector<cv::Point>& polygon, double area) {
    int edge = 0;
    for (int j = 1; j < 5; ++j) {
        const auto a = polygon[(j + 1) % 5] - polygon[j];
        const auto b = polygon[(edge + 1) % 5] - polygon[edge];
        if (a.dot(a) < b.dot(b))
            edge = j;
    }
    const cv::Point2f a = polygon[edge], b = polygon[(edge + 1) % 5];
    const cv::Point2f u = a - cv::Point2f(polygon[(edge + 4) % 5]);
    const cv::Point2f v = cv::Point2f(polygon[(edge + 2) % 5]) - b;
    const double limit = .55 * std::sqrt(area);
    const double denominator = u.cross(v);
    if (cv::norm(b - a) >= limit || std::abs(denominator) < 1e-5)
        return false;
    const double t = (b - a).cross(v) / denominator;
    const cv::Point2f point = a + u * t;
    if (t < 0 || (point - b).dot(v) > 0 || cv::norm(point - a) >= limit ||
        cv::norm(point - b) >= limit)
        return false;
    polygon[edge] = {cvRound(point.x), cvRound(point.y)};
    polygon.erase(polygon.begin() + (edge + 1) % 5);
    return true;
}

}  // namespace

void ComponentCandidates::Stats::merge(const Stats& other) {
    this->left_ = std::min(this->left_, other.left_);
    this->top_ = std::min(this->top_, other.top_);
    this->right_ = std::max(this->right_, other.right_);
    this->bottom_ = std::max(this->bottom_, other.bottom_);
    this->area_ += other.area_;
}

void ComponentCandidates::Envelope::clear() {
    this->row_ = -1;
    this->left_chain_.clear();
    this->right_chain_.clear();
}

void ComponentCandidates::Envelope::append(const Run& run) {
    if (run.row_ != this->row_) {
        this->flush();
        this->row_ = run.row_;
        this->left_ = run.left_;
    }
    this->right_ = run.end_ - 1;
}

void ComponentCandidates::Envelope::flush() {
    if (this->row_ < 0)
        return;
    append_extreme(this->left_chain_, {this->left_, this->row_}, -1);
    append_extreme(this->right_chain_, {this->right_, this->row_}, 1);
}

int ComponentCandidates::root(int label) {
    while (label != this->parents_[label]) {
        this->parents_[label] = this->parents_[this->parents_[label]];
        label = this->parents_[label];
    }
    return label;
}

void ComponentCandidates::extract(const cv::Mat& mask, const ComponentConfig& config,
                                  std::vector<std::vector<cv::Point>>& candidates,
                                  SupplementalComponents* supplemental) {
    this->extract_impl<Input::binary>(mask, config, candidates, supplemental, 0);
}

bool ComponentCandidates::native_stride2_available([[maybe_unused]] int width) {
#if defined(__aarch64__) && defined(__AARCH64EL__)
    return true;
#elif defined(__x86_64__) && defined(__GNUC__)
    static const bool available = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("bmi2");
    return available && width >= 256;
#else
    return false;
#endif
}

static bool pack_accelerated(const cv::Mat& gray, int threshold,
                             std::vector<std::uint64_t>& packed) {
    if (gray.dims != 2 || gray.type() != CV_8UC1 || gray.rows <= 0 || threshold < 0 ||
        threshold > 255)
        return false;
#if defined(__aarch64__) && defined(__AARCH64EL__) && defined(__linux__) && __has_include(<arm_sve.h>)
    static const bool has_sve = (getauxval(AT_HWCAP) & HWCAP_SVE) != 0;
    if (has_sve && gray.cols >= 256) {
        pack_native_stride_two(gray, threshold, packed);
        return true;
    }
#endif
#if defined(__x86_64__) && defined(__GNUC__)
    if (ComponentCandidates::native_stride2_available(gray.cols)) {
        pack_stride_two_x86(gray, threshold, packed);
        return true;
    }
#endif
    return false;
}

bool ComponentCandidates::pack_trace_samples(const cv::Mat& gray, int threshold,
                                             std::vector<std::uint64_t>& packed) {
    if (pack_accelerated(gray, threshold, packed))
        return true;
#if defined(__aarch64__) && defined(__AARCH64EL__)
    if (gray.dims != 2 || gray.type() != CV_8UC1 || gray.rows <= 0 || gray.cols <= 0 ||
        threshold < 0 || threshold > 255)
        return false;
    const int width = (gray.cols + 1) / 2;
    const int height = (gray.rows + 1) / 2;
    const int words = (width + 63) / 64;
    packed.resize(std::size_t(words) * height);
    const DarkWord pack(threshold);
    for (int y = 0; y < height; ++y) {
        const auto* row = gray.ptr<std::uint8_t>(2 * y);
        auto* bits = packed.data() + std::size_t(y) * words;
        int x = 0;
        for (; 2 * (x + 64) <= gray.cols; x += 64)
            bits[x / 64] = pack.full<2>(row + 2 * x);
        if (x < width) {
            std::uint64_t end = 0;
            for (int i = x; i < width; ++i)
                end |= std::uint64_t(row[2 * i] <= threshold) << (i - x);
            bits[x / 64] = end;
        }
    }
    return true;
#else
    return false;
#endif
}

void ComponentCandidates::extract_native_stride2(const cv::Mat& gray, int threshold,
                                                 const ComponentConfig& config,
                                                 std::vector<std::vector<cv::Point>>& candidates) {
    if (pack_accelerated(gray, threshold, this->packed_rows_)) {
        const cv::Size extent((gray.cols + 1) / 2, (gray.rows + 1) / 2);
        this->extract_impl<Input::packed>(gray(cv::Rect({}, extent)), config, candidates, nullptr,
                                          0, this->packed_rows_.data());
        return;
    }
    this->extract_impl<Input::stride_two>(gray, config, candidates, nullptr, threshold);
}

void ComponentCandidates::extract_dark_cross(const cv::Mat& gray, int threshold,
                                             const ComponentConfig& config,
                                             std::vector<std::vector<cv::Point>>& candidates) {
    CV_Assert(!gray.empty() && threshold >= 0 && threshold <= 255);
    this->extract_impl<Input::dark_cross>(gray, config, candidates, nullptr, threshold);
}

void ComponentCandidates::extract_packed(const cv::Mat& gray,
                                         const std::vector<std::uint64_t>& bits,
                                         const ComponentConfig& config,
                                         std::vector<std::vector<cv::Point>>& candidates,
                                         SupplementalComponents* supplemental) {
    CV_Assert(gray.dims == 2 && !gray.empty() &&
              bits.size() == static_cast<std::size_t>((gray.cols + 63) / 64) * gray.rows);
    this->extract_impl<Input::packed>(gray, config, candidates, supplemental, 0, bits.data());
}

template <ComponentCandidates::Input kInput>
void ComponentCandidates::extract_impl(const cv::Mat& input, const ComponentConfig& config,
                                       std::vector<std::vector<cv::Point>>& candidates,
                                       SupplementalComponents* supplemental, int threshold,
                                       const std::uint64_t* bits) {
    constexpr bool kNative = kInput == Input::stride_two;
    const int width = kNative ? (input.cols + 1) / 2 : input.cols;
    const int height = kNative ? (input.rows + 1) / 2 : input.rows;
    const int minimum_side = config.minimum_quad_side_samples_;
    const int maximum_side = config.maximum_quad_side_samples_;
    CV_Assert(input.dims == 2 && input.type() == CV_8UC1 && width < 32768 && height < 32768 &&
              minimum_side >= 3 && minimum_side <= maximum_side && maximum_side < 32768);
    this->runs_.clear();
    this->parents_.clear();
    this->stats_.clear();
    int previous_begin = 0;
    int previous_end = 0;
    DarkCrossRows cross(input, threshold, this->packed_rows_);
    if constexpr (kInput == Input::dark_cross)
        cross.prepare();
    for (int y = 0; y < height; ++y) {
        const auto* row = input.ptr<std::uint8_t>(kNative ? 2 * y : y);
        const int words = (width + 63) / 64;
        const auto* below =
                kInput == Input::packed && width >= 512 && !supplemental && y + 1 < height
                        ? bits + std::size_t(y + 1) * words
                        : nullptr;
        int previous = previous_begin;
        const int begin = static_cast<int>(this->runs_.size());
        const auto submit = [&](int x, int end) {
            // Run ends are exclusive. Equality still connects diagonal neighbors.
            while (previous < previous_end && this->runs_[previous].end_ < x)
                ++previous;
            int label = -1;
            for (int j = previous; j < previous_end && this->runs_[j].left_ <= end; ++j) {
                const int adjacent = this->root(this->runs_[j].label_);
                if (label < 0) {
                    label = adjacent;
                } else if (adjacent != label) {
                    this->parents_[adjacent] = label;
                    this->stats_[label].merge(this->stats_[adjacent]);
                }
            }
            if (label < 0) {
                if constexpr (kInput == Input::packed) {
                    constexpr int kShort = 4;
                    if (below && end - x <= kShort) {
                        const int left = std::max(0, x - 1);
                        const int right = std::min(width, end + 1);
                        const int within = left % 64;
                        const int count = right - left;
                        // No adjacent run above or below gives a one-row
                        // component, which cannot pass the configured minimum.
                        if (within + count <= 64 &&
                            ((below[left / 64] >> within) & ((std::uint64_t{1} << count) - 1)) == 0)
                            return;
                    }
                }
                label = static_cast<int>(this->parents_.size());
                this->parents_.push_back(label);
                this->stats_.push_back({x, y, end, y + 1, end - x, -1});
            } else {
                auto& stats = this->stats_[label];
                stats.left_ = std::min(stats.left_, x);
                stats.right_ = std::max(stats.right_, end);
                // Runs are visited by increasing row, including across a union.
                stats.bottom_ = y + 1;
                stats.area_ += end - x;
            }
            this->runs_.push_back({x, end, y, label});
        };
        if constexpr (kNative) {
            native_runs(row, input.cols, threshold, submit);
        } else if constexpr (kInput == Input::dark_cross) {
            cross.runs(y, submit);
        } else if constexpr (kInput == Input::packed) {
            const auto* packed_row = bits + std::size_t(y) * words;
            std::uint64_t carry = 0;
            int start = -1;
            for (int i = 0; i < words; ++i)
                emit_word(packed_row[i], 64 * i, start, carry, submit);
            if (start >= 0)
                submit(start, width);
        } else {
            for (int x = 0; x < width;) {
                x = skip_equal(row, x, width, 0);
                if (x == width)
                    break;
                const int end = skip_equal(row, x, width, 255);
                CV_Assert(end > x);
                submit(x, end);
                x = end;
            }
        }
        previous_begin = begin;
        previous_end = static_cast<int>(this->runs_.size());
    }

    if (supplemental) {
        supplemental->polygons_.clear();
        supplemental_envelopes_.clear();
    }
    const double minimum_fill =
            supplemental ? supplemental->minimum_fill_ratio_ : config.minimum_fill_ratio_;
    int count = 0;
    for (int i = 0; i < static_cast<int>(this->stats_.size()); ++i) {
        auto& stats = this->stats_[i];
        if (i != this->parents_[i] || stats.area_ < minimum_side * minimum_side / 4)
            continue;
        const int width = stats.right_ - stats.left_;
        const int height = stats.bottom_ - stats.top_;
        const int shorter_side = std::min(width, height);
        const int longer_side = std::max(width, height);
        if (shorter_side >= minimum_side && longer_side <= maximum_side &&
            longer_side <= config.maximum_aspect_ratio_ * shorter_side &&
            stats.area_ >= minimum_fill * width * height &&
            stats.area_ <= config.maximum_fill_ratio_ * width * height) {
            stats.envelope_ = count++;
            if (supplemental)
                supplemental_envelopes_.push_back(stats.area_ <
                                                  config.minimum_fill_ratio_ * width * height);
        }
    }
    candidates.clear();
    if (count == 0)
        return;
    if (this->envelopes_.size() < static_cast<std::size_t>(count))
        this->envelopes_.resize(count);
    for (int i = 0; i < count; ++i)
        this->envelopes_[i].clear();
    for (int i = 0; i < static_cast<int>(this->parents_.size()); ++i)
        this->parents_[i] = this->root(i);
    if (kInput == Input::packed && width >= 512) {
        // Parent links have been flattened and are no longer needed by the scan.
        for (auto& label : this->parents_)
            label = this->stats_[label].envelope_;
        for (const Run& run : this->runs_) {
            const int envelope = this->parents_[run.label_];
            if (envelope >= 0)
                this->envelopes_[envelope].append(run);
        }
    } else {
        for (const Run& run : this->runs_) {
            const int envelope = this->stats_[this->parents_[run.label_]].envelope_;
            if (envelope >= 0)
                this->envelopes_[envelope].append(run);
        }
    }

    for (int i = 0; i < count; ++i) {
        auto& envelope = this->envelopes_[i];
        envelope.flush();
        this->hull_.assign(envelope.left_chain_.begin(), envelope.left_chain_.end());
        this->hull_.insert(this->hull_.end(), envelope.right_chain_.rbegin(),
                           envelope.right_chain_.rend());
        const float area = static_cast<float>(cv::contourArea(this->hull_));
        if (area < 5)
            continue;
        const double epsilon = config.hull_tolerance_ratio_ * cv::arcLength(this->hull_, true);
        if (this->hull_.size() >= 4 && this->hull_.size() <= kClosedPathMaximumVertices &&
            maximum_side <= kClosedPathMaximumExtent) {
            const bool accepted =
                    supplemental
                            ? approximate_closed_bounded<4, 5>(this->hull_, epsilon, this->quad_)
                            : approximate_closed_quad(this->hull_, epsilon, this->quad_);
            if (!accepted)
                continue;
        } else {
            cv::approxPolyDP(this->hull_, this->quad_, epsilon, true);
        }
        bool extra = supplemental && supplemental_envelopes_[i];
        if (supplemental && this->quad_.size() == 5)
            extra = complete_bevel(this->quad_, area);
        if (this->quad_.size() == 4 &&
            cv::contourArea(this->quad_) > config.minimum_quad_hull_ratio_ * area) {
            auto& output = extra ? supplemental->polygons_ : candidates;
            output.push_back(this->quad_);
        }
    }
}

}  // namespace aruco3cuda::apriltag_small_cpu::detail
