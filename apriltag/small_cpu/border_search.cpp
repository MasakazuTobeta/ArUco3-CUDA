// SPDX-License-Identifier: Apache-2.0
#include "border_search.hpp"
#include "aruco3cuda/dictionary.hpp"
#include "cpu_grid_refine.hpp"
#include "grid_alignment.hpp"
#include "tag_decode.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace aruco3cuda::generated {
extern const DictionaryTable kDictApriltag36h11Table;
}

namespace aruco3cuda::apriltag_small_cpu::detail {
namespace {
using aruco3cuda::MarkerCode;
using aruco3cuda::generated::kDictApriltag36h11Table;
using Result = Detection;
constexpr MarkerCode all = (MarkerCode{1} << 36) - 1;
struct Check {
    int index = -1, errors = 100;
};
class Decode {
    using Bucket = std::vector<std::uint16_t>;
    std::array<std::array<Bucket, 64>, 6> words_;

public:
    Decode() {
        for (int i = 0; i < kDictApriltag36h11Table.code_count_ * 4; ++i)
            for (int g = 0; g < 6; ++g)
                this->words_[g][chunk(kDictApriltag36h11Table.codes_[i], g)].push_back(i);
    }
    Check operator()(const cv::Mat& gray, const Quad& quad, const BorderConfig& config,
                     int correction_limit) const {
        Projection p;
        if (!project(quad, p))
            return {};
        std::array<std::array<LumaSample, 32>, 2> ring;
        int positive = 0, visible = 0;
        for (int i = 0; i < 8; ++i) {
            const float t = (i + .5F) / 8;
            for (int s = 0; s < 4; ++s) {
                for (int c = 0; c < 2; ++c) {
                    const float in = c ? -.0625F : .0625F;
                    const float u = s == 0 ? in : s == 2 ? 1 - in : t;
                    const float v = s == 1 ? in : s == 3 ? 1 - in : t;
                    ring[c][i * 4 + s] = {u - .5F, v - .5F, p.sample(gray, u, v)};
                }
                const float a = ring[0][4 * i + s].intensity, b = ring[1][4 * i + s].intensity;
                if (a >= 0 && b >= 0) {
                    ++visible;
                    positive += b - a >= 12;
                }
            }
        }
        if (positive < config.minimum_ring_support_ || positive * 5 < visible * 3)
            return {};
        Plane dark, light;
        if (!fit_plane(ring[0], dark) || !fit_plane(ring[1], light) ||
            light.center - dark.center < 16)
            return {};
        MarkerCode known = 0, code = 0;
        for (int y = 0; y < 6; ++y)
            for (int x = 0; x < 6; ++x) {
                const float u = (x + 1.5F) / 8, v = (y + 1.5F) / 8;
                const float measured = p.sample(gray, u, v);
                if (measured < 0)
                    continue;
                known |= MarkerCode{1} << (6 * y + x);
                code |= MarkerCode{measured >
                                   .5F * (dark.at(u - .5F, v - .5F) + light.at(u - .5F, v - .5F))}
                        << (6 * y + x);
            }
        // Payload samples outside the visible image are erasures. Narrow the
        // correction budget and require a unique dictionary match on observed bits.
        const int unknown = bit_count(all ^ known);
        if (36 - unknown < config.minimum_known_cells_)
            return {};
        const int budget = std::clamp((9 - unknown) / 2, 0, std::min(2, correction_limit));
        int index = -1, distance = budget + 1;
        bool ambiguous = false;
        const auto consider = [&](int i) {
            const int error = bit_count((kDictApriltag36h11Table.codes_[i] ^ code) & known);
            if (error < distance) {
                distance = error;
                index = i;
                ambiguous = false;
            } else if (error == distance && index >= 0 && index != i)
                ambiguous = true;
        };
        int intact = 0;
        for (int g = 0; g < 6; ++g)
            intact += chunk(known, g) == 63;
        if (intact > budget) {
            for (int g = 0; g < 6; ++g)
                if (chunk(known, g) == 63)
                    for (auto i : this->words_[g][chunk(code, g)])
                        consider(i);
        } else {
            for (int i = 0; i < kDictApriltag36h11Table.code_count_ * 4; ++i)
                consider(i);
        }
        if (ambiguous || index < 0)
            return {};
        return {index, distance};
    }

private:
    static int chunk(MarkerCode x, int g) { return (x >> (6 * g)) & 63; }
};

Point center(const Quad& q) {
    return .25F * (q[0] + q[1] + q[2] + q[3]);
}
bool nearby(const Result& a, const Result& b) {
    return length(center(a.corners_px_) - center(b.corners_px_)) <=
           .32F * std::min(length(a.corners_px_[0] - a.corners_px_[1]),
                           length(b.corners_px_[0] - b.corners_px_[1]));
}
Quad expansion(const Quad& q, float offset) {
    const auto origin = center(q);
    Quad out;
    const auto side = .25F * (length(q[0] - q[1]) + length(q[1] - q[2]) + length(q[2] - q[3]) +
                              length(q[3] - q[0]));
    for (int i = 0; i < 4; ++i)
        out[i] = origin + (1 + offset / std::max(1.F, side)) * (q[i] - origin);
    return out;
}

}  // namespace

void validate_border_config(const BorderConfig& config) {
    if (config.luma_threshold_ < 0 || config.luma_threshold_ > 255 ||
        config.maximum_strip_depth_px_ < 24 || config.maximum_strip_depth_px_ > 1024 ||
        config.maximum_clip_depth_px_ < 0 ||
        config.maximum_clip_depth_px_ >= config.maximum_strip_depth_px_ ||
        config.maximum_candidates_per_side_ < 1 || config.maximum_candidates_per_side_ > 64 ||
        config.maximum_window_pixels_per_side_ < 24 * 24 ||
        config.maximum_window_pixels_per_side_ > 1024 * 32768 || config.minimum_dark_run_px_ < 1 ||
        config.minimum_dark_run_px_ > 32 || config.window_margin_px_ < 8 ||
        config.window_margin_px_ > 1024 || config.minimum_ring_support_ < 1 ||
        config.minimum_ring_support_ > 32 || config.minimum_known_cells_ < 24 ||
        config.minimum_known_cells_ > 36 || config.grid_refinement_passes_ < 0 ||
        config.grid_refinement_passes_ > 4)
        throw std::invalid_argument("Invalid AprilTag border candidate parameters");
}

class BorderSearch::Impl {
public:
    explicit Impl(const BorderConfig& config, int correction_limit)
        : config_(config), correction_limit_(correction_limit) {}
    void align_visible_tags(const cv::Mat& gray, std::vector<Result>& results, float margin_ratio) {
        for (auto& tag : results) {
            const auto close = [&](Point pt) {
                return pt.x < 8 || pt.y < 8 || pt.x >= gray.cols - 8 || pt.y >= gray.rows - 8;
            };
            const auto adopt = [&](const Quad& fit) {
                const auto check = decode_(gray, fit, config_, correction_limit_);
                if (check.index == 4 * tag.id_ && check.errors <= tag.corrected_bits_) {
                    tag.corners_px_ = fit;
                    tag.corrected_bits_ = check.errors;
                }
            };
            Quad fit;
            if (std::any_of(tag.corners_px_.begin(), tag.corners_px_.end(), close) &&
                propose_visible_grid(gray, tag, fit))
                adopt(fit);
            if (margin_ratio <= 0)
                continue;
            float clearance = std::numeric_limits<float>::max();
            for (const auto q : tag.corners_px_)
                clearance =
                        std::min({clearance, q.x, q.y, gray.cols - 1 - q.x, gray.rows - 1 - q.y});
            const float side = length(tag.corners_px_[1] - tag.corners_px_[0]);
            if (clearance < 1 || clearance >= std::max(8.F, margin_ratio * side))
                continue;
            fit = tag.corners_px_;
            if (refine_quad(gray, fit, false))
                adopt(fit);
        }
    }
    void append(const cv::Mat& gray, std::vector<Result>& results) {
        for (int side = 0; side < 4; ++side) {
            const int depth = std::min(this->config_.maximum_strip_depth_px_,
                                       (side % 2 ? gray.cols : gray.rows) / 3);
            if (depth < 24)
                continue;
            const int extent = side % 2 ? gray.rows : gray.cols;
            int remaining = this->config_.maximum_window_pixels_per_side_;
            int candidates = this->config_.maximum_candidates_per_side_;
            for (const auto [first, end] : this->windows(gray, side, extent)) {
                const int pixels = depth * (end - first);
                if (pixels > remaining)
                    continue;
                remaining -= pixels;
                this->scan_window(gray, side, depth, first, end, candidates, results);
                if (candidates == 0)
                    break;
            }
        }
    }

private:
    struct Window {
        int first, end;
    };
    const std::vector<Window>& windows(const cv::Mat& gray, int side, int extent) {
        auto& result = this->windows_;
        result.clear();
        const auto pixel = [&](int along, int inward) {
            switch (side) {
                case 0:
                    return gray.ptr<std::uint8_t>(inward)[along];
                case 1:
                    return gray.ptr<std::uint8_t>(along)[gray.cols - 1 - inward];
                case 2:
                    return gray.ptr<std::uint8_t>(gray.rows - 1 - inward)[gray.cols - 1 - along];
                default:
                    return gray.ptr<std::uint8_t>(gray.rows - 1 - along)[inward];
            }
        };
        for (int y : {0, 4, 7}) {
            int x = 0;
            while (x < extent) {
                while (x < extent && pixel(x, y) > this->config_.luma_threshold_)
                    ++x;
                const int start = x;
                while (x < extent && pixel(x, y) <= this->config_.luma_threshold_)
                    ++x;
                const int width = x - start;
                if (width < this->config_.minimum_dark_run_px_ ||
                    width > this->config_.maximum_strip_depth_px_ * 3 / 2)
                    continue;
                result.push_back({std::max(0, start - this->config_.window_margin_px_),
                                  std::min(extent, x + this->config_.window_margin_px_)});
            }
        }
        std::sort(result.begin(), result.end(),
                  [](const Window& a, const Window& b) { return a.first < b.first; });
        std::size_t out = 0;
        for (const auto region : result) {
            if (out && region.first <= result[out - 1].end)
                result[out - 1].end = std::max(result[out - 1].end, region.end);
            else
                result[out++] = region;
        }
        result.resize(out);
        return result;
    }
    void scan_window(const cv::Mat& gray, int side, int depth, int first, int end, int& candidates,
                     std::vector<Result>& results) {
        const int extent = end - first;
        cv::Mat local;
        switch (side) {
            case 0:
                local = gray(cv::Rect(first, 0, extent, depth));
                break;
            case 1:
                cv::rotate(gray(cv::Rect(gray.cols - depth, first, depth, extent)), this->rotated_,
                           cv::ROTATE_90_COUNTERCLOCKWISE);
                local = this->rotated_;
                break;
            case 2:
                cv::rotate(gray(cv::Rect(gray.cols - end, gray.rows - depth, extent, depth)),
                           this->rotated_, cv::ROTATE_180);
                local = this->rotated_;
                break;
            case 3:
                cv::rotate(gray(cv::Rect(0, gray.rows - end, depth, extent)), this->rotated_,
                           cv::ROTATE_90_CLOCKWISE);
                local = this->rotated_;
                break;
        }
        this->offset_ = first;
        this->propose(gray, local, side, candidates, results);
    }
    Point global(Point p, const cv::Mat& img, int side) {
        p.x += this->offset_;
        switch (side) {
            case 0:
                return p;
            case 1:
                return {img.cols - 1 - p.y, p.x};
            case 2:
                return {img.cols - 1 - p.x, img.rows - 1 - p.y};
            default:
                return {p.y, img.rows - 1 - p.x};
        }
    }
    void propose(const cv::Mat& img, const cv::Mat& local, int side, int& candidates,
                 std::vector<Result>& results) {
        cv::threshold(local, this->binary_, this->config_.luma_threshold_, 255,
                      cv::THRESH_BINARY_INV);
        cv::findContours(this->binary_, this->contours_, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        for (const auto& v : this->contours_) {
            if (v.size() < 4)
                continue;
            const auto rect = cv::boundingRect(v);
            if (this->offset_ > 0 && rect.x == 0)
                continue;
            if (this->offset_ + local.cols < (side % 2 ? img.rows : img.cols) &&
                rect.x + rect.width >= local.cols)
                continue;
            if (rect.y > 7 || rect.width < 16 || rect.height < 14 ||
                rect.width > this->config_.maximum_strip_depth_px_ * 1.5 ||
                rect.height >= local.rows - 2 || std::abs(cv::contourArea(v)) < 130)
                continue;
            if (candidates == 0)
                break;
            --candidates;
            cv::convexHull(v, this->hull_, false);
            const auto peri = cv::arcLength(this->hull_, true);
            for (float epsilon : {.016F, .028F}) {
                cv::approxPolyDP(this->hull_, this->polygon_, epsilon * peri, true);
                if (this->consider(img, side, rect, results))
                    break;
            }
        }
    }
    bool consider(const cv::Mat& img, int side, const cv::Rect& rect,
                  std::vector<Result>& results) {
        const int n = static_cast<int>(this->polygon_.size());
        if (n < 4 || n > 5)
            return false;
        if (cv::contourArea(this->polygon_, true) < 0)
            std::reverse(this->polygon_.begin(), this->polygon_.end());
        int missing = -1;
        if (rect.y == 0) {
            for (int i = 0; i < n; ++i)
                if (this->polygon_[i].y <= 1 && this->polygon_[(i + 1) % n].y <= 1) {
                    missing = i;
                    break;
                }
        }
        if (missing < 0) {
            if (n != 4)
                return false;
            Quad q;
            for (int i = 0; i < 4; ++i)
                q[i] = this->global(this->polygon_[i], img, side);
            return this->accept(img, q, results);
        }
        // Rotate the polygon so the segment along the frame is first. It is a
        // clipping artifact, so its endpoints do not define the hidden tag edge.
        const auto vertex = [&](int i) { return Point(this->polygon_[(missing + i + n) % n]); };
        const Point a = vertex(0), b = vertex(1), c = vertex(2), z = vertex(-1);
        if (n == 5) {
            const auto d = b - c, e = a - z;
            const float denom = cross(d, e);
            if (std::abs(denom) <= .12F * length(d) * length(e))
                return false;
            const auto apex = c + (cross(z - c, e) / denom) * d;
            if (apex.y > 2 || apex.y < -this->config_.maximum_clip_depth_px_ ||
                !(std::abs(apex.x - .5F * (a.x + b.x)) < rect.width))
                return false;
            const Quad q{this->global(apex, img, side), this->global(c, img, side),
                         this->global(vertex(3), img, side), this->global(z, img, side)};
            return this->accept(img, q, results);
        }
        if (n != 4 || c.y < 12 || z.y < 12)
            return false;
        const float estimate = .5F * (length(a - b) + length(z - c)) - .5F * (z.y + c.y);
        const float center_depth =
                std::clamp(estimate, 0.F, float(this->config_.maximum_clip_depth_px_));
        for (float d : {0.F, -3.F, 3.F, -6.F, 6.F, -10.F, 10.F, -14.F, 14.F, -18.F, 18.F}) {
            const float depth = center_depth + d;
            if (depth < -.5F || depth > this->config_.maximum_clip_depth_px_)
                continue;
            const float ay = a.y + depth, by = b.y + depth;
            const auto p = a + (a - z) * (ay / (z.y - a.y));
            const auto r = b + (b - c) * (by / (c.y - b.y));
            const Quad q{this->global(p, img, side), this->global(r, img, side),
                         this->global(c, img, side), this->global(z, img, side)};
            if (this->accept(img, q, results))
                return true;
        }
        return false;
    }
    bool accept(const cv::Mat& img, const Quad& source, std::vector<Result>& results) {
        if (std::min({length(source[0] - source[1]), length(source[1] - source[2]),
                      length(source[2] - source[3]), length(source[3] - source[0])}) < 12)
            return false;
        for (const auto& old : results)
            if (length(center(source) - center(old.corners_px_)) <
                .23F * length(source[0] - source[1]))
                return true;
        for (const float delta : {1.0F, 0.F, 2.0F, -1.0F}) {
            auto q = expansion(source, delta);
            auto code = this->decode_(img, q, this->config_, this->correction_limit_);
            if (code.index < 0)
                continue;
            const auto word = kDictApriltag36h11Table.codes_[code.index];
            if (this->config_.grid_refinement_passes_ > 0 &&
                (!config_.fit_only_corrected_candidates_ || code.errors > 0)) {
                cv::Matx<std::uint8_t, 8, 8> cells{};
                for (int y = 0; y < 6; ++y)
                    for (int x = 0; x < 6; ++x)
                        cells(y + 1, x + 1) = (word >> (6 * y + x)) & 1;
                const cv::Mat view(cells, false);
                this->grid_.assign(q.begin(), q.end());
                for (int trial = 0; trial < this->config_.grid_refinement_passes_; ++trial) {
                    if (!aruco3cuda::cpu::detail::fit_visible_payload_grid(img, view, 4,
                                                                           this->grid_))
                        break;
                    Quad improved;
                    std::copy(this->grid_.begin(), this->grid_.end(), improved.begin());
                    const auto check =
                            this->decode_(img, improved, this->config_, this->correction_limit_);
                    if (check.index != code.index || check.errors > code.errors)
                        break;
                    code = check;
                    q = improved;
                }
            }
            Quad oriented;
            for (int i = 0; i < 4; ++i)
                oriented[i] = q[(4 + i - code.index % 4) % 4];
            const Result result{code.index / 4, oriented, code.errors};
            if (std::none_of(results.begin(), results.end(),
                             [&](const auto& old) { return nearby(old, result); }))
                results.push_back(result);
            return true;
        }
        return false;
    }
    int offset_ = 0;
    Decode decode_;
    const BorderConfig config_;
    const int correction_limit_;
    std::vector<Window> windows_;
    cv::Mat rotated_, binary_;
    std::vector<std::vector<cv::Point>> contours_;
    std::vector<cv::Point> hull_, polygon_;
    std::vector<Point> grid_;
};

BorderSearch::BorderSearch(const BorderConfig& config, int correction_limit)
    : impl_(std::make_unique<Impl>(config, correction_limit)) {}
BorderSearch::~BorderSearch() = default;
void BorderSearch::append(const cv::Mat& gray, std::vector<Detection>& results) {
    this->impl_->append(gray, results);
}
void BorderSearch::align_visible(const cv::Mat& gray, std::vector<Detection>& results,
                                 float margin_ratio) {
    this->impl_->align_visible_tags(gray, results, margin_ratio);
}

}  // namespace aruco3cuda::apriltag_small_cpu::detail
