// SPDX-License-Identifier: Apache-2.0
#include "cpu_edge_regions.hpp"

#include <algorithm>
#include <cstdlib>

namespace aruco3cuda::cpu::detail {
namespace {
constexpr int kMinimumRunPx = 8;
constexpr int kMaximumRunPx = 640;
constexpr int kRegionsPerSide = 2;
constexpr std::size_t kRegionsPerImage = 6;

struct Boundary {
    const cv::Mat& gray_;
    bool vertical_;
    bool far_;
    int length_;
    int depth_;

    int pixel(int along, int inward) const {
        const int across = this->far_ ? this->depth_ - 1 - inward : inward;
        return this->vertical_ ? this->gray_.ptr<unsigned char>(along)[across]
                               : this->gray_.ptr<unsigned char>(across)[along];
    }

    cv::Rect region(int first, int end) const {
        const int span = end - first;
        const int margin = std::max(16, span / 4);
        const int start = std::max(0, first - margin);
        const int extent = std::min(this->length_, end + margin) - start;
        const int inward = std::min(this->depth_, std::max(42, 3 * span / 2));
        const int across = this->far_ ? this->depth_ - inward : 0;
        return this->vertical_ ? cv::Rect(across, start, inward, extent)
                               : cv::Rect(start, across, extent, inward);
    }
};

struct Run {
    int first_;
    int end_;
    int contrast_;
    int threshold_;

    int score() const { return this->contrast_ * std::min(this->end_ - this->first_, 100); }
};

std::vector<Run> find_runs(const Boundary& boundary, std::vector<unsigned char>& samples) {
    const int step = boundary.length_ >= 1000 ? 4 : 2;
    const int count = boundary.length_ / step;
    samples.resize(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        samples[i] = static_cast<unsigned char>(
                std::min(boundary.pixel(i * step, 0), boundary.pixel(i * step, 1)));
    }

    std::vector<Run> runs;
    for (int first = 3; first + 5 < count;) {
        const int left =
                std::min(samples[first - 1], std::max(samples[first - 2], samples[first - 3]));
        const int dark = std::min(samples[first], samples[first + 1]);
        const int threshold = (left + dark) / 2;
        if (left - dark < 20 || samples[first + 1] >= threshold) {
            ++first;
            continue;
        }
        int end = first + 2;
        int sum = samples[first] + samples[first + 1];
        for (; end + 2 < count && samples[end] < threshold; ++end)
            sum += samples[end];
        if (end + 2 >= count)
            break;

        const int right = std::min(samples[end], std::max(samples[end + 1], samples[end + 2]));
        const int light = std::min(left, right);
        const int mean = sum / (end - first);
        const int contrast = light - mean;
        const int span = (end - first) * step;
        const int split = (light + mean) / 2;
        if (span >= kMinimumRunPx && span <= kMaximumRunPx && std::abs(left - right) <= 85 &&
            contrast >= std::max(26, light / 4) &&
            boundary.pixel((first + end) * step / 2,
                           std::min(boundary.depth_ - 1, std::max(2, span / 24))) < split) {
            runs.push_back({first * step, end * step, contrast, split});
        }
        first = end;
    }
    std::stable_sort(runs.begin(), runs.end(),
                     [](const Run& a, const Run& b) { return a.score() > b.score(); });
    return runs;
}

bool supported_flanks(const Boundary& boundary, const Run& run) {
    const int span = run.end_ - run.first_;
    const int radius = std::clamp(span / 16, 3, 10);
    if (run.first_ < radius || run.end_ + radius >= boundary.length_ || 2 * radius >= span)
        return false;
    const int depth = std::min((boundary.depth_ - 1) / 2, std::clamp(span / 24, 2, 10));
    int support = 0;
    for (const int inward : {depth, 2 * depth}) {
        support += boundary.pixel(run.first_ - radius, inward) > run.threshold_ &&
                   boundary.pixel(run.first_ + radius, inward) < run.threshold_;
        support += boundary.pixel(run.end_ + radius, inward) > run.threshold_ &&
                   boundary.pixel(run.end_ - radius, inward) < run.threshold_;
    }
    return support >= 3;
}

bool supported_interior(const Boundary& boundary, const Run& run) {
    const int span = run.end_ - run.first_;
    // A clipped, roughly square tag has light and dark cells before an in-frame
    // border. Check both cues to avoid spending a full decode on isolated dark bands.
    int dark = 0;
    for (const int y : {22, 37, 50, 64}) {
        const int inward = span * y / 100;
        if (inward >= boundary.depth_)
            return false;
        for (const int x : {22, 37, 50, 64})
            dark += boundary.pixel(run.first_ + span * x / 100, inward) < run.threshold_;
    }
    if (dark < 3 || dark > 13)
        return false;

    const int rim = std::clamp(span / 24, 2, 9);
    const int step = std::clamp(span / 32, 1, 6);
    for (int inward = span * 7 / 10; inward + rim < boundary.depth_ && inward < span * 14 / 10;
         inward += step) {
        int support = 0;
        for (const int x : {23, 39, 61, 77}) {
            const int along = run.first_ + span * x / 100;
            support += boundary.pixel(along, inward - rim) < run.threshold_ &&
                       boundary.pixel(along, inward + rim) > run.threshold_;
        }
        if (support >= 3)
            return true;
    }
    return false;
}
}  // namespace

std::vector<cv::Rect> edge_regions(const cv::Mat& gray, std::size_t max_pixels) {
    if (gray.empty() || gray.dims != 2 || gray.type() != CV_8UC1 || gray.cols < 24 ||
        gray.rows < 24 || gray.cols > 65536 || gray.rows > 65536 ||
        gray.step[0] < static_cast<std::size_t>(gray.cols) || max_pixels < 24U * kMinimumRunPx)
        return {};

    std::vector<cv::Rect> regions;
    std::vector<unsigned char> samples;
    for (const bool vertical : {true, false}) {
        for (const bool far : {false, true}) {
            const Boundary boundary{gray, vertical, far, vertical ? gray.rows : gray.cols,
                                    vertical ? gray.cols : gray.rows};
            int accepted = 0;
            for (const Run& run : find_runs(boundary, samples)) {
                const cv::Rect region = boundary.region(run.first_, run.end_);
                const auto area = static_cast<std::size_t>(region.width) * region.height;
                if (area > max_pixels || !supported_flanks(boundary, run) ||
                    !supported_interior(boundary, run))
                    continue;
                regions.push_back(region);
                max_pixels -= area;
                if (regions.size() == kRegionsPerImage || max_pixels < 24U * kMinimumRunPx)
                    return regions;
                if (++accepted == kRegionsPerSide)
                    break;
            }
        }
    }
    return regions;
}
}  // namespace aruco3cuda::cpu::detail
