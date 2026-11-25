// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace aruco3cuda::apriltag_small_cpu::detail {

// Each scan line can contribute at most two hull vertices. The extrema allow
// two monotone sweeps without sorting the contour points.
class OutlineHull {
public:
    void build(const std::vector<cv::Point>& contour, const cv::Rect& box,
               std::vector<cv::Point>& hull) {
        const bool by_row = box.height <= box.width;
        const int lines = by_row ? box.height : box.width;
        constexpr int kMinimumRemovedPoints = 8;
        if (contour.size() <= static_cast<std::size_t>(2 * lines + kMinimumRemovedPoints)) {
            cv::convexHull(contour, hull);
            return;
        }
        if (by_row)
            this->build_impl<true>(contour, box.y, lines, hull);
        else
            this->build_impl<false>(contour, box.x, lines, hull);
    }

private:
    struct Extreme {
        int coordinate_ = 0;
        int index_ = -1;
        bool repeated_ = false;
    };

    struct Range {
        Extreme low_;
        Extreme high_;
    };

    template <bool by_row>
    void build_impl(const std::vector<cv::Point>& contour, int origin, int lines,
                    std::vector<cv::Point>& hull) {
        this->ranges_.assign(lines, {});
        const auto range_at = [&](cv::Point point) -> Range& {
            return this->ranges_[(by_row ? point.y : point.x) - origin];
        };
        const auto value_at = [](cv::Point point) { return by_row ? point.x : point.y; };
        for (int index = 0; index < static_cast<int>(contour.size()); ++index) {
            const auto point = contour[index];
            const int value = value_at(point);
            auto& range = range_at(point);
            if (range.low_.index_ < 0 || value < range.low_.coordinate_)
                range.low_ = {value, index, false};
            else if (value == range.low_.coordinate_)
                range.low_.repeated_ = true;
            if (range.high_.index_ < 0 || value > range.high_.coordinate_)
                range.high_ = {value, index, false};
            else if (value == range.high_.coordinate_)
                range.high_.repeated_ = true;
        }

        const auto append = [&](std::vector<Extreme>& chain, const Extreme& point) {
            while (chain.size() >= 2) {
                const cv::Point middle = contour[chain.back().index_];
                const cv::Point previous = contour[chain[chain.size() - 2].index_];
                const cv::Point a = middle - previous, b = contour[point.index_] - middle;
                const auto cross = std::int64_t{a.x} * b.y - std::int64_t{a.y} * b.x;
                if ((by_row ? -cross : cross) > 0)
                    break;
                chain.pop_back();
            }
            chain.push_back(point);
        };

        this->low_chain_.clear();
        this->high_chain_.clear();
        for (const auto& range : this->ranges_)
            if (range.low_.index_ >= 0)
                append(this->low_chain_, range.low_);
        for (auto range = this->ranges_.rbegin(); range != this->ranges_.rend(); ++range)
            if (range->high_.index_ >= 0)
                append(this->high_chain_, range->high_);
        const bool share_last = this->low_chain_.back().index_ == this->high_chain_.front().index_;
        const bool share_first = this->low_chain_.front().index_ == this->high_chain_.back().index_;
        if (this->high_chain_.size() > static_cast<std::size_t>(share_last + share_first))
            this->low_chain_.insert(this->low_chain_.end(), this->high_chain_.begin() + share_last,
                                    this->high_chain_.end() - share_first);

        if constexpr (by_row)
            std::reverse(this->low_chain_.begin(), this->low_chain_.end());

        const auto points = static_cast<int>(this->low_chain_.size());
        int rises = 0, least = 0, greatest = 0;
        bool unique = true;
        for (int i = 0; i < points; ++i) {
            const auto& point = this->low_chain_[i];
            unique &= !point.repeated_;
            rises += point.index_ < this->low_chain_[(i + 1) % points].index_;
            if (point.index_ < this->low_chain_[least].index_)
                least = i;
            if (point.index_ > this->low_chain_[greatest].index_)
                greatest = i;
        }

        if (points >= 3 && unique && (rises == 1 || rises == points - 1)) {
            // For circularly ordered unique contour indices, OpenCV starts
            // at the first point in their ascending or descending traversal.
            const int first = rises == 1 ? greatest : least;
            hull.clear();
            for (int i = first; i < points; ++i)
                hull.push_back(contour[this->low_chain_[i].index_]);
            for (int i = 0; i < first; ++i)
                hull.push_back(contour[this->low_chain_[i].index_]);
            return;
        }

        // Repeated hull vertices and nonmonotone contours use OpenCV's tie and
        // start rules. Preserve the order and multiplicity of their endpoints.
        this->extrema_.clear();
        for (const auto& point : contour) {
            const auto& range = range_at(point);
            const auto value = value_at(point);
            if (value == range.low_.coordinate_ || value == range.high_.coordinate_)
                this->extrema_.push_back(point);
        }
        cv::convexHull(this->extrema_, hull);
    }

    std::vector<Range> ranges_;
    std::vector<Extreme> low_chain_, high_chain_;
    std::vector<cv::Point> extrema_;
};

}  // namespace aruco3cuda::apriltag_small_cpu::detail
