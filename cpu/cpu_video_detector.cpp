// SPDX-License-Identifier: Apache-2.0
#include "cpu_video_detector.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace aruco3cuda::cpu {
namespace {

Status failure(Status status, const char* message, std::string* out_message) {
    if (out_message != nullptr)
        *out_message = message;
    return status;
}

bool valid_image(const cv::Mat& image) {
    if (image.empty() || image.type() != CV_8UC1 || image.dims != 2 || image.cols > 65536 ||
        image.rows > 65536 ||
        image.total() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        return false;
    const auto start = reinterpret_cast<std::uintptr_t>(image.data);
    const auto end = reinterpret_cast<std::uintptr_t>(image.dataend);
    const auto width = static_cast<std::size_t>(image.cols);
    const auto last_row = static_cast<std::size_t>(image.rows - 1);
    return image.step[1] == 1U && image.step[0] >= width && end >= start && end - start >= width &&
           (last_row == 0U || image.step[0] <= (end - start - width) / last_row);
}

cv::Rect expand(const cv::Rect& rectangle, int padding, const cv::Rect& frame) {
    return (rectangle + cv::Size(2 * padding, 2 * padding) - cv::Point(padding, padding)) & frame;
}

// Merge before applying the region budget. Overlapping searches would discover duplicate
// detections and could handle a shared marker with different crop geometry.
bool add_region(std::vector<cv::Rect>* regions, cv::Rect rectangle, const CpuVideoOptions& options,
                double area_limit) {
    for (std::size_t i = 0; i < regions->size();) {
        if (((*regions)[i] & rectangle).empty()) {
            ++i;
            continue;
        }
        rectangle |= (*regions)[i];
        (*regions)[i] = regions->back();
        regions->pop_back();
        i = 0;
    }
    regions->push_back(rectangle);
    if (regions->size() > options.max_rois_)
        return false;
    double area = 0.0;
    for (const auto& region : *regions)
        area += static_cast<double>(region.area());
    return area <= area_limit;
}

// Map nearest-neighbor samples back to source ranges. Each tile checks up to four
// samples per axis and adds context so a changed cell can propose an entire marker.
bool changed_regions(const cv::Mat& current, const cv::Mat& previous, const cv::Rect& frame,
                     const CpuVideoOptions& options, double area_limit,
                     std::vector<cv::Rect>* regions) {
    constexpr int kSamplesPerTile = 4;
    const auto cell_begin = [](int index, int extent, int samples) {
        return static_cast<int>(static_cast<std::int64_t>(index) * extent / samples);
    };
    for (int y = 0; y < current.rows; y += kSamplesPerTile) {
        const int y_end = std::min(y + kSamplesPerTile, current.rows);
        for (int x = 0; x < current.cols; x += kSamplesPerTile) {
            const int x_end = std::min(x + kSamplesPerTile, current.cols);
            bool changed = false;
            for (int sy = y; sy < y_end && !changed; ++sy) {
                const auto* now = current.ptr<std::uint8_t>(sy);
                const auto* prior = previous.ptr<std::uint8_t>(sy);
                for (int sx = x; sx < x_end; ++sx) {
                    if (std::abs(static_cast<int>(now[sx]) - static_cast<int>(prior[sx])) >=
                        options.change_threshold_) {
                        changed = true;
                        break;
                    }
                }
            }
            if (!changed)
                continue;
            const cv::Point first(cell_begin(x, frame.width, current.cols),
                                  cell_begin(y, frame.height, current.rows));
            const cv::Point last(cell_begin(x_end, frame.width, current.cols),
                                 cell_begin(y_end, frame.height, current.rows));
            if (!add_region(regions, expand(cv::Rect(first, last), options.padding_px_, frame),
                            options, area_limit))
                return false;
        }
    }
    return true;
}

void append_detections(CpuResult&& local, const cv::Rect& region, CpuResult* global) {
    global->candidate_count_ += local.candidate_count_;
    global->rejected_count_ += local.rejected_count_;
    for (std::size_t i = 0; i < local.ids_.size(); ++i) {
        auto& corners = local.corners_[i];
        for (auto& corner : corners) {
            corner.x += static_cast<float>(region.x);
            corner.y += static_cast<float>(region.y);
        }
        global->ids_.push_back(local.ids_[i]);
        global->corners_.push_back(std::move(corners));
    }
}

}  // namespace

Status CpuVideoDetector::initialize(const std::string& dictionary_name,
                                    const DetectorConfig& config, std::string* out_message,
                                    const CpuVideoOptions& options) {
    if (options.refresh_interval_ < 1 || options.tile_side_px_ < 4 ||
        options.tile_side_px_ > 65536 || options.change_threshold_ < 1 ||
        options.change_threshold_ > 255 || options.padding_px_ < 0 || options.padding_px_ > 65536 ||
        options.max_rois_ == 0U || options.max_rois_ > 1024U ||
        !std::isfinite(options.max_region_fraction_) || options.max_region_fraction_ <= 0.0 ||
        options.max_region_fraction_ > 1.0)
        return failure(Status::kInvalidConfig, "invalid video search bounds", out_message);
    try {
        CpuDetector full;
        Status status = full.initialize(dictionary_name, config, out_message);
        if (status != Status::kOk)
            return status;
        CpuDetector regional;
        status = regional.initialize(dictionary_name, config, out_message);
        if (status != Status::kOk)
            return status;
        this->full_ = std::move(full);
        this->regional_ = std::move(regional);
        this->options_ = options;
        this->reset();
        this->initialized_ = true;
        return Status::kOk;
    } catch (const std::exception& error) {
        return failure(Status::kInvalidConfig, error.what(), out_message);
    }
}

void CpuVideoDetector::reset() noexcept {
    this->previous_size_ = {};
    this->previous_corners_.clear();
    this->previous_ids_.clear();
    this->current_ids_.clear();
    this->previous_samples_.release();
    this->current_samples_.release();
    this->frames_since_full_ = 0;
}

Status CpuVideoDetector::detect(const cv::Mat& image, CpuVideoResult* out,
                                std::string* out_message) {
    if (!this->initialized_)
        return failure(Status::kNotInitialized, "video detector is not initialized", out_message);
    if (out == nullptr || !valid_image(image)) {
        this->reset();
        return out == nullptr
                       ? failure(Status::kInvalidArgument, "result must not be null", out_message)
                       : failure(Status::kInvalidImage, "expected a valid Gray8 frame",
                                 out_message);
    }

    try {
        const int step = this->options_.tile_side_px_ / 4;
        const cv::Size sample_size((image.cols + step - 1) / step, (image.rows + step - 1) / step);
        cv::resize(image, this->current_samples_, sample_size, 0.0, 0.0, cv::INTER_NEAREST);
        const cv::Rect frame(0, 0, image.cols, image.rows);
        const double area_limit = this->options_.max_region_fraction_ * frame.area();
        bool full = image.size() != this->previous_size_ ||
                    this->frames_since_full_ >= this->options_.refresh_interval_ - 1;
        std::vector<cv::Rect> regions;
        if (!full) {
            for (const auto& corners : this->previous_corners_) {
                if (!add_region(
                            &regions,
                            expand(cv::boundingRect(corners), this->options_.padding_px_, frame),
                            this->options_, area_limit)) {
                    full = true;
                    break;
                }
            }
            if (!full)
                full = !changed_regions(this->current_samples_, this->previous_samples_, frame,
                                        this->options_, area_limit, &regions);
        }

        CpuVideoResult result;
        if (!full) {
            for (const auto& region : regions) {
                CpuResult local;
                const Status status = this->regional_.detect(image(region), &local);
                if (status != Status::kOk) {
                    full = true;
                    break;
                }
                append_detections(std::move(local), region, &result.detections_);
            }
        }
        if (!full) {
            this->current_ids_ = result.detections_.ids_;
            std::sort(this->current_ids_.begin(), this->current_ids_.end());
            // Crops can miss a marker that was visible one frame earlier. Recheck its
            // disappearance with the full detector, including repeated copies of an ID.
            full = !std::includes(this->current_ids_.begin(), this->current_ids_.end(),
                                  this->previous_ids_.begin(), this->previous_ids_.end());
        }
        if (full) {
            result.detections_ = {};
            const Status status = this->full_.detect(image, &result.detections_, out_message);
            if (status != Status::kOk) {
                this->reset();
                return status;
            }
            this->current_ids_ = result.detections_.ids_;
            std::sort(this->current_ids_.begin(), this->current_ids_.end());
        }
        result.full_frame_ = full;
        result.roi_count_ = full ? 0U : regions.size();
        this->previous_corners_ = result.detections_.corners_;
        this->previous_size_ = image.size();
        std::swap(this->previous_samples_, this->current_samples_);
        std::swap(this->previous_ids_, this->current_ids_);
        this->frames_since_full_ = full ? 0 : this->frames_since_full_ + 1;
        *out = std::move(result);
        return Status::kOk;
    } catch (const std::exception& error) {
        this->reset();
        return failure(Status::kInvalidImage, error.what(), out_message);
    }
}

}  // namespace aruco3cuda::cpu
