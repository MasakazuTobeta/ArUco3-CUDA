// SPDX-License-Identifier: Apache-2.0
#include "cpu_detector.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core/utils/tls.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cpu_box_sample.hpp"
#include "cpu_candidates.hpp"
#include "cpu_component_candidates.hpp"
#include "cpu_contours.hpp"
#include "cpu_edge_regions.hpp"
#include "cpu_grid_refine.hpp"
#include "cpu_grouping.hpp"
#include "cpu_sparse_pyramid.hpp"
#include "cpu_threshold.hpp"

namespace aruco3cuda::cpu {
namespace {
using Clock = std::chrono::steady_clock;
using hybrid::average_module_size;
using hybrid::CandidateNode;
using hybrid::filter_too_close_candidates;
using hybrid::find_quad_candidates;
using hybrid::MarkerCandidate;
using hybrid::reorder_corners;

void message(std::string* out, const std::string& value) {
    if (out != nullptr) {
        *out = value;
    }
}

// Return the rows that can affect a threshold contour, with its full local window.
cv::Range informative_rows(const cv::Mat& image, int halo) {
    const auto* first = image.ptr<unsigned char>(0);
    const auto* last = image.ptr<unsigned char>(image.rows - 1);
    const auto uniform = [&](const unsigned char* row) {
        return std::memcmp(row, row + 1, static_cast<std::size_t>(image.cols - 1)) == 0;
    };
    int start = 0, end = image.rows;
    if (uniform(first)) {
        while (start < end && std::memcmp(first, image.ptr<unsigned char>(start), image.cols) == 0)
            ++start;
        if (start == end)
            return {0, 0};
    }
    if (uniform(last)) {
        while (end > start && std::memcmp(last, image.ptr<unsigned char>(end - 1), image.cols) == 0)
            --end;
    }
    return {std::max(0, start - halo), std::min(image.rows, end + halo)};
}

// Only an edge with real contrast can enclose a tag and its outer quiet zone.
bool edge_has_contrast(const cv::Mat& image, bool horizontal, bool last) {
    const int count = horizontal ? image.cols : image.rows;
    const std::size_t step = horizontal ? 1 : image.step[0];
    const auto* first = image.ptr<unsigned char>(horizontal && last ? image.rows - 1 : 0) +
                        (!horizontal && last ? image.cols - 1 : 0);
    int lowest = 255, highest = 0;
    for (int i = 0; i < count; ++i) {
        const int value = first[static_cast<std::size_t>(i) * step];
        lowest = std::min(lowest, value);
        highest = std::max(highest, value);
        if (highest - lowest >= 8)
            return true;
    }
    return false;
}

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

const std::map<std::string, int>& dictionary_table() {
    static const std::map<std::string, int> kTable = {
            {"DICT_4X4_50", cv::aruco::DICT_4X4_50},
            {"DICT_4X4_100", cv::aruco::DICT_4X4_100},
            {"DICT_4X4_250", cv::aruco::DICT_4X4_250},
            {"DICT_4X4_1000", cv::aruco::DICT_4X4_1000},
            {"DICT_5X5_50", cv::aruco::DICT_5X5_50},
            {"DICT_5X5_100", cv::aruco::DICT_5X5_100},
            {"DICT_5X5_250", cv::aruco::DICT_5X5_250},
            {"DICT_5X5_1000", cv::aruco::DICT_5X5_1000},
            {"DICT_6X6_50", cv::aruco::DICT_6X6_50},
            {"DICT_6X6_100", cv::aruco::DICT_6X6_100},
            {"DICT_6X6_250", cv::aruco::DICT_6X6_250},
            {"DICT_6X6_1000", cv::aruco::DICT_6X6_1000},
            {"DICT_7X7_50", cv::aruco::DICT_7X7_50},
            {"DICT_7X7_100", cv::aruco::DICT_7X7_100},
            {"DICT_7X7_250", cv::aruco::DICT_7X7_250},
            {"DICT_7X7_1000", cv::aruco::DICT_7X7_1000},
            {"DICT_APRILTAG_36h11", cv::aruco::DICT_APRILTAG_36h11},
            {"DICT_ARUCO_ORIGINAL", cv::aruco::DICT_ARUCO_ORIGINAL},
            {"DICT_ARUCO_MIP_36h12", cv::aruco::DICT_ARUCO_MIP_36h12},
    };
    return kTable;
}

cv::aruco::DetectorParameters to_detector_parameters(const DetectorConfig& config) {
    cv::aruco::DetectorParameters params;
    params.adaptiveThreshWinSizeMin = config.adaptive_thresh_win_size_min_px_;
    params.adaptiveThreshWinSizeMax = config.adaptive_thresh_win_size_max_px_;
    params.adaptiveThreshWinSizeStep = config.adaptive_thresh_win_size_step_px_;
    params.adaptiveThreshConstant = config.adaptive_thresh_constant_;
    params.minMarkerPerimeterRate = config.min_marker_perimeter_rate_;
    params.maxMarkerPerimeterRate = config.max_marker_perimeter_rate_;
    params.polygonalApproxAccuracyRate = config.polygonal_approx_accuracy_rate_;
    params.minCornerDistanceRate = config.min_corner_distance_rate_;
    params.minDistanceToBorder = config.min_distance_to_border_px_;
    params.markerBorderBits = config.marker_border_bits_;
    params.perspectiveRemovePixelPerCell = config.perspective_remove_pixel_per_cell_;
    params.perspectiveRemoveIgnoredMarginPerCell =
            config.perspective_remove_ignored_margin_per_cell_;
    params.maxErroneousBitsInBorderRate = config.max_erroneous_bits_in_border_rate_;
    params.minOtsuStdDev = config.min_otsu_std_dev_;
    params.errorCorrectionRate = config.error_correction_rate_;
    params.useAruco3Detection = config.use_aruco3_detection_;
    params.minSideLengthCanonicalImg = config.min_side_length_canonical_img_px_;
    params.minMarkerLengthRatioOriginalImg = config.min_marker_length_ratio_original_img_;
    params.minMarkerDistanceRate = config.min_marker_distance_rate_;
    params.minGroupDistance = static_cast<float>(config.min_group_distance_);
    params.cornerRefinementWinSize = config.corner_refinement_win_size_px_;
    params.relativeCornerRefinmentWinSize =
            static_cast<float>(config.relative_corner_refinement_win_size_);
    params.cornerRefinementMaxIterations = config.corner_refinement_max_iterations_;
    params.cornerRefinementMinAccuracy = config.corner_refinement_min_accuracy_px_;
    return params;
}

/// Computes the ratio of white pixels per cell. Same as OpenCV's
/// _extractCellPixelRatio.
cv::Mat extract_cell_pixel_ratio(const cv::Mat& image, const std::vector<cv::Point2f>& corners,
                                 int marker_size, int border_bits, int cell_size,
                                 double cell_margin_rate, double min_std_dev) {
    const int size_with_borders = marker_size + 2 * border_bits;
    const int cell_margin = static_cast<int>(cell_margin_rate * cell_size);
    const int result_size = size_with_borders * cell_size;

    std::vector<cv::Point2f> destination(4);
    destination[0] = cv::Point2f(0.0F, 0.0F);
    destination[1] = cv::Point2f(static_cast<float>(result_size) - 1.0F, 0.0F);
    destination[2] = cv::Point2f(static_cast<float>(result_size) - 1.0F,
                                 static_cast<float>(result_size) - 1.0F);
    destination[3] = cv::Point2f(0.0F, static_cast<float>(result_size) - 1.0F);

    const cv::Mat transformation = cv::getPerspectiveTransform(corners, destination);
    cv::Mat canonical;
    cv::warpPerspective(image, canonical, transformation, cv::Size(result_size, result_size),
                        cv::INTER_NEAREST);

    cv::Mat ratio(size_with_borders, size_with_borders, CV_32FC1, cv::Scalar::all(0));
    cv::Mat mean;
    cv::Mat stddev;
    const cv::Mat inner = canonical.colRange(cell_size / 2, canonical.cols - cell_size / 2)
                                  .rowRange(cell_size / 2, canonical.rows - cell_size / 2);
    cv::meanStdDev(inner, mean, stddev);
    if (stddev.ptr<double>(0)[0] < min_std_dev) {
        // A candidate with little variance is taken as all white or all black.
        // Otsu carries no meaning there.
        ratio.setTo(mean.ptr<double>(0)[0] > 127.0 ? 1.0 : 0.0);
        return ratio;
    }
    cv::threshold(canonical, canonical, 125, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    for (int y = 0; y < size_with_borders; ++y) {
        for (int x = 0; x < size_with_borders; ++x) {
            const cv::Rect cell(x * cell_size + cell_margin, y * cell_size + cell_margin,
                                cell_size - 2 * cell_margin, cell_size - 2 * cell_margin);
            const cv::Mat square = canonical(cell);
            ratio.at<float>(y, x) = static_cast<float>(cv::countNonZero(square)) /
                                    static_cast<float>(square.total());
        }
    }
    return ratio;
}

/// Counts the errors in the border cells. Same as OpenCV's _getBorderErrors.
int count_border_errors(const cv::Mat& ratio, int marker_size, int border_bits,
                        float valid_bit_threshold) {
    const int size_with_borders = marker_size + 2 * border_bits;
    int errors = 0;
    for (int y = 0; y < size_with_borders; ++y) {
        const auto* row = ratio.ptr<float>(y);
        for (int k = 0; k < border_bits; ++k) {
            if (row[k] > valid_bit_threshold) {
                ++errors;
            }
            if (row[size_with_borders - 1 - k] > valid_bit_threshold) {
                ++errors;
            }
        }
    }
    for (int x = border_bits; x < size_with_borders - border_bits; ++x) {
        for (int k = 0; k < border_bits; ++k) {
            if (ratio.ptr<float>(k)[x] > valid_bit_threshold) {
                ++errors;
            }
            if (ratio.ptr<float>(size_with_borders - 1 - k)[x] > valid_bit_threshold) {
                ++errors;
            }
        }
    }
    return errors;
}

/// Reject compact contours by their exact unit-chain length before expanding surviving ones.
/// CHAIN_APPROX_SIMPLE removes only collinear unit steps, so a segment contributes its
/// Chebyshev length. Expanding preserves findContours' starting vertex and traversal order.
void find_compact_quad_candidates(const cv::Mat& binary, const DetectorConfig& config,
                                  std::vector<std::vector<cv::Point2f>>& candidates,
                                  std::vector<std::vector<cv::Point>>& contours_out) {
    const int longest = std::max(binary.cols, binary.rows);
    std::size_t min_perimeter =
            static_cast<std::size_t>(config.min_marker_perimeter_rate_ * longest);
    const std::size_t max_perimeter =
            static_cast<std::size_t>(config.max_marker_perimeter_rate_ * longest);
    if (config.use_aruco3_detection_ && config.min_side_length_canonical_img_px_ != 0) {
        min_perimeter = static_cast<std::size_t>(4 * config.min_side_length_canonical_img_px_);
    }
    std::vector<std::vector<cv::Point>> compact_contours;
    cv::findContours(binary, compact_contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    std::vector<cv::Point> contour;
    std::vector<cv::Point> approximation;
    for (const std::vector<cv::Point>& compact : compact_contours) {
        std::size_t length = compact.size() == 1U ? 1U : 0U;
        for (std::size_t i = 0; i < compact.size(); ++i) {
            const cv::Point delta = compact[(i + 1U) % compact.size()] - compact[i];
            length += static_cast<std::size_t>(std::max(std::abs(delta.x), std::abs(delta.y)));
        }
        if (length < min_perimeter || length > max_perimeter) {
            continue;
        }
        contour.clear();
        contour.reserve(length);
        if (compact.size() == 1U) {
            contour.push_back(compact[0]);
        } else {
            for (std::size_t i = 0; i < compact.size(); ++i) {
                cv::Point point = compact[i];
                const cv::Point delta = compact[(i + 1U) % compact.size()] - point;
                const int steps = std::max(std::abs(delta.x), std::abs(delta.y));
                const cv::Point step((delta.x > 0) - (delta.x < 0), (delta.y > 0) - (delta.y < 0));
                for (int j = 0; j < steps; ++j) {
                    contour.push_back(point);
                    point += step;
                }
            }
        }
        cv::approxPolyDP(contour, approximation,
                         static_cast<double>(length) * config.polygonal_approx_accuracy_rate_,
                         true);
        if (approximation.size() != 4U || !cv::isContourConvex(approximation)) {
            continue;
        }
        double min_distance_squared = static_cast<double>(longest) * longest;
        for (std::size_t j = 0; j < 4U; ++j) {
            const double dx = approximation[j].x - approximation[(j + 1U) % 4U].x;
            const double dy = approximation[j].y - approximation[(j + 1U) % 4U].y;
            min_distance_squared = std::min(min_distance_squared, dx * dx + dy * dy);
        }
        const double distance = static_cast<double>(length) * config.min_corner_distance_rate_;
        if (min_distance_squared < distance * distance) {
            continue;
        }
        std::vector<cv::Point2f> candidate(4);
        for (std::size_t j = 0; j < 4U; ++j) {
            candidate[j] = cv::Point2f(static_cast<float>(approximation[j].x),
                                       static_cast<float>(approximation[j].y));
        }
        candidates.push_back(std::move(candidate));
        contours_out.push_back(contour);
    }
}

int bit_count(std::uint64_t value) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(value);
#else
    int count = 0;
    while (value != 0U) {
        value &= value - 1U;
        ++count;
    }
    return count;
#endif
}

/// Immutable dictionary-only result for a fully definite exact codeword.
struct ExactDictionaryEntry {
    std::uint64_t code_ = 0U;
    int id_ = -1;
    int rotation_ = 0;
};

/// Each decode worker owns all mutable candidate sampling buffers.
struct DecodeWorkspace {
    std::vector<cv::Point2f> scaled_{4};
    cv::Mat canonical_;
    cv::Mat ratio_;
    detail::SparsePyramid sparse_;
    std::uint64_t sparse_epoch_ = 0U;
    std::vector<cv::Size> full_refinement_sizes_{1};
};

}  // namespace

class CpuDetector::Impl {
public:
    Status initialize(const std::string& dictionary_name, const DetectorConfig& config,
                      std::string* out_message, const CpuOptions& options);
    Status detect(const cv::Mat& image, CpuResult* out, std::string* out_message);

private:
    Status detect_pass(const cv::Mat& image, CpuResult* out, std::string* out_message,
                       unsigned edge_mask);
    void ensure_level(std::size_t level);
    bool sparse_work_is_smaller(const std::vector<CandidateNode>& nodes, int segmentation_width,
                                int closest) const;
    cv::Mat prefilter_components(const cv::Mat& binary);
    bool identify_one(const std::vector<cv::Point2f>& corners, const cv::Mat& image, float scale,
                      int* id, int* rotation, DecodeWorkspace& workspace, int pyramid_level);
    cv::Mat extract_ratios(const cv::Mat& image, DecodeWorkspace& workspace, int pyramid_level);
    bool identify_packed(const cv::Mat& ratio, int* id, int* rotation) const;

    bool initialized_ = false;
    bool sparse_active_ = false;
    std::uint64_t frame_epoch_ = 0U;
    DetectorConfig config_;
    CpuOptions options_;
    cv::aruco::DetectorParameters parameters_;
    cv::aruco::Dictionary dictionary_;
    std::vector<cv::Mat> decoded_grids_;
    std::vector<std::array<std::uint64_t, 4>> codes_;
    std::vector<ExactDictionaryEntry> exact_dictionary_;
    std::uint64_t valid_code_bits_ = 0U;
    std::vector<cv::Mat> pyramid_;
    std::vector<cv::Size> pyramid_sizes_;
    std::size_t built_level_ = 0U;
    cv::Mat segmentation_;
    cv::Mat padded_segmentation_;
    std::vector<cv::Mat> binary_;
    std::vector<detail::ContourWorkspace> contour_workspaces_;
    std::vector<int> windows_;
    std::array<cv::Point2f, 4> destination_{};
    DecodeWorkspace serial_workspace_;
    cv::TLSData<DecodeWorkspace> parallel_workspaces_;
    cv::Mat component_labels_;
    cv::Mat component_statistics_;
    cv::Mat component_centroids_;
    cv::Mat filtered_binary_;
    std::vector<std::uint8_t> component_values_;
    detail::ThresholdWorkspace threshold_workspace_;
    detail::ComponentCandidates component_candidates_;
};

Status CpuDetector::Impl::initialize(const std::string& dictionary_name,
                                     const DetectorConfig& config, std::string* out_message,
                                     const CpuOptions& options) {
    this->initialized_ = false;
    const Status valid = config.validate(out_message);
    if (valid != Status::kOk) {
        return valid;
    }
    if (config.use_aruco3_detection_ && config.min_side_length_canonical_img_px_ <= 0) {
        message(out_message, "ArUco3 requires a positive canonical image side length");
        return Status::kInvalidConfig;
    }
    const int margin = static_cast<int>(config.perspective_remove_ignored_margin_per_cell_ *
                                        config.perspective_remove_pixel_per_cell_);
    if (config.perspective_remove_pixel_per_cell_ <= 2 * margin) {
        message(out_message, "cell margin leaves no sampled pixels");
        return Status::kInvalidConfig;
    }
    if (config.corner_refine_method_ != CornerRefineMethod::kNone &&
        config.corner_refine_method_ != CornerRefineMethod::kSubpix) {
        message(out_message, "unsupported corner refinement method");
        return Status::kInvalidConfig;
    }
    if (!std::isfinite(options.component_edge_band_px_) || options.component_edge_band_px_ < 0.0 ||
        options.component_edge_band_px_ > 64.0) {
        message(out_message, "component edge support band must be finite and within 0..64 pixels");
        return Status::kInvalidConfig;
    }
    if (options.threshold_parallel_min_pixels_ < 0) {
        message(out_message, "threshold_parallel_min_pixels must be nonnegative");
        return Status::kInvalidConfig;
    }
    if (options.min_parallel_candidates_ < 1) {
        message(out_message, "min_parallel_candidates must be positive");
        return Status::kInvalidConfig;
    }
    if (options.sparse_tile_side_px_ < 1 || options.sparse_tile_side_px_ > 65536) {
        message(out_message, "sparse_tile_side_px must be within 1..65536");
        return Status::kInvalidConfig;
    }
    if (!std::isfinite(options.sparse_full_level_ratio_) ||
        options.sparse_full_level_ratio_ <= 0.0 || options.sparse_full_level_ratio_ > 1.0) {
        message(out_message, "sparse_full_level_ratio must be finite and within (0,1]");
        return Status::kInvalidConfig;
    }
    if (!std::isfinite(options.sparse_max_work_ratio_) || options.sparse_max_work_ratio_ < 0.0) {
        message(out_message, "sparse_max_work_ratio must be finite and nonnegative");
        return Status::kInvalidConfig;
    }
    if (options.decoded_grid_radius_px_ != 0 &&
        (options.decoded_grid_radius_px_ < 2 || options.decoded_grid_radius_px_ > 32 ||
         dictionary_name != "DICT_APRILTAG_36h11" || config.marker_border_bits_ != 1)) {
        message(out_message,
                "decoded grid refinement requires tag36h11, one border cell and a radius of 2..32 "
                "pixels");
        return Status::kInvalidConfig;
    }
    if (options.decode_at_source_resolution_ &&
        (!config.use_aruco3_detection_ || dictionary_name != "DICT_APRILTAG_36h11")) {
        message(out_message, "source-resolution decoding requires the tag36h11 ArUco3 search");
        return Status::kInvalidConfig;
    }
    if (options.candidate_box_stride_px_ != 0 &&
        (!options.decode_at_source_resolution_ || options.candidate_box_stride_px_ < 2 ||
         options.candidate_box_stride_px_ > 8)) {
        message(out_message,
                "candidate box stride must be zero, or 2..8 with source-resolution decoding");
        return Status::kInvalidConfig;
    }
    if (options.search_frame_edges_ &&
        (!options.decode_at_source_resolution_ || config.min_distance_to_border_px_ != 0 ||
         options.decoded_grid_radius_px_ == 0)) {
        message(out_message,
                "frame-edge search requires source-resolution decoding, grid refinement and zero "
                "border distance");
        return Status::kInvalidConfig;
    }
    const auto found = dictionary_table().find(dictionary_name);
    if (found == dictionary_table().end()) {
        message(out_message, "unsupported predefined dictionary: " + dictionary_name);
        return Status::kUnsupportedDictionary;
    }
    try {
        this->dictionary_ = cv::aruco::getPredefinedDictionary(found->second);
        if (this->dictionary_.markerSize > 7) {
            message(out_message, "packed dictionary supports marker sizes up to seven");
            return Status::kUnsupportedDictionary;
        }
        this->decoded_grids_.clear();
        if (options.decoded_grid_radius_px_ != 0) {
            this->decoded_grids_.resize(this->dictionary_.bytesList.rows);
            for (int id = 0; id < this->dictionary_.bytesList.rows; ++id)
                this->dictionary_.generateImageMarker(id, 8, this->decoded_grids_[id], 1);
        }
        this->config_ = config;
        this->options_ = options;
        this->parameters_ = to_detector_parameters(config);
        this->parameters_.validBitIdThreshold = static_cast<float>(config.valid_bit_threshold_);
        this->windows_.clear();
        for (int window = config.adaptive_thresh_win_size_min_px_;
             window <= config.adaptive_thresh_win_size_max_px_;
             window += config.adaptive_thresh_win_size_step_px_) {
            this->windows_.push_back(window + ((window % 2 == 0) ? 1 : 0));
        }
        this->binary_.resize(this->windows_.size());
        this->contour_workspaces_.resize(this->windows_.size());
        const int bytes = (this->dictionary_.markerSize * this->dictionary_.markerSize + 7) / 8;
        this->codes_.resize(static_cast<std::size_t>(this->dictionary_.bytesList.rows));
        for (int id = 0; id < this->dictionary_.bytesList.rows; ++id) {
            const std::uint8_t* row = this->dictionary_.bytesList.ptr<std::uint8_t>(id);
            for (int rotation = 0; rotation < 4; ++rotation) {
                std::uint64_t code = 0U;
                for (int b = 0; b < bytes; ++b) {
                    code |= static_cast<std::uint64_t>(row[rotation * bytes + b]) << (8 * b);
                }
                this->codes_[static_cast<std::size_t>(id)][static_cast<std::size_t>(rotation)] =
                        code;
            }
        }
        this->exact_dictionary_.clear();
        this->valid_code_bits_ = 0U;
        const int total_bits = this->dictionary_.markerSize * this->dictionary_.markerSize;
        for (int index = 0; index < total_bits; ++index) {
            const int base = (index / 8) * 8;
            const int bit = base + std::min(8, total_bits - base) - 1 - index % 8;
            this->valid_code_bits_ |= std::uint64_t{1} << bit;
        }
        if (options.exact_dictionary_lookup_) {
            this->exact_dictionary_.reserve(this->codes_.size() * 4U);
            for (const auto& rotations : this->codes_) {
                for (const std::uint64_t code : rotations) {
                    this->exact_dictionary_.push_back({code & this->valid_code_bits_, -1, 0});
                }
            }
            std::sort(this->exact_dictionary_.begin(), this->exact_dictionary_.end(),
                      [](const ExactDictionaryEntry& first, const ExactDictionaryEntry& second) {
                          return first.code_ < second.code_;
                      });
            this->exact_dictionary_.erase(
                    std::unique(this->exact_dictionary_.begin(), this->exact_dictionary_.end(),
                                [](const ExactDictionaryEntry& first,
                                   const ExactDictionaryEntry& second) {
                                    return first.code_ == second.code_;
                                }),
                    this->exact_dictionary_.end());
            const int max_errors = static_cast<int>(this->dictionary_.maxCorrectionBits *
                                                    config.error_correction_rate_);
            // Precompute the complete ordinary scan, including an earlier non-perfect ID
            // if it satisfies the configured correction limit. No dictionary distance
            // invariant is assumed, and initialization work never depends on a frame.
            for (ExactDictionaryEntry& entry : this->exact_dictionary_) {
                for (std::size_t index = 0; index < this->codes_.size(); ++index) {
                    int best = total_bits + 1;
                    int best_rotation = 0;
                    for (int rotation = 0; rotation < 4; ++rotation) {
                        const int distance = bit_count(
                                entry.code_ ^
                                (this->valid_code_bits_ &
                                 this->codes_[index][static_cast<std::size_t>(rotation)]));
                        if (distance < best) {
                            best = distance;
                            best_rotation = rotation;
                            if (best == 0) {
                                break;
                            }
                        }
                    }
                    if (best <= max_errors) {
                        entry.id_ = static_cast<int>(index);
                        entry.rotation_ = best_rotation;
                        break;
                    }
                }
            }
        }
        const int side = (this->dictionary_.markerSize + 2 * config.marker_border_bits_) *
                         config.perspective_remove_pixel_per_cell_;
        const float end = static_cast<float>(side - 1);
        this->destination_ = {cv::Point2f(0.0F, 0.0F), cv::Point2f(end, 0.0F),
                              cv::Point2f(end, end), cv::Point2f(0.0F, end)};
        this->pyramid_.clear();
        this->pyramid_sizes_.clear();
        this->initialized_ = true;
        return Status::kOk;
    } catch (const std::exception& error) {
        message(out_message, std::string("CPU detector initialization: ") + error.what());
        return Status::kInvalidConfig;
    }
}

bool CpuDetector::Impl::sparse_work_is_smaller(const std::vector<CandidateNode>& nodes,
                                               int segmentation_width, int closest) const {
    if (this->options_.sparse_max_work_ratio_ == 0.0 || nodes.empty()) {
        return true;
    }
    const int tile = this->options_.sparse_tile_side_px_;
    const int minimum_perimeter = 4 * this->config_.min_side_length_canonical_img_px_;
    int maximum_level = closest;
    double regional_work = 0.0;
    for (const CandidateNode& node : nodes) {
        int decode_level = 0;
        float best = std::numeric_limits<float>::max();
        for (std::size_t p = 0; p < this->pyramid_sizes_.size(); ++p) {
            const float scale = static_cast<float>(this->pyramid_sizes_[p].width) /
                                static_cast<float>(segmentation_width);
            const float distance = static_cast<float>(node.contour_.size()) * scale -
                                   static_cast<float>(minimum_perimeter);
            if (distance > 0.0F && distance < best) {
                best = distance;
                decode_level = static_cast<int>(p);
            }
        }
        const int target_level = std::max(decode_level, std::max(0, closest - 1));
        maximum_level = std::max(maximum_level, target_level);
        const cv::Size size = this->pyramid_sizes_[static_cast<std::size_t>(target_level)];
        const float scale = static_cast<float>(size.width) / static_cast<float>(segmentation_width);
        float minimum_x = std::numeric_limits<float>::max();
        float minimum_y = minimum_x;
        float maximum_x = -minimum_x;
        float maximum_y = -minimum_y;
        for (const cv::Point2f& corner : node.corners_) {
            minimum_x = std::min(minimum_x, corner.x * scale);
            minimum_y = std::min(minimum_y, corner.y * scale);
            maximum_x = std::max(maximum_x, corner.x * scale);
            maximum_y = std::max(maximum_y, corner.y * scale);
        }
        // Include a typical refinement patch in the work estimate. This is a cost
        // decision only: actual sampling still validates every iteration footprint.
        const int refinement_level = std::max(0, closest - 1);
        const cv::Size refinement_size =
                this->pyramid_sizes_[static_cast<std::size_t>(refinement_level)];
        const int radius = std::max(refinement_size.width, refinement_size.height) > 1080 ? 5 : 3;
        const double margin =
                std::ldexp(static_cast<double>(radius + 3), refinement_level - target_level);
        const int left = std::max(0, cvFloor(static_cast<double>(minimum_x) - margin));
        const int top = std::max(0, cvFloor(static_cast<double>(minimum_y) - margin));
        const int right =
                std::min(size.width, cvCeil(static_cast<double>(maximum_x) + margin + 1.0));
        const int bottom =
                std::min(size.height, cvCeil(static_cast<double>(maximum_y) + margin + 1.0));
        cv::Rect request(left, top, std::max(0, right - left), std::max(0, bottom - top));
        for (int level = target_level; level > 0 && !request.empty(); --level) {
            const cv::Size current = this->pyramid_sizes_[static_cast<std::size_t>(level)];
            const int x = request.x / tile * tile;
            const int y = request.y / tile * tile;
            const int end_x = std::min(current.width, (request.br().x + tile - 1) / tile * tile);
            const int end_y = std::min(current.height, (request.br().y + tile - 1) / tile * tile);
            regional_work += static_cast<double>(end_x - x) * (end_y - y);
            const cv::Size parent = this->pyramid_sizes_[static_cast<std::size_t>(level - 1)];
            const int source_x = std::max(0, 2 * x - 2);
            const int source_y = std::max(0, 2 * y - 2);
            const int source_end_x = std::min(parent.width, 2 * (end_x - 1) + 3);
            const int source_end_y = std::min(parent.height, 2 * (end_y - 1) + 3);
            request =
                    cv::Rect(source_x, source_y, source_end_x - source_x, source_end_y - source_y);
        }
    }
    double full_work = 0.0;
    for (int level = 1; level <= maximum_level; ++level) {
        const cv::Size size = this->pyramid_sizes_[static_cast<std::size_t>(level)];
        full_work += static_cast<double>(size.width) * size.height;
    }
    return regional_work < full_work * this->options_.sparse_max_work_ratio_;
}

void CpuDetector::Impl::ensure_level(std::size_t level) {
    while (this->built_level_ < level) {
        const std::size_t next = this->built_level_ + 1U;
        cv::pyrDown(this->pyramid_[this->built_level_], this->pyramid_[next],
                    this->pyramid_sizes_[next]);
        this->built_level_ = next;
    }
}

cv::Mat CpuDetector::Impl::prefilter_components(const cv::Mat& binary) {
    const int longest = std::max(binary.cols, binary.rows);
    std::size_t minimum =
            static_cast<std::size_t>(this->config_.min_marker_perimeter_rate_ * longest);
    if (this->config_.use_aruco3_detection_ &&
        this->config_.min_side_length_canonical_img_px_ != 0) {
        minimum = static_cast<std::size_t>(4 * this->config_.min_side_length_canonical_img_px_);
    }
    // A boundary visit has one directed neighbor state; there are at most eight per
    // foreground pixel. An eight-connected component below this conservative bound
    // cannot contain a contour surviving the perimeter filter. Removing it cannot
    // change the boundaries of another eight-disconnected foreground component.
    if (minimum <= 8U) {
        return binary;
    }
    const int count = cv::connectedComponentsWithStats(
            binary, this->component_labels_, this->component_statistics_,
            this->component_centroids_, 8, CV_32S, cv::CCL_GRANA);
    this->component_values_.assign(static_cast<std::size_t>(count), 255U);
    this->component_values_[0] = 0U;
    bool changed = false;
    for (int label = 1; label < count; ++label) {
        const int area = this->component_statistics_.at<int>(label, cv::CC_STAT_AREA);
        if (static_cast<std::size_t>(area) * 8U < minimum) {
            this->component_values_[static_cast<std::size_t>(label)] = 0U;
            changed = true;
        }
    }
    if (!changed) {
        return binary;
    }
    this->filtered_binary_.create(binary.size(), CV_8UC1);
    for (int y = 0; y < binary.rows; ++y) {
        const int* labels = this->component_labels_.ptr<int>(y);
        std::uint8_t* destination = this->filtered_binary_.ptr<std::uint8_t>(y);
        for (int x = 0; x < binary.cols; ++x) {
            destination[x] = this->component_values_[static_cast<std::size_t>(labels[x])];
        }
    }
    return this->filtered_binary_;
}

cv::Mat CpuDetector::Impl::extract_ratios(const cv::Mat& image, DecodeWorkspace& workspace,
                                          int pyramid_level) {
    const int marker_size = this->dictionary_.markerSize;
    const int border = this->config_.marker_border_bits_;
    const int cell_size = this->config_.perspective_remove_pixel_per_cell_;
    if (!this->options_.fused_cells_ && !this->sparse_active_) {
        return extract_cell_pixel_ratio(image, workspace.scaled_, marker_size, border, cell_size,
                                        this->config_.perspective_remove_ignored_margin_per_cell_,
                                        this->config_.min_otsu_std_dev_);
    }
    const int cells = marker_size + 2 * border;
    const int side = cells * cell_size;
    const int margin =
            static_cast<int>(this->config_.perspective_remove_ignored_margin_per_cell_ * cell_size);
    if (this->sparse_active_) {
        if (workspace.sparse_epoch_ != this->frame_epoch_) {
            workspace.sparse_.reset(this->options_.sparse_tile_side_px_,
                                    this->options_.sparse_full_level_ratio_);
            workspace.sparse_epoch_ = this->frame_epoch_;
        }
        workspace.sparse_.warp(this->pyramid_[0], this->pyramid_sizes_, pyramid_level,
                               workspace.scaled_, cv::Size(side, side), &workspace.canonical_);
    } else {
        const cv::Mat transform =
                cv::getPerspectiveTransform(workspace.scaled_.data(), this->destination_.data());
        cv::warpPerspective(image, workspace.canonical_, transform, cv::Size(side, side),
                            cv::INTER_NEAREST);
    }
    workspace.ratio_.create(cells, cells, CV_32FC1);
    cv::Scalar mean;
    cv::Scalar stddev;
    cv::meanStdDev(
            workspace.canonical_(cv::Rect(cell_size / 2, cell_size / 2, side - 2 * (cell_size / 2),
                                          side - 2 * (cell_size / 2))),
            mean, stddev);
    if (stddev[0] < this->config_.min_otsu_std_dev_) {
        workspace.ratio_.setTo(mean[0] > 127.0 ? 1.0 : 0.0);
        return workspace.ratio_;
    }
    const int threshold_mode = cv::THRESH_BINARY | cv::THRESH_OTSU |
                               (this->options_.dryrun_otsu_ ? cv::THRESH_DRYRUN : 0);
    const int otsu = static_cast<int>(cv::threshold(workspace.canonical_, workspace.canonical_,
                                                    125.0, 255.0, threshold_mode));
    const int samples = cell_size - 2 * margin;
    const float area = static_cast<float>(samples * samples);
    for (int y = 0; y < cells; ++y) {
        float* ratios = workspace.ratio_.ptr<float>(y);
        for (int x = 0; x < cells; ++x) {
            int count = 0;
            for (int sy = 0; sy < samples; ++sy) {
                const std::uint8_t* source =
                        workspace.canonical_.ptr<std::uint8_t>(y * cell_size + margin + sy) +
                        x * cell_size + margin;
                for (int sx = 0; sx < samples; ++sx) {
                    count += (this->options_.dryrun_otsu_ ? source[sx] > otsu : source[sx] != 0U)
                                     ? 1
                                     : 0;
                }
            }
            ratios[x] = static_cast<float>(count) / area;
        }
    }
    return workspace.ratio_;
}

bool CpuDetector::Impl::identify_packed(const cv::Mat& ratio, int* id, int* rotation) const {
    const int size = this->dictionary_.markerSize;
    const int total = size * size;
    const int border = this->config_.marker_border_bits_;
    const float lower = this->parameters_.validBitIdThreshold;
    const float upper = 1.0F - lower;
    std::uint64_t not_black = 0U;
    std::uint64_t not_white = 0U;
    for (int y = 0; y < size; ++y) {
        const float* row = ratio.ptr<float>(y + border) + border;
        for (int x = 0; x < size; ++x) {
            const int index = y * size + x;
            const int base = (index / 8) * 8;
            const int bit = base + std::min(8, total - base) - 1 - index % 8;
            const std::uint64_t mask = std::uint64_t{1} << bit;
            if (row[x] > lower) {
                not_black |= mask;
            }
            if (row[x] < upper) {
                not_white |= mask;
            }
        }
    }
    const std::uint64_t selector = not_black ^ not_white;
    if (this->options_.exact_dictionary_lookup_ && selector == this->valid_code_bits_) {
        const auto found = std::lower_bound(this->exact_dictionary_.begin(),
                                            this->exact_dictionary_.end(), not_black,
                                            [](const ExactDictionaryEntry& entry,
                                               std::uint64_t code) { return entry.code_ < code; });
        if (found != this->exact_dictionary_.end() && found->code_ == not_black &&
            found->id_ >= 0) {
            *id = found->id_;
            *rotation = found->rotation_;
            return true;
        }
    }
    const int max_errors = static_cast<int>(this->dictionary_.maxCorrectionBits *
                                            this->config_.error_correction_rate_);
    for (std::size_t index = 0; index < this->codes_.size(); ++index) {
        int best = total + 1;
        int best_rotation = 0;
        for (int r = 0; r < 4; ++r) {
            const int distance = bit_count(
                    not_black ^ (selector & this->codes_[index][static_cast<std::size_t>(r)]));
            if (distance < best) {
                best = distance;
                best_rotation = r;
                if (best == 0) {
                    break;
                }
            }
        }
        if (best <= max_errors) {
            *id = static_cast<int>(index);
            *rotation = best_rotation;
            return true;
        }
    }
    return false;
}

bool CpuDetector::Impl::identify_one(const std::vector<cv::Point2f>& corners, const cv::Mat& image,
                                     float scale, int* id, int* rotation,
                                     DecodeWorkspace& workspace, int pyramid_level) {
    for (std::size_t c = 0; c < 4U; ++c) {
        workspace.scaled_[c] = corners[c] * scale;
    }
    const cv::Mat ratio = this->extract_ratios(image, workspace, pyramid_level);
    const int size = this->dictionary_.markerSize;
    const int border = this->config_.marker_border_bits_;
    const int max_errors =
            static_cast<int>(size * size * this->config_.max_erroneous_bits_in_border_rate_);
    if (count_border_errors(ratio, size, border, this->parameters_.validBitIdThreshold) >
        max_errors) {
        return false;
    }
    if (this->options_.packed_dictionary_ || this->options_.exact_dictionary_lookup_) {
        return this->identify_packed(ratio, id, rotation);
    }
    return this->dictionary_.identify(ratio(cv::Rect(border, border, size, size)), *id, *rotation,
                                      this->config_.error_correction_rate_,
                                      this->parameters_.validBitIdThreshold);
}

Status CpuDetector::Impl::detect(const cv::Mat& image, CpuResult* out, std::string* out_message) {
    if (!this->options_.search_frame_edges_ || out == nullptr || !this->initialized_) {
        return this->detect_pass(image, out, out_message, 0U);
    }
    CpuResult frame;
    const Status status = this->detect_pass(image, &frame, out_message, 0U);
    if (status != Status::kOk)
        return status;
    try {
        const bool profile = this->options_.profile_stages_;
        auto start = profile ? Clock::now() : Clock::time_point{};
        const auto regions = detail::edge_regions(image, this->options_.frame_edge_max_pixels_);
        if (profile)
            frame.stage_times_.preprocess_ms_ += elapsed_ms(start);
        for (const auto& region : regions) {
            // Only the original field of view can be extended. The other ROI sides
            // are ordinary cuts through real image pixels.
            const unsigned mask =
                    (region.x == 0 ? 1U : 0U) | (region.br().x == image.cols ? 2U : 0U) |
                    (region.y == 0 ? 4U : 0U) | (region.br().y == image.rows ? 8U : 0U);
            CpuResult local;
            const auto retry_status = this->detect_pass(image(region), &local, out_message, mask);
            if (retry_status != Status::kOk)
                return retry_status;
            const std::size_t original_count = frame.ids_.size();
            if (profile)
                start = Clock::now();
            for (std::size_t i = 0; i < local.ids_.size(); ++i) {
                auto& corners = local.corners_[i];
                for (auto& point : corners)
                    point += cv::Point2f(region.tl());
                const auto touches = [&](cv::Point2f p) {
                    return p.x <= 3 || p.y <= 3 || p.x >= image.cols - 4 || p.y >= image.rows - 4;
                };
                if (!std::any_of(corners.begin(), corners.end(), touches))
                    continue;
                const auto duplicate = [&](const auto& previous) {
                    const double radius = .2 * std::min(cv::norm(corners[0] - corners[2]),
                                                        cv::norm(previous[0] - previous[2]));
                    for (int offset = 0; offset < 4; ++offset) {
                        bool same = true;
                        for (int c = 0; c < 4; ++c)
                            same &= cv::norm(previous[(c + offset) % 4] - corners[c]) <= radius;
                        if (same)
                            return true;
                    }
                    return false;
                };
                if (std::any_of(frame.corners_.begin(), frame.corners_.end(), duplicate))
                    continue;
                const int id = local.ids_[i];
                const int budget = static_cast<int>(this->dictionary_.maxCorrectionBits *
                                                    this->config_.error_correction_rate_);
                const int ring_budget = static_cast<int>(
                        this->dictionary_.markerSize * this->dictionary_.markerSize *
                        this->config_.max_erroneous_bits_in_border_rate_);
                if (!detail::visible_grid_matches(image, this->decoded_grids_[id], corners, budget,
                                                  ring_budget))
                    continue;
                frame.ids_.push_back(id);
                frame.corners_.push_back(std::move(corners));
            }
            frame.candidate_count_ += local.candidate_count_;
            frame.rejected_count_ += local.rejected_count_ + local.ids_.size() -
                                     (frame.ids_.size() - original_count);
            if (profile) {
                auto& times = frame.stage_times_;
                times.preprocess_ms_ += local.stage_times_.preprocess_ms_;
                times.threshold_ms_ += local.stage_times_.threshold_ms_;
                times.candidates_ms_ += local.stage_times_.candidates_ms_;
                times.trace_ms_ += local.stage_times_.trace_ms_;
                times.group_ms_ += local.stage_times_.group_ms_;
                times.decode_ms_ += local.stage_times_.decode_ms_;
                times.refine_ms_ += local.stage_times_.refine_ms_ + elapsed_ms(start);
            }
        }
        *out = std::move(frame);
        return Status::kOk;
    } catch (const std::exception& error) {
        message(out_message, std::string("frame-edge search: ") + error.what());
        return Status::kInvalidImage;
    }
}
Status CpuDetector::Impl::detect_pass(const cv::Mat& image, CpuResult* out,
                                      std::string* out_message, unsigned edge_mask) {
    if (out == nullptr) {
        message(out_message, "result must not be null");
        return Status::kInvalidArgument;
    }
    if (!this->initialized_) {
        message(out_message, "initialize() has not been called");
        return Status::kNotInitialized;
    }
    if (image.empty() || image.type() != CV_8UC1 || image.dims != 2 || image.cols > 65536 ||
        image.rows > 65536 ||
        image.total() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        image.step < static_cast<std::size_t>(image.cols)) {
        message(out_message,
                "image must be nonempty CV_8UC1, with valid stride, sides <=65536 and at most "
                "INT_MAX pixels");
        return Status::kInvalidImage;
    }
    const auto first = reinterpret_cast<std::uintptr_t>(image.data);
    const auto end = reinterpret_cast<std::uintptr_t>(image.dataend);
    const std::size_t width = static_cast<std::size_t>(image.cols);
    const std::size_t last_row = static_cast<std::size_t>(image.rows - 1);
    if (image.step[1] != 1U || end < first || end - first < width ||
        (last_row != 0U && image.step[0] > (end - first - width) / last_row)) {
        message(out_message, "image storage must cover all visible bytes of each Gray8 row");
        return Status::kInvalidImage;
    }
    try {
        CpuResult result;
        ++this->frame_epoch_;
        if (this->frame_epoch_ == 0U) {
            this->parallel_workspaces_.cleanup();
            this->serial_workspace_.sparse_epoch_ = 0U;
            ++this->frame_epoch_;
        }
        this->sparse_active_ = this->options_.sparse_pyramid_ &&
                               this->config_.use_aruco3_detection_ && !image.isSubmatrix();
        const bool profile = this->options_.profile_stages_;
        Clock::time_point start = profile ? Clock::now() : Clock::time_point{};
        const int box_step = this->options_.candidate_box_stride_px_;
        float scale = 1.0F;
        int levels = 0;
        int closest = 0;
        if (this->config_.use_aruco3_detection_) {
            const float canonical =
                    static_cast<float>(this->config_.min_side_length_canonical_img_px_);
            scale = canonical /
                    (canonical + static_cast<float>(std::max(image.cols, image.rows)) *
                                         this->config_.min_marker_length_ratio_original_img_);
            if (box_step > 0)
                scale = 1.0F / box_step;
            const float image_area =
                    static_cast<float>(static_cast<double>(image.cols) * image.rows);
            const float marker_area = canonical * canonical;
            levels = std::max(0, static_cast<int>(std::log2(image_area / marker_area) / 2.0F));
            const float scaled_area = image_area * scale * scale;
            closest = std::max(0, cvRound(std::log2(image_area / scaled_area) / 2.0F));
            levels = std::max(levels, closest);
            if (this->options_.decode_at_source_resolution_) {
                levels = 0;
                closest = 0;
            }
        }
        const int minimum_extent = box_step > 0 ? 1 : 0;
        const cv::Size segmentation_size(std::max(minimum_extent, cvRound(scale * image.cols)),
                                         std::max(minimum_extent, cvRound(scale * image.rows)));
        if (segmentation_size.width <= 0 || segmentation_size.height <= 0) {
            message(out_message, "downscaling produces an empty segmentation image");
            return Status::kInvalidImage;
        }
        cv::Mat segmentation = image;
        if (scale != 1.0F) {
            if (box_step != 0) {
                detail::sample_box_grid(image, this->segmentation_, segmentation_size, box_step);
            } else {
                cv::resize(image, this->segmentation_, segmentation_size);
            }
            segmentation = this->segmentation_;
        }
        const int neighborhood = this->windows_.back() / 2 + 2;
        int trim_top = 0;
        if (this->options_.decode_at_source_resolution_ && segmentation.cols >= segmentation.rows &&
            this->config_.adaptive_thresh_constant_ >= 1.0) {
            const cv::Range rows = informative_rows(segmentation, neighborhood);
            if (rows.empty()) {
                *out = std::move(result);
                return Status::kOk;
            }
            trim_top = rows.start;
            segmentation = segmentation.rowRange(rows);
        }
        int pad_top = 0, pad_bottom = 0, pad_left = 0, pad_right = 0;
        if (edge_mask != 0U) {
            if ((edge_mask & 4U) && trim_top == 0 && edge_has_contrast(segmentation, true, false))
                pad_top = neighborhood;
            if ((edge_mask & 8U) && trim_top + segmentation.rows == segmentation_size.height &&
                edge_has_contrast(segmentation, true, true))
                pad_bottom = neighborhood;
            if ((edge_mask & 1U) && edge_has_contrast(segmentation, false, false))
                pad_left = neighborhood;
            if ((edge_mask & 2U) && edge_has_contrast(segmentation, false, true))
                pad_right = neighborhood;
        }
        const bool padded = pad_top || pad_bottom || pad_left || pad_right;
        if (padded) {
            if (static_cast<std::int64_t>(segmentation.cols + pad_left + pad_right) *
                        (segmentation.rows + pad_top + pad_bottom) >
                std::numeric_limits<int>::max()) {
                message(out_message, "padded candidate image exceeds INT_MAX pixels");
                return Status::kInvalidImage;
            }
            cv::copyMakeBorder(segmentation, this->padded_segmentation_, pad_top, pad_bottom,
                               pad_left, pad_right, cv::BORDER_CONSTANT | cv::BORDER_ISOLATED,
                               cv::Scalar(255));
            segmentation = this->padded_segmentation_;
        }
        this->pyramid_.resize(static_cast<std::size_t>(levels) + 1U);
        this->pyramid_sizes_.resize(this->pyramid_.size());
        this->pyramid_[0] = image;
        this->pyramid_sizes_[0] = image.size();
        for (std::size_t level = 1U; level < this->pyramid_.size(); ++level) {
            const cv::Size previous = this->pyramid_sizes_[level - 1U];
            this->pyramid_sizes_[level] =
                    cv::Size((previous.width + 1) / 2, (previous.height + 1) / 2);
        }
        this->built_level_ = 0U;
        if (!this->options_.reuse_pyramid_) {
            this->pyramid_.clear();
            cv::buildPyramid(image, this->pyramid_, levels);
            this->built_level_ = static_cast<std::size_t>(levels);
        }
        if (profile) {
            result.stage_times_.preprocess_ms_ = elapsed_ms(start);
            start = Clock::now();
        }
        if (this->options_.multiwindow_threshold_) {
            this->threshold_workspace_.apply(
                    segmentation, this->windows_, this->config_.adaptive_thresh_constant_,
                    &this->binary_, this->options_.threshold_parallel_min_pixels_,
                    this->options_.threshold_max_prefix_bytes_);
        } else {
            const auto threshold_windows = [&](const cv::Range& range) {
                for (int index = range.start; index < range.end; ++index) {
                    const std::size_t w = static_cast<std::size_t>(index);
                    cv::adaptiveThreshold(segmentation, this->binary_[w], 255.0,
                                          cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV,
                                          this->windows_[w],
                                          this->config_.adaptive_thresh_constant_);
                }
            };
            const cv::Range range(0, static_cast<int>(this->windows_.size()));
            if (this->options_.parallel_windows_ && cv::getNumThreads() > 1) {
                cv::parallel_for_(range, threshold_windows);
            } else {
                threshold_windows(range);
            }
        }
        if (profile) {
            result.stage_times_.threshold_ms_ = elapsed_ms(start);
            start = Clock::now();
        }
        const Clock::time_point candidates_start = start;
        std::vector<std::vector<cv::Point2f>> candidates;
        std::vector<std::vector<cv::Point>> contours;
        const bool parallel = this->options_.parallel_windows_ && cv::getNumThreads() > 1 &&
                              !this->options_.component_prefilter_ &&
                              !this->options_.component_candidates_;
        if (parallel) {
            std::vector<std::vector<std::vector<cv::Point2f>>> window_candidates(
                    this->binary_.size());
            std::vector<std::vector<std::vector<cv::Point>>> window_contours(this->binary_.size());
            cv::parallel_for_(
                    cv::Range(0, static_cast<int>(this->binary_.size())),
                    [&](const cv::Range& range) {
                        for (int index = range.start; index < range.end; ++index) {
                            const std::size_t w = static_cast<std::size_t>(index);
                            if (this->options_.simple_contours_) {
                                find_compact_quad_candidates(this->binary_[w], this->config_,
                                                             window_candidates[w],
                                                             window_contours[w]);
                            } else if (this->options_.filtered_contours_) {
                                this->contour_workspaces_[w].append(this->binary_[w], this->config_,
                                                                    window_candidates[w],
                                                                    window_contours[w]);
                            } else {
                                find_quad_candidates(this->binary_[w], this->config_,
                                                     window_candidates[w], window_contours[w]);
                            }
                        }
                    });
            // Keep the serial window order, including ties in the later stable perimeter sort.
            for (std::size_t w = 0; w < this->binary_.size(); ++w) {
                candidates.insert(candidates.end(),
                                  std::make_move_iterator(window_candidates[w].begin()),
                                  std::make_move_iterator(window_candidates[w].end()));
                contours.insert(contours.end(), std::make_move_iterator(window_contours[w].begin()),
                                std::make_move_iterator(window_contours[w].end()));
            }
        } else {
            for (std::size_t w = 0; w < this->binary_.size(); ++w) {
                const cv::Mat& binary = this->binary_[w];
                const cv::Mat filtered = this->options_.component_prefilter_
                                                 ? this->prefilter_components(binary)
                                                 : binary;
                if (this->options_.component_candidates_) {
                    this->component_candidates_.append(filtered, this->config_,
                                                       this->options_.component_edge_band_px_,
                                                       candidates, contours);
                } else if (this->options_.simple_contours_) {
                    find_compact_quad_candidates(filtered, this->config_, candidates, contours);
                } else if (this->options_.filtered_contours_) {
                    this->contour_workspaces_[w].append(filtered, this->config_, candidates,
                                                        contours);
                } else {
                    find_quad_candidates(filtered, this->config_, candidates, contours);
                }
            }
        }
        if (!padded && trim_top != 0) {
            const cv::Point offset(0, trim_top);
            for (auto& quad : candidates)
                for (auto& corner : quad)
                    corner += cv::Point2f(offset);
            for (auto& contour : contours)
                for (auto& point : contour)
                    point += offset;
        }
        result.candidate_count_ = candidates.size();
        for (std::vector<cv::Point2f>& candidate : candidates) {
            reorder_corners(candidate);
        }
        if (profile) {
            result.stage_times_.trace_ms_ = elapsed_ms(start);
            start = Clock::now();
        }
        // Test group bounds in the coordinates used for extraction. A fitted
        // padded contour may be just outside the source while another member is visible.
        const cv::Size group_size = padded ? segmentation.size() : segmentation_size;
        std::vector<CandidateNode> nodes =
                this->options_.spatial_grouping_
                        ? detail::filter_spatial_candidates(group_size, candidates, contours,
                                                            this->config_,
                                                            this->dictionary_.markerSize)
                        : filter_too_close_candidates(group_size, candidates, contours,
                                                      this->config_, this->dictionary_.markerSize);
        if (profile) {
            result.stage_times_.group_ms_ = elapsed_ms(start);
        }
        if (this->sparse_active_ &&
            !this->sparse_work_is_smaller(nodes, segmentation_size.width, closest)) {
            this->sparse_active_ = false;
        }
        if (profile) {
            result.stage_times_.candidates_ms_ = elapsed_ms(candidates_start);
            start = Clock::now();
        }
        if (box_step != 0 || padded) {
            // Group in the segmentation grid, then map each sampling center to its
            // original pixel position. The existing decoder applies the width ratio.
            const float ratio = static_cast<float>(segmentation_size.width) / image.cols;
            const float center = (box_step - 2) / 2 + .5F;
            const auto align = [&](std::vector<cv::Point2f>& corners) {
                for (auto& point : corners) {
                    if (padded)
                        point += cv::Point2f(-pad_left, trim_top - pad_top);
                    if (box_step != 0)
                        point = ratio * (box_step * point + cv::Point2f(center, center));
                }
            };
            for (auto& node : nodes) {
                align(node.corners_);
                for (auto& close : node.close_contours_)
                    align(close.corners_);
            }
        }
        const std::size_t count = nodes.size();
        std::vector<int> ids(count, -1);
        std::vector<int> rotations(count, 0);
        std::vector<std::uint8_t> valid(count, 0U);
        std::vector<std::uint8_t> visited(count, 0U);
        int max_depth = 0;
        for (const CandidateNode& node : nodes) {
            max_depth = std::max(max_depth, node.depth_);
        }
        std::vector<std::vector<std::size_t>> by_depth(static_cast<std::size_t>(max_depth) + 1U);
        for (std::size_t v = 0; v < count; ++v) {
            by_depth[static_cast<std::size_t>(nodes[v].depth_)].push_back(v);
        }
        std::vector<std::size_t> levels_for_node(count, 0U);
        std::vector<float> scales_for_node(count, 1.0F);
        std::size_t counter = 0U;
        std::size_t depth = 0U;
        while (counter < count && depth < by_depth.size()) {
            const std::vector<std::size_t>& batch = by_depth[depth];
            std::size_t maximum_level = 0U;
            for (const std::size_t v : batch) {
                visited[v] = 1U;
                std::size_t level = 0U;
                float candidate_scale = 1.0F;
                if (this->config_.use_aruco3_detection_) {
                    // Dimensions suffice to choose the level. Pixels are generated only when
                    // needed.
                    float best = std::numeric_limits<float>::max();
                    const int perimeter = this->config_.min_side_length_canonical_img_px_ * 4;
                    for (std::size_t p = 0; p < this->pyramid_sizes_.size(); ++p) {
                        const float ratio = static_cast<float>(this->pyramid_sizes_[p].width) /
                                            static_cast<float>(segmentation_size.width);
                        const float distance =
                                static_cast<float>(nodes[v].contour_.size()) * ratio -
                                static_cast<float>(perimeter);
                        if (distance < best && distance > 0.0F) {
                            best = distance;
                            level = p;
                        }
                    }
                    candidate_scale = static_cast<float>(this->pyramid_sizes_[level].width) /
                                      static_cast<float>(segmentation_size.width);
                }

                levels_for_node[v] = level;
                scales_for_node[v] = candidate_scale;
                maximum_level = std::max(maximum_level, level);
            }
            // No worker may mutate the shared pyramid. Generate pixels before publishing
            // the immutable level views, then join the batch before suppression advances.
            if (!this->sparse_active_) {
                this->ensure_level(maximum_level);
            }
            const bool parallel_decode =
                    this->options_.parallel_decode_ && cv::getNumThreads() > 1 &&
                    batch.size() > 1U &&
                    batch.size() >=
                            static_cast<std::size_t>(this->options_.min_parallel_candidates_);
            const auto decode_batch = [&](const cv::Range& range) {
                DecodeWorkspace& workspace = parallel_decode ? this->parallel_workspaces_.getRef()
                                                             : this->serial_workspace_;
                for (int index = range.start; index < range.end; ++index) {
                    const std::size_t v = batch[static_cast<std::size_t>(index)];
                    const cv::Mat& level_image = this->pyramid_[levels_for_node[v]];
                    const float candidate_scale = scales_for_node[v];
                    if (this->identify_one(nodes[v].corners_, level_image, candidate_scale, &ids[v],
                                           &rotations[v], workspace,
                                           static_cast<int>(levels_for_node[v]))) {
                        valid[v] = 1U;
                        continue;
                    }
                    for (const MarkerCandidate& close : nodes[v].close_contours_) {
                        if (this->identify_one(close.corners_, level_image, candidate_scale,
                                               &ids[v], &rotations[v], workspace,
                                               static_cast<int>(levels_for_node[v]))) {
                            nodes[v].corners_ = close.corners_;
                            nodes[v].contour_ = close.contour_;
                            valid[v] = 1U;
                            break;
                        }
                    }
                }
            };
            const cv::Range batch_range(0, static_cast<int>(batch.size()));
            if (parallel_decode) {
                cv::parallel_for_(batch_range, decode_batch, cv::getNumThreads());
            } else {
                decode_batch(batch_range);
            }
            for (const std::size_t v : by_depth[depth]) {
                if (valid[v] != 0U) {
                    int parent = nodes[v].parent_;
                    while (parent != -1) {
                        const std::size_t index = static_cast<std::size_t>(parent);
                        if (visited[index] == 0U) {
                            visited[index] = 1U;
                            ++counter;
                        }
                        parent = nodes[index].parent_;
                    }
                }
                ++counter;
            }
            ++depth;
        }
        if (profile) {
            result.stage_times_.decode_ms_ = elapsed_ms(start);
            start = Clock::now();
        }
        std::vector<std::size_t> accepted;
        accepted.reserve(count);
        for (std::size_t v = 0; v < count; ++v) {
            if (valid[v] != 0U) {
                accepted.push_back(v);
            }
        }
        result.ids_.resize(accepted.size());
        result.corners_.resize(accepted.size());
        if (!accepted.empty() && this->config_.use_aruco3_detection_ && !this->sparse_active_) {
            this->ensure_level(static_cast<std::size_t>(closest));
        }
        const bool parallel_refine =
                this->options_.parallel_decode_ && cv::getNumThreads() > 1 &&
                accepted.size() > 1U &&
                accepted.size() >=
                        static_cast<std::size_t>(this->options_.min_parallel_candidates_);
        const auto refine_batch = [&](const cv::Range& range) {
            DecodeWorkspace& workspace =
                    parallel_refine ? this->parallel_workspaces_.getRef() : this->serial_workspace_;
            for (int index = range.start; index < range.end; ++index) {
                const std::size_t destination = static_cast<std::size_t>(index);
                const std::size_t v = accepted[destination];
                std::vector<cv::Point2f> corners = nodes[v].corners_;
                std::rotate(corners.begin(), corners.begin() + (4 - rotations[v]), corners.end());
                if (this->config_.use_aruco3_detection_) {
                    if (this->sparse_active_ && workspace.sparse_epoch_ != this->frame_epoch_) {
                        workspace.sparse_.reset(this->options_.sparse_tile_side_px_,
                                                this->options_.sparse_full_level_ratio_);
                        workspace.sparse_epoch_ = this->frame_epoch_;
                    }
                    const float initial =
                            static_cast<float>(
                                    this->pyramid_sizes_[static_cast<std::size_t>(closest)].width) /
                            static_cast<float>(segmentation_size.width);
                    cv::Mat matrix(corners);
                    if (initial != 1.0F) {
                        matrix *= static_cast<double>(initial);
                    }
                    for (int p = closest - 1; p >= 0; --p) {
                        matrix *= 2.0;
                        const cv::Size level_size =
                                this->pyramid_sizes_[static_cast<std::size_t>(p)];
                        const int window =
                                std::max(level_size.width, level_size.height) > 1080 ? 5 : 3;
                        if (this->sparse_active_) {
                            workspace.sparse_.refine(
                                    this->pyramid_[0], this->pyramid_sizes_, p, &corners, window,
                                    this->config_.corner_refinement_max_iterations_,
                                    this->config_.corner_refinement_min_accuracy_px_);
                        } else if (this->options_.cached_refinement_) {
                            const cv::Mat& level = this->pyramid_[static_cast<std::size_t>(p)];
                            workspace.full_refinement_sizes_[0] = level_size;
                            workspace.sparse_.refine(
                                    level, workspace.full_refinement_sizes_, 0, &corners, window,
                                    this->config_.corner_refinement_max_iterations_,
                                    this->config_.corner_refinement_min_accuracy_px_);
                        } else {
                            const cv::Mat& level = this->pyramid_[static_cast<std::size_t>(p)];
                            cv::cornerSubPix(
                                    level, matrix, cv::Size(window, window), cv::Size(-1, -1),
                                    cv::TermCriteria(
                                            cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
                                            this->config_.corner_refinement_max_iterations_,
                                            this->config_.corner_refinement_min_accuracy_px_));
                        }
                    }
                } else if (this->config_.corner_refine_method_ == CornerRefineMethod::kSubpix) {
                    const float module = average_module_size(corners, this->dictionary_.markerSize,
                                                             this->config_.marker_border_bits_);
                    const int window = std::min(
                            this->config_.corner_refinement_win_size_px_,
                            std::max(
                                    1,
                                    cvRound(static_cast<float>(
                                                    this->config_
                                                            .relative_corner_refinement_win_size_) *
                                            module)));
                    if (this->options_.cached_refinement_) {
                        workspace.full_refinement_sizes_[0] = image.size();
                        workspace.sparse_.refine(image, workspace.full_refinement_sizes_, 0,
                                                 &corners, window,
                                                 this->config_.corner_refinement_max_iterations_,
                                                 this->config_.corner_refinement_min_accuracy_px_);
                    } else {
                        cv::cornerSubPix(
                                image, corners, cv::Size(window, window), cv::Size(-1, -1),
                                cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
                                                 this->config_.corner_refinement_max_iterations_,
                                                 this->config_.corner_refinement_min_accuracy_px_));
                    }
                }

                if (this->options_.decoded_grid_radius_px_ != 0) {
                    auto fitted = corners;
                    int verified_id = -1, rotation = -1;
                    if (detail::fit_decoded_grid(image, this->decoded_grids_[ids[v]],
                                                 this->options_.decoded_grid_radius_px_, fitted) &&
                        this->identify_one(fitted, image, 1.0F, &verified_id, &rotation, workspace,
                                           0) &&
                        verified_id == ids[v] && rotation == 0) {
                        corners.swap(fitted);
                    }
                }
                result.ids_[destination] = ids[v];
                result.corners_[destination] = std::move(corners);
            }
        };
        const cv::Range refine_range(0, static_cast<int>(accepted.size()));
        if (parallel_refine) {
            cv::parallel_for_(refine_range, refine_batch, cv::getNumThreads());
        } else {
            refine_batch(refine_range);
        }
        result.rejected_count_ = count - result.ids_.size();
        if (profile) {
            result.stage_times_.refine_ms_ = elapsed_ms(start);
        }
        this->pyramid_[0].release();
        *out = std::move(result);
        return Status::kOk;
    } catch (const std::exception& error) {
        if (!this->pyramid_.empty()) {
            this->pyramid_[0].release();
        }
        message(out_message, std::string("CPU detection: ") + error.what());
        return Status::kInvalidImage;
    }
}

CpuDetector::CpuDetector() : impl_(std::make_unique<Impl>()) {}
CpuDetector::~CpuDetector() = default;
CpuDetector::CpuDetector(CpuDetector&&) noexcept = default;
CpuDetector& CpuDetector::operator=(CpuDetector&&) noexcept = default;
Status CpuDetector::initialize(const std::string& name, const DetectorConfig& config,
                               std::string* out_message, const CpuOptions& options) {
    if (!this->impl_) {
        this->impl_ = std::make_unique<Impl>();
    }
    return this->impl_->initialize(name, config, out_message, options);
}
Status CpuDetector::detect(const cv::Mat& image, CpuResult* out, std::string* out_message) {
    if (!this->impl_) {
        message(out_message,
                out == nullptr ? "result must not be null" : "initialize() has not been called");
        return out == nullptr ? Status::kInvalidArgument : Status::kNotInitialized;
    }
    return this->impl_->detect(image, out, out_message);
}
}  // namespace aruco3cuda::cpu
