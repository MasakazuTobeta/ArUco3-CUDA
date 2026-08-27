// SPDX-License-Identifier: Apache-2.0
#include "hybrid_detector.hpp"

#include <cuda_runtime_api.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <ratio>
#include <string>
#include <utility>
#include <vector>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"
#include "aruco3cuda/workspace.hpp"
#include "preprocess.hpp"
#include "threshold.hpp"

namespace aruco3cuda::hybrid {
namespace {

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

/// 二値化画像から四角形候補を抽出する。
///
/// OpenCV の _findMarkerContours と同じ判定を行う。ArUco3 有効時は
/// 周長の下限を minSideLengthCanonicalImg * 4 へ置き換える。
void find_quad_candidates(const cv::Mat& binary, const DetectorConfig& config,
                          std::vector<std::vector<cv::Point2f>>& candidates,
                          std::vector<std::vector<cv::Point>>& contours_out) {
    const int longest = std::max(binary.cols, binary.rows);
    auto min_perimeter =
            static_cast<unsigned int>(config.min_marker_perimeter_rate_ * longest);
    const auto max_perimeter =
            static_cast<unsigned int>(config.max_marker_perimeter_rate_ * longest);
    if (config.use_aruco3_detection_ && config.min_side_length_canonical_img_px_ != 0) {
        min_perimeter = static_cast<unsigned int>(4 * config.min_side_length_canonical_img_px_);
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    for (const auto& contour : contours) {
        if (contour.size() < min_perimeter || contour.size() > max_perimeter) {
            continue;
        }
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx,
                         static_cast<double>(contour.size()) *
                                 config.polygonal_approx_accuracy_rate_,
                         true);
        if (approx.size() != 4U || !cv::isContourConvex(approx)) {
            continue;
        }
        double min_distance_squared = static_cast<double>(longest) * longest;
        for (int j = 0; j < 4; ++j) {
            const double dx = approx[j].x - approx[(j + 1) % 4].x;
            const double dy = approx[j].y - approx[(j + 1) % 4].y;
            min_distance_squared = std::min(min_distance_squared, dx * dx + dy * dy);
        }
        const double min_corner_distance =
                static_cast<double>(contour.size()) * config.min_corner_distance_rate_;
        if (min_distance_squared < min_corner_distance * min_corner_distance) {
            continue;
        }
        std::vector<cv::Point2f> candidate(4);
        for (int j = 0; j < 4; ++j) {
            candidate[j] = cv::Point2f(static_cast<float>(approx[j].x),
                                       static_cast<float>(approx[j].y));
        }
        candidates.push_back(candidate);
        contours_out.push_back(contour);
    }
}

/// 四隅を時計回りへ揃える。OpenCV の _reorderCandidatesCorners と同じ。
void reorder_corners(std::vector<cv::Point2f>& candidate) {
    // 差を float で取ってから double へ広げる。OpenCV と同じ丸めにする。
    const double dx1 = static_cast<double>(candidate[1].x - candidate[0].x);
    const double dy1 = static_cast<double>(candidate[1].y - candidate[0].y);
    const double dx2 = static_cast<double>(candidate[2].x - candidate[0].x);
    const double dy2 = static_cast<double>(candidate[2].y - candidate[0].y);
    if ((dx1 * dy2) - (dy1 * dx2) < 0.0) {
        std::swap(candidate[1], candidate[3]);
    }
}

/// 候補の周長から、識別に使う pyramid level を選ぶ。
///
/// OpenCV の _findOptPyrImageForCanonicalImg と同じ。距離が正のものだけを
/// 対象とし、より大きい level を優先する。
std::size_t find_optimal_level(const std::vector<cv::Mat>& pyramid, int scaled_width,
                               int contour_size, int min_perimeter) {
    std::size_t optimal = 0;
    float distance = std::numeric_limits<float>::max();
    for (std::size_t i = 0; i < pyramid.size(); ++i) {
        const float scale = static_cast<float>(pyramid[i].cols) / static_cast<float>(scaled_width);
        const float scaled_perimeter = static_cast<float>(contour_size) * scale;
        const float new_distance = scaled_perimeter - static_cast<float>(min_perimeter);
        if (new_distance < distance && new_distance > 0.0F) {
            distance = new_distance;
            optimal = i;
        }
    }
    return optimal;
}

/// セルごとの白画素比を求める。OpenCV の _extractCellPixelRatio と同じ。
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
        // 分散が小さい候補は全て白か全て黒とみなす。Otsu が意味を持たない。
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

/// 外周セルの誤り数を数える。OpenCV の _getBorderErrors と同じ。
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

/// 候補 1 つ分の四隅・輪郭・周長。
///
/// 周長は輪郭長ではなく四隅を結んだ四角形の辺長和とする。OpenCV の
/// MarkerCandidate と同じ定義であり、グループ内で採用する候補の選択に使う。
struct MarkerCandidate {
    std::vector<cv::Point2f> corners_;
    std::vector<cv::Point> contour_;
    float perimeter_ = 0.0F;
};

/// 候補を包含関係の木として保持する。OpenCV の MarkerCandidateTree と同じ。
///
/// parent_ は自分を内側に含む候補の index、depth_ は自分より内側にある
/// 候補の最大段数である。識別は depth_ の小さい順に行い、マーカーが確定
/// したらその親を識別対象から外す。
struct CandidateNode : MarkerCandidate {
    int parent_ = -1;
    int depth_ = 0;
    std::vector<MarkerCandidate> close_contours_;
};

/// 四隅を結んだ四角形の辺長和。
float quad_perimeter(const std::vector<cv::Point2f>& corners) {
    float perimeter = 0.0F;
    for (std::size_t i = 0; i < 4U; ++i) {
        const cv::Point2f edge = corners[i] - corners[(i + 1U) % 4U];
        perimeter += std::sqrt((edge.x * edge.x) + (edge.y * edge.y));
    }
    return perimeter;
}

/// 2 つの候補の四隅間平均距離。開始頂点の 4 通りの対応のうち最小を採る。
///
/// OpenCV の getAverageDistance と同じ。頂点の並びが 1 つずれていても
/// 同じマーカーだと判定できるようにするため、対応を総当たりする。
float average_quad_distance(const std::vector<cv::Point2f>& first,
                            const std::vector<cv::Point2f>& second) {
    float min_distance_squared = std::numeric_limits<float>::max();
    for (int fc = 0; fc < 4; ++fc) {
        float distance_squared = 0.0F;
        for (int c = 0; c < 4; ++c) {
            const int mod_c = (c + fc) % 4;
            const cv::Point2f delta = first[static_cast<std::size_t>(mod_c)] -
                                      second[static_cast<std::size_t>(c)];
            distance_squared += (delta.x * delta.x) + (delta.y * delta.y);
        }
        distance_squared /= 4.0F;
        min_distance_squared = std::min(min_distance_squared, distance_squared);
    }
    return std::sqrt(min_distance_squared);
}

/// 四隅から求めたセル 1 辺の平均画素数。OpenCV の getAverageModuleSize と同じ。
float average_module_size(const std::vector<cv::Point2f>& corners, int marker_size,
                          int border_bits) {
    const int module_count = marker_size + (border_bits * 2);
    return quad_perimeter(corners) / (4.0F * static_cast<float>(module_count));
}

/// first の四隅が全て second の内側にあるか。OpenCV の checkMarker1InMarker2 と同じ。
bool quad_inside_quad(const std::vector<cv::Point2f>& first,
                      const std::vector<cv::Point2f>& second) {
    for (std::size_t i = 0; i < 4U; ++i) {
        if (cv::pointPolygonTest(second, first[i], false) < 0.0) {
            return false;
        }
    }
    return true;
}

/// 近接候補をグループ化し、各グループの代表を選んで包含関係の木を作る。
///
/// 二値化 window を変えると同じマーカーから少しずつ違う候補が得られる。
/// OpenCV の filterTooCloseCandidates と同じく、周長の降順に並べたうえで
/// 近接するものを 1 グループとし、グループ内で最大周長の候補を代表に採る。
/// 代表以外のうち代表から離れているものは close_contours_ として残し、
/// 代表の識別が失敗した場合の代替に使う。
///
/// 代表が画像端に近すぎる場合はグループごと捨てる。OpenCV も同じ扱いで、
/// 端に掛かったマーカーは四隅が信用できないためである。
std::vector<CandidateNode> filter_too_close_candidates(
        const cv::Size& image_size, std::vector<std::vector<cv::Point2f>>& candidates,
        std::vector<std::vector<cv::Point>>& contours, const DetectorConfig& config,
        int marker_size) {
    std::vector<CandidateNode> nodes(candidates.size());
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        nodes[i].corners_ = std::move(candidates[i]);
        nodes[i].contour_ = std::move(contours[i]);
        nodes[i].perimeter_ = quad_perimeter(nodes[i].corners_);
    }
    // 周長の降順へ並べる。同じ周長のときの順序は window の順序を保つ。
    std::stable_sort(nodes.begin(), nodes.end(),
                     [](const CandidateNode& lhs, const CandidateNode& rhs) {
                         return lhs.perimeter_ > rhs.perimeter_;
                     });

    std::vector<int> group_id(nodes.size(), -1);
    std::vector<std::vector<std::size_t>> groups;
    std::vector<bool> selected(nodes.size(), true);
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        for (std::size_t j = i + 1U; j < nodes.size(); ++j) {
            const float distance = average_quad_distance(nodes[i].corners_, nodes[j].corners_);
            if (distance >= nodes[j].perimeter_ *
                                    static_cast<float>(config.min_marker_distance_rate_)) {
                continue;
            }
            selected[i] = false;
            selected[j] = false;
            if (group_id[i] < 0 && group_id[j] < 0) {
                group_id[i] = static_cast<int>(groups.size());
                group_id[j] = group_id[i];
                groups.push_back({i, j});
            } else if (group_id[i] > -1 && group_id[j] == -1) {
                group_id[j] = group_id[i];
                groups[static_cast<std::size_t>(group_id[i])].push_back(j);
            } else if (group_id[j] > -1 && group_id[i] == -1) {
                group_id[i] = group_id[j];
                groups[static_cast<std::size_t>(group_id[j])].push_back(i);
            }
            // 双方が別の group に属する場合は統合しない。OpenCV と同じ挙動。
        }
        if (selected[i]) {
            // どの候補とも近接しなかったものは 1 つだけの group とする。
            selected[i] = false;
            group_id[i] = static_cast<int>(groups.size());
            groups.push_back({i});
        }
    }

    for (std::vector<std::size_t>& group : groups) {
        // index は周長の降順なので、昇順に並べると先頭が最大周長になる。
        std::stable_sort(group.begin(), group.end());
        std::size_t current = group[0];
        bool too_near_border = false;
        for (const cv::Point2f& corner : nodes[current].corners_) {
            const auto margin = static_cast<float>(config.min_distance_to_border_px_);
            if (corner.x < margin || corner.y < margin ||
                corner.x > static_cast<float>(image_size.width - 1) - margin ||
                corner.y > static_cast<float>(image_size.height - 1) - margin) {
                too_near_border = true;
                break;
            }
        }
        if (too_near_border) {
            continue;
        }
        selected[current] = true;
        for (std::size_t k = 1U; k < group.size(); ++k) {
            const std::size_t id = group[k];
            const float distance =
                    average_quad_distance(nodes[id].corners_, nodes[current].corners_);
            const float module_size = average_module_size(nodes[id].corners_, marker_size,
                                                          config.marker_border_bits_);
            if (distance > static_cast<float>(config.min_group_distance_) * module_size) {
                current = id;
                nodes[group[0]].close_contours_.push_back(nodes[id]);
            }
        }
    }

    std::vector<CandidateNode> result;
    result.reserve(groups.size());
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        if (selected[i]) {
            result.push_back(std::move(nodes[i]));
        }
    }

    // 包含関係を辿って親と段数を決める。result は周長の降順なので、
    // 内側の候補ほど後ろにある。
    for (int i = static_cast<int>(result.size()) - 1; i >= 0; --i) {
        for (int j = i - 1; j >= 0; --j) {
            const auto outer = static_cast<std::size_t>(j);
            const auto inner = static_cast<std::size_t>(i);
            if (quad_inside_quad(result[inner].corners_, result[outer].corners_)) {
                result[inner].parent_ = j;
                result[outer].depth_ = std::max(result[outer].depth_, result[inner].depth_ + 1);
                break;
            }
        }
    }
    return result;
}

}  // namespace

/// 実装本体。公開 header が OpenCV へ依存しないよう pimpl とする。
class HybridDetector::Impl {
public:
    Status initialize(const DetectorConfig& config, const std::string& dictionary_name,
                      int max_width_px, int max_height_px, std::string* out_message);
    Status detect(const ImageViewU8& image, HybridResult* out, std::string* out_message);
    const WorkspaceStatistics& workspace_statistics() const {
        return this->workspace_.statistics();
    }

private:
    Status run_gpu_stages(const ImageViewU8& image, std::string* out_message);
    void run_cpu_stages(const ImageViewU8& image, HybridResult* out);

    bool initialized_ = false;
    DetectorConfig config_;
    cv::aruco::Dictionary dictionary_;
    cv::aruco::DetectorParameters parameters_;
    Workspace workspace_;
    detail::ScalePlan plan_;
    detail::PreprocessBuffers preprocess_;
    detail::ThresholdBuffers threshold_;
    std::vector<cv::Mat> binary_images_;
    std::vector<cv::Mat> pyramid_;
};

Status HybridDetector::Impl::initialize(const DetectorConfig& config,
                                        const std::string& dictionary_name, int max_width_px,
                                        int max_height_px, std::string* out_message) {
    const Status config_status = config.validate(out_message);
    if (config_status != Status::kOk) {
        return config_status;
    }
    if (max_width_px < 1 || max_height_px < 1) {
        if (out_message != nullptr) {
            *out_message = "最大解像度は 1 以上である必要がある";
        }
        return Status::kInvalidArgument;
    }
    const auto entry = dictionary_table().find(dictionary_name);
    if (entry == dictionary_table().end()) {
        if (out_message != nullptr) {
            *out_message = "未対応の Dictionary: " + dictionary_name;
        }
        return Status::kUnsupportedDictionary;
    }

    this->config_ = config;
    this->dictionary_ = cv::aruco::getPredefinedDictionary(entry->second);
    this->parameters_ = to_detector_parameters(config);

    // 最大解像度から必要な容量を求め、初期化時に一括で確保する。
    detail::ScalePlan plan;
    const Status plan_status = detail::plan_scales(config, max_width_px, max_height_px, &plan);
    if (plan_status != Status::kOk) {
        return plan_status;
    }
    const std::size_t preprocess_bytes =
            detail::preprocess_workspace_bytes(plan, max_width_px, max_height_px);
    const std::size_t threshold_bytes = detail::threshold_workspace_bytes(
            config, plan.segmentation_width_px_, plan.segmentation_height_px_);
    if (preprocess_bytes == 0U || threshold_bytes == 0U) {
        if (out_message != nullptr) {
            *out_message = "workspace の必要量を算出できない";
        }
        return Status::kInvalidConfig;
    }
    const Status capacity_status = this->workspace_.ensure_capacity(
            preprocess_bytes + threshold_bytes, MemorySpace::kDevice, out_message);
    if (capacity_status != Status::kOk) {
        return capacity_status;
    }
    this->initialized_ = true;
    return Status::kOk;
}

Status HybridDetector::Impl::run_gpu_stages(const ImageViewU8& image, std::string* out_message) {
    const Status plan_status =
            detail::plan_scales(this->config_, image.width_px_, image.height_px_, &this->plan_);
    if (plan_status != Status::kOk) {
        return plan_status;
    }
    // フレームの先頭で切り出し位置を戻す。容量は保持されるため確保は起きない。
    this->workspace_.reset();

    const Status reserve_status = detail::reserve_preprocess(this->plan_, image, this->workspace_,
                                                             &this->preprocess_);
    if (reserve_status != Status::kOk) {
        if (out_message != nullptr) {
            *out_message = "前処理の領域を確保できない。初期化時の最大解像度を超えている";
        }
        return reserve_status;
    }
    const Status threshold_reserve = detail::reserve_threshold(
            this->config_, this->plan_.segmentation_width_px_, this->plan_.segmentation_height_px_,
            this->workspace_, &this->threshold_);
    if (threshold_reserve != Status::kOk) {
        if (out_message != nullptr) {
            *out_message = "二値化の領域を確保できない。初期化時の最大解像度を超えている";
        }
        return threshold_reserve;
    }

    Status status = detail::build_pyramid_async(&this->preprocess_, this->config_, nullptr);
    if (status != Status::kOk) {
        return status;
    }
    status = detail::build_segmentation_async(this->plan_, &this->preprocess_, this->config_,
                                              nullptr);
    if (status != Status::kOk) {
        return status;
    }
    const ImageViewU8 segmentation{this->preprocess_.segmentation_.data_,
                                   this->preprocess_.segmentation_.width_px_,
                                   this->preprocess_.segmentation_.height_px_,
                                   this->preprocess_.segmentation_.pitch_bytes_, image.space_};
    status = detail::build_threshold_async(segmentation, &this->threshold_, this->config_,
                                           nullptr);
    if (status != Status::kOk) {
        return status;
    }

    // 二値化画像を host へ戻すため、ここで同期する。案 C の構造上避けられない。
    const cudaError_t sync = cudaDeviceSynchronize();
    if (sync != cudaSuccess) {
        if (out_message != nullptr) {
            *out_message = std::string("GPU の同期に失敗した: ") + cudaGetErrorString(sync);
        }
        return Status::kCudaError;
    }

    // 二値化画像と pyramid を host へ取り出す。
    this->binary_images_.resize(static_cast<std::size_t>(this->threshold_.window_count_));
    for (int i = 0; i < this->threshold_.window_count_; ++i) {
        const detail::ImagePlaneU8& plane = this->threshold_.binary_[i];
        cv::Mat& destination = this->binary_images_[static_cast<std::size_t>(i)];
        destination.create(plane.height_px_, plane.width_px_, CV_8UC1);
        if (cudaMemcpy2D(destination.data, static_cast<std::size_t>(destination.step),
                         plane.data_, plane.pitch_bytes_,
                         static_cast<std::size_t>(plane.width_px_),
                         static_cast<std::size_t>(plane.height_px_),
                         cudaMemcpyDeviceToHost) != cudaSuccess) {
            if (out_message != nullptr) {
                *out_message = "二値化画像を host へ戻せない";
            }
            return Status::kCudaError;
        }
    }

    this->pyramid_.resize(static_cast<std::size_t>(this->preprocess_.level_count_));
    for (int level = 0; level < this->preprocess_.level_count_; ++level) {
        const ImageViewU8 view = detail::level_view(this->preprocess_, level);
        cv::Mat& destination = this->pyramid_[static_cast<std::size_t>(level)];
        destination.create(view.height_px_, view.width_px_, CV_8UC1);
        if (cudaMemcpy2D(destination.data, static_cast<std::size_t>(destination.step), view.data_,
                         view.pitch_bytes_, static_cast<std::size_t>(view.width_px_),
                         static_cast<std::size_t>(view.height_px_),
                         cudaMemcpyDeviceToHost) != cudaSuccess) {
            if (out_message != nullptr) {
                *out_message = "pyramid を host へ戻せない";
            }
            return Status::kCudaError;
        }
    }
    return Status::kOk;
}

void HybridDetector::Impl::run_cpu_stages(const ImageViewU8& image, HybridResult* out) {
    const int segmentation_width = this->plan_.segmentation_width_px_;
    const int segmentation_height = this->plan_.segmentation_height_px_;
    const int min_perimeter = this->config_.min_side_length_canonical_img_px_ * 4;
    const int cell_size = this->config_.perspective_remove_pixel_per_cell_;
    const int marker_size = this->dictionary_.markerSize;
    const int border_bits = this->config_.marker_border_bits_;
    const int max_border_errors = static_cast<int>(
            marker_size * marker_size * this->config_.max_erroneous_bits_in_border_rate_);

    std::vector<std::vector<cv::Point2f>> candidates;
    std::vector<std::vector<cv::Point>> contours;
    for (const cv::Mat& binary : this->binary_images_) {
        find_quad_candidates(binary, this->config_, candidates, contours);
    }
    out->candidate_count_ = candidates.size();
    for (std::vector<cv::Point2f>& candidate : candidates) {
        reorder_corners(candidate);
    }

    // 二値化 window ごとに同じマーカーが複数回検出されるため、近接候補を
    // まとめて 1 つに絞る。ここで最大周長の候補を選ばないと四隅が内側へ
    // 寄り、原寸へ戻したときに数 px の誤差になる。
    std::vector<CandidateNode> nodes = filter_too_close_candidates(
            cv::Size(segmentation_width, segmentation_height), candidates, contours,
            this->config_, marker_size);

    // 1 つの候補を識別する。成功したら id と rotation を書き込む。
    const auto identify_one = [&](const std::vector<cv::Point2f>& corners, const cv::Mat& level,
                                  float scale, int* out_id, int* out_rotation) {
        std::vector<cv::Point2f> scaled(4);
        for (int c = 0; c < 4; ++c) {
            scaled[static_cast<std::size_t>(c)] = corners[static_cast<std::size_t>(c)] * scale;
        }
        const cv::Mat ratio = extract_cell_pixel_ratio(
                level, scaled, marker_size, border_bits, cell_size,
                this->config_.perspective_remove_ignored_margin_per_cell_,
                this->config_.min_otsu_std_dev_);
        if (count_border_errors(ratio, marker_size, border_bits,
                                this->parameters_.validBitIdThreshold) > max_border_errors) {
            return false;
        }
        const cv::Mat inner =
                ratio(cv::Rect(border_bits, border_bits, ratio.cols - (2 * border_bits),
                               ratio.rows - (2 * border_bits)));
        return this->dictionary_.identify(inner, *out_id, *out_rotation,
                                          this->config_.error_correction_rate_,
                                          this->parameters_.validBitIdThreshold);
    };

    const std::size_t node_count = nodes.size();
    std::vector<int> ids(node_count, -1);
    std::vector<int> rotations(node_count, 0);
    std::vector<std::uint8_t> valid(node_count, 0U);
    std::vector<std::uint8_t> visited(node_count, 0U);

    int max_depth = 0;
    for (const CandidateNode& node : nodes) {
        max_depth = std::max(max_depth, node.depth_);
    }
    std::vector<std::vector<std::size_t>> by_depth(static_cast<std::size_t>(max_depth) + 1U);
    for (std::size_t i = 0; i < node_count; ++i) {
        by_depth[static_cast<std::size_t>(nodes[i].depth_)].push_back(i);
    }

    // 内側の候補から識別し、マーカーが確定したらその外側 (親) を数え上げ
    // 済みとして扱う。マーカーの黒枠は外側の輪郭も候補になるため、内側が
    // 当たった時点で外側を識別する必要がない。
    std::size_t counter = 0;
    std::size_t depth = 0;
    while (counter < node_count && depth < by_depth.size()) {
        for (const std::size_t v : by_depth[depth]) {
            visited[v] = 1U;
            std::size_t level = 0;
            float scale = 1.0F;
            if (this->config_.use_aruco3_detection_) {
                level = find_optimal_level(this->pyramid_, segmentation_width,
                                           static_cast<int>(nodes[v].contour_.size()),
                                           min_perimeter);
                scale = static_cast<float>(this->pyramid_[level].cols) /
                        static_cast<float>(segmentation_width);
            }
            const cv::Mat& level_image = this->pyramid_[level];
            if (identify_one(nodes[v].corners_, level_image, scale, &ids[v], &rotations[v])) {
                valid[v] = 1U;
                continue;
            }
            // 代表で失敗した場合、同じ group の離れた候補を試す。
            for (const MarkerCandidate& close : nodes[v].close_contours_) {
                if (identify_one(close.corners_, level_image, scale, &ids[v], &rotations[v])) {
                    nodes[v].corners_ = close.corners_;
                    nodes[v].contour_ = close.contour_;
                    valid[v] = 1U;
                    break;
                }
            }
        }
        for (const std::size_t v : by_depth[depth]) {
            if (valid[v] != 0U) {
                int parent = nodes[v].parent_;
                while (parent != -1) {
                    const auto index = static_cast<std::size_t>(parent);
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

    // ArUco3 は縮小画像で輪郭を取るため、原寸へ戻すには subpixel 補正が要る。
    // OpenCV も useAruco3Detection が有効なら補正方法の指定を上書きする。
    const bool use_subpix = this->config_.use_aruco3_detection_ ||
                            this->config_.corner_refine_method_ == CornerRefineMethod::kSubpix;
    for (std::size_t v = 0; v < node_count; ++v) {
        if (valid[v] == 0U) {
            continue;
        }
        std::vector<cv::Point2f> corners = nodes[v].corners_;
        // Dictionary 照合で得た回転を打ち消す。OpenCV の correctCornerPosition と同じ。
        std::rotate(corners.begin(),
                    corners.begin() + (4 - static_cast<std::ptrdiff_t>(rotations[v])),
                    corners.end());

        // 四隅を原寸座標へ戻す。ArUco3 は pyramid を 1 段ずつ上げながら
        // 各 level で subpixel 補正する。
        if (this->config_.use_aruco3_detection_) {
            const int closest = this->plan_.closest_level_index_;
            const float initial_scale =
                    static_cast<float>(
                            this->pyramid_[static_cast<std::size_t>(closest)].cols) /
                    static_cast<float>(segmentation_width);
            cv::Mat corner_matrix(corners);
            if (initial_scale != 1.0F) {
                corner_matrix *= static_cast<double>(initial_scale);
            }
            for (int idx = closest - 1; idx >= 0; --idx) {
                corner_matrix *= 2.0;
                const cv::Mat& level_image = this->pyramid_[static_cast<std::size_t>(idx)];
                const int window = std::max(level_image.cols, level_image.rows) > 1080 ? 5 : 3;
                cv::cornerSubPix(level_image, corner_matrix, cv::Size(window, window),
                                 cv::Size(-1, -1),
                                 cv::TermCriteria(cv::TermCriteria::MAX_ITER |
                                                          cv::TermCriteria::EPS,
                                                  this->config_.corner_refinement_max_iterations_,
                                                  this->config_.corner_refinement_min_accuracy_px_));
            }
        } else if (use_subpix) {
            // ArUco3 無効時は原寸画像で 1 度だけ補正する。window はセル 1 辺
            // から決め、上限で頭打ちにする。OpenCV の非 ArUco3 経路と同じ。
            const float module_size =
                    average_module_size(corners, marker_size, border_bits);
            int window = std::max(
                    1, cvRound(static_cast<float>(
                                       this->config_.relative_corner_refinement_win_size_) *
                               module_size));
            window = std::min(window, this->config_.corner_refinement_win_size_px_);
            cv::Mat corner_matrix(corners);
            cv::cornerSubPix(this->pyramid_[0], corner_matrix, cv::Size(window, window),
                             cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
                                              this->config_.corner_refinement_max_iterations_,
                                              this->config_.corner_refinement_min_accuracy_px_));
        }
        // ArUco3 が無効なら縮小しないため、原寸へ戻す拡大は不要である。

        HybridDetection detection;
        detection.id_ = ids[v];
        detection.rotation_ = rotations[v];
        for (int c = 0; c < 4; ++c) {
            const std::size_t index = static_cast<std::size_t>(c);
            detection.corners_[index * 2U] = static_cast<double>(corners[index].x);
            detection.corners_[(index * 2U) + 1U] = static_cast<double>(corners[index].y);
        }
        out->detections_.push_back(detection);
    }

    // 結果の順序を入力順序に依存させないため、ID と位置で整列する。
    std::sort(out->detections_.begin(), out->detections_.end(),
              [](const HybridDetection& a, const HybridDetection& b) {
                  if (a.id_ != b.id_) {
                      return a.id_ < b.id_;
                  }
                  if (a.corners_[0] != b.corners_[0]) {
                      return a.corners_[0] < b.corners_[0];
                  }
                  return a.corners_[1] < b.corners_[1];
              });
    (void)image;
}

Status HybridDetector::Impl::detect(const ImageViewU8& image, HybridResult* out,
                                    std::string* out_message) {
    if (out == nullptr) {
        return Status::kInvalidArgument;
    }
    if (!this->initialized_) {
        if (out_message != nullptr) {
            *out_message = "initialize() が呼ばれていない";
        }
        return Status::kNotInitialized;
    }
    const Status image_status = validate_image_view(image, out_message);
    if (image_status != Status::kOk) {
        return image_status;
    }

    HybridResult result;
    const auto gpu_start = std::chrono::steady_clock::now();
    const Status gpu_status = this->run_gpu_stages(image, out_message);
    const auto gpu_finish = std::chrono::steady_clock::now();
    if (gpu_status != Status::kOk) {
        return gpu_status;
    }
    result.gpu_ms_ = std::chrono::duration<double, std::milli>(gpu_finish - gpu_start).count();

    const auto cpu_start = std::chrono::steady_clock::now();
    this->run_cpu_stages(image, &result);
    const auto cpu_finish = std::chrono::steady_clock::now();
    result.cpu_ms_ = std::chrono::duration<double, std::milli>(cpu_finish - cpu_start).count();

    *out = std::move(result);
    return Status::kOk;
}

HybridDetector::HybridDetector() : impl_(std::make_unique<Impl>()) {}
HybridDetector::~HybridDetector() = default;
HybridDetector::HybridDetector(HybridDetector&&) noexcept = default;
HybridDetector& HybridDetector::operator=(HybridDetector&&) noexcept = default;

Status HybridDetector::initialize(const DetectorConfig& config, const std::string& dictionary_name,
                                  int max_width_px, int max_height_px, std::string* out_message) {
    return this->impl_->initialize(config, dictionary_name, max_width_px, max_height_px,
                                   out_message);
}

Status HybridDetector::detect(const ImageViewU8& image, HybridResult* out,
                              std::string* out_message) {
    return this->impl_->detect(image, out, out_message);
}

const WorkspaceStatistics& HybridDetector::workspace_statistics() const {
    return this->impl_->workspace_statistics();
}

}  // namespace aruco3cuda::hybrid
