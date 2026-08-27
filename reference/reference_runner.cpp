// SPDX-License-Identifier: Apache-2.0
#include "reference_runner.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <ios>
#include <map>
#include <memory>
#include <ostream>
#include <ratio>
#include <string>
#include <utility>
#include <vector>

#include "aruco3cuda/util/json_writer.hpp"
#include "aruco3cuda/util/sha256.hpp"

namespace aruco3cuda::reference {
namespace {

using aruco3cuda::util::JsonWriter;

/// 対応する定義済み Dictionary。OpenCV の enum への対応表を 1 箇所へ集約する。
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

cv::aruco::DetectorParameters to_detector_parameters(const ReferenceConfig& config) {
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
    params.minMarkerDistanceRate = config.min_marker_distance_rate_;
    params.markerBorderBits = config.marker_border_bits_;
    params.perspectiveRemovePixelPerCell = config.perspective_remove_pixel_per_cell_;
    params.perspectiveRemoveIgnoredMarginPerCell =
            config.perspective_remove_ignored_margin_per_cell_;
    params.maxErroneousBitsInBorderRate = config.max_erroneous_bits_in_border_rate_;
    params.minOtsuStdDev = config.min_otsu_std_dev_;
    params.errorCorrectionRate = config.error_correction_rate_;
    params.useAruco3Detection = config.use_aruco3_detection_;
    params.cornerRefinementMethod = config.use_corner_subpix_refinement_
                                            ? static_cast<int>(cv::aruco::CORNER_REFINE_SUBPIX)
                                            : static_cast<int>(cv::aruco::CORNER_REFINE_NONE);
    params.cornerRefinementWinSize = config.corner_refinement_win_size_px_;
    params.relativeCornerRefinmentWinSize =
            static_cast<float>(config.relative_corner_refinement_win_size_);
    params.cornerRefinementMaxIterations = config.corner_refinement_max_iterations_;
    params.cornerRefinementMinAccuracy = config.corner_refinement_min_accuracy_px_;
    params.minSideLengthCanonicalImg = config.min_side_length_canonical_img_px_;
    params.minMarkerLengthRatioOriginalImg = config.min_marker_length_ratio_original_img_;
    return params;
}

/// ArUco3 の実効縮小率。OpenCV 4.x が内部で用いる式と同じものを再現する。
/// 詳細は docs/design/detector-pipeline.md の観測仕様を参照。
double effective_fxfy(const ReferenceConfig& config, int width_px, int height_px) {
    if (!config.use_aruco3_detection_) {
        return 1.0;
    }
    const double side = static_cast<double>(config.min_side_length_canonical_img_px_);
    const double longest = static_cast<double>(std::max(width_px, height_px));
    const double denominator =
            side + longest * static_cast<double>(config.min_marker_length_ratio_original_img_);
    if (denominator <= 0.0) {
        return 1.0;
    }
    return side / denominator;
}

/// 小さな設定 file を読み込む。上限を超える file は読まない。
///
/// 外部 file を無制限に memory へ読み込まないため、上限を明示する。
/// 対象は image が持つ provenance JSON であり、数 KB に収まる。
std::string read_file_text(const std::string& path) {
    constexpr std::streamsize kMaxFileBytes = 1 << 20;  // 1 MiB
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return std::string();
    }
    input.seekg(0, std::ios::end);
    const std::streamsize size = input.tellg();
    if (size < 0 || size > kMaxFileBytes) {
        return std::string();
    }
    input.seekg(0, std::ios::beg);
    std::string content(static_cast<std::size_t>(size), '\0');
    if (size > 0 && !input.read(content.data(), size)) {
        return std::string();
    }
    return content;
}

}  // namespace

bool validate_config(const ReferenceConfig& config, std::string* out_error) {
    if (out_error == nullptr) {
        return false;
    }
    // OpenCV が assert する条件と、値として意味を成さない範囲を検査する。
    // 検査項目と条件は cv::aruco::DetectorParameters の仕様に対応する。
    struct IntRange {
        const char* name;
        int value;
        int minimum;
        int maximum;
    };
    const IntRange int_ranges[] = {
            {"adaptive_thresh_win_size_min", config.adaptive_thresh_win_size_min_px_, 3, 4096},
            {"adaptive_thresh_win_size_max", config.adaptive_thresh_win_size_max_px_, 3, 4096},
            {"adaptive_thresh_win_size_step", config.adaptive_thresh_win_size_step_px_, 1, 4096},
            {"marker_border_bits", config.marker_border_bits_, 1, 16},
            {"perspective_remove_pixel_per_cell", config.perspective_remove_pixel_per_cell_, 1,
             256},
            {"min_distance_to_border_px", config.min_distance_to_border_px_, 0, 4096},
            {"min_side_length_canonical_img_px", config.min_side_length_canonical_img_px_, 0, 4096},
            {"num_threads", config.num_threads_, 0, 1024},
    };
    for (const IntRange& range : int_ranges) {
        if (range.value < range.minimum || range.value > range.maximum) {
            *out_error = std::string("設定値が範囲外: ") + range.name + "=" +
                         std::to_string(range.value) + " (有効範囲 " +
                         std::to_string(range.minimum) + " から " + std::to_string(range.maximum) +
                         ")";
            return false;
        }
    }
    if (config.adaptive_thresh_win_size_max_px_ < config.adaptive_thresh_win_size_min_px_) {
        *out_error = "設定値が矛盾: adaptive_thresh_win_size_max=" +
                     std::to_string(config.adaptive_thresh_win_size_max_px_) +
                     " が adaptive_thresh_win_size_min=" +
                     std::to_string(config.adaptive_thresh_win_size_min_px_) + " より小さい";
        return false;
    }

    struct DoubleRange {
        const char* name;
        double value;
        double minimum;
        double maximum;
    };
    const DoubleRange double_ranges[] = {
            {"min_marker_perimeter_rate", config.min_marker_perimeter_rate_, 0.0, 8.0},
            {"max_marker_perimeter_rate", config.max_marker_perimeter_rate_, 0.0, 64.0},
            {"polygonal_approx_accuracy_rate", config.polygonal_approx_accuracy_rate_, 0.0, 1.0},
            {"min_corner_distance_rate", config.min_corner_distance_rate_, 0.0, 1.0},
            {"min_marker_distance_rate", config.min_marker_distance_rate_, 0.0, 4.0},
            {"perspective_remove_ignored_margin_per_cell",
             config.perspective_remove_ignored_margin_per_cell_, 0.0, 0.5},
            {"max_erroneous_bits_in_border_rate", config.max_erroneous_bits_in_border_rate_, 0.0,
             1.0},
            {"min_otsu_std_dev", config.min_otsu_std_dev_, 0.0, 255.0},
            {"error_correction_rate", config.error_correction_rate_, 0.0, 1.0},
            {"adaptive_thresh_constant", config.adaptive_thresh_constant_, -255.0, 255.0},
            {"min_marker_length_ratio_original_img",
             static_cast<double>(config.min_marker_length_ratio_original_img_), 0.0, 1.0},
    };
    for (const DoubleRange& range : double_ranges) {
        if (!(range.value >= range.minimum) || !(range.value <= range.maximum)) {
            // NaN は比較が全て false になるため、この書き方で同時に弾ける。
            *out_error = std::string("設定値が範囲外: ") + range.name + "=" +
                         std::to_string(range.value) + " (有効範囲 " +
                         std::to_string(range.minimum) + " から " + std::to_string(range.maximum) +
                         ")";
            return false;
        }
    }
    if (config.max_marker_perimeter_rate_ <= config.min_marker_perimeter_rate_) {
        *out_error = "設定値が矛盾: max_marker_perimeter_rate が min_marker_perimeter_rate 以下";
        return false;
    }
    // OpenCV は ArUco3 有効時にこの組み合わせを assert で拒否する。
    if (config.use_aruco3_detection_ && config.min_side_length_canonical_img_px_ == 0 &&
        config.min_marker_length_ratio_original_img_ == 0.0F) {
        *out_error =
                "設定値が矛盾: use_aruco3_detection が有効で "
                "min_side_length_canonical_img と min_marker_length_ratio_original_img が共に 0";
        return false;
    }
    return true;
}

bool is_known_dictionary(const std::string& name) {
    return dictionary_table().find(name) != dictionary_table().end();
}

std::vector<std::string> known_dictionary_names() {
    std::vector<std::string> names;
    names.reserve(dictionary_table().size());
    for (const auto& entry : dictionary_table()) {
        names.push_back(entry.first);
    }
    return names;
}

namespace {

/// 読み込み済みの画像を検出し、結果を並べ替えて格納する。
///
/// 検出そのものだけを行う。file 読み込みと checksum は呼出側の責務とする。
void run_detection(const cv::aruco::ArucoDetector& detector, const cv::Mat& image,
                   ReferenceResult* result) {
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> rejected;

    const auto start = std::chrono::steady_clock::now();
    detector.detectMarkers(image, corners, ids, rejected);
    const auto finish = std::chrono::steady_clock::now();
    result->detect_ms_ = std::chrono::duration<double, std::milli>(finish - start).count();
    result->rejected_count_ = rejected.size();

    result->detections_.clear();
    result->detections_.reserve(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) {
        ReferenceDetection detection;
        detection.id_ = ids[i];
        for (std::size_t c = 0; c < 4; ++c) {
            detection.corners_[c * 2] = static_cast<double>(corners[i][c].x);
            detection.corners_[c * 2 + 1] = static_cast<double>(corners[i][c].y);
        }
        result->detections_.push_back(detection);
    }

    // OpenCV が返す順序は候補の抽出順に依存する。比較を容易にするため安定に並べ替える。
    std::sort(result->detections_.begin(), result->detections_.end(),
              [](const ReferenceDetection& a, const ReferenceDetection& b) {
                  if (a.id_ != b.id_) {
                      return a.id_ < b.id_;
                  }
                  if (a.corners_[0] != b.corners_[0]) {
                      return a.corners_[0] < b.corners_[0];
                  }
                  return a.corners_[1] < b.corners_[1];
              });
}

/// 画像を読み込み、寸法と checksum と縮小率を求める。
///
/// 成功時のみ out_image と out_result を書き換える。
bool load_image(const std::string& image_path, const ReferenceConfig& config, cv::Mat* out_image,
                ReferenceResult* out_result, std::string* out_error) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        *out_error = "画像を読み込めない: " + image_path;
        return false;
    }
    ReferenceResult result;
    result.image_path_ = image_path;
    result.width_px_ = image.cols;
    result.height_px_ = image.rows;
    if (!aruco3cuda::util::sha256_file(image_path, &result.image_sha256_)) {
        *out_error = "checksum を計算できない: " + image_path;
        return false;
    }
    result.fxfy_effective_ = effective_fxfy(config, image.cols, image.rows);
    result.segmentation_width_px_ = cvRound(result.fxfy_effective_ * image.cols);
    result.segmentation_height_px_ = cvRound(result.fxfy_effective_ * image.rows);

    *out_image = std::move(image);
    *out_result = result;
    return true;
}

/// 設定と Dictionary 名を検証し、OpenCV の Dictionary を返す。
bool resolve_dictionary(const ReferenceConfig& config, cv::aruco::Dictionary* out_dictionary,
                        std::string* out_error) {
    // 設定値を最初に検証する。範囲外の値を OpenCV へ渡すと cv::Exception が
    // 送出され、bool と out_error で失敗を通知する契約を破る。
    if (!validate_config(config, out_error)) {
        return false;
    }
    const auto dictionary_entry = dictionary_table().find(config.dictionary_name_);
    if (dictionary_entry == dictionary_table().end()) {
        *out_error = "未対応の Dictionary: " + config.dictionary_name_;
        return false;
    }
    *out_dictionary = cv::aruco::getPredefinedDictionary(dictionary_entry->second);
    return true;
}

}  // namespace

bool detect_image(const std::string& image_path, const ReferenceConfig& config,
                  ReferenceResult* out_result, std::string* out_error) {
    if (out_result == nullptr || out_error == nullptr) {
        return false;
    }
    cv::aruco::Dictionary dictionary;
    if (!resolve_dictionary(config, &dictionary, out_error)) {
        return false;
    }
    cv::Mat image;
    ReferenceResult result;
    if (!load_image(image_path, config, &image, &result, out_error)) {
        return false;
    }
    const cv::aruco::ArucoDetector detector(dictionary, to_detector_parameters(config));
    run_detection(detector, image, &result);
    *out_result = result;
    return true;
}

/// ReferenceDetector の実装本体。公開 header が OpenCV へ依存しないよう pimpl とする。
class ReferenceDetector::Impl {
public:
    bool initialize(const std::string& image_path, const ReferenceConfig& config,
                    std::string* out_error) {
        cv::aruco::Dictionary dictionary;
        if (!resolve_dictionary(config, &dictionary, out_error)) {
            return false;
        }
        if (!load_image(image_path, config, &this->image_, &this->metadata_, out_error)) {
            return false;
        }
        this->detector_ = cv::aruco::ArucoDetector(dictionary, to_detector_parameters(config));
        this->initialized_ = true;
        return true;
    }

    bool detect(ReferenceResult* out_result, std::string* out_error) {
        if (!this->initialized_) {
            *out_error = "initialize() が呼ばれていない";
            return false;
        }
        ReferenceResult result = this->metadata_;
        run_detection(this->detector_, this->image_, &result);
        *out_result = std::move(result);
        return true;
    }

    const ReferenceResult& metadata() const { return this->metadata_; }

private:
    bool initialized_ = false;
    cv::Mat image_;
    ReferenceResult metadata_;
    cv::aruco::ArucoDetector detector_;
};

ReferenceDetector::ReferenceDetector() : impl_(std::make_unique<Impl>()) {}
ReferenceDetector::~ReferenceDetector() = default;
ReferenceDetector::ReferenceDetector(ReferenceDetector&&) noexcept = default;
ReferenceDetector& ReferenceDetector::operator=(ReferenceDetector&&) noexcept = default;

bool ReferenceDetector::initialize(const std::string& image_path, const ReferenceConfig& config,
                                   std::string* out_error) {
    if (out_error == nullptr) {
        return false;
    }
    return this->impl_->initialize(image_path, config, out_error);
}

bool ReferenceDetector::detect(ReferenceResult* out_result, std::string* out_error) {
    if (out_result == nullptr || out_error == nullptr) {
        return false;
    }
    return this->impl_->detect(out_result, out_error);
}

const ReferenceResult& ReferenceDetector::metadata() const {
    return this->impl_->metadata();
}

ReferenceEnvironment collect_environment(const ReferenceConfig& config) {
    ReferenceEnvironment environment;
    environment.opencv_version_ = CV_VERSION;
    if (config.num_threads_ > 0) {
        cv::setNumThreads(config.num_threads_);
    }
    environment.opencv_threads_ = cv::getNumThreads();
    // container image が持つ OpenCV の取得元情報。存在しない環境では空になる。
    environment.opencv_provenance_json_ =
            read_file_text("/opt/opencv/share/aruco3cuda/opencv-provenance.json");
    return environment;
}

void write_results_json(std::ostream& out, const ReferenceConfig& config,
                        const ReferenceEnvironment& environment,
                        const std::vector<ReferenceResult>& results) {
    JsonWriter writer(out);
    writer.begin_object();

    writer.member_int("schema_version", 1);
    writer.member_string("producer", "aruco3cuda_reference_runner");

    writer.key("environment");
    writer.begin_object();
    writer.member_string("opencv_version", environment.opencv_version_);
    writer.member_int("opencv_threads", environment.opencv_threads_);
    writer.member_bool("opencv_provenance_available", !environment.opencv_provenance_json_.empty());
    writer.end_object();

    writer.key("detector");
    writer.begin_object();
    writer.member_string("dictionary", config.dictionary_name_);
    writer.member_int("adaptiveThreshWinSizeMin", config.adaptive_thresh_win_size_min_px_);
    writer.member_int("adaptiveThreshWinSizeMax", config.adaptive_thresh_win_size_max_px_);
    writer.member_int("adaptiveThreshWinSizeStep", config.adaptive_thresh_win_size_step_px_);
    writer.member_double("adaptiveThreshConstant", config.adaptive_thresh_constant_, 6);
    writer.member_double("minMarkerPerimeterRate", config.min_marker_perimeter_rate_, 6);
    writer.member_double("maxMarkerPerimeterRate", config.max_marker_perimeter_rate_, 6);
    writer.member_double("polygonalApproxAccuracyRate", config.polygonal_approx_accuracy_rate_, 6);
    writer.member_double("minCornerDistanceRate", config.min_corner_distance_rate_, 6);
    writer.member_int("minDistanceToBorder", config.min_distance_to_border_px_);
    writer.member_double("minMarkerDistanceRate", config.min_marker_distance_rate_, 6);
    writer.member_int("markerBorderBits", config.marker_border_bits_);
    writer.member_int("perspectiveRemovePixelPerCell", config.perspective_remove_pixel_per_cell_);
    writer.member_double("perspectiveRemoveIgnoredMarginPerCell",
                         config.perspective_remove_ignored_margin_per_cell_, 6);
    writer.member_double("maxErroneousBitsInBorderRate", config.max_erroneous_bits_in_border_rate_,
                         6);
    writer.member_double("minOtsuStdDev", config.min_otsu_std_dev_, 6);
    writer.member_double("errorCorrectionRate", config.error_correction_rate_, 6);
    writer.member_bool("useAruco3Detection", config.use_aruco3_detection_);
    writer.member_int("minSideLengthCanonicalImg", config.min_side_length_canonical_img_px_);
    writer.member_double("minMarkerLengthRatioOriginalImg",
                         static_cast<double>(config.min_marker_length_ratio_original_img_), 6);
    writer.member_bool("useCornerSubpixRefinement", config.use_corner_subpix_refinement_);
    writer.member_int("cornerRefinementWinSize", config.corner_refinement_win_size_px_);
    writer.member_double("relativeCornerRefinmentWinSize",
                         config.relative_corner_refinement_win_size_, 6);
    writer.member_int("cornerRefinementMaxIterations", config.corner_refinement_max_iterations_);
    writer.member_double("cornerRefinementMinAccuracy", config.corner_refinement_min_accuracy_px_,
                         6);
    writer.end_object();

    writer.key("images");
    writer.begin_array();
    for (const ReferenceResult& result : results) {
        writer.begin_object();
        writer.member_string("path", result.image_path_);
        writer.member_string("sha256", result.image_sha256_);
        writer.member_int("width_px", result.width_px_);
        writer.member_int("height_px", result.height_px_);
        writer.member_double("fxfy_effective", result.fxfy_effective_, 6);
        writer.member_int("segmentation_width_px", result.segmentation_width_px_);
        writer.member_int("segmentation_height_px", result.segmentation_height_px_);
        writer.member_int("rejected_count", static_cast<long long>(result.rejected_count_));
        if (!config.omit_timing_) {
            writer.member_double("detect_ms", result.detect_ms_, 3);
        }

        writer.key("detections");
        writer.begin_array();
        for (const ReferenceDetection& detection : result.detections_) {
            writer.begin_object();
            writer.member_int("id", detection.id_);
            writer.key("corners");
            writer.begin_array();
            for (std::size_t c = 0; c < 4; ++c) {
                writer.begin_array();
                writer.value_double(detection.corners_[c * 2], 4);
                writer.value_double(detection.corners_[c * 2 + 1], 4);
                writer.end_array();
            }
            writer.end_array();
            writer.end_object();
        }
        writer.end_array();
        writer.end_object();
    }
    writer.end_array();

    writer.end_object();
    out << '\n';
}

}  // namespace aruco3cuda::reference
