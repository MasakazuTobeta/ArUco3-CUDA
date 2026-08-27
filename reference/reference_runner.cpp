// SPDX-License-Identifier: Apache-2.0
#include "reference_runner.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <cstddef>
#include <map>
#include <ostream>
#include <ratio>
#include <sstream>
#include <string>
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
    params.adaptiveThreshWinSizeMin = config.adaptive_thresh_win_size_min_;
    params.adaptiveThreshWinSizeMax = config.adaptive_thresh_win_size_max_;
    params.adaptiveThreshWinSizeStep = config.adaptive_thresh_win_size_step_;
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

std::string read_file_text(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        return std::string();
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

}  // namespace

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

bool detect_image(const std::string& image_path, const ReferenceConfig& config,
                  ReferenceResult* out_result, std::string* out_error) {
    if (out_result == nullptr || out_error == nullptr) {
        return false;
    }
    const auto dictionary_entry = dictionary_table().find(config.dictionary_name_);
    if (dictionary_entry == dictionary_table().end()) {
        *out_error = "未対応の Dictionary: " + config.dictionary_name_;
        return false;
    }

    const cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
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

    const cv::aruco::Dictionary dictionary =
            cv::aruco::getPredefinedDictionary(dictionary_entry->second);
    const cv::aruco::ArucoDetector detector(dictionary, to_detector_parameters(config));

    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> rejected;

    const auto start = std::chrono::steady_clock::now();
    detector.detectMarkers(image, corners, ids, rejected);
    const auto finish = std::chrono::steady_clock::now();
    result.detect_ms_ =
            std::chrono::duration<double, std::milli>(finish - start).count();
    result.rejected_count_ = rejected.size();

    result.detections_.reserve(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) {
        ReferenceDetection detection;
        detection.id_ = ids[i];
        for (std::size_t c = 0; c < 4; ++c) {
            detection.corners_[c * 2] = static_cast<double>(corners[i][c].x);
            detection.corners_[c * 2 + 1] = static_cast<double>(corners[i][c].y);
        }
        result.detections_.push_back(detection);
    }

    // OpenCV が返す順序は候補の抽出順に依存する。比較を容易にするため安定に並べ替える。
    std::sort(result.detections_.begin(), result.detections_.end(),
              [](const ReferenceDetection& a, const ReferenceDetection& b) {
                  if (a.id_ != b.id_) {
                      return a.id_ < b.id_;
                  }
                  if (a.corners_[0] != b.corners_[0]) {
                      return a.corners_[0] < b.corners_[0];
                  }
                  return a.corners_[1] < b.corners_[1];
              });

    *out_result = result;
    return true;
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
    writer.member_bool("opencv_provenance_available",
                       !environment.opencv_provenance_json_.empty());
    writer.end_object();

    writer.key("detector");
    writer.begin_object();
    writer.member_string("dictionary", config.dictionary_name_);
    writer.member_int("adaptiveThreshWinSizeMin", config.adaptive_thresh_win_size_min_);
    writer.member_int("adaptiveThreshWinSizeMax", config.adaptive_thresh_win_size_max_);
    writer.member_int("adaptiveThreshWinSizeStep", config.adaptive_thresh_win_size_step_);
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
    writer.member_double("maxErroneousBitsInBorderRate",
                         config.max_erroneous_bits_in_border_rate_, 6);
    writer.member_double("minOtsuStdDev", config.min_otsu_std_dev_, 6);
    writer.member_double("errorCorrectionRate", config.error_correction_rate_, 6);
    writer.member_bool("useAruco3Detection", config.use_aruco3_detection_);
    writer.member_int("minSideLengthCanonicalImg", config.min_side_length_canonical_img_px_);
    writer.member_double("minMarkerLengthRatioOriginalImg",
                         static_cast<double>(config.min_marker_length_ratio_original_img_), 6);
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
