// SPDX-License-Identifier: Apache-2.0
//
// 正確性評価 tool の CLI。
//
// 目的:
//   合成 corpus の ground truth に対して CPU 基準、hybrid、CUDA の 3 経路を
//   走らせ、precision、recall、ID と rotation の一致率、四隅の RMSE を条件別に
//   報告する。差分レポート (tools/report) は CPU 基準を基準に据えるため、基準
//   自身が取りこぼしたマーカーは差異として現れない。真値と突き合わせる経路を
//   別に持つ。
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cuda_runtime_api.h>

#include "accuracy.hpp"
#include "aruco3cuda/config.hpp"
#include "aruco3cuda/detections.hpp"
#include "aruco3cuda/detector.hpp"
#include "aruco3cuda/dictionary.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"
#include "aruco3cuda/util/json_writer.hpp"
#include "aruco3cuda/workspace.hpp"
#include "corpus_generator.hpp"
#include "device_image.hpp"
#include "hybrid_detector.hpp"
#include "reference_runner.hpp"
#include "report_diff.hpp"

namespace {

using aruco3cuda::evaluate::AccuracySummary;
using aruco3cuda::evaluate::ImageAccuracy;
using aruco3cuda::evaluate::MatchConfig;
using aruco3cuda::evaluate::Observation;
using aruco3cuda::evaluate::TruthMarker;

/// 評価する経路。
enum class Route {
    kCpu,
    kHybrid,
    kCuda,
};

/// 経路の表示名。報告と JSON の key に使う。
const char* route_name(Route route) {
    switch (route) {
        case Route::kCpu:
            return "CPU";
        case Route::kHybrid:
            return "Hybrid";
        case Route::kCuda:
            return "CUDA";
    }
    return "不明";
}

constexpr std::size_t kRouteCount = 3U;
const Route kRoutes[kRouteCount] = {Route::kCpu, Route::kHybrid, Route::kCuda};

/// 経路 1 つ分の集計。
struct RouteTotals {
    /// 真値との突き合わせ。
    AccuracySummary truth_;
    /// 条件ごとの真値との突き合わせ。key は場面名の先頭語。
    std::map<std::string, AccuracySummary> by_condition_;
    /// 解像度ごと。key は "1280x720"。
    std::map<std::string, AccuracySummary> by_resolution_;
    /// マーカー 1 辺ごと。key は pixel 値。
    std::map<int, AccuracySummary> by_side_px_;
    /// ArUco3 の下限以上の大きさを持つ真値だけの集計。recall はここを読む。
    AccuracySummary detectable_;
    /// 下限を下回る真値だけの集計。原理上検出されないことの確認に使う。
    AccuracySummary below_limit_;
    /// CPU 基準との差異。CPU 経路自身では使わない。
    std::vector<aruco3cuda::report::ImageComparison> comparisons_;
};

/// 実行中に集めた device memory の使用量。
struct MemoryUsage {
    std::size_t peak_used_bytes_ = 0;
    std::size_t capacity_bytes_ = 0;
    std::size_t allocation_count_ = 0;
    std::size_t reallocation_count_ = 0;
    /// 検出 1 回あたりに増えた確保回数。定常状態では 0 になる。
    std::size_t per_frame_allocation_count_ = 0;
};

void print_usage(std::ostream& out) {
    out << "使用方法: aruco3cuda_evaluate [option]...\n"
        << "\n"
        << "  --preset <name>                corpus preset。既定 full\n"
        << "  --seed <u64>                   corpus の乱数種。既定 20260827\n"
        << "  --corpus-dir <path>            corpus の生成先。既定 /tmp/aruco3eval\n"
        << "  --dictionary <name>            既定 DICT_ARUCO_MIP_36h12\n"
        << "  --output <path>                報告 JSON の出力先。既定は出力しない\n"
        << "  --match-radius-ratio <f>       対応付けの半径。辺長比。既定 0.5\n"
        << "  --corner-tolerance-px <f>      CPU 基準との比較の許容差。既定 1.0\n"
        << "  --use-aruco3 <0|1>             ArUco3 検出戦略。既定 1\n"
        << "  --help                         この説明を表示して終了\n";
}

bool take_value(int argc, char** argv, int* index, const char* option, std::string* out) {
    if (*index + 1 >= argc) {
        std::cerr << "引数が不足している: " << option << '\n';
        return false;
    }
    ++(*index);
    *out = argv[*index];
    return true;
}

bool parse_double(const std::string& text, double* out) {
    try {
        std::size_t consumed = 0;
        const double value = std::stod(text, &consumed);
        if (consumed != text.size()) {
            return false;
        }
        *out = value;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool parse_u64(const std::string& text, std::uint64_t* out) {
    try {
        std::size_t consumed = 0;
        const unsigned long long value = std::stoull(text, &consumed);
        if (consumed != text.size()) {
            return false;
        }
        *out = static_cast<std::uint64_t>(value);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool parse_bool_flag(const std::string& text, bool* out) {
    if (text == "0") {
        *out = false;
        return true;
    }
    if (text == "1") {
        *out = true;
        return true;
    }
    return false;
}

/// 場面名の先頭語。条件別の集計の key に使う。
///
/// 場面名は "clean_1280x720_n4_s128" や "blur_1920x1080" の形であり、
/// 先頭語がそのまま条件を表す。
std::string condition_of(const std::string& scene_name) {
    const std::size_t separator = scene_name.find('_');
    if (separator == std::string::npos) {
        return scene_name;
    }
    return scene_name.substr(0, separator);
}

/// 真値を評価用の形へ移す。
std::vector<TruthMarker> to_truth(const aruco3cuda::corpusgen::GeneratedScene& scene) {
    std::vector<TruthMarker> truth;
    truth.reserve(scene.markers_.size());
    for (const auto& marker : scene.markers_) {
        TruthMarker entry;
        entry.id_ = marker.id_;
        entry.corners_ = marker.corners_;
        entry.side_px_ = marker.side_px_;
        entry.fully_inside_ = marker.fully_inside_;
        entry.occlusion_ratio_ = marker.occlusion_ratio_;
        truth.push_back(entry);
    }
    return truth;
}

/// 4 隅の float 配列を double の四隅へ移す。
aruco3cuda::report::Quad to_quad(const float* corners) {
    aruco3cuda::report::Quad quad{};
    for (std::size_t c = 0; c < 8U; ++c) {
        quad[c] = static_cast<double>(corners[c]);
    }
    return quad;
}

/// 1 枚分の結果を経路の集計へ足し込む。
///
/// @param detectable_limit_px ArUco3 が原理上検出できる 1 辺の下限。
void record(const aruco3cuda::corpusgen::GeneratedScene& scene,
            const std::vector<Observation>& observed, const MatchConfig& match,
            double detectable_limit_px, RouteTotals* out_totals) {
    const std::vector<TruthMarker> truth = to_truth(scene);
    const ImageAccuracy image =
            aruco3cuda::evaluate::compare_to_truth(scene.name_, truth, observed, match);
    std::vector<bool> detectable(truth.size(), false);
    std::vector<bool> below_limit(truth.size(), false);
    for (std::size_t m = 0; m < truth.size(); ++m) {
        detectable[m] = truth[m].side_px_ >= detectable_limit_px;
        below_limit[m] = !detectable[m];
    }
    aruco3cuda::evaluate::accumulate_selected(image, detectable, &out_totals->detectable_);
    aruco3cuda::evaluate::accumulate_selected(image, below_limit, &out_totals->below_limit_);
    aruco3cuda::evaluate::accumulate(image, &out_totals->truth_);
    aruco3cuda::evaluate::accumulate(image, &out_totals->by_condition_[condition_of(scene.name_)]);
    const std::string resolution =
            std::to_string(scene.width_px_) + "x" + std::to_string(scene.height_px_);
    aruco3cuda::evaluate::accumulate(image, &out_totals->by_resolution_[resolution]);
    if (!scene.markers_.empty()) {
        aruco3cuda::evaluate::accumulate(
                image, &out_totals->by_side_px_[static_cast<int>(scene.spec_.marker_side_px_)]);
    }
}

/// 検出を差分レポートの形へ移す。
std::vector<aruco3cuda::report::Detection> to_detections(const std::vector<Observation>& observed) {
    std::vector<aruco3cuda::report::Detection> result;
    result.reserve(observed.size());
    for (const Observation& entry : observed) {
        result.push_back({entry.id_, entry.corners_});
    }
    return result;
}

/// 割合を百分率の文字列にする。定義できない場合は "-" を返す。
std::string percent_or_dash(bool defined, double value) {
    if (!defined) {
        return "-";
    }
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.2f%%", value * 100.0);
    return std::string(buffer);
}

/// 集計 1 件を 1 行で書く。
///
/// @param recall_only 真値の部分集合だけを集めた集計なら true。対応の付かない
///                    検出を含まないため precision と検出数を表示しない。
///                    表示すると「検出 0 件で precision 100%」のような、
///                    数え方の都合でしかない値が読まれてしまう。
void write_summary_line(std::ostream& out, const std::string& label, const AccuracySummary& summary,
                        bool recall_only) {
    double precision_value = 0.0;
    double recall_value = 0.0;
    double rmse_value = 0.0;
    const bool has_precision =
            !recall_only && aruco3cuda::evaluate::precision(summary, &precision_value);
    const bool has_recall = aruco3cuda::evaluate::recall(summary, &recall_value);
    const bool has_rmse = aruco3cuda::evaluate::corner_rmse_px(summary, &rmse_value);
    out << "  " << std::left << std::setw(18) << label << std::right << " 真値 " << std::setw(5)
        << summary.truth_count_;
    if (recall_only) {
        out << "                 ";
    } else {
        out << " 検出 " << std::setw(5) << summary.observed_count_;
    }
    out << " TP " << std::setw(5) << summary.true_positive_;
    if (recall_only) {
        out << "         ";
    } else {
        out << " FP " << std::setw(4) << summary.false_positive_;
    }
    out << " FN " << std::setw(4) << summary.false_negative_ << "  precision " << std::setw(8)
        << percent_or_dash(has_precision, precision_value) << "  recall " << std::setw(8)
        << percent_or_dash(has_recall, recall_value) << "  rotation 一致 " << std::setw(8)
        << percent_or_dash(summary.true_positive_ != 0U,
                           summary.true_positive_ == 0U
                                   ? 0.0
                                   : static_cast<double>(summary.rotation_agreed_) /
                                             static_cast<double>(summary.true_positive_));
    if (has_rmse) {
        out << "  四隅 RMSE " << std::fixed << std::setprecision(4) << rmse_value << " px 最大 "
            << summary.corner_max_px_ << " px" << std::defaultfloat;
    }
    out << '\n';
}

/// 集計 1 件を JSON へ書く。
void write_summary_json(aruco3cuda::util::JsonWriter& writer, const AccuracySummary& summary,
                        bool recall_only) {
    writer.begin_object();
    writer.member_int("imageCount", static_cast<long long>(summary.image_count_));
    writer.member_int("truthCount", static_cast<long long>(summary.truth_count_));
    writer.member_int("observedCount", static_cast<long long>(summary.observed_count_));
    writer.member_int("truePositive", static_cast<long long>(summary.true_positive_));
    writer.member_int("falsePositive", static_cast<long long>(summary.false_positive_));
    writer.member_int("falseNegative", static_cast<long long>(summary.false_negative_));
    writer.member_int("rotationAgreed", static_cast<long long>(summary.rotation_agreed_));
    double value = 0.0;
    if (!recall_only && aruco3cuda::evaluate::precision(summary, &value)) {
        writer.member_double("precision", value, 6);
    }
    if (aruco3cuda::evaluate::recall(summary, &value)) {
        writer.member_double("recall", value, 6);
    }
    if (aruco3cuda::evaluate::corner_rmse_px(summary, &value)) {
        writer.member_double("cornerRmsePx", value, 6);
    }
    writer.member_double("cornerMaxPx", summary.corner_max_px_, 6);
    writer.end_object();
}

}  // namespace

int main(int argc, char** argv) {
    std::string preset = "full";
    std::uint64_t seed = 20260827U;
    std::string corpus_dir = "/tmp/aruco3eval";
    std::string dictionary_name = "DICT_ARUCO_MIP_36h12";
    std::string output_path;
    MatchConfig match;
    aruco3cuda::report::CompareConfig compare_config;
    bool use_aruco3 = true;

    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        std::string value;
        if (argument == "--help") {
            print_usage(std::cout);
            return EXIT_SUCCESS;
        }
        if (argument == "--preset") {
            if (!take_value(argc, argv, &i, "--preset", &preset)) {
                return EXIT_FAILURE;
            }
            continue;
        }
        if (argument == "--seed") {
            if (!take_value(argc, argv, &i, "--seed", &value) || !parse_u64(value, &seed)) {
                std::cerr << "--seed は符号なし整数である必要がある\n";
                return EXIT_FAILURE;
            }
            continue;
        }
        if (argument == "--corpus-dir") {
            if (!take_value(argc, argv, &i, "--corpus-dir", &corpus_dir)) {
                return EXIT_FAILURE;
            }
            continue;
        }
        if (argument == "--dictionary") {
            if (!take_value(argc, argv, &i, "--dictionary", &dictionary_name)) {
                return EXIT_FAILURE;
            }
            continue;
        }
        if (argument == "--output") {
            if (!take_value(argc, argv, &i, "--output", &output_path)) {
                return EXIT_FAILURE;
            }
            continue;
        }
        if (argument == "--match-radius-ratio") {
            if (!take_value(argc, argv, &i, "--match-radius-ratio", &value) ||
                !parse_double(value, &match.match_radius_ratio_) ||
                match.match_radius_ratio_ <= 0.0) {
                std::cerr << "--match-radius-ratio は正の実数である必要がある\n";
                return EXIT_FAILURE;
            }
            continue;
        }
        if (argument == "--corner-tolerance-px") {
            if (!take_value(argc, argv, &i, "--corner-tolerance-px", &value) ||
                !parse_double(value, &compare_config.corner_tolerance_px_) ||
                compare_config.corner_tolerance_px_ < 0.0) {
                std::cerr << "--corner-tolerance-px は 0 以上の実数である必要がある\n";
                return EXIT_FAILURE;
            }
            continue;
        }
        if (argument == "--use-aruco3") {
            if (!take_value(argc, argv, &i, "--use-aruco3", &value) ||
                !parse_bool_flag(value, &use_aruco3)) {
                std::cerr << "--use-aruco3 は 0 か 1 である必要がある\n";
                return EXIT_FAILURE;
            }
            continue;
        }
        std::cerr << "不明な option: " << argument << '\n';
        print_usage(std::cerr);
        return EXIT_FAILURE;
    }
    compare_config.match_radius_ratio_ = match.match_radius_ratio_;

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        std::cerr << "CUDA device が見つからない\n";
        return EXIT_FAILURE;
    }

    std::vector<aruco3cuda::corpusgen::SceneSpec> specs;
    if (!aruco3cuda::corpusgen::build_preset(preset, &specs)) {
        std::cerr << "未知の preset: " << preset << '\n';
        return EXIT_FAILURE;
    }

    aruco3cuda::corpusgen::CorpusConfig corpus_config;
    corpus_config.dictionary_name_ = dictionary_name;
    corpus_config.output_dir_ = corpus_dir;
    corpus_config.seed_ = seed;

    aruco3cuda::DetectorConfig detector_config;
    if (!use_aruco3) {
        detector_config = aruco3cuda::DetectorConfig::opencv_defaults();
    }
    detector_config.use_aruco3_detection_ = use_aruco3;

    aruco3cuda::reference::ReferenceConfig reference_config;
    reference_config.dictionary_name_ = dictionary_name;
    reference_config.use_aruco3_detection_ = detector_config.use_aruco3_detection_;
    reference_config.min_side_length_canonical_img_px_ =
            detector_config.min_side_length_canonical_img_px_;
    reference_config.min_marker_length_ratio_original_img_ =
            detector_config.min_marker_length_ratio_original_img_;
    reference_config.use_corner_subpix_refinement_ =
            detector_config.corner_refine_method_ == aruco3cuda::CornerRefineMethod::kSubpix;

    const aruco3cuda::DictionaryTable* table =
            aruco3cuda::find_builtin_dictionary(dictionary_name.c_str());
    if (table == nullptr) {
        std::cerr << "未対応の Dictionary: " << dictionary_name << '\n';
        return EXIT_FAILURE;
    }

    RouteTotals totals[kRouteCount];
    MemoryUsage memory;
    std::size_t detect_call_count = 0;

    for (std::size_t index = 0; index < specs.size(); ++index) {
        aruco3cuda::corpusgen::GeneratedScene scene;
        std::string error;
        if (!aruco3cuda::corpusgen::generate_scene(corpus_config, specs[index], index, &scene,
                                                   &error)) {
            std::cerr << "場面を生成できない: " << error << '\n';
            return EXIT_FAILURE;
        }

        const cv::Mat image = cv::imread(scene.path_, cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            std::cerr << "画像を読めない: " << scene.path_ << '\n';
            return EXIT_FAILURE;
        }

        std::vector<Observation> observed[kRouteCount];

        aruco3cuda::reference::ReferenceResult reference_result;
        if (!aruco3cuda::reference::detect_image(scene.path_, reference_config, &reference_result,
                                                 &error)) {
            std::cerr << "CPU 基準の検出に失敗した: " << error << '\n';
            return EXIT_FAILURE;
        }
        for (const auto& detection : reference_result.detections_) {
            observed[0].push_back({detection.id_, detection.corners_});
        }

        std::string message;
        aruco3cuda::hybrid::DeviceImage device;
        if (device.reserve(aruco3cuda::MemorySpace::kDevice, image.cols, image.rows, &message) !=
            aruco3cuda::Status::kOk) {
            std::cerr << "device buffer を確保できない: " << message << '\n';
            return EXIT_FAILURE;
        }
        if (device.upload(image.data, image.cols, image.rows, static_cast<std::size_t>(image.step),
                          &message) != aruco3cuda::Status::kOk) {
            std::cerr << "device への転送に失敗した: " << message << '\n';
            return EXIT_FAILURE;
        }

        aruco3cuda::hybrid::HybridDetector hybrid;
        if (hybrid.initialize(detector_config, dictionary_name, image.cols, image.rows, &message) !=
            aruco3cuda::Status::kOk) {
            std::cerr << "hybrid 経路の初期化に失敗した: " << message << '\n';
            return EXIT_FAILURE;
        }
        aruco3cuda::hybrid::HybridResult hybrid_result;
        if (hybrid.detect(device.view(), &hybrid_result, &message) != aruco3cuda::Status::kOk) {
            std::cerr << "hybrid 経路の検出に失敗した: " << message << '\n';
            return EXIT_FAILURE;
        }
        for (const auto& detection : hybrid_result.detections_) {
            observed[1].push_back({detection.id_, detection.corners_});
        }

        // workspace の確保量は画像に合わせる。既定の上限のままだと、小さい
        // 画像でも 4K 相当の領域を抱えたまま測ることになる。
        aruco3cuda::DetectorConfig cuda_config = detector_config;
        cuda_config.max_width_px_ = image.cols;
        cuda_config.max_height_px_ = image.rows;
        aruco3cuda::Detector cuda;
        if (cuda.initialize(*table, cuda_config, &message) != aruco3cuda::Status::kOk) {
            std::cerr << "CUDA 経路の初期化に失敗した: " << message << '\n';
            return EXIT_FAILURE;
        }
        cudaStream_t stream = nullptr;
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
            std::cerr << "stream を作れない\n";
            return EXIT_FAILURE;
        }
        const std::size_t before = cuda.workspace_statistics().allocation_count_;
        aruco3cuda::HostDetections host_detections;
        const aruco3cuda::Status detected = cuda.detect_async(device.view(), stream, &message);
        aruco3cuda::Status downloaded = aruco3cuda::Status::kOk;
        if (detected == aruco3cuda::Status::kOk) {
            downloaded = cuda.download(&host_detections, stream, &message);
        }
        const aruco3cuda::WorkspaceStatistics statistics = cuda.workspace_statistics();
        static_cast<void>(cudaStreamDestroy(stream));
        if (detected != aruco3cuda::Status::kOk || downloaded != aruco3cuda::Status::kOk) {
            std::cerr << "CUDA 経路の検出に失敗した: " << message << '\n';
            return EXIT_FAILURE;
        }
        for (std::size_t d = 0; d < host_detections.ids_.size(); ++d) {
            observed[2].push_back(
                    {host_detections.ids_[d], to_quad(&host_detections.corners_[d * 8U])});
        }
        ++detect_call_count;
        memory.peak_used_bytes_ = std::max(memory.peak_used_bytes_, statistics.peak_used_bytes_);
        memory.capacity_bytes_ = std::max(memory.capacity_bytes_, statistics.capacity_bytes_);
        memory.allocation_count_ += statistics.allocation_count_;
        memory.reallocation_count_ += statistics.reallocation_count_;
        // 初期化で確保した分を差し引き、検出そのものが増やした回数だけを数える。
        memory.per_frame_allocation_count_ += statistics.allocation_count_ - before;

        // ArUco3 は縮小後の 1 辺が min_side_length_canonical_img を下回る
        // マーカーを検出しない。下限を下回る真値を含めたまま recall を出すと、
        // 実装の取りこぼしと戦略上の下限が混ざる。
        const double detectable_limit_px =
                use_aruco3 ? aruco3cuda::corpusgen::minimum_detectable_side_px(
                                     detector_config.min_side_length_canonical_img_px_,
                                     std::max(image.cols, image.rows),
                                     detector_config.min_marker_length_ratio_original_img_)
                           : 0.0;
        for (std::size_t r = 0; r < kRouteCount; ++r) {
            record(scene, observed[r], match, detectable_limit_px, &totals[r]);
        }
        const std::vector<aruco3cuda::report::Detection> baseline = to_detections(observed[0]);
        for (std::size_t r = 1; r < kRouteCount; ++r) {
            totals[r].comparisons_.push_back(aruco3cuda::report::compare_detections(
                    scene.name_, baseline, to_detections(observed[r]), compare_config));
        }
    }

    std::cout << "=== 正確性評価 ===\n";
    std::cout << "preset " << preset << " / 場面 " << specs.size() << " 枚 / seed " << seed << '\n';
    std::cout << "\n真値との突き合わせ (全て)\n";
    for (std::size_t r = 0; r < kRouteCount; ++r) {
        write_summary_line(std::cout, route_name(kRoutes[r]), totals[r].truth_, false);
    }
    std::cout << "\n真値との突き合わせ (ArUco3 の下限以上の大きさのみ。precision は"
                 "この区分では定義しない)\n";
    for (std::size_t r = 0; r < kRouteCount; ++r) {
        write_summary_line(std::cout, route_name(kRoutes[r]), totals[r].detectable_, true);
    }
    std::cout << "\n真値との突き合わせ (下限を下回る大きさのみ)\n";
    for (std::size_t r = 0; r < kRouteCount; ++r) {
        write_summary_line(std::cout, route_name(kRoutes[r]), totals[r].below_limit_, true);
    }

    for (std::size_t r = 0; r < kRouteCount; ++r) {
        std::cout << "\n条件別 (" << route_name(kRoutes[r]) << ")\n";
        for (const auto& entry : totals[r].by_condition_) {
            write_summary_line(std::cout, entry.first, entry.second, false);
        }
    }
    std::cout << "\nマーカー 1 辺別 (CPU)\n";
    for (const auto& entry : totals[0].by_side_px_) {
        write_summary_line(std::cout, std::to_string(entry.first) + " px", entry.second, false);
    }
    std::cout << "\n解像度別 (CPU)\n";
    for (const auto& entry : totals[0].by_resolution_) {
        write_summary_line(std::cout, entry.first, entry.second, false);
    }

    std::cout << "\nCPU 基準との差異\n";
    for (std::size_t r = 1; r < kRouteCount; ++r) {
        const aruco3cuda::report::Summary summary =
                aruco3cuda::report::summarize(totals[r].comparisons_);
        std::cout << "  " << route_name(kRoutes[r]) << " 一致 " << summary.agreed_image_count_
                  << " / " << summary.image_count_ << " 枚、四隅の最大差 "
                  << summary.worst_corner_error_px_ << " px";
        const char* keys[] = {"未検出", "過検出", "ID 不一致", "rotation 不一致", "四隅ずれ"};
        for (std::size_t k = 0; k < 5U; ++k) {
            if (summary.kind_counts_[k] != 0U) {
                std::cout << "、" << keys[k] << ' ' << summary.kind_counts_[k] << " 件";
            }
        }
        std::cout << '\n';
        // 差異のある場面を全て挙げる。件数だけでは、どの条件で起きたのかが
        // 分からず、原因を追えない。
        for (const aruco3cuda::report::ImageComparison& comparison : totals[r].comparisons_) {
            if (comparison.agrees()) {
                continue;
            }
            for (const aruco3cuda::report::Diff& diff : comparison.diffs_) {
                std::cout << "    " << comparison.image_path_ << ' '
                          << aruco3cuda::report::diff_kind_name(diff.kind_)
                          << " id=" << diff.baseline_id_ << " 位置 (" << diff.center_x_px_ << ", "
                          << diff.center_y_px_ << ") 差 " << diff.corner_error_px_ << " px\n";
            }
        }
    }

    std::cout << "\ndevice memory (CUDA 経路)\n";
    std::cout << "  workspace の最大使用量 " << memory.peak_used_bytes_ << " byte / 最大容量 "
              << memory.capacity_bytes_ << " byte\n";
    std::cout << "  検出 " << detect_call_count << " 回で増えた確保回数 "
              << memory.per_frame_allocation_count_ << " 回\n";

    if (!output_path.empty()) {
        std::ofstream output(output_path);
        if (!output.is_open()) {
            std::cerr << "出力先を開けない: " << output_path << '\n';
            return EXIT_FAILURE;
        }
        aruco3cuda::util::JsonWriter writer(output);
        writer.begin_object();
        writer.member_int("schema_version", 1);
        writer.member_string("preset", preset);
        writer.member_int("sceneCount", static_cast<long long>(specs.size()));
        writer.member_int("seed", static_cast<long long>(seed));
        writer.member_string("dictionary", dictionary_name);
        writer.member_double("matchRadiusRatio", match.match_radius_ratio_, 6);
        writer.member_double("cornerTolerancePx", compare_config.corner_tolerance_px_, 6);
        writer.key("routes");
        writer.begin_object();
        for (std::size_t r = 0; r < kRouteCount; ++r) {
            writer.key(route_name(kRoutes[r]));
            writer.begin_object();
            writer.key("truth");
            write_summary_json(writer, totals[r].truth_, false);
            writer.key("detectable");
            write_summary_json(writer, totals[r].detectable_, true);
            writer.key("belowLimit");
            write_summary_json(writer, totals[r].below_limit_, true);
            writer.key("byCondition");
            writer.begin_object();
            for (const auto& entry : totals[r].by_condition_) {
                writer.key(entry.first);
                write_summary_json(writer, entry.second, false);
            }
            writer.end_object();
            writer.key("byResolution");
            writer.begin_object();
            for (const auto& entry : totals[r].by_resolution_) {
                writer.key(entry.first);
                write_summary_json(writer, entry.second, false);
            }
            writer.end_object();
            writer.key("bySidePx");
            writer.begin_object();
            for (const auto& entry : totals[r].by_side_px_) {
                writer.key(std::to_string(entry.first));
                write_summary_json(writer, entry.second, false);
            }
            writer.end_object();
            if (r != 0U) {
                const aruco3cuda::report::Summary summary =
                        aruco3cuda::report::summarize(totals[r].comparisons_);
                writer.key("versusCpu");
                writer.begin_object();
                writer.member_int("imageCount", static_cast<long long>(summary.image_count_));
                writer.member_int("agreedImageCount",
                                  static_cast<long long>(summary.agreed_image_count_));
                writer.member_double("worstCornerErrorPx", summary.worst_corner_error_px_, 6);
                const char* keys[] = {"missed", "extra", "idMismatch", "rotationMismatch",
                                      "cornerShift"};
                for (std::size_t k = 0; k < 5U; ++k) {
                    writer.member_int(keys[k], static_cast<long long>(summary.kind_counts_[k]));
                }
                writer.end_object();
            }
            writer.end_object();
        }
        writer.end_object();
        writer.key("deviceMemory");
        writer.begin_object();
        writer.member_int("workspacePeakUsedBytes",
                          static_cast<long long>(memory.peak_used_bytes_));
        writer.member_int("workspaceCapacityBytes", static_cast<long long>(memory.capacity_bytes_));
        writer.member_int("detectCallCount", static_cast<long long>(detect_call_count));
        writer.member_int("allocationsDuringDetect",
                          static_cast<long long>(memory.per_frame_allocation_count_));
        writer.end_object();
        writer.end_object();
        output << '\n';
        if (!output) {
            std::cerr << "出力の書き込みに失敗した: " << output_path << '\n';
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
