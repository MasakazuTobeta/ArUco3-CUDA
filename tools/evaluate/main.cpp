// SPDX-License-Identifier: Apache-2.0
//
// CLI of the accuracy evaluation tool.
//
// Purpose:
//   Runs the three routes (CPU baseline, hybrid, CUDA) against the ground truth of
//   the synthetic corpus and reports precision, recall, the ID and rotation agreement
//   rates, and the corner RMSE, broken down by condition. The difference report
//   (tools/report) takes the CPU baseline as its reference, so a marker the baseline
//   itself missed never appears as a difference. This tool provides a separate path
//   that matches against the true values.
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

/// Route under evaluation.
enum class Route {
    kCpu,
    kHybrid,
    kCuda,
};

/// Display name of a route. Used in the report and as a JSON key.
const char* route_name(Route route) {
    switch (route) {
        case Route::kCpu:
            return "CPU";
        case Route::kHybrid:
            return "Hybrid";
        case Route::kCuda:
            return "CUDA";
    }
    return "unknown";
}

constexpr std::size_t kRouteCount = 3U;
const Route kRoutes[kRouteCount] = {Route::kCpu, Route::kHybrid, Route::kCuda};

/// Totals for one route.
struct RouteTotals {
    /// Matching against the true values.
    AccuracySummary truth_;
    /// Matching against the true values per condition. The key is the leading word of
    /// the scene name.
    std::map<std::string, AccuracySummary> by_condition_;
    /// Per resolution. The key looks like "1280x720".
    std::map<std::string, AccuracySummary> by_resolution_;
    /// Per marker side length. The key is the value in pixels.
    std::map<int, AccuracySummary> by_side_px_;
    /// Totals over only the true values at or above the ArUco3 lower bound. Read
    /// recall from here.
    AccuracySummary detectable_;
    /// Totals over only the true values below the lower bound. Used to confirm that
    /// they go undetected, as the strategy dictates.
    AccuracySummary below_limit_;
    /// Per condition, over only the true values at or above the lower bound. Read the
    /// per-condition recall from here.
    std::map<std::string, AccuracySummary> detectable_by_condition_;
    /// Differences against the CPU baseline. Unused for the CPU route itself.
    std::vector<aruco3cuda::report::ImageComparison> comparisons_;
};

/// Device memory usage collected during the run.
struct MemoryUsage {
    std::size_t peak_used_bytes_ = 0;
    std::size_t capacity_bytes_ = 0;
    std::size_t allocation_count_ = 0;
    std::size_t reallocation_count_ = 0;
    /// Allocations added per detection call. Reaches 0 in the steady state.
    std::size_t per_frame_allocation_count_ = 0;
};

void print_usage(std::ostream& out) {
    out << "usage: aruco3cuda_evaluate [option]...\n"
        << "\n"
        << "  --preset <name>                corpus preset; default full\n"
        << "  --seed <u64>                   random seed of the corpus; default 20260827\n"
        << "  --corpus-dir <path>            where the corpus is generated; default "
           "/tmp/aruco3eval\n"
        << "  --dictionary <name>            default DICT_ARUCO_MIP_36h12\n"
        << "  --output <path>                destination of the JSON report; none by default\n"
        << "  --match-radius-ratio <f>       matching radius as a side-length ratio; "
           "default 0.5\n"
        << "  --corner-tolerance-px <f>      tolerance when comparing against the CPU "
           "baseline; default 1.0\n"
        << "  --use-aruco3 <0|1>             ArUco3 detection strategy; default 1\n"
        << "  --help                         print this help and exit\n";
}

bool take_value(int argc, char** argv, int* index, const char* option, std::string* out) {
    if (*index + 1 >= argc) {
        std::cerr << "missing argument for: " << option << '\n';
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

/// The leading word of a scene name. Used as the key of the per-condition totals.
///
/// A scene name has the form "clean_1280x720_n4_s128" or "blur_1920x1080", where the
/// leading word is the condition itself.
std::string condition_of(const std::string& scene_name) {
    const std::size_t separator = scene_name.find('_');
    if (separator == std::string::npos) {
        return scene_name;
    }
    return scene_name.substr(0, separator);
}

/// Converts the true values into the form the evaluation uses.
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

/// Converts a float array of four corners into the double corner form.
aruco3cuda::report::Quad to_quad(const float* corners) {
    aruco3cuda::report::Quad quad{};
    for (std::size_t c = 0; c < 8U; ++c) {
        quad[c] = static_cast<double>(corners[c]);
    }
    return quad;
}

/// Accumulates the result for one image into the totals of a route.
///
/// @param detectable_limit_px Lower bound on the side length that ArUco3 can detect
///        in principle.
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
    aruco3cuda::evaluate::accumulate_selected(
            image, detectable, &out_totals->detectable_by_condition_[condition_of(scene.name_)]);
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

/// Converts detections into the form the difference report uses.
std::vector<aruco3cuda::report::Detection> to_detections(const std::vector<Observation>& observed) {
    std::vector<aruco3cuda::report::Detection> result;
    result.reserve(observed.size());
    for (const Observation& entry : observed) {
        result.push_back({entry.id_, entry.corners_});
    }
    return result;
}

/// Formats a ratio as a percentage string. Returns "-" when it is undefined.
std::string percent_or_dash(bool defined, double value) {
    if (!defined) {
        return "-";
    }
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.2f%%", value * 100.0);
    return std::string(buffer);
}

/// Writes one aggregate as a single line.
///
/// @param recall_only true when the aggregate covers only a subset of the true
///        values. It holds no unmatched detections, so precision and the detection
///        count are not displayed. Displaying them would invite reading a value that
///        is an artifact of the counting, such as "precision 100% with 0 detections".
void write_summary_line(std::ostream& out, const std::string& label, const AccuracySummary& summary,
                        bool recall_only) {
    double precision_value = 0.0;
    double recall_value = 0.0;
    double rmse_value = 0.0;
    const bool has_precision =
            !recall_only && aruco3cuda::evaluate::precision(summary, &precision_value);
    const bool has_recall = aruco3cuda::evaluate::recall(summary, &recall_value);
    const bool has_rmse = aruco3cuda::evaluate::corner_rmse_px(summary, &rmse_value);
    out << "  " << std::left << std::setw(18) << label << std::right << " truth " << std::setw(5)
        << summary.truth_count_;
    if (recall_only) {
        out << "                 ";
    } else {
        out << " detections " << std::setw(5) << summary.observed_count_;
    }
    out << " TP " << std::setw(5) << summary.true_positive_;
    if (recall_only) {
        out << "         ";
    } else {
        out << " FP " << std::setw(4) << summary.false_positive_;
    }
    out << " FN " << std::setw(4) << summary.false_negative_ << "  precision " << std::setw(8)
        << percent_or_dash(has_precision, precision_value) << "  recall " << std::setw(8)
        << percent_or_dash(has_recall, recall_value) << "  rotation match " << std::setw(8)
        << percent_or_dash(summary.true_positive_ != 0U,
                           summary.true_positive_ == 0U
                                   ? 0.0
                                   : static_cast<double>(summary.rotation_agreed_) /
                                             static_cast<double>(summary.true_positive_));
    if (has_rmse) {
        out << "  corner RMSE " << std::fixed << std::setprecision(4) << rmse_value << " px max "
            << summary.corner_max_px_ << " px" << std::defaultfloat;
    }
    out << '\n';
}

/// Writes one aggregate as JSON.
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
                std::cerr << "--seed must be an unsigned integer\n";
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
                std::cerr << "--match-radius-ratio must be a positive real number\n";
                return EXIT_FAILURE;
            }
            continue;
        }
        if (argument == "--corner-tolerance-px") {
            if (!take_value(argc, argv, &i, "--corner-tolerance-px", &value) ||
                !parse_double(value, &compare_config.corner_tolerance_px_) ||
                compare_config.corner_tolerance_px_ < 0.0) {
                std::cerr << "--corner-tolerance-px must be a real number >= 0\n";
                return EXIT_FAILURE;
            }
            continue;
        }
        if (argument == "--use-aruco3") {
            if (!take_value(argc, argv, &i, "--use-aruco3", &value) ||
                !parse_bool_flag(value, &use_aruco3)) {
                std::cerr << "--use-aruco3 must be 0 or 1\n";
                return EXIT_FAILURE;
            }
            continue;
        }
        std::cerr << "unknown option: " << argument << '\n';
        print_usage(std::cerr);
        return EXIT_FAILURE;
    }
    compare_config.match_radius_ratio_ = match.match_radius_ratio_;

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        std::cerr << "no CUDA device found\n";
        return EXIT_FAILURE;
    }

    std::vector<aruco3cuda::corpusgen::SceneSpec> specs;
    if (!aruco3cuda::corpusgen::build_preset(preset, &specs)) {
        std::cerr << "unknown preset: " << preset << '\n';
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
        std::cerr << "unsupported Dictionary: " << dictionary_name << '\n';
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
            std::cerr << "cannot generate the scene: " << error << '\n';
            return EXIT_FAILURE;
        }

        const cv::Mat image = cv::imread(scene.path_, cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            std::cerr << "cannot read image: " << scene.path_ << '\n';
            return EXIT_FAILURE;
        }

        std::vector<Observation> observed[kRouteCount];

        aruco3cuda::reference::ReferenceResult reference_result;
        if (!aruco3cuda::reference::detect_image(scene.path_, reference_config, &reference_result,
                                                 &error)) {
            std::cerr << "CPU baseline detection failed: " << error << '\n';
            return EXIT_FAILURE;
        }
        for (const auto& detection : reference_result.detections_) {
            observed[0].push_back({detection.id_, detection.corners_});
        }

        std::string message;
        aruco3cuda::hybrid::DeviceImage device;
        if (device.reserve(aruco3cuda::MemorySpace::kDevice, image.cols, image.rows, &message) !=
            aruco3cuda::Status::kOk) {
            std::cerr << "cannot allocate the device buffer: " << message << '\n';
            return EXIT_FAILURE;
        }
        if (device.upload(image.data, image.cols, image.rows, static_cast<std::size_t>(image.step),
                          &message) != aruco3cuda::Status::kOk) {
            std::cerr << "transfer to the device failed: " << message << '\n';
            return EXIT_FAILURE;
        }

        aruco3cuda::hybrid::HybridDetector hybrid;
        if (hybrid.initialize(detector_config, dictionary_name, image.cols, image.rows, &message) !=
            aruco3cuda::Status::kOk) {
            std::cerr << "initialization of the hybrid route failed: " << message << '\n';
            return EXIT_FAILURE;
        }
        aruco3cuda::hybrid::HybridResult hybrid_result;
        if (hybrid.detect(device.view(), &hybrid_result, &message) != aruco3cuda::Status::kOk) {
            std::cerr << "detection on the hybrid route failed: " << message << '\n';
            return EXIT_FAILURE;
        }
        for (const auto& detection : hybrid_result.detections_) {
            observed[1].push_back({detection.id_, detection.corners_});
        }

        // Size the workspace allocation to the image. Left at the default upper
        // bound, even a small image would be measured while holding a 4K-sized region.
        aruco3cuda::DetectorConfig cuda_config = detector_config;
        cuda_config.max_width_px_ = image.cols;
        cuda_config.max_height_px_ = image.rows;
        aruco3cuda::Detector cuda;
        if (cuda.initialize(*table, cuda_config, &message) != aruco3cuda::Status::kOk) {
            std::cerr << "initialization of the CUDA route failed: " << message << '\n';
            return EXIT_FAILURE;
        }
        cudaStream_t stream = nullptr;
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
            std::cerr << "cannot create a stream\n";
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
            std::cerr << "detection on the CUDA route failed: " << message << '\n';
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
        // Subtract what initialization allocated, so only the allocations the
        // detection itself added are counted.
        memory.per_frame_allocation_count_ += statistics.allocation_count_ - before;

        // ArUco3 does not detect a marker whose side after downscaling falls below
        // min_side_length_canonical_img. Computing recall while still including true
        // values below that bound mixes the implementation's misses with the
        // strategy's inherent lower bound.
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

    std::cout << "=== Accuracy evaluation ===\n";
    std::cout << "preset " << preset << " / scenes " << specs.size() << " / seed " << seed << '\n';
    std::cout << "\nMatched against the true values (all)\n";
    for (std::size_t r = 0; r < kRouteCount; ++r) {
        write_summary_line(std::cout, route_name(kRoutes[r]), totals[r].truth_, false);
    }
    std::cout << "\nMatched against the true values (only sizes at or above the ArUco3 "
                 "lower bound; precision is not defined for this group)\n";
    for (std::size_t r = 0; r < kRouteCount; ++r) {
        write_summary_line(std::cout, route_name(kRoutes[r]), totals[r].detectable_, true);
    }
    std::cout << "\nMatched against the true values (only sizes below the lower bound)\n";
    for (std::size_t r = 0; r < kRouteCount; ++r) {
        write_summary_line(std::cout, route_name(kRoutes[r]), totals[r].below_limit_, true);
    }

    for (std::size_t r = 0; r < kRouteCount; ++r) {
        std::cout << "\nBy condition (" << route_name(kRoutes[r]) << ")\n";
        for (const auto& entry : totals[r].by_condition_) {
            write_summary_line(std::cout, entry.first, entry.second, false);
        }
    }
    for (std::size_t r = 0; r < kRouteCount; ++r) {
        std::cout << "\nBy condition, at or above the lower bound only (" << route_name(kRoutes[r])
                  << ")\n";
        for (const auto& entry : totals[r].detectable_by_condition_) {
            write_summary_line(std::cout, entry.first, entry.second, true);
        }
    }
    std::cout << "\nBy marker side length (CPU)\n";
    for (const auto& entry : totals[0].by_side_px_) {
        write_summary_line(std::cout, std::to_string(entry.first) + " px", entry.second, false);
    }
    std::cout << "\nBy resolution (CPU)\n";
    for (const auto& entry : totals[0].by_resolution_) {
        write_summary_line(std::cout, entry.first, entry.second, false);
    }

    std::cout << "\nDifferences against the CPU baseline\n";
    for (std::size_t r = 1; r < kRouteCount; ++r) {
        const aruco3cuda::report::Summary summary =
                aruco3cuda::report::summarize(totals[r].comparisons_);
        std::cout << "  " << route_name(kRoutes[r]) << " agreed " << summary.agreed_image_count_
                  << " / " << summary.image_count_ << " images, worst corner difference "
                  << summary.worst_corner_error_px_ << " px";
        const char* keys[] = {"missed", "extra", "id mismatch", "rotation mismatch",
                              "corner shift"};
        for (std::size_t k = 0; k < 5U; ++k) {
            if (summary.kind_counts_[k] != 0U) {
                std::cout << ", " << keys[k] << ' ' << summary.kind_counts_[k];
            }
        }
        std::cout << '\n';
        // List every scene with a difference. A count alone does not say under which
        // condition it happened, which leaves the cause untraceable.
        for (const aruco3cuda::report::ImageComparison& comparison : totals[r].comparisons_) {
            if (comparison.agrees()) {
                continue;
            }
            for (const aruco3cuda::report::Diff& diff : comparison.diffs_) {
                std::cout << "    " << comparison.image_path_ << ' '
                          << aruco3cuda::report::diff_kind_name(diff.kind_)
                          << " id=" << diff.baseline_id_ << " at (" << diff.center_x_px_ << ", "
                          << diff.center_y_px_ << ") difference " << diff.corner_error_px_
                          << " px\n";
            }
        }
    }

    std::cout << "\ndevice memory (CUDA route)\n";
    std::cout << "  workspace peak usage " << memory.peak_used_bytes_ << " bytes / peak capacity "
              << memory.capacity_bytes_ << " bytes\n";
    std::cout << "  allocations added over " << detect_call_count
              << " detections: " << memory.per_frame_allocation_count_ << "\n";

    if (!output_path.empty()) {
        std::ofstream output(output_path);
        if (!output.is_open()) {
            std::cerr << "cannot open the output destination: " << output_path << '\n';
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
            writer.key("detectableByCondition");
            writer.begin_object();
            for (const auto& entry : totals[r].detectable_by_condition_) {
                writer.key(entry.first);
                write_summary_json(writer, entry.second, true);
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
            std::cerr << "writing the output failed: " << output_path << '\n';
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
