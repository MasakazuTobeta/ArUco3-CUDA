// SPDX-License-Identifier: Apache-2.0
// Paired, host-only detection measurements. Loading, hashing, and validation are untimed.
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "aruco3cuda/util/json_writer.hpp"
#include "aruco3cuda/util/sha256.hpp"
#include "aruco3cuda/util/statistics.hpp"
#include "corpus_generator.hpp"
#include "cpu_detector.hpp"

namespace {
using Clock = std::chrono::steady_clock;
using aruco3cuda::cpu::CpuDetector;
using aruco3cuda::cpu::CpuOptions;
using aruco3cuda::cpu::CpuResult;
using aruco3cuda::util::JsonWriter;

struct Settings {
    std::string input_dir_;
    std::string filter_;
    std::string variants_ = "opencv,control,lazy,packed,cells,optimized";
    std::string dictionary_ = "DICT_ARUCO_MIP_36h12";
    int iterations_ = 200;
    int warmup_ = 30;
    int threads_ = 1;
    int throughput_frames_ = 0;
    bool aruco3_ = true;
    bool profile_ = false;
    bool generate_ = false;
};

int parse_integer(const std::string& value) {
    std::size_t consumed = 0;
    const int parsed = std::stoi(value, &consumed);
    if (consumed != value.size()) {
        throw std::invalid_argument("invalid integer: " + value);
    }
    return parsed;
}

Settings parse(int argc, char** argv) {
    Settings settings;
    for (int i = 1; i < argc; ++i) {
        const std::string name = argv[i];
        if (name == "--profile") {
            settings.profile_ = true;
            continue;
        }
        if (name == "--generate") {
            settings.generate_ = true;
            continue;
        }
        if (name == "--help") {
            std::cout << "usage: aruco3cuda_cpu_benchmark --input-dir DIR [--generate] "
                         "[--filter TEXT] [--iterations N] [--warmup N] [--threads N] "
                         "[--aruco3 0|1] [--dictionary NAME] [--profile] [--throughput-frames N] "
                         "[--variants "
                         "opencv,previous,optimized,control,lazy,packed,cells,threshold,spatial,"
                         "sparse,contours,cached,lookup,round2]\n";
            std::exit(EXIT_SUCCESS);
        }
        if (++i >= argc) {
            throw std::invalid_argument("missing argument: " + name);
        }
        const std::string value = argv[i];
        if (name == "--input-dir")
            settings.input_dir_ = value;
        else if (name == "--filter")
            settings.filter_ = value;
        else if (name == "--variants")
            settings.variants_ = value;
        else if (name == "--dictionary")
            settings.dictionary_ = value;
        else if (name == "--iterations")
            settings.iterations_ = parse_integer(value);
        else if (name == "--warmup")
            settings.warmup_ = parse_integer(value);
        else if (name == "--threads")
            settings.threads_ = parse_integer(value);
        else if (name == "--throughput-frames")
            settings.throughput_frames_ = parse_integer(value);
        else if (name == "--aruco3") {
            const int flag = parse_integer(value);
            if (flag < 0 || flag > 1)
                throw std::invalid_argument("--aruco3 must be 0 or 1");
            settings.aruco3_ = flag != 0;
        } else
            throw std::invalid_argument("unknown argument: " + name);
    }
    if (settings.input_dir_.empty() || settings.iterations_ < 1 || settings.warmup_ < 0 ||
        settings.threads_ < 1 || settings.throughput_frames_ < 0) {
        throw std::invalid_argument(
                "provide --input-dir, iterations > 0, warmup >= 0, threads > 0");
    }
    return settings;
}

void generate(const Settings& settings) {
    aruco3cuda::corpusgen::CorpusConfig config;
    config.output_dir_ = settings.input_dir_;
    config.dictionary_name_ = settings.dictionary_;
    std::vector<aruco3cuda::corpusgen::SceneSpec> specs;
    if (!aruco3cuda::corpusgen::build_preset("full", &specs)) {
        throw std::runtime_error("cannot construct full corpus");
    }
    // The historical timing sweep detects nothing in most high-resolution scenes.
    // Add larger markers, dense scenes, and textured scenes with detectable markers.
    const std::vector<cv::Size> sizes = {{640, 480}, {1280, 720}, {1920, 1080}, {3840, 2160}};
    for (const cv::Size& size : sizes) {
        for (const int count : {1, 4, 16}) {
            for (const double noise : {0.0, 20.0}) {
                aruco3cuda::corpusgen::SceneSpec spec;
                spec.width_px_ = size.width;
                spec.height_px_ = size.height;
                spec.marker_count_ = count;
                spec.marker_side_px_ = std::max(128.0, 64.0 + 0.065 * size.width);
                if (count == 16)
                    spec.marker_side_px_ = std::min(spec.marker_side_px_, size.height / 5.0);
                spec.noise_sigma_levels_ = noise;
                spec.name_ = "cpu_" + std::to_string(size.width) + "x" +
                             std::to_string(size.height) + "_n" + std::to_string(count) +
                             (noise == 0 ? "_clean" : "_noise");
                specs.push_back(spec);
            }
        }
    }
    std::vector<aruco3cuda::corpusgen::GeneratedScene> scenes;
    for (std::size_t i = 0; i < specs.size(); ++i) {
        aruco3cuda::corpusgen::GeneratedScene scene;
        std::string error;
        if (!aruco3cuda::corpusgen::generate_scene(config, specs[i], i, &scene, &error)) {
            throw std::runtime_error(error);
        }
        scenes.push_back(std::move(scene));
    }
    std::ofstream manifest(std::filesystem::path(settings.input_dir_) / "manifest.json");
    aruco3cuda::corpusgen::write_manifest_json(manifest, config, "cpu-extended", scenes);
    if (!manifest)
        throw std::runtime_error("cannot write corpus manifest");
}

CpuResult ordered(CpuResult result) {
    std::vector<std::size_t> order(result.ids_.size());
    std::iota(order.begin(), order.end(), 0U);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        if (result.ids_[a] != result.ids_[b])
            return result.ids_[a] < result.ids_[b];
        if (result.corners_[a][0].x != result.corners_[b][0].x)
            return result.corners_[a][0].x < result.corners_[b][0].x;
        return result.corners_[a][0].y < result.corners_[b][0].y;
    });
    CpuResult sorted;
    sorted.rejected_count_ = result.rejected_count_;
    for (std::size_t index : order) {
        sorted.ids_.push_back(result.ids_[index]);
        sorted.corners_.push_back(result.corners_[index]);
    }
    return sorted;
}

double compare(const CpuResult& expected_raw, const CpuResult& actual_raw) {
    const CpuResult expected = ordered(expected_raw);
    const CpuResult actual = ordered(actual_raw);
    if (actual.rejected_count_ != expected.rejected_count_)
        throw std::runtime_error("rejected counts differ");
    if (actual.ids_ != expected.ids_)
        throw std::runtime_error("detected IDs differ");
    double largest = 0.0;
    for (std::size_t i = 0; i < expected.ids_.size(); ++i) {
        if (actual.corners_[i].size() != 4)
            throw std::runtime_error("invalid corner count");
        for (std::size_t c = 0; c < 4; ++c) {
            const double distance = cv::norm(expected.corners_[i][c] - actual.corners_[i][c]);
            if (!std::isfinite(distance))
                throw std::runtime_error("non-finite corner");
            largest = std::max(largest, distance);
        }
    }
    if (largest != 0.0)
        throw std::runtime_error("corner discrepancy: " + std::to_string(largest));
    return largest;
}

struct Variant {
    std::string name_;
    CpuDetector detector_;
    CpuResult result_;
    std::vector<double> samples_;
    std::array<std::vector<double>, 7> stage_samples_;
    bool reference_ = false;
};

CpuOptions options_for(const std::string& name, bool profile) {
    CpuOptions options;
    options.reuse_pyramid_ = false;
    options.packed_dictionary_ = false;
    options.fused_cells_ = false;
    options.multiwindow_threshold_ = false;
    options.simple_contours_ = false;
    options.component_prefilter_ = false;
    options.dryrun_otsu_ = false;
    options.component_candidates_ = false;
    options.parallel_windows_ = false;
    options.parallel_decode_ = false;
    options.profile_stages_ = profile;
    const bool optimized = name == "optimized" || name == "optimized_serial" ||
                           name == "optimized_windows" || name == "optimized_legacy" ||
                           name == "components" || name == "previous" || name == "spatial" ||
                           name == "sparse" || name == "contours" || name == "round2" ||
                           name == "cached" || name == "lookup";
    if (name == "optimized_legacy")
        options.threshold_max_prefix_bytes_ = 0U;
    options.reuse_pyramid_ = name == "lazy" || optimized;
    options.packed_dictionary_ = name == "packed" || optimized;
    options.fused_cells_ = name == "cells" || name == "dryrun" || optimized;
    options.multiwindow_threshold_ = name == "threshold" || optimized;
    options.simple_contours_ = name == "simple" || name == "combined";
    options.component_prefilter_ = name == "prefilter" || name == "combined";
    options.dryrun_otsu_ = name == "dryrun";
    options.component_candidates_ = name == "components";
    options.parallel_windows_ = name == "parallel" || name == "optimized" ||
                                name == "optimized_windows" || name == "optimized_legacy" ||
                                name == "previous" || name == "spatial" || name == "sparse" ||
                                name == "contours" || name == "round2" || name == "cached" ||
                                name == "lookup";
    options.parallel_decode_ = name == "decode" || name == "optimized" || name == "components" ||
                               name == "optimized_legacy" || name == "previous" ||
                               name == "spatial" || name == "sparse" || name == "contours" ||
                               name == "round2" || name == "cached" || name == "lookup";
    const bool current = name == "optimized" || name == "optimized_serial" ||
                         name == "optimized_windows" || name == "optimized_legacy";
    options.exact_dictionary_lookup_ = current || name == "lookup" || name == "round2";
    options.spatial_grouping_ = name == "spatial" || name == "round2";
    options.sparse_pyramid_ = current || name == "sparse" || name == "round2";
    options.filtered_contours_ = current || name == "contours" || name == "round2";
    options.cached_refinement_ = current || name == "cached" || name == "round2";
    const std::vector<std::string> names = {"control",
                                            "lazy",
                                            "packed",
                                            "cells",
                                            "threshold",
                                            "simple",
                                            "prefilter",
                                            "combined",
                                            "dryrun",
                                            "parallel",
                                            "decode",
                                            "optimized",
                                            "optimized_serial",
                                            "optimized_windows",
                                            "optimized_legacy",
                                            "components",
                                            "previous",
                                            "spatial",
                                            "sparse",
                                            "contours",
                                            "round2",
                                            "cached",
                                            "lookup"};
    if (std::find(names.begin(), names.end(), name) == names.end()) {
        throw std::invalid_argument("unknown variant: " + name);
    }
    return options;
}

void run(const Settings& settings) {
    cv::setNumThreads(settings.threads_);
    aruco3cuda::DetectorConfig config;
    config.use_aruco3_detection_ = settings.aruco3_;
    config.corner_refine_method_ = settings.aruco3_ ? aruco3cuda::CornerRefineMethod::kSubpix
                                                    : aruco3cuda::CornerRefineMethod::kNone;
    std::string message;
    // Names are resolved through the installed OpenCV dictionary registry.
    const std::vector<std::pair<std::string, cv::aruco::PredefinedDictionaryType>> dictionaries = {
            {"DICT_ARUCO_MIP_36h12", cv::aruco::DICT_ARUCO_MIP_36h12},
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
            {"DICT_ARUCO_ORIGINAL", cv::aruco::DICT_ARUCO_ORIGINAL},
            {"DICT_7X7_1000", cv::aruco::DICT_7X7_1000}};
    auto dictionary =
            std::find_if(dictionaries.begin(), dictionaries.end(),
                         [&](const auto& entry) { return entry.first == settings.dictionary_; });
    if (dictionary == dictionaries.end())
        throw std::invalid_argument("benchmark dictionary unsupported");
    cv::aruco::DetectorParameters parameters;
    parameters.useAruco3Detection = settings.aruco3_;
    parameters.minSideLengthCanonicalImg = config.min_side_length_canonical_img_px_;
    parameters.minMarkerLengthRatioOriginalImg = config.min_marker_length_ratio_original_img_;
    parameters.cornerRefinementMethod =
            settings.aruco3_ ? cv::aruco::CORNER_REFINE_SUBPIX : cv::aruco::CORNER_REFINE_NONE;
    cv::aruco::ArucoDetector reference(cv::aruco::getPredefinedDictionary(dictionary->second),
                                       parameters);
    std::vector<Variant> variants;
    std::size_t position = 0;
    while (position <= settings.variants_.size()) {
        const auto end = settings.variants_.find(',', position);
        Variant variant;
        variant.name_ = settings.variants_.substr(position, end - position);
        variant.reference_ = variant.name_ == "opencv";
        if (!variant.reference_ &&
            variant.detector_.initialize(settings.dictionary_, config, &message,
                                         options_for(variant.name_, settings.profile_)) !=
                    aruco3cuda::Status::kOk)
            throw std::runtime_error(message);
        variants.push_back(std::move(variant));
        if (end == std::string::npos)
            break;
        position = end + 1;
    }
    JsonWriter environment(std::cout, 0);
    environment.begin_object();
    environment.member_string("type", "environment");
    environment.member_string("opencv_version", CV_VERSION);
    environment.member_string("compiler", __VERSION__);
#ifdef ARUCO3CPU_BUILD_FLAGS
    environment.member_string("cpu_compile_flags", ARUCO3CPU_BUILD_FLAGS);
#endif
    environment.member_int("opencv_threads", cv::getNumThreads());
    environment.member_int("iterations", settings.iterations_);
    environment.member_int("warmup", settings.warmup_);
    environment.member_int("throughput_frames", settings.throughput_frames_);
    environment.member_int("default_threshold_max_prefix_bytes",
                           CpuOptions{}.threshold_max_prefix_bytes_);
    environment.member_bool("aruco3", settings.aruco3_);
    environment.member_bool("profile_stages", settings.profile_);
    environment.member_string("dictionary", settings.dictionary_);
    environment.key("cpu_options");
    environment.begin_array();
    for (const Variant& variant : variants) {
        if (variant.reference_)
            continue;
        const CpuOptions options = options_for(variant.name_, settings.profile_);
        environment.begin_object();
        environment.member_string("variant", variant.name_);
        environment.member_bool("reuse_pyramid", options.reuse_pyramid_);
        environment.member_bool("packed_dictionary", options.packed_dictionary_);
        environment.member_bool("fused_cells", options.fused_cells_);
        environment.member_bool("multiwindow_threshold", options.multiwindow_threshold_);
        environment.member_bool("parallel_windows", options.parallel_windows_);
        environment.member_bool("parallel_decode", options.parallel_decode_);
        environment.member_bool("exact_dictionary_lookup", options.exact_dictionary_lookup_);
        environment.member_bool("filtered_contours", options.filtered_contours_);
        environment.member_bool("sparse_pyramid", options.sparse_pyramid_);
        environment.member_bool("cached_refinement", options.cached_refinement_);
        environment.member_bool("spatial_grouping", options.spatial_grouping_);
        environment.member_int("sparse_tile_side_px", options.sparse_tile_side_px_);
        environment.member_double("sparse_full_level_ratio", options.sparse_full_level_ratio_, 9);
        environment.member_double("sparse_max_work_ratio", options.sparse_max_work_ratio_, 9);
        environment.member_int("threshold_max_prefix_bytes", options.threshold_max_prefix_bytes_);
        environment.end_object();
    }
    environment.end_array();
    environment.member_int("min_side_length_canonical_img_px",
                           config.min_side_length_canonical_img_px_);
    environment.member_double("min_marker_length_ratio_original_img",
                              config.min_marker_length_ratio_original_img_, 9);
    environment.member_int("threshold_window_min_px", config.adaptive_thresh_win_size_min_px_);
    environment.member_int("threshold_window_max_px", config.adaptive_thresh_win_size_max_px_);
    environment.member_int("threshold_window_step_px", config.adaptive_thresh_win_size_step_px_);
    environment.member_double("error_correction_rate", config.error_correction_rate_, 9);
    environment.member_string("measurement",
                              "host grayscale input to host results; alternating route order");
    environment.end_object();
    std::cout << '\n';
    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::directory_iterator(settings.input_dir_)) {
        const std::string extension = entry.path().extension().string();
        if (entry.is_regular_file() &&
            (extension == ".png" || extension == ".jpg" || extension == ".jpeg") &&
            entry.path().filename().string().find(settings.filter_) != std::string::npos)
            files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    if (files.empty())
        throw std::runtime_error("no matching PNG/JPEG images");
    for (const auto& path : files) {
        const cv::Mat image = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
        if (image.empty())
            throw std::runtime_error("cannot read " + path.string());
        std::string digest;
        if (!aruco3cuda::util::sha256_file(path.string(), &digest))
            throw std::runtime_error(message);
        CpuResult expected;
        std::vector<std::vector<cv::Point2f>> rejected;
        reference.detectMarkers(image, expected.corners_, expected.ids_, rejected);
        expected.rejected_count_ = rejected.size();
        const auto detect = [&](Variant& variant) {
            if (variant.reference_) {
                // Rejected corner coordinates are absent from CpuResult. Request them only
                // in the untimed validation call, so the baseline has the same outputs.
                reference.detectMarkers(image, variant.result_.corners_, variant.result_.ids_);
                variant.result_.rejected_count_ = expected.rejected_count_;
            } else if (variant.detector_.detect(image, &variant.result_, &message) !=
                       aruco3cuda::Status::kOk) {
                throw std::runtime_error(variant.name_ + ": " + message);
            }
        };
        for (Variant& variant : variants) {
            variant.samples_.clear();
            for (auto& samples : variant.stage_samples_)
                samples.clear();
            detect(variant);
            try {
                if (variant.name_ != "components")
                    (void)compare(expected, variant.result_);
            } catch (const std::exception& e) {
                throw std::runtime_error(path.string() + " " + variant.name_ + ": " + e.what());
            }
            for (int i = 0; i < settings.warmup_; ++i)
                detect(variant);
        }
        for (int i = 0; i < settings.iterations_; ++i) {
            for (std::size_t j = 0; j < variants.size(); ++j) {
                const std::size_t index = (static_cast<std::size_t>(i) + j) % variants.size();
                Variant& variant = variants[index];
                const auto start = Clock::now();
                detect(variant);
                const double elapsed_ms =
                        std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                variant.samples_.push_back(elapsed_ms);
                const auto& stage = variant.result_.stage_times_;
                const std::array<double, 7> times = {stage.preprocess_ms_, stage.threshold_ms_,
                                                     stage.candidates_ms_, stage.decode_ms_,
                                                     stage.refine_ms_,     stage.trace_ms_,
                                                     stage.group_ms_};
                for (std::size_t s = 0; s < times.size(); ++s) {
                    variant.stage_samples_[s].push_back(times[s]);
                }
            }
        }
        for (Variant& variant : variants) {
            const auto throughput_start = Clock::now();
            for (int frame = 0; frame < settings.throughput_frames_; ++frame)
                detect(variant);
            const double throughput_seconds =
                    std::chrono::duration<double>(Clock::now() - throughput_start).count();
            aruco3cuda::util::SampleStatistics statistics;
            (void)aruco3cuda::util::compute_statistics(variant.samples_, &statistics);
            JsonWriter row(std::cout, 0);
            row.begin_object();
            row.member_string("type", "measurement");
            row.member_string("scene", path.filename().string());
            row.member_string("sha256", digest);
            row.member_string("variant", variant.name_);
            row.member_int("width", image.cols);
            row.member_int("height", image.rows);
            const float scale =
                    settings.aruco3_
                            ? static_cast<float>(config.min_side_length_canonical_img_px_) /
                                      (static_cast<float>(
                                               config.min_side_length_canonical_img_px_) +
                                       static_cast<float>(std::max(image.cols, image.rows)) *
                                               config.min_marker_length_ratio_original_img_)
                            : 1.0F;
            row.member_double("fxfy_effective", scale, 9);
            row.member_int("segmentation_width", cvRound(scale * image.cols));
            row.member_int("segmentation_height", cvRound(scale * image.rows));
            row.member_int("detections", static_cast<int>(variant.result_.ids_.size()));
            row.member_int("rejected", static_cast<int>(variant.result_.rejected_count_));
            row.member_int("reference_rejected", static_cast<int>(expected.rejected_count_));
            double corner_error = std::numeric_limits<double>::quiet_NaN();
            std::string discrepancy;
            try {
                corner_error = compare(expected, variant.result_);
            } catch (const std::exception& error) {
                discrepancy = error.what();
                if (variant.name_ != "components")
                    throw;
            }
            row.member_string("compatibility_discrepancy", discrepancy);
            row.member_double("max_corner_error_px", corner_error, 9);
            row.member_int("reference_detections", static_cast<int>(expected.ids_.size()));
            row.member_double("p50_ms", statistics.p50_, 9);
            row.member_double("p95_ms", statistics.p95_, 9);
            row.member_double("p99_ms", statistics.p99_, 9);
            row.member_double("throughput_fps",
                              settings.throughput_frames_ > 0
                                      ? settings.throughput_frames_ / throughput_seconds
                                      : std::numeric_limits<double>::quiet_NaN(),
                              6);
            const std::array<const char*, 7> stage_names = {
                    "preprocess_ms", "threshold_ms", "candidates_ms", "decode_ms",
                    "refine_ms",     "trace_ms",     "group_ms"};
            for (std::size_t s = 0; s < stage_names.size(); ++s) {
                aruco3cuda::util::SampleStatistics stage_statistics;
                (void)aruco3cuda::util::compute_statistics(variant.stage_samples_[s],
                                                           &stage_statistics);
                row.member_double(stage_names[s], stage_statistics.p50_, 9);
            }
            row.end_object();
            std::cout << '\n';
        }
        std::cout.flush();
    }
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const Settings settings = parse(argc, argv);
        if (settings.generate_)
            generate(settings);
        run(settings);
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
