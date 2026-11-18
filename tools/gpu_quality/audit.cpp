// SPDX-License-Identifier: Apache-2.0
// Correctness-only audit. All observations are retained; no timing claim is made.
#include <cuda_runtime_api.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "accuracy.hpp"
#include "aruco3cuda/config.hpp"
#include "aruco3cuda/detector.hpp"
#include "aruco3cuda/dictionary.hpp"
#include "aruco3cuda/util/json_writer.hpp"
#include "aruco3cuda/util/sha256.hpp"
#include "cpu_detector.hpp"
#include "device_image.hpp"
#include "quality_scenes.hpp"
#include "report_diff.hpp"

namespace {

using Observation = aruco3cuda::evaluate::Observation;
using TruthMarker = aruco3cuda::evaluate::TruthMarker;
using JsonWriter = aruco3cuda::util::JsonWriter;
using FrameSpec = aruco3cuda::cpu_quality::FrameSpec;

struct Frame {
    std::string name_;
    std::string family_;
    cv::Mat pixels_;
    std::vector<TruthMarker> truth_;
    bool annotated_ = true;
    bool positive_control_ = false;
};

void require(aruco3cuda::Status status, const std::string& message) {
    if (status != aruco3cuda::Status::kOk) {
        throw std::runtime_error(std::string(aruco3cuda::to_string(status)) + ": " + message);
    }
}

cv::aruco::PredefinedDictionaryType dictionary_type(const std::string& name) {
    if (name == "DICT_ARUCO_MIP_36h12") {
        return cv::aruco::DICT_ARUCO_MIP_36h12;
    }
    const std::array<int, 4> counts{50, 100, 250, 1000};
    for (int side = 4; side <= 7; ++side) {
        for (std::size_t i = 0; i < counts.size(); ++i) {
            if (name == "DICT_" + std::to_string(side) + "X" + std::to_string(side) + "_" +
                                std::to_string(counts[i])) {
                return static_cast<cv::aruco::PredefinedDictionaryType>((side - 4) * 4 +
                                                                        static_cast<int>(i));
            }
        }
    }
    throw std::invalid_argument("unsupported built-in dictionary: " + name);
}

std::vector<TruthMarker> make_truth(const FrameSpec& spec) {
    std::vector<TruthMarker> result;
    for (const auto& marker : spec.markers_) {
        TruthMarker truth;
        truth.id_ = marker.id_;
        truth.fully_inside_ = true;
        for (std::size_t i = 0; i < 4U; ++i) {
            truth.corners_[i * 2U] = marker.corners_[i].x;
            truth.corners_[i * 2U + 1U] = marker.corners_[i].y;
            truth.fully_inside_ = truth.fully_inside_ && marker.corners_[i].x >= -0.5F &&
                                  marker.corners_[i].y >= -0.5F &&
                                  marker.corners_[i].x <= spec.size_.width - 0.5F &&
                                  marker.corners_[i].y <= spec.size_.height - 0.5F;
        }
        truth.side_px_ = aruco3cuda::report::quad_average_side(truth.corners_);
        // The renderer's occluder width fraction is not a measured area fraction.
        result.push_back(truth);
    }
    return result;
}

std::vector<Observation> from_host(const std::vector<int>& ids,
                                   const std::vector<std::vector<cv::Point2f>>& corners) {
    std::vector<Observation> result;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        Observation observation;
        observation.id_ = ids[i];
        for (std::size_t c = 0; c < 4U; ++c) {
            observation.corners_[c * 2U] = corners.at(i).at(c).x;
            observation.corners_[c * 2U + 1U] = corners.at(i).at(c).y;
        }
        result.push_back(observation);
    }
    return result;
}

std::vector<aruco3cuda::report::Detection> as_detections(
        const std::vector<Observation>& observations) {
    std::vector<aruco3cuda::report::Detection> result;
    for (const Observation& observation : observations) {
        result.push_back({observation.id_, observation.corners_});
    }
    return result;
}

void write_observations(JsonWriter& writer, const std::vector<Observation>& observations) {
    writer.begin_array();
    for (const Observation& observation : observations) {
        writer.begin_object();
        writer.member_int("id", observation.id_);
        writer.key("corners");
        writer.begin_array();
        for (double coordinate : observation.corners_) {
            if (!std::isfinite(coordinate)) {
                throw std::runtime_error("nonfinite output corner");
            }
            writer.value_double(coordinate, 17);
        }
        writer.end_array();
        writer.end_object();
    }
    writer.end_array();
}

void write_metrics(JsonWriter& writer, const aruco3cuda::evaluate::ImageAccuracy& accuracy) {
    writer.begin_object();
    writer.member_int("tp", static_cast<long long>(accuracy.true_positive_));
    writer.member_int("fp", static_cast<long long>(accuracy.false_positive_));
    writer.member_int("fn", static_cast<long long>(accuracy.false_negative_));
    writer.member_int("rotation_agreed", static_cast<long long>(accuracy.rotation_agreed_));
    writer.member_double("corner_squared_sum_px2", accuracy.corner_squared_sum_px2_, 17);
    writer.member_int("corner_samples", static_cast<long long>(accuracy.corner_sample_count_));
    writer.member_double("corner_max_px", accuracy.corner_max_px_, 17);
    writer.key("per_truth");
    writer.begin_array();
    for (std::size_t i = 0; i < accuracy.truth_outcomes_.size(); ++i) {
        writer.begin_object();
        writer.member_int("outcome", static_cast<int>(accuracy.truth_outcomes_[i]));
        writer.member_bool("rotation_agreed", accuracy.truth_rotation_agreed_[i]);
        writer.member_double("corner_squared_sum_px2", accuracy.truth_squared_error_px2_[i], 17);
        writer.member_double("corner_max_px", accuracy.truth_corner_max_px_[i], 17);
        writer.end_object();
    }
    writer.end_array();
    writer.end_object();
}

class Stream {
public:
    explicit Stream(bool direct) {
        if (!direct &&
            cudaStreamCreateWithFlags(&this->stream_, cudaStreamNonBlocking) != cudaSuccess) {
            throw std::runtime_error("cannot create CUDA stream");
        }
    }
    ~Stream() {
        if (this->stream_ != nullptr) {
            static_cast<void>(cudaStreamDestroy(this->stream_));
        }
    }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    cudaStream_t get() const { return this->stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        bool aruco3 = true;
        bool direct = false;
        int width = 641;
        int height = 479;
        aruco3cuda::DetectorConfig config;
        std::string dictionary_name = "DICT_ARUCO_MIP_36h12";
        std::filesystem::path image_directory;
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--direct") {
                direct = true;
            } else if (i + 1 < argc && argument == "--dictionary") {
                dictionary_name = argv[++i];
            } else if (i + 1 < argc && argument == "--images") {
                image_directory = argv[++i];
            } else if (i + 1 < argc && argument == "--aruco3") {
                const std::string value = argv[++i];
                if (value != "0" && value != "1") {
                    throw std::invalid_argument("--aruco3 requires 0 or 1");
                }
                aruco3 = value == "1";
            } else if (i + 1 < argc &&
                       (argument == "--width" || argument == "--height" ||
                        argument == "--cell-size" || argument == "--min-canonical-side" ||
                        argument == "--refine-iterations" || argument == "--max-candidates" ||
                        argument == "--max-markers")) {
                const std::string value = argv[++i];
                std::size_t used = 0;
                const int parsed = std::stoi(value, &used);
                if (used != value.size() || parsed <= 0) {
                    throw std::invalid_argument(argument + " requires a positive integer");
                }
                if (argument == "--width")
                    width = parsed;
                if (argument == "--height")
                    height = parsed;
                if (argument == "--cell-size")
                    config.perspective_remove_pixel_per_cell_ = parsed;
                if (argument == "--min-canonical-side")
                    config.min_side_length_canonical_img_px_ = parsed;
                if (argument == "--refine-iterations")
                    config.corner_refinement_max_iterations_ = parsed;
                if (argument == "--max-candidates")
                    config.max_candidates_ = parsed;
                if (argument == "--max-markers")
                    config.max_markers_ = parsed;
            } else if (i + 1 < argc &&
                       (argument == "--min-marker-ratio" || argument == "--refine-accuracy")) {
                const std::string value = argv[++i];
                std::size_t used = 0;
                const double parsed = std::stod(value, &used);
                if (used != value.size() || !std::isfinite(parsed) || parsed < 0.0) {
                    throw std::invalid_argument(argument + " requires a finite nonnegative number");
                }
                if (argument == "--min-marker-ratio")
                    config.min_marker_length_ratio_original_img_ = parsed;
                if (argument == "--refine-accuracy")
                    config.corner_refinement_min_accuracy_px_ = parsed;
            } else {
                throw std::invalid_argument(
                        "usage: gpu_quality [--aruco3 0|1] [--dictionary NAME] [--direct] "
                        "[--images DIRECTORY] [--width N] [--height N] [--cell-size N] "
                        "[--min-canonical-side N] [--min-marker-ratio R] "
                        "[--refine-iterations N] [--refine-accuracy R] "
                        "[--max-candidates N] [--max-markers N]; "
                        "without images, render 96 adversarial frames");
            }
        }
        cv::setNumThreads(1);
        const auto dictionary =
                cv::aruco::getPredefinedDictionary(dictionary_type(dictionary_name));
        std::vector<Frame> frames;
        if (image_directory.empty()) {
            for (const auto& spec : aruco3cuda::cpu_quality::make_adversarial_specs(
                         cv::Size(width, height), dictionary.bytesList.rows)) {
                frames.push_back({spec.name_, spec.family_,
                                  aruco3cuda::cpu_quality::render_frame(spec, dictionary),
                                  make_truth(spec), true, spec.require_detection_});
            }
        } else {
            std::vector<std::filesystem::path> paths;
            for (const auto& entry : std::filesystem::directory_iterator(image_directory)) {
                if (entry.is_regular_file()) {
                    const std::string extension = entry.path().extension().string();
                    if (extension == ".png" || extension == ".pgm" || extension == ".jpg" ||
                        extension == ".jpeg") {
                        paths.push_back(entry.path());
                    }
                }
            }
            std::sort(paths.begin(), paths.end());
            for (const auto& path : paths) {
                cv::Mat image = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
                if (image.empty()) {
                    throw std::runtime_error("cannot read " + path.string());
                }
                frames.push_back(
                        {path.filename().string(), "unannotated", image, {}, false, false});
            }
        }
        if (frames.empty()) {
            throw std::invalid_argument("empty input set");
        }
        config.use_aruco3_detection_ = aruco3;
        config.corner_refine_method_ = aruco3 ? aruco3cuda::CornerRefineMethod::kSubpix
                                              : aruco3cuda::CornerRefineMethod::kNone;
        config.max_width_px_ = 1;
        config.max_height_px_ = 1;
        for (const Frame& frame : frames) {
            config.max_width_px_ = std::max(config.max_width_px_, frame.pixels_.cols);
            config.max_height_px_ = std::max(config.max_height_px_, frame.pixels_.rows);
        }
        cv::aruco::DetectorParameters parameters;
        parameters.useAruco3Detection = aruco3;
        parameters.minSideLengthCanonicalImg = config.min_side_length_canonical_img_px_;
        parameters.minMarkerLengthRatioOriginalImg = config.min_marker_length_ratio_original_img_;
        parameters.cornerRefinementMethod =
                aruco3 ? cv::aruco::CORNER_REFINE_SUBPIX : cv::aruco::CORNER_REFINE_NONE;
        parameters.perspectiveRemovePixelPerCell = config.perspective_remove_pixel_per_cell_;
        parameters.cornerRefinementMaxIterations = config.corner_refinement_max_iterations_;
        parameters.cornerRefinementMinAccuracy = config.corner_refinement_min_accuracy_px_;
        cv::aruco::ArucoDetector reference(dictionary, parameters);
        aruco3cuda::cpu::CpuDetector optimized;
        std::string message;
        require(optimized.initialize(dictionary_name, config, &message), message);
        aruco3cuda::Detector gpu;
        const auto* table = aruco3cuda::find_builtin_dictionary(dictionary_name.c_str());
        if (table == nullptr) {
            throw std::invalid_argument("dictionary not bundled in GPU build");
        }
        require(gpu.initialize(*table, config, &message), message);
        Stream stream(direct);
        aruco3cuda::hybrid::DeviceImage device;
        cv::Mat storage(config.max_height_px_ + 4, config.max_width_px_ + 11, CV_8UC1);
        std::vector<std::size_t> order(frames.size());
        std::iota(order.begin(), order.end(), 0U);
        std::mt19937 random(20261117U);
        std::shuffle(order.begin(), order.end(), random);
        // Revisit the same frame after other content to expose stale current-frame workspaces.
        const std::vector<std::size_t> revisited(
                order.begin(), order.begin() + std::min<std::size_t>(8U, frames.size()));
        order.insert(order.end(), revisited.begin(), revisited.end());
        std::size_t cpu_mismatch_frames = 0U;
        std::size_t gpu_discrete_regression_frames = 0U;
        std::size_t failed_positive_controls = 0U;
        std::size_t unannotated_discrete_differences = 0U;
        std::size_t sequence = 0U;
        const auto process = [&](const Frame& frame) {
            storage.setTo(cv::Scalar(sequence % 2U == 0U ? 17 : 231));
            cv::Mat image = storage(cv::Rect(3, 2, frame.pixels_.cols, frame.pixels_.rows));
            frame.pixels_.copyTo(image);
            aruco3cuda::util::Sha256 hash;
            for (int y = 0; y < image.rows; ++y) {
                hash.update(image.ptr(y), static_cast<std::size_t>(image.cols));
            }
            const std::string digest = hash.finalize();
            std::vector<int> ids;
            std::vector<std::vector<cv::Point2f>> corners;
            reference.detectMarkers(image, corners, ids);
            std::array<std::vector<Observation>, 3> observations;
            observations[0] = from_host(ids, corners);
            aruco3cuda::cpu::CpuResult cpu;
            require(optimized.detect(image, &cpu, &message), message);
            observations[1] = from_host(cpu.ids_, cpu.corners_);
            require(device.reserve(image.cols, image.rows, &message), message);
            require(device.upload(image.data, image.cols, image.rows, image.step, &message),
                    message);
            require(gpu.detect_async(device.view(), stream.get(), &message), message);
            aruco3cuda::HostDetections result;
            require(gpu.download(&result, stream.get(), &message), message);
            if (result.marker_overflow_) {
                throw std::runtime_error("GPU marker overflow");
            }
            for (std::size_t i = 0; i < result.ids_.size(); ++i) {
                Observation observation;
                observation.id_ = result.ids_[i];
                std::copy_n(result.corners_.begin() + static_cast<std::ptrdiff_t>(i * 8U), 8U,
                            observation.corners_.begin());
                observations[2].push_back(observation);
            }
            if (cv::norm(image, frame.pixels_, cv::NORM_INF) != 0.0) {
                throw std::runtime_error("detector modified input pixels");
            }
            aruco3cuda::report::CompareConfig compare;
            compare.corner_tolerance_px_ = 0.0;
            const auto cpu_difference = aruco3cuda::report::compare_detections(
                    frame.name_, as_detections(observations[0]), as_detections(observations[1]),
                    compare);
            cpu_mismatch_frames += cpu_difference.agrees() ? 0U : 1U;
            compare.corner_tolerance_px_ = 1.0;
            const auto gpu_difference = aruco3cuda::report::compare_detections(
                    frame.name_, as_detections(observations[0]), as_detections(observations[2]),
                    compare);
            const bool discrete_difference = std::any_of(
                    gpu_difference.diffs_.begin(), gpu_difference.diffs_.end(),
                    [](const auto& difference) {
                        return difference.kind_ != aruco3cuda::report::DiffKind::kCornerShift;
                    });
            unannotated_discrete_differences += !frame.annotated_ && discrete_difference ? 1U : 0U;
            const bool failed_positive =
                    frame.positive_control_ &&
                    (observations[0].empty() || observations[1].empty() || observations[2].empty());
            failed_positive_controls += failed_positive ? 1U : 0U;
            JsonWriter writer(std::cout, 0);
            writer.begin_object();
            writer.member_string("type", "frame");
            writer.member_int("sequence", static_cast<long long>(sequence++));
            writer.member_string("name", frame.name_);
            writer.member_string("family", frame.family_);
            writer.member_string("dictionary", dictionary_name);
            writer.member_bool("aruco3", aruco3);
            writer.member_bool("graph", !direct);
            writer.key("configuration");
            writer.begin_object();
            writer.member_int("max_candidates", config.max_candidates_);
            writer.member_int("max_markers", config.max_markers_);
            writer.member_int("perspective_remove_pixel_per_cell",
                              config.perspective_remove_pixel_per_cell_);
            writer.member_int("min_side_length_canonical_img",
                              config.min_side_length_canonical_img_px_);
            writer.member_double("min_marker_length_ratio_original_img",
                                 config.min_marker_length_ratio_original_img_, 17);
            writer.member_int("corner_refinement_max_iterations",
                              config.corner_refinement_max_iterations_);
            writer.member_double("corner_refinement_min_accuracy",
                                 config.corner_refinement_min_accuracy_px_, 17);
            writer.end_object();
            writer.member_string("opencv_version", CV_VERSION);
            writer.member_string("gpu_resize_profile", aruco3cuda::compiled_resize_profile());
            writer.member_string("gpu_resize_backend", aruco3cuda::compiled_resize_backend());
            writer.member_string("gpu_carotene_x_fma", aruco3cuda::compiled_carotene_x_fma());
            writer.member_string("gpu_strategy_identity",
                                 aruco3cuda::compiled_gpu_strategy_identity());
            writer.member_string("pixel_sha256", digest);
            writer.member_int("width", image.cols);
            writer.member_int("height", image.rows);
            writer.member_bool("has_ground_truth", frame.annotated_);
            writer.member_bool("positive_control", frame.positive_control_);
            writer.member_bool("cpu_exact", cpu_difference.agrees());
            writer.member_bool("failed_positive_control", failed_positive);
            writer.key("cpu_gpu_differences_at_1px");
            writer.begin_array();
            for (const auto& difference : gpu_difference.diffs_) {
                writer.begin_object();
                writer.member_string("kind", aruco3cuda::report::diff_kind_name(difference.kind_));
                writer.member_int("baseline_id", difference.baseline_id_);
                writer.member_int("target_id", difference.target_id_);
                writer.member_double("corner_error_px", difference.corner_error_px_, 17);
                writer.end_object();
            }
            writer.end_array();
            if (frame.annotated_) {
                std::vector<Observation> truth_observations;
                for (const auto& marker : frame.truth_) {
                    truth_observations.push_back({marker.id_, marker.corners_});
                }
                writer.key("truth");
                write_observations(writer, truth_observations);
            }
            const std::array<const char*, 3> names{"opencv", "optimized_cpu", "gpu"};
            std::array<aruco3cuda::evaluate::ImageAccuracy, 3> metrics;
            for (std::size_t i = 0; i < names.size(); ++i) {
                writer.key(names[i]);
                write_observations(writer, observations[i]);
                if (frame.annotated_) {
                    metrics[i] = aruco3cuda::evaluate::compare_to_truth(frame.name_, frame.truth_,
                                                                        observations[i], {});
                    writer.key(std::string(names[i]) + "_quality");
                    write_metrics(writer, metrics[i]);
                }
            }
            bool discrete_regression = false;
            if (frame.annotated_) {
                discrete_regression = metrics[2].false_positive_ > metrics[0].false_positive_;
                for (std::size_t i = 0; i < frame.truth_.size(); ++i) {
                    if (metrics[0].truth_outcomes_[i] ==
                        aruco3cuda::evaluate::TruthOutcome::kDetected) {
                        discrete_regression =
                                discrete_regression ||
                                metrics[2].truth_outcomes_[i] !=
                                        aruco3cuda::evaluate::TruthOutcome::kDetected ||
                                (metrics[0].truth_rotation_agreed_[i] &&
                                 !metrics[2].truth_rotation_agreed_[i]);
                    }
                }
            }
            gpu_discrete_regression_frames += discrete_regression ? 1U : 0U;
            writer.member_bool("gpu_discrete_regression", discrete_regression);
            writer.end_object();
            std::cout << '\n';
        };
        for (std::size_t i = 0; i < order.size(); ++i) {
            process(frames[order[i]]);
            if (i % 7U == 6U) {
                Frame blank{"inserted_blank",
                            "sequence_cut",
                            cv::Mat(frames[order[i]].pixels_.size(), CV_8UC1, cv::Scalar(245)),
                            {},
                            true,
                            false};
                process(blank);
            }
        }
        JsonWriter writer(std::cout, 0);
        writer.begin_object();
        writer.member_string("type", "completion");
        writer.member_int("frames", static_cast<long long>(sequence));
        writer.member_int("cpu_mismatch_frames", static_cast<long long>(cpu_mismatch_frames));
        writer.member_int("gpu_discrete_regression_frames",
                          static_cast<long long>(gpu_discrete_regression_frames));
        writer.member_int("failed_positive_controls",
                          static_cast<long long>(failed_positive_controls));
        writer.member_int("unannotated_discrete_difference_frames",
                          static_cast<long long>(unannotated_discrete_differences));
        writer.member_string(
                "corner_gate",
                "unclassified: compare common-marker GT errors before accepting quality");
        writer.end_object();
        std::cout << '\n';
        return cpu_mismatch_frames != 0U || gpu_discrete_regression_frames != 0U ||
                               failed_positive_controls != 0U ||
                               unannotated_discrete_differences != 0U
                       ? 2
                       : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
