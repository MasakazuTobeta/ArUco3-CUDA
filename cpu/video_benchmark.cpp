// SPDX-License-Identifier: Apache-2.0
// Persistent workers rerun detection and preprocessing on each indexed input.
#include <sched.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "aruco3cuda/util/json_writer.hpp"
#include "aruco3cuda/util/sha256.hpp"
#include "aruco3cuda/util/statistics.hpp"
#include "cpu_detector.hpp"

namespace {
using Clock = std::chrono::steady_clock;
using aruco3cuda::cpu::CpuDetector;
using aruco3cuda::cpu::CpuOptions;
using aruco3cuda::cpu::CpuResult;
using aruco3cuda::util::JsonWriter;

struct Settings {
    std::string input_dir_ = "data/cpu-experiment";
    std::string filter_ = "cpu_1280x720_";
    std::vector<std::string> inputs_;
    std::string route_ = "both";
    std::string layout_ = "both";
    std::string tuning_ = "default";
    std::string test_failure_ = "none";
    std::string cpu_list_;
    int frames_ = 240;
    int workers_ = 4;
    int opencv_threads_ = 1;
    int warmup_ = 18;
    bool aruco3_ = true;
    bool validate_only_ = false;
    bool save_frames_ = false;
    bool reverse_order_ = false;
};

struct Input {
    std::string path_;
    std::string file_sha256_;
    std::string pixels_sha256_;
    cv::Mat image_;
};

struct Layout {
    std::string name_;
    int workers_ = 1;
    int opencv_threads_ = 1;
};

int integer(const std::string& text) {
    std::size_t consumed = 0U;
    const int value = std::stoi(text, &consumed);
    if (consumed != text.size()) {
        throw std::invalid_argument("invalid integer: " + text);
    }
    return value;
}

bool one_of(const std::string& value, const std::vector<std::string>& choices) {
    return std::find(choices.begin(), choices.end(), value) != choices.end();
}

Settings parse(int argc, char** argv) {
    Settings settings;
    for (int index = 1; index < argc; ++index) {
        const std::string name = argv[index];
        if (name == "--help") {
            std::cout << "usage: aruco3cuda_cpu_video [--input-dir DIR --filter TEXT | "
                         "--input FILE ...] [--frames N] [--workers W] [--opencv-threads T] "
                         "[--cpu-list LIST] [--layout both|intra|workers] "
                         "[--route both|opencv|cpu] [--aruco3 0|1] [--warmup N] "
                         "[--tuning "
                         "default|previous|control|spatial|sparse|contours|cached|lookup|round2] "
                         "[--validate-only] [--save-frames] [--reverse-order] "
                         "[--test-worker-failure setup|frame|spawn]\n"
                         "Pin exactly W*T CPUs; both layouts use that same CPU budget.\n";
            std::exit(EXIT_SUCCESS);
        }
        if (name == "--validate-only") {
            settings.validate_only_ = true;
            continue;
        }
        if (name == "--save-frames") {
            settings.save_frames_ = true;
            continue;
        }
        if (name == "--reverse-order") {
            settings.reverse_order_ = true;
            continue;
        }
        if (++index >= argc) {
            throw std::invalid_argument("missing value for " + name);
        }
        const std::string value = argv[index];
        if (name == "--input-dir")
            settings.input_dir_ = value;
        else if (name == "--input")
            settings.inputs_.push_back(value);
        else if (name == "--filter")
            settings.filter_ = value;
        else if (name == "--frames")
            settings.frames_ = integer(value);
        else if (name == "--workers")
            settings.workers_ = integer(value);
        else if (name == "--opencv-threads")
            settings.opencv_threads_ = integer(value);
        else if (name == "--warmup")
            settings.warmup_ = integer(value);
        else if (name == "--cpu-list")
            settings.cpu_list_ = value;
        else if (name == "--route")
            settings.route_ = value;
        else if (name == "--layout")
            settings.layout_ = value;
        else if (name == "--tuning")
            settings.tuning_ = value;
        else if (name == "--test-worker-failure")
            settings.test_failure_ = value;
        else if (name == "--aruco3") {
            const int enabled = integer(value);
            if (enabled < 0 || enabled > 1) {
                throw std::invalid_argument("--aruco3 must be 0 or 1");
            }
            settings.aruco3_ = enabled != 0;
        } else {
            throw std::invalid_argument("unknown argument: " + name);
        }
    }
    if (settings.workers_ < 1 || settings.workers_ > 64 || settings.opencv_threads_ < 1 ||
        settings.opencv_threads_ > 64 || settings.workers_ * settings.opencv_threads_ > 64 ||
        settings.frames_ < 2 || settings.frames_ > 1000000 || settings.warmup_ < 0 ||
        settings.warmup_ > 1000000) {
        throw std::invalid_argument(
                "require 2..1000000 frames, positive W*T <= 64, "
                "and 0..1000000 warmup calls");
    }
    if (!one_of(settings.route_, {"both", "opencv", "cpu"}) ||
        !one_of(settings.layout_, {"both", "intra", "workers"}) ||
        !one_of(settings.tuning_, {"default", "previous", "control", "spatial", "sparse",
                                   "contours", "cached", "lookup", "round2"})) {
        throw std::invalid_argument("invalid route, layout, or CPU tuning");
    }
    if (settings.layout_ != "intra" && settings.frames_ < settings.workers_) {
        throw std::invalid_argument("provide at least one measured frame per worker");
    }
    if (!one_of(settings.test_failure_, {"none", "setup", "frame", "spawn"}) ||
        (settings.test_failure_ != "none" && !settings.validate_only_)) {
        throw std::invalid_argument("worker failure injection requires --validate-only");
    }
    if (settings.test_failure_ == "spawn" &&
        (settings.workers_ < 2 || settings.layout_ != "workers")) {
        throw std::invalid_argument("spawn failure injection requires worker layout with W >= 2");
    }
    return settings;
}

std::vector<int> set_and_read_affinity(const std::string& cpu_list) {
    if (!cpu_list.empty()) {
        cpu_set_t requested;
        CPU_ZERO(&requested);
        std::size_t begin = 0U;
        while (begin <= cpu_list.size()) {
            const std::size_t end = cpu_list.find(',', begin);
            const int cpu = integer(cpu_list.substr(begin, end - begin));
            if (cpu < 0 || cpu >= CPU_SETSIZE || CPU_ISSET(cpu, &requested) != 0) {
                throw std::invalid_argument("CPU list contains an invalid or repeated index");
            }
            CPU_SET(cpu, &requested);
            if (end == std::string::npos) {
                break;
            }
            begin = end + 1U;
        }
        if (::sched_setaffinity(0, sizeof(requested), &requested) != 0) {
            throw std::runtime_error("sched_setaffinity failed for " + cpu_list);
        }
    }
    cpu_set_t active;
    CPU_ZERO(&active);
    if (::sched_getaffinity(0, sizeof(active), &active) != 0) {
        throw std::runtime_error("sched_getaffinity failed");
    }
    std::vector<int> result;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &active) != 0) {
            result.push_back(cpu);
        }
    }
    return result;
}

std::vector<Input> load_inputs(const Settings& settings) {
    std::vector<std::filesystem::path> paths;
    if (!settings.inputs_.empty()) {
        for (const std::string& path : settings.inputs_) {
            paths.emplace_back(path);
        }
    } else {
        for (const auto& entry : std::filesystem::directory_iterator(settings.input_dir_)) {
            const std::string extension = entry.path().extension().string();
            if (entry.is_regular_file() && one_of(extension, {".png", ".jpg", ".jpeg", ".pgm"}) &&
                entry.path().filename().string().find(settings.filter_) != std::string::npos) {
                paths.push_back(entry.path());
            }
        }
        std::sort(paths.begin(), paths.end());
    }
    if (paths.size() < 2U || paths.size() > 256U) {
        throw std::invalid_argument("select 2..256 input images for the changing-frame sequence");
    }
    std::vector<Input> inputs;
    for (const auto& path : paths) {
        Input input;
        input.path_ = path.string();
        input.image_ = cv::imread(input.path_, cv::IMREAD_GRAYSCALE);
        if (input.image_.empty() || input.image_.cols > 65536 || input.image_.rows > 65536 ||
            input.image_.total() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("invalid image: " + input.path_);
        }
        if (!aruco3cuda::util::sha256_file(input.path_, &input.file_sha256_)) {
            throw std::runtime_error("cannot hash " + input.path_);
        }
        aruco3cuda::util::Sha256 pixels;
        const std::string dimensions =
                std::to_string(input.image_.cols) + "x" + std::to_string(input.image_.rows) + ":";
        pixels.update(dimensions.data(), dimensions.size());
        for (int row = 0; row < input.image_.rows; ++row) {
            pixels.update(input.image_.ptr(row), static_cast<std::size_t>(input.image_.cols));
        }
        input.pixels_sha256_ = pixels.finalize();
        inputs.push_back(std::move(input));
    }
    const std::size_t used = std::min(inputs.size(), static_cast<std::size_t>(settings.frames_));
    bool changing = false;
    for (std::size_t index = 1U; index < used; ++index) {
        changing = changing || inputs[index].pixels_sha256_ != inputs[0].pixels_sha256_;
    }
    if (!changing) {
        throw std::invalid_argument("the measured sequence must contain different decoded frames");
    }
    return inputs;
}

aruco3cuda::DetectorConfig detector_config(const Settings& settings) {
    aruco3cuda::DetectorConfig config;
    config.use_aruco3_detection_ = settings.aruco3_;
    config.corner_refine_method_ = settings.aruco3_ ? aruco3cuda::CornerRefineMethod::kSubpix
                                                    : aruco3cuda::CornerRefineMethod::kNone;
    return config;
}

CpuOptions cpu_options(const std::string& tuning) {
    CpuOptions options;
    if (tuning == "default")
        return options;
    // Ablations remain relative to the previous six-switch detector even when
    // newly selected defaults change. The default route above preserves them.
    options.reuse_pyramid_ = true;
    options.packed_dictionary_ = true;
    options.fused_cells_ = true;
    options.multiwindow_threshold_ = true;
    options.parallel_windows_ = true;
    options.parallel_decode_ = true;
    options.spatial_grouping_ = false;
    options.sparse_pyramid_ = false;
    options.filtered_contours_ = false;
    options.cached_refinement_ = false;
    options.exact_dictionary_lookup_ = false;
    options.simple_contours_ = false;
    options.component_prefilter_ = false;
    options.dryrun_otsu_ = false;
    options.component_candidates_ = false;
    if (tuning == "control") {
        options.reuse_pyramid_ = false;
        options.packed_dictionary_ = false;
        options.exact_dictionary_lookup_ = false;
        options.fused_cells_ = false;
        options.multiwindow_threshold_ = false;
        options.parallel_windows_ = false;
        options.parallel_decode_ = false;
        options.spatial_grouping_ = false;
        options.sparse_pyramid_ = false;
        options.filtered_contours_ = false;
        options.cached_refinement_ = false;
        options.simple_contours_ = false;
        options.component_prefilter_ = false;
        options.dryrun_otsu_ = false;
        options.component_candidates_ = false;
        return options;
    }
    if (tuning == "spatial" || tuning == "round2")
        options.spatial_grouping_ = true;
    if (tuning == "sparse" || tuning == "round2")
        options.sparse_pyramid_ = true;
    if (tuning == "contours" || tuning == "round2")
        options.filtered_contours_ = true;
    if (tuning == "cached" || tuning == "round2")
        options.cached_refinement_ = true;
    if (tuning == "lookup" || tuning == "round2")
        options.exact_dictionary_lookup_ = true;
    return options;
}

class Runner {
public:
    Runner(const Settings& settings, const std::string& route) {
        const auto config = detector_config(settings);
        if (route == "opencv") {
            cv::aruco::DetectorParameters parameters;
            parameters.useAruco3Detection = settings.aruco3_;
            parameters.minSideLengthCanonicalImg = config.min_side_length_canonical_img_px_;
            parameters.minMarkerLengthRatioOriginalImg =
                    config.min_marker_length_ratio_original_img_;
            parameters.cornerRefinementMethod = settings.aruco3_ ? cv::aruco::CORNER_REFINE_SUBPIX
                                                                 : cv::aruco::CORNER_REFINE_NONE;
            this->reference_ = std::make_unique<cv::aruco::ArucoDetector>(
                    cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_MIP_36h12),
                    parameters);
        } else {
            this->candidate_ = std::make_unique<CpuDetector>();
            std::string message;
            if (this->candidate_->initialize("DICT_ARUCO_MIP_36h12", config, &message,
                                             cpu_options(settings.tuning_)) !=
                aruco3cuda::Status::kOk) {
                throw std::runtime_error(message);
            }
        }
    }

    void detect(const cv::Mat& image, CpuResult* out, bool rejected) {
        if (this->reference_) {
            if (rejected) {
                this->reference_->detectMarkers(image, out->corners_, out->ids_, this->rejected_);
                out->rejected_count_ = this->rejected_.size();
            } else {
                this->reference_->detectMarkers(image, out->corners_, out->ids_);
            }
        } else {
            std::string message;
            if (this->candidate_->detect(image, out, &message) != aruco3cuda::Status::kOk) {
                throw std::runtime_error(message);
            }
        }
    }

private:
    std::unique_ptr<cv::aruco::ArucoDetector> reference_;
    std::unique_ptr<CpuDetector> candidate_;
    std::vector<std::vector<cv::Point2f>> rejected_;
};

CpuResult ordered(const CpuResult& result) {
    if (result.ids_.size() != result.corners_.size()) {
        throw std::runtime_error("ID/corner vector lengths differ");
    }
    for (const auto& corners : result.corners_) {
        if (corners.size() != 4U) {
            throw std::runtime_error("invalid corner count");
        }
        for (const cv::Point2f& corner : corners) {
            if (!std::isfinite(corner.x) || !std::isfinite(corner.y)) {
                throw std::runtime_error("non-finite corner");
            }
        }
    }
    std::vector<std::size_t> order(result.ids_.size());
    std::iota(order.begin(), order.end(), 0U);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t first, std::size_t second) {
        if (result.ids_[first] != result.ids_[second]) {
            return result.ids_[first] < result.ids_[second];
        }
        if (result.corners_[first][0].x != result.corners_[second][0].x) {
            return result.corners_[first][0].x < result.corners_[second][0].x;
        }
        return result.corners_[first][0].y < result.corners_[second][0].y;
    });
    CpuResult sorted;
    sorted.rejected_count_ = result.rejected_count_;
    for (const std::size_t index : order) {
        sorted.ids_.push_back(result.ids_[index]);
        sorted.corners_.push_back(result.corners_[index]);
    }
    return sorted;
}

void compare(const CpuResult& expected, const CpuResult& raw_actual, bool rejected,
             std::size_t frame) {
    const CpuResult actual = ordered(raw_actual);
    if (actual.ids_ != expected.ids_ || actual.corners_ != expected.corners_ ||
        (rejected && actual.rejected_count_ != expected.rejected_count_)) {
        throw std::runtime_error("frame " + std::to_string(frame) +
                                 " differs in IDs, ordered corners, or rejected count");
    }
}

enum class Phase { kValidate, kMeasure };

struct Shared {
    std::mutex mutex_;
    std::condition_variable condition_;
    std::atomic<bool> cancelled_{false};
    std::atomic<std::size_t> cursor_{0U};
    std::size_t generation_ = 0U;
    int ready_ = 0;
    int finished_ = 0;
    Phase phase_ = Phase::kValidate;
    std::exception_ptr error_;
};

struct Frame {
    CpuResult result_;
    Clock::time_point start_{};
    Clock::time_point stop_{};
    int worker_ = -1;
};

// Cancelling before joining also covers partial thread creation and setup errors.
class Pool {
public:
    explicit Pool(Shared& shared) : shared_(shared) {}
    ~Pool() { this->stop_and_join(); }
    std::vector<std::thread>& threads() { return this->threads_; }
    void stop_and_join() {
        {
            std::lock_guard<std::mutex> lock(this->shared_.mutex_);
            this->shared_.cancelled_.store(true);
        }
        this->shared_.condition_.notify_all();
        for (std::thread& thread : this->threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

private:
    Shared& shared_;
    std::vector<std::thread> threads_;
};

void phase(Shared& shared, Phase requested, int workers) {
    std::unique_lock<std::mutex> lock(shared.mutex_);
    shared.finished_ = 0;
    shared.cursor_.store(0U);
    shared.phase_ = requested;
    ++shared.generation_;
    shared.condition_.notify_all();
    shared.condition_.wait(lock,
                           [&] { return shared.cancelled_.load() || shared.finished_ == workers; });
    if (shared.error_) {
        std::rethrow_exception(shared.error_);
    }
}

void run_layout(const Settings& settings, const std::vector<Input>& inputs,
                const std::vector<CpuResult>& expected, const Layout& layout,
                const std::string& route) {
    cv::setNumThreads(layout.opencv_threads_);
    std::vector<Frame> frames(static_cast<std::size_t>(settings.frames_));
    Shared shared;
    Pool pool(shared);
    pool.threads().reserve(static_cast<std::size_t>(layout.workers_));
    for (int worker = 0; worker < layout.workers_; ++worker) {
        if (settings.test_failure_ == "spawn" && worker == 1) {
            throw std::runtime_error("injected failure after partial thread creation");
        }
        pool.threads().emplace_back([&, worker] {
            try {
                if (settings.test_failure_ == "setup" && worker == 0) {
                    throw std::runtime_error("injected worker setup failure");
                }
                Runner runner(settings, route);
                CpuResult warm_result;
                // Every detector sees every input before measurement. Storage
                // reuse is allowed, but detection is rerun for each call.
                for (const Input& input : inputs) {
                    if (shared.cancelled_.load())
                        return;
                    runner.detect(input.image_, &warm_result, false);
                }
                for (int warmup = 0; warmup < settings.warmup_; ++warmup) {
                    if (shared.cancelled_.load())
                        return;
                    runner.detect(inputs[static_cast<std::size_t>(warmup) % inputs.size()].image_,
                                  &warm_result, false);
                }
                std::size_t generation = 0U;
                {
                    std::lock_guard<std::mutex> lock(shared.mutex_);
                    ++shared.ready_;
                    shared.condition_.notify_all();
                }
                for (;;) {
                    Phase current;
                    {
                        std::unique_lock<std::mutex> lock(shared.mutex_);
                        shared.condition_.wait(lock, [&] {
                            return shared.cancelled_.load() || shared.generation_ != generation;
                        });
                        if (shared.cancelled_.load())
                            return;
                        generation = shared.generation_;
                        current = shared.phase_;
                    }
                    while (!shared.cancelled_.load()) {
                        const std::size_t index = shared.cursor_.fetch_add(1U);
                        if (index >= frames.size())
                            break;
                        Frame& frame = frames[index];
                        const Input& input = inputs[index % inputs.size()];
                        if (current == Phase::kMeasure) {
                            frame.worker_ = worker;
                            frame.start_ = Clock::now();
                            runner.detect(input.image_, &frame.result_, false);
                            frame.stop_ = Clock::now();
                        } else {
                            if (settings.test_failure_ == "frame" && index == 0U) {
                                throw std::runtime_error("injected worker frame failure");
                            }
                            if (settings.validate_only_)
                                frame.worker_ = worker;
                            runner.detect(input.image_, &frame.result_, true);
                            compare(expected[index], frame.result_, true, index);
                        }
                    }
                    {
                        std::lock_guard<std::mutex> lock(shared.mutex_);
                        ++shared.finished_;
                        shared.condition_.notify_all();
                    }
                }
            } catch (...) {
                {
                    std::lock_guard<std::mutex> lock(shared.mutex_);
                    if (!shared.error_)
                        shared.error_ = std::current_exception();
                    shared.cancelled_.store(true);
                }
                shared.condition_.notify_all();
            }
        });
    }
    {
        std::unique_lock<std::mutex> lock(shared.mutex_);
        shared.condition_.wait(
                lock, [&] { return shared.cancelled_.load() || shared.ready_ == layout.workers_; });
        if (shared.error_)
            std::rethrow_exception(shared.error_);
    }
    phase(shared, Phase::kValidate, layout.workers_);
    std::vector<double> service_ms;
    double span_ms = 0.0;
    aruco3cuda::util::SampleStatistics statistics;
    if (!settings.validate_only_) {
        phase(shared, Phase::kMeasure, layout.workers_);
        Clock::time_point first = frames.front().start_;
        Clock::time_point last = frames.front().stop_;
        for (std::size_t index = 0U; index < frames.size(); ++index) {
            compare(expected[index], frames[index].result_, route == "cpu", index);
            first = std::min(first, frames[index].start_);
            last = std::max(last, frames[index].stop_);
            service_ms.push_back(std::chrono::duration<double, std::milli>(frames[index].stop_ -
                                                                           frames[index].start_)
                                         .count());
        }
        span_ms = std::chrono::duration<double, std::milli>(last - first).count();
        if (!(span_ms > 0.0) || !aruco3cuda::util::compute_statistics(service_ms, &statistics)) {
            throw std::runtime_error("invalid timing span");
        }
        // The baseline does not materialize rejected coordinates in its timed
        // calls. Validate that output separately on every frame after timing.
        phase(shared, Phase::kValidate, layout.workers_);
    }
    pool.stop_and_join();
    JsonWriter writer(std::cout, 0);
    writer.begin_object();
    writer.member_string("type", "video_measurement");
    writer.member_string("route", route);
    writer.member_string("layout", layout.name_);
    writer.member_string("tuning", settings.tuning_);
    writer.member_int("workers", layout.workers_);
    writer.member_int("requested_opencv_threads", layout.opencv_threads_);
    writer.member_int("opencv_threads", cv::getNumThreads());
    writer.member_int("frames", settings.frames_);
    writer.member_int("validated_frames", settings.frames_);
    writer.member_int("untimed_validation_passes", settings.validate_only_ ? 1 : 2);
    writer.member_int("timed_frames_checked", settings.validate_only_ ? 0 : settings.frames_);
    writer.member_bool("timed_rejected_counts_checked", !settings.validate_only_ && route == "cpu");
    writer.member_bool("exact_ids_and_ordered_corners", true);
    writer.member_bool("exact_untimed_rejected_counts", true);
    writer.member_bool("validate_only", settings.validate_only_);
    writer.member_string("throughput_scope",
                         "first service start to last completion; includes dispatch gaps");
    writer.member_string("service_scope",
                         "one detector call with host IDs and corners; no queue wait");
    for (const auto& entry : std::vector<std::pair<std::string, double>>{
                 {"service_p50_ms", statistics.p50_},
                 {"service_p95_ms", statistics.p95_},
                 {"service_p99_ms", statistics.p99_},
                 {"saturated_span_ms", span_ms},
                 {"saturated_throughput_fps",
                  settings.frames_ * 1000.0 / (span_ms > 0.0 ? span_ms : 1.0)}}) {
        writer.key(entry.first);
        if (settings.validate_only_)
            writer.value_null();
        else
            writer.value_double(entry.second, 9);
    }
    writer.end_object();
    std::cout << '\n';
    if (settings.save_frames_) {
        for (std::size_t index = 0U; index < frames.size(); ++index) {
            JsonWriter frame(std::cout, 0);
            frame.begin_object();
            frame.member_string("type", "video_frame");
            frame.member_string("route", route);
            frame.member_string("layout", layout.name_);
            frame.member_int("frame", static_cast<long long>(index));
            frame.member_int("input", static_cast<long long>(index % inputs.size()));
            frame.member_int("worker", frames[index].worker_);
            frame.member_int("detections",
                             static_cast<long long>(frames[index].result_.ids_.size()));
            frame.member_int("validated_rejected_count",
                             static_cast<long long>(frames[index].result_.rejected_count_));
            frame.key("service_ms");
            if (settings.validate_only_)
                frame.value_null();
            else
                frame.value_double(service_ms[index], 9);
            frame.end_object();
            std::cout << '\n';
        }
    }
}

void run(const Settings& settings) {
    const std::vector<int> cpus = set_and_read_affinity(settings.cpu_list_);
    const int budget = settings.workers_ * settings.opencv_threads_;
    if (cpus.size() != static_cast<std::size_t>(budget)) {
        throw std::invalid_argument(
                "pin exactly workers*opencv-threads CPUs with taskset or --cpu-list");
    }
    const std::vector<Input> inputs = load_inputs(settings);
    std::vector<CpuResult> expected(static_cast<std::size_t>(settings.frames_));
    cv::setNumThreads(1);
    Runner oracle(settings, "opencv");
    for (std::size_t frame = 0U; frame < expected.size(); ++frame) {
        oracle.detect(inputs[frame % inputs.size()].image_, &expected[frame], true);
        expected[frame] = ordered(expected[frame]);
    }
    aruco3cuda::util::Sha256 sequence;
    for (int frame = 0; frame < settings.frames_; ++frame) {
        const std::string entry =
                std::to_string(frame) + ":" +
                inputs[static_cast<std::size_t>(frame) % inputs.size()].pixels_sha256_ + "\n";
        sequence.update(entry.data(), entry.size());
    }
    std::string binary_sha256;
    if (!aruco3cuda::util::sha256_file("/proc/self/exe", &binary_sha256)) {
        throw std::runtime_error("cannot hash executable");
    }
    JsonWriter environment(std::cout, 0);
    environment.begin_object();
    environment.member_string("type", "video_environment");
    environment.member_string("opencv_version", CV_VERSION);
    environment.member_string("compiler", __VERSION__);
    environment.member_string("binary_sha256", binary_sha256);
    environment.member_string("dictionary", "DICT_ARUCO_MIP_36h12");
    environment.member_bool("aruco3", settings.aruco3_);
    environment.member_int("cpu_budget", budget);
    environment.member_int("frames", settings.frames_);
    environment.member_int("warmup_calls_per_worker", settings.warmup_);
    environment.member_int("input_preparation_calls_per_worker",
                           static_cast<long long>(inputs.size()));
    environment.member_bool("tracking", false);
    environment.member_bool("startup_excluded", true);
    environment.member_bool("validate_only", settings.validate_only_);
    environment.member_bool("reverse_order", settings.reverse_order_);
    environment.member_string("test_worker_failure", settings.test_failure_);
    const auto config = detector_config(settings);
    const CpuOptions options = cpu_options(settings.tuning_);
    environment.member_int("min_side_length_canonical_img_px",
                           config.min_side_length_canonical_img_px_);
    environment.member_double("min_marker_length_ratio_original_img",
                              config.min_marker_length_ratio_original_img_, 9);
    environment.member_int("threshold_window_min_px", config.adaptive_thresh_win_size_min_px_);
    environment.member_int("threshold_window_max_px", config.adaptive_thresh_win_size_max_px_);
    environment.member_int("threshold_window_step_px", config.adaptive_thresh_win_size_step_px_);
    environment.member_double("threshold_constant", config.adaptive_thresh_constant_, 9);
    environment.member_double("error_correction_rate", config.error_correction_rate_, 9);
    environment.member_bool("reuse_pyramid", options.reuse_pyramid_);
    environment.member_bool("packed_dictionary", options.packed_dictionary_);
    environment.member_bool("exact_dictionary_lookup", options.exact_dictionary_lookup_);
    environment.member_bool("fused_cells", options.fused_cells_);
    environment.member_bool("multiwindow_threshold", options.multiwindow_threshold_);
    environment.member_bool("parallel_windows", options.parallel_windows_);
    environment.member_bool("parallel_decode", options.parallel_decode_);
    environment.member_bool("spatial_grouping", options.spatial_grouping_);
    environment.member_bool("sparse_pyramid", options.sparse_pyramid_);
    environment.member_bool("filtered_contours", options.filtered_contours_);
    environment.member_bool("cached_refinement", options.cached_refinement_);
    environment.member_int("threshold_parallel_min_pixels", options.threshold_parallel_min_pixels_);
    environment.member_int("threshold_max_prefix_bytes",
                           static_cast<long long>(options.threshold_max_prefix_bytes_));
    environment.member_int("sparse_tile_side_px", options.sparse_tile_side_px_);
    environment.member_double("sparse_full_level_ratio", options.sparse_full_level_ratio_, 9);
    environment.member_string("sequence",
                              "frame index modulo sorted inputs, or explicit --input order");
    environment.member_string("sequence_sha256", sequence.finalize());
    environment.key("cpu_affinity");
    environment.begin_array();
    for (const int cpu : cpus)
        environment.value_int(cpu);
    environment.end_array();
    environment.key("inputs");
    environment.begin_array();
    for (const Input& input : inputs) {
        environment.begin_object();
        environment.member_string("path", input.path_);
        environment.member_string("file_sha256", input.file_sha256_);
        environment.member_string("pixels_sha256", input.pixels_sha256_);
        environment.member_int("width", input.image_.cols);
        environment.member_int("height", input.image_.rows);
        environment.end_object();
    }
    environment.end_array();
    environment.end_object();
    std::cout << '\n';
    std::vector<Layout> layouts;
    if (settings.layout_ != "workers")
        layouts.push_back({"intra", 1, budget});
    if (settings.layout_ != "intra")
        layouts.push_back({"workers", settings.workers_, settings.opencv_threads_});
    std::vector<std::string> routes = {"opencv", "cpu"};
    if (settings.reverse_order_) {
        std::reverse(routes.begin(), routes.end());
        std::reverse(layouts.begin(), layouts.end());
    }
    for (const std::string& route : routes) {
        if (settings.route_ != "both" && settings.route_ != route)
            continue;
        for (const Layout& layout : layouts) {
            run_layout(settings, inputs, expected, layout, route);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        run(parse(argc, argv));
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "video benchmark: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
