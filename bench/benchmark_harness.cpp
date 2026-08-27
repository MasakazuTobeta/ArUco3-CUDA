// SPDX-License-Identifier: Apache-2.0
#include "benchmark_harness.hpp"

#include <stdio.h>  // popen と pclose は POSIX であり <cstdio> の std 名前空間に無い
#include <sys/utsname.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <ostream>
#include <ratio>
#include <string>
#include <vector>

#include "aruco3cuda/device_probe.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/util/json_writer.hpp"
#include "aruco3cuda/util/statistics.hpp"
#include "reference_runner.hpp"

namespace aruco3cuda::bench {
namespace {

using aruco3cuda::util::JsonWriter;
using aruco3cuda::util::SampleStatistics;

/// command を実行し標準出力を 1 行取得する。取得できない場合は空文字列を返す。
///
/// nvidia-smi や nvpmodel のように、library 経由で取得できない環境情報を
/// 記録するために使用する。失敗しても測定自体は継続する。
std::string read_command_line(const std::string& command) {
    std::array<char, 512> buffer{};
    // popen は RAII 対象にならないため、この関数内で確実に閉じる。
    std::FILE* pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) {
        return std::string();
    }
    std::string result;
    if (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        result = buffer.data();
    }
    pclose(pipe);
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r' ||
                              result.back() == ' ')) {
        result.pop_back();
    }
    return result;
}

std::string read_os_pretty_name() {
    std::ifstream release("/etc/os-release");
    if (!release) {
        return std::string();
    }
    std::string line;
    while (std::getline(release, line)) {
        const std::string prefix = "PRETTY_NAME=";
        if (line.rfind(prefix, 0) == 0) {
            std::string value = line.substr(prefix.size());
            if (value.size() >= 2U && value.front() == '"' && value.back() == '"') {
                value = value.substr(1, value.size() - 2);
            }
            return value;
        }
    }
    return std::string();
}

void write_statistics(JsonWriter& writer, const std::string& name,
                      const SampleStatistics& stats) {
    writer.key(name);
    writer.begin_object();
    writer.member_int("count", static_cast<long long>(stats.count_));
    writer.member_double("min_ms", stats.min_, 4);
    writer.member_double("max_ms", stats.max_, 4);
    writer.member_double("mean_ms", stats.mean_, 4);
    writer.member_double("stddev_ms", stats.stddev_, 4);
    writer.member_double("p50_ms", stats.p50_, 4);
    writer.member_double("p95_ms", stats.p95_, 4);
    writer.member_double("p99_ms", stats.p99_, 4);
    writer.end_object();
}

}  // namespace

const char* to_string(Route route) {
    switch (route) {
        case Route::kCpu:
            return "CPU";
        case Route::kCudaEndToEnd:
            return "CUDA-E2E";
        case Route::kCudaResident:
            return "CUDA-Resident";
        case Route::kHybrid:
            return "Hybrid";
    }
    return "Unknown";
}

const char* to_string(MemoryMode mode) {
    switch (mode) {
        case MemoryMode::kNotApplicable:
            return "N/A";
        case MemoryMode::kHostPageable:
            return "M-Pageable";
        case MemoryMode::kHostPinned:
            return "M-Pinned";
        case MemoryMode::kManaged:
            return "M-Managed";
        case MemoryMode::kDevice:
            return "M-Device";
    }
    return "Unknown";
}

EnvironmentRecord collect_environment(const BenchmarkConfig& config) {
    EnvironmentRecord environment;

    std::array<char, 256> hostname{};
    if (gethostname(hostname.data(), hostname.size() - 1U) == 0) {
        environment.hostname_ = hostname.data();
    }
    utsname system_info{};
    if (uname(&system_info) == 0) {
        environment.kernel_ = system_info.release;
        environment.architecture_ = system_info.machine;
    }
    environment.os_ = read_os_pretty_name();
    environment.cpu_online_cores_ = static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));

    const aruco3cuda::reference::ReferenceEnvironment detector_environment =
            aruco3cuda::reference::collect_environment(config.detector_);
    environment.opencv_version_ = detector_environment.opencv_version_;
    environment.opencv_threads_ = detector_environment.opencv_threads_;

    // GPU 情報。device が無い環境でも測定自体は成立するため、失敗を致命的に扱わない。
    int device_count = 0;
    if (aruco3cuda::device_count(&device_count) == aruco3cuda::Status::kOk && device_count > 0) {
        aruco3cuda::DeviceProbeResult probe;
        if (aruco3cuda::probe_device(0, &probe) == aruco3cuda::Status::kOk) {
            // device 名と性質は CUDA から取得する。nvidia-smi が無い環境でも記録できる。
            environment.gpu_name_ = probe.name_;
            environment.gpu_integrated_ = probe.integrated_;
            environment.gpu_compute_capability_ =
                    std::to_string(probe.compute_capability_major_) + "." +
                    std::to_string(probe.compute_capability_minor_);
        }
    }
    // driver version と power mode は library から取得できないため外部 command を使う。
    environment.driver_version_ = read_command_line(
            "nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null");
    environment.cuda_toolkit_version_ = read_command_line(
            "nvcc --version 2>/dev/null | sed -n 's/.*release \\([0-9.]*\\).*/\\1/p'");
    // Jetson の power mode は測定条件に直結する。存在しない環境では空になる。
    environment.power_mode_ =
            read_command_line("nvpmodel -q 2>/dev/null | tr '\\n' ' ' | sed 's/  */ /g'");

    return environment;
}

bool measure_image(const std::string& image_path, const BenchmarkConfig& config,
                   MeasurementRecord* out_record, std::string* out_error) {
    if (out_record == nullptr || out_error == nullptr) {
        return false;
    }
    if (config.route_ != Route::kCpu) {
        // 未実装の経路を CPU で代替すると、測定結果が経路名と食い違う。
        *out_error = std::string("経路 ") + to_string(config.route_) +
                     " は未実装。現在測定できるのは CPU 経路のみ";
        return false;
    }
    if (config.latency_iterations_ <= 0) {
        *out_error = "latency_iterations は 1 以上である必要がある";
        return false;
    }

    aruco3cuda::reference::ReferenceResult first;
    if (!aruco3cuda::reference::detect_image(image_path, config.detector_, &first, out_error)) {
        return false;
    }

    MeasurementRecord record;
    record.image_path_ = image_path;
    record.image_sha256_ = first.image_sha256_;
    record.width_px_ = first.width_px_;
    record.height_px_ = first.height_px_;
    record.detection_count_ = first.detections_.size();
    record.fxfy_effective_ = first.fxfy_effective_;

    // warm-up。cache と分岐予測を定常状態へ寄せるため、測定区間から分離する。
    for (int i = 0; i < config.warmup_iterations_; ++i) {
        aruco3cuda::reference::ReferenceResult warmup;
        if (!aruco3cuda::reference::detect_image(image_path, config.detector_, &warmup,
                                                 out_error)) {
            return false;
        }
    }

    // 単一フレーム遅延。1 回ずつ独立に測る。
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(config.latency_iterations_));
    for (int i = 0; i < config.latency_iterations_; ++i) {
        aruco3cuda::reference::ReferenceResult iteration;
        const auto start = std::chrono::steady_clock::now();
        if (!aruco3cuda::reference::detect_image(image_path, config.detector_, &iteration,
                                                 out_error)) {
            return false;
        }
        const auto finish = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(finish - start).count());
    }
    if (!aruco3cuda::util::compute_statistics(samples, &record.end_to_end_ms_)) {
        *out_error = "統計を計算できない";
        return false;
    }
    if (config.save_all_samples_) {
        record.end_to_end_samples_ms_ = samples;
    }

    // throughput。連続処理の総時間から frame/s を求める。遅延とは別に測る。
    if (config.throughput_frames_ > 0) {
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < config.throughput_frames_; ++i) {
            aruco3cuda::reference::ReferenceResult iteration;
            if (!aruco3cuda::reference::detect_image(image_path, config.detector_, &iteration,
                                                     out_error)) {
                return false;
            }
        }
        const auto finish = std::chrono::steady_clock::now();
        const double elapsed_s = std::chrono::duration<double>(finish - start).count();
        if (elapsed_s > 0.0) {
            record.throughput_available_ = true;
            record.throughput_fps_ = static_cast<double>(config.throughput_frames_) / elapsed_s;
        }
    }

    *out_record = record;
    return true;
}

void write_environment_line(std::ostream& out, const EnvironmentRecord& environment) {
    JsonWriter writer(out, 0);
    writer.begin_object();
    writer.member_string("type", "environment");
    writer.member_int("schema_version", 1);
    writer.member_string("hostname", environment.hostname_);
    writer.member_string("os", environment.os_);
    writer.member_string("kernel", environment.kernel_);
    writer.member_string("architecture", environment.architecture_);
    writer.member_int("cpu_online_cores", environment.cpu_online_cores_);
    writer.member_string("opencv_version", environment.opencv_version_);
    writer.member_int("opencv_threads", environment.opencv_threads_);
    writer.member_string("cuda_toolkit", environment.cuda_toolkit_version_);
    writer.member_string("gpu_name", environment.gpu_name_);
    writer.member_string("gpu_compute_capability", environment.gpu_compute_capability_);
    writer.member_bool("gpu_integrated", environment.gpu_integrated_);
    writer.member_string("driver_version", environment.driver_version_);
    writer.member_string("power_mode", environment.power_mode_);
    writer.end_object();
    out << '\n';
}

void write_measurement_line(std::ostream& out, const BenchmarkConfig& config,
                            const MeasurementRecord& record) {
    JsonWriter writer(out, 0);
    writer.begin_object();
    writer.member_string("type", "measurement");
    writer.member_int("schema_version", 1);
    writer.member_string("route", to_string(config.route_));
    writer.member_string("memory_mode", to_string(config.memory_mode_));
    writer.member_string("image_path", record.image_path_);
    writer.member_string("image_sha256", record.image_sha256_);
    writer.member_int("width_px", record.width_px_);
    writer.member_int("height_px", record.height_px_);
    writer.member_int("detection_count", static_cast<long long>(record.detection_count_));

    writer.key("conditions");
    writer.begin_object();
    writer.member_int("warmup_iterations", config.warmup_iterations_);
    writer.member_int("latency_iterations", config.latency_iterations_);
    writer.member_int("throughput_frames", config.throughput_frames_);
    writer.member_bool("use_aruco3_detection", config.detector_.use_aruco3_detection_);
    writer.member_int("min_side_length_canonical_img",
                      config.detector_.min_side_length_canonical_img_px_);
    writer.member_double("min_marker_length_ratio_original_img",
                         static_cast<double>(
                                 config.detector_.min_marker_length_ratio_original_img_),
                         6);
    writer.member_double("fxfy_effective", record.fxfy_effective_, 6);
    writer.member_string("dictionary", config.detector_.dictionary_name_);
    writer.end_object();

    write_statistics(writer, "end_to_end", record.end_to_end_ms_);

    writer.key("kernel");
    if (record.kernel_time_available_) {
        writer.begin_object();
        writer.member_double("p50_ms", record.kernel_ms_.p50_, 4);
        writer.member_double("p95_ms", record.kernel_ms_.p95_, 4);
        writer.member_double("p99_ms", record.kernel_ms_.p99_, 4);
        writer.end_object();
    } else {
        // CPU 経路には kernel 時間が存在しない。0 で埋めず未測定を明示する。
        writer.value_null();
    }

    writer.key("throughput_fps");
    if (record.throughput_available_) {
        writer.value_double(record.throughput_fps_, 3);
    } else {
        writer.value_null();
    }

    if (!record.end_to_end_samples_ms_.empty()) {
        writer.key("end_to_end_samples_ms");
        writer.begin_array();
        for (const double sample : record.end_to_end_samples_ms_) {
            writer.value_double(sample, 4);
        }
        writer.end_array();
    }

    writer.end_object();
    out << '\n';
}

}  // namespace aruco3cuda::bench
