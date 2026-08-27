// SPDX-License-Identifier: Apache-2.0
#include "benchmark_harness.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "device_image.hpp"
#include "hybrid_detector.hpp"

#include <sched.h>  // sched_setaffinity は POSIX 拡張であり標準 header に無い
#include <sys/personality.h>  // ASLR の状態確認。標準 header には無い
#include <stdio.h>  // popen と pclose は POSIX であり <cstdio> の std 名前空間に無い
#include <sys/utsname.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <fstream>
#include <map>
#include <memory>
#include <ostream>
#include <ratio>
#include <string>
#include <utility>
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

/// popen が返す pipe を RAII で保持する。
///
/// popen / pclose を手動で対にすると、途中の早期 return や例外で子 process が
/// 残る。unique_ptr の deleter へ pclose を委ね、経路によらず必ず閉じる。
struct PipeCloser {
    void operator()(std::FILE* pipe) const noexcept {
        if (pipe != nullptr) {
            pclose(pipe);
        }
    }
};
using UniquePipe = std::unique_ptr<std::FILE, PipeCloser>;

UniquePipe open_command(const std::string& command) {
    return UniquePipe(popen(command.c_str(), "r"));
}

/// command を実行し標準出力を 1 行取得する。取得できない場合は空文字列を返す。
///
/// nvpmodel のように library 経由で取得できない環境情報を記録するために使用する。
/// 失敗しても測定自体は継続する。値が取れないことと空であることを
/// 呼出側が区別する必要はなく、いずれも「未取得」として扱う。
std::string read_command_line(const std::string& command) {
    std::array<char, 512> buffer{};
    const UniquePipe pipe = open_command(command);
    if (!pipe) {
        return std::string();
    }
    std::string result;
    if (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) != nullptr) {
        result = buffer.data();
    }
    while (!result.empty() &&
           (result.back() == '\n' || result.back() == '\r' || result.back() == ' ')) {
        result.pop_back();
    }
    return result;
}

/// command の標準出力を全行取得する。
///
/// 想定するのは query-platform-info.sh の数行の出力である。外部 command の
/// 出力を無制限に取り込まないよう行数の上限を設ける。
std::vector<std::string> read_command_lines(const std::string& command) {
    constexpr std::size_t kMaxLines = 256U;
    std::vector<std::string> lines;
    std::array<char, 512> buffer{};
    const UniquePipe pipe = open_command(command);
    if (!pipe) {
        return lines;
    }
    while (lines.size() < kMaxLines &&
           std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) != nullptr) {
        std::string line = buffer.data();
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
            line.pop_back();
        }
        if (!line.empty()) {
            lines.push_back(line);
        }
    }
    return lines;
}

/// query-platform-info.sh の key=value 出力を map へ読み込む。
///
/// 機種ごとに取得元が異なる項目をここで吸収する。script が無い環境では
/// 空の map を返し、呼出側は未取得として扱う。推測で埋めない。
std::map<std::string, std::string> read_platform_info() {
    std::map<std::string, std::string> info;
    for (const std::string& line : read_command_lines("query-platform-info.sh 2>/dev/null")) {
        const std::size_t separator = line.find('=');
        if (separator == std::string::npos || separator == 0U) {
            continue;
        }
        info[line.substr(0, separator)] = line.substr(separator + 1U);
    }
    return info;
}

/// 整数として解釈できる場合のみ値を返す。
bool parse_int_field(const std::map<std::string, std::string>& info, const std::string& key,
                     int* out) {
    const auto entry = info.find(key);
    if (entry == info.end() || entry->second.empty()) {
        return false;
    }
    try {
        std::size_t consumed = 0;
        const int value = std::stoi(entry->second, &consumed);
        if (consumed != entry->second.size()) {
            return false;
        }
        *out = value;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

std::string field_or_empty(const std::map<std::string, std::string>& info,
                           const std::string& key) {
    const auto entry = info.find(key);
    return entry == info.end() ? std::string() : entry->second;
}

/// /proc/cpuinfo から core 種別ごとの CPU 番号をまとめた文字列を作る。
///
/// 性能 core と効率 core が混在する機では、どの種別で測ったかが分からないと
/// 測定値を比較できない。marketing 名は既知のものだけ補い、未知の実装 ID は
/// そのまま記録する。推測で名前を付けない。
std::string read_cpu_topology() {
    static const std::map<std::string, std::string> kKnownParts = {
            {"0xd85", "Cortex-X925"}, {"0xd87", "Cortex-A725"}, {"0xd8e", "Cortex-A720"},
            {"0xd41", "Cortex-A78AE"}, {"0xd42", "Cortex-A78AE"},
    };
    std::ifstream info("/proc/cpuinfo");
    if (!info) {
        return std::string();
    }
    // 出現順を保つため vector で持つ。core 種別は数種類しかない。
    std::vector<std::pair<std::string, std::vector<int>>> groups;
    std::string line;
    int processor = -1;
    while (std::getline(info, line)) {
        const std::size_t separator = line.find(':');
        if (separator == std::string::npos) {
            continue;
        }
        std::string key = line.substr(0, separator);
        std::string value = line.substr(separator + 1);
        while (!key.empty() && (key.back() == ' ' || key.back() == '\t')) {
            key.pop_back();
        }
        while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) {
            value.erase(value.begin());
        }
        if (key == "processor") {
            try {
                processor = std::stoi(value);
            } catch (const std::exception&) {
                processor = -1;
            }
        } else if (key == "CPU part" && processor >= 0) {
            const auto known = kKnownParts.find(value);
            const std::string name = (known == kKnownParts.end()) ? value : known->second;
            auto entry = std::find_if(groups.begin(), groups.end(),
                                      [&name](const auto& g) { return g.first == name; });
            if (entry == groups.end()) {
                groups.emplace_back(name, std::vector<int>{processor});
            } else {
                entry->second.push_back(processor);
            }
        }
    }
    std::string result;
    for (const auto& group : groups) {
        if (!result.empty()) {
            result += ", ";
        }
        result += group.first + " x" + std::to_string(group.second.size());
    }
    return result;
}

/// 測定に使用する CPU を固定する。
///
/// @return 固定できた場合は使用した CPU 番号の文字列。固定しない場合は "unpinned"。
///         失敗した場合は理由を含む文字列を返し、測定自体は継続する。
std::string apply_cpu_affinity(const std::vector<int>& cpu_list) {
    if (cpu_list.empty()) {
        return "unpinned";
    }
    cpu_set_t mask;
    CPU_ZERO(&mask);
    std::string names;
    for (const int cpu : cpu_list) {
        if (cpu < 0 || cpu >= CPU_SETSIZE) {
            return "invalid-cpu-" + std::to_string(cpu);
        }
        CPU_SET(cpu, &mask);
        if (!names.empty()) {
            names += ",";
        }
        names += std::to_string(cpu);
    }
    if (sched_setaffinity(0, sizeof(mask), &mask) != 0) {
        return "affinity-failed:" + names;
    }
    return names;
}

/// ASLR の状態を調べる。
///
/// process 自身の personality と system 設定の両方を見る。setarch -R などで
/// process 単位に無効化されている場合、system 設定だけでは判別できない。
///
/// @return "disabled(process)"、"disabled(system)"、"enabled"、
///         または判別できない場合は空文字列。
std::string read_address_randomization() {
    const int persona = personality(0xffffffffU);
    if (persona >= 0 && (static_cast<unsigned int>(persona) & ADDR_NO_RANDOMIZE) != 0U) {
        return "disabled(process)";
    }
    std::ifstream setting("/proc/sys/kernel/randomize_va_space");
    if (!setting) {
        return std::string();
    }
    std::string value;
    if (!(setting >> value)) {
        return std::string();
    }
    return value == "0" ? "disabled(system)" : "enabled";
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

/// 結果 JSONL の schema 版。key を追加または改名したら上げる。
///
/// 3 での変更: measurement 行へ stages を追加した。あわせて CPU 経路の
/// 測定区間から画像の読み込みと checksum を外した。version 2 以前の
/// 結果と混ぜると、同じ key が違う区間を指すことになる。
constexpr int kSchemaVersion = 3;

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
    environment.cpu_topology_ = read_cpu_topology();
    // 測定前に CPU を固定する。以降の測定はこの core 集合で行われる。
    environment.cpu_affinity_ = apply_cpu_affinity(config.cpu_affinity_);
    environment.address_randomization_ = read_address_randomization();
    // sysconf は失敗時に -1 を返す。core 数として -1 を記録すると
    // 測定条件の解釈を誤るため、取得できない場合は 0 のままとする。
    const long online_cores = sysconf(_SC_NPROCESSORS_ONLN);
    if (online_cores > 0) {
        environment.cpu_online_cores_ = static_cast<int>(online_cores);
    }

    const aruco3cuda::reference::ReferenceEnvironment detector_environment =
            aruco3cuda::reference::collect_environment(config.detector_);
    environment.opencv_version_ = detector_environment.opencv_version_;
    environment.opencv_threads_ = detector_environment.opencv_threads_;

    // GPU 情報。device が無い環境でも測定自体は成立するため、失敗を致命的に扱わない。
    // ただし無言では継続しない。CUDA 側の失敗は記録し、環境情報から
    // GPU の項目が欠けている理由を後から追えるようにする。
    int device_count = 0;
    const aruco3cuda::Status count_status = aruco3cuda::device_count(&device_count);
    if (count_status != aruco3cuda::Status::kOk) {
        environment.gpu_probe_error_ = std::string("device_count が失敗した: ") +
                                       aruco3cuda::to_string(count_status) + " " +
                                       aruco3cuda::last_cuda_error_message();
    } else if (device_count == 0) {
        environment.gpu_probe_error_ = "CUDA device が 1 つも見つからない";
    }
    if (count_status == aruco3cuda::Status::kOk && device_count > 0) {
        aruco3cuda::DeviceProbeResult probe;
        const aruco3cuda::Status probe_status = aruco3cuda::probe_device(0, &probe);
        if (probe_status != aruco3cuda::Status::kOk) {
            environment.gpu_probe_error_ = std::string("probe_device が失敗した: ") +
                                           aruco3cuda::to_string(probe_status) + " " +
                                           aruco3cuda::last_cuda_error_message();
        }
        if (probe_status == aruco3cuda::Status::kOk) {
            // device 名と性質は CUDA から取得する。nvidia-smi が無い環境でも記録できる。
            environment.gpu_name_ = probe.name_;
            environment.gpu_integrated_ = probe.integrated_;
            environment.gpu_compute_capability_ =
                    std::to_string(probe.compute_capability_major_) + "." +
                    std::to_string(probe.compute_capability_minor_);
        }
    }
    environment.cuda_toolkit_version_ = read_command_line(
            "nvcc --version 2>/dev/null | sed -n 's/.*release \\([0-9.]*\\).*/\\1/p'");

    // driver version、power mode、clock、L4T release は library から取得できない。
    // 取得元が機種ごとに異なるため query-platform-info.sh が差を吸収する。
    const std::map<std::string, std::string> platform_info = read_platform_info();
    environment.driver_version_ = field_or_empty(platform_info, "driver_version");
    environment.platform_release_ = field_or_empty(platform_info, "platform_release");
    environment.platform_model_ = field_or_empty(platform_info, "platform_model");
    environment.power_mode_ = field_or_empty(platform_info, "power_mode");
    environment.gpu_clock_available_ =
            parse_int_field(platform_info, "gpu_max_clock_mhz", &environment.gpu_max_clock_mhz_);
    parse_int_field(platform_info, "gpu_current_clock_mhz", &environment.gpu_current_clock_mhz_);

    return environment;
}

namespace {

/// 測定の段階。step へ渡し、どの段階の反復かを伝える。
///
/// 経路側が段階ごとの内訳を記録する場合、どの反復を数えるかで分布が変わる。
/// warm-up と throughput を混ぜると、遅延の分位点と直接比べられなくなる。
enum class Phase : int {
    kWarmup,
    kLatency,
    kThroughput,
};

/// 1 frame 分の処理を反復し、遅延と throughput を測る。
///
/// 経路ごとに違うのは 1 frame の中身だけである。warm-up の分離、分位点の
/// 求め方、throughput の測り方は評価計画が定めた共通の手順であり、経路を
/// 増やすたびに書き写すと手順がずれる。
///
/// step は Phase を受け取り 1 frame を処理し、失敗時に out_error を埋めて
/// false を返すこと。
template <typename Step>
bool measure_iterations(const BenchmarkConfig& config, Step step, MeasurementRecord* record,
                        std::string* out_error) {
    // warm-up。cache と分岐予測を定常状態へ寄せるため、測定区間から分離する。
    for (int i = 0; i < config.warmup_iterations_; ++i) {
        if (!step(Phase::kWarmup)) {
            return false;
        }
    }

    // 単一フレーム遅延。1 回ずつ独立に測る。
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(config.latency_iterations_));
    for (int i = 0; i < config.latency_iterations_; ++i) {
        const auto start = std::chrono::steady_clock::now();
        if (!step(Phase::kLatency)) {
            return false;
        }
        const auto finish = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(finish - start).count());
    }
    if (!aruco3cuda::util::compute_statistics(samples, &record->end_to_end_ms_)) {
        *out_error = "統計を計算できない";
        return false;
    }
    if (config.save_all_samples_) {
        record->end_to_end_samples_ms_ = samples;
    }

    // throughput。連続処理の総時間から frame/s を求める。遅延とは別に測る。
    if (config.throughput_frames_ > 0) {
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < config.throughput_frames_; ++i) {
            if (!step(Phase::kThroughput)) {
                return false;
            }
        }
        const auto finish = std::chrono::steady_clock::now();
        const double elapsed_s = std::chrono::duration<double>(finish - start).count();
        if (elapsed_s > 0.0) {
            record->throughput_available_ = true;
            record->throughput_fps_ = static_cast<double>(config.throughput_frames_) / elapsed_s;
        }
    }
    return true;
}

/// 測定条件の境界を検証する。
///
/// 負値は反復 loop を素通りし、標本 0 件のまま「測定した」ことになってしまう。
bool validate_iteration_counts(const BenchmarkConfig& config, std::string* out_error) {
    if (config.latency_iterations_ <= 0) {
        *out_error = "latency_iterations は 1 以上である必要がある: " +
                     std::to_string(config.latency_iterations_);
        return false;
    }
    if (config.warmup_iterations_ < 0) {
        *out_error = "warmup_iterations は 0 以上である必要がある: " +
                     std::to_string(config.warmup_iterations_);
        return false;
    }
    if (config.throughput_frames_ < 0) {
        *out_error = "throughput_frames は 0 以上である必要がある: " +
                     std::to_string(config.throughput_frames_);
        return false;
    }
    return true;
}

/// CPU 基準経路を測定する。
///
/// 画像の読み込みと checksum は初期化側へ寄せ、測定区間は検出だけにする。
/// 読み込みを含めると、合成 corpus の 1280x720 PNG では測定区間の 6 割から
/// 8 割が PNG の復号になり、検出時間の比較にならない。
bool measure_cpu(const std::string& image_path, const BenchmarkConfig& config,
                 MeasurementRecord* record, std::string* out_error) {
    aruco3cuda::reference::ReferenceDetector detector;
    if (!detector.initialize(image_path, config.detector_, out_error)) {
        return false;
    }
    const aruco3cuda::reference::ReferenceResult& metadata = detector.metadata();
    record->image_path_ = image_path;
    record->image_sha256_ = metadata.image_sha256_;
    record->width_px_ = metadata.width_px_;
    record->height_px_ = metadata.height_px_;
    record->fxfy_effective_ = metadata.fxfy_effective_;

    aruco3cuda::reference::ReferenceResult result;
    if (!detector.detect(&result, out_error)) {
        return false;
    }
    record->detection_count_ = result.detections_.size();

    return measure_iterations(
            config, [&](Phase) { return detector.detect(&result, out_error); }, record,
            out_error);
}

/// hybrid 経路を測定する。
///
/// memory 種別で測定区間が変わる。kDevice は画像が既に device にある想定で
/// 転送を測定区間の外へ置き、kHostPageable は host 入力を毎 frame 転送する。
/// 前者は camera から GPU へ直接入る構成の上限、後者は CPU 経路と同じく
/// host の画像から始める場合の値である。
bool measure_hybrid(const std::string& image_path, const BenchmarkConfig& config,
                    MeasurementRecord* record, std::string* out_error) {
    if (config.memory_mode_ != MemoryMode::kDevice &&
        config.memory_mode_ != MemoryMode::kHostPageable) {
        *out_error = std::string("hybrid 経路が対応する memory 種別は M-Device と M-Pageable。"
                                 "指定された種別: ") +
                     to_string(config.memory_mode_);
        return false;
    }

    // 画像の読み込みと checksum は測定区間の外で 1 度だけ行う。
    aruco3cuda::reference::ReferenceDetector loader;
    if (!loader.initialize(image_path, config.detector_, out_error)) {
        return false;
    }
    const aruco3cuda::reference::ReferenceResult& metadata = loader.metadata();
    const cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        *out_error = "画像を読み込めない: " + image_path;
        return false;
    }

    record->image_path_ = image_path;
    record->image_sha256_ = metadata.image_sha256_;
    record->width_px_ = image.cols;
    record->height_px_ = image.rows;
    record->fxfy_effective_ = metadata.fxfy_effective_;

    aruco3cuda::hybrid::DeviceImage device;
    std::string message;
    if (device.reserve(image.cols, image.rows, &message) != aruco3cuda::Status::kOk) {
        *out_error = message;
        return false;
    }
    const auto upload = [&]() {
        return device.upload(image.data, image.cols, image.rows,
                             static_cast<std::size_t>(image.step), &message) ==
               aruco3cuda::Status::kOk;
    };
    if (!upload()) {
        *out_error = message;
        return false;
    }

    aruco3cuda::hybrid::HybridDetector detector;
    if (detector.initialize(config.cuda_detector_, config.detector_.dictionary_name_, image.cols,
                            image.rows, &message) != aruco3cuda::Status::kOk) {
        *out_error = "hybrid 検出器を初期化できない: " + message;
        return false;
    }

    const bool upload_inside = config.memory_mode_ == MemoryMode::kHostPageable;
    aruco3cuda::hybrid::HybridResult result;
    std::vector<double> gpu_samples;
    std::vector<double> cpu_samples;
    gpu_samples.reserve(static_cast<std::size_t>(config.latency_iterations_));
    cpu_samples.reserve(static_cast<std::size_t>(config.latency_iterations_));

    const auto step = [&](Phase phase) {
        if (upload_inside && !upload()) {
            *out_error = message;
            return false;
        }
        const aruco3cuda::Status status = detector.detect(device.view(), &result, &message);
        if (status != aruco3cuda::Status::kOk) {
            *out_error = "hybrid 検出に失敗した: " + message;
            return false;
        }
        // 段階の内訳は遅延測定の反復だけから求める。warm-up と throughput を
        // 混ぜると、end_to_end の分位点と直接比べられる保証が無くなる。
        if (phase == Phase::kLatency) {
            gpu_samples.push_back(result.gpu_ms_);
            cpu_samples.push_back(result.cpu_ms_);
        }
        return true;
    };

    if (!step(Phase::kWarmup)) {
        return false;
    }
    record->detection_count_ = result.detections_.size();

    if (!measure_iterations(config, step, record, out_error)) {
        return false;
    }
    record->stage_times_available_ =
            aruco3cuda::util::compute_statistics(gpu_samples, &record->gpu_stage_ms_) &&
            aruco3cuda::util::compute_statistics(cpu_samples, &record->cpu_stage_ms_);
    return true;
}

}  // namespace

aruco3cuda::DetectorConfig cuda_config_from_reference(
        const aruco3cuda::reference::ReferenceConfig& config) {
    aruco3cuda::DetectorConfig result;
    result.adaptive_thresh_win_size_min_px_ = config.adaptive_thresh_win_size_min_px_;
    result.adaptive_thresh_win_size_max_px_ = config.adaptive_thresh_win_size_max_px_;
    result.adaptive_thresh_win_size_step_px_ = config.adaptive_thresh_win_size_step_px_;
    result.adaptive_thresh_constant_ = config.adaptive_thresh_constant_;
    result.min_marker_perimeter_rate_ = config.min_marker_perimeter_rate_;
    result.max_marker_perimeter_rate_ = config.max_marker_perimeter_rate_;
    result.polygonal_approx_accuracy_rate_ = config.polygonal_approx_accuracy_rate_;
    result.min_corner_distance_rate_ = config.min_corner_distance_rate_;
    result.min_distance_to_border_px_ = config.min_distance_to_border_px_;
    result.min_marker_distance_rate_ = config.min_marker_distance_rate_;
    result.marker_border_bits_ = config.marker_border_bits_;
    result.perspective_remove_pixel_per_cell_ = config.perspective_remove_pixel_per_cell_;
    result.perspective_remove_ignored_margin_per_cell_ =
            config.perspective_remove_ignored_margin_per_cell_;
    result.max_erroneous_bits_in_border_rate_ = config.max_erroneous_bits_in_border_rate_;
    result.min_otsu_std_dev_ = config.min_otsu_std_dev_;
    result.error_correction_rate_ = config.error_correction_rate_;
    result.use_aruco3_detection_ = config.use_aruco3_detection_;
    result.min_side_length_canonical_img_px_ = config.min_side_length_canonical_img_px_;
    result.min_marker_length_ratio_original_img_ = config.min_marker_length_ratio_original_img_;
    result.corner_refine_method_ = config.use_corner_subpix_refinement_
                                           ? aruco3cuda::CornerRefineMethod::kSubpix
                                           : aruco3cuda::CornerRefineMethod::kNone;
    result.corner_refinement_win_size_px_ = config.corner_refinement_win_size_px_;
    result.relative_corner_refinement_win_size_ = config.relative_corner_refinement_win_size_;
    result.corner_refinement_max_iterations_ = config.corner_refinement_max_iterations_;
    result.corner_refinement_min_accuracy_px_ = config.corner_refinement_min_accuracy_px_;
    return result;
}

bool measure_image(const std::string& image_path, const BenchmarkConfig& config,
                   MeasurementRecord* out_record, std::string* out_error) {
    if (out_record == nullptr || out_error == nullptr) {
        return false;
    }
    if (!validate_iteration_counts(config, out_error)) {
        return false;
    }

    MeasurementRecord record;
    bool ok = false;
    switch (config.route_) {
        case Route::kCpu:
            ok = measure_cpu(image_path, config, &record, out_error);
            break;
        case Route::kHybrid:
            ok = measure_hybrid(image_path, config, &record, out_error);
            break;
        case Route::kCudaEndToEnd:
        case Route::kCudaResident:
        default:
            // 未実装の経路を CPU で代替すると、測定結果が経路名と食い違う。
            *out_error = std::string("経路 ") + to_string(config.route_) +
                         " は未実装。現在測定できるのは CPU 経路と Hybrid 経路";
            return false;
    }
    if (!ok) {
        return false;
    }
    *out_record = record;
    return true;
}

void write_environment_line(std::ostream& out, const EnvironmentRecord& environment) {
    JsonWriter writer(out, 0);
    writer.begin_object();
    writer.member_string("type", "environment");
    // version 2: cpu_topology、cpu_affinity、address_randomization、
    // gpu clock、platform 情報を追加した。測定条件の再現に必要なため。
    writer.member_int("schema_version", kSchemaVersion);
    writer.member_string("hostname", environment.hostname_);
    writer.member_string("os", environment.os_);
    writer.member_string("kernel", environment.kernel_);
    writer.member_string("architecture", environment.architecture_);
    writer.member_int("cpu_online_cores", environment.cpu_online_cores_);
    writer.member_string("cpu_topology", environment.cpu_topology_);
    writer.member_string("cpu_affinity", environment.cpu_affinity_);
    writer.member_string("address_randomization", environment.address_randomization_);
    writer.member_string("opencv_version", environment.opencv_version_);
    writer.member_int("opencv_threads", environment.opencv_threads_);
    writer.member_string("cuda_toolkit", environment.cuda_toolkit_version_);
    writer.member_string("gpu_name", environment.gpu_name_);
    writer.member_string("gpu_compute_capability", environment.gpu_compute_capability_);
    writer.member_bool("gpu_integrated", environment.gpu_integrated_);
    writer.member_string("driver_version", environment.driver_version_);
    writer.member_string("platform_release", environment.platform_release_);
    writer.member_string("platform_model", environment.platform_model_);
    writer.member_string("power_mode", environment.power_mode_);
    // GPU 情報が欠けている場合の理由。空なら取得に成功している。
    writer.member_string("gpu_probe_error", environment.gpu_probe_error_);
    // clock は測定条件に直結する。取得できない場合に 0 を書くと
    // 「clock が 0」と誤読されるため null とする。
    writer.key("gpu_max_clock_mhz");
    if (environment.gpu_clock_available_) {
        writer.value_int(environment.gpu_max_clock_mhz_);
    } else {
        writer.value_null();
    }
    writer.key("gpu_current_clock_mhz");
    if (environment.gpu_current_clock_mhz_ > 0) {
        writer.value_int(environment.gpu_current_clock_mhz_);
    } else {
        writer.value_null();
    }
    writer.end_object();
    out << '\n';
}

void write_measurement_line(std::ostream& out, const BenchmarkConfig& config,
                            const MeasurementRecord& record) {
    JsonWriter writer(out, 0);
    writer.begin_object();
    writer.member_string("type", "measurement");
    writer.member_int("schema_version", kSchemaVersion);
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

    // 段階時間。CPU 経路には存在しないため未測定を明示する。
    writer.key("stages");
    if (record.stage_times_available_) {
        writer.begin_object();
        write_statistics(writer, "gpu", record.gpu_stage_ms_);
        write_statistics(writer, "cpu", record.cpu_stage_ms_);
        writer.end_object();
    } else {
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
