// SPDX-License-Identifier: Apache-2.0
#include "benchmark_harness.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/detections.hpp"
#include "aruco3cuda/detector.hpp"
#include "aruco3cuda/dictionary.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"
#include "device_image.hpp"
#include "hybrid_detector.hpp"

#include <sched.h>  // sched_setaffinity is a POSIX extension and is not in a standard header
#include <stdio.h>  // popen and pclose are POSIX and are not in the std namespace of <cstdio>
#include <sys/personality.h>  // Checking the ASLR state. Not in a standard header
#include <sys/utsname.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

/// Hold the pipe returned by popen with RAII.
///
/// Pairing popen and pclose by hand leaves the child process behind on an early
/// return or an exception in between. Delegating pclose to the deleter of a
/// unique_ptr closes it on every path.
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

/// Run a command and take one line from its standard output. Returns an empty
/// string when nothing can be obtained.
///
/// Used to record environment information that cannot be obtained through a
/// library, such as nvpmodel. A failure does not stop the measurement. The
/// caller has no need to distinguish "no value" from "empty"; both are treated
/// as "not obtained".
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

/// Take every line of a command's standard output.
///
/// The expected case is the few lines printed by query-platform-info.sh. A line
/// limit keeps the output of an external command from being taken in without
/// bound.
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

/// Read the key=value output of query-platform-info.sh into a map.
///
/// Absorbs here the items whose source differs from machine to machine. In an
/// environment without the script, an empty map is returned and the caller
/// treats the items as not obtained. Nothing is filled in by guesswork.
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

/// Return the value only when it can be interpreted as an integer.
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

std::string field_or_empty(const std::map<std::string, std::string>& info, const std::string& key) {
    const auto entry = info.find(key);
    return entry == info.end() ? std::string() : entry->second;
}

/// Read the maximum frequency of each CPU from sysfs and use it in place of the
/// core type.
///
/// An x86 hybrid CPU has no implementation ID in /proc/cpuinfo equivalent to the
/// aarch64 "CPU part", and the model name is the same for every core. The
/// performance and efficiency cores differ in maximum frequency, so that is used
/// in place of the type. No name such as P-core or E-core is guessed from the
/// frequency.
///
/// @param cpu_count Number of CPUs to scan.
/// @return The CPU numbers grouped by frequency (kHz), keeping the order of
///         appearance. Empty when nothing can be read.
std::vector<std::pair<long long, std::vector<int>>> read_frequency_groups(int cpu_count) {
    std::vector<std::pair<long long, std::vector<int>>> groups;
    for (int cpu = 0; cpu < cpu_count; ++cpu) {
        const std::string path =
                "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cpufreq/cpuinfo_max_freq";
        std::ifstream file(path);
        long long value = 0;
        if (!file || !(file >> value) || value <= 0) {
            continue;
        }
        auto entry = std::find_if(groups.begin(), groups.end(),
                                  [value](const auto& g) { return g.first == value; });
        if (entry == groups.end()) {
            groups.emplace_back(value, std::vector<int>{cpu});
        } else {
            entry->second.push_back(cpu);
        }
    }
    return groups;
}

/// Convert kHz into a GHz string, to two decimal places.
std::string format_ghz(long long khz) {
    // Two decimal places, truncated rather than rounded. Keep enough digits that
    // rounding for display does not erase the distinction between core types.
    const long long hundredths = khz / 10000;
    const long long integer_part = hundredths / 100;
    const long long fraction = hundredths % 100;
    std::string text = std::to_string(integer_part) + ".";
    if (fraction < 10) {
        text += "0";
    }
    return text + std::to_string(fraction) + "GHz";
}

/// Build a string that groups the CPU numbers by core type.
///
/// On a machine that mixes performance and efficiency cores, measured values
/// cannot be compared without knowing which type they were taken on. On aarch64
/// the classification uses the "CPU part" field of /proc/cpuinfo, and a
/// marketing name is supplied only for the known values. An unknown
/// implementation ID is recorded as is; no name is guessed for it.
///
/// x86 has no "CPU part", and this path alone yields an empty string. It did
/// come out empty on a machine with an Intel Core Ultra 7 265 and a GeForce RTX
/// 5070 Ti, and the environment information test failed. On x86 the
/// classification falls back to the maximum frequency.
/// Collapses an ascending run of processor numbers into the form "0-4,10-14".
///
/// Consecutive numbers are folded into a range. Listing 20 of them makes the
/// environment information hard to read.
std::string format_processor_ranges(const std::vector<int>& processors) {
    std::string result;
    for (std::size_t i = 0; i < processors.size();) {
        std::size_t j = i;
        while (j + 1 < processors.size() && processors[j + 1] == processors[j] + 1) {
            ++j;
        }
        if (!result.empty()) {
            result += ",";
        }
        result += std::to_string(processors[i]);
        if (j > i) {
            result += "-" + std::to_string(processors[j]);
        }
        i = j + 1;
    }
    return result;
}

std::string read_cpu_topology() {
    static const std::map<std::string, std::string> kKnownParts = {
            {"0xd85", "Cortex-X925"},  {"0xd87", "Cortex-A725"},  {"0xd8e", "Cortex-A720"},
            {"0xd41", "Cortex-A78AE"}, {"0xd42", "Cortex-A78AE"},
    };
    std::ifstream info("/proc/cpuinfo");
    if (!info) {
        return std::string();
    }
    // Held in a vector to keep the order of appearance. There are only a few core types.
    std::vector<std::pair<std::string, std::vector<int>>> groups;
    std::string model_name;
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
        } else if (key == "model name" && model_name.empty()) {
            model_name = value;
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
    if (!groups.empty()) {
        // Also keep the range of processor numbers for each core type. With the
        // count alone, the results cannot be checked against which core the
        // measurement ran on (whether --cpu-list named performance or efficiency
        // cores). The CPU route differs by roughly a factor of 1.6 between the
        // two, so without this correspondence the measurement conditions cannot
        // be verified.
        std::string result;
        for (const auto& group : groups) {
            if (!result.empty()) {
                result += ", ";
            }
            result += group.first + " x" + std::to_string(group.second.size());
            result += " (cpu " + format_processor_ranges(group.second) + ")";
        }
        return result;
    }

    // An environment without "CPU part". Classify by maximum frequency.
    const std::vector<std::pair<long long, std::vector<int>>> by_frequency =
            read_frequency_groups(processor + 1);
    if (by_frequency.empty()) {
        // The frequency cannot be read either. At least keep the model name.
        return model_name;
    }
    std::string result = model_name.empty() ? std::string("(unknown CPU)") : model_name;
    result += ":";
    for (const auto& group : by_frequency) {
        result += " " + format_ghz(group.first) + " x" + std::to_string(group.second.size()) + ",";
    }
    result.pop_back();
    return result;
}

/// Pin the CPUs used for the measurement.
///
/// @return The string of CPU numbers used when pinning succeeded, or "unpinned"
///         when no pinning was requested. On failure a string containing the
///         reason is returned and the measurement continues.
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

/// Determine the state of ASLR.
///
/// Looks both at the personality of the process itself and at the system
/// setting. When it has been disabled per process, for instance with setarch -R,
/// the system setting alone cannot tell.
///
/// @return "disabled(process)", "disabled(system)", "enabled", or an empty
///         string when it cannot be determined.
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

/// Schema version of the result JSONL. Raise it whenever a key is added or
/// renamed, or when the meaning or the format of a recorded value changes.
///
/// Change in 3: stages was added to the measurement line, and image loading and
/// the checksum were taken out of the measured interval of the CPU route. Mixing
/// these with results from version 2 or earlier would make the same key refer to
/// a different interval.
/// Change in 4: startup was added to the measurement line.
/// Change in 5: cuda_toolkit carries the patch version, taken from the V token
/// of nvcc --version, where it used to stop at the minor version. No key was
/// added or renamed and the measured intervals are identical to 4, so 4 and 5
/// may be aggregated together; see SUPPORTED_SCHEMA_VERSIONS in aggregate.py.
constexpr int kSchemaVersion = 5;

void write_statistics(JsonWriter& writer, const std::string& name, const SampleStatistics& stats) {
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
    // Pin the CPUs before measuring. Every measurement below runs on this core set.
    environment.cpu_affinity_ = apply_cpu_affinity(config.cpu_affinity_);
    environment.address_randomization_ = read_address_randomization();
    // sysconf returns -1 on failure. Recording -1 as a core count would lead to
    // a wrong reading of the measurement conditions, so it is left at 0 when it
    // cannot be obtained.
    const long online_cores = sysconf(_SC_NPROCESSORS_ONLN);
    if (online_cores > 0) {
        environment.cpu_online_cores_ = static_cast<int>(online_cores);
    }

    const aruco3cuda::reference::ReferenceEnvironment detector_environment =
            aruco3cuda::reference::collect_environment(config.detector_);
    environment.opencv_version_ = detector_environment.opencv_version_;
    environment.opencv_threads_ = detector_environment.opencv_threads_;

    // GPU information. The measurement itself still holds in an environment with
    // no device, so a failure is not treated as fatal. It is not passed over
    // silently either: a CUDA-side failure is recorded so that the reason a GPU
    // item is missing can be traced afterwards from the environment information.
    // The first CUDA call implicitly creates the context. It happens once per
    // process and does not show up in the per-route measurements. For a one-shot
    // detection it is hundreds of times the detection itself, so it is recorded
    // as environment information.
    int device_count = 0;
    const auto context_start = std::chrono::steady_clock::now();
    const aruco3cuda::Status count_status = aruco3cuda::device_count(&device_count);
    const auto context_finish = std::chrono::steady_clock::now();
    if (count_status == aruco3cuda::Status::kOk && device_count > 0) {
        environment.cuda_context_ms_ =
                std::chrono::duration<double, std::milli>(context_finish - context_start).count();
    }
    if (count_status != aruco3cuda::Status::kOk) {
        environment.gpu_probe_error_ = std::string("device_count failed: ") +
                                       aruco3cuda::to_string(count_status) + " " +
                                       aruco3cuda::last_cuda_error_message();
    } else if (device_count == 0) {
        environment.gpu_probe_error_ = "no CUDA device was found";
    }
    if (count_status == aruco3cuda::Status::kOk && device_count > 0) {
        aruco3cuda::DeviceProbeResult probe;
        const aruco3cuda::Status probe_status = aruco3cuda::probe_device(0, &probe);
        if (probe_status != aruco3cuda::Status::kOk) {
            environment.gpu_probe_error_ = std::string("probe_device failed: ") +
                                           aruco3cuda::to_string(probe_status) + " " +
                                           aruco3cuda::last_cuda_error_message();
        }
        if (probe_status == aruco3cuda::Status::kOk) {
            // The device name and properties come from CUDA, so they can be
            // recorded even in an environment without nvidia-smi.
            environment.gpu_name_ = probe.name_;
            environment.gpu_integrated_ = probe.integrated_;
            environment.gpu_compute_capability_ = std::to_string(probe.compute_capability_major_) +
                                                  "." +
                                                  std::to_string(probe.compute_capability_minor_);
        }
    }
    // The V token carries the patch version; the release field stops at the
    // minor version. Two machines can run the same minor version and different
    // patch versions, which is not visible in the record otherwise. The V
    // expression has to come first: it replaces the pattern space, so the
    // release expression that follows only fires when there was no V token to
    // find. read_command_line keeps the first line of output and drops the
    // rest, so printing twice would hide the answer behind the fallback.
    environment.cuda_toolkit_version_ = read_command_line(
            "nvcc --version 2>/dev/null | "
            "sed -n 's/.*, V\\([0-9][0-9.]*\\).*/\\1/p; "
            "s/.*release \\([0-9][0-9.]*\\).*/\\1/p'");

    // The driver version, the power mode, the clocks and the L4T release cannot
    // be obtained from a library. Their source differs from machine to machine,
    // so query-platform-info.sh absorbs the difference.
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

/// Phase of the measurement. Passed to step to say which phase an iteration
/// belongs to.
///
/// When a route records a per-stage breakdown, the distribution depends on which
/// iterations are counted. Mixing warm-up and throughput in makes it impossible
/// to compare directly against the latency percentiles.
enum class Phase : int {
    kWarmup,
    kLatency,
    kThroughput,
};

/// Repeat the processing of one frame and measure latency and throughput.
///
/// Only the contents of a single frame differ from route to route. Separating
/// warm-up, computing the percentiles and measuring throughput are the common
/// procedure laid down by the evaluation plan; copying it out again for every
/// new route would let the procedure drift.
///
/// step takes a Phase, processes one frame, and on failure must fill in
/// out_error and return false.
template <typename Step>
bool measure_iterations(const BenchmarkConfig& config, Step step, MeasurementRecord* record,
                        std::string* out_error) {
    // Warm-up. Separated from the measured interval so that the caches and the
    // branch predictors settle into their steady state.
    for (int i = 0; i < config.warmup_iterations_; ++i) {
        if (!step(Phase::kWarmup)) {
            return false;
        }
    }

    // Single-frame latency. Measured one iteration at a time, independently.
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
        *out_error = "cannot compute the statistics";
        return false;
    }
    if (config.save_all_samples_) {
        record->end_to_end_samples_ms_ = samples;
    }

    // Throughput. Frames per second derived from the total time of continuous
    // processing. Measured separately from latency.
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

/// Validate the bounds of the measurement conditions.
///
/// A negative value slips straight through the iteration loop and would count as
/// "measured" with zero samples.
bool validate_iteration_counts(const BenchmarkConfig& config, std::string* out_error) {
    if (config.latency_iterations_ <= 0) {
        *out_error = "latency_iterations must be 1 or more: " +
                     std::to_string(config.latency_iterations_);
        return false;
    }
    if (config.warmup_iterations_ < 0) {
        *out_error = "warmup_iterations must be 0 or more: " +
                     std::to_string(config.warmup_iterations_);
        return false;
    }
    if (config.throughput_frames_ < 0) {
        *out_error = "throughput_frames must be 0 or more: " +
                     std::to_string(config.throughput_frames_);
        return false;
    }
    return true;
}

/// Measure the CPU baseline route.
///
/// Image loading and the checksum are moved into initialization so that the
/// measured interval covers detection alone. Including the loading would make
/// PNG decoding 60 to 80 percent of the measured interval for the 1280x720 PNGs
/// of the synthetic corpus, which is no longer a comparison of detection time.
bool measure_cpu(const std::string& image_path, const BenchmarkConfig& config,
                 MeasurementRecord* record, std::string* out_error) {
    // The CPU route has no memory kind. Accepting one would line up, in the
    // aggregation, a measurement unrelated to that kind as its result.
    if (config.memory_mode_ != MemoryMode::kNotApplicable) {
        *out_error =
                std::string("the CPU route supports only the N/A memory kind; requested: ") +
                to_string(config.memory_mode_);
        return false;
    }
    // The startup cost appears only once. Rebuild the instance and measure the
    // first image.
    const auto setup_start = std::chrono::steady_clock::now();
    aruco3cuda::reference::ReferenceDetector detector;
    if (!detector.initialize(image_path, config.detector_, out_error)) {
        return false;
    }
    const auto first_start = std::chrono::steady_clock::now();
    aruco3cuda::reference::ReferenceResult result;
    if (!detector.detect(&result, out_error)) {
        return false;
    }
    const auto first_finish = std::chrono::steady_clock::now();
    record->first_frame_ms_ =
            std::chrono::duration<double, std::milli>(first_finish - first_start).count();
    record->time_to_first_result_ms_ =
            std::chrono::duration<double, std::milli>(first_finish - setup_start).count();

    const aruco3cuda::reference::ReferenceResult& metadata = detector.metadata();
    record->image_path_ = image_path;
    record->image_sha256_ = metadata.image_sha256_;
    record->width_px_ = metadata.width_px_;
    record->height_px_ = metadata.height_px_;
    record->fxfy_effective_ = metadata.fxfy_effective_;
    record->detection_count_ = result.detections_.size();

    return measure_iterations(
            config, [&](Phase) { return detector.detect(&result, out_error); }, record, out_error);
}

/// Measure the hybrid route.
///
/// The measured interval depends on the memory kind. kDevice assumes the image
/// is already on the device and places the transfer outside the interval, while
/// kHostPageable transfers the host input every frame. The former is the upper
/// bound for a configuration where the camera feeds the GPU directly; the latter
/// is the value for starting from a host image, as the CPU route does.
bool measure_hybrid(const std::string& image_path, const BenchmarkConfig& config,
                    MeasurementRecord* record, std::string* out_error) {
    if (config.memory_mode_ != MemoryMode::kDevice &&
        config.memory_mode_ != MemoryMode::kHostPageable) {
        *out_error = std::string("the hybrid route supports only the M-Device and M-Pageable "
                                 "memory kinds; requested: ") +
                     to_string(config.memory_mode_);
        return false;
    }

    // Image loading and the checksum happen once, outside the measured interval.
    aruco3cuda::reference::ReferenceDetector loader;
    if (!loader.initialize(image_path, config.detector_, out_error)) {
        return false;
    }
    const aruco3cuda::reference::ReferenceResult& metadata = loader.metadata();
    const cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        *out_error = "cannot load image: " + image_path;
        return false;
    }

    record->image_path_ = image_path;
    record->image_sha256_ = metadata.image_sha256_;
    record->width_px_ = image.cols;
    record->height_px_ = image.rows;
    record->fxfy_effective_ = metadata.fxfy_effective_;

    // Measure the startup cost. CUDA context creation happens on the first CUDA
    // call, so preparation is counted from the device buffer allocation onward.
    const auto setup_start = std::chrono::steady_clock::now();
    aruco3cuda::hybrid::DeviceImage device;
    std::string message;
    if (device.reserve(image.cols, image.rows, &message) != aruco3cuda::Status::kOk) {
        *out_error = message;
        return false;
    }
    const auto upload = [&]() {
        return device.upload(image.data, image.cols, image.rows,
                             static_cast<std::size_t>(image.step),
                             &message) == aruco3cuda::Status::kOk;
    };
    if (!upload()) {
        *out_error = message;
        return false;
    }

    aruco3cuda::hybrid::HybridDetector detector;
    if (detector.initialize(config.cuda_detector_, config.detector_.dictionary_name_, image.cols,
                            image.rows, &message) != aruco3cuda::Status::kOk) {
        *out_error = "cannot initialize the hybrid detector: " + message;
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
            *out_error = "hybrid detection failed: " + message;
            return false;
        }
        // The per-stage breakdown comes only from the latency iterations. Mixing
        // warm-up and throughput in would remove the guarantee that it can be
        // compared directly against the end_to_end percentiles.
        if (phase == Phase::kLatency) {
            gpu_samples.push_back(result.gpu_ms_);
            cpu_samples.push_back(result.cpu_ms_);
        }
        return true;
    };

    const auto first_start = std::chrono::steady_clock::now();
    if (!step(Phase::kWarmup)) {
        return false;
    }
    const auto first_finish = std::chrono::steady_clock::now();
    record->first_frame_ms_ =
            std::chrono::duration<double, std::milli>(first_finish - first_start).count();
    record->time_to_first_result_ms_ =
            std::chrono::duration<double, std::milli>(first_finish - setup_start).count();
    record->detection_count_ = result.detections_.size();

    if (!measure_iterations(config, step, record, out_error)) {
        return false;
    }
    record->stage_times_available_ =
            aruco3cuda::util::compute_statistics(gpu_samples, &record->gpu_stage_ms_) &&
            aruco3cuda::util::compute_statistics(cpu_samples, &record->cpu_stage_ms_);
    return true;
}

/// Always release the page-locked input buffer.
struct PinnedSource {
    void* data_ = nullptr;
    PinnedSource() = default;
    PinnedSource(const PinnedSource&) = delete;
    PinnedSource& operator=(const PinnedSource&) = delete;
    ~PinnedSource() {
        if (this->data_ != nullptr) {
            static_cast<void>(cudaFreeHost(this->data_));
        }
    }
};

/// Always destroy the stream. There are many early returns, so RAII is used.
struct StreamGuard {
    cudaStream_t stream_;
    StreamGuard(const StreamGuard&) = delete;
    StreamGuard& operator=(const StreamGuard&) = delete;
    ~StreamGuard() {
        if (this->stream_ != nullptr) {
            static_cast<void>(cudaStreamDestroy(this->stream_));
        }
    }
};

/// Measure the full GPU routes.
///
/// The measured interval depends on the route.
///
/// | Route | Measured interval |
/// | --- | --- |
/// | kCudaResident | Issuing detect_async and synchronizing the stream |
/// | kCudaEndToEnd | Transfer from the host, issuing, synchronizing, retrieving the results |
///
/// **Both intervals include the stream synchronization.** detect_async only
/// issues kernels and returns, so without the synchronization only the cost of
/// issuing would be measured.
bool measure_cuda(const std::string& image_path, const BenchmarkConfig& config,
                  MeasurementRecord* record, std::string* out_error) {
    const bool resident = config.route_ == Route::kCudaResident;
    if (resident && config.memory_mode_ != MemoryMode::kDevice) {
        *out_error =
                std::string("the CUDA-Resident route supports only the M-Device memory kind; "
                            "requested: ") +
                to_string(config.memory_mode_);
        return false;
    }
    if (!resident && config.memory_mode_ != MemoryMode::kHostPageable &&
        config.memory_mode_ != MemoryMode::kHostPinned &&
        config.memory_mode_ != MemoryMode::kManaged) {
        *out_error = std::string("the CUDA-EndToEnd route supports only the M-Pageable, "
                                 "M-Pinned, and M-Managed memory kinds; requested: ") +
                     to_string(config.memory_mode_);
        return false;
    }

    // Image loading and the checksum happen once, outside the measured interval.
    aruco3cuda::reference::ReferenceDetector loader;
    if (!loader.initialize(image_path, config.detector_, out_error)) {
        return false;
    }
    const aruco3cuda::reference::ReferenceResult& metadata = loader.metadata();
    const cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        *out_error = "cannot load image: " + image_path;
        return false;
    }

    record->image_path_ = image_path;
    record->image_sha256_ = metadata.image_sha256_;
    record->width_px_ = image.cols;
    record->height_px_ = image.rows;
    record->fxfy_effective_ = metadata.fxfy_effective_;

    const aruco3cuda::DictionaryTable* table =
            aruco3cuda::find_builtin_dictionary(config.detector_.dictionary_name_.c_str());
    if (table == nullptr) {
        *out_error = "unsupported Dictionary: " + config.detector_.dictionary_name_;
        return false;
    }

    // Size the allocation to the image. Leaving the default 3840x2160 in place
    // would measure even a small image while holding a 4K-sized workspace.
    aruco3cuda::DetectorConfig detector_config = config.cuda_detector_;
    detector_config.max_width_px_ = image.cols;
    detector_config.max_height_px_ = image.rows;

    // Measure the startup cost. CUDA context creation happens on the first CUDA
    // call, so preparation is counted from the device buffer allocation onward.
    const auto setup_start = std::chrono::steady_clock::now();
    aruco3cuda::hybrid::DeviceImage device;
    std::string message;
    // The memory kind of the input buffer is a measurement axis. Both how it is
    // allocated and what a transfer means change with the kind.
    const aruco3cuda::MemorySpace space = (config.memory_mode_ == MemoryMode::kManaged)
                                                  ? aruco3cuda::MemorySpace::kManaged
                                                  : aruco3cuda::MemorySpace::kDevice;
    if (device.reserve(space, image.cols, image.rows, &message) != aruco3cuda::Status::kOk) {
        *out_error = message;
        return false;
    }

    // M-Pinned measures the case where the input buffer is page-locked. The
    // buffer of a cv::Mat is pageable, so it is copied into a page-locked region
    // **once, outside the measured interval**. Copying it every frame would
    // measure the cost of the copy rather than the difference between kinds.
    PinnedSource pinned;
    const std::uint8_t* source = image.data;
    auto source_pitch = static_cast<std::size_t>(image.step);
    if (config.memory_mode_ == MemoryMode::kHostPinned) {
        const auto bytes =
                static_cast<std::size_t>(image.cols) * static_cast<std::size_t>(image.rows);
        if (cudaMallocHost(&pinned.data_, bytes) != cudaSuccess) {
            *out_error = "cannot allocate a page-locked input buffer";
            return false;
        }
        for (int row = 0; row < image.rows; ++row) {
            std::memcpy(
                    static_cast<std::uint8_t*>(pinned.data_) +
                            (static_cast<std::size_t>(row) * static_cast<std::size_t>(image.cols)),
                    image.data +
                            (static_cast<std::size_t>(row) * static_cast<std::size_t>(image.step)),
                    static_cast<std::size_t>(image.cols));
        }
        source = static_cast<const std::uint8_t*>(pinned.data_);
        source_pitch = static_cast<std::size_t>(image.cols);
    }

    const auto upload = [&]() {
        return device.upload(source, image.cols, image.rows, source_pitch, &message) ==
               aruco3cuda::Status::kOk;
    };
    if (!upload()) {
        *out_error = message;
        return false;
    }

    aruco3cuda::Detector detector;
    if (detector.initialize(*table, detector_config, &message) != aruco3cuda::Status::kOk) {
        *out_error = "cannot initialize the CUDA detector: " + message;
        return false;
    }

    // Pass an explicit stream. CUDA does not allow the default stream to be
    // captured, so it cannot take the path that folds the issue sequence into a
    // CUDA Graph. A dedicated stream is the natural choice in production, and the
    // measurement follows that shape.
    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
        *out_error = "cannot create the stream";
        return false;
    }
    // Destroyed on every path, including an early return.
    const StreamGuard guard{stream};

    aruco3cuda::HostDetections detections;
    const auto step = [&](Phase phase) {
        (void)phase;
        if (!resident && !upload()) {
            *out_error = message;
            return false;
        }
        const aruco3cuda::Status status = detector.detect_async(device.view(), stream, &message);
        if (status != aruco3cuda::Status::kOk) {
            *out_error = "CUDA detection failed: " + message;
            return false;
        }
        if (resident) {
            // Device resident, so the results are not brought back to the host.
            // The stream synchronization is still part of the interval; without
            // it only the cost of issuing would be measured.
            const cudaError_t synchronized = cudaStreamSynchronize(stream);
            if (synchronized != cudaSuccess) {
                *out_error =
                        std::string("cannot synchronize the stream: ") +
                        cudaGetErrorString(synchronized);
                return false;
            }
            return true;
        }
        // download synchronizes the stream.
        const aruco3cuda::Status downloaded = detector.download(&detections, stream, &message);
        if (downloaded != aruco3cuda::Status::kOk &&
            downloaded != aruco3cuda::Status::kMarkerOverflow) {
            *out_error = "cannot retrieve the results: " + message;
            return false;
        }
        return true;
    };

    const auto first_start = std::chrono::steady_clock::now();
    if (!step(Phase::kWarmup)) {
        return false;
    }
    const auto first_finish = std::chrono::steady_clock::now();
    record->first_frame_ms_ =
            std::chrono::duration<double, std::milli>(first_finish - first_start).count();
    record->time_to_first_result_ms_ =
            std::chrono::duration<double, std::milli>(first_finish - setup_start).count();

    // The detection count is read once, outside the measured interval. It is
    // needed even on the device-resident route.
    const aruco3cuda::Status counted = detector.download(&detections, stream, &message);
    if (counted != aruco3cuda::Status::kOk && counted != aruco3cuda::Status::kMarkerOverflow) {
        *out_error = "cannot read the detection count: " + message;
        return false;
    }
    record->detection_count_ = detections.ids_.size();

    return measure_iterations(config, step, record, out_error);
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
    // When ArUco3 is enabled, OpenCV unconditionally overwrites
    // cornerRefinementMethod with SUBPIX (at the top of detectMarkers in
    // aruco_detector.cpp), because the corners detected on the downscaled image
    // have to be walked back up the pyramid to full resolution. The CPU baseline
    // therefore refines regardless of use_corner_subpix_refinement_ when ArUco3
    // is enabled, and the copy follows suit.
    result.corner_refine_method_ =
            (config.use_aruco3_detection_ || config.use_corner_subpix_refinement_)
                    ? aruco3cuda::CornerRefineMethod::kSubpix
                    : aruco3cuda::CornerRefineMethod::kNone;
    result.corner_refinement_win_size_px_ = config.corner_refinement_win_size_px_;
    result.relative_corner_refinement_win_size_ = config.relative_corner_refinement_win_size_;
    result.corner_refinement_max_iterations_ = config.corner_refinement_max_iterations_;
    result.corner_refinement_min_accuracy_px_ = config.corner_refinement_min_accuracy_px_;
    // The items that govern the allocation size are not copied. Sizing them to
    // the image is the caller's responsibility; leaving the default (3840x2160)
    // here would make the recorded measurement conditions disagree with reality.
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
        case Route::kCudaResident:
        case Route::kCudaEndToEnd:
            ok = measure_cuda(image_path, config, &record, out_error);
            break;
        default:
            // Substituting CPU for an unknown route would make the measurement
            // disagree with the route name.
            *out_error = std::string("route ") + to_string(config.route_) + " is unimplemented";
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
    // The per-version history is with kSchemaVersion, so that there is one
    // changelog rather than two that drift apart.
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
    writer.member_double("cuda_context_ms", environment.cuda_context_ms_, 3);
    writer.member_string("driver_version", environment.driver_version_);
    writer.member_string("platform_release", environment.platform_release_);
    writer.member_string("platform_model", environment.platform_model_);
    writer.member_string("power_mode", environment.power_mode_);
    // Reason a GPU item is missing. Empty when it was obtained successfully.
    writer.member_string("gpu_probe_error", environment.gpu_probe_error_);
    // The clocks bear directly on the measurement conditions. Writing 0 when they
    // cannot be obtained would be misread as "the clock is 0", so null is used.
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
    writer.member_double(
            "min_marker_length_ratio_original_img",
            static_cast<double>(config.detector_.min_marker_length_ratio_original_img_), 6);
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
        // The CPU route has no kernel time. It is not filled with 0; "unmeasured"
        // is stated explicitly.
        writer.value_null();
    }

    // Startup cost. It does not appear in the percentiles after warm-up, so it is
    // recorded separately.
    writer.key("startup");
    writer.begin_object();
    writer.member_double("time_to_first_result_ms", record.time_to_first_result_ms_, 4);
    writer.member_double("first_frame_ms", record.first_frame_ms_, 4);
    writer.end_object();

    // Stage times. The CPU route has none, so "unmeasured" is stated explicitly.
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
