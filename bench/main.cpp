// SPDX-License-Identifier: Apache-2.0
//
// CLI for the benchmark harness.
//
// Purpose:
//   Keep the measurement conditions and the environment information together
//   with the results, so that performance is recorded in a form that can be
//   reproduced later. The results are written as JSONL: the first line holds the
//   environment information and the lines after it hold the measurements.
#include <cstdlib>
#include <exception>
#include <string>
#include <fstream>
#include <iostream>
#include <cstddef>
#include <string>
#include <vector>

#include "benchmark_harness.hpp"

namespace {

using aruco3cuda::bench::BenchmarkConfig;
using aruco3cuda::bench::Route;
using aruco3cuda::bench::MemoryMode;

void print_usage(std::ostream& out) {
    out << "Usage: aruco3cuda_bench [option]... --input <image>...\n"
        << "\n"
        << "  --input <path>                 Input image. May be given several times\n"
        << "  --output <path>                Destination of the result JSONL. Default is stdout\n"
        << "  --route <name>                 CPU, Hybrid, CUDA-Resident, CUDA-EndToEnd.\n"
        << "                                 Default CPU\n"
        << "  --memory-mode <name>           N/A, M-Pageable, M-Pinned, M-Managed, M-Device.\n"
        << "                                 The default is chosen from the route\n"
        << "                                 M-Device places the transfer outside the measured\n"
        << "                                 interval; M-Pageable and M-Pinned include the\n"
        << "                                 per-frame transfer in it\n"
        << "                                 CUDA-Resident supports only M-Device, and\n"
        << "                                 CUDA-EndToEnd only M-Pageable and M-Pinned\n"
        << "  --warmup <n>                   Number of preparatory runs. Default 20\n"
        << "  --latency-iterations <n>       Number of latency measurements. Default 200\n"
        << "  --throughput-frames <n>        Frames for the throughput measurement.\n"
        << "                                 0 disables it. Default 100\n"
        << "  --save-samples                 Include every sample in the results\n"
        << "  --cpu-list <n[,n...]>          CPU numbers to measure on. On a machine that mixes\n"
        << "                                 performance and efficiency cores, values become\n"
        << "                                 bimodal unless they are pinned\n"
        << "  --dictionary <name>            Default DICT_ARUCO_MIP_36h12\n"
        << "  --threads <n>                  OpenCV thread count. Default 1\n"
        << "  --use-aruco3 <0|1>             Default 1\n"
        << "  --min-marker-length-ratio <f>  Default 0.05. No downscaling happens at 0\n"
        << "  --min-side-length-canonical <n>  Default 32\n"
        << "  --help                         Print this help and exit\n";
}

bool parse_route(const std::string& name, Route* out) {
    if (name == "CPU") {
        *out = Route::kCpu;
        return true;
    }
    if (name == "CUDA-E2E") {
        *out = Route::kCudaEndToEnd;
        return true;
    }
    if (name == "CUDA-Resident") {
        *out = Route::kCudaResident;
        return true;
    }
    if (name == "Hybrid") {
        *out = Route::kHybrid;
        return true;
    }
    return false;
}

/// Resolve a memory kind from its name.
bool parse_memory_mode(const std::string& name, MemoryMode* out) {
    if (name == "N/A") {
        *out = MemoryMode::kNotApplicable;
        return true;
    }
    if (name == "M-Pageable") {
        *out = MemoryMode::kHostPageable;
        return true;
    }
    if (name == "M-Pinned") {
        *out = MemoryMode::kHostPinned;
        return true;
    }
    if (name == "M-Managed") {
        *out = MemoryMode::kManaged;
        return true;
    }
    if (name == "M-Device") {
        *out = MemoryMode::kDevice;
        return true;
    }
    return false;
}

/// Parse a comma-separated list of CPU numbers.
///
/// @param text For example "5,6,7". Whitespace is not allowed.
/// @param out Receives the CPU numbers on success.
/// @return true when every element is a non-negative integer.
bool parse_cpu_list(const std::string& text, std::vector<int>* out) {
    out->clear();
    std::size_t begin = 0;
    while (begin <= text.size()) {
        const std::size_t end = text.find(',', begin);
        const std::string token =
                text.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        if (token.empty() || token.find_first_not_of("0123456789") != std::string::npos) {
            return false;
        }
        try {
            out->push_back(std::stoi(token));
        } catch (const std::exception&) {
            return false;
        }
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return !out->empty();
}

/// Confirm that a parsed integer lies within the given range.
///
/// argv is external input and is not trusted, so an out-of-range value is never
/// written into the measurement conditions.
bool check_range(const std::string& option, int value, int minimum, int maximum) {
    if (value < minimum || value > maximum) {
        std::cerr << option << " must be between " << minimum << " and " << maximum
                  << " inclusive: " << value << '\n';
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    BenchmarkConfig config;
    // When no memory kind is given, the default follows the route. The CPU route
    // has no memory kind, and hybrid is measured with device-resident input as
    // its upper bound.
    bool memory_mode_given = false;
    std::vector<std::string> inputs;
    std::string output_path;

    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--help") {
            print_usage(std::cout);
            return EXIT_SUCCESS;
        }
        if (option == "--save-samples") {
            config.save_all_samples_ = true;
            continue;
        }
        if (i + 1 >= argc) {
            std::cerr << "missing argument: " << option << '\n';
            return EXIT_FAILURE;
        }
        const std::string value = argv[++i];
        try {
            if (option == "--input") {
                inputs.push_back(value);
            } else if (option == "--output") {
                output_path = value;
            } else if (option == "--route") {
                if (!parse_route(value, &config.route_)) {
                    std::cerr << "unknown route: " << value << '\n';
                    return EXIT_FAILURE;
                }
            } else if (option == "--memory-mode") {
                if (!parse_memory_mode(value, &config.memory_mode_)) {
                    std::cerr << "unknown memory kind: " << value << '\n';
                    return EXIT_FAILURE;
                }
                memory_mode_given = true;
            } else if (option == "--warmup") {
                const int parsed = std::stoi(value);
                if (!check_range(option, parsed, 0, 100000)) {
                    return EXIT_FAILURE;
                }
                config.warmup_iterations_ = parsed;
            } else if (option == "--latency-iterations") {
                const int parsed = std::stoi(value);
                if (!check_range(option, parsed, 1, 1000000)) {
                    return EXIT_FAILURE;
                }
                config.latency_iterations_ = parsed;
            } else if (option == "--throughput-frames") {
                const int parsed = std::stoi(value);
                if (!check_range(option, parsed, 0, 1000000)) {
                    return EXIT_FAILURE;
                }
                config.throughput_frames_ = parsed;
            } else if (option == "--cpu-list") {
                if (!parse_cpu_list(value, &config.cpu_affinity_)) {
                    std::cerr << "--cpu-list must be comma-separated non-negative integers: "
                              << value << '\n';
                    return EXIT_FAILURE;
                }
            } else if (option == "--dictionary") {
                config.detector_.dictionary_name_ = value;
            } else if (option == "--threads") {
                const int parsed = std::stoi(value);
                if (!check_range(option, parsed, 0, 1024)) {
                    return EXIT_FAILURE;
                }
                config.detector_.num_threads_ = parsed;
            } else if (option == "--use-aruco3") {
                const int parsed = std::stoi(value);
                if (!check_range(option, parsed, 0, 1)) {
                    return EXIT_FAILURE;
                }
                config.detector_.use_aruco3_detection_ = parsed != 0;
            } else if (option == "--min-marker-length-ratio") {
                const float parsed = std::stof(value);
                if (!(parsed >= 0.0F) || !(parsed <= 1.0F)) {
                    std::cerr << option << " must be between 0 and 1 inclusive: " << parsed
                              << '\n';
                    return EXIT_FAILURE;
                }
                config.detector_.min_marker_length_ratio_original_img_ = parsed;
            } else if (option == "--min-side-length-canonical") {
                const int parsed = std::stoi(value);
                if (!check_range(option, parsed, 0, 4096)) {
                    return EXIT_FAILURE;
                }
                config.detector_.min_side_length_canonical_img_px_ = parsed;
            } else {
                std::cerr << "unknown option: " << option << '\n';
                print_usage(std::cerr);
                return EXIT_FAILURE;
            }
        } catch (const std::exception&) {
            std::cerr << "cannot parse the value: " << option << ' ' << value << '\n';
            return EXIT_FAILURE;
        }
    }

    if (inputs.empty()) {
        std::cerr << "--input was not specified\n";
        print_usage(std::cerr);
        return EXIT_FAILURE;
    }

    // Copy the CUDA route settings from the CPU baseline settings. Leaving the
    // default-constructed value in place would let the downscale ratio and
    // whether ArUco3 is enabled diverge, so that two different conditions were
    // being compared.
    config.cuda_detector_ = aruco3cuda::bench::cuda_config_from_reference(config.detector_);
    if (!memory_mode_given) {
        // For each route, default to a kind that route supports. CUDA-EndToEnd is
        // a route that includes the transfer in the measured interval, so
        // device-resident cannot be its default.
        switch (config.route_) {
            case aruco3cuda::bench::Route::kCpu:
                config.memory_mode_ = aruco3cuda::bench::MemoryMode::kNotApplicable;
                break;
            case aruco3cuda::bench::Route::kCudaEndToEnd:
                config.memory_mode_ = aruco3cuda::bench::MemoryMode::kHostPageable;
                break;
            case aruco3cuda::bench::Route::kCudaResident:
            case aruco3cuda::bench::Route::kHybrid:
            default:
                config.memory_mode_ = aruco3cuda::bench::MemoryMode::kDevice;
                break;
        }
    }

    std::ofstream file_output;
    if (!output_path.empty()) {
        file_output.open(output_path);
        if (!file_output) {
            std::cerr << "cannot open the output file: " << output_path << '\n';
            return EXIT_FAILURE;
        }
    }
    std::ostream& out = output_path.empty() ? std::cout : file_output;

    const aruco3cuda::bench::EnvironmentRecord environment =
            aruco3cuda::bench::collect_environment(config);
    aruco3cuda::bench::write_environment_line(out, environment);

    for (const std::string& input : inputs) {
        aruco3cuda::bench::MeasurementRecord record;
        std::string error;
        if (!aruco3cuda::bench::measure_image(input, config, &record, &error)) {
            std::cerr << error << '\n';
            return EXIT_FAILURE;
        }
        aruco3cuda::bench::write_measurement_line(out, config, record);
        std::cerr << input << ": p50=" << record.end_to_end_ms_.p50_
                  << " ms p95=" << record.end_to_end_ms_.p95_
                  << " ms p99=" << record.end_to_end_ms_.p99_ << " ms\n";
    }

    // Never report a write failure as success. Missing measurements cannot be
    // noticed afterwards.
    if (output_path.empty()) {
        std::cout.flush();
        if (!std::cout) {
            std::cerr << "failed to write to standard output\n";
            return EXIT_FAILURE;
        }
    } else {
        file_output.close();
        if (!file_output) {
            std::cerr << "failed to write to the output file: " << output_path << '\n';
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
