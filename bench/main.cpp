// SPDX-License-Identifier: Apache-2.0
//
// benchmark harness の CLI。
//
// 目的:
//   測定条件と環境情報を結果と一体で残し、後から再現できる形で性能を記録する。
//   結果は JSONL で出力し、1 行目に環境情報、以降に測定結果を並べる。
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "benchmark_harness.hpp"

namespace {

using aruco3cuda::bench::BenchmarkConfig;
using aruco3cuda::bench::Route;

void print_usage(std::ostream& out) {
    out << "使用方法: aruco3cuda_bench [option]... --input <画像>...\n"
        << "\n"
        << "  --input <path>                 入力画像。複数回指定できる\n"
        << "  --output <path>                結果 JSONL の出力先。既定は標準出力\n"
        << "  --route <name>                 CPU のみ実装済み。既定 CPU\n"
        << "  --warmup <n>                   準備実行の回数。既定 20\n"
        << "  --latency-iterations <n>       遅延測定の回数。既定 200\n"
        << "  --throughput-frames <n>        throughput 測定のフレーム数。0 で無効。既定 100\n"
        << "  --save-samples                 全標本を結果へ含める\n"
        << "  --dictionary <name>            既定 DICT_ARUCO_MIP_36h12\n"
        << "  --threads <n>                  OpenCV の thread 数。既定 1\n"
        << "  --use-aruco3 <0|1>             既定 1\n"
        << "  --min-marker-length-ratio <f>  既定 0.05。0 では縮小が発生しない\n"
        << "  --min-side-length-canonical <n>  既定 32\n"
        << "  --help                         この説明を表示して終了\n";
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

}  // namespace

int main(int argc, char** argv) {
    BenchmarkConfig config;
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
            std::cerr << "引数が不足している: " << option << '\n';
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
                    std::cerr << "未知の経路: " << value << '\n';
                    return EXIT_FAILURE;
                }
            } else if (option == "--warmup") {
                config.warmup_iterations_ = std::stoi(value);
            } else if (option == "--latency-iterations") {
                config.latency_iterations_ = std::stoi(value);
            } else if (option == "--throughput-frames") {
                config.throughput_frames_ = std::stoi(value);
            } else if (option == "--dictionary") {
                config.detector_.dictionary_name_ = value;
            } else if (option == "--threads") {
                config.detector_.num_threads_ = std::stoi(value);
            } else if (option == "--use-aruco3") {
                config.detector_.use_aruco3_detection_ = std::stoi(value) != 0;
            } else if (option == "--min-marker-length-ratio") {
                config.detector_.min_marker_length_ratio_original_img_ =
                        std::stof(value);
            } else if (option == "--min-side-length-canonical") {
                config.detector_.min_side_length_canonical_img_px_ = std::stoi(value);
            } else {
                std::cerr << "未知の option: " << option << '\n';
                print_usage(std::cerr);
                return EXIT_FAILURE;
            }
        } catch (const std::exception&) {
            std::cerr << "値を解釈できない: " << option << ' ' << value << '\n';
            return EXIT_FAILURE;
        }
    }

    if (inputs.empty()) {
        std::cerr << "--input が指定されていない\n";
        print_usage(std::cerr);
        return EXIT_FAILURE;
    }

    std::ofstream file_output;
    if (!output_path.empty()) {
        file_output.open(output_path);
        if (!file_output) {
            std::cerr << "出力 file を開けない: " << output_path << '\n';
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
    return EXIT_SUCCESS;
}
