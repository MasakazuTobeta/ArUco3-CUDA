// SPDX-License-Identifier: Apache-2.0
//
// benchmark harness の CLI。
//
// 目的:
//   測定条件と環境情報を結果と一体で残し、後から再現できる形で性能を記録する。
//   結果は JSONL で出力し、1 行目に環境情報、以降に測定結果を並べる。
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
    out << "使用方法: aruco3cuda_bench [option]... --input <画像>...\n"
        << "\n"
        << "  --input <path>                 入力画像。複数回指定できる\n"
        << "  --output <path>                結果 JSONL の出力先。既定は標準出力\n"
        << "  --route <name>                 CPU、Hybrid、CUDA-Resident、CUDA-EndToEnd。既定 CPU\n"
        << "  --memory-mode <name>           N/A、M-Pageable、M-Pinned、M-Managed、M-Device。\n"
        << "                                 既定は経路に応じて選ぶ\n"
        << "                                 M-Device は転送を測定区間の外へ置き、M-Pageable と\n"
        << "                                 M-Pinned は毎 frame の転送を測定区間へ含める\n"
        << "                                 CUDA-Resident は M-Device のみ、CUDA-EndToEnd は\n"
        << "                                 M-Pageable と M-Pinned のみに対応する\n"
        << "  --warmup <n>                   準備実行の回数。既定 20\n"
        << "  --latency-iterations <n>       遅延測定の回数。既定 200\n"
        << "  --throughput-frames <n>        throughput 測定のフレーム数。0 で無効。既定 100\n"
        << "  --save-samples                 全標本を結果へ含める\n"
        << "  --cpu-list <n[,n...]>          測定に使う CPU 番号。性能 core と効率 core が\n"
        << "                                 混在する機では固定しないと値が二極化する\n"
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

/// 名前から memory 種別を求める。
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

/// カンマ区切りの CPU 番号一覧を解析する。
///
/// @param text 例: "5,6,7"。空白は許さない。
/// @param out 成功時に CPU 番号を格納する。
/// @return 全ての要素が非負整数なら true。
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

/// 解析した整数が指定範囲にあることを確認する。
///
/// 外部入力である argv を信頼せず、範囲外の値を測定条件へ書き込まない。
bool check_range(const std::string& option, int value, int minimum, int maximum) {
    if (value < minimum || value > maximum) {
        std::cerr << option << " は " << minimum << " 以上 " << maximum << " 以下である必要がある: "
                  << value << '\n';
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    BenchmarkConfig config;
    // memory 種別を指定しなかった場合の既定は経路で決める。CPU 経路に
    // memory 種別は無く、hybrid は device 常駐入力を上限として測る。
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
            } else if (option == "--memory-mode") {
                if (!parse_memory_mode(value, &config.memory_mode_)) {
                    std::cerr << "未知の memory 種別: " << value << '\n';
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
                    std::cerr << "--cpu-list は カンマ区切りの非負整数である必要がある: " << value
                              << '\n';
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
                    std::cerr << option << " は 0 以上 1 以下である必要がある: " << parsed << '\n';
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

    // CUDA 経路の設定を CPU 基準の設定から写す。既定構築のままにすると
    // 縮小率や ArUco3 の有無が食い違い、別の条件を比べることになる。
    config.cuda_detector_ = aruco3cuda::bench::cuda_config_from_reference(config.detector_);
    if (!memory_mode_given) {
        // 経路ごとに、その経路が対応する種別を既定にする。CUDA-EndToEnd は
        // 転送を測定区間に含める経路であり、device 常駐を既定にできない。
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

    // 書き込み失敗を成功として報告しない。測定結果の欠落は後から気付けない。
    if (output_path.empty()) {
        std::cout.flush();
        if (!std::cout) {
            std::cerr << "標準出力への書き込みに失敗した\n";
            return EXIT_FAILURE;
        }
    } else {
        file_output.close();
        if (!file_output) {
            std::cerr << "出力 file への書き込みに失敗した: " << output_path << '\n';
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
