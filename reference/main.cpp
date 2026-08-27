// SPDX-License-Identifier: Apache-2.0
//
// CPU 基準 runner の CLI。
//
// 目的:
//   OpenCV の ArUco 検出を固定した設定で実行し、結果を機械可読形式で保存する。
//   CUDA 実装との差分比較および crossover point の測定に使用する基準値を作る。
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "reference_runner.hpp"

namespace {

using aruco3cuda::reference::ReferenceConfig;

void print_usage(std::ostream& out) {
    out << "使用方法: aruco3cuda_reference_runner [option]... --input <画像>...\n"
        << "\n"
        << "  --input <path>                 入力画像。複数回指定できる\n"
        << "  --output <path>                結果 JSON の出力先。既定は標準出力\n"
        << "  --dictionary <name>            既定 DICT_ARUCO_MIP_36h12\n"
        << "  --threads <n>                  OpenCV の thread 数。既定 1。0 で OpenCV の既定\n"
        << "  --use-aruco3 <0|1>             ArUco3 検出戦略。既定 1\n"
        << "  --min-side-length-canonical <n>  既定 32\n"
        << "  --min-marker-length-ratio <f>  既定 0.05。0 では縮小が発生しない\n"
        << "  --adaptive-thresh-win-min <n>  既定 3\n"
        << "  --adaptive-thresh-win-max <n>  既定 23\n"
        << "  --adaptive-thresh-win-step <n> 既定 10\n"
        << "  --error-correction-rate <f>    既定 0.6\n"
        << "  --omit-timing                  実行時間を出力へ含めない。golden 比較用\n"
        << "  --list-dictionaries            対応する Dictionary 名を表示して終了\n"
        << "  --help                         この説明を表示して終了\n";
}

/// 次の引数を取り出す。不足している場合は false を返す。
bool take_value(int argc, char** argv, int* index, const char* option, std::string* out) {
    if (*index + 1 >= argc) {
        std::cerr << "引数が不足している: " << option << '\n';
        return false;
    }
    ++(*index);
    *out = argv[*index];
    return true;
}

/// 解析した整数が指定範囲にあることを確認する。
///
/// 外部入力である argv を信頼せず、範囲外の値を設定へ書き込まない。
/// 範囲の検証を CLI 側でも行うのは、どの option が不正かを利用者へ示すためである。
bool check_range(const std::string& option, int value, int minimum, int maximum) {
    if (value < minimum || value > maximum) {
        std::cerr << option << " は " << minimum << " 以上 " << maximum
                  << " 以下である必要がある: " << value << '\n';
        return false;
    }
    return true;
}

bool parse_int(const std::string& text, int* out) {
    try {
        std::size_t consumed = 0;
        const int value = std::stoi(text, &consumed);
        if (consumed != text.size()) {
            return false;
        }
        *out = value;
        return true;
    } catch (const std::exception&) {
        return false;
    }
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

}  // namespace

int main(int argc, char** argv) {
    ReferenceConfig config;
    std::vector<std::string> inputs;
    std::string output_path;

    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        std::string value;
        int int_value = 0;
        double double_value = 0.0;

        if (option == "--help") {
            print_usage(std::cout);
            return EXIT_SUCCESS;
        }
        if (option == "--list-dictionaries") {
            for (const std::string& name : aruco3cuda::reference::known_dictionary_names()) {
                std::cout << name << '\n';
            }
            return EXIT_SUCCESS;
        }
        if (option == "--input") {
            if (!take_value(argc, argv, &i, "--input", &value)) {
                return EXIT_FAILURE;
            }
            inputs.push_back(value);
            continue;
        }
        if (option == "--omit-timing") {
            config.omit_timing_ = true;
            continue;
        }
        if (option == "--output") {
            if (!take_value(argc, argv, &i, "--output", &output_path)) {
                return EXIT_FAILURE;
            }
            continue;
        }
        if (option == "--dictionary") {
            if (!take_value(argc, argv, &i, "--dictionary", &config.dictionary_name_)) {
                return EXIT_FAILURE;
            }
            continue;
        }

        // 数値 option をまとめて扱う。
        struct IntOption {
            const char* name;
            int* target;
            int minimum;
            int maximum;
        };
        // 範囲は reference_runner の validate_config と一致させる。
        const IntOption int_options[] = {
                {"--threads", &config.num_threads_, 0, 1024},
                {"--min-side-length-canonical", &config.min_side_length_canonical_img_px_, 0, 4096},
                {"--adaptive-thresh-win-min", &config.adaptive_thresh_win_size_min_px_, 3, 4096},
                {"--adaptive-thresh-win-max", &config.adaptive_thresh_win_size_max_px_, 3, 4096},
                {"--adaptive-thresh-win-step", &config.adaptive_thresh_win_size_step_px_, 1, 4096},
        };
        bool handled = false;
        for (const IntOption& entry : int_options) {
            if (option == entry.name) {
                if (!take_value(argc, argv, &i, entry.name, &value)) {
                    return EXIT_FAILURE;
                }
                if (!parse_int(value, &int_value)) {
                    std::cerr << "整数として解釈できない: " << entry.name << ' ' << value << '\n';
                    return EXIT_FAILURE;
                }
                if (!check_range(entry.name, int_value, entry.minimum, entry.maximum)) {
                    return EXIT_FAILURE;
                }
                *entry.target = int_value;
                handled = true;
                break;
            }
        }
        if (handled) {
            continue;
        }

        if (option == "--use-aruco3") {
            if (!take_value(argc, argv, &i, "--use-aruco3", &value) ||
                !parse_int(value, &int_value)) {
                std::cerr << "0 または 1 を指定すること: --use-aruco3\n";
                return EXIT_FAILURE;
            }
            config.use_aruco3_detection_ = int_value != 0;
            continue;
        }
        if (option == "--min-marker-length-ratio") {
            if (!take_value(argc, argv, &i, "--min-marker-length-ratio", &value) ||
                !parse_double(value, &double_value)) {
                std::cerr << "実数として解釈できない: --min-marker-length-ratio\n";
                return EXIT_FAILURE;
            }
            if (!(double_value >= 0.0) || !(double_value <= 1.0)) {
                std::cerr << "--min-marker-length-ratio は 0 以上 1 以下である必要がある: "
                          << double_value << '\n';
                return EXIT_FAILURE;
            }
            config.min_marker_length_ratio_original_img_ = static_cast<float>(double_value);
            continue;
        }
        if (option == "--error-correction-rate") {
            if (!take_value(argc, argv, &i, "--error-correction-rate", &value) ||
                !parse_double(value, &double_value)) {
                std::cerr << "実数として解釈できない: --error-correction-rate\n";
                return EXIT_FAILURE;
            }
            if (!(double_value >= 0.0) || !(double_value <= 1.0)) {
                std::cerr << "--error-correction-rate は 0 以上 1 以下である必要がある: "
                          << double_value << '\n';
                return EXIT_FAILURE;
            }
            config.error_correction_rate_ = double_value;
            continue;
        }

        std::cerr << "未知の option: " << option << '\n';
        print_usage(std::cerr);
        return EXIT_FAILURE;
    }

    if (inputs.empty()) {
        std::cerr << "--input が指定されていない\n";
        print_usage(std::cerr);
        return EXIT_FAILURE;
    }
    if (!aruco3cuda::reference::is_known_dictionary(config.dictionary_name_)) {
        std::cerr << "未対応の Dictionary: " << config.dictionary_name_ << '\n'
                  << "--list-dictionaries で対応名を確認できる\n";
        return EXIT_FAILURE;
    }
    // OpenCV 側の制約。ArUco3 有効時にどちらも 0 だと detectMarkers が失敗する。
    if (config.use_aruco3_detection_ && config.min_side_length_canonical_img_px_ == 0 &&
        config.min_marker_length_ratio_original_img_ == 0.0F) {
        std::cerr << "--use-aruco3 1 では min-side-length-canonical と "
                     "min-marker-length-ratio の両方を 0 にできない\n";
        return EXIT_FAILURE;
    }

    const aruco3cuda::reference::ReferenceEnvironment environment =
            aruco3cuda::reference::collect_environment(config);

    std::vector<aruco3cuda::reference::ReferenceResult> results;
    results.reserve(inputs.size());
    for (const std::string& input : inputs) {
        aruco3cuda::reference::ReferenceResult result;
        std::string error;
        if (!aruco3cuda::reference::detect_image(input, config, &result, &error)) {
            std::cerr << error << '\n';
            return EXIT_FAILURE;
        }
        results.push_back(result);
    }

    if (output_path.empty()) {
        aruco3cuda::reference::write_results_json(std::cout, config, environment, results);
        // 書き込み失敗を成功として報告しない。pipe が閉じられた場合などに起きる。
        std::cout.flush();
        if (!std::cout) {
            std::cerr << "標準出力への書き込みに失敗した\n";
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }
    std::ofstream output(output_path);
    if (!output) {
        std::cerr << "出力 file を開けない: " << output_path << '\n';
        return EXIT_FAILURE;
    }
    aruco3cuda::reference::write_results_json(output, config, environment, results);
    // close まで行って状態を確認する。buffer に残ったまま失敗する場合があるため。
    output.close();
    if (!output) {
        std::cerr << "出力 file への書き込みに失敗した: " << output_path << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
