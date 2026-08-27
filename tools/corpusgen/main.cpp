// SPDX-License-Identifier: Apache-2.0
//
// 合成 corpus 生成器の CLI。
//
// 目的:
//   四隅の ground truth を持つ合成画像を、seed 固定で再生成できる形で作る。
//   CPU 基準結果は互換性の基準であり ground truth ではないため、正確性評価には
//   生成時に既知である真値が必要になる。
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "corpus_generator.hpp"

namespace {

using aruco3cuda::corpusgen::CorpusConfig;

void print_usage(std::ostream& out) {
    out << "使用方法: aruco3cuda_corpusgen [option]...\n"
        << "\n"
        << "  --output-dir <dir>   画像の出力先。既定 data/corpus\n"
        << "  --manifest <path>    manifest JSON の出力先。既定 <output-dir>/manifest.json\n"
        << "  --preset <name>      smoke | basic | full。既定 smoke\n"
        << "  --dictionary <name>  既定 DICT_ARUCO_MIP_36h12\n"
        << "  --seed <n>           乱数種。既定 20260827\n"
        << "  --canonical-px <n>   マーカー描画の canonical 解像度。既定 512\n"
        << "  --list-presets       preset 名を表示して終了\n"
        << "  --help               この説明を表示して終了\n";
}

/// 文字列全体が整数として解釈できることを確認する。
///
/// std::stoi は末尾の余分な文字を無視するため、"512xyz" が 512 として
/// 無言で受理される。外部入力である argv を信頼せず、全体一致を要求する。
bool parse_int_strict(const std::string& text, int* out) {
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

/// 文字列全体が符号なし整数として解釈できることを確認する。
bool parse_uint64_strict(const std::string& text, std::uint64_t* out) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
        return false;
    }
    try {
        std::size_t consumed = 0;
        const unsigned long long value = std::stoull(text, &consumed);
        if (consumed != text.size()) {
            return false;
        }
        *out = static_cast<std::uint64_t>(value);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

}  // namespace

int main(int argc, char** argv) {
    CorpusConfig config;
    std::string preset = "smoke";
    std::string manifest_path;

    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--help") {
            print_usage(std::cout);
            return EXIT_SUCCESS;
        }
        if (option == "--list-presets") {
            for (const std::string& name : aruco3cuda::corpusgen::known_presets()) {
                std::cout << name << '\n';
            }
            return EXIT_SUCCESS;
        }
        if (i + 1 >= argc) {
            std::cerr << "引数が不足している: " << option << '\n';
            return EXIT_FAILURE;
        }
        const std::string value = argv[++i];
        try {
            if (option == "--output-dir") {
                config.output_dir_ = value;
            } else if (option == "--manifest") {
                manifest_path = value;
            } else if (option == "--preset") {
                preset = value;
            } else if (option == "--dictionary") {
                config.dictionary_name_ = value;
            } else if (option == "--seed") {
                if (!parse_uint64_strict(value, &config.seed_)) {
                    std::cerr << "--seed は符号なし整数である必要がある: " << value << '\n';
                    return EXIT_FAILURE;
                }
            } else if (option == "--canonical-px") {
                int canonical_px = 0;
                if (!parse_int_strict(value, &canonical_px)) {
                    std::cerr << "--canonical-px は整数である必要がある: " << value << '\n';
                    return EXIT_FAILURE;
                }
                // 上限は memory 使用量から決める。範囲は corpus_generator の
                // validate_scene と一致させる。
                if (canonical_px < 8 || canonical_px > 8192) {
                    std::cerr << "--canonical-px は 8 以上 8192 以下である必要がある: "
                              << canonical_px << '\n';
                    return EXIT_FAILURE;
                }
                config.canonical_marker_px_ = canonical_px;
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

    std::vector<aruco3cuda::corpusgen::SceneSpec> specs;
    if (!aruco3cuda::corpusgen::build_preset(preset, &specs)) {
        std::cerr << "未知の preset: " << preset << '\n' << "--list-presets で対応名を確認できる\n";
        return EXIT_FAILURE;
    }
    if (manifest_path.empty()) {
        manifest_path = config.output_dir_ + "/manifest.json";
    }

    std::vector<aruco3cuda::corpusgen::GeneratedScene> scenes;
    scenes.reserve(specs.size());
    for (std::size_t index = 0; index < specs.size(); ++index) {
        aruco3cuda::corpusgen::GeneratedScene scene;
        std::string error;
        if (!aruco3cuda::corpusgen::generate_scene(config, specs[index], index, &scene, &error)) {
            std::cerr << error << '\n';
            return EXIT_FAILURE;
        }
        scenes.push_back(scene);
    }

    std::ofstream manifest(manifest_path);
    if (!manifest) {
        std::cerr << "manifest を開けない: " << manifest_path << '\n'
                  << "出力先 directory が存在するか確認すること\n";
        return EXIT_FAILURE;
    }
    aruco3cuda::corpusgen::write_manifest_json(manifest, config, preset, scenes);
    // 書き込み失敗を成功として報告しない。manifest が欠けると corpus を
    // 後から参照できなくなる。
    manifest.close();
    if (!manifest) {
        std::cerr << "manifest への書き込みに失敗した: " << manifest_path << '\n';
        return EXIT_FAILURE;
    }

    std::size_t marker_total = 0;
    for (const auto& scene : scenes) {
        marker_total += scene.markers_.size();
    }
    std::cout << "生成: " << scenes.size() << " 枚、マーカー " << marker_total << " 個\n"
              << "manifest: " << manifest_path << '\n';
    return EXIT_SUCCESS;
}
