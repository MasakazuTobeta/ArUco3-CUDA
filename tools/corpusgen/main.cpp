// SPDX-License-Identifier: Apache-2.0
//
// 合成 corpus 生成器の CLI。
//
// 目的:
//   四隅の ground truth を持つ合成画像を、seed 固定で再生成できる形で作る。
//   CPU 基準結果は互換性の基準であり ground truth ではないため、正確性評価には
//   生成時に既知である真値が必要になる。
#include <cstddef>
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
                config.seed_ = std::stoull(value);
            } else if (option == "--canonical-px") {
                config.canonical_marker_px_ = std::stoi(value);
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

    std::size_t marker_total = 0;
    for (const auto& scene : scenes) {
        marker_total += scene.markers_.size();
    }
    std::cout << "生成: " << scenes.size() << " 枚、マーカー " << marker_total << " 個\n"
              << "manifest: " << manifest_path << '\n';
    return EXIT_SUCCESS;
}
