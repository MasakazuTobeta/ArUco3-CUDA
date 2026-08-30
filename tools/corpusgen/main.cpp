// SPDX-License-Identifier: Apache-2.0
//
// CLI of the synthetic corpus generator.
//
// Purpose:
//   Produces synthetic images carrying four-corner ground truth, in a form that can
//   be regenerated from a fixed seed. The CPU baseline results are a compatibility
//   reference, not ground truth, so an accuracy evaluation needs true values that are
//   known at generation time.
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
    out << "usage: aruco3cuda_corpusgen [option]...\n"
        << "\n"
        << "  --output-dir <dir>   destination of the images; default data/corpus\n"
        << "  --manifest <path>    destination of the manifest JSON; default "
           "<output-dir>/manifest.json\n"
        << "  --preset <name>      smoke | basic | full; default smoke\n"
        << "  --dictionary <name>  default DICT_ARUCO_MIP_36h12\n"
        << "  --seed <n>           random seed; default 20260827\n"
        << "  --canonical-px <n>   canonical resolution for rendering markers; "
           "default 512\n"
        << "  --list-presets       print the preset names and exit\n"
        << "  --help               print this help and exit\n";
}

/// Confirms that the entire string parses as an integer.
///
/// std::stoi ignores trailing characters, so "512xyz" would be silently accepted as
/// 512. argv is external input and is not trusted, so a full-string match is
/// required.
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

/// Confirms that the entire string parses as an unsigned integer.
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
            std::cerr << "missing argument for: " << option << '\n';
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
                    std::cerr << "--seed must be an unsigned integer: " << value << '\n';
                    return EXIT_FAILURE;
                }
            } else if (option == "--canonical-px") {
                int canonical_px = 0;
                if (!parse_int_strict(value, &canonical_px)) {
                    std::cerr << "--canonical-px must be an integer: " << value << '\n';
                    return EXIT_FAILURE;
                }
                // The upper bound comes from memory use. The range is kept in step
                // with validate_scene in corpus_generator.
                if (canonical_px < 8 || canonical_px > 8192) {
                    std::cerr << "--canonical-px must be between 8 and 8192 inclusive: "
                              << canonical_px << '\n';
                    return EXIT_FAILURE;
                }
                config.canonical_marker_px_ = canonical_px;
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

    std::vector<aruco3cuda::corpusgen::SceneSpec> specs;
    if (!aruco3cuda::corpusgen::build_preset(preset, &specs)) {
        std::cerr << "unknown preset: " << preset << '\n'
                  << "--list-presets prints the supported names\n";
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
        std::cerr << "cannot open the manifest: " << manifest_path << '\n'
                  << "check that the destination directory exists\n";
        return EXIT_FAILURE;
    }
    aruco3cuda::corpusgen::write_manifest_json(manifest, config, preset, scenes);
    // Never report a failed write as a success. Without the manifest the corpus
    // cannot be referred to later.
    manifest.close();
    if (!manifest) {
        std::cerr << "writing the manifest failed: " << manifest_path << '\n';
        return EXIT_FAILURE;
    }

    std::size_t marker_total = 0;
    for (const auto& scene : scenes) {
        marker_total += scene.markers_.size();
    }
    std::cout << "generated: " << scenes.size() << " images, " << marker_total << " markers\n"
              << "manifest: " << manifest_path << '\n';
    return EXIT_SUCCESS;
}
