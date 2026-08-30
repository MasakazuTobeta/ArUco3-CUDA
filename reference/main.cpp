// SPDX-License-Identifier: Apache-2.0
//
// CLI for the CPU baseline runner.
//
// Purpose:
//   Run OpenCV ArUco detection with a fixed configuration and store the results
//   in a machine-readable format. Produces the baseline used for diffing
//   against the CUDA implementation and for measuring the crossover point.
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
    out << "Usage: aruco3cuda_reference_runner [option]... --input <image>...\n"
        << "\n"
        << "  --input <path>                 Input image. May be given several times\n"
        << "  --output <path>                Destination of the result JSON. Default is stdout\n"
        << "  --dictionary <name>            Default DICT_ARUCO_MIP_36h12\n"
        << "  --threads <n>                  OpenCV thread count. Default 1. 0 keeps\n"
        << "                                 the OpenCV default\n"
        << "  --use-aruco3 <0|1>             ArUco3 detection strategy. Default 1\n"
        << "  --min-side-length-canonical <n>  Default 32\n"
        << "  --min-marker-length-ratio <f>  Default 0.05. No downscaling happens at 0\n"
        << "  --adaptive-thresh-win-min <n>  Default 3\n"
        << "  --adaptive-thresh-win-max <n>  Default 23\n"
        << "  --adaptive-thresh-win-step <n> Default 10\n"
        << "  --error-correction-rate <f>    Default 0.6\n"
        << "  --omit-timing                  Omit the execution time. For golden comparison\n"
        << "  --list-dictionaries            Print the supported dictionary names and exit\n"
        << "  --help                         Print this help and exit\n";
}

/// Take the next argument. Returns false when it is missing.
bool take_value(int argc, char** argv, int* index, const char* option, std::string* out) {
    if (*index + 1 >= argc) {
        std::cerr << "missing argument: " << option << '\n';
        return false;
    }
    ++(*index);
    *out = argv[*index];
    return true;
}

/// Confirm that a parsed integer lies within the given range.
///
/// argv is external input and is not trusted, so an out-of-range value is never
/// written into the settings. The range is validated on the CLI side as well so
/// that the user is told which option was invalid.
bool check_range(const std::string& option, int value, int minimum, int maximum) {
    if (value < minimum || value > maximum) {
        std::cerr << option << " must be between " << minimum << " and " << maximum
                  << " inclusive: " << value << '\n';
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

        // Handle the numeric options together.
        struct IntOption {
            const char* name;
            int* target;
            int minimum;
            int maximum;
        };
        // Keep the ranges consistent with validate_config in reference_runner.
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
                    std::cerr << "cannot parse as an integer: " << entry.name << ' ' << value
                              << '\n';
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
                std::cerr << "specify either 0 or 1: --use-aruco3\n";
                return EXIT_FAILURE;
            }
            config.use_aruco3_detection_ = int_value != 0;
            continue;
        }
        if (option == "--min-marker-length-ratio") {
            if (!take_value(argc, argv, &i, "--min-marker-length-ratio", &value) ||
                !parse_double(value, &double_value)) {
                std::cerr << "cannot parse as a real number: --min-marker-length-ratio\n";
                return EXIT_FAILURE;
            }
            if (!(double_value >= 0.0) || !(double_value <= 1.0)) {
                std::cerr << "--min-marker-length-ratio must be between 0 and 1 inclusive: "
                          << double_value << '\n';
                return EXIT_FAILURE;
            }
            config.min_marker_length_ratio_original_img_ = static_cast<float>(double_value);
            continue;
        }
        if (option == "--error-correction-rate") {
            if (!take_value(argc, argv, &i, "--error-correction-rate", &value) ||
                !parse_double(value, &double_value)) {
                std::cerr << "cannot parse as a real number: --error-correction-rate\n";
                return EXIT_FAILURE;
            }
            if (!(double_value >= 0.0) || !(double_value <= 1.0)) {
                std::cerr << "--error-correction-rate must be between 0 and 1 inclusive: "
                          << double_value << '\n';
                return EXIT_FAILURE;
            }
            config.error_correction_rate_ = double_value;
            continue;
        }

        std::cerr << "unknown option: " << option << '\n';
        print_usage(std::cerr);
        return EXIT_FAILURE;
    }

    if (inputs.empty()) {
        std::cerr << "--input was not specified\n";
        print_usage(std::cerr);
        return EXIT_FAILURE;
    }
    if (!aruco3cuda::reference::is_known_dictionary(config.dictionary_name_)) {
        std::cerr << "unsupported Dictionary: " << config.dictionary_name_ << '\n'
                  << "--list-dictionaries lists the supported names\n";
        return EXIT_FAILURE;
    }
    // Constraint imposed by OpenCV. With ArUco3 enabled, detectMarkers fails
    // when both are 0.
    if (config.use_aruco3_detection_ && config.min_side_length_canonical_img_px_ == 0 &&
        config.min_marker_length_ratio_original_img_ == 0.0F) {
        std::cerr << "with --use-aruco3 1, min-side-length-canonical and "
                     "min-marker-length-ratio cannot both be 0\n";
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
        // Never report a write failure as success. It happens for instance when
        // the pipe has been closed.
        std::cout.flush();
        if (!std::cout) {
            std::cerr << "failed to write to standard output\n";
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }
    std::ofstream output(output_path);
    if (!output) {
        std::cerr << "cannot open the output file: " << output_path << '\n';
        return EXIT_FAILURE;
    }
    aruco3cuda::reference::write_results_json(output, config, environment, results);
    // Close first and then check the state, because a write can still fail
    // while data is sitting in the buffer.
    output.close();
    if (!output) {
        std::cerr << "failed to write to the output file: " << output_path << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
