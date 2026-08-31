// SPDX-License-Identifier: Apache-2.0
//
// Purpose:
//   Count the contour points of a corpus image, which is the quantity the
//   benchmark report identifies as setting the boundary between the CPU and the
//   GPU routes.
//
// Why this exists:
//   The counting procedure was written down in docs/benchmark-report.md but not
//   provided as a tool, and the regression input was not kept. That left the
//   published coefficients unverifiable by a reader, and it left the analysis
//   stuck at the three machines it was first run on. This tool closes both: it
//   is the procedure, and its output regenerates the regression input from the
//   corpus, which is reproducible from its seed.
//
// The procedure, unchanged from the report:
//   1. Downscale by the ArUco3 factor fxfy = S / (S + max(W, H) * tau) to obtain
//      the segmentation image.
//   2. Adaptive threshold it three times, with window 3, 13 and 23
//      (ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, constant 7).
//   3. findContours on each with RETR_LIST and CHAIN_APPROX_NONE, summing the
//      point counts of every contour.
//   4. Sum over the three windows.
//
//   The window sizes and the constant are the defaults of DetectorConfig, so
//   they are read from there rather than repeated here.
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "aruco3cuda/config.hpp"

namespace {

void print_usage(std::ostream& out) {
    out << "Usage: aruco3cuda_contourcount --input <image>... [option]...\n"
        << "\n"
        << "  --input <path>                 Corpus image. May be given several times\n"
        << "  --output <path>                Destination JSON. Default is stdout\n"
        << "  --min-side-length-canonical <n>  ArUco3 S. Default from DetectorConfig\n"
        << "  --min-marker-length-ratio <f>  ArUco3 tau. Default from DetectorConfig\n"
        << "  --help                         Print this help and exit\n";
}

bool take_value(int argc, char** argv, int* index, const char* option, std::string* out) {
    if (*index + 1 >= argc) {
        std::cerr << "missing argument: " << option << '\n';
        return false;
    }
    ++(*index);
    *out = argv[*index];
    return true;
}

/// The downscale factor the ArUco3 strategy applies before segmentation.
///
/// Identical to the rule in the detector: a factor of 1 means no downscaling,
/// and the factor never exceeds 1.
/// The segmentation image size, computed exactly as detail::plan_scales does.
///
/// Both halves of this matter and neither is obvious.
///
/// The factor is computed in float and the size rounded with lrint, because
/// that is what the detector does: a double would land on the other side of a
/// rounding boundary and give a size one pixel different.
///
/// The size, not the factor, is what gets handed to cv::resize. OpenCV
/// recomputes the interpolation coefficients as dst/src when it is given a
/// destination size, and uses the factor as passed when it is given one, so the
/// two produce different pixels at the same output size. The detector takes the
/// first path, and src/core/preprocess.cu says so where it computes
/// inverse_scale = dst / src. Passing the factor here instead changes the
/// contour count by up to a factor of three on a high-texture scene.
cv::Size segmentation_size(const aruco3cuda::DetectorConfig& config, int width_px, int height_px) {
    const float side = static_cast<float>(config.min_side_length_canonical_img_px_);
    const float longest = static_cast<float>(width_px > height_px ? width_px : height_px);
    const float denominator = side + longest * config.min_marker_length_ratio_original_img_;
    if (denominator <= 0.0F) {
        return cv::Size(width_px, height_px);
    }
    const float fxfy = side / denominator;
    if (fxfy >= 1.0F) {
        return cv::Size(width_px, height_px);
    }
    const int scaled_width =
            static_cast<int>(std::lrint(static_cast<double>(fxfy * static_cast<float>(width_px))));
    const int scaled_height =
            static_cast<int>(std::lrint(static_cast<double>(fxfy * static_cast<float>(height_px))));
    return cv::Size(scaled_width < 1 ? 1 : scaled_width, scaled_height < 1 ? 1 : scaled_height);
}

/// Total number of contour points over the three adaptive threshold windows.
long long count_contour_points(const cv::Mat& gray, const aruco3cuda::DetectorConfig& config,
                               long long* out_per_window) {
    long long total = 0;
    int index = 0;
    for (int window = config.adaptive_thresh_win_size_min_px_;
         window <= config.adaptive_thresh_win_size_max_px_;
         window += config.adaptive_thresh_win_size_step_px_) {
        // OpenCV requires an odd window of at least 3.
        const int odd = (window % 2 == 0) ? window + 1 : window;
        cv::Mat binary;
        cv::adaptiveThreshold(gray, binary, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV,
                              odd, config.adaptive_thresh_constant_);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
        long long window_total = 0;
        for (const std::vector<cv::Point>& contour : contours) {
            window_total += static_cast<long long>(contour.size());
        }
        if (out_per_window != nullptr && index < 3) {
            out_per_window[index] = window_total;
        }
        total += window_total;
        ++index;
    }
    return total;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> inputs;
    std::string output_path;
    aruco3cuda::DetectorConfig config;

    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        std::string value;
        if (option == "--help") {
            print_usage(std::cout);
            return EXIT_SUCCESS;
        }
        if (option == "--input") {
            if (!take_value(argc, argv, &i, "--input", &value)) {
                return EXIT_FAILURE;
            }
            inputs.push_back(value);
        } else if (option == "--output") {
            if (!take_value(argc, argv, &i, "--output", &output_path)) {
                return EXIT_FAILURE;
            }
        } else if (option == "--min-side-length-canonical") {
            if (!take_value(argc, argv, &i, option.c_str(), &value)) {
                return EXIT_FAILURE;
            }
            try {
                config.min_side_length_canonical_img_px_ = std::stoi(value);
            } catch (const std::exception&) {
                std::cerr << option << " is not an integer: " << value << '\n';
                return EXIT_FAILURE;
            }
        } else if (option == "--min-marker-length-ratio") {
            if (!take_value(argc, argv, &i, option.c_str(), &value)) {
                return EXIT_FAILURE;
            }
            try {
                config.min_marker_length_ratio_original_img_ = std::stof(value);
            } catch (const std::exception&) {
                std::cerr << option << " is not a number: " << value << '\n';
                return EXIT_FAILURE;
            }
        } else {
            std::cerr << "unknown option: " << option << '\n';
            print_usage(std::cerr);
            return EXIT_FAILURE;
        }
    }

    if (inputs.empty()) {
        std::cerr << "--input was not specified\n";
        return EXIT_FAILURE;
    }

    std::string json = "{\n  \"schema_version\": 1,\n  \"scenes\": [\n";
    bool first = true;
    for (const std::string& path : inputs) {
        const cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            std::cerr << "cannot read the image: " << path << '\n';
            return EXIT_FAILURE;
        }
        const cv::Size size = segmentation_size(config, image.cols, image.rows);
        cv::Mat segmentation;
        if (size.width == image.cols && size.height == image.rows) {
            segmentation = image;
        } else {
            // INTER_LINEAR, matching the detector: src/core/preprocess.cu reproduces
            // OpenCV's INTER_LINEAR resize down to its fixed-point coefficients.
            cv::resize(image, segmentation, size, 0.0, 0.0, cv::INTER_LINEAR);
        }
        const double scale = static_cast<double>(size.width) / static_cast<double>(image.cols);
        long long per_window[3] = {0, 0, 0};
        const long long total = count_contour_points(segmentation, config, per_window);

        char buffer[768];
        std::snprintf(buffer, sizeof(buffer),
                      "%s    {\"path\": \"%s\", \"width_px\": %d, \"height_px\": %d, "
                      "\"fxfy\": %.6f, \"segmentation_width_px\": %d, "
                      "\"segmentation_height_px\": %d, \"contour_points\": %lld, "
                      "\"per_window\": [%lld, %lld, %lld]}",
                      first ? "" : ",\n", path.c_str(), image.cols, image.rows, scale,
                      segmentation.cols, segmentation.rows, total, per_window[0], per_window[1],
                      per_window[2]);
        json += buffer;
        first = false;
    }
    json += "\n  ]\n}\n";

    if (output_path.empty()) {
        std::cout << json;
    } else {
        std::ofstream out(output_path, std::ios::trunc);
        if (!out.is_open()) {
            std::cerr << "cannot open the file for writing: " << output_path << '\n';
            return EXIT_FAILURE;
        }
        out << json;
        if (!out.good()) {
            std::cerr << "writing failed: " << output_path << '\n';
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
