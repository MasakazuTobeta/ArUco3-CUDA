// SPDX-License-Identifier: Apache-2.0
//
// CLI of the difference report tool.
//
// Purpose:
//   Runs the CPU baseline implementation and the route under evaluation on the same
//   images and reports the differences classified by kind. A single agreement rate
//   cannot distinguish a slight corner displacement from a missed marker.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cuda_runtime_api.h>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"
#include "hybrid_detector.hpp"
#include "reference_runner.hpp"
#include "report_diff.hpp"

namespace {

using aruco3cuda::report::CompareConfig;

void print_usage(std::ostream& out) {
    out << "usage: aruco3cuda_report [option]... --input <image>...\n"
        << "\n"
        << "  --input <path>                 input image; may be given several times\n"
        << "  --output <path>                destination of the JSON report; none by default\n"
        << "  --dictionary <name>            default DICT_ARUCO_MIP_36h12\n"
        << "  --corner-tolerance-px <f>      corner tolerance; default 1.0\n"
        << "  --match-radius-ratio <f>       matching radius as a side-length ratio; "
           "default 0.5\n"
        << "  --use-aruco3 <0|1>             ArUco3 detection strategy; default 1\n"
        << "  --fail-on-diff                 exit with code 1 if there is any difference\n"
        << "  --help                         print this help and exit\n";
}

bool take_value(int argc, char** argv, int* index, const char* option, std::string* out) {
    if (*index + 1 >= argc) {
        std::cerr << "missing argument for: " << option << '\n';
        return false;
    }
    ++(*index);
    *out = argv[*index];
    return true;
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

bool parse_bool_flag(const std::string& text, bool* out) {
    if (text == "0") {
        *out = false;
        return true;
    }
    if (text == "1") {
        *out = true;
        return true;
    }
    return false;
}

/// Copies an image to the device and builds a view over it.
///
/// The route under evaluation takes a view of device memory, so a host image cannot
/// be passed as is. The transfer pitch is chosen independently of the source pitch,
/// which also confirms that a pitched input works.
class DeviceImage {
public:
    DeviceImage() = default;
    DeviceImage(const DeviceImage&) = delete;
    DeviceImage& operator=(const DeviceImage&) = delete;
    DeviceImage(DeviceImage&&) = delete;
    DeviceImage& operator=(DeviceImage&&) = delete;
    ~DeviceImage() {
        if (this->data_ != nullptr) {
            static_cast<void>(cudaFree(this->data_));
        }
    }

    bool upload(const cv::Mat& image) {
        this->pitch_bytes_ = static_cast<std::size_t>(image.cols);
        if (cudaMalloc(&this->data_, this->pitch_bytes_ * static_cast<std::size_t>(image.rows)) !=
            cudaSuccess) {
            return false;
        }
        const cudaError_t status = cudaMemcpy2D(
                this->data_, this->pitch_bytes_, image.data, static_cast<std::size_t>(image.step),
                static_cast<std::size_t>(image.cols), static_cast<std::size_t>(image.rows),
                cudaMemcpyHostToDevice);
        if (status != cudaSuccess) {
            return false;
        }
        this->view_.data_ = static_cast<const std::uint8_t*>(this->data_);
        this->view_.width_px_ = image.cols;
        this->view_.height_px_ = image.rows;
        this->view_.pitch_bytes_ = this->pitch_bytes_;
        this->view_.space_ = aruco3cuda::MemorySpace::kDevice;
        return true;
    }

    const aruco3cuda::ImageViewU8& view() const { return this->view_; }

private:
    void* data_ = nullptr;
    std::size_t pitch_bytes_ = 0;
    aruco3cuda::ImageViewU8 view_;
};

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> inputs;
    std::string output_path;
    std::string dictionary_name = "DICT_ARUCO_MIP_36h12";
    CompareConfig compare_config;
    bool use_aruco3 = true;
    bool fail_on_diff = false;

    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        std::string value;
        if (argument == "--help") {
            print_usage(std::cout);
            return EXIT_SUCCESS;
        }
        if (argument == "--fail-on-diff") {
            fail_on_diff = true;
            continue;
        }
        if (argument == "--input") {
            if (!take_value(argc, argv, &i, "--input", &value)) {
                return EXIT_FAILURE;
            }
            inputs.push_back(value);
            continue;
        }
        if (argument == "--output") {
            if (!take_value(argc, argv, &i, "--output", &output_path)) {
                return EXIT_FAILURE;
            }
            continue;
        }
        if (argument == "--dictionary") {
            if (!take_value(argc, argv, &i, "--dictionary", &dictionary_name)) {
                return EXIT_FAILURE;
            }
            continue;
        }
        if (argument == "--corner-tolerance-px") {
            if (!take_value(argc, argv, &i, "--corner-tolerance-px", &value) ||
                !parse_double(value, &compare_config.corner_tolerance_px_) ||
                compare_config.corner_tolerance_px_ < 0.0) {
                std::cerr << "--corner-tolerance-px must be a real number >= 0\n";
                return EXIT_FAILURE;
            }
            continue;
        }
        if (argument == "--match-radius-ratio") {
            if (!take_value(argc, argv, &i, "--match-radius-ratio", &value) ||
                !parse_double(value, &compare_config.match_radius_ratio_) ||
                compare_config.match_radius_ratio_ <= 0.0) {
                std::cerr << "--match-radius-ratio must be a positive real number\n";
                return EXIT_FAILURE;
            }
            continue;
        }
        if (argument == "--use-aruco3") {
            if (!take_value(argc, argv, &i, "--use-aruco3", &value) ||
                !parse_bool_flag(value, &use_aruco3)) {
                std::cerr << "--use-aruco3 must be 0 or 1\n";
                return EXIT_FAILURE;
            }
            continue;
        }
        std::cerr << "unknown option: " << argument << '\n';
        print_usage(std::cerr);
        return EXIT_FAILURE;
    }

    if (inputs.empty()) {
        std::cerr << "no --input was given\n";
        print_usage(std::cerr);
        return EXIT_FAILURE;
    }

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        std::cerr << "no CUDA device found\n";
        return EXIT_FAILURE;
    }

    aruco3cuda::DetectorConfig detector_config;
    detector_config.use_aruco3_detection_ = use_aruco3;
    if (!use_aruco3) {
        detector_config = aruco3cuda::DetectorConfig::opencv_defaults();
    }

    aruco3cuda::reference::ReferenceConfig reference_config;
    reference_config.dictionary_name_ = dictionary_name;
    reference_config.use_aruco3_detection_ = detector_config.use_aruco3_detection_;
    reference_config.min_side_length_canonical_img_px_ =
            detector_config.min_side_length_canonical_img_px_;
    reference_config.min_marker_length_ratio_original_img_ =
            detector_config.min_marker_length_ratio_original_img_;
    reference_config.use_corner_subpix_refinement_ =
            detector_config.corner_refine_method_ == aruco3cuda::CornerRefineMethod::kSubpix;

    std::vector<aruco3cuda::report::ImageComparison> comparisons;
    for (const std::string& path : inputs) {
        const cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            std::cerr << "cannot read image: " << path << '\n';
            return EXIT_FAILURE;
        }

        aruco3cuda::reference::ReferenceResult reference_result;
        std::string error;
        if (!aruco3cuda::reference::detect_image(path, reference_config, &reference_result,
                                                 &error)) {
            std::cerr << "CPU baseline detection failed: " << error << '\n';
            return EXIT_FAILURE;
        }

        DeviceImage device;
        if (!device.upload(image)) {
            std::cerr << "transfer to the device failed: " << path << '\n';
            return EXIT_FAILURE;
        }
        aruco3cuda::hybrid::HybridDetector detector;
        std::string message;
        if (detector.initialize(detector_config, dictionary_name, image.cols, image.rows,
                                &message) != aruco3cuda::Status::kOk) {
            std::cerr << "initialization of the route under evaluation failed: " << message << '\n';
            return EXIT_FAILURE;
        }
        aruco3cuda::hybrid::HybridResult hybrid_result;
        if (detector.detect(device.view(), &hybrid_result, &message) != aruco3cuda::Status::kOk) {
            std::cerr << "detection by the route under evaluation failed: " << message << '\n';
            return EXIT_FAILURE;
        }

        std::vector<aruco3cuda::report::Detection> baseline;
        baseline.reserve(reference_result.detections_.size());
        for (const auto& detection : reference_result.detections_) {
            baseline.push_back({detection.id_, detection.corners_});
        }
        std::vector<aruco3cuda::report::Detection> target;
        target.reserve(hybrid_result.detections_.size());
        for (const auto& detection : hybrid_result.detections_) {
            target.push_back({detection.id_, detection.corners_});
        }
        comparisons.push_back(
                aruco3cuda::report::compare_detections(path, baseline, target, compare_config));
    }

    const aruco3cuda::report::Summary summary = aruco3cuda::report::summarize(comparisons);
    aruco3cuda::report::write_text_report(std::cout, comparisons, summary);

    if (!output_path.empty()) {
        std::ofstream output(output_path);
        if (!output.is_open()) {
            std::cerr << "cannot open the output destination: " << output_path << '\n';
            return EXIT_FAILURE;
        }
        aruco3cuda::report::write_json_report(output, comparisons, summary, compare_config);
        if (!output) {
            std::cerr << "writing the output failed: " << output_path << '\n';
            return EXIT_FAILURE;
        }
    }

    if (fail_on_diff && summary.agreed_image_count_ != summary.image_count_) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
