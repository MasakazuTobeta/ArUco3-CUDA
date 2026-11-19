// SPDX-License-Identifier: Apache-2.0
#include "quality_scenes.hpp"

#include <opencv2/imgcodecs.hpp>

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "aruco3cuda/util/json_writer.hpp"
#include "aruco3cuda/util/sha256.hpp"

namespace {
int integer(const char* text) {
    std::size_t consumed = 0;
    const int value = std::stoi(text, &consumed);
    if (consumed != std::string(text).size()) {
        throw std::invalid_argument("invalid dimension");
    }
    return value;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2 && argc != 4 && argc != 5) {
            throw std::invalid_argument(
                    "usage: aruco3cuda_cpu_quality_corpus OUTPUT_DIR [WIDTH HEIGHT "
                    "[APRILTAG_FAMILY]]");
        }
        const std::filesystem::path directory(argv[1]);
        if (std::filesystem::exists(directory) && !std::filesystem::is_empty(directory)) {
            throw std::invalid_argument("quality output directory must be empty");
        }
        const cv::Size size =
                argc >= 4 ? cv::Size(integer(argv[2]), integer(argv[3])) : cv::Size(641, 479);
        const std::string family = argc == 5 ? argv[4] : "DICT_ARUCO_MIP_36h12";
        cv::aruco::PredefinedDictionaryType type = cv::aruco::DICT_ARUCO_MIP_36h12;
        if (argc == 5) {
            if (family == "DICT_APRILTAG_16h5") {
                type = cv::aruco::DICT_APRILTAG_16h5;
            } else if (family == "DICT_APRILTAG_25h9") {
                type = cv::aruco::DICT_APRILTAG_25h9;
            } else if (family == "DICT_APRILTAG_36h10") {
                type = cv::aruco::DICT_APRILTAG_36h10;
            } else if (family == "DICT_APRILTAG_36h11") {
                type = cv::aruco::DICT_APRILTAG_36h11;
            } else {
                throw std::invalid_argument("unknown AprilTag family: " + family);
            }
        }
        const cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(type);
        const auto specs =
                aruco3cuda::cpu_quality::make_adversarial_specs(size, dictionary.bytesList.rows);
        std::filesystem::create_directories(directory);
        std::ofstream manifest(directory / "manifest.json");
        if (!manifest) {
            throw std::runtime_error("cannot create quality manifest");
        }
        aruco3cuda::util::JsonWriter writer(manifest, 2);
        writer.begin_object();
        writer.member_int("schema_version", 1);
        writer.member_string("generator", "cpu-quality-adversarial-v1");
        writer.member_string("opencv_version", CV_VERSION);
        writer.member_string("dictionary", family);
        writer.member_string("source_kind", "synthetic; not real camera images");
        writer.member_string("corner_convention",
                             "oriented outer pixel boundaries; canonical -0.5 through 255.5");
        writer.key("scenes");
        writer.begin_array();
        for (const auto& spec : specs) {
            const cv::Mat image = aruco3cuda::cpu_quality::render_frame(spec, dictionary);
            const std::filesystem::path path =
                    directory / (spec.name_ + (argc == 5 ? ".pgm" : ".png"));
            if (!cv::imwrite(path.string(), image)) {
                throw std::runtime_error("cannot write quality image");
            }
            std::string digest;
            if (!aruco3cuda::util::sha256_file(path.string(), &digest)) {
                throw std::runtime_error("cannot hash quality image");
            }
            writer.begin_object();
            writer.member_string("name", spec.name_);
            writer.member_string("path", path.filename().string());
            writer.member_string("sha256", digest);
            writer.member_string("family", spec.family_);
            writer.member_int("width_px", size.width);
            writer.member_int("height_px", size.height);
            writer.member_int("seed", static_cast<long long>(spec.seed_));
            writer.member_bool("require_detection", spec.require_detection_);
            writer.member_double("blur_sigma_px", spec.blur_sigma_px_, 6);
            writer.member_double("noise_sigma_levels", spec.noise_sigma_levels_, 6);
            writer.member_double("contrast", spec.contrast_, 6);
            writer.member_double("illumination_gradient", spec.illumination_gradient_, 6);
            writer.member_double("occlusion_fraction", spec.occlusion_fraction_, 6);
            writer.member_int("background_kind", spec.background_kind_);
            writer.key("markers");
            writer.begin_array();
            for (const auto& marker : spec.markers_) {
                writer.begin_object();
                writer.member_int("id", marker.id_);
                writer.key("corners_px");
                writer.begin_array();
                for (const cv::Point2f& point : marker.corners_) {
                    writer.value_double(point.x, 9);
                    writer.value_double(point.y, 9);
                }
                writer.end_array();
                writer.end_object();
            }
            writer.end_array();
            writer.end_object();
        }
        writer.end_array();
        writer.end_object();
        manifest << '\n';
        if (!manifest) {
            throw std::runtime_error("cannot finish quality manifest");
        }
        std::cout << "Generated " << specs.size() << " synthetic quality scenes in " << directory
                  << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
