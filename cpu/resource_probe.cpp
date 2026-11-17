// SPDX-License-Identifier: Apache-2.0
// Isolated process measurement of setup, first detection, throughput, and peak RSS.
#include <sys/resource.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "aruco3cuda/util/json_writer.hpp"
#include "cpu_detector.hpp"

namespace {
using Clock = std::chrono::steady_clock;
double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
int integer(const char* text) {
    std::size_t consumed = 0;
    const int value = std::stoi(text, &consumed);
    if (consumed != std::string(text).size())
        throw std::invalid_argument("invalid integer");
    return value;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 6) {
            throw std::invalid_argument(
                    "usage: aruco3cuda_cpu_resources IMAGE opencv|control|previous|optimized "
                    "ARUCO3 THREADS "
                    "FRAMES");
        }
        const std::string route = argv[2];
        const int aruco3 = integer(argv[3]);
        const int threads = integer(argv[4]);
        const int frames = integer(argv[5]);
        if (aruco3 < 0 || aruco3 > 1 || threads < 1 || frames < 1 ||
            (route != "opencv" && route != "control" && route != "previous" &&
             route != "optimized")) {
            throw std::invalid_argument("invalid route, strategy, thread count, or frame count");
        }
        cv::setNumThreads(threads);
        const cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
        if (image.empty())
            throw std::runtime_error("cannot load image");
        aruco3cuda::DetectorConfig config;
        config.use_aruco3_detection_ = aruco3 != 0;
        config.corner_refine_method_ = aruco3 ? aruco3cuda::CornerRefineMethod::kSubpix
                                              : aruco3cuda::CornerRefineMethod::kNone;
        std::unique_ptr<aruco3cuda::cpu::CpuDetector> candidate;
        std::unique_ptr<cv::aruco::ArucoDetector> reference;
        aruco3cuda::cpu::CpuResult result;
        std::string message;
        const auto setup_start = Clock::now();
        if (route == "opencv") {
            cv::aruco::DetectorParameters parameters;
            parameters.useAruco3Detection = aruco3 != 0;
            parameters.minSideLengthCanonicalImg = config.min_side_length_canonical_img_px_;
            parameters.minMarkerLengthRatioOriginalImg =
                    config.min_marker_length_ratio_original_img_;
            parameters.cornerRefinementMethod =
                    aruco3 ? cv::aruco::CORNER_REFINE_SUBPIX : cv::aruco::CORNER_REFINE_NONE;
            reference = std::make_unique<cv::aruco::ArucoDetector>(
                    cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_MIP_36h12),
                    parameters);
        } else {
            candidate = std::make_unique<aruco3cuda::cpu::CpuDetector>();
            aruco3cuda::cpu::CpuOptions options;
            if (route == "previous") {
                options.exact_dictionary_lookup_ = false;
                options.filtered_contours_ = false;
                options.sparse_pyramid_ = false;
                options.cached_refinement_ = false;
                options.spatial_grouping_ = false;
            }
            if (route == "control") {
                options.reuse_pyramid_ = false;
                options.packed_dictionary_ = false;
                options.exact_dictionary_lookup_ = false;
                options.fused_cells_ = false;
                options.multiwindow_threshold_ = false;
                options.parallel_windows_ = false;
                options.parallel_decode_ = false;
                options.filtered_contours_ = false;
                options.sparse_pyramid_ = false;
                options.cached_refinement_ = false;
                options.spatial_grouping_ = false;
            }
            if (candidate->initialize("DICT_ARUCO_MIP_36h12", config, &message, options) !=
                aruco3cuda::Status::kOk)
                throw std::runtime_error(message);
        }
        const double setup_ms = elapsed_ms(setup_start);
        const auto detect = [&] {
            if (reference)
                reference->detectMarkers(image, result.corners_, result.ids_);
            else if (candidate->detect(image, &result, &message) != aruco3cuda::Status::kOk)
                throw std::runtime_error(message);
        };
        const auto first_start = Clock::now();
        detect();
        const double first_ms = elapsed_ms(first_start);
        const double first_result_ms = elapsed_ms(setup_start);
        for (int warmup = 0; warmup < 30; ++warmup)
            detect();
        const auto throughput_start = Clock::now();
        for (int frame = 0; frame < frames; ++frame)
            detect();
        const double throughput_ms = elapsed_ms(throughput_start);
        struct rusage usage {};
        if (::getrusage(RUSAGE_SELF, &usage) != 0)
            throw std::runtime_error("getrusage failed");
        aruco3cuda::util::JsonWriter writer(std::cout, 0);
        writer.begin_object();
        writer.member_string("image", argv[1]);
        writer.member_string("route", route);
        writer.member_string("opencv_version", CV_VERSION);
        writer.member_bool("aruco3", aruco3 != 0);
        writer.member_int("threads", threads);
        writer.member_int("detections", static_cast<int>(result.ids_.size()));
        writer.member_int("warmup", 30);
        writer.member_int("throughput_frames", frames);
        writer.member_double("setup_ms", setup_ms, 9);
        writer.member_double("first_detection_ms", first_ms, 9);
        writer.member_double("time_to_first_result_ms", first_result_ms, 9);
        writer.member_double("throughput_fps", frames * 1000.0 / throughput_ms, 6);
        writer.member_int("peak_process_rss_kib", usage.ru_maxrss);
        writer.member_string("scope",
                             "Setup begins after image decoding; RSS includes the input and "
                             "OpenCV, with no compatibility oracle in the process");
        writer.end_object();
        std::cout << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
