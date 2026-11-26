// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <opencv2/imgcodecs.hpp>

#include <charconv>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace apriltag_benchmark {
inline int integer(std::string_view arg, int low, int high) {
    int value = -1;
    const auto result = std::from_chars(arg.data(), arg.data() + arg.size(), value);
    if (result.ec != std::errc{} || result.ptr != arg.data() + arg.size() || value < low ||
        value > high)
        throw std::invalid_argument("invalid benchmark argument");
    return value;
}

template <class Corners>
void write_tag(std::size_t index, int id, const Corners& corners) {
    std::cout << "tag " << index << ' ' << id;
    for (const auto& point : corners)
        std::cout << ' ' << point.x << ' ' << point.y;
    std::cout << '\n';
}

template <class Backend, class PreparePass, class... Options>
int run_with_prepare(int argc, char** argv, PreparePass prepare_pass, bool reset_each_frame,
                     Options... options) {
    if (argc < 5)
        throw std::invalid_argument("usage: sequence STRIDE WARMUP PASSES image.pgm [...]");
    const int stride = integer(argv[1], 0, 8);
    const int warmup = integer(argv[2], 0, 1000000), passes = integer(argv[3], 1, 1000000);
    std::vector<cv::Mat> frames;
    for (int i = 4; i < argc; ++i) {
        auto image = cv::imread(argv[i], cv::IMREAD_UNCHANGED);
        if (image.empty() || image.type() != CV_8UC1)
            throw std::runtime_error("expected a Gray8 frame");
        frames.push_back(std::move(image));
    }
    cv::setNumThreads(1);
    Backend detector(stride, options...);
    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (int pass = -warmup; pass < passes; ++pass) {
        if (!reset_each_frame)
            prepare_pass(detector);
        for (std::size_t i = 0; i < frames.size(); ++i) {
            if (reset_each_frame)
                prepare_pass(detector);
            const auto start = std::chrono::steady_clock::now();
            detector.detect(frames[i]);
            const auto elapsed = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start);
            if (pass >= 0)
                std::cout << "ms " << i << ' ' << elapsed.count() << '\n';
            if (pass == 0)
                detector.write(i);
        }
    }
    return std::cout ? 0 : 1;
}

template <class Backend, class... Options>
int run(int argc, char** argv, Options... options) {
    return run_with_prepare<Backend>(argc, argv, [](Backend&) {}, false, options...);
}

template <class Backend>
int main_impl(int argc, char** argv) {
    try {
        return run<Backend>(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
}  // namespace apriltag_benchmark
