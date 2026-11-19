// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "pgm.hpp"

namespace apriltag_benchmark {

using aruco3cuda_examples::GrayImage;
using Clock = std::chrono::steady_clock;

inline void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

inline int integer(std::string_view text, int lower, int upper) {
    int value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || value < lower ||
        value > upper) {
        throw std::runtime_error("integer outside the accepted range: " + std::string(text));
    }
    return value;
}

// Both routes receive the same source pixels at a stable device address across a
// sequence. RGB preparation is charged only to the native RGB route in host scope.
struct DeviceInput {
    cudaStream_t stream_ = nullptr;
    void* pixels_ = nullptr;
    size_t pitch_ = 0;

    ~DeviceInput() {
        if (this->stream_ != nullptr) {
            cudaStreamSynchronize(this->stream_);
        }
        if (this->pixels_ != nullptr) {
            cudaFree(this->pixels_);
        }
        if (this->stream_ != nullptr) {
            cudaStreamDestroy(this->stream_);
        }
    }
};

template <typename Backend>
int run(int argc, char** argv) {
    if (argc < 7) {
        throw std::runtime_error(
                "usage: benchmark device|host WARMUP_LOOPS TIMED_LOOPS "
                "ROUTE_PARAMETER MAX_TAGS frame.pgm [frame.pgm ...]");
    }
    const std::string scope = argv[1];
    if (scope != "device" && scope != "host") {
        throw std::runtime_error("scope must be device or host");
    }
    const bool host_scope = scope == "host";
    const int warmup = integer(argv[2], 0, 1000000);
    const int measured = integer(argv[3], 1, 1000000);
    const int parameter = integer(argv[4], Backend::kMinParameter, Backend::kMaxParameter);
    const int capacity = integer(argv[5], 1, 65535);
    std::vector<GrayImage> frames;
    for (int i = 6; i < argc; ++i) {
        GrayImage frame;
        std::string error;
        if (!aruco3cuda_examples::read_pgm(argv[i], &frame, &error)) {
            throw std::runtime_error(error);
        }
        if (!frames.empty() && (frame.width_px_ != frames.front().width_px_ ||
                                frame.height_px_ != frames.front().height_px_)) {
            throw std::runtime_error("input dimensions must match within one sequence");
        }
        frames.push_back(std::move(frame));
    }
    const auto width = static_cast<uint16_t>(frames.front().width_px_);
    const auto height = static_cast<uint16_t>(frames.front().height_px_);
    Backend detector(width, height, parameter, capacity);
    DeviceInput input;
    check_cuda(cudaStreamCreateWithFlags(&input.stream_, cudaStreamNonBlocking), "create stream");
    const size_t row = Backend::kBytesPerPixel * width;
    check_cuda(cudaMallocPitch(&input.pixels_, &input.pitch_, row, height), "allocate input");
    std::cout.precision(std::numeric_limits<double>::max_digits10);
    for (int cycle = -warmup; cycle < measured; ++cycle) {
        for (size_t i = 0; i < frames.size(); ++i) {
            auto start = Clock::now();
            const void* pixels = detector.prepare(frames[i].pixels_);
            check_cuda(cudaMemcpy2DAsync(input.pixels_, input.pitch_, pixels, row, row, height,
                                         cudaMemcpyHostToDevice, input.stream_),
                       "upload input");
            if (!host_scope) {
                check_cuda(cudaStreamSynchronize(input.stream_), "make device input ready");
                start = Clock::now();
            }
            detector.detect(input);
            check_cuda(cudaStreamSynchronize(input.stream_), "complete detect");
            const double elapsed_ms =
                    std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            const size_t count = detector.size();
            if (count >= static_cast<size_t>(capacity)) {
                throw std::runtime_error("possible output truncation: raise MAX_TAGS and repeat");
            }
            if (cycle < 0) {
                continue;
            }
            std::cout << "{\"backend\":\"" << Backend::kName << "\",\"cycle\":" << cycle
                      << ",\"frame\":" << i << ",\"scope\":\"" << scope
                      << "\",\"ms\":" << elapsed_ms << ",\"tags\":[";
            for (size_t j = 0; j < count; ++j) {
                std::cout << (j == 0 ? "" : ",") << "{\"id\":" << detector.id(j)
                          << ",\"corners\":[";
                for (size_t c = 0; c < 4; ++c) {
                    const auto point = detector.corner(j, c);
                    if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
                        throw std::runtime_error("non-finite detection corner");
                    }
                    std::cout << (c == 0 ? "" : ",") << "[" << point.x << "," << point.y << "]";
                }
                std::cout << "]}";
            }
            std::cout << "]}\n";
        }
    }
    if (!std::cout.flush()) {
        throw std::runtime_error("failed to write benchmark output");
    }
    return 0;
}

template <typename Backend>
int main_impl(int argc, char** argv) {
    try {
        return run<Backend>(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

}  // namespace apriltag_benchmark
