// SPDX-License-Identifier: Apache-2.0
// Synchronized ArUco latency on changing frames with one reusable input buffer.
#include <cuda_runtime_api.h>
#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/detector.hpp"
#include "aruco3cuda/dictionary.hpp"

namespace {

using Clock = std::chrono::steady_clock;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void require(aruco3cuda::Status status, const std::string& message) {
    if (status != aruco3cuda::Status::kOk) {
        throw std::runtime_error(std::string(aruco3cuda::to_string(status)) + ": " + message);
    }
}

struct Stream {
    Stream() {
        check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "create stream");
    }
    ~Stream() { static_cast<void>(cudaStreamDestroy(stream_)); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    cudaStream_t stream_{};
};

struct DeviceInput {
    DeviceInput(int width, int height) {
        check_cuda(cudaMallocPitch(&pixels_, &pitch_, static_cast<std::size_t>(width),
                                   static_cast<std::size_t>(height)),
                   "allocate input");
    }
    ~DeviceInput() { static_cast<void>(cudaFree(pixels_)); }
    DeviceInput(const DeviceInput&) = delete;
    DeviceInput& operator=(const DeviceInput&) = delete;

    void* pixels_ = nullptr;
    std::size_t pitch_ = 0;
};

int integer(const char* text, int lower, int upper) {
    const std::string value(text);
    std::size_t parsed = 0;
    const int result = std::stoi(value, &parsed);
    if (parsed != value.size() || result < lower || result > upper) {
        throw std::invalid_argument("out-of-range integer: " + value);
    }
    return result;
}

void observation(int cycle, std::size_t frame, double ms,
                 const aruco3cuda::HostDetections& result) {
    if (!std::isfinite(ms) || ms <= 0.0) {
        throw std::runtime_error("invalid elapsed time");
    }
    if (result.corners_.size() != 8U * result.ids_.size()) {
        throw std::runtime_error("invalid detection result");
    }
    std::cout << "{\"cycle\":" << cycle << ",\"frame\":" << frame << ",\"ms\":" << ms
              << ",\"detections\":[";
    for (std::size_t i = 0; i < result.ids_.size(); ++i) {
        std::cout << (i == 0 ? "" : ",") << "{\"id\":" << result.ids_[i] << ",\"corners\":[";
        for (std::size_t corner = 0; corner < 8U; ++corner) {
            const float value = result.corners_[i * 8U + corner];
            if (!std::isfinite(value)) {
                throw std::runtime_error("non-finite detection corner");
            }
            std::cout << (corner == 0 ? "" : ",") << value;
        }
        std::cout << "]}";
    }
    std::cout << "]}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 8) {
            throw std::invalid_argument(
                    "usage: aruco3cuda_gpu_sequence resident|host ARUCO3 WARMUP_PASSES "
                    "TIMED_PASSES MAX_CANDIDATES image image ...");
        }
        const std::string scope(argv[1]);
        if (scope != "resident" && scope != "host") {
            throw std::invalid_argument("expected resident or host input");
        }
        const bool resident = scope == "resident";
        const bool aruco3 = integer(argv[2], 0, 1) == 1;
        const int warmup = integer(argv[3], 0, 100000);
        const int measured = integer(argv[4], 1, 100000);
        aruco3cuda::DetectorConfig config;
        config.max_candidates_ = integer(argv[5], 1024, 65535);
        config.max_markers_ = 1024;
        config.use_aruco3_detection_ = aruco3;
        config.corner_refine_method_ = aruco3 ? aruco3cuda::CornerRefineMethod::kSubpix
                                              : aruco3cuda::CornerRefineMethod::kNone;

        std::vector<cv::Mat> images;
        for (int i = 6; i < argc; ++i) {
            cv::Mat image = cv::imread(argv[i], cv::IMREAD_GRAYSCALE);
            if (image.empty() || (!images.empty() && image.size() != images.front().size())) {
                throw std::runtime_error("unreadable image or mixed dimensions: " +
                                         std::string(argv[i]));
            }
            images.push_back(image);
        }
        cv::setNumThreads(1);
        config.max_width_px_ = images.front().cols;
        config.max_height_px_ = images.front().rows;
        const auto* dictionary = aruco3cuda::find_builtin_dictionary("DICT_ARUCO_MIP_36h12");
        if (dictionary == nullptr) {
            throw std::runtime_error("built-in ArUco Dictionary is missing");
        }

        Stream stream;
        DeviceInput input(config.max_width_px_, config.max_height_px_);
        const aruco3cuda::ImageViewU8 view{static_cast<const std::uint8_t*>(input.pixels_),
                                           config.max_width_px_, config.max_height_px_,
                                           input.pitch_, aruco3cuda::MemorySpace::kDevice};
        aruco3cuda::Detector detector;
        std::string message;
        require(detector.initialize(*dictionary, config, &message), message);
        aruco3cuda::HostDetections detections;
        std::cout.precision(std::numeric_limits<double>::max_digits10);
        for (int cycle = -warmup; cycle < measured; ++cycle) {
            for (std::size_t frame = 0; frame < images.size(); ++frame) {
                const cv::Mat& image = images[frame];
                auto start = Clock::now();
                check_cuda(cudaMemcpy2DAsync(input.pixels_, input.pitch_, image.data, image.step,
                                             static_cast<std::size_t>(image.cols),
                                             static_cast<std::size_t>(image.rows),
                                             cudaMemcpyHostToDevice, stream.stream_),
                           "upload input");
                if (resident) {
                    check_cuda(cudaStreamSynchronize(stream.stream_), "make input ready");
                    start = Clock::now();
                }
                require(detector.detect_async(view, stream.stream_, &message), message);
                if (resident) {
                    check_cuda(cudaStreamSynchronize(stream.stream_), "complete detection");
                } else {
                    require(detector.download(&detections, stream.stream_, &message), message);
                }
                const auto end = Clock::now();
                if (resident) {
                    require(detector.download(&detections, stream.stream_, &message), message);
                }
                if (cycle >= 0) {
                    observation(cycle, frame,
                                std::chrono::duration<double, std::milli>(end - start).count(),
                                detections);
                }
            }
        }
        if (!std::cout.flush()) {
            throw std::runtime_error("cannot write timing output");
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
