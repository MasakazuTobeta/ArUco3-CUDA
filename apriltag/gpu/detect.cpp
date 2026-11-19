// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime_api.h>
#include <charconv>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include "aruco3cuda/apriltag/gpu/detector.hpp"
#include "aruco3cuda/apriltag_dictionary.hpp"
#include "pgm.hpp"

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
struct DeviceFrame {
    std::uint8_t* data_ = nullptr;
    std::size_t pitch_ = 0;
    DeviceFrame(int width, int height) {
        check(cudaMallocPitch(reinterpret_cast<void**>(&this->data_), &this->pitch_, width,
                              height));
    }
    ~DeviceFrame() { cudaFree(this->data_); }
};
}  // namespace
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: aruco3cuda_apriltag_gpu_detect DICT_APRILTAG_36h11 [--decimate N] "
                     "frame.pgm [frame.pgm ...]\n";
        return 2;
    }
    try {
        const auto* family = aruco3cuda::find_apriltag_dictionary(argv[1]);
        if (!family)
            throw std::invalid_argument("unknown AprilTag family");
        int width = 0, height = 0;
        std::unique_ptr<DeviceFrame> frame;
        std::unique_ptr<aruco3cuda::apriltag_gpu::Detector> detector;
        int first_image = 2;
        aruco3cuda::apriltag_gpu::Config cfg;
        if (std::string(argv[2]) == "--decimate") {
            if (argc < 5)
                throw std::invalid_argument("expected decimation and an image");
            const char* end = argv[3] + std::strlen(argv[3]);
            const auto parsed = std::from_chars(argv[3], end, cfg.decimate_);
            if (parsed.ec != std::errc{} || parsed.ptr != end)
                throw std::invalid_argument("invalid decimation");
            first_image = 4;
        }
        for (int i = first_image; i < argc; ++i) {
            aruco3cuda_examples::GrayImage image;
            std::string error;
            if (!aruco3cuda_examples::read_pgm(argv[i], &image, &error))
                throw std::runtime_error(error);
            if (width != image.width_px_ || height != image.height_px_) {
                width = image.width_px_;
                height = image.height_px_;
                frame = std::make_unique<DeviceFrame>(width, height);
                detector = std::make_unique<aruco3cuda::apriltag_gpu::Detector>(*family, width,
                                                                                height, cfg);
            }
            check(cudaMemcpy2D(frame->data_, frame->pitch_, image.pixels_.data(), width, width,
                               height, cudaMemcpyHostToDevice));
            const auto tags = detector->detect(frame->data_, frame->pitch_, nullptr);
            std::cout << "image," << argv[i] << '\n';
            for (const auto& tag : tags) {
                std::cout << "gpu_detection," << tag.id_;
                for (float value : tag.corners_)
                    std::cout << ',' << std::setprecision(9) << value;
                std::cout << '\n';
            }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}
