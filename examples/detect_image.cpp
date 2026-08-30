// SPDX-License-Identifier: Apache-2.0
//
// Sample: detect markers in a PGM image with the device-resident detector.
//
// Purpose:
//   Shows the whole public API in the order a caller uses it: size the workspace
//   from the input, initialize once, upload the image, issue the detection on a
//   stream, read the results on the device, and only then bring them back to the
//   host. The steps that cost a synchronization are the ones worth being able to
//   point at, so they are marked in the comments.
//
// Usage:
//   generate_marker --id 42 --output marker.pgm
//   detect_image --input marker.pgm
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/detections.hpp"
#include "aruco3cuda/detector.hpp"
#include "aruco3cuda/dictionary.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"
#include "aruco3cuda/version.hpp"

#include "pgm.hpp"

namespace {

using aruco3cuda::Status;
using aruco3cuda_examples::GrayImage;

/// Owns a pitched device allocation holding one grayscale image.
///
/// cudaMallocPitch is used rather than cudaMalloc because it pads each row to the
/// alignment the device prefers. That padding is exactly why ImageViewU8 carries
/// pitch_bytes_ separately from width_px_: assuming the two are equal works on the
/// host and then reads the wrong pixels on the device.
class DeviceImage {
public:
    DeviceImage() = default;
    ~DeviceImage() { this->release(); }
    DeviceImage(const DeviceImage&) = delete;
    DeviceImage& operator=(const DeviceImage&) = delete;
    DeviceImage(DeviceImage&&) = delete;
    DeviceImage& operator=(DeviceImage&&) = delete;

    /// Allocates device memory and copies the host image into it.
    bool upload(const GrayImage& image, std::string* out_message) {
        this->release();
        const std::size_t row_bytes = static_cast<std::size_t>(image.width_px_);
        cudaError_t error =
                cudaMallocPitch(reinterpret_cast<void**>(&this->data_), &this->pitch_bytes_,
                                row_bytes, static_cast<std::size_t>(image.height_px_));
        if (error != cudaSuccess) {
            *out_message = std::string("cudaMallocPitch failed: ") + cudaGetErrorString(error);
            this->data_ = nullptr;
            return false;
        }
        error = cudaMemcpy2D(this->data_, this->pitch_bytes_, image.pixels_.data(), row_bytes,
                             row_bytes, static_cast<std::size_t>(image.height_px_),
                             cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            *out_message = std::string("cudaMemcpy2D failed: ") + cudaGetErrorString(error);
            this->release();
            return false;
        }
        return true;
    }

    const std::uint8_t* data() const { return this->data_; }
    std::size_t pitch_bytes() const { return this->pitch_bytes_; }

private:
    void release() {
        if (this->data_ != nullptr) {
            // A destructor must not throw, and there is nothing left to recover
            // from at this point, so the status is deliberately dropped.
            static_cast<void>(cudaFree(this->data_));
            this->data_ = nullptr;
        }
        this->pitch_bytes_ = 0;
    }

    std::uint8_t* data_ = nullptr;
    std::size_t pitch_bytes_ = 0;
};

/// Owns a CUDA stream.
///
/// Passing an explicit stream is what lets the detector capture one frame worth of
/// launches as a CUDA Graph. CUDA cannot capture on the default stream, so running
/// with --default-stream takes the slower path that issues each step separately.
class CudaStream {
public:
    CudaStream() = default;
    ~CudaStream() {
        if (this->stream_ != nullptr) {
            static_cast<void>(cudaStreamDestroy(this->stream_));
        }
    }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    CudaStream(CudaStream&&) = delete;
    CudaStream& operator=(CudaStream&&) = delete;

    bool create(std::string* out_message) {
        const cudaError_t error = cudaStreamCreate(&this->stream_);
        if (error != cudaSuccess) {
            *out_message = std::string("cudaStreamCreate failed: ") + cudaGetErrorString(error);
            this->stream_ = nullptr;
            return false;
        }
        return true;
    }

    cudaStream_t get() const { return this->stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

void print_usage(std::ostream& out) {
    out << "Usage: detect_image --input <path.pgm> [option]...\n"
        << "\n"
        << "  --input <path>         Input image, binary PGM (P5). Required\n"
        << "  --dictionary <name>    Dictionary name. Default DICT_ARUCO_MIP_36h12\n"
        << "  --no-aruco3            Disable the ArUco3 strategy and corner refinement.\n"
        << "                         These two travel together: the detector rejects\n"
        << "                         any other combination of the pair\n"
        << "  --default-stream       Use the default stream instead of an explicit one.\n"
        << "                         CUDA cannot capture a graph on it, so this is the\n"
        << "                         slower path\n"
        << "  --repeat <n>           Run the detection n times. Default 1. Useful for\n"
        << "                         showing that the workspace stops allocating\n"
        << "  --help                 Print this help and exit\n";
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

/// Report a failed library call together with whatever context CUDA recorded.
void report(const char* what, Status status, const std::string& message) {
    std::cerr << what << " failed: " << aruco3cuda::to_string(status) << '\n';
    if (!message.empty()) {
        std::cerr << "  " << message << '\n';
    }
    const char* cuda_message = aruco3cuda::last_cuda_error_message();
    if (cuda_message != nullptr && cuda_message[0] != '\0') {
        std::cerr << "  " << cuda_message << '\n';
    }
}

double to_mib(std::size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

}  // namespace

int main(int argc, char** argv) {
    std::string input_path;
    std::string dictionary_name = "DICT_ARUCO_MIP_36h12";
    bool use_aruco3 = true;
    bool use_explicit_stream = true;
    int repeat = 1;

    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--help") {
            print_usage(std::cout);
            return EXIT_SUCCESS;
        }
        if (option == "--input") {
            if (!take_value(argc, argv, &i, "--input", &input_path)) {
                return EXIT_FAILURE;
            }
        } else if (option == "--dictionary") {
            if (!take_value(argc, argv, &i, "--dictionary", &dictionary_name)) {
                return EXIT_FAILURE;
            }
        } else if (option == "--no-aruco3") {
            use_aruco3 = false;
        } else if (option == "--default-stream") {
            use_explicit_stream = false;
        } else if (option == "--repeat") {
            std::string value;
            if (!take_value(argc, argv, &i, "--repeat", &value)) {
                return EXIT_FAILURE;
            }
            if (!parse_int(value, &repeat) || repeat < 1) {
                std::cerr << "--repeat must be a positive integer: " << value << '\n';
                return EXIT_FAILURE;
            }
        } else {
            std::cerr << "unknown option: " << option << '\n';
            print_usage(std::cerr);
            return EXIT_FAILURE;
        }
    }

    if (input_path.empty()) {
        std::cerr << "--input was not specified\n";
        return EXIT_FAILURE;
    }

    GrayImage image;
    std::string message;
    if (!aruco3cuda_examples::read_pgm(input_path, &image, &message)) {
        std::cerr << message << '\n';
        return EXIT_FAILURE;
    }

    const aruco3cuda::DictionaryTable* dictionary =
            aruco3cuda::find_builtin_dictionary(dictionary_name.c_str());
    if (dictionary == nullptr) {
        std::cerr << "unknown dictionary: " << dictionary_name << '\n';
        return EXIT_FAILURE;
    }

    // Size the workspace from this image rather than leaving the default of
    // 3840x2160. The workspace is allocated once for the worst case these limits
    // imply, so overstating them costs memory on every run.
    aruco3cuda::DetectorConfig config;
    config.max_width_px_ = image.width_px_;
    config.max_height_px_ = image.height_px_;
    if (!use_aruco3) {
        // The detector accepts only ArUco3 with refinement, or neither. With
        // ArUco3 on and refinement off the corners would stay in the coordinates
        // of the downscaled image.
        config.use_aruco3_detection_ = false;
        config.corner_refine_method_ = aruco3cuda::CornerRefineMethod::kNone;
    }

    aruco3cuda::Detector detector;
    // Synchronizes the whole device, once, so that the dictionary transfer is
    // visible from every stream afterwards.
    Status status = detector.initialize(*dictionary, config, &message);
    if (status != Status::kOk) {
        report("initialize", status, message);
        return EXIT_FAILURE;
    }

    DeviceImage device_image;
    if (!device_image.upload(image, &message)) {
        std::cerr << message << '\n';
        return EXIT_FAILURE;
    }

    CudaStream stream;
    if (use_explicit_stream && !stream.create(&message)) {
        std::cerr << message << '\n';
        return EXIT_FAILURE;
    }

    aruco3cuda::ImageViewU8 view;
    view.data_ = device_image.data();
    view.width_px_ = image.width_px_;
    view.height_px_ = image.height_px_;
    view.pitch_bytes_ = device_image.pitch_bytes();
    view.space_ = aruco3cuda::MemorySpace::kDevice;

    aruco3cuda::HostDetections result;
    for (int iteration = 0; iteration < repeat; ++iteration) {
        // Issues the kernels and returns; nothing has run yet at this point.
        status = detector.detect_async(view, stream.get(), &message);
        if (status != Status::kOk) {
            report("detect_async", status, message);
            return EXIT_FAILURE;
        }

        // The results are already addressable on the device here. A pose
        // estimation stage running on the same device would read them from this
        // struct and never touch the host.
        aruco3cuda::DeviceDetections on_device;
        status = detector.device_detections(&on_device);
        if (status != Status::kOk) {
            report("device_detections", status, std::string());
            return EXIT_FAILURE;
        }

        // Waits for the stream. In steady state this is the only synchronization
        // point, and it exists only because this sample prints from the host.
        status = detector.download(&result, stream.get(), &message);
        if (status != Status::kOk && status != Status::kMarkerOverflow) {
            report("download", status, message);
            return EXIT_FAILURE;
        }
    }

    const std::size_t detection_count = result.ids_.size();
    std::cout << "aruco3cuda   : " << aruco3cuda::version_string() << '\n'
              << "input        : " << input_path << " (" << image.width_px_ << "x"
              << image.height_px_ << ")\n"
              << "dictionary   : " << dictionary->name_ << " (" << dictionary->marker_size_ << "x"
              << dictionary->marker_size_ << " bits, " << dictionary->code_count_ << " ids)\n"
              << "configuration: ArUco3 " << (config.use_aruco3_detection_ ? "on" : "off")
              << ", corner refinement " << aruco3cuda::to_string(config.corner_refine_method_)
              << ", " << (use_explicit_stream ? "explicit stream" : "default stream") << ", "
              << repeat << " iteration(s)\n";

    if (config.use_aruco3_detection_) {
        // The single most common reason for finding nothing: the ArUco3 strategy
        // segments a downscaled image, so a marker below this side length in the
        // original image cannot be found at all.
        const double longest_side =
                static_cast<double>(std::max(image.width_px_, image.height_px_));
        const double lower_bound_px =
                static_cast<double>(config.min_side_length_canonical_img_px_) +
                longest_side * static_cast<double>(config.min_marker_length_ratio_original_img_);
        std::cout << "detectable   : markers of side " << std::fixed << std::setprecision(1)
                  << lower_bound_px << " px or more in this image\n";
    }

    const aruco3cuda::WorkspaceStatistics& workspace = detector.workspace_statistics();
    std::cout << "workspace    : " << workspace.allocation_count_ << " allocation(s), "
              << workspace.reallocation_count_ << " reallocation(s), peak " << std::fixed
              << std::setprecision(2) << to_mib(workspace.peak_used_bytes_) << " MiB\n";

    std::cout << "detections   : " << detection_count << " (accepted " << result.accepted_total_
              << ")\n";
    if (result.marker_overflow_) {
        std::cout << "  the results were truncated at max_markers_ = " << config.max_markers_
                  << '\n';
    }
    for (std::size_t index = 0; index < detection_count; ++index) {
        std::cout << "  [" << index << "] id=" << result.ids_[index]
                  << " rotation=" << result.rotations_[index] << " corners=";
        for (std::size_t corner = 0; corner < 4U; ++corner) {
            const std::size_t base = index * 8U + corner * 2U;
            std::cout << " (" << std::fixed << std::setprecision(2) << result.corners_[base] << ", "
                      << result.corners_[base + 1U] << ")";
        }
        std::cout << '\n';
    }
    return EXIT_SUCCESS;
}
