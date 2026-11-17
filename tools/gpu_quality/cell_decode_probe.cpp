// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/util/json_writer.hpp"
#include "aruco3cuda/util/sha256.hpp"
#include "aruco3cuda/util/statistics.hpp"
#include "aruco3cuda/workspace.hpp"
#include "candidate_filter.hpp"
#include "cell_decode.hpp"
#include "cell_sample.hpp"

namespace {
using aruco3cuda::Status;
constexpr int kWarmup = 50;
constexpr int kIterations = 150;
constexpr int kMarkerCells = 6;

void require(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void require(Status status, const char* operation) {
    if (status != Status::kOk) {
        throw std::runtime_error(std::string(operation) + ": " + aruco3cuda::to_string(status));
    }
}

// One case owns its handles. Construction is separate so partial initialization
// also reaches the destructor. Timed work always completes before output reads.
struct Handles {
    cudaStream_t stream_ = nullptr;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t executable_ = nullptr;
    cudaEvent_t begin_ = nullptr;
    cudaEvent_t end_ = nullptr;

    Handles() = default;
    Handles(const Handles&) = delete;
    Handles& operator=(const Handles&) = delete;
    ~Handles() {
        if (end_ != nullptr) {
            static_cast<void>(cudaEventDestroy(end_));
        }
        if (begin_ != nullptr) {
            static_cast<void>(cudaEventDestroy(begin_));
        }
        if (executable_ != nullptr) {
            static_cast<void>(cudaGraphExecDestroy(executable_));
        }
        if (graph_ != nullptr) {
            static_cast<void>(cudaGraphDestroy(graph_));
        }
        if (stream_ != nullptr) {
            static_cast<void>(cudaStreamDestroy(stream_));
        }
    }
};

template <typename T>
std::vector<T> download(const T* device, std::size_t count) {
    std::vector<T> host(count);
    if (count != 0U) {
        require(cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost),
                "download result");
    }
    return host;
}

template <typename T>
void hash_values(aruco3cuda::util::Sha256& hash, const std::vector<T>& values) {
    hash.update(values.data(), values.size() * sizeof(T));
}

void run_case(int count, int levels, const std::string& label, const cudaDeviceProp& device,
              int driver_version, int runtime_version) {
    Handles handles;
    require(cudaStreamCreateWithFlags(&handles.stream_, cudaStreamNonBlocking), "create stream");
    require(cudaEventCreate(&handles.begin_), "create begin event");
    require(cudaEventCreate(&handles.end_), "create end event");
    aruco3cuda::DetectorConfig config;
    config.max_candidates_ = std::max(count, 1);
    config.max_markers_ = config.max_candidates_;
    aruco3cuda::Workspace workspace;
    const std::size_t bytes = aruco3cuda::detail::candidate_workspace_bytes(config, 64, 64) +
                              aruco3cuda::detail::canonical_workspace_bytes(config, kMarkerCells) +
                              aruco3cuda::detail::cell_ratio_workspace_bytes(config, kMarkerCells);
    require(workspace.ensure_capacity(bytes, aruco3cuda::MemorySpace::kDevice), "reserve storage");
    workspace.reset();
    aruco3cuda::detail::CandidateFilterBuffers filter;
    aruco3cuda::detail::DeviceCandidates candidates;
    aruco3cuda::detail::CanonicalBuffers canonical;
    aruco3cuda::detail::CellRatioBuffers ratios;
    require(aruco3cuda::detail::reserve_candidates(config, 64, 64, workspace, &filter, &candidates),
            "reserve candidates");
    require(aruco3cuda::detail::reserve_canonical(config, kMarkerCells, workspace, &canonical),
            "reserve canonical images");
    require(aruco3cuda::detail::reserve_cell_ratios(config, kMarkerCells, workspace, &ratios),
            "reserve ratios");
    const auto capacity = static_cast<std::size_t>(config.max_candidates_);
    const auto side = static_cast<std::size_t>(canonical.side_px_);
    std::vector<std::uint8_t> pixels(side * side * capacity);
    for (std::size_t i = 0; i < pixels.size(); ++i) {
        pixels[i] = levels == 1 ? 127U
                                : static_cast<std::uint8_t>(
                                          ((i * 37U + i / 13U) % static_cast<std::size_t>(levels)) *
                                          255U / static_cast<std::size_t>(levels - 1));
    }
    require(cudaMemcpyAsync(canonical.images_, pixels.data(), pixels.size(), cudaMemcpyHostToDevice,
                            handles.stream_),
            "upload canonical images");
    require(cudaMemcpyAsync(candidates.count_, &count, sizeof(count), cudaMemcpyHostToDevice,
                            handles.stream_),
            "upload count");
    require(cudaStreamSynchronize(handles.stream_), "complete input preparation");
    require(cudaStreamBeginCapture(handles.stream_, cudaStreamCaptureModeThreadLocal), "capture");
    const Status submitted = aruco3cuda::detail::build_cell_ratios_async(
            canonical, candidates, config, kMarkerCells, &ratios, handles.stream_);
    // Always end capture, including a failed submission, before reporting it.
    const cudaError_t ended = cudaStreamEndCapture(handles.stream_, &handles.graph_);
    require(submitted, "capture cell decode");
    require(ended, "end capture");
#if CUDART_VERSION >= 12000
    require(cudaGraphInstantiate(&handles.executable_, handles.graph_, 0), "instantiate graph");
#else
    require(cudaGraphInstantiate(&handles.executable_, handles.graph_, nullptr, nullptr, 0),
            "instantiate graph");
#endif
    for (int i = 0; i < kWarmup; ++i) {
        require(cudaGraphLaunch(handles.executable_, handles.stream_), "warmup launch");
    }
    require(cudaStreamSynchronize(handles.stream_), "complete warmup");
    std::vector<double> samples_us;
    samples_us.reserve(kIterations);
    for (int i = 0; i < kIterations; ++i) {
        require(cudaEventRecord(handles.begin_, handles.stream_), "record begin");
        require(cudaGraphLaunch(handles.executable_, handles.stream_), "timed launch");
        require(cudaEventRecord(handles.end_, handles.stream_), "record end");
        require(cudaEventSynchronize(handles.end_), "complete timed launch");
        float elapsed_ms = 0.0F;
        require(cudaEventElapsedTime(&elapsed_ms, handles.begin_, handles.end_), "read interval");
        samples_us.push_back(static_cast<double>(elapsed_ms) * 1000.0);
    }
    const auto active = static_cast<std::size_t>(count);
    aruco3cuda::util::Sha256 output_hash;
    hash_values(output_hash, download(ratios.thresholds_, active));
    hash_values(output_hash, download(ratios.ratios_, active * 64U));
    hash_values(output_hash, download(ratios.border_errors_, active));
    hash_values(output_hash, download(ratios.accepted_, active));
    aruco3cuda::util::SampleStatistics statistics;
    if (!aruco3cuda::util::compute_statistics(samples_us, &statistics)) {
        throw std::runtime_error("empty timing samples");
    }
    aruco3cuda::util::JsonWriter writer(std::cout, 0);
    writer.begin_object();
    writer.member_string("scope", "cell-decode CUDA-event graph interval; not end-to-end");
    writer.member_string("variant", label);
    writer.member_string("gpu", device.name);
    writer.member_int("cuda_driver_version", driver_version);
    writer.member_int("cuda_runtime_version", runtime_version);
    writer.member_int("active_candidates", count);
    writer.member_int("capacity", config.max_candidates_);
    writer.member_int("intensity_levels", levels);
    writer.member_int("canonical_side_px", canonical.side_px_);
    writer.member_int("marker_cells", kMarkerCells);
    writer.member_int("warmup", kWarmup);
    writer.member_int("iterations", kIterations);
    writer.member_string("input_sha256",
                         aruco3cuda::util::sha256_bytes(pixels.data(), pixels.size()));
    writer.member_string("output_sha256", output_hash.finalize());
    writer.member_double("p50_us", statistics.p50_, 9);
    writer.member_double("p95_us", statistics.p95_, 9);
    writer.member_double("p99_us", statistics.p99_, 9);
    writer.key("samples_us");
    writer.begin_array();
    for (double sample : samples_us) {
        writer.value_double(sample, 9);
    }
    writer.end_array();
    writer.end_object();
    std::cout << '\n';
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 2) {
            throw std::invalid_argument("usage: cell_decode_probe [VARIANT_LABEL]");
        }
        const std::string label = argc == 2 ? argv[1] : "unspecified";
        int ordinal = 0;
        require(cudaGetDevice(&ordinal), "get device");
        cudaDeviceProp device{};
        require(cudaGetDeviceProperties(&device, ordinal), "get device properties");
        int driver_version = 0;
        int runtime_version = 0;
        require(cudaDriverGetVersion(&driver_version), "get driver version");
        require(cudaRuntimeGetVersion(&runtime_version), "get runtime version");
        for (int count : {0, 1, 8, 128}) {
            for (int levels : {1, 2, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256}) {
                run_case(count, levels, label, device, driver_version, runtime_version);
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
