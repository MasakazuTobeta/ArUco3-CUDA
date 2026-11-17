// SPDX-License-Identifier: Apache-2.0
#include "aruco3cuda/config.hpp"
#include "benchmark_harness.hpp"

#include <iostream>
#include <sstream>
#include <string>

int main() {
    // Pull the shared config object from the first static library in link order.
    // The CPU library compiles that file without CUDA-private build definitions.
    aruco3cuda::DetectorConfig config;
    if (config.validate() != aruco3cuda::Status::kOk) {
        return 1;
    }
    const std::string profile = aruco3cuda::compiled_resize_profile();
    const std::string fma = aruco3cuda::compiled_carotene_x_fma();
    if (profile != EXPECTED_RESIZE_PROFILE || fma != EXPECTED_CAROTENE_X_FMA ||
        aruco3cuda::resize_carotene_x_fma() != (fma == "FUSED")) {
        std::cerr << "static link order selected incorrect core descriptor: " << profile << "/"
                  << fma << " expected " << EXPECTED_RESIZE_PROFILE << "/"
                  << EXPECTED_CAROTENE_X_FMA << '\n';
        return 2;
    }
    const auto& strategies = aruco3cuda::compiled_gpu_strategies();
    if (std::string(strategies.contour_) != EXPECTED_CONTOUR ||
        std::string(strategies.threshold_) != EXPECTED_THRESHOLD ||
        std::string(strategies.pyramid_) != EXPECTED_PYRAMID ||
        std::string(strategies.otsu_) != EXPECTED_OTSU ||
        std::string(strategies.candidate_grid_) != EXPECTED_GRID ||
        strategies.otsu_max_occupied_bins_ != EXPECTED_OTSU_BINS ||
        strategies.candidate_grid_blocks_ != EXPECTED_GRID_BLOCKS) {
        std::cerr << "compiled strategy profile disagrees with CMake selection\n";
        return 4;
    }
    if (std::string(strategies.contour_count_) != EXPECTED_CONTOUR_COUNT ||
        std::string(strategies.contour_rdp_) != EXPECTED_CONTOUR_RDP ||
        std::string(strategies.contour_materialize_) != EXPECTED_CONTOUR_MATERIALIZE ||
        std::string(strategies.contour_materialize_compaction_) !=
                EXPECTED_CONTOUR_MATERIALIZE_COMPACTION ||
        std::string(strategies.contour_topology_) != "ORIGINAL_SLOTS" ||
        std::string(strategies.resize_coefficients_) != EXPECTED_RESIZE_COEFFICIENTS) {
        std::cerr << "compiled contour variant identity disagrees with selection\n";
        return 6;
    }
    aruco3cuda::bench::BenchmarkConfig benchmark;
    aruco3cuda::bench::MeasurementRecord record;
    std::ostringstream json;
    aruco3cuda::bench::write_measurement_line(json, benchmark, record);
    if (json.str().find(std::string("\"gpu_resize_profile\":\"") + EXPECTED_RESIZE_PROFILE +
                        "\"") == std::string::npos ||
        json.str().find(std::string("\"gpu_carotene_x_fma\":\"") + EXPECTED_CAROTENE_X_FMA +
                        "\"") == std::string::npos) {
        std::cerr << "benchmark metadata disagrees with requested CMake descriptor\n";
        return 3;
    }
    if (json.str().find(std::string("\"gpu_strategy_identity\":\"") +
                        aruco3cuda::compiled_gpu_strategy_identity() + "\"") == std::string::npos) {
        std::cerr << "benchmark strategy metadata missing\n";
        return 5;
    }
    std::cout << profile << "/" << aruco3cuda::compiled_resize_backend() << "/" << fma << '\n';
    return 0;
}
