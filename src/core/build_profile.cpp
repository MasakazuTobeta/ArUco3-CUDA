// SPDX-License-Identifier: Apache-2.0
#include "aruco3cuda/config.hpp"

namespace aruco3cuda {
const GpuBuildStrategies& compiled_gpu_strategies() noexcept {
    static constexpr GpuBuildStrategies profile{
            ARUCO3CUDA_CONTOUR_STRATEGY_NAME,
            ARUCO3CUDA_THRESHOLD_STRATEGY_NAME,
            ARUCO3CUDA_PYRAMID_STRATEGY_NAME,
            ARUCO3CUDA_OTSU_STRATEGY_NAME,
            ARUCO3CUDA_CANDIDATE_GRID_STRATEGY_NAME,
            ARUCO3CUDA_RESIZE_COEFFICIENT_STRATEGY_NAME,
            "ORIGINAL_SLOTS",
            ARUCO3CUDA_CONTOUR_COUNT_STRATEGY_NAME,
            ARUCO3CUDA_CONTOUR_RDP_STRATEGY_NAME,
            ARUCO3CUDA_CONTOUR_MATERIALIZE_STRATEGY_NAME,
            ARUCO3CUDA_CONTOUR_MATERIALIZE_COMPACTION_STRATEGY_NAME,
            ARUCO3CUDA_OTSU_FIXED_POINT_MAX_OCCUPIED_BINS,
            ARUCO3CUDA_CANDIDATE_GRID_BLOCKS};
    return profile;
}
std::string compiled_gpu_strategy_identity() {
    const auto& p = compiled_gpu_strategies();
    return std::string("contour=") + p.contour_ + ";threshold=" + p.threshold_ +
           ";pyramid=" + p.pyramid_ + ";otsu=" + p.otsu_ +
           ";resize_coefficients=" + p.resize_coefficients_ +
           ";contour_topology=" + p.contour_topology_ + ";contour_count=" + p.contour_count_ +
           ";contour_rdp=" + p.contour_rdp_ + ";contour_materialize=" + p.contour_materialize_ +
           ";contour_materialize_compaction=" + p.contour_materialize_compaction_ +
           ";otsu_max_occupied_bins=" + std::to_string(p.otsu_max_occupied_bins_) +
           ";candidate_grid=" + p.candidate_grid_ +
           ";candidate_grid_blocks=" + std::to_string(p.candidate_grid_blocks_);
}
}  // namespace aruco3cuda
