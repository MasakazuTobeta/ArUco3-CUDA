// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CORE_PYRAMID_SAMPLING_CUH
#define ARUCO3CUDA_CORE_PYRAMID_SAMPLING_CUH

#include "preprocess.hpp"

namespace aruco3cuda::detail {

/// Selects the canonical sampling level with the existing non-fused float rule.
///
/// @param pyramid Global level dimensions; no pixel data is read.
/// @param segmentation_width Positive segmentation width.
/// @param perimeter Current candidate's chain-code perimeter.
/// @param min_perimeter Minimum canonical perimeter.
/// @return Index of the level with the smallest strictly positive scaled distance.
///
/// Ownership: references values only and retains nothing.
/// Synchronization: device-local arithmetic, with no synchronization.
///
/// Example input: widths 1024, 512, 256; segmentation width 1024; perimeter 513; minimum 128
/// Example output: level 2
__device__ inline int choose_canonical_level(const PyramidRef& pyramid, int segmentation_width,
                                             int perimeter, int min_perimeter) {
    int optimal = 0;
    float distance = 3.402823466e+38F;
    for (int i = 0; i < pyramid.level_count_; ++i) {
        const float scale =
                static_cast<float>(pyramid.width_[i]) / static_cast<float>(segmentation_width);
        // The planner is compiled in a different translation unit from the warp.
        // Explicit rounding prevents compiler FMA settings from changing demand.
        const float scaled = __fmul_rn(static_cast<float>(perimeter), scale);
        const float value = __fsub_rn(scaled, static_cast<float>(min_perimeter));
        if (value < distance && value > 0.0F) {
            distance = value;
            optimal = i;
        }
    }
    return optimal;
}

}  // namespace aruco3cuda::detail

#endif  // ARUCO3CUDA_CORE_PYRAMID_SAMPLING_CUH
