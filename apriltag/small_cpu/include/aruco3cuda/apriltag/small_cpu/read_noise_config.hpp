// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace aruco3cuda::apriltag_small_cpu {

/// Local source verification on a matte background with spatially distributed camera noise.
struct ReadNoiseConfig {
    bool enabled_ = true;
    /// Required fraction of native high-pass residuals in each image quadrant.
    int minimum_residual_per_mille_ = 250;
    /// Extra span allowed in the raw midtones before checking their denoised distribution.
    int extra_scene_span_ = 40;
    /// Guide mask mean minus pixel threshold, in Gray8 levels.
    int guide_difference_ = 3;
    /// Source dark-to-light contrast after local filtering, in Gray8 levels.
    int minimum_contrast_ = 6;
    /// Area that must remain when chamfered outline proposals are reduced to four vertices.
    float minimum_hull_retention_ = .90F;
    /// Standard deviation on the half-resolution crop after area averaging.
    float crop_blur_sigma_ = .8F;
    /// Per-corner displacement bound in printed cells. Zero disables the internal-grid fit.
    float maximum_grid_shift_cells_ = .64F;
};
}  // namespace aruco3cuda::apriltag_small_cpu
