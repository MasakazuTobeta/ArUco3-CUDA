// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "aruco3cuda/apriltag/small_cpu/detector.hpp"

namespace aruco3cuda::apriltag_small_cpu::detail {

// Native outer-border evidence required to retain the incoming geometry after a large grid fit.
struct GridBorderConfig {
    float minimum_side_px_ = 32.0F;
    float minimum_displacement_px_ = 1.25F;
    float minimum_contrast_ = 20.0F;
    int minimum_shared_samples_ = 9;
    float maximum_original_mean_square_ = 0.05F;
    float minimum_error_ratio_ = 1.75F;
    float minimum_mean_square_increase_ = 0.02F;
};

// Refines canonical corners from the decoded grid's oriented edges in the current Gray8 frame.
// Returns true only after verifying the fitted corners against the same oriented payload.
bool align_decoded_grid(const cv::Mat& gray, Detection& detection,
                        const GridBorderConfig& border = {});

// A verified rectangular initialization can recover a poorly localized isolated candidate.
bool align_isolated_grid(const cv::Mat& gray, Detection& detection);

// Uses straight or curved native edge crossings and verifies the oriented payload at the
// fitted intersections. The input corners are preserved if the measurements fail.
bool try_align_curved_grid(const cv::Mat& gray, Detection& detection);

// Fits a diffuse optical transition on the native oriented grid. The caller verifies the payload.
bool propose_diffuse_grid(const cv::Mat& gray, int id, int contrast,
                          std::array<cv::Point2f, 4>& corners);

// Measures localized low-contrast transitions. The caller verifies the oriented code.
bool propose_faint_grid(const cv::Mat& gray, int id, int minimum_contrast,
                        std::array<cv::Point2f, 4>& corners);

// Tries only the narrow integrated-edge fit on a sharp, well-localized decoded grid.
bool try_align_sharp_grid(const cv::Mat& gray, Detection& detection);

// Fits visible transitions of a partially clipped tag. The caller must verify
// the returned hypothesis using an erasure-aware decoder before accepting it.
bool propose_visible_grid(const cv::Mat& gray, const Detection& tag,
                          std::array<cv::Point2f, 4>& hypothesis);

// Fits the four outer borders from their signed transitions; failure preserves the proposal.
bool try_fit_sharp_border(const cv::Mat& gray, std::array<cv::Point2f, 4>& corners);

}  // namespace aruco3cuda::apriltag_small_cpu::detail
