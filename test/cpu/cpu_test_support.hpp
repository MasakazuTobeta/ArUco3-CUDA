// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CPU_TEST_SUPPORT_HPP
#define ARUCO3CUDA_CPU_TEST_SUPPORT_HPP
#include <gtest/gtest.h>
#include <algorithm>
#include <cstddef>
#include <numeric>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <vector>
#include "cpu_detector.hpp"
namespace aruco3cuda::cpu::test_support {
using aruco3cuda::CornerRefineMethod;
inline cv::aruco::DetectorParameters reference_parameters(const DetectorConfig& config) {
    cv::aruco::DetectorParameters params;
    params.adaptiveThreshWinSizeMin = config.adaptive_thresh_win_size_min_px_;
    params.adaptiveThreshWinSizeMax = config.adaptive_thresh_win_size_max_px_;
    params.adaptiveThreshWinSizeStep = config.adaptive_thresh_win_size_step_px_;
    params.adaptiveThreshConstant = config.adaptive_thresh_constant_;
    params.minMarkerPerimeterRate = config.min_marker_perimeter_rate_;
    params.maxMarkerPerimeterRate = config.max_marker_perimeter_rate_;
    params.polygonalApproxAccuracyRate = config.polygonal_approx_accuracy_rate_;
    params.minCornerDistanceRate = config.min_corner_distance_rate_;
    params.minDistanceToBorder = config.min_distance_to_border_px_;
    params.minMarkerDistanceRate = config.min_marker_distance_rate_;
    params.minGroupDistance = static_cast<float>(config.min_group_distance_);
    params.markerBorderBits = config.marker_border_bits_;
    params.perspectiveRemovePixelPerCell = config.perspective_remove_pixel_per_cell_;
    params.perspectiveRemoveIgnoredMarginPerCell =
            config.perspective_remove_ignored_margin_per_cell_;
    params.maxErroneousBitsInBorderRate = config.max_erroneous_bits_in_border_rate_;
    params.minOtsuStdDev = config.min_otsu_std_dev_;
    params.errorCorrectionRate = config.error_correction_rate_;
    params.validBitIdThreshold = static_cast<float>(config.valid_bit_threshold_);
    params.cornerRefinementMethod = config.corner_refine_method_ == CornerRefineMethod::kSubpix
                                            ? cv::aruco::CORNER_REFINE_SUBPIX
                                            : cv::aruco::CORNER_REFINE_NONE;
    params.cornerRefinementWinSize = config.corner_refinement_win_size_px_;
    params.relativeCornerRefinmentWinSize =
            static_cast<float>(config.relative_corner_refinement_win_size_);
    params.cornerRefinementMaxIterations = config.corner_refinement_max_iterations_;
    params.cornerRefinementMinAccuracy = config.corner_refinement_min_accuracy_px_;
    params.useAruco3Detection = config.use_aruco3_detection_;
    params.minSideLengthCanonicalImg = config.min_side_length_canonical_img_px_;
    params.minMarkerLengthRatioOriginalImg = config.min_marker_length_ratio_original_img_;
    return params;
}

inline CpuResult reference_detect(const cv::Mat& image, const DetectorConfig& config,
                                  cv::aruco::PredefinedDictionaryType dictionary_id) {
    cv::aruco::ArucoDetector reference(cv::aruco::getPredefinedDictionary(dictionary_id),
                                       reference_parameters(config));
    CpuResult out;
    std::vector<std::vector<cv::Point2f>> rejected;
    reference.detectMarkers(image, out.corners_, out.ids_, rejected);
    out.rejected_count_ = rejected.size();
    return out;
}

inline std::vector<std::size_t> detection_order(const CpuResult& result) {
    std::vector<std::size_t> order(result.ids_.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        if (result.ids_[a] != result.ids_[b]) {
            return result.ids_[a] < result.ids_[b];
        }
        if (result.corners_[a][0].x != result.corners_[b][0].x) {
            return result.corners_[a][0].x < result.corners_[b][0].x;
        }
        return result.corners_[a][0].y < result.corners_[b][0].y;
    });
    return order;
}

inline void expect_same_detections(const CpuResult& actual, const CpuResult& expected) {
    ASSERT_EQ(actual.ids_.size(), actual.corners_.size());
    ASSERT_EQ(actual.ids_.size(), expected.ids_.size());
    for (const std::vector<cv::Point2f>& corner : actual.corners_) {
        ASSERT_EQ(corner.size(), 4U);
    }
    const std::vector<std::size_t> actual_order = detection_order(actual);
    const std::vector<std::size_t> expected_order = detection_order(expected);
    for (std::size_t i = 0; i < actual_order.size(); ++i) {
        const std::size_t a = actual_order[i];
        const std::size_t e = expected_order[i];
        ASSERT_EQ(actual.ids_[a], expected.ids_[e]);
        // Corner order contains the decoded rotation; comparing without a cyclic
        // shift catches orientation errors as well as inaccurate coordinates.
        for (std::size_t c = 0; c < 4U; ++c) {
            EXPECT_EQ(actual.corners_[a][c].x, expected.corners_[e][c].x);
            EXPECT_EQ(actual.corners_[a][c].y, expected.corners_[e][c].y);
        }
    }
    EXPECT_EQ(actual.rejected_count_, expected.rejected_count_);
}

inline cv::Mat marker_scene(cv::aruco::PredefinedDictionaryType dictionary_id, int marker_id,
                            int clockwise_quarter_turns, int border_bits = 1) {
    cv::Mat marker;
    cv::aruco::generateImageMarker(cv::aruco::getPredefinedDictionary(dictionary_id), marker_id,
                                   160, marker, border_bits);
    if (clockwise_quarter_turns == 1) {
        cv::rotate(marker, marker, cv::ROTATE_90_CLOCKWISE);
    } else if (clockwise_quarter_turns == 2) {
        cv::rotate(marker, marker, cv::ROTATE_180);
    } else if (clockwise_quarter_turns == 3) {
        cv::rotate(marker, marker, cv::ROTATE_90_COUNTERCLOCKWISE);
    }
    cv::Mat image(320, 320, CV_8UC1, cv::Scalar(255));
    marker.copyTo(image(cv::Rect(80, 80, marker.cols, marker.rows)));
    return image;
}

}  // namespace aruco3cuda::cpu::test_support
#endif
