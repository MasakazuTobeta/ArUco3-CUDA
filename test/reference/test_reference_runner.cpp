// SPDX-License-Identifier: Apache-2.0
//
// Verifies the nominal, failure, and boundary behavior of the CPU reference runner.
//
// The synthetic images are generated inside the test. Markers are drawn at known
// positions, so the detection results can be judged against ground-truth corners.
#include "reference_runner.hpp"

#include <gtest/gtest.h>
#include <unistd.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using aruco3cuda::reference::ReferenceConfig;
using aruco3cuda::reference::ReferenceResult;

constexpr int kMarkerId = 42;
constexpr int kMarkerOriginX = 400;
constexpr int kMarkerOriginY = 260;
constexpr int kMarkerSidePx = 160;

/// Builds an image with one marker drawn at a known position and saves it to a path.
std::string write_synthetic_image(const std::string& name, int width_px, int height_px) {
    const cv::aruco::Dictionary dictionary =
            cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_MIP_36h12);
    cv::Mat marker;
    dictionary.generateImageMarker(kMarkerId, kMarkerSidePx, marker, 1);

    cv::Mat scene(height_px, width_px, CV_8UC1, cv::Scalar(255));
    marker.copyTo(scene(cv::Rect(kMarkerOriginX, kMarkerOriginY, kMarkerSidePx, kMarkerSidePx)));

    // Keep the path per process so that concurrent runs of the same binary do not collide.
    const std::string path = "/tmp/" + std::to_string(::getpid()) + "_" + name;
    EXPECT_TRUE(cv::imwrite(path, scene));
    return path;
}

class ReferenceRunnerTest : public ::testing::Test {
protected:
    void TearDown() override {
        for (const std::string& path : this->created_files_) {
            std::remove(path.c_str());
        }
    }
    std::string make_image(const std::string& name, int width_px = 1280, int height_px = 720) {
        const std::string path = write_synthetic_image(name, width_px, height_px);
        this->created_files_.push_back(path);
        return path;
    }

private:
    std::vector<std::string> created_files_;
};

// Nominal: the known marker is detected and its corners match where it was drawn.
TEST_F(ReferenceRunnerTest, detects_known_marker_at_expected_corners) {
    const std::string path = this->make_image("aruco3cuda_ref_basic.png");
    ReferenceConfig config;
    ReferenceResult result;
    std::string error;
    ASSERT_TRUE(aruco3cuda::reference::detect_image(path, config, &result, &error)) << error;

    ASSERT_EQ(result.detections_.size(), 1U);
    EXPECT_EQ(result.detections_[0].id_, kMarkerId);

    // Tolerance against the drawn position. Subpixel refinement is included, so about
    // one pixel is allowed.
    constexpr double kTolerancePx = 1.0;
    EXPECT_NEAR(result.detections_[0].corners_[0], kMarkerOriginX, kTolerancePx);
    EXPECT_NEAR(result.detections_[0].corners_[1], kMarkerOriginY, kTolerancePx);
    EXPECT_NEAR(result.detections_[0].corners_[4], kMarkerOriginX + kMarkerSidePx, kTolerancePx);
    EXPECT_NEAR(result.detections_[0].corners_[5], kMarkerOriginY + kMarkerSidePx, kTolerancePx);
}

// Nominal: the same input yields the same result.
TEST_F(ReferenceRunnerTest, results_are_deterministic) {
    const std::string path = this->make_image("aruco3cuda_ref_determinism.png");
    ReferenceConfig config;
    ReferenceResult first;
    ReferenceResult second;
    std::string error;
    ASSERT_TRUE(aruco3cuda::reference::detect_image(path, config, &first, &error)) << error;
    ASSERT_TRUE(aruco3cuda::reference::detect_image(path, config, &second, &error)) << error;

    ASSERT_EQ(first.detections_.size(), second.detections_.size());
    for (std::size_t i = 0; i < first.detections_.size(); ++i) {
        EXPECT_EQ(first.detections_[i].id_, second.detections_[i].id_);
        EXPECT_EQ(first.detections_[i].corners_, second.detections_[i].corners_);
    }
    EXPECT_EQ(first.image_sha256_, second.image_sha256_);
}

// Nominal: the effective ArUco3 downscale factor matches the observed specification.
// This pins the result to the formula recorded in docs/design/detector-pipeline.md.
TEST_F(ReferenceRunnerTest, reports_expected_aruco3_downscale) {
    const std::string path = this->make_image("aruco3cuda_ref_scale.png");
    ReferenceConfig config;
    config.use_aruco3_detection_ = true;
    config.min_side_length_canonical_img_px_ = 32;
    config.min_marker_length_ratio_original_img_ = 0.05F;

    ReferenceResult result;
    std::string error;
    ASSERT_TRUE(aruco3cuda::reference::detect_image(path, config, &result, &error)) << error;

    // fxfy = 32 / (32 + 1280 * 0.05) = 32 / 96 = 0.3333...
    EXPECT_NEAR(result.fxfy_effective_, 1.0 / 3.0, 1e-6);
    EXPECT_EQ(result.segmentation_width_px_, 427);
    EXPECT_EQ(result.segmentation_height_px_, 240);
}

// Boundary: with minMarkerLengthRatioOriginalImg at 0 no downscaling happens.
TEST_F(ReferenceRunnerTest, no_downscale_when_length_ratio_is_zero) {
    const std::string path = this->make_image("aruco3cuda_ref_noscale.png");
    ReferenceConfig config;
    config.use_aruco3_detection_ = true;
    config.min_marker_length_ratio_original_img_ = 0.0F;

    ReferenceResult result;
    std::string error;
    ASSERT_TRUE(aruco3cuda::reference::detect_image(path, config, &result, &error)) << error;
    EXPECT_DOUBLE_EQ(result.fxfy_effective_, 1.0);
    EXPECT_EQ(result.segmentation_width_px_, result.width_px_);
}

// Boundary: with ArUco3 disabled the scale factor is 1.
TEST_F(ReferenceRunnerTest, no_downscale_when_aruco3_disabled) {
    const std::string path = this->make_image("aruco3cuda_ref_disabled.png");
    ReferenceConfig config;
    config.use_aruco3_detection_ = false;
    ReferenceResult result;
    std::string error;
    ASSERT_TRUE(aruco3cuda::reference::detect_image(path, config, &result, &error)) << error;
    EXPECT_DOUBLE_EQ(result.fxfy_effective_, 1.0);
}

// Failure: a missing image fails and reports the reason.
TEST_F(ReferenceRunnerTest, fails_for_missing_image) {
    ReferenceConfig config;
    ReferenceResult result;
    std::string error;
    EXPECT_FALSE(
            aruco3cuda::reference::detect_image("/nonexistent/img.png", config, &result, &error));
    EXPECT_FALSE(error.empty());
}

// Failure: an unsupported dictionary fails.
TEST_F(ReferenceRunnerTest, fails_for_unknown_dictionary) {
    const std::string path = this->make_image("aruco3cuda_ref_dict.png");
    ReferenceConfig config;
    config.dictionary_name_ = "DICT_DOES_NOT_EXIST";
    ReferenceResult result;
    std::string error;
    EXPECT_FALSE(aruco3cuda::reference::detect_image(path, config, &result, &error));
    EXPECT_FALSE(error.empty());
}

// Failure: a null output pointer fails safely.
TEST_F(ReferenceRunnerTest, rejects_null_outputs) {
    ReferenceConfig config;
    ReferenceResult result;
    std::string error;
    EXPECT_FALSE(aruco3cuda::reference::detect_image("/tmp/x.png", config, nullptr, &error));
    EXPECT_FALSE(aruco3cuda::reference::detect_image("/tmp/x.png", config, &result, nullptr));
}

// Nominal: the environment is collected and the fields that must be recorded are filled in.
// If empty or wrong values were recorded with the results, the measurement conditions
// could not be reproduced afterwards.
TEST(ReferenceEnvironmentTest, collects_opencv_version_and_thread_count) {
    aruco3cuda::reference::ReferenceConfig config;
    config.num_threads_ = 1;
    const aruco3cuda::reference::ReferenceEnvironment environment =
            aruco3cuda::reference::collect_environment(config);
    EXPECT_FALSE(environment.opencv_version_.empty());
    // The thread count is pinned for reproducibility, so the value given must actually
    // take effect.
    EXPECT_EQ(environment.opencv_threads_, 1);
}

// Nominal: a thread count of 0 follows the OpenCV default.
TEST(ReferenceEnvironmentTest, thread_count_zero_uses_opencv_default) {
    aruco3cuda::reference::ReferenceConfig config;
    config.num_threads_ = 0;
    const aruco3cuda::reference::ReferenceEnvironment environment =
            aruco3cuda::reference::collect_environment(config);
    EXPECT_GT(environment.opencv_threads_, 0);
}

// Failure: an out-of-range setting returns false before detection runs.
// Passing an out-of-range value to OpenCV lets a cv::Exception escape and break the contract.
TEST(ReferenceConfigTest, rejects_out_of_range_values) {
    std::string error;
    aruco3cuda::reference::ReferenceConfig valid;
    EXPECT_TRUE(aruco3cuda::reference::validate_config(valid, &error)) << error;

    aruco3cuda::reference::ReferenceConfig small_window = valid;
    small_window.adaptive_thresh_win_size_min_px_ = 2;
    EXPECT_FALSE(aruco3cuda::reference::validate_config(small_window, &error));
    EXPECT_NE(error.find("adaptive_thresh_win_size_min"), std::string::npos) << error;

    aruco3cuda::reference::ReferenceConfig inverted = valid;
    inverted.adaptive_thresh_win_size_max_px_ = 3;
    inverted.adaptive_thresh_win_size_min_px_ = 21;
    EXPECT_FALSE(aruco3cuda::reference::validate_config(inverted, &error));

    aruco3cuda::reference::ReferenceConfig bad_rate = valid;
    bad_rate.error_correction_rate_ = 1.5;
    EXPECT_FALSE(aruco3cuda::reference::validate_config(bad_rate, &error));
    EXPECT_NE(error.find("error_correction_rate"), std::string::npos) << error;

    // Every comparison against NaN is false, so the range check has to reject it explicitly.
    aruco3cuda::reference::ReferenceConfig nan_rate = valid;
    nan_rate.min_otsu_std_dev_ = std::nan("");
    EXPECT_FALSE(aruco3cuda::reference::validate_config(nan_rate, &error));

    aruco3cuda::reference::ReferenceConfig zero_aruco3 = valid;
    zero_aruco3.use_aruco3_detection_ = true;
    zero_aruco3.min_side_length_canonical_img_px_ = 0;
    zero_aruco3.min_marker_length_ratio_original_img_ = 0.0F;
    EXPECT_FALSE(aruco3cuda::reference::validate_config(zero_aruco3, &error));

    EXPECT_FALSE(aruco3cuda::reference::validate_config(valid, nullptr));
}

// Failure: with an out-of-range setting, detect_image fails by returning false rather
// than by throwing.
TEST_F(ReferenceRunnerTest, detect_rejects_invalid_config_without_exception) {
    const std::string path = this->make_image("aruco3cuda_ref_badcfg.png");
    ReferenceConfig config;
    config.marker_border_bits_ = 0;
    ReferenceResult result;
    std::string error;
    EXPECT_FALSE(aruco3cuda::reference::detect_image(path, config, &result, &error));
    EXPECT_NE(error.find("marker_border_bits"), std::string::npos) << error;
}

// Nominal: dictionary names resolve.
TEST(ReferenceDictionaryTest, resolves_known_names_only) {
    EXPECT_TRUE(aruco3cuda::reference::is_known_dictionary("DICT_ARUCO_MIP_36h12"));
    EXPECT_TRUE(aruco3cuda::reference::is_known_dictionary("DICT_6X6_250"));
    EXPECT_FALSE(aruco3cuda::reference::is_known_dictionary("DICT_NOPE"));
    EXPECT_FALSE(aruco3cuda::reference::known_dictionary_names().empty());
}

// Nominal: the JSON output is well formed and carries the detections.
TEST_F(ReferenceRunnerTest, writes_json_containing_detections) {
    const std::string path = this->make_image("aruco3cuda_ref_json.png");
    ReferenceConfig config;
    ReferenceResult result;
    std::string error;
    ASSERT_TRUE(aruco3cuda::reference::detect_image(path, config, &result, &error)) << error;

    const aruco3cuda::reference::ReferenceEnvironment environment =
            aruco3cuda::reference::collect_environment(config);
    std::ostringstream out;
    aruco3cuda::reference::write_results_json(out, config, environment, {result});
    const std::string json = out.str();

    EXPECT_NE(json.find("\"schema_version\""), std::string::npos);
    EXPECT_NE(json.find("\"DICT_ARUCO_MIP_36h12\""), std::string::npos);
    EXPECT_NE(json.find("\"id\": 42"), std::string::npos);
    EXPECT_NE(json.find("\"fxfy_effective\""), std::string::npos);
    EXPECT_NE(json.find(result.image_sha256_), std::string::npos);
}

// Nominal: with the equivalent of --omit-timing set, the output is deterministic byte for byte.
TEST_F(ReferenceRunnerTest, json_is_byte_identical_when_timing_omitted) {
    const std::string path = this->make_image("aruco3cuda_ref_golden.png");
    ReferenceConfig config;
    config.omit_timing_ = true;

    auto render = [&]() {
        ReferenceResult result;
        std::string error;
        EXPECT_TRUE(aruco3cuda::reference::detect_image(path, config, &result, &error)) << error;
        const aruco3cuda::reference::ReferenceEnvironment environment =
                aruco3cuda::reference::collect_environment(config);
        std::ostringstream out;
        aruco3cuda::reference::write_results_json(out, config, environment, {result});
        return out.str();
    };
    const std::string first = render();
    EXPECT_EQ(first, render());
    EXPECT_EQ(first.find("detect_ms"), std::string::npos);
}

// Nominal: splitting the load off still gives the same result as detect_image.
//
// Measurements move the load into the initialization step. If that changed the
// result, what is being measured would no longer be the same thing.
TEST_F(ReferenceRunnerTest, detector_matches_detect_image) {
    const std::string path = this->make_image("aruco3cuda_ref_detector.png");
    ReferenceConfig config;

    ReferenceResult expected;
    std::string error;
    ASSERT_TRUE(aruco3cuda::reference::detect_image(path, config, &expected, &error)) << error;

    aruco3cuda::reference::ReferenceDetector detector;
    ASSERT_TRUE(detector.initialize(path, config, &error)) << error;
    EXPECT_EQ(detector.metadata().image_sha256_, expected.image_sha256_);
    EXPECT_EQ(detector.metadata().width_px_, expected.width_px_);
    EXPECT_EQ(detector.metadata().height_px_, expected.height_px_);
    EXPECT_EQ(detector.metadata().fxfy_effective_, expected.fxfy_effective_);
    // The metadata carries no detections.
    EXPECT_TRUE(detector.metadata().detections_.empty());

    // Repeated calls give the same result.
    for (int i = 0; i < 3; ++i) {
        ReferenceResult actual;
        ASSERT_TRUE(detector.detect(&actual, &error)) << error;
        ASSERT_EQ(actual.detections_.size(), expected.detections_.size());
        for (std::size_t k = 0; k < expected.detections_.size(); ++k) {
            EXPECT_EQ(actual.detections_[k].id_, expected.detections_[k].id_);
            EXPECT_EQ(actual.detections_[k].corners_, expected.detections_[k].corners_);
        }
        EXPECT_EQ(actual.image_sha256_, expected.image_sha256_);
        EXPECT_EQ(actual.rejected_count_, expected.rejected_count_);
    }
}

// Nominal: the detector remains usable after being moved.
TEST_F(ReferenceRunnerTest, detector_can_be_moved) {
    const std::string path = this->make_image("aruco3cuda_ref_detector_move.png");
    const ReferenceConfig config;
    std::string error;

    aruco3cuda::reference::ReferenceDetector source;
    ASSERT_TRUE(source.initialize(path, config, &error)) << error;

    aruco3cuda::reference::ReferenceDetector moved(std::move(source));
    ReferenceResult first;
    ASSERT_TRUE(moved.detect(&first, &error)) << error;
    EXPECT_FALSE(first.detections_.empty());

    aruco3cuda::reference::ReferenceDetector assigned;
    assigned = std::move(moved);
    ReferenceResult second;
    ASSERT_TRUE(assigned.detect(&second, &error)) << error;
    EXPECT_EQ(second.detections_.size(), first.detections_.size());
}

// Failure: detection before initialization, and invalid arguments, are rejected.
TEST_F(ReferenceRunnerTest, detector_rejects_invalid_use) {
    aruco3cuda::reference::ReferenceDetector detector;
    ReferenceResult result;
    std::string error;

    EXPECT_FALSE(detector.detect(&result, &error));
    EXPECT_NE(error.find("initialize"), std::string::npos);
    EXPECT_FALSE(detector.detect(nullptr, &error));
    EXPECT_FALSE(detector.detect(&result, nullptr));
    EXPECT_FALSE(detector.initialize("/nonexistent/image.png", ReferenceConfig(), nullptr));

    // An image that cannot be loaded.
    EXPECT_FALSE(detector.initialize("/nonexistent/image.png", ReferenceConfig(), &error));
    EXPECT_NE(error.find("cannot load image"), std::string::npos);

    // An unsupported dictionary.
    ReferenceConfig bad_dictionary;
    bad_dictionary.dictionary_name_ = "DICT_NOPE";
    EXPECT_FALSE(detector.initialize("/nonexistent/image.png", bad_dictionary, &error));
    EXPECT_NE(error.find("Dictionary"), std::string::npos);

    // A configuration that contradicts itself.
    ReferenceConfig bad_range;
    bad_range.max_marker_perimeter_rate_ = bad_range.min_marker_perimeter_rate_;
    EXPECT_FALSE(detector.initialize("/nonexistent/image.png", bad_range, &error));
    EXPECT_NE(error.find("inconsistent config"), std::string::npos);
}

}  // namespace
