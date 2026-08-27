// SPDX-License-Identifier: Apache-2.0
//
// CPU 基準 runner の正常系、異常系、境界値を検証する。
//
// 合成画像は test 内で生成する。既知の配置へマーカーを描画するため、
// 四隅の ground truth を持った状態で検出結果を評価できる。
#include "reference_runner.hpp"

#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

namespace {

using aruco3cuda::reference::ReferenceConfig;
using aruco3cuda::reference::ReferenceResult;

constexpr int kMarkerId = 42;
constexpr int kMarkerOriginX = 400;
constexpr int kMarkerOriginY = 260;
constexpr int kMarkerSidePx = 160;

/// 既知の位置へ 1 個のマーカーを描いた画像を作り、path へ保存する。
std::string write_synthetic_image(const std::string& name, int width_px, int height_px) {
    const cv::aruco::Dictionary dictionary =
            cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_MIP_36h12);
    cv::Mat marker;
    dictionary.generateImageMarker(kMarkerId, kMarkerSidePx, marker, 1);

    cv::Mat scene(height_px, width_px, CV_8UC1, cv::Scalar(255));
    marker.copyTo(scene(cv::Rect(kMarkerOriginX, kMarkerOriginY, kMarkerSidePx, kMarkerSidePx)));

    const std::string path = "/tmp/" + name;
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

// 正常系: 既知のマーカーを検出し、四隅が描画位置と一致する。
TEST_F(ReferenceRunnerTest, detects_known_marker_at_expected_corners) {
    const std::string path = this->make_image("aruco3cuda_ref_basic.png");
    ReferenceConfig config;
    ReferenceResult result;
    std::string error;
    ASSERT_TRUE(aruco3cuda::reference::detect_image(path, config, &result, &error)) << error;

    ASSERT_EQ(result.detections_.size(), 1U);
    EXPECT_EQ(result.detections_[0].id_, kMarkerId);

    // 描画位置に対する許容誤差。subpixel 補正を含むため 1 pixel 程度を許す。
    constexpr double kTolerancePx = 1.0;
    EXPECT_NEAR(result.detections_[0].corners_[0], kMarkerOriginX, kTolerancePx);
    EXPECT_NEAR(result.detections_[0].corners_[1], kMarkerOriginY, kTolerancePx);
    EXPECT_NEAR(result.detections_[0].corners_[4], kMarkerOriginX + kMarkerSidePx, kTolerancePx);
    EXPECT_NEAR(result.detections_[0].corners_[5], kMarkerOriginY + kMarkerSidePx, kTolerancePx);
}

// 正常系: 同じ入力からは同じ結果が得られる。
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

// 正常系: ArUco3 の実効縮小率が観測仕様と一致する。
// docs/design/detector-pipeline.md に記録した式と同じ結果になることを固定する。
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

// 境界値: minMarkerLengthRatioOriginalImg が 0 だと縮小が発生しない。
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

// 境界値: ArUco3 を無効にすると縮小率は 1 になる。
TEST_F(ReferenceRunnerTest, no_downscale_when_aruco3_disabled) {
    const std::string path = this->make_image("aruco3cuda_ref_disabled.png");
    ReferenceConfig config;
    config.use_aruco3_detection_ = false;
    ReferenceResult result;
    std::string error;
    ASSERT_TRUE(aruco3cuda::reference::detect_image(path, config, &result, &error)) << error;
    EXPECT_DOUBLE_EQ(result.fxfy_effective_, 1.0);
}

// 異常系: 存在しない画像では失敗し理由を返す。
TEST_F(ReferenceRunnerTest, fails_for_missing_image) {
    ReferenceConfig config;
    ReferenceResult result;
    std::string error;
    EXPECT_FALSE(
            aruco3cuda::reference::detect_image("/nonexistent/img.png", config, &result, &error));
    EXPECT_FALSE(error.empty());
}

// 異常系: 未対応の Dictionary では失敗する。
TEST_F(ReferenceRunnerTest, fails_for_unknown_dictionary) {
    const std::string path = this->make_image("aruco3cuda_ref_dict.png");
    ReferenceConfig config;
    config.dictionary_name_ = "DICT_DOES_NOT_EXIST";
    ReferenceResult result;
    std::string error;
    EXPECT_FALSE(aruco3cuda::reference::detect_image(path, config, &result, &error));
    EXPECT_FALSE(error.empty());
}

// 異常系: 出力先が nullptr でも安全に失敗する。
TEST_F(ReferenceRunnerTest, rejects_null_outputs) {
    ReferenceConfig config;
    ReferenceResult result;
    std::string error;
    EXPECT_FALSE(aruco3cuda::reference::detect_image("/tmp/x.png", config, nullptr, &error));
    EXPECT_FALSE(aruco3cuda::reference::detect_image("/tmp/x.png", config, &result, nullptr));
}

// 正常系: Dictionary 名の解決。
TEST(ReferenceDictionaryTest, resolves_known_names_only) {
    EXPECT_TRUE(aruco3cuda::reference::is_known_dictionary("DICT_ARUCO_MIP_36h12"));
    EXPECT_TRUE(aruco3cuda::reference::is_known_dictionary("DICT_6X6_250"));
    EXPECT_FALSE(aruco3cuda::reference::is_known_dictionary("DICT_NOPE"));
    EXPECT_FALSE(aruco3cuda::reference::known_dictionary_names().empty());
}

// 正常系: JSON 出力が有効な構造を持ち、検出内容を含む。
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

// 正常系: --omit-timing 相当の設定では出力が byte 単位で決定的になる。
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

}  // namespace
