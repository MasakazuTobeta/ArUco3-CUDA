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

// 正常系: 環境情報を収集でき、記録すべき項目が埋まる。
// 空や誤値のまま結果へ記録されると、測定条件を後から再現できない。
TEST(ReferenceEnvironmentTest, collects_opencv_version_and_thread_count) {
    aruco3cuda::reference::ReferenceConfig config;
    config.num_threads_ = 1;
    const aruco3cuda::reference::ReferenceEnvironment environment =
            aruco3cuda::reference::collect_environment(config);
    EXPECT_FALSE(environment.opencv_version_.empty());
    // 再現性のため thread 数を固定する。指定した値が実際に反映される必要がある。
    EXPECT_EQ(environment.opencv_threads_, 1);
}

// 正常系: thread 数に 0 を指定すると OpenCV の既定に従う。
TEST(ReferenceEnvironmentTest, thread_count_zero_uses_opencv_default) {
    aruco3cuda::reference::ReferenceConfig config;
    config.num_threads_ = 0;
    const aruco3cuda::reference::ReferenceEnvironment environment =
            aruco3cuda::reference::collect_environment(config);
    EXPECT_GT(environment.opencv_threads_, 0);
}

// 異常系: 設定値が範囲外なら検出前に false を返す。
// 範囲外の値を OpenCV へ渡すと cv::Exception が契約を破って抜ける。
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

    // NaN は比較が全て false になるため、範囲検査で弾けている必要がある。
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

// 異常系: 設定値が範囲外の場合、detect_image は例外ではなく false で失敗する。
TEST_F(ReferenceRunnerTest, detect_rejects_invalid_config_without_exception) {
    const std::string path = this->make_image("aruco3cuda_ref_badcfg.png");
    ReferenceConfig config;
    config.marker_border_bits_ = 0;
    ReferenceResult result;
    std::string error;
    EXPECT_FALSE(aruco3cuda::reference::detect_image(path, config, &result, &error));
    EXPECT_NE(error.find("marker_border_bits"), std::string::npos) << error;
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

// 正常系: 読み込みを分けても detect_image と同じ結果になる。
//
// 測定では読み込みを初期化側へ寄せる。結果が変わってしまえば、測定して
// いる対象が別物になる。
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
    // metadata は検出結果を含まない。
    EXPECT_TRUE(detector.metadata().detections_.empty());

    // 繰り返し呼んでも同じ結果になる。
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

// 正常系: move してもそのまま使える。
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

// 異常系: 初期化前の検出と、不正な引数を拒否する。
TEST_F(ReferenceRunnerTest, detector_rejects_invalid_use) {
    aruco3cuda::reference::ReferenceDetector detector;
    ReferenceResult result;
    std::string error;

    EXPECT_FALSE(detector.detect(&result, &error));
    EXPECT_NE(error.find("initialize"), std::string::npos);
    EXPECT_FALSE(detector.detect(nullptr, &error));
    EXPECT_FALSE(detector.detect(&result, nullptr));
    EXPECT_FALSE(detector.initialize("/nonexistent/image.png", ReferenceConfig(), nullptr));

    // 読み込めない画像。
    EXPECT_FALSE(detector.initialize("/nonexistent/image.png", ReferenceConfig(), &error));
    EXPECT_NE(error.find("読み込めない"), std::string::npos);

    // 未対応の Dictionary。
    ReferenceConfig bad_dictionary;
    bad_dictionary.dictionary_name_ = "DICT_NOPE";
    EXPECT_FALSE(detector.initialize("/nonexistent/image.png", bad_dictionary, &error));
    EXPECT_NE(error.find("Dictionary"), std::string::npos);

    // 設定が矛盾している場合。
    ReferenceConfig bad_range;
    bad_range.max_marker_perimeter_rate_ = bad_range.min_marker_perimeter_rate_;
    EXPECT_FALSE(detector.initialize("/nonexistent/image.png", bad_range, &error));
    EXPECT_NE(error.find("矛盾"), std::string::npos);
}

}  // namespace
