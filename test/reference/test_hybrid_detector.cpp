// SPDX-License-Identifier: Apache-2.0
//
// ハイブリッド経路を CPU 基準実装と突き合わせる。
//
// WP-1.5 の完了条件は「合成画像の基本条件で ID と四隅を取得できる」ことである。
// あわせて、CPU 基準との差異を分類できる状態にする。前処理と二値化が
// OpenCV と一致しているため、ここで残る差は CPU 側の実装差に絞られる。
#include "hybrid_detector.hpp"

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"
#include "corpus_generator.hpp"
#include "reference_runner.hpp"

namespace {

using aruco3cuda::DetectorConfig;
using aruco3cuda::ImageViewU8;
using aruco3cuda::MemorySpace;
using aruco3cuda::Status;
using aruco3cuda::hybrid::HybridDetector;
using aruco3cuda::hybrid::HybridResult;

bool has_cuda_device() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

class DeviceImage {
public:
    DeviceImage() = default;
    ~DeviceImage() {
        if (this->data_ != nullptr) {
            (void)cudaFree(this->data_);
        }
    }
    DeviceImage(const DeviceImage&) = delete;
    DeviceImage& operator=(const DeviceImage&) = delete;

    bool upload(const cv::Mat& image) {
        this->width_px_ = image.cols;
        this->height_px_ = image.rows;
        this->pitch_bytes_ = static_cast<std::size_t>(image.cols) + 48U;
        const std::size_t bytes = this->pitch_bytes_ * static_cast<std::size_t>(image.rows);
        if (cudaMalloc(&this->data_, bytes) != cudaSuccess) {
            return false;
        }
        return cudaMemcpy2D(
                       this->data_, this->pitch_bytes_, image.data,
                       static_cast<std::size_t>(image.step), static_cast<std::size_t>(image.cols),
                       static_cast<std::size_t>(image.rows), cudaMemcpyHostToDevice) == cudaSuccess;
    }

    ImageViewU8 view() const {
        ImageViewU8 result;
        result.data_ = static_cast<const std::uint8_t*>(this->data_);
        result.width_px_ = this->width_px_;
        result.height_px_ = this->height_px_;
        result.pitch_bytes_ = this->pitch_bytes_;
        result.space_ = MemorySpace::kDevice;
        return result;
    }

private:
    void* data_ = nullptr;
    std::size_t pitch_bytes_ = 0;
    int width_px_ = 0;
    int height_px_ = 0;
};

/// 合成 corpus の 1 枚を仕様から生成して読み込む。
cv::Mat make_scene_from_spec(const aruco3cuda::corpusgen::SceneSpec& spec,
                             aruco3cuda::corpusgen::GeneratedScene* out_scene) {
    aruco3cuda::corpusgen::CorpusConfig config;
    config.output_dir_ = "/tmp/aruco3cuda_hybrid_test";
    config.seed_ = 20260827U;
    std::string error;
    EXPECT_TRUE(aruco3cuda::corpusgen::generate_scene(config, spec, 0, out_scene, &error)) << error;
    return cv::imread(out_scene->path_, cv::IMREAD_GRAYSCALE);
}

/// 歪みの無い 1280x720 の場面を生成して読み込む。
cv::Mat make_scene(const std::string& name, int marker_count, double side_px,
                   aruco3cuda::corpusgen::GeneratedScene* out_scene) {
    aruco3cuda::corpusgen::SceneSpec spec;
    spec.name_ = name;
    spec.width_px_ = 1280;
    spec.height_px_ = 720;
    spec.marker_count_ = marker_count;
    spec.marker_side_px_ = side_px;
    return make_scene_from_spec(spec, out_scene);
}

/// CPU 基準実装で同じ画像を検出する。
aruco3cuda::reference::ReferenceResult detect_with_reference(const std::string& path,
                                                             const DetectorConfig& config) {
    aruco3cuda::reference::ReferenceConfig reference_config;
    reference_config.use_aruco3_detection_ = config.use_aruco3_detection_;
    reference_config.min_side_length_canonical_img_px_ = config.min_side_length_canonical_img_px_;
    reference_config.min_marker_length_ratio_original_img_ =
            config.min_marker_length_ratio_original_img_;
    reference_config.use_corner_subpix_refinement_ =
            config.corner_refine_method_ == aruco3cuda::CornerRefineMethod::kSubpix;
    aruco3cuda::reference::ReferenceResult result;
    std::string error;
    EXPECT_TRUE(aruco3cuda::reference::detect_image(path, reference_config, &result, &error))
            << error;
    return result;
}

/// 検出結果を ID で対応付け、四隅の最大誤差を返す。対応が無ければ -1。
double corner_error_for_id(const aruco3cuda::reference::ReferenceResult& reference,
                           const aruco3cuda::hybrid::HybridDetection& detection) {
    for (const auto& expected : reference.detections_) {
        if (expected.id_ != detection.id_) {
            continue;
        }
        double worst = 0.0;
        for (std::size_t c = 0; c < 4; ++c) {
            const double dx = expected.corners_[c * 2] - detection.corners_[c * 2];
            const double dy = expected.corners_[c * 2 + 1] - detection.corners_[c * 2 + 1];
            worst = std::max(worst, std::sqrt(dx * dx + dy * dy));
        }
        return worst;
    }
    return -1.0;
}

// 異常系: 初期化前は検出できない。
TEST(HybridDetectorTest, detect_before_initialize_fails) {
    HybridDetector detector;
    ImageViewU8 view;
    HybridResult result;
    std::string message;
    EXPECT_EQ(detector.detect(view, &result, &message), Status::kNotInitialized);
    EXPECT_FALSE(message.empty());
}

// 異常系: 不正な設定と Dictionary を拒否する。
TEST(HybridDetectorTest, rejects_invalid_initialization) {
    HybridDetector detector;
    std::string message;

    DetectorConfig invalid;
    invalid.max_candidates_ = 0;
    EXPECT_EQ(detector.initialize(invalid, "DICT_ARUCO_MIP_36h12", 640, 480, &message),
              Status::kInvalidConfig);

    EXPECT_EQ(detector.initialize(DetectorConfig(), "DICT_NOPE", 640, 480, &message),
              Status::kUnsupportedDictionary);
    EXPECT_NE(message.find("DICT_NOPE"), std::string::npos) << message;

    EXPECT_EQ(detector.initialize(DetectorConfig(), "DICT_ARUCO_MIP_36h12", 0, 480, &message),
              Status::kInvalidArgument);
}

// 異常系: 不正な画像 view を境界で拒否する。
TEST(HybridDetectorTest, rejects_invalid_image) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    HybridDetector detector;
    std::string message;
    ASSERT_EQ(detector.initialize(DetectorConfig(), "DICT_ARUCO_MIP_36h12", 640, 480, &message),
              Status::kOk)
            << message;
    ImageViewU8 view;
    HybridResult result;
    EXPECT_EQ(detector.detect(view, &result, &message), Status::kInvalidImage);
    EXPECT_EQ(detector.detect(view, nullptr, &message), Status::kInvalidArgument);
}

// 正常系: 合成画像から ID と四隅を取得できる。WP-1.5 の完了条件である。
TEST(HybridDetectorTest, detects_markers_in_synthetic_scene) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    aruco3cuda::corpusgen::GeneratedScene scene;
    const cv::Mat image = make_scene("hybrid_basic", 4, 128.0, &scene);
    ASSERT_FALSE(image.empty());
    ASSERT_EQ(scene.markers_.size(), 4U);

    DeviceImage device;
    ASSERT_TRUE(device.upload(image));

    DetectorConfig config;
    HybridDetector detector;
    std::string message;
    ASSERT_EQ(detector.initialize(config, "DICT_ARUCO_MIP_36h12", image.cols, image.rows, &message),
              Status::kOk)
            << message;

    HybridResult result;
    ASSERT_EQ(detector.detect(device.view(), &result, &message), Status::kOk) << message;

    std::printf("[hybrid] 候補 %zu 検出 %zu GPU %.3f ms CPU %.3f ms\n", result.candidate_count_,
                result.detections_.size(), result.gpu_ms_, result.cpu_ms_);

    // ground truth の ID を全て検出する。
    ASSERT_EQ(result.detections_.size(), scene.markers_.size());
    for (const auto& truth : scene.markers_) {
        const bool found = std::any_of(result.detections_.begin(), result.detections_.end(),
                                       [&truth](const auto& d) { return d.id_ == truth.id_; });
        EXPECT_TRUE(found) << "id=" << truth.id_;
    }
}

// 正常系: CPU 基準実装と ID が一致し、四隅の差を分類できる。
// Phase 1 のゲート G1 に対応する。
TEST(HybridDetectorTest, matches_cpu_reference_and_reports_differences) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    // 素の場面だけでなく、劣化と解像度も変えて比較する。縮小画像の生成は
    // OpenCV と bit 単位では一致しないため、輪郭が動きやすい条件を含める。
    std::vector<aruco3cuda::corpusgen::SceneSpec> cases;
    {
        // 既定は 1280x720、マーカー 4 枚、辺 128 px。ここから 1 つずつ変える。
        aruco3cuda::corpusgen::SceneSpec base;
        base.width_px_ = 1280;
        base.height_px_ = 720;
        base.marker_count_ = 4;
        base.marker_side_px_ = 128.0;

        aruco3cuda::corpusgen::SceneSpec spec = base;
        spec.name_ = "hybrid_n1";
        spec.marker_count_ = 1;
        spec.marker_side_px_ = 160.0;
        cases.push_back(spec);

        spec = base;
        spec.name_ = "hybrid_n4";
        cases.push_back(spec);

        spec = base;
        spec.name_ = "hybrid_n9";
        spec.marker_count_ = 9;
        cases.push_back(spec);

        spec = base;
        spec.name_ = "hybrid_small";
        spec.marker_count_ = 6;
        spec.marker_side_px_ = 112.0;
        cases.push_back(spec);

        // ArUco3 の下限を下回る大きさ。両者とも検出しないことを確かめる。
        spec = base;
        spec.name_ = "hybrid_too_small";
        spec.marker_count_ = 6;
        spec.marker_side_px_ = 48.0;
        cases.push_back(spec);

        spec = base;
        spec.name_ = "hybrid_rot";
        spec.rotation_deg_ = 23.0;
        cases.push_back(spec);

        spec = base;
        spec.name_ = "hybrid_persp";
        spec.perspective_strength_ = 0.5;
        cases.push_back(spec);

        spec = base;
        spec.name_ = "hybrid_blur";
        spec.blur_sigma_px_ = 1.2;
        cases.push_back(spec);

        spec = base;
        spec.name_ = "hybrid_noise";
        spec.noise_sigma_levels_ = 6.0;
        cases.push_back(spec);

        spec = base;
        spec.name_ = "hybrid_illum";
        spec.illumination_strength_ = 0.6;
        cases.push_back(spec);

        spec = base;
        spec.name_ = "hybrid_1080p";
        spec.width_px_ = 1920;
        spec.height_px_ = 1080;
        spec.marker_side_px_ = 160.0;
        cases.push_back(spec);

        spec = base;
        spec.name_ = "hybrid_vga";
        spec.width_px_ = 640;
        spec.height_px_ = 480;
        spec.marker_count_ = 3;
        spec.marker_side_px_ = 96.0;
        cases.push_back(spec);
    }

    // ArUco3 有効時 (既定) と無効時 (OpenCV 既定) の両方で比較する。
    // 原寸へ戻す処理が経路ごとに異なるため、片方だけでは足りない。
    struct ConfigCase {
        const char* label;
        DetectorConfig config;
    };
    const auto plain_with_subpix = [] {
        DetectorConfig config = DetectorConfig::opencv_defaults();
        config.corner_refine_method_ = aruco3cuda::CornerRefineMethod::kSubpix;
        return config;
    };
    const ConfigCase config_cases[] = {{"aruco3", DetectorConfig{}},
                                       {"plain", DetectorConfig::opencv_defaults()},
                                       {"plain_subpix", plain_with_subpix()}};

    double worst_overall = 0.0;
    for (const ConfigCase& config_case : config_cases) {
        for (const aruco3cuda::corpusgen::SceneSpec& spec : cases) {
            aruco3cuda::corpusgen::GeneratedScene scene;
            const cv::Mat image = make_scene_from_spec(spec, &scene);
            ASSERT_FALSE(image.empty()) << spec.name_;

            DeviceImage device;
            ASSERT_TRUE(device.upload(image)) << spec.name_;

            const DetectorConfig& config = config_case.config;
            HybridDetector detector;
            std::string message;
            ASSERT_EQ(detector.initialize(config, "DICT_ARUCO_MIP_36h12", image.cols, image.rows,
                                          &message),
                      Status::kOk)
                    << message;
            HybridResult result;
            ASSERT_EQ(detector.detect(device.view(), &result, &message), Status::kOk) << message;

            const auto reference = detect_with_reference(scene.path_, config);

            // 差異を分類する。未検出、過検出、四隅ずれ。
            std::size_t matched = 0;
            double worst_corner_error = 0.0;
            for (const auto& detection : result.detections_) {
                const double error = corner_error_for_id(reference, detection);
                if (error < 0.0) {
                    continue;  // CPU 基準に無い ID。過検出として数える
                }
                ++matched;
                worst_corner_error = std::max(worst_corner_error, error);
            }
            const std::size_t missed = reference.detections_.size() - matched;
            const std::size_t extra = result.detections_.size() - matched;
            worst_overall = std::max(worst_overall, worst_corner_error);

            std::printf(
                    "[hybrid vs CPU] %s/%s: 基準 %zu 一致 %zu 未検出 %zu 過検出 %zu 四隅最大差 "
                    "%.4f px\n",
                    config_case.label, spec.name_.c_str(), reference.detections_.size(), matched,
                    missed, extra, worst_corner_error);

            EXPECT_EQ(missed, 0U) << spec.name_;
            EXPECT_EQ(extra, 0U) << spec.name_;
            // 前処理と二値化が OpenCV と一致し、候補の grouping も同じ代表を
            // 選ぶため、素の場面では四隅が完全に一致する。差が出るのは縮小
            // 画像の生成が bit 単位では一致しないためで、上限は実測に基づく。
            EXPECT_LT(worst_corner_error, 0.5) << spec.name_;
        }
    }
    std::printf("[hybrid vs CPU] 全 %zu 場面 x %zu 設定の四隅最大差 %.4f px\n", cases.size(),
                std::size(config_cases), worst_overall);
}

// 正常系: 定常状態でフレームごとの確保が発生しない。
TEST(HybridDetectorTest, steady_state_does_not_allocate_per_frame) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    aruco3cuda::corpusgen::GeneratedScene scene;
    const cv::Mat image = make_scene("hybrid_steady", 4, 128.0, &scene);
    ASSERT_FALSE(image.empty());
    DeviceImage device;
    ASSERT_TRUE(device.upload(image));

    HybridDetector detector;
    std::string message;
    ASSERT_EQ(detector.initialize(DetectorConfig(), "DICT_ARUCO_MIP_36h12", image.cols, image.rows,
                                  &message),
              Status::kOk)
            << message;

    HybridResult result;
    ASSERT_EQ(detector.detect(device.view(), &result, &message), Status::kOk) << message;
    const std::size_t after_first = detector.workspace_statistics().allocation_count_;

    for (int frame = 0; frame < 20; ++frame) {
        ASSERT_EQ(detector.detect(device.view(), &result, &message), Status::kOk) << message;
        ASSERT_EQ(detector.workspace_statistics().allocation_count_, after_first)
                << "frame=" << frame;
        ASSERT_EQ(detector.workspace_statistics().reallocation_count_, 0U) << "frame=" << frame;
    }
}

// 正常系: 同じ入力からは同じ結果が得られる。
TEST(HybridDetectorTest, results_are_deterministic) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    aruco3cuda::corpusgen::GeneratedScene scene;
    const cv::Mat image = make_scene("hybrid_determinism", 4, 128.0, &scene);
    ASSERT_FALSE(image.empty());
    DeviceImage device;
    ASSERT_TRUE(device.upload(image));

    HybridDetector detector;
    std::string message;
    ASSERT_EQ(detector.initialize(DetectorConfig(), "DICT_ARUCO_MIP_36h12", image.cols, image.rows,
                                  &message),
              Status::kOk);
    HybridResult first;
    HybridResult second;
    ASSERT_EQ(detector.detect(device.view(), &first, &message), Status::kOk);
    ASSERT_EQ(detector.detect(device.view(), &second, &message), Status::kOk);

    ASSERT_EQ(first.detections_.size(), second.detections_.size());
    for (std::size_t i = 0; i < first.detections_.size(); ++i) {
        EXPECT_EQ(first.detections_[i].id_, second.detections_[i].id_);
        EXPECT_EQ(first.detections_[i].corners_, second.detections_[i].corners_);
    }
}

// 境界値: マーカーが無い画像では検出が 0 件になる。
TEST(HybridDetectorTest, reports_no_detection_for_empty_scene) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    aruco3cuda::corpusgen::GeneratedScene scene;
    const cv::Mat image = make_scene("hybrid_empty", 0, 128.0, &scene);
    ASSERT_FALSE(image.empty());
    DeviceImage device;
    ASSERT_TRUE(device.upload(image));

    HybridDetector detector;
    std::string message;
    ASSERT_EQ(detector.initialize(DetectorConfig(), "DICT_ARUCO_MIP_36h12", image.cols, image.rows,
                                  &message),
              Status::kOk);
    HybridResult result;
    ASSERT_EQ(detector.detect(device.view(), &result, &message), Status::kOk);
    EXPECT_TRUE(result.detections_.empty());
}

// 境界値: マーカーを黒枠が囲む場面でも CPU 基準と一致する。
//
// 黒枠の外周も四角形候補になり、マーカーを内側に含む。候補の木では
// マーカーが子、枠が親になる。親を識別対象から外す経路を通す。
TEST(HybridDetectorTest, matches_cpu_reference_for_nested_quads) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const cv::aruco::Dictionary dictionary =
            cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_MIP_36h12);
    cv::Mat marker;
    dictionary.generateImageMarker(101, 192, marker, 1);

    cv::Mat scene(720, 1280, CV_8UC1, cv::Scalar(255));
    // マーカーの周りに余白を置き、その外側を黒枠で囲む。
    const cv::Rect marker_rect(544, 264, 192, 192);
    marker.copyTo(scene(marker_rect));
    const cv::Rect frame_rect(marker_rect.x - 40, marker_rect.y - 40, marker_rect.width + 80,
                              marker_rect.height + 80);
    cv::rectangle(scene, frame_rect, cv::Scalar(0), 12);

    const std::string path = "/tmp/aruco3cuda_hybrid_nested.png";
    ASSERT_TRUE(cv::imwrite(path, scene));

    DeviceImage device;
    ASSERT_TRUE(device.upload(scene));

    const DetectorConfig config;
    HybridDetector detector;
    std::string message;
    ASSERT_EQ(detector.initialize(config, "DICT_ARUCO_MIP_36h12", scene.cols, scene.rows, &message),
              Status::kOk)
            << message;
    HybridResult result;
    ASSERT_EQ(detector.detect(device.view(), &result, &message), Status::kOk) << message;

    const auto reference = detect_with_reference(path, config);
    std::remove(path.c_str());

    ASSERT_EQ(result.detections_.size(), reference.detections_.size());
    double worst = 0.0;
    for (const auto& detection : result.detections_) {
        const double error = corner_error_for_id(reference, detection);
        ASSERT_GE(error, 0.0) << "CPU 基準に無い id=" << detection.id_;
        worst = std::max(worst, error);
    }
    std::printf("[hybrid vs CPU] nested: 検出 %zu 四隅最大差 %.4f px\n", result.detections_.size(),
                worst);
    EXPECT_LT(worst, 0.5);
}

// 正常系: move してもそのまま検出できる。pimpl の move を確認する。
TEST(HybridDetectorTest, detector_can_be_moved) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    aruco3cuda::corpusgen::GeneratedScene scene;
    const cv::Mat image = make_scene("hybrid_move", 4, 128.0, &scene);
    ASSERT_FALSE(image.empty());
    DeviceImage device;
    ASSERT_TRUE(device.upload(image));

    HybridDetector source;
    std::string message;
    ASSERT_EQ(source.initialize(DetectorConfig(), "DICT_ARUCO_MIP_36h12", image.cols, image.rows,
                                &message),
              Status::kOk)
            << message;

    HybridDetector moved(std::move(source));
    HybridResult first;
    ASSERT_EQ(moved.detect(device.view(), &first, &message), Status::kOk) << message;
    EXPECT_FALSE(first.detections_.empty());

    HybridDetector assigned;
    assigned = std::move(moved);
    HybridResult second;
    ASSERT_EQ(assigned.detect(device.view(), &second, &message), Status::kOk) << message;
    ASSERT_EQ(second.detections_.size(), first.detections_.size());
    for (std::size_t i = 0; i < first.detections_.size(); ++i) {
        EXPECT_EQ(second.detections_[i].id_, first.detections_[i].id_);
        EXPECT_EQ(second.detections_[i].corners_, first.detections_[i].corners_);
    }
}

}  // namespace
