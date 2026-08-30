// SPDX-License-Identifier: Apache-2.0
//
// Cross-checks the hybrid path against the CPU reference implementation.
//
// The condition the hybrid path has to meet is that it recovers IDs and corners
// under the basic conditions of the synthetic images. Beyond that, the test keeps
// the differences against the CPU reference classifiable. Preprocessing and
// binarization already agree with OpenCV, so the differences that remain here are
// narrowed down to implementation differences on the CPU side.
#include "hybrid_detector.hpp"

#include <gtest/gtest.h>
#include <unistd.h>

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

/// Generates one image of the synthetic corpus from its spec and loads it.
cv::Mat make_scene_from_spec(const aruco3cuda::corpusgen::SceneSpec& spec,
                             aruco3cuda::corpusgen::GeneratedScene* out_scene) {
    aruco3cuda::corpusgen::CorpusConfig config;
    // Keep the directory per process so that concurrent runs of the same binary do not collide.
    config.output_dir_ = "/tmp/aruco3cuda_hybrid_test_" + std::to_string(::getpid());
    config.seed_ = 20260827U;
    std::string error;
    EXPECT_TRUE(aruco3cuda::corpusgen::generate_scene(config, spec, 0, out_scene, &error)) << error;
    return cv::imread(out_scene->path_, cv::IMREAD_GRAYSCALE);
}

/// Generates an undistorted 1280x720 scene and loads it.
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

/// Runs detection on the same image with the CPU reference implementation.
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

/// Matches detections by ID and returns the largest corner error. Returns -1 when
/// there is no match.
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

// Failure: detection is not possible before initialization.
TEST(HybridDetectorTest, detect_before_initialize_fails) {
    HybridDetector detector;
    ImageViewU8 view;
    HybridResult result;
    std::string message;
    EXPECT_EQ(detector.detect(view, &result, &message), Status::kNotInitialized);
    EXPECT_FALSE(message.empty());
}

// Failure: an invalid configuration and an invalid dictionary are rejected.
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

// Failure: an invalid image view is rejected at the boundary.
TEST(HybridDetectorTest, rejects_invalid_image) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
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

// Nominal: IDs and corners are recovered from a synthetic image. This is the minimum
// condition for the hybrid path.
TEST(HybridDetectorTest, detects_markers_in_synthetic_scene) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
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

    std::printf("[hybrid] candidates %zu detections %zu GPU %.3f ms CPU %.3f ms\n",
                result.candidate_count_, result.detections_.size(), result.gpu_ms_, result.cpu_ms_);

    // Every ground-truth ID is detected.
    ASSERT_EQ(result.detections_.size(), scene.markers_.size());
    for (const auto& truth : scene.markers_) {
        const bool found = std::any_of(result.detections_.begin(), result.detections_.end(),
                                       [&truth](const auto& d) { return d.id_ == truth.id_; });
        EXPECT_TRUE(found) << "id=" << truth.id_;
    }
}

// Nominal: the IDs agree with the CPU reference implementation and the corner
// differences can be classified. This is the confirmation that the hybrid path holds up.
TEST(HybridDetectorTest, matches_cpu_reference_and_reports_differences) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
    }
    // Compare not only plain scenes but also varying degradation and resolution.
    // Generating the downscaled image does not agree with OpenCV bit for bit, so the
    // cases include conditions under which the contours move easily.
    std::vector<aruco3cuda::corpusgen::SceneSpec> cases;
    {
        // The baseline is 1280x720, four markers, 128 px per side. Each case changes
        // one thing from there.
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

        // A size below the ArUco3 lower bound. This confirms that neither side detects it.
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

    // Compare both with ArUco3 enabled (the default) and disabled (the OpenCV default).
    // The step that maps back to the original scale differs between the two paths, so
    // one of them alone would not be enough.
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

            // Classify the differences: missed, extra, and corner shift.
            std::size_t matched = 0;
            double worst_corner_error = 0.0;
            for (const auto& detection : result.detections_) {
                const double error = corner_error_for_id(reference, detection);
                if (error < 0.0) {
                    continue;  // An ID absent from the CPU reference; counted as extra.
                }
                ++matched;
                worst_corner_error = std::max(worst_corner_error, error);
            }
            const std::size_t missed = reference.detections_.size() - matched;
            const std::size_t extra = result.detections_.size() - matched;
            worst_overall = std::max(worst_overall, worst_corner_error);

            std::printf(
                    "[hybrid vs CPU] %s/%s: baseline %zu matched %zu missed %zu extra %zu "
                    "largest corner difference %.4f px\n",
                    config_case.label, spec.name_.c_str(), reference.detections_.size(), matched,
                    missed, extra, worst_corner_error);

            EXPECT_EQ(missed, 0U) << spec.name_;
            EXPECT_EQ(extra, 0U) << spec.name_;
            // Preprocessing and binarization agree with OpenCV, and candidate grouping
            // picks the same representative, so on plain scenes the corners agree exactly.
            // Differences appear because generating the downscaled image does not agree
            // bit for bit, and the upper bound is based on measurement.
            EXPECT_LT(worst_corner_error, 0.5) << spec.name_;
        }
    }
    std::printf(
            "[hybrid vs CPU] largest corner difference over all %zu scenes x %zu configs "
            "%.4f px\n",
            cases.size(), std::size(config_cases), worst_overall);
}

// Nominal: in steady state no allocation happens per frame.
TEST(HybridDetectorTest, steady_state_does_not_allocate_per_frame) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
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

// Nominal: the same input yields the same result.
TEST(HybridDetectorTest, results_are_deterministic) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
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

// Boundary: an image without markers yields no detections.
TEST(HybridDetectorTest, reports_no_detection_for_empty_scene) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
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

// Boundary: a scene where a black frame surrounds the marker still agrees with the
// CPU reference.
//
// The outer edge of the black frame is a quadrilateral candidate too, and it
// contains the marker inside it. In the candidate tree the marker is the child and
// the frame is the parent. This exercises the path that drops the parent from
// identification.
TEST(HybridDetectorTest, matches_cpu_reference_for_nested_quads) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
    }
    const cv::aruco::Dictionary dictionary =
            cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_MIP_36h12);
    cv::Mat marker;
    dictionary.generateImageMarker(101, 192, marker, 1);

    cv::Mat scene(720, 1280, CV_8UC1, cv::Scalar(255));
    // Leave a margin around the marker and surround it with a black frame.
    const cv::Rect marker_rect(544, 264, 192, 192);
    marker.copyTo(scene(marker_rect));
    const cv::Rect frame_rect(marker_rect.x - 40, marker_rect.y - 40, marker_rect.width + 80,
                              marker_rect.height + 80);
    cv::rectangle(scene, frame_rect, cv::Scalar(0), 12);

    const std::string path = "/tmp/aruco3cuda_hybrid_nested_" + std::to_string(::getpid()) + ".png";
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
        ASSERT_GE(error, 0.0) << "id absent from the CPU reference: id=" << detection.id_;
        worst = std::max(worst, error);
    }
    std::printf("[hybrid vs CPU] nested: detections %zu largest corner difference %.4f px\n",
                result.detections_.size(), worst);
    EXPECT_LT(worst, 0.5);
}

// Nominal: detection still works after a move. This exercises the move of the pimpl.
TEST(HybridDetectorTest, detector_can_be_moved) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
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
