// SPDX-License-Identifier: Apache-2.0
//
// 公開 API の検出器を検証する。
//
// 完了条件は「host 同期なしで検出結果を参照できることを test で確認できる」
// である。単に呼べるだけでは足りないため、stream を占有する kernel を先に
// 積み、detect_async が戻った時点で stream がまだ動いていることを確かめる。
#include "aruco3cuda/detector.hpp"

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <ratio>
#include <string>
#include <utility>
#include <vector>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/detections.hpp"
#include "aruco3cuda/dictionary.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"
#include "device_image.hpp"
#include "test_detector_probe.hpp"

namespace {

using aruco3cuda::Detector;
using aruco3cuda::DetectorConfig;
using aruco3cuda::DeviceDetections;
using aruco3cuda::HostDetections;
using aruco3cuda::Status;

constexpr int kMarkerId = 42;
constexpr int kMarkerSidePx = 160;

bool has_cuda_device() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

const aruco3cuda::DictionaryTable& table() {
    const aruco3cuda::DictionaryTable* found =
            aruco3cuda::find_builtin_dictionary("DICT_ARUCO_MIP_36h12");
    EXPECT_NE(found, nullptr);
    return *found;
}

/// マーカーを 1 枚置いた場面を作る。
cv::Mat make_scene(int width, int height, const cv::Point& origin, int id = kMarkerId) {
    const cv::aruco::Dictionary dictionary =
            cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_MIP_36h12);
    cv::Mat marker;
    dictionary.generateImageMarker(id, kMarkerSidePx, marker, 1);
    cv::Mat scene(height, width, CV_8UC1, cv::Scalar(235));
    marker.copyTo(scene(cv::Rect(origin.x, origin.y, kMarkerSidePx, kMarkerSidePx)));
    return scene;
}

/// 入力上限を場面へ合わせた設定を作る。
DetectorConfig config_for(const cv::Mat& scene) {
    DetectorConfig config;
    config.max_width_px_ = scene.cols;
    config.max_height_px_ = scene.rows;
    return config;
}

/// 場面を device へ載せる。
class SceneImage {
public:
    SceneImage() = default;
    SceneImage(const SceneImage&) = delete;
    SceneImage& operator=(const SceneImage&) = delete;

    bool upload(const cv::Mat& scene) {
        return this->image_.reserve(scene.cols, scene.rows) == Status::kOk &&
               this->image_.upload(scene.data, scene.cols, scene.rows, scene.step) == Status::kOk;
    }
    const aruco3cuda::ImageViewU8& view() const { return this->image_.view(); }

private:
    aruco3cuda::hybrid::DeviceImage image_;
};

/// OpenCV の検出結果。突き合わせの基準にする。
struct ReferenceMarker {
    int id_ = -1;
    std::vector<cv::Point2f> corners_;
};

std::vector<ReferenceMarker> detect_with_opencv(const cv::Mat& scene,
                                                const DetectorConfig& config) {
    cv::aruco::DetectorParameters parameters;
    parameters.useAruco3Detection = config.use_aruco3_detection_;
    parameters.minSideLengthCanonicalImg = config.min_side_length_canonical_img_px_;
    parameters.minMarkerLengthRatioOriginalImg = config.min_marker_length_ratio_original_img_;
    parameters.cornerRefinementMethod = static_cast<int>(cv::aruco::CORNER_REFINE_SUBPIX);
    const cv::aruco::Dictionary dictionary =
            cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_MIP_36h12);
    const cv::aruco::ArucoDetector detector(dictionary, parameters);

    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> ids;
    detector.detectMarkers(scene, corners, ids);

    std::vector<ReferenceMarker> result;
    result.reserve(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) {
        ReferenceMarker marker;
        marker.id_ = ids[i];
        marker.corners_ = corners[i];
        result.push_back(marker);
    }
    return result;
}

/// 検出 1 件の四隅を取り出す。
std::vector<cv::Point2f> corners_of(const HostDetections& detections, std::size_t index) {
    std::vector<cv::Point2f> result;
    result.reserve(4U);
    for (int corner = 0; corner < 4; ++corner) {
        const std::size_t base = (index * 8U) + (static_cast<std::size_t>(corner) * 2U);
        result.emplace_back(detections.corners_[base], detections.corners_[base + 1U]);
    }
    return result;
}

/// 巡回のずれを許した四隅の最大頂点距離。
double quad_distance(const std::vector<cv::Point2f>& a, const std::vector<cv::Point2f>& b) {
    double best = 1e30;
    for (int shift = 0; shift < 4; ++shift) {
        double worst = 0.0;
        for (int i = 0; i < 4; ++i) {
            const cv::Point2f& first = a[static_cast<std::size_t>(i)];
            const cv::Point2f& second = b[static_cast<std::size_t>((i + shift) % 4)];
            const double dx = static_cast<double>(first.x) - static_cast<double>(second.x);
            const double dy = static_cast<double>(first.y) - static_cast<double>(second.y);
            worst = std::max(worst, std::sqrt((dx * dx) + (dy * dy)));
        }
        best = std::min(best, worst);
    }
    return best;
}

// 正常系: 初期化してマーカーを検出できる。
TEST(DetectorTest, detects_marker_and_matches_opencv) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const cv::Mat scene = make_scene(1280, 720, cv::Point(400, 260));
    const DetectorConfig config = config_for(scene);

    SceneImage image;
    ASSERT_TRUE(image.upload(scene));

    Detector detector;
    std::string message;
    ASSERT_EQ(detector.initialize(table(), config, &message), Status::kOk) << message;
    EXPECT_TRUE(detector.initialized());

    ASSERT_EQ(detector.detect_async(image.view(), nullptr, &message), Status::kOk) << message;
    HostDetections detections;
    ASSERT_EQ(detector.download(&detections, nullptr, &message), Status::kOk) << message;

    const std::vector<ReferenceMarker> expected = detect_with_opencv(scene, config);
    ASSERT_EQ(expected.size(), 1U);
    ASSERT_EQ(detections.ids_.size(), 1U);
    EXPECT_EQ(detections.ids_[0], kMarkerId);
    EXPECT_EQ(detections.ids_[0], expected[0].id_);

    const double distance = quad_distance(corners_of(detections, 0U), expected[0].corners_);
    std::printf("[detector] ID %d、四隅の差 %.4f px\n", detections.ids_[0], distance);
    EXPECT_LT(distance, 1.0);
}

// 正常系: detect_async は host 同期しない。
//
// stream を長く占有する kernel を先に積み、その後ろへ detect_async を積む。
// host 同期していれば呼び出しから戻るまでに占有が終わっているはずであり、
// stream は完了している。同期していなければ stream はまだ動いている。
// suite 名に Timing を含める。Compute Sanitizer の下では占有 kernel の
// 実時間が読めず、発行と占有の大小が崩れる。既定の除外規則に載せる。
TEST(DetectorTimingTest, detect_async_does_not_synchronize) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const cv::Mat scene = make_scene(1280, 720, cv::Point(400, 260));
    const DetectorConfig config = config_for(scene);

    SceneImage image;
    ASSERT_TRUE(image.upload(scene));

    Detector detector;
    std::string message;
    ASSERT_EQ(detector.initialize(table(), config, &message), Status::kOk) << message;

    // 既定 stream は他の stream と暗黙に同期する。専用の stream を作る。
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    int* sink = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&sink), sizeof(int)), cudaSuccess);

    // 1 度流して配置を確定させる。以降は同じ入力なので配置を組み直さない。
    ASSERT_EQ(detector.detect_async(image.view(), stream, &message), Status::kOk) << message;
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    // clock の周波数は機で違う。十分に大きい値を積み、下限だけを検査する。
    constexpr long long kSpinCycles = 2000000000LL;
    ASSERT_TRUE(aruco3cuda::test::enqueue_spin(kSpinCycles, sink, stream));

    const auto start = std::chrono::steady_clock::now();
    const Status status = detector.detect_async(image.view(), stream, &message);
    DeviceDetections device;
    const Status view_status = detector.device_detections(&device);
    const auto finish = std::chrono::steady_clock::now();
    const cudaError_t query = cudaStreamQuery(stream);
    const double elapsed_ms = std::chrono::duration<double, std::milli>(finish - start).count();

    // 後始末は先に済ませる。assert が落ちても資源を残さない。
    const cudaError_t sync = cudaStreamSynchronize(stream);
    const auto spin_finish = std::chrono::steady_clock::now();
    const double spin_ms = std::chrono::duration<double, std::milli>(spin_finish - start).count();
    EXPECT_EQ(cudaFree(sink), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);

    ASSERT_EQ(status, Status::kOk) << message;
    ASSERT_EQ(view_status, Status::kOk);
    ASSERT_EQ(sync, cudaSuccess);
    EXPECT_NE(device.count_, nullptr);
    EXPECT_GT(device.capacity_, 0);

    std::printf("[detector] 発行 %.3f ms、占有の完了まで %.3f ms、query=%s\n", elapsed_ms, spin_ms,
                cudaGetErrorName(query));
    // 占有が十分に長いことを先に確かめる。短ければこの test は何も測っていない。
    ASSERT_GT(spin_ms, 50.0) << "占有 kernel が短すぎる。cycle 数を増やすこと";
    // 発行が占有の完了を待っていないこと。
    EXPECT_EQ(query, cudaErrorNotReady);
    EXPECT_LT(elapsed_ms, spin_ms / 2.0);
}

// 正常系: device 常駐の結果を host 同期なしで参照できる。
TEST(DetectorTest, device_detections_needs_no_synchronization) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const cv::Mat scene = make_scene(1280, 720, cv::Point(400, 260));
    const DetectorConfig config = config_for(scene);
    SceneImage image;
    ASSERT_TRUE(image.upload(scene));

    Detector detector;
    std::string message;
    ASSERT_EQ(detector.initialize(table(), config, &message), Status::kOk) << message;
    ASSERT_EQ(detector.detect_async(image.view(), nullptr, &message), Status::kOk) << message;

    DeviceDetections device;
    ASSERT_EQ(detector.device_detections(&device), Status::kOk);
    ASSERT_NE(device.count_, nullptr);
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);

    // device pointer をそのまま読み、host 側の詰め替えを介さずに一致を見る。
    std::int32_t count = 0;
    ASSERT_EQ(cudaMemcpy(&count, device.count_, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(count, 1);
    std::int32_t id = 0;
    ASSERT_EQ(cudaMemcpy(&id, device.ids_, sizeof(std::int32_t), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(id, kMarkerId);

    HostDetections host;
    ASSERT_EQ(detector.download(&host, nullptr, &message), Status::kOk) << message;
    ASSERT_EQ(host.ids_.size(), 1U);
    EXPECT_EQ(host.ids_[0], id);
}

// 正常系: workspace を frame ごとに確保し直さない。
TEST(DetectorTest, allocates_workspace_once) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const cv::Mat scene = make_scene(1280, 720, cv::Point(400, 260));
    const DetectorConfig config = config_for(scene);
    SceneImage image;
    ASSERT_TRUE(image.upload(scene));

    Detector detector;
    std::string message;
    ASSERT_EQ(detector.initialize(table(), config, &message), Status::kOk) << message;
    const std::size_t after_initialize = detector.workspace_statistics().allocation_count_;
    EXPECT_EQ(after_initialize, 1U);

    for (int i = 0; i < 8; ++i) {
        ASSERT_EQ(detector.detect_async(image.view(), nullptr, &message), Status::kOk) << message;
    }
    ASSERT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
    EXPECT_EQ(detector.workspace_statistics().allocation_count_, after_initialize);
    EXPECT_EQ(detector.workspace_statistics().exhausted_count_, 0U);
    std::printf("[detector] workspace %zu byte、確保 %zu 回\n",
                detector.workspace_statistics().capacity_bytes_,
                detector.workspace_statistics().allocation_count_);
}

// 正常系: 複数のマーカーを検出できる。
TEST(DetectorTest, detects_multiple_markers) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const cv::aruco::Dictionary dictionary =
            cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_MIP_36h12);
    cv::Mat scene(720, 1280, CV_8UC1, cv::Scalar(235));
    const std::vector<int> ids = {7, 42, 100};
    const std::vector<cv::Point> origins = {cv::Point(120, 120), cv::Point(520, 120),
                                            cv::Point(320, 420)};
    for (std::size_t i = 0; i < ids.size(); ++i) {
        cv::Mat marker;
        dictionary.generateImageMarker(ids[i], kMarkerSidePx, marker, 1);
        marker.copyTo(scene(cv::Rect(origins[i].x, origins[i].y, kMarkerSidePx, kMarkerSidePx)));
    }
    const DetectorConfig config = config_for(scene);

    SceneImage image;
    ASSERT_TRUE(image.upload(scene));
    Detector detector;
    std::string message;
    ASSERT_EQ(detector.initialize(table(), config, &message), Status::kOk) << message;
    ASSERT_EQ(detector.detect_async(image.view(), nullptr, &message), Status::kOk) << message;
    HostDetections detections;
    ASSERT_EQ(detector.download(&detections, nullptr, &message), Status::kOk) << message;

    std::vector<int> found(detections.ids_.begin(), detections.ids_.end());
    std::sort(found.begin(), found.end());
    std::printf("[detector] %zu 件を検出\n", found.size());
    EXPECT_EQ(found, ids);
    EXPECT_EQ(detections.corners_.size(), found.size() * 8U);
    EXPECT_FALSE(detections.marker_overflow_);
}

// 境界値: 入力の寸法を変えても正しく検出できる。
TEST(DetectorTest, handles_changed_input_size) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const cv::Mat large = make_scene(1280, 720, cv::Point(400, 260));
    const cv::Mat small = make_scene(960, 540, cv::Point(300, 180));
    DetectorConfig config = config_for(large);

    SceneImage large_image;
    SceneImage small_image;
    ASSERT_TRUE(large_image.upload(large));
    ASSERT_TRUE(small_image.upload(small));

    Detector detector;
    std::string message;
    ASSERT_EQ(detector.initialize(table(), config, &message), Status::kOk) << message;

    HostDetections detections;
    ASSERT_EQ(detector.detect_async(large_image.view(), nullptr, &message), Status::kOk) << message;
    ASSERT_EQ(detector.download(&detections, nullptr, &message), Status::kOk) << message;
    ASSERT_EQ(detections.ids_.size(), 1U);

    ASSERT_EQ(detector.detect_async(small_image.view(), nullptr, &message), Status::kOk) << message;
    ASSERT_EQ(detector.download(&detections, nullptr, &message), Status::kOk) << message;
    ASSERT_EQ(detections.ids_.size(), 1U);
    EXPECT_EQ(detections.ids_[0], kMarkerId);
    // 小さい画像でも上限を確保し直していない。
    EXPECT_EQ(detector.workspace_statistics().allocation_count_, 1U);

    // もう一度大きい方へ戻しても壊れない。
    ASSERT_EQ(detector.detect_async(large_image.view(), nullptr, &message), Status::kOk) << message;
    ASSERT_EQ(detector.download(&detections, nullptr, &message), Status::kOk) << message;
    EXPECT_EQ(detections.ids_.size(), 1U);
}

// 境界値: 検出上限を超えたら kMarkerOverflow を返す。
TEST(DetectorTest, reports_marker_overflow) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const cv::aruco::Dictionary dictionary =
            cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_MIP_36h12);
    cv::Mat scene(720, 1280, CV_8UC1, cv::Scalar(235));
    for (int i = 0; i < 3; ++i) {
        cv::Mat marker;
        dictionary.generateImageMarker(i, kMarkerSidePx, marker, 1);
        marker.copyTo(scene(cv::Rect(60 + (i * 400), 260, kMarkerSidePx, kMarkerSidePx)));
    }
    DetectorConfig config = config_for(scene);
    config.max_markers_ = 2;

    SceneImage image;
    ASSERT_TRUE(image.upload(scene));
    Detector detector;
    std::string message;
    ASSERT_EQ(detector.initialize(table(), config, &message), Status::kOk) << message;
    ASSERT_EQ(detector.detect_async(image.view(), nullptr, &message), Status::kOk) << message;

    HostDetections detections;
    EXPECT_EQ(detector.download(&detections, nullptr, &message), Status::kMarkerOverflow);
    EXPECT_EQ(detections.ids_.size(), 2U);
    EXPECT_TRUE(detections.marker_overflow_);
    EXPECT_EQ(detections.accepted_total_, 3);
}

// 異常系: 支持しない設定の組み合わせを拒否する。
TEST(DetectorTest, rejects_unsupported_combinations) {
    DetectorConfig aruco3_without_refine;
    aruco3_without_refine.use_aruco3_detection_ = true;
    aruco3_without_refine.corner_refine_method_ = aruco3cuda::CornerRefineMethod::kNone;
    Detector first;
    std::string message;
    EXPECT_EQ(first.initialize(table(), aruco3_without_refine, &message), Status::kInvalidConfig);
    EXPECT_NE(message.find("corner_refine_method"), std::string::npos) << message;
    EXPECT_FALSE(first.initialized());

    DetectorConfig plain_with_refine = DetectorConfig::opencv_defaults();
    plain_with_refine.corner_refine_method_ = aruco3cuda::CornerRefineMethod::kSubpix;
    Detector second;
    EXPECT_EQ(second.initialize(table(), plain_with_refine, &message), Status::kInvalidConfig);
    EXPECT_FALSE(second.initialized());
}

// 異常系: 引数が不正なら実行しない。
TEST(DetectorTest, rejects_invalid_arguments) {
    Detector detector;
    std::string message;
    DeviceDetections device;
    HostDetections host;

    // initialize 前。
    EXPECT_EQ(detector.device_detections(&device), Status::kNotInitialized);
    EXPECT_EQ(detector.download(&host, nullptr, &message), Status::kNotInitialized);
    aruco3cuda::ImageViewU8 empty;
    EXPECT_EQ(detector.detect_async(empty, nullptr, &message), Status::kNotInitialized);
    EXPECT_EQ(detector.workspace_statistics().allocation_count_, 0U);

    aruco3cuda::DictionaryTable broken;
    EXPECT_EQ(detector.initialize(broken, DetectorConfig(), &message),
              Status::kUnsupportedDictionary);

    DetectorConfig invalid;
    invalid.adaptive_thresh_win_size_min_px_ = 2;
    EXPECT_EQ(detector.initialize(table(), invalid, &message), Status::kInvalidConfig);

    if (!has_cuda_device()) {
        return;
    }
    const cv::Mat scene = make_scene(640, 480, cv::Point(200, 150));
    ASSERT_EQ(detector.initialize(table(), config_for(scene), &message), Status::kOk) << message;
    EXPECT_EQ(detector.device_detections(nullptr), Status::kInvalidArgument);
    EXPECT_EQ(detector.download(nullptr, nullptr, &message), Status::kInvalidArgument);

    // host memory の画像は受け取らない。kernel が直接読むためである。
    aruco3cuda::ImageViewU8 host_view;
    host_view.data_ = scene.data;
    host_view.width_px_ = scene.cols;
    host_view.height_px_ = scene.rows;
    host_view.pitch_bytes_ = scene.step;
    host_view.space_ = aruco3cuda::MemorySpace::kHostPageable;
    EXPECT_EQ(detector.detect_async(host_view, nullptr, &message), Status::kInvalidImage);

    // 上限を超える寸法。
    const cv::Mat oversized = make_scene(1280, 720, cv::Point(400, 260));
    SceneImage big;
    ASSERT_TRUE(big.upload(oversized));
    EXPECT_EQ(detector.detect_async(big.view(), nullptr, &message), Status::kInvalidArgument);
}

// 正常系: move しても使える。
TEST(DetectorTest, supports_move) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const cv::Mat scene = make_scene(640, 480, cv::Point(200, 150));
    SceneImage image;
    ASSERT_TRUE(image.upload(scene));

    Detector source;
    std::string message;
    ASSERT_EQ(source.initialize(table(), config_for(scene), &message), Status::kOk) << message;
    Detector moved(std::move(source));
    EXPECT_TRUE(moved.initialized());
    ASSERT_EQ(moved.detect_async(image.view(), nullptr, &message), Status::kOk) << message;
    HostDetections detections;
    ASSERT_EQ(moved.download(&detections, nullptr, &message), Status::kOk) << message;
    EXPECT_EQ(detections.ids_.size(), 1U);

    // move 元は初期化前の状態として扱える。落ちてはならない。
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_FALSE(source.initialized());
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_EQ(source.device_detections(nullptr), Status::kNotInitialized);
}

}  // namespace
