// SPDX-License-Identifier: Apache-2.0
//
// Verifies the detector exposed through the public API.
//
// The completion criterion is that a test can confirm the detection results are
// reachable without a host synchronization. Merely being able to call the API
// is not enough, so a kernel that occupies the stream is enqueued first, and we
// confirm that the stream is still running by the time detect_async returns.
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
#include <sstream>
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

/// Builds a scene with a single marker placed in it.
cv::Mat make_scene(int width, int height, const cv::Point& origin, int id = kMarkerId) {
    const cv::aruco::Dictionary dictionary =
            cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_MIP_36h12);
    cv::Mat marker;
    dictionary.generateImageMarker(id, kMarkerSidePx, marker, 1);
    cv::Mat scene(height, width, CV_8UC1, cv::Scalar(235));
    marker.copyTo(scene(cv::Rect(origin.x, origin.y, kMarkerSidePx, kMarkerSidePx)));
    return scene;
}

/// Builds a configuration whose input limits match the scene.
DetectorConfig config_for(const cv::Mat& scene) {
    DetectorConfig config;
    config.max_width_px_ = scene.cols;
    config.max_height_px_ = scene.rows;
    return config;
}

/// Uploads the scene to the device.
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

/// OpenCV's detection result, used as the comparison reference.
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

/// Extracts the four corners of a single detection.
std::vector<cv::Point2f> corners_of(const HostDetections& detections, std::size_t index) {
    std::vector<cv::Point2f> result;
    result.reserve(4U);
    for (int corner = 0; corner < 4; ++corner) {
        const std::size_t base = (index * 8U) + (static_cast<std::size_t>(corner) * 2U);
        result.emplace_back(detections.corners_[base], detections.corners_[base + 1U]);
    }
    return result;
}

/// Maximum vertex distance between two quads, allowing for a cyclic shift.
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

// Happy path: initializes and detects a marker.
TEST(DetectorTest, detects_marker_and_matches_opencv) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
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
    std::printf("[detector] ID %d, corner deviation %.4f px\n", detections.ids_[0], distance);
    EXPECT_LT(distance, 1.0);
}

// Happy path: detect_async does not synchronize with the host.
//
// A kernel that occupies the stream for a long time is enqueued first, and
// detect_async is enqueued behind it. If the call synchronized with the host,
// the occupation would have finished before the call returned and the stream
// would be complete. If it does not synchronize, the stream is still running.
// The suite name contains Timing on purpose: under Compute Sanitizer the wall
// time of the occupying kernel cannot be read and the ordering between dispatch
// and occupation breaks down, so this suite is on the default exclusion list.
TEST(DetectorTimingTest, detect_async_does_not_synchronize) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    const cv::Mat scene = make_scene(1280, 720, cv::Point(400, 260));
    const DetectorConfig config = config_for(scene);

    SceneImage image;
    ASSERT_TRUE(image.upload(scene));

    Detector detector;
    std::string message;
    ASSERT_EQ(detector.initialize(table(), config, &message), Status::kOk) << message;

    // The default stream synchronizes implicitly with other streams, so create
    // a dedicated one.
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    int* sink = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&sink), sizeof(int)), cudaSuccess);

    // Run once to settle the layout. The input is identical afterwards, so the
    // layout is not rebuilt.
    ASSERT_EQ(detector.detect_async(image.view(), stream, &message), Status::kOk) << message;
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    // The clock frequency differs across machines. Enqueue a generously large
    // value and check only the lower bound.
    constexpr long long kSpinCycles = 2000000000LL;
    ASSERT_TRUE(aruco3cuda::test::enqueue_spin(kSpinCycles, sink, stream));

    const auto start = std::chrono::steady_clock::now();
    const Status status = detector.detect_async(image.view(), stream, &message);
    DeviceDetections device;
    const Status view_status = detector.device_detections(&device);
    const auto finish = std::chrono::steady_clock::now();
    const cudaError_t query = cudaStreamQuery(stream);
    const double elapsed_ms = std::chrono::duration<double, std::milli>(finish - start).count();

    // Clean up first so that no resource is left behind if an assertion fails.
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

    std::printf("[detector] dispatch %.3f ms, until occupation completes %.3f ms, query=%s\n",
                elapsed_ms, spin_ms, cudaGetErrorName(query));
    // First confirm the occupation was long enough; if it was short, this test
    // measured nothing.
    ASSERT_GT(spin_ms, 50.0) << "the spin kernel is too short; increase the cycle count";
    // The dispatch must not have waited for the occupation to complete.
    EXPECT_EQ(query, cudaErrorNotReady);
    EXPECT_LT(elapsed_ms, spin_ms / 2.0);
}

// Happy path: the device-resident results are reachable without a host
// synchronization.
TEST(DetectorTest, device_detections_needs_no_synchronization) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
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

    // Read the device pointer directly, checking agreement without going
    // through the host-side repacking.
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

// Happy path: the workspace is not reallocated on every frame.
TEST(DetectorTest, allocates_workspace_once) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
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
    std::printf("[detector] workspace %zu bytes, %zu allocations\n",
                detector.workspace_statistics().capacity_bytes_,
                detector.workspace_statistics().allocation_count_);
}

// Happy path: detects multiple markers.
TEST(DetectorTest, detects_multiple_markers) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
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
    std::printf("[detector] detected %zu markers\n", found.size());
    EXPECT_EQ(found, ids);
    EXPECT_EQ(detections.corners_.size(), found.size() * 8U);
    EXPECT_FALSE(detections.marker_overflow_);
}

// Boundary: detection stays correct when the input dimensions change.
TEST(DetectorTest, handles_changed_input_size) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
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
    // Even for the smaller image the capacity was not reallocated.
    EXPECT_EQ(detector.workspace_statistics().allocation_count_, 1U);

    // Switching back to the larger image does not break anything.
    ASSERT_EQ(detector.detect_async(large_image.view(), nullptr, &message), Status::kOk) << message;
    ASSERT_EQ(detector.download(&detections, nullptr, &message), Status::kOk) << message;
    EXPECT_EQ(detections.ids_.size(), 1U);
}

/// Creates an explicit stream and always destroys it.
class OwnedStream {
public:
    OwnedStream() {
        if (cudaStreamCreateWithFlags(&this->stream_, cudaStreamNonBlocking) != cudaSuccess) {
            this->stream_ = nullptr;
        }
    }
    ~OwnedStream() {
        if (this->stream_ != nullptr) {
            static_cast<void>(cudaStreamDestroy(this->stream_));
        }
    }
    OwnedStream(const OwnedStream&) = delete;
    OwnedStream& operator=(const OwnedStream&) = delete;
    cudaStream_t get() const { return this->stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

/// Reduces a detection result to a comparable form.
std::string digest_of(const HostDetections& detections) {
    std::ostringstream out;
    for (std::size_t i = 0; i < detections.ids_.size(); ++i) {
        out << detections.ids_[i] << ':' << detections.rotations_[i] << '(';
        for (int corner = 0; corner < 8; ++corner) {
            out << detections.corners_[(i * 8U) + static_cast<std::size_t>(corner)] << ',';
        }
        out << ") ";
    }
    return out.str();
}

// Happy path: folding the dispatch sequence does not change the result.
//
// Passing an explicit stream folds the dispatch sequence into a CUDA graph.
// Because a graph bakes in the kernel arguments, the hazard is not rounding but
// **stale references**. We compare directly against the default stream (the
// path that does not fold).
TEST(DetectorTest, graph_path_agrees_with_direct_dispatch) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    const cv::Mat scene = make_scene(1280, 720, cv::Point(400, 260));
    const DetectorConfig config = config_for(scene);
    SceneImage image;
    ASSERT_TRUE(image.upload(scene));
    OwnedStream stream;
    ASSERT_NE(stream.get(), nullptr);

    Detector direct;
    Detector graph;
    std::string message;
    ASSERT_EQ(direct.initialize(table(), config, &message), Status::kOk) << message;
    ASSERT_EQ(graph.initialize(table(), config, &message), Status::kOk) << message;

    HostDetections expected;
    ASSERT_EQ(direct.detect_async(image.view(), nullptr, &message), Status::kOk) << message;
    ASSERT_EQ(direct.download(&expected, nullptr, &message), Status::kOk) << message;
    ASSERT_EQ(expected.ids_.size(), 1U);

    // Run three frames: the first captures, and the rest launch the folded
    // sequence.
    for (int frame = 0; frame < 3; ++frame) {
        HostDetections actual;
        ASSERT_EQ(graph.detect_async(image.view(), stream.get(), &message), Status::kOk)
                << "frame " << frame << " " << message;
        ASSERT_EQ(graph.download(&actual, stream.get(), &message), Status::kOk) << message;
        EXPECT_EQ(digest_of(actual), digest_of(expected)) << "frame " << frame;
    }
}

// Boundary: the dispatch sequence is recaptured when the input dimensions
// change.
//
// A captured sequence bakes in the workspace pointers. A change in dimensions
// rebuilds the layout, so without a recapture the graph would **silently poke
// at stale pointers**.
TEST(DetectorTest, graph_is_rebuilt_when_layout_changes) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    const cv::Mat large = make_scene(1280, 720, cv::Point(400, 260));
    const cv::Mat small = make_scene(960, 540, cv::Point(300, 180), 7);
    DetectorConfig config = config_for(large);
    SceneImage large_image;
    SceneImage small_image;
    ASSERT_TRUE(large_image.upload(large));
    ASSERT_TRUE(small_image.upload(small));
    OwnedStream stream;
    ASSERT_NE(stream.get(), nullptr);

    // Produce the expected values through the non-folding path.
    Detector direct;
    std::string message;
    ASSERT_EQ(direct.initialize(table(), config, &message), Status::kOk) << message;
    HostDetections expected_large;
    HostDetections expected_small;
    ASSERT_EQ(direct.detect_async(large_image.view(), nullptr, &message), Status::kOk) << message;
    ASSERT_EQ(direct.download(&expected_large, nullptr, &message), Status::kOk) << message;
    ASSERT_EQ(direct.detect_async(small_image.view(), nullptr, &message), Status::kOk) << message;
    ASSERT_EQ(direct.download(&expected_small, nullptr, &message), Status::kOk) << message;
    ASSERT_EQ(expected_large.ids_.size(), 1U);
    ASSERT_EQ(expected_small.ids_.size(), 1U);
    ASSERT_NE(expected_large.ids_[0], expected_small.ids_[0]);

    Detector graph;
    ASSERT_EQ(graph.initialize(table(), config, &message), Status::kOk) << message;
    // Alternate large, small, large, small; a recapture happens every time.
    const std::vector<std::pair<const SceneImage*, const HostDetections*>> sequence = {
            {&large_image, &expected_large},
            {&small_image, &expected_small},
            {&large_image, &expected_large},
            {&small_image, &expected_small},
    };
    for (std::size_t step = 0; step < sequence.size(); ++step) {
        HostDetections actual;
        ASSERT_EQ(graph.detect_async(sequence[step].first->view(), stream.get(), &message),
                  Status::kOk)
                << "step " << step << " " << message;
        ASSERT_EQ(graph.download(&actual, stream.get(), &message), Status::kOk) << message;
        EXPECT_EQ(digest_of(actual), digest_of(*sequence[step].second)) << "step " << step;
    }
    // Even across layout rebuilds the workspace was not reallocated.
    EXPECT_EQ(graph.workspace_statistics().allocation_count_, 1U);
}

// Boundary: the dispatch sequence is recaptured when the stream changes.
TEST(DetectorTest, graph_is_rebuilt_when_stream_changes) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    const cv::Mat scene = make_scene(1280, 720, cv::Point(400, 260));
    SceneImage image;
    ASSERT_TRUE(image.upload(scene));
    OwnedStream first;
    OwnedStream second;
    ASSERT_NE(first.get(), nullptr);
    ASSERT_NE(second.get(), nullptr);

    Detector detector;
    std::string message;
    ASSERT_EQ(detector.initialize(table(), config_for(scene), &message), Status::kOk) << message;

    // Hand over the first stream, the second stream, the default stream, then
    // the first stream again.
    const std::vector<cudaStream_t> streams = {first.get(), second.get(), nullptr, first.get()};
    for (std::size_t step = 0; step < streams.size(); ++step) {
        HostDetections actual;
        ASSERT_EQ(detector.detect_async(image.view(), streams[step], &message), Status::kOk)
                << "step " << step << " " << message;
        ASSERT_EQ(detector.download(&actual, streams[step], &message), Status::kOk) << message;
        ASSERT_EQ(actual.ids_.size(), 1U) << "step " << step;
        EXPECT_EQ(actual.ids_[0], kMarkerId) << "step " << step;
    }
}

// Boundary: reinitializing with a changed configuration discards the dispatch
// sequence.
TEST(DetectorTest, graph_is_rebuilt_after_reinitialize) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    const cv::Mat scene = make_scene(1280, 720, cv::Point(400, 260));
    SceneImage image;
    ASSERT_TRUE(image.upload(scene));
    OwnedStream stream;
    ASSERT_NE(stream.get(), nullptr);

    Detector detector;
    std::string message;
    ASSERT_EQ(detector.initialize(table(), config_for(scene), &message), Status::kOk) << message;
    HostDetections first;
    ASSERT_EQ(detector.detect_async(image.view(), stream.get(), &message), Status::kOk) << message;
    ASSERT_EQ(detector.download(&first, stream.get(), &message), Status::kOk) << message;
    ASSERT_EQ(first.ids_.size(), 1U);

    // Switch to a configuration without downscaling; the segmentation
    // dimensions and the pyramid level count both change.
    DetectorConfig changed = config_for(scene);
    changed.min_marker_length_ratio_original_img_ = 0.0F;
    changed.min_side_length_canonical_img_px_ = 32;
    ASSERT_EQ(detector.initialize(table(), changed, &message), Status::kOk) << message;
    HostDetections second;
    ASSERT_EQ(detector.detect_async(image.view(), stream.get(), &message), Status::kOk) << message;
    ASSERT_EQ(detector.download(&second, stream.get(), &message), Status::kOk) << message;
    ASSERT_EQ(second.ids_.size(), 1U);
    EXPECT_EQ(second.ids_[0], kMarkerId);
    // Without downscaling the corners come out at full resolution. They are not
    // guaranteed to land where the previous configuration put them, but they
    // must not stray from the marker's position.
    EXPECT_NEAR(static_cast<double>(second.corners_[0]), 400.0, 3.0);
    EXPECT_NEAR(static_cast<double>(second.corners_[1]), 260.0, 3.0);
}

// Boundary: exceeding the detection limit returns kMarkerOverflow.
TEST(DetectorTest, reports_marker_overflow) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
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

// Failure path: rejects unsupported configuration combinations.
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

// Failure path: invalid arguments perform no work.
TEST(DetectorTest, rejects_invalid_arguments) {
    Detector detector;
    std::string message;
    DeviceDetections device;
    HostDetections host;

    // Before initialize.
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

    // An image in host memory is not accepted, because the kernels read it
    // directly.
    aruco3cuda::ImageViewU8 host_view;
    host_view.data_ = scene.data;
    host_view.width_px_ = scene.cols;
    host_view.height_px_ = scene.rows;
    host_view.pitch_bytes_ = scene.step;
    host_view.space_ = aruco3cuda::MemorySpace::kHostPageable;
    EXPECT_EQ(detector.detect_async(host_view, nullptr, &message), Status::kInvalidImage);

    // Dimensions beyond the configured limits.
    const cv::Mat oversized = make_scene(1280, 720, cv::Point(400, 260));
    SceneImage big;
    ASSERT_TRUE(big.upload(oversized));
    EXPECT_EQ(detector.detect_async(big.view(), nullptr, &message), Status::kInvalidArgument);
}

// Happy path: still usable after a move.
TEST(DetectorTest, supports_move) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
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

    // The moved-from object behaves as if uninitialized; it must not crash.
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_FALSE(source.initialized());
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_EQ(source.device_detections(nullptr), Status::kNotInitialized);
}

}  // namespace
