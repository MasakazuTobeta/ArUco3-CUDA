// SPDX-License-Identifier: Apache-2.0
//
// 適応的二値化を OpenCV の adaptiveThreshold と突き合わせる。
//
// この段階の出力が候補抽出の入力になる。画素の白黒が入れ替わると輪郭が
// 変わるため、一致率を実測して固定する。あわせて、前段の resize に残る
// 1 階調差が二値化へ与える実際の影響も測る。
#include "threshold.hpp"

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"
#include "aruco3cuda/workspace.hpp"
#include "preprocess.hpp"

namespace {

using aruco3cuda::DetectorConfig;
using aruco3cuda::ImageViewU8;
using aruco3cuda::MemorySpace;
using aruco3cuda::Status;
using aruco3cuda::Workspace;
using aruco3cuda::detail::ThresholdBuffers;

bool has_cuda_device() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

/// 決定的な試験画像。マーカーに似た矩形と勾配と noise を混ぜる。
cv::Mat make_test_image(int width, int height, std::uint64_t seed) {
    cv::Mat image(height, width, CV_8UC1);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> noise(0, 30);
    for (int y = 0; y < height; ++y) {
        auto* row = image.ptr<std::uint8_t>(y);
        for (int x = 0; x < width; ++x) {
            const int gradient = (x * 150) / std::max(width - 1, 1);
            row[x] = cv::saturate_cast<std::uint8_t>(60 + gradient + noise(rng));
        }
    }
    // 黒い矩形を並べる。二値化の境界が輪郭として現れる条件を作る。
    for (int i = 0; i < 6; ++i) {
        const int size = 20 + i * 11;
        const cv::Rect rect((i * 53 + 11) % std::max(width - size - 1, 1),
                            (i * 37 + 7) % std::max(height - size - 1, 1), size, size);
        cv::rectangle(image, rect, cv::Scalar(15), cv::FILLED);
    }
    return image;
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
        // 幅と異なる pitch を使い、非連続配置でも正しく扱えることを同時に確認する。
        this->pitch_bytes_ = static_cast<std::size_t>(image.cols) + 32U;
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

cv::Mat download(const aruco3cuda::detail::ImagePlaneU8& plane) {
    cv::Mat result(plane.height_px_, plane.width_px_, CV_8UC1);
    const cudaError_t error =
            cudaMemcpy2D(result.data, static_cast<std::size_t>(result.step), plane.data_,
                         plane.pitch_bytes_, static_cast<std::size_t>(plane.width_px_),
                         static_cast<std::size_t>(plane.height_px_), cudaMemcpyDeviceToHost);
    EXPECT_EQ(error, cudaSuccess) << cudaGetErrorString(error);
    return result;
}

double mismatch_ratio(const cv::Mat& a, const cv::Mat& b) {
    EXPECT_EQ(a.size(), b.size());
    if (a.size() != b.size()) {
        return 1.0;
    }
    std::size_t mismatched = 0;
    for (int y = 0; y < a.rows; ++y) {
        const auto* pa = a.ptr<std::uint8_t>(y);
        const auto* pb = b.ptr<std::uint8_t>(y);
        for (int x = 0; x < a.cols; ++x) {
            if (pa[x] != pb[x]) {
                ++mismatched;
            }
        }
    }
    return static_cast<double>(mismatched) / static_cast<double>(a.total());
}

/// 二値化を実行して buffer を返す。
class ThresholdFixture {
public:
    bool run(const cv::Mat& image, const DetectorConfig& config) {
        if (!this->input_.upload(image)) {
            return false;
        }
        const std::size_t bytes =
                aruco3cuda::detail::threshold_workspace_bytes(config, image.cols, image.rows);
        if (bytes == 0U) {
            return false;
        }
        if (this->workspace_.ensure_capacity(bytes, MemorySpace::kDevice, nullptr) != Status::kOk) {
            return false;
        }
        this->workspace_.reset();
        if (aruco3cuda::detail::reserve_threshold(config, image.cols, image.rows, this->workspace_,
                                                  &this->buffers_) != Status::kOk) {
            return false;
        }
        if (aruco3cuda::detail::build_threshold_async(this->input_.view(), &this->buffers_, config,
                                                      nullptr) != Status::kOk) {
            return false;
        }
        return cudaDeviceSynchronize() == cudaSuccess;
    }

    Workspace workspace_;
    ThresholdBuffers buffers_;
    DeviceImage input_;
};

// 正常系: window 一覧が OpenCV の走査と一致する。
TEST(ThresholdWindowTest, matches_opencv_window_sequence) {
    DetectorConfig config;
    int sizes[aruco3cuda::kMaxAdaptiveThresholdWindows] = {};
    int count = 0;
    ASSERT_EQ(aruco3cuda::detail::threshold_window_sizes(
                      config, sizes, aruco3cuda::kMaxAdaptiveThresholdWindows, &count),
              Status::kOk);
    // 既定は min 3、max 23、step 10 であり 3、13、23 の 3 通り。
    ASSERT_EQ(count, 3);
    EXPECT_EQ(sizes[0], 3);
    EXPECT_EQ(sizes[1], 13);
    EXPECT_EQ(sizes[2], 23);
}

// 境界値: 偶数の window は奇数へ切り上げる。OpenCV の _threshold と同じ。
TEST(ThresholdWindowTest, rounds_even_windows_up) {
    DetectorConfig config;
    config.adaptive_thresh_win_size_min_px_ = 4;
    config.adaptive_thresh_win_size_max_px_ = 10;
    config.adaptive_thresh_win_size_step_px_ = 3;
    int sizes[aruco3cuda::kMaxAdaptiveThresholdWindows] = {};
    int count = 0;
    ASSERT_EQ(aruco3cuda::detail::threshold_window_sizes(
                      config, sizes, aruco3cuda::kMaxAdaptiveThresholdWindows, &count),
              Status::kOk);
    // 4、7、10 が 5、7、11 になる。
    ASSERT_EQ(count, 3);
    EXPECT_EQ(sizes[0], 5);
    EXPECT_EQ(sizes[1], 7);
    EXPECT_EQ(sizes[2], 11);
}

// 異常系: 不正な引数と容量不足を拒否する。
TEST(ThresholdWindowTest, rejects_invalid_arguments) {
    const DetectorConfig config;
    int sizes[4] = {};
    int count = 0;
    EXPECT_EQ(aruco3cuda::detail::threshold_window_sizes(config, nullptr, 4, &count),
              Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::detail::threshold_window_sizes(config, sizes, 4, nullptr),
              Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::detail::threshold_window_sizes(config, sizes, 0, &count),
              Status::kInvalidArgument);
    // 走査数より小さい capacity は拒否する。
    EXPECT_EQ(aruco3cuda::detail::threshold_window_sizes(config, sizes, 2, &count),
              Status::kInvalidConfig);
}

// 正常系: 3 通りの window で OpenCV の二値化と完全に一致する。
// 完了条件そのものである。
TEST(ThresholdTest, matches_opencv_adaptive_threshold) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    struct Case {
        int width;
        int height;
    };
    const Case cases[] = {{427, 240}, {640, 480}, {129, 97}};
    for (const Case& item : cases) {
        const cv::Mat image = make_test_image(item.width, item.height, 20260827U);
        DetectorConfig config;
        ThresholdFixture fixture;
        ASSERT_TRUE(fixture.run(image, config)) << item.width << "x" << item.height;

        for (int i = 0; i < fixture.buffers_.window_count_; ++i) {
            const int window = fixture.buffers_.window_sizes_px_[i];
            cv::Mat expected;
            cv::adaptiveThreshold(image, expected, 255, cv::ADAPTIVE_THRESH_MEAN_C,
                                  cv::THRESH_BINARY_INV, window, config.adaptive_thresh_constant_);
            const double ratio = mismatch_ratio(expected, download(fixture.buffers_.binary_[i]));
            std::printf("[threshold] %dx%d window=%2d 不一致率 %.6f\n", item.width, item.height,
                        window, ratio);
            EXPECT_DOUBLE_EQ(ratio, 0.0)
                    << item.width << "x" << item.height << " window=" << window;
        }
    }
}

// 正常系: 定数を変えても OpenCV と一致する。
TEST(ThresholdTest, matches_opencv_for_various_constants) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const cv::Mat image = make_test_image(320, 240, 5U);
    for (const double constant : {0.0, 3.0, 7.0, 7.9, 15.0}) {
        DetectorConfig config;
        config.adaptive_thresh_constant_ = constant;
        ThresholdFixture fixture;
        ASSERT_TRUE(fixture.run(image, config)) << constant;

        cv::Mat expected;
        cv::adaptiveThreshold(image, expected, 255, cv::ADAPTIVE_THRESH_MEAN_C,
                              cv::THRESH_BINARY_INV, fixture.buffers_.window_sizes_px_[1],
                              constant);
        EXPECT_DOUBLE_EQ(mismatch_ratio(expected, download(fixture.buffers_.binary_[1])), 0.0)
                << "定数 " << constant;
    }
}

// 正常系: block 寸法を変えても結果が変わらない。
TEST(ThresholdTest, result_is_independent_of_block_dim) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const cv::Mat image = make_test_image(320, 240, 9U);
    cv::Mat reference;
    for (const int block_dim : {8, 16, 32}) {
        DetectorConfig config;
        config.cuda_block_dim_ = block_dim;
        ThresholdFixture fixture;
        ASSERT_TRUE(fixture.run(image, config)) << block_dim;
        const cv::Mat actual = download(fixture.buffers_.binary_[2]);
        if (reference.empty()) {
            reference = actual;
        } else {
            EXPECT_DOUBLE_EQ(mismatch_ratio(reference, actual), 0.0) << "block_dim=" << block_dim;
        }
    }
}

// 異常系: 不正な引数を拒否する。
TEST(ThresholdTest, rejects_invalid_arguments) {
    const DetectorConfig config;
    ThresholdBuffers buffers;
    ImageViewU8 view;
    EXPECT_EQ(aruco3cuda::detail::build_threshold_async(view, nullptr, config, nullptr),
              Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::detail::build_threshold_async(view, &buffers, config, nullptr),
              Status::kInvalidArgument);

    Workspace workspace;
    EXPECT_EQ(aruco3cuda::detail::reserve_threshold(config, 0, 10, workspace, &buffers),
              Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::detail::reserve_threshold(config, 10, 10, workspace, nullptr),
              Status::kInvalidArgument);
    // 容量不足。
    EXPECT_EQ(aruco3cuda::detail::reserve_threshold(config, 640, 480, workspace, &buffers),
              Status::kInvalidConfig);
}

// 異常系: 入力の寸法が予約時と違えば拒否する。
TEST(ThresholdTest, rejects_size_mismatch) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const cv::Mat image = make_test_image(320, 240, 1U);
    DetectorConfig config;
    ThresholdFixture fixture;
    ASSERT_TRUE(fixture.run(image, config));

    DeviceImage other;
    ASSERT_TRUE(other.upload(make_test_image(160, 120, 1U)));
    EXPECT_EQ(aruco3cuda::detail::build_threshold_async(other.view(), &fixture.buffers_, config,
                                                        nullptr),
              Status::kInvalidArgument);
}

// 前段の resize に残る 1 階調差が、二値化へ与える実際の影響を測る。
// WP-1.3 で受け入れた差の下流影響を、模擬ではなく実測で確認する。
TEST(ThresholdTest, measures_impact_of_resize_difference) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const cv::Mat source = make_test_image(1280, 720, 20260827U);
    DetectorConfig config;

    // CUDA の前処理で作った segmentation 画像を取り出す。
    DeviceImage input;
    ASSERT_TRUE(input.upload(source));
    aruco3cuda::detail::ScalePlan plan;
    ASSERT_EQ(aruco3cuda::detail::plan_scales(config, source.cols, source.rows, &plan),
              Status::kOk);
    Workspace preprocess_workspace;
    ASSERT_EQ(preprocess_workspace.ensure_capacity(aruco3cuda::detail::preprocess_workspace_bytes(
                                                           plan, source.cols, source.rows),
                                                   MemorySpace::kDevice, nullptr),
              Status::kOk);
    aruco3cuda::detail::PreprocessBuffers preprocess;
    ASSERT_EQ(aruco3cuda::detail::reserve_preprocess(plan, input.view(), preprocess_workspace,
                                                     &preprocess),
              Status::kOk);
    ASSERT_EQ(aruco3cuda::detail::build_segmentation_async(plan, &preprocess, config, nullptr),
              Status::kOk);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const cv::Mat cuda_segmentation = download(preprocess.segmentation_);

    cv::Mat opencv_segmentation;
    cv::resize(source, opencv_segmentation,
               cv::Size(plan.segmentation_width_px_, plan.segmentation_height_px_), 0.0, 0.0,
               cv::INTER_LINEAR);

    // 同じ二値化を両方へ適用し、白黒が入れ替わる割合を測る。
    for (int window : {3, 13, 23}) {
        cv::Mat from_cuda;
        cv::Mat from_opencv;
        cv::adaptiveThreshold(cuda_segmentation, from_cuda, 255, cv::ADAPTIVE_THRESH_MEAN_C,
                              cv::THRESH_BINARY_INV, window, config.adaptive_thresh_constant_);
        cv::adaptiveThreshold(opencv_segmentation, from_opencv, 255, cv::ADAPTIVE_THRESH_MEAN_C,
                              cv::THRESH_BINARY_INV, window, config.adaptive_thresh_constant_);
        const double ratio = mismatch_ratio(from_opencv, from_cuda);
        std::printf("[resize の影響] window=%2d 二値化の反転 %.6f\n", window, ratio);
        // 実測に基づく上限。大きく増える変更は前処理の劣化を意味する。
        EXPECT_LT(ratio, 0.01) << "window=" << window;
    }
}

}  // namespace
