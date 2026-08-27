// SPDX-License-Identifier: Apache-2.0
//
// 前処理 kernel を OpenCV の結果と突き合わせる。
//
// 候補抽出の入力になるため、ここでの差は以降の全段階へ伝播する。
// pyrDown は整数演算であり完全一致を要求する。resize は浮動小数点の
// 丸めが残るため、許容差を実測して固定する。
#include "preprocess.hpp"

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
#include <vector>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"
#include "aruco3cuda/workspace.hpp"

namespace {

using aruco3cuda::DetectorConfig;
using aruco3cuda::ImageViewU8;
using aruco3cuda::MemorySpace;
using aruco3cuda::Status;
using aruco3cuda::Workspace;
using aruco3cuda::detail::PreprocessBuffers;
using aruco3cuda::detail::ScalePlan;

bool has_cuda_device() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

/// 決定的な試験画像を作る。
///
/// 勾配と市松模様と noise を混ぜる。平坦な画像では補間の差が現れず、
/// 一致していると誤って判断してしまう。
cv::Mat make_test_image(int width, int height, std::uint64_t seed) {
    cv::Mat image(height, width, CV_8UC1);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> noise(0, 40);
    for (int y = 0; y < height; ++y) {
        auto* row = image.ptr<std::uint8_t>(y);
        for (int x = 0; x < width; ++x) {
            const int gradient = (x * 200) / std::max(width - 1, 1);
            const int checker = ((x / 17 + y / 13) % 2) * 40;
            row[x] = cv::saturate_cast<std::uint8_t>(gradient + checker + noise(rng));
        }
    }
    return image;
}

/// device 上の画像を保持する。test の中でだけ使う簡易な RAII。
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
        this->pitch_bytes_ = static_cast<std::size_t>(image.cols) + 64U;
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

/// device 上の面を host の Mat へ取り出す。
cv::Mat download(const ImageViewU8& view) {
    cv::Mat result(view.height_px_, view.width_px_, CV_8UC1);
    const cudaError_t error =
            cudaMemcpy2D(result.data, static_cast<std::size_t>(result.step), view.data_,
                         view.pitch_bytes_, static_cast<std::size_t>(view.width_px_),
                         static_cast<std::size_t>(view.height_px_), cudaMemcpyDeviceToHost);
    EXPECT_EQ(error, cudaSuccess) << cudaGetErrorString(error);
    return result;
}

struct Difference {
    int max_abs = 0;
    double mismatch_ratio = 0.0;
};

Difference compare(const cv::Mat& expected, const cv::Mat& actual) {
    Difference difference;
    EXPECT_EQ(expected.size(), actual.size());
    if (expected.size() != actual.size()) {
        return difference;
    }
    std::size_t mismatched = 0;
    for (int y = 0; y < expected.rows; ++y) {
        const auto* e = expected.ptr<std::uint8_t>(y);
        const auto* a = actual.ptr<std::uint8_t>(y);
        for (int x = 0; x < expected.cols; ++x) {
            const int diff = std::abs(static_cast<int>(e[x]) - static_cast<int>(a[x]));
            difference.max_abs = std::max(difference.max_abs, diff);
            if (diff != 0) {
                ++mismatched;
            }
        }
    }
    const auto total = static_cast<double>(expected.total());
    difference.mismatch_ratio = total > 0.0 ? static_cast<double>(mismatched) / total : 0.0;
    return difference;
}

/// 前処理を一式実行して buffer を返す。
class PreprocessFixture {
public:
    bool run(const cv::Mat& image, const DetectorConfig& config) {
        if (!this->input_.upload(image)) {
            return false;
        }
        if (aruco3cuda::detail::plan_scales(config, image.cols, image.rows, &this->plan_) !=
            Status::kOk) {
            return false;
        }
        const std::size_t bytes =
                aruco3cuda::detail::preprocess_workspace_bytes(this->plan_, image.cols, image.rows);
        if (bytes == 0U) {
            return false;
        }
        if (this->workspace_.ensure_capacity(bytes, MemorySpace::kDevice, nullptr) != Status::kOk) {
            return false;
        }
        this->workspace_.reset();
        if (aruco3cuda::detail::reserve_preprocess(this->plan_, this->input_.view(),
                                                   this->workspace_,
                                                   &this->buffers_) != Status::kOk) {
            return false;
        }
        if (aruco3cuda::detail::build_pyramid_async(&this->buffers_, config, nullptr) !=
            Status::kOk) {
            return false;
        }
        if (aruco3cuda::detail::build_segmentation_async(this->plan_, &this->buffers_, config,
                                                         nullptr) != Status::kOk) {
            return false;
        }
        return cudaDeviceSynchronize() == cudaSuccess;
    }

    ScalePlan plan_;
    Workspace workspace_;
    PreprocessBuffers buffers_;
    DeviceImage input_;
};

// 正常系: 縮小計画が観測仕様どおりの値を返す。
// docs/design/detector-pipeline.md に記録した式と一致することを固定する。
TEST(ScalePlanTest, matches_documented_formulas) {
    DetectorConfig config;
    config.use_aruco3_detection_ = true;
    config.min_side_length_canonical_img_px_ = 32;
    config.min_marker_length_ratio_original_img_ = 0.05F;

    ScalePlan plan;
    ASSERT_EQ(aruco3cuda::detail::plan_scales(config, 1280, 720, &plan), Status::kOk);
    // fxfy = 32 / (32 + 1280 * 0.05) = 32 / 96
    // OpenCV と同じく単精度で計算するため、許容差は単精度の分解能に合わせる。
    EXPECT_NEAR(plan.fxfy_, 1.0 / 3.0, 1e-6);
    EXPECT_EQ(plan.segmentation_width_px_, 427);
    EXPECT_EQ(plan.segmentation_height_px_, 240);
    // num_levels = (int)(log2(1280*720 / 32^2) / 2) = (int)(log2(900) / 2) = 4
    EXPECT_EQ(plan.level_count_, 5);
    EXPECT_GE(plan.closest_level_index_, 0);
    EXPECT_LT(plan.closest_level_index_, plan.level_count_);
}

// 正常系: segmentation の寸法が OpenCV の計算と一致する。
// OpenCV は fxfy を単精度で求め、寸法を cvRound で丸める。演算の型と
// 丸め規約を揃えないと、境界で 1 pixel 食い違う。
TEST(ScalePlanTest, segmentation_size_matches_opencv_arithmetic) {
    struct Case {
        int width;
        int height;
        int side;
        float ratio;
    };
    const Case cases[] = {
            {640, 480, 32, 0.05F},   {1280, 720, 32, 0.05F}, {1920, 1080, 32, 0.05F},
            {3840, 2160, 32, 0.05F}, {1280, 720, 32, 0.02F}, {1280, 720, 64, 0.1F},
            {1000, 1000, 32, 0.03F}, {333, 777, 48, 0.07F},
    };
    for (const Case& item : cases) {
        DetectorConfig config;
        config.use_aruco3_detection_ = true;
        config.min_side_length_canonical_img_px_ = item.side;
        config.min_marker_length_ratio_original_img_ = item.ratio;

        ScalePlan plan;
        ASSERT_EQ(aruco3cuda::detail::plan_scales(config, item.width, item.height, &plan),
                  Status::kOk);

        // OpenCV と同じ式を同じ型で計算する。
        const float side = static_cast<float>(item.side);
        const float longest = static_cast<float>(std::max(item.width, item.height));
        const float fxfy = side / (side + longest * item.ratio);
        EXPECT_EQ(plan.segmentation_width_px_, cvRound(fxfy * item.width))
                << item.width << "x" << item.height;
        EXPECT_EQ(plan.segmentation_height_px_, cvRound(fxfy * item.height))
                << item.width << "x" << item.height;
    }
}

// 境界値: ArUco3 を無効にすると縮小せず pyramid も 1 段になる。
TEST(ScalePlanTest, disabled_aruco3_has_no_downscale) {
    DetectorConfig config = DetectorConfig::opencv_defaults();
    ScalePlan plan;
    ASSERT_EQ(aruco3cuda::detail::plan_scales(config, 1280, 720, &plan), Status::kOk);
    EXPECT_DOUBLE_EQ(plan.fxfy_, 1.0);
    EXPECT_EQ(plan.segmentation_width_px_, 1280);
    EXPECT_EQ(plan.segmentation_height_px_, 720);
    EXPECT_EQ(plan.level_count_, 1);
}

// 異常系: 不正な引数を拒否する。
TEST(ScalePlanTest, rejects_invalid_arguments) {
    const DetectorConfig config;
    ScalePlan plan;
    EXPECT_EQ(aruco3cuda::detail::plan_scales(config, 1280, 720, nullptr),
              Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::detail::plan_scales(config, 0, 720, &plan), Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::detail::plan_scales(config, 1280, 0, &plan), Status::kInvalidArgument);
}

// 正常系: pyramid が OpenCV の buildPyramid と完全に一致する。
// pyrDown は整数演算であり、丸めまで含めて再現できる。
TEST(PyramidTest, matches_opencv_build_pyramid_exactly) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const cv::Mat image = make_test_image(640, 480, 20260827U);
    DetectorConfig config;

    PreprocessFixture fixture;
    ASSERT_TRUE(fixture.run(image, config));

    std::vector<cv::Mat> expected;
    cv::buildPyramid(image, expected, fixture.plan_.level_count_ - 1);
    ASSERT_EQ(static_cast<int>(expected.size()), fixture.plan_.level_count_);

    for (int level = 1; level < fixture.plan_.level_count_; ++level) {
        const ImageViewU8 view = aruco3cuda::detail::level_view(fixture.buffers_, level);
        ASSERT_NE(view.data_, nullptr) << "level=" << level;
        EXPECT_EQ(view.width_px_, expected[level].cols) << "level=" << level;
        EXPECT_EQ(view.height_px_, expected[level].rows) << "level=" << level;

        const Difference difference = compare(expected[level], download(view));
        EXPECT_EQ(difference.max_abs, 0)
                << "level=" << level << " 一致率 " << (1.0 - difference.mismatch_ratio);
    }
}

// 正常系: level 0 は入力そのものを指し、複製しない。
TEST(PyramidTest, level_zero_references_input) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const cv::Mat image = make_test_image(320, 240, 7U);
    PreprocessFixture fixture;
    ASSERT_TRUE(fixture.run(image, DetectorConfig()));
    const ImageViewU8 level0 = aruco3cuda::detail::level_view(fixture.buffers_, 0);
    EXPECT_EQ(level0.data_, fixture.input_.view().data_);
    EXPECT_EQ(compare(image, download(level0)).max_abs, 0);
}

// 境界値: 範囲外の level は空の view を返す。
TEST(PyramidTest, out_of_range_level_returns_empty_view) {
    PreprocessBuffers buffers;
    buffers.level_count_ = 3;
    EXPECT_EQ(aruco3cuda::detail::level_view(buffers, -1).data_, nullptr);
    EXPECT_EQ(aruco3cuda::detail::level_view(buffers, 3).data_, nullptr);
}

// 正常系: segmentation 画像が OpenCV の resize と 1 階調以内で一致する。
//
// OpenCV の 8-bit INTER_LINEAR は INTER_LINEAR_EXACT と同じ結果を返す。
// これは softdouble による軟件浮動小数点と ufixedpoint16 で構成された
// bit exact 経路であり、平台間で同じ結果になることを目的としている。
// kernel 内で再現するにはこの 2 つの数値型を移植する必要があり、前処理
// 段階の代償として見合わない。
//
// 差は最大 1 階調に収まる。この差が下流の適応的二値化で画素の白黒を
// 入れ替える割合は、25% の画素へ無作為に 1 階調を加えた場合で 0.45% である。
// 既知かつ定量化された差異として記録し、Phase 2 の差異分類ではこの分を
// 候補抽出の設計差と区別する。
//
// 上限を 1 に固定することで、補間規約の食い違いのような別種の劣化を検出できる。
TEST(SegmentationTest, matches_opencv_resize_within_one_level) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    struct Case {
        int width;
        int height;
    };
    const Case cases[] = {{640, 480},   {1280, 720}, {1920, 1080},
                          {3840, 2160}, {333, 777},  {1000, 1000}};
    for (const Case& item : cases) {
        const cv::Mat image = make_test_image(item.width, item.height, 20260827U);
        DetectorConfig config;
        PreprocessFixture fixture;
        ASSERT_TRUE(fixture.run(image, config)) << item.width << "x" << item.height;

        cv::Mat expected;
        cv::resize(image, expected,
                   cv::Size(fixture.plan_.segmentation_width_px_,
                            fixture.plan_.segmentation_height_px_),
                   0.0, 0.0, cv::INTER_LINEAR);

        const cv::Mat actual = download(aruco3cuda::detail::level_view(fixture.buffers_, 0));
        (void)actual;
        const ImageViewU8 segmentation_view{
                fixture.buffers_.segmentation_.data_, fixture.buffers_.segmentation_.width_px_,
                fixture.buffers_.segmentation_.height_px_,
                fixture.buffers_.segmentation_.pitch_bytes_, MemorySpace::kDevice};
        const Difference difference = compare(expected, download(segmentation_view));

        // 1 階調を超える差は補間規約の食い違いを意味する。
        EXPECT_LE(difference.max_abs, 1)
                << item.width << "x" << item.height << " 最大差 " << difference.max_abs;
        // 不一致の割合も上限を設ける。実測は最大 0.372 であり、
        // 大きく増える変更は補間の劣化を意味する。
        EXPECT_LE(difference.mismatch_ratio, 0.45)
                << item.width << "x" << item.height << " 不一致率 " << difference.mismatch_ratio;
        std::printf("[resize] %dx%d -> %dx%d  最大差 %d  不一致率 %.4f\n", item.width, item.height,
                    fixture.plan_.segmentation_width_px_, fixture.plan_.segmentation_height_px_,
                    difference.max_abs, difference.mismatch_ratio);
    }
}

// 境界値: 縮小率が 1 のとき segmentation は入力の複製になる。
TEST(SegmentationTest, copies_input_when_scale_is_one) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const cv::Mat image = make_test_image(320, 240, 11U);
    const DetectorConfig config = DetectorConfig::opencv_defaults();
    PreprocessFixture fixture;
    ASSERT_TRUE(fixture.run(image, config));
    ASSERT_DOUBLE_EQ(fixture.plan_.fxfy_, 1.0);

    const ImageViewU8 segmentation_view{
            fixture.buffers_.segmentation_.data_, fixture.buffers_.segmentation_.width_px_,
            fixture.buffers_.segmentation_.height_px_, fixture.buffers_.segmentation_.pitch_bytes_,
            MemorySpace::kDevice};
    EXPECT_EQ(compare(image, download(segmentation_view)).max_abs, 0);
}

// 正常系: block 寸法を変えても結果が変わらない。
// 機種別 tuning で block 寸法を動かすため、結果が寸法に依存しないことを固定する。
TEST(PreprocessTest, result_is_independent_of_block_dim) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const cv::Mat image = make_test_image(640, 480, 3U);
    cv::Mat reference;
    for (const int block_dim : {8, 16, 32}) {
        DetectorConfig config;
        config.cuda_block_dim_ = block_dim;
        ASSERT_EQ(config.validate(nullptr), Status::kOk) << block_dim;

        PreprocessFixture fixture;
        ASSERT_TRUE(fixture.run(image, config)) << block_dim;
        const cv::Mat actual = download(aruco3cuda::detail::level_view(fixture.buffers_, 1));
        if (reference.empty()) {
            reference = actual;
        } else {
            EXPECT_EQ(compare(reference, actual).max_abs, 0) << "block_dim=" << block_dim;
        }
    }
}

// 異常系: 不正な引数を拒否する。
TEST(PreprocessTest, rejects_invalid_arguments) {
    const DetectorConfig config;
    EXPECT_EQ(aruco3cuda::detail::build_pyramid_async(nullptr, config, nullptr),
              Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::detail::build_segmentation_async(ScalePlan(), nullptr, config, nullptr),
              Status::kInvalidArgument);

    PreprocessBuffers empty;
    EXPECT_EQ(aruco3cuda::detail::build_pyramid_async(&empty, config, nullptr),
              Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::detail::build_segmentation_async(ScalePlan(), &empty, config, nullptr),
              Status::kInvalidArgument);

    ScalePlan plan;
    Workspace workspace;
    ImageViewU8 invalid;
    PreprocessBuffers buffers;
    EXPECT_EQ(aruco3cuda::detail::reserve_preprocess(plan, invalid, workspace, &buffers),
              Status::kInvalidImage);
    EXPECT_EQ(aruco3cuda::detail::reserve_preprocess(plan, invalid, workspace, nullptr),
              Status::kInvalidArgument);
}

// 異常系: workspace の容量が足りなければ切り出しに失敗する。
TEST(PreprocessTest, fails_when_workspace_is_too_small) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const cv::Mat image = make_test_image(640, 480, 5U);
    DeviceImage input;
    ASSERT_TRUE(input.upload(image));

    ScalePlan plan;
    ASSERT_EQ(aruco3cuda::detail::plan_scales(DetectorConfig(), image.cols, image.rows, &plan),
              Status::kOk);

    Workspace workspace;
    ASSERT_EQ(workspace.ensure_capacity(1024U, MemorySpace::kDevice, nullptr), Status::kOk);
    PreprocessBuffers buffers;
    EXPECT_EQ(aruco3cuda::detail::reserve_preprocess(plan, input.view(), workspace, &buffers),
              Status::kInvalidConfig);
}

}  // namespace
