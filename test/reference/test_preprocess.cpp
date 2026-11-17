// SPDX-License-Identifier: Apache-2.0
//
// Cross-checks the preprocessing kernels against the OpenCV results.
//
// This stage feeds candidate extraction, so any difference here propagates to
// every stage downstream. pyrDown is integer arithmetic, so an exact match is
// required. Segmentation resize is exact for the matching pinned CPU HAL
// profile; the legacy GENERIC profile retains its cross-HAL diagnostic bound.
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
#include <cstring>
#include <random>
#include <string>
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

/// Builds a deterministic test image.
///
/// Mixes a gradient, a checkerboard, and noise. On a flat image the differences
/// in interpolation never show up, which would lead to the wrong conclusion that
/// the two implementations agree.
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

/// Holds an image on the device. A minimal RAII helper used only inside this test.
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

    bool upload(const cv::Mat& image, std::size_t padding = 64U) {
        this->width_px_ = image.cols;
        this->height_px_ = image.rows;
        // Use a pitch that differs from the width, so that a non-contiguous layout
        // is exercised at the same time.
        this->pitch_bytes_ = static_cast<std::size_t>(image.cols) + padding;
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

/// Copies a device-side plane back into a host Mat.
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

/// Runs the full preprocessing chain and holds on to the resulting buffers.
class PreprocessFixture {
public:
    bool run(const cv::Mat& image, const DetectorConfig& config, cv::Size target = {},
             bool prepare_coefficients = true, bool all_resize_modes = true) {
        if (!this->input_.upload(image)) {
            return false;
        }
        if (aruco3cuda::detail::plan_scales(config, image.cols, image.rows, &this->plan_) !=
            Status::kOk) {
            return false;
        }
        if (target.width > 0 && target.height > 0) {
            this->plan_.segmentation_width_px_ = target.width;
            this->plan_.segmentation_height_px_ = target.height;
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
                                                   this->workspace_, &this->buffers_,
                                                   all_resize_modes) != Status::kOk) {
            return false;
        }
        if (prepare_coefficients && aruco3cuda::detail::prepare_segmentation_resize(
                                            &this->buffers_, config, nullptr) != Status::kOk)
            return false;
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

// Nominal: the downscale plan returns the values the observed specification calls for.
// This pins the plan to the formulas recorded in docs/design/detector-pipeline.md.
TEST(ScalePlanTest, matches_documented_formulas) {
    DetectorConfig config;
    config.use_aruco3_detection_ = true;
    config.min_side_length_canonical_img_px_ = 32;
    config.min_marker_length_ratio_original_img_ = 0.05F;

    ScalePlan plan;
    ASSERT_EQ(aruco3cuda::detail::plan_scales(config, 1280, 720, &plan), Status::kOk);
    // fxfy = 32 / (32 + 1280 * 0.05) = 32 / 96
    // The computation runs in single precision, as OpenCV does, so the tolerance
    // matches the resolution of single precision.
    EXPECT_NEAR(plan.fxfy_, 1.0 / 3.0, 1e-6);
    EXPECT_EQ(plan.segmentation_width_px_, 427);
    EXPECT_EQ(plan.segmentation_height_px_, 240);
    // num_levels = (int)(log2(1280*720 / 32^2) / 2) = (int)(log2(900) / 2) = 4
    EXPECT_EQ(plan.level_count_, 5);
    EXPECT_GE(plan.closest_level_index_, 0);
    EXPECT_LT(plan.closest_level_index_, plan.level_count_);
}

// Nominal: the segmentation size matches the OpenCV arithmetic.
// OpenCV computes fxfy in single precision and rounds the size with cvRound.
// Unless the arithmetic type and the rounding convention are matched, the sizes
// disagree by one pixel at the boundaries.
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

        // Compute the same formula in the same type OpenCV uses.
        const float side = static_cast<float>(item.side);
        const float longest = static_cast<float>(std::max(item.width, item.height));
        const float fxfy = side / (side + longest * item.ratio);
        EXPECT_EQ(plan.segmentation_width_px_, cvRound(fxfy * item.width))
                << item.width << "x" << item.height;
        EXPECT_EQ(plan.segmentation_height_px_, cvRound(fxfy * item.height))
                << item.width << "x" << item.height;
    }
}

// Boundary: with ArUco3 disabled there is no downscaling and the pyramid has a single level.
TEST(ScalePlanTest, disabled_aruco3_has_no_downscale) {
    DetectorConfig config = DetectorConfig::opencv_defaults();
    ScalePlan plan;
    ASSERT_EQ(aruco3cuda::detail::plan_scales(config, 1280, 720, &plan), Status::kOk);
    EXPECT_DOUBLE_EQ(plan.fxfy_, 1.0);
    EXPECT_EQ(plan.segmentation_width_px_, 1280);
    EXPECT_EQ(plan.segmentation_height_px_, 720);
    EXPECT_EQ(plan.level_count_, 1);
}

// Failure: invalid arguments are rejected.
TEST(ScalePlanTest, rejects_invalid_arguments) {
    const DetectorConfig config;
    ScalePlan plan;
    EXPECT_EQ(aruco3cuda::detail::plan_scales(config, 1280, 720, nullptr),
              Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::detail::plan_scales(config, 0, 720, &plan), Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::detail::plan_scales(config, 1280, 0, &plan), Status::kInvalidArgument);
}

// Nominal: the pyramid matches the OpenCV buildPyramid exactly.
// pyrDown is integer arithmetic, so it can be reproduced down to the rounding.
TEST(PyramidTest, matches_opencv_build_pyramid_exactly) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
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
                << "level=" << level << " agreement ratio " << (1.0 - difference.mismatch_ratio);
    }
}

// Nominal: level 0 points at the input itself rather than copying it.
TEST(PyramidTest, level_zero_references_input) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
    }
    const cv::Mat image = make_test_image(320, 240, 7U);
    PreprocessFixture fixture;
    ASSERT_TRUE(fixture.run(image, DetectorConfig()));
    const ImageViewU8 level0 = aruco3cuda::detail::level_view(fixture.buffers_, 0);
    EXPECT_EQ(level0.data_, fixture.input_.view().data_);
    EXPECT_EQ(compare(image, download(level0)).max_abs, 0);
}

// Boundary: a level outside the valid range returns an empty view.
TEST(PyramidTest, out_of_range_level_returns_empty_view) {
    PreprocessBuffers buffers;
    buffers.level_count_ = 3;
    EXPECT_EQ(aruco3cuda::detail::level_view(buffers, -1).data_, nullptr);
    EXPECT_EQ(aruco3cuda::detail::level_view(buffers, 3).data_, nullptr);
}

// The GENERIC profile can differ from an ARM HAL by one gray level. This
// historical diagnostic is not a detector quality acceptance criterion.
TEST(SegmentationTest, matches_opencv_resize_within_one_level) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
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

        // A difference beyond one level means the interpolation conventions disagree.
        const bool arm_profile =
                std::string(aruco3cuda::compiled_resize_profile()) == "OPENCV_ARM_HAL";
        EXPECT_LE(difference.max_abs, arm_profile ? 0 : 1)
                << item.width << "x" << item.height << " max difference " << difference.max_abs;
        // Bound the mismatch ratio as well. The measured maximum is 0.372, and a
        // change that raises it markedly means the interpolation has degraded.
        EXPECT_LE(difference.mismatch_ratio, 0.45)
                << item.width << "x" << item.height << " mismatch ratio "
                << difference.mismatch_ratio;
        std::printf("[resize] %dx%d -> %dx%d  max difference %d  mismatch ratio %.4f\n", item.width,
                    item.height, fixture.plan_.segmentation_width_px_,
                    fixture.plan_.segmentation_height_px_, difference.max_abs,
                    difference.mismatch_ratio);
    }
}

// Public CPU compatibility includes backend-specific vector-coordinate rounding,
// overlapping NEON tails, one-axis reductions and independently owned ROI storage.
TEST(SegmentationTest, matching_profile_is_exact_on_random_strided_resize_boundaries) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
    }
#if defined(__aarch64__) || defined(__arm__)
    if (std::string(aruco3cuda::compiled_resize_profile()) != "OPENCV_ARM_HAL") {
        GTEST_SKIP() << "GENERIC does not select the pinned ARM HAL arithmetic";
    }
#else
    if (std::string(aruco3cuda::compiled_resize_profile()) != "GENERIC") {
        GTEST_SKIP() << "ARM HAL profile requires its matching CPU oracle";
    }
#endif
    struct Case {
        int width;
        int height;
        int target_width;
        int target_height;
    };
    std::vector<Case> cases = {
            {641, 479, 213, 159}, {641, 479, 384, 287}, {641, 479, 320, 239},  {640, 480, 320, 240},
            {32, 39, 10, 13},     {47, 51, 15, 17},     {64, 71, 21, 23},      {641, 49, 641, 31},
            {641, 49, 320, 49},   {31, 29, 31, 17},     {31, 29, 19, 29},      {1, 79, 1, 23},
            {79, 1, 23, 1},       {7, 39, 3, 17},       {39, 7, 17, 3},        {17, 19, 7, 8},
            {17, 19, 8, 7},       {9001, 17, 8201, 11}, {18001, 17, 16401, 11}};
    for (int width : {16, 17, 23, 31, 32, 33, 47, 63, 64, 65, 127, 128, 129}) {
        for (int target = std::max(8, width / 3); target < width; ++target) {
            if (target < 40 || target % 8 == 1 || target == width - 1) {
                cases.push_back({width, 47, target, 29});
            }
        }
    }
    cases.push_back({32, 49, 12, 49});
    cases.push_back({47, 49, 15, 16});
    cases.push_back({9001, 49, 2250, 12});
    cv::RNG rng(20261117U);
    for (const Case& item : cases) {
        SCOPED_TRACE(::testing::Message()
                     << item.width << "x" << item.height << " -> " << item.target_width << "x"
                     << item.target_height << " backend " << aruco3cuda::compiled_resize_backend());
        cv::Mat parent(item.height + 4, item.width + 19, CV_8UC1, cv::Scalar(231));
        cv::Mat roi = parent(cv::Rect(7, 2, item.width, item.height));
        rng.fill(roi, cv::RNG::UNIFORM, 0, 256);
        cv::Mat expected;
        cv::resize(roi.clone(), expected, cv::Size(item.target_width, item.target_height), 0.0, 0.0,
                   cv::INTER_LINEAR);
        PreprocessFixture fixture;
        ASSERT_TRUE(fixture.run(roi, DetectorConfig{}, expected.size()));
        const auto& plane = fixture.buffers_.segmentation_;
        const ImageViewU8 view{plane.data_, plane.width_px_, plane.height_px_, plane.pitch_bytes_,
                               MemorySpace::kDevice};
        const Difference difference = compare(expected, download(view));
        EXPECT_EQ(difference.max_abs, 0);
        EXPECT_EQ(difference.mismatch_ratio, 0.0);
        PreprocessFixture direct;
        ASSERT_TRUE(direct.run(roi, DetectorConfig{}, expected.size(), false));
        const auto& direct_plane = direct.buffers_.segmentation_;
        const ImageViewU8 direct_view{direct_plane.data_, direct_plane.width_px_,
                                      direct_plane.height_px_, direct_plane.pitch_bytes_,
                                      MemorySpace::kDevice};
        EXPECT_EQ(compare(download(direct_view), download(view)).max_abs, 0);
    }
}

TEST(SegmentationTest, production_coefficients_are_reserved_only_for_kleidi) {
    if (!has_cuda_device())
        GTEST_SKIP() << "no CUDA device";
    if (std::string(aruco3cuda::compiled_resize_profile()) != "OPENCV_ARM_HAL")
        GTEST_SKIP() << "ARM coefficient path only";
    for (const cv::Size target : {cv::Size(213, 159), cv::Size(383, 287), cv::Size(320, 479)}) {
        PreprocessFixture fixture;
        ASSERT_TRUE(
                fixture.run(make_test_image(641, 479, 81U), DetectorConfig{}, target, true, false));
        const bool kleidi = target.width == 383;
        EXPECT_EQ(fixture.buffers_.resize_coefficients_ready_, kleidi);
        EXPECT_EQ(fixture.buffers_.resize_x_ != nullptr, kleidi);
        EXPECT_EQ(fixture.buffers_.resize_y_ != nullptr, kleidi);
    }
}

struct ResizeCapture {
    cudaStream_t stream_ = nullptr;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t executable_ = nullptr;
    ~ResizeCapture() {
        if (executable_ != nullptr)
            (void)cudaGraphExecDestroy(executable_);
        if (graph_ != nullptr)
            (void)cudaGraphDestroy(graph_);
        if (stream_ != nullptr)
            (void)cudaStreamDestroy(stream_);
    }
};

TEST(SegmentationTest, immutable_coefficients_preserve_changed_frames_and_strides_under_capture) {
    if (!has_cuda_device())
        GTEST_SKIP() << "no CUDA device";
    if (std::string(aruco3cuda::compiled_resize_profile()) != "OPENCV_ARM_HAL")
        GTEST_SKIP() << "ARM coefficient path only";
    const DetectorConfig config;
    for (const cv::Size target : {cv::Size(213, 159), cv::Size(383, 287), cv::Size(320, 479)}) {
        PreprocessFixture fixture;
        cv::Mat initial(479, 641, CV_8UC1, cv::Scalar(17));
        ASSERT_TRUE(fixture.run(initial, config, target));
        ASSERT_TRUE(fixture.buffers_.resize_coefficients_ready_);
        auto& b = fixture.buffers_;
        const std::size_t xbytes = static_cast<std::size_t>(target.width) *
                                   sizeof(aruco3cuda::detail::ResizeAxisCoefficients);
        const std::size_t ybytes = static_cast<std::size_t>(target.height) *
                                   sizeof(aruco3cuda::detail::ResizeAxisCoefficients);
        std::vector<std::uint8_t> original_x(xbytes), original_y(ybytes), after_x(xbytes),
                after_y(ybytes);
        ASSERT_EQ(cudaMemcpy(original_x.data(), b.resize_x_, xbytes, cudaMemcpyDeviceToHost),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpy(original_y.data(), b.resize_y_, ybytes, cudaMemcpyDeviceToHost),
                  cudaSuccess);
        DeviceImage changed_stride;
        ASSERT_TRUE(changed_stride.upload(initial, 97U));
        for (int layout = 0; layout < 2; ++layout) {
            if (layout == 1)
                b.level0_ = changed_stride.view();
            ResizeCapture capture;
            ASSERT_EQ(cudaStreamCreateWithFlags(&capture.stream_, cudaStreamNonBlocking),
                      cudaSuccess);
            ASSERT_EQ(cudaStreamBeginCapture(capture.stream_, cudaStreamCaptureModeThreadLocal),
                      cudaSuccess);
            ASSERT_EQ(aruco3cuda::detail::build_segmentation_async(fixture.plan_, &b, config,
                                                                   capture.stream_),
                      Status::kOk);
            ASSERT_EQ(cudaStreamEndCapture(capture.stream_, &capture.graph_), cudaSuccess);
#if CUDART_VERSION >= 12000
            ASSERT_EQ(cudaGraphInstantiate(&capture.executable_, capture.graph_, 0), cudaSuccess);
#else
            ASSERT_EQ(
                    cudaGraphInstantiate(&capture.executable_, capture.graph_, nullptr, nullptr, 0),
                    cudaSuccess);
#endif
            for (int frame = 0; frame < 4; ++frame) {
                cv::Mat parent(483, 641 + 13 + frame * 11, CV_8UC1,
                               cv::Scalar(frame % 2 ? 17 : 231));
                cv::Mat image = parent(cv::Rect(5, 2, 641, 479));
                cv::RNG rng(static_cast<std::uint64_t>(20261117 + layout * 10 + frame));
                rng.fill(image, cv::RNG::UNIFORM, 0, 256);
                auto* input = const_cast<std::uint8_t*>(b.level0_.data_);
                ASSERT_EQ(cudaMemset2DAsync(input, b.level0_.pitch_bytes_, frame % 2 ? 0x17 : 0xE7,
                                            b.level0_.pitch_bytes_, 479, capture.stream_),
                          cudaSuccess);
                ASSERT_EQ(cudaMemcpy2DAsync(input, b.level0_.pitch_bytes_, image.data, image.step,
                                            641, 479, cudaMemcpyHostToDevice, capture.stream_),
                          cudaSuccess);
                ASSERT_EQ(cudaMemset2DAsync(b.segmentation_.data_, b.segmentation_.pitch_bytes_,
                                            0xA5, b.segmentation_.pitch_bytes_, target.height,
                                            capture.stream_),
                          cudaSuccess);
                ASSERT_EQ(cudaGraphLaunch(capture.executable_, capture.stream_), cudaSuccess);
                ASSERT_EQ(cudaStreamSynchronize(capture.stream_), cudaSuccess);
                cv::Mat expected;
                cv::resize(image.clone(), expected, target, 0, 0, cv::INTER_LINEAR);
                const ImageViewU8 output{b.segmentation_.data_, target.width, target.height,
                                         b.segmentation_.pitch_bytes_, MemorySpace::kDevice};
                EXPECT_EQ(compare(expected, download(output)).max_abs, 0)
                        << target << " layout " << layout << " frame " << frame;
            }
        }
        ASSERT_EQ(cudaMemcpy(after_x.data(), b.resize_x_, xbytes, cudaMemcpyDeviceToHost),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpy(after_y.data(), b.resize_y_, ybytes, cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_EQ(original_x, after_x);
        EXPECT_EQ(original_y, after_y);
    }
}

TEST(SegmentationTest, coefficient_key_changes_fall_back_and_reprepare_outside_capture) {
    if (!has_cuda_device())
        GTEST_SKIP() << "no CUDA device";
    if (std::string(aruco3cuda::compiled_resize_profile()) != "OPENCV_ARM_HAL")
        GTEST_SKIP() << "ARM coefficient path only";
    const DetectorConfig config;
    for (const cv::Size target : {cv::Size(213, 159), cv::Size(383, 287)}) {
        PreprocessFixture fixture;
        ASSERT_TRUE(fixture.run(make_test_image(641, 479, 45U), config, target));
        auto& b = fixture.buffers_;
        ASSERT_TRUE(b.resize_coefficients_ready_);
        for (int change = 0; change < 3; ++change) {
            if (change == 0) {
                b.resize_sve_vector_bytes_ = b.resize_sve_vector_bytes_ == 0 ? 16 : 0;
                b.resize_carotene_x_fma_ = !b.resize_carotene_x_fma_;
            } else if (change == 1) {
                --b.level0_.width_px_;
            } else {
                --b.segmentation_.width_px_;
            }
            const ImageViewU8 output{b.segmentation_.data_, b.segmentation_.width_px_,
                                     b.segmentation_.height_px_, b.segmentation_.pitch_bytes_,
                                     MemorySpace::kDevice};
            ASSERT_EQ(aruco3cuda::detail::build_segmentation_async(fixture.plan_, &b, config,
                                                                   nullptr),
                      Status::kOk);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            const cv::Mat direct = download(output);
            ResizeCapture capture;
            ASSERT_EQ(cudaStreamCreateWithFlags(&capture.stream_, cudaStreamNonBlocking),
                      cudaSuccess);
            ASSERT_EQ(cudaStreamBeginCapture(capture.stream_, cudaStreamCaptureModeThreadLocal),
                      cudaSuccess);
            EXPECT_EQ(aruco3cuda::detail::prepare_segmentation_resize(&b, config, capture.stream_),
                      Status::kInvalidArgument);
            EXPECT_FALSE(b.resize_coefficients_ready_);
            ASSERT_EQ(cudaStreamEndCapture(capture.stream_, &capture.graph_), cudaSuccess);
            ASSERT_EQ(aruco3cuda::detail::prepare_segmentation_resize(&b, config, capture.stream_),
                      Status::kOk);
            EXPECT_TRUE(b.resize_coefficients_ready_);
            ASSERT_EQ(aruco3cuda::detail::build_segmentation_async(fixture.plan_, &b, config,
                                                                   capture.stream_),
                      Status::kOk);
            ASSERT_EQ(cudaStreamSynchronize(capture.stream_), cudaSuccess);
            EXPECT_EQ(compare(direct, download(output)).max_abs, 0);
        }
    }
}

TEST(SegmentationTest, coefficient_capacity_covers_nearby_size_after_quarter_maximum) {
    if (!has_cuda_device())
        GTEST_SKIP() << "no CUDA device";
    if (std::string(aruco3cuda::compiled_resize_profile()) != "OPENCV_ARM_HAL")
        GTEST_SKIP() << "ARM coefficient path only";
    DetectorConfig config;
    config.min_side_length_canonical_img_px_ = 64;
    ScalePlan maximum;
    ScalePlan actual;
    ASSERT_EQ(aruco3cuda::detail::plan_scales(config, 1280, 960, &maximum), Status::kOk);
    ASSERT_EQ(aruco3cuda::detail::plan_scales(config, 1279, 959, &actual), Status::kOk);
    ASSERT_EQ(maximum.segmentation_width_px_, 640);
    ASSERT_EQ(maximum.segmentation_height_px_, 480);
    ASSERT_EQ(actual.segmentation_width_px_, maximum.segmentation_width_px_);
    ASSERT_EQ(actual.segmentation_height_px_, maximum.segmentation_height_px_);
    const auto maximum_bytes = aruco3cuda::detail::preprocess_workspace_bytes(maximum, 1280, 960);
    const auto actual_bytes = aruco3cuda::detail::preprocess_workspace_bytes(actual, 1279, 959);
    ASSERT_LE(actual_bytes, maximum_bytes);
    const cv::Mat image = make_test_image(1279, 959, 346U);
    DeviceImage input;
    ASSERT_TRUE(input.upload(image));
    Workspace workspace;
    ASSERT_EQ(workspace.ensure_capacity(maximum_bytes, MemorySpace::kDevice, nullptr), Status::kOk);
    PreprocessBuffers buffers;
    ASSERT_EQ(aruco3cuda::detail::reserve_preprocess(actual, input.view(), workspace, &buffers),
              Status::kOk);
    ASSERT_NE(buffers.resize_x_, nullptr);
    ASSERT_EQ(aruco3cuda::detail::prepare_segmentation_resize(&buffers, config, nullptr),
              Status::kOk);
    ASSERT_TRUE(buffers.resize_coefficients_ready_);
    ASSERT_EQ(aruco3cuda::detail::build_segmentation_async(actual, &buffers, config, nullptr),
              Status::kOk);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    cv::Mat expected;
    cv::resize(image, expected, {640, 480}, 0, 0, cv::INTER_LINEAR);
    const ImageViewU8 output{buffers.segmentation_.data_, 640, 480,
                             buffers.segmentation_.pitch_bytes_, MemorySpace::kDevice};
    EXPECT_EQ(compare(expected, download(output)).max_abs, 0);
    Workspace too_small;
    ASSERT_EQ(too_small.ensure_capacity(actual_bytes - 256U, MemorySpace::kDevice, nullptr),
              Status::kOk);
    PreprocessBuffers unavailable;
    EXPECT_EQ(aruco3cuda::detail::reserve_preprocess(actual, input.view(), too_small, &unavailable),
              Status::kInvalidConfig);
    EXPECT_FALSE(unavailable.resize_coefficients_ready_);
}

TEST(SegmentationTest, coefficient_prepare_rejects_invalid_storage_without_publishing_ready) {
    const DetectorConfig config;
    EXPECT_EQ(aruco3cuda::detail::prepare_segmentation_resize(nullptr, config, nullptr),
              Status::kInvalidArgument);
    PreprocessBuffers b;
    b.resize_coefficients_ready_ = true;
    EXPECT_EQ(aruco3cuda::detail::prepare_segmentation_resize(&b, config, nullptr),
              Status::kInvalidArgument);
    EXPECT_FALSE(b.resize_coefficients_ready_);
    if (std::string(aruco3cuda::compiled_resize_profile()) == "OPENCV_ARM_HAL") {
        b.level0_.width_px_ = 641;
        b.level0_.height_px_ = 479;
        b.segmentation_.width_px_ = 383;
        b.segmentation_.height_px_ = 287;
        b.resize_x_capacity_ = 383;
        b.resize_y_capacity_ = 287;
        EXPECT_EQ(aruco3cuda::detail::prepare_segmentation_resize(&b, config, nullptr),
                  Status::kInvalidArgument);
        EXPECT_FALSE(b.resize_coefficients_ready_);
    }
}

// Boundary: when the scale factor is 1 the segmentation is a copy of the input.
TEST(SegmentationTest, copies_input_when_scale_is_one) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
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

// Nominal: the result does not depend on the block dimension.
// Per-device tuning varies the block dimension, so the independence of the result
// from that dimension is pinned down here.
TEST(PreprocessTest, result_is_independent_of_block_dim) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
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

// Failure: invalid arguments are rejected.
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

// Failure: carving out the buffers fails when the workspace capacity is too small.
TEST(PreprocessTest, fails_when_workspace_is_too_small) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
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
