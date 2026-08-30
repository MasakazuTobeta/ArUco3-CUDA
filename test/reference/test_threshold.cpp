// SPDX-License-Identifier: Apache-2.0
//
// Cross-checks the adaptive threshold against the OpenCV adaptiveThreshold.
//
// The output of this stage feeds candidate extraction. Any pixel whose black and
// white flip changes the contours, so the agreement rate is measured and pinned
// down here. The test also measures the real effect on binarization of the
// one-level difference left over by the resize in the preceding stage.
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

/// Deterministic test image mixing marker-like rectangles, a gradient, and noise.
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
    // Lay out black rectangles. This creates the conditions under which the
    // binarization boundaries show up as contours.
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
        // Use a pitch that differs from the width, so that a non-contiguous layout
        // is exercised at the same time.
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

/// Runs the binarization and holds on to the resulting buffers.
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

// Nominal: the list of windows matches the OpenCV sweep.
TEST(ThresholdWindowTest, matches_opencv_window_sequence) {
    DetectorConfig config;
    int sizes[aruco3cuda::kMaxAdaptiveThresholdWindows] = {};
    int count = 0;
    ASSERT_EQ(aruco3cuda::detail::threshold_window_sizes(
                      config, sizes, aruco3cuda::kMaxAdaptiveThresholdWindows, &count),
              Status::kOk);
    // The defaults are min 3, max 23, step 10, which yields the three windows 3, 13, and 23.
    ASSERT_EQ(count, 3);
    EXPECT_EQ(sizes[0], 3);
    EXPECT_EQ(sizes[1], 13);
    EXPECT_EQ(sizes[2], 23);
}

// Boundary: even windows are rounded up to odd, the same as OpenCV _threshold does.
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
    // 4, 7, and 10 become 5, 7, and 11.
    ASSERT_EQ(count, 3);
    EXPECT_EQ(sizes[0], 5);
    EXPECT_EQ(sizes[1], 7);
    EXPECT_EQ(sizes[2], 11);
}

// Failure: invalid arguments and insufficient capacity are rejected.
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
    // A capacity smaller than the number of windows in the sweep is rejected.
    EXPECT_EQ(aruco3cuda::detail::threshold_window_sizes(config, sizes, 2, &count),
              Status::kInvalidConfig);
}

// Nominal: for all three windows the result matches the OpenCV binarization exactly.
// This is the completion criterion itself.
TEST(ThresholdTest, matches_opencv_adaptive_threshold) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
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
            std::printf("[threshold] %dx%d window=%2d mismatch ratio %.6f\n", item.width,
                        item.height, window, ratio);
            EXPECT_DOUBLE_EQ(ratio, 0.0)
                    << item.width << "x" << item.height << " window=" << window;
        }
    }
}

// Nominal: the result still matches OpenCV when the constant is changed.
TEST(ThresholdTest, matches_opencv_for_various_constants) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
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
                << "constant " << constant;
    }
}

// Nominal: the result does not depend on the block dimension.
TEST(ThresholdTest, result_is_independent_of_block_dim) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
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

// Failure: invalid arguments are rejected.
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
    // Insufficient capacity.
    EXPECT_EQ(aruco3cuda::detail::reserve_threshold(config, 640, 480, workspace, &buffers),
              Status::kInvalidConfig);
}

// Failure: an input whose size differs from the one reserved for is rejected.
TEST(ThresholdTest, rejects_size_mismatch) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
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

// Measures the real effect on binarization of the one-level difference left over by
// the resize in the preceding stage. The downstream impact of the difference accepted
// at the downscaling stage is confirmed by measurement, not by simulation.
TEST(ThresholdTest, measures_impact_of_resize_difference) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
    }
    const cv::Mat source = make_test_image(1280, 720, 20260827U);
    DetectorConfig config;

    // Take out the segmentation image produced by the CUDA preprocessing.
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

    // Apply the same binarization to both and measure the fraction of pixels whose
    // black and white are swapped.
    for (int window : {3, 13, 23}) {
        cv::Mat from_cuda;
        cv::Mat from_opencv;
        cv::adaptiveThreshold(cuda_segmentation, from_cuda, 255, cv::ADAPTIVE_THRESH_MEAN_C,
                              cv::THRESH_BINARY_INV, window, config.adaptive_thresh_constant_);
        cv::adaptiveThreshold(opencv_segmentation, from_opencv, 255, cv::ADAPTIVE_THRESH_MEAN_C,
                              cv::THRESH_BINARY_INV, window, config.adaptive_thresh_constant_);
        const double ratio = mismatch_ratio(from_opencv, from_cuda);
        std::printf("[resize impact] window=%2d binarization flips %.6f\n", window, ratio);
        // Upper bound based on measurement. A change that raises this markedly means
        // the preprocessing has degraded.
        EXPECT_LT(ratio, 0.01) << "window=" << window;
    }
}

}  // namespace
