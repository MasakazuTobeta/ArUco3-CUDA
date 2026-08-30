// SPDX-License-Identifier: Apache-2.0
//
// Cross-checks connected-component labeling against the OpenCV connectedComponents.
//
// The label values themselves depend on the numbering scheme of each
// implementation. What matters is which pixels belong to the same component, so
// the test checks that the two partitions of the pixels agree. It also pins down
// that every run produces the same labels.
#include "labeling.hpp"

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"
#include "aruco3cuda/workspace.hpp"
#include "preprocess.hpp"

namespace {

using aruco3cuda::MemorySpace;
using aruco3cuda::Status;
using aruco3cuda::Workspace;
using aruco3cuda::detail::kBackgroundLabel;
using aruco3cuda::detail::LabelBuffers;

bool has_cuda_device() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

/// Harness that uploads a binary image to the device and runs the labeling.
///
/// The allocation uses a pitch that differs from the width, which also confirms
/// that a non-contiguous layout is read correctly.
class LabelingRun {
public:
    LabelingRun() = default;
    LabelingRun(const LabelingRun&) = delete;
    LabelingRun& operator=(const LabelingRun&) = delete;
    ~LabelingRun() {
        if (this->binary_ != nullptr) {
            static_cast<void>(cudaFree(this->binary_));
        }
    }

    /// Processes a binary image and copies the label array back to the host.
    /// Returns whether it succeeded.
    bool run(const cv::Mat& binary) {
        const std::size_t pitch = static_cast<std::size_t>(binary.cols) + 32U;
        if (cudaMalloc(&this->binary_, pitch * static_cast<std::size_t>(binary.rows)) !=
            cudaSuccess) {
            return false;
        }
        if (cudaMemcpy2D(this->binary_, pitch, binary.data, static_cast<std::size_t>(binary.step),
                         static_cast<std::size_t>(binary.cols),
                         static_cast<std::size_t>(binary.rows),
                         cudaMemcpyHostToDevice) != cudaSuccess) {
            return false;
        }
        aruco3cuda::detail::ImagePlaneU8 plane;
        plane.data_ = static_cast<std::uint8_t*>(this->binary_);
        plane.width_px_ = binary.cols;
        plane.height_px_ = binary.rows;
        plane.pitch_bytes_ = pitch;

        const std::size_t bytes =
                aruco3cuda::detail::labeling_workspace_bytes(binary.cols, binary.rows);
        if (bytes == 0U) {
            return false;
        }
        if (this->workspace_.ensure_capacity(bytes, MemorySpace::kDevice, nullptr) != Status::kOk) {
            return false;
        }
        this->workspace_.reset();
        if (aruco3cuda::detail::reserve_labeling(binary.cols, binary.rows, this->workspace_,
                                                 &this->buffers_) != Status::kOk) {
            return false;
        }
        if (aruco3cuda::detail::build_labels_async(plane, &this->buffers_, nullptr) !=
            Status::kOk) {
            return false;
        }
        if (aruco3cuda::detail::read_label_count(this->buffers_, &this->label_count_, nullptr) !=
            Status::kOk) {
            return false;
        }
        this->labels_.resize(static_cast<std::size_t>(binary.cols) *
                             static_cast<std::size_t>(binary.rows));
        return cudaMemcpy(this->labels_.data(), this->buffers_.labels_,
                          this->labels_.size() * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost) == cudaSuccess;
    }

    const std::vector<std::int32_t>& labels() const { return this->labels_; }
    int label_count() const { return this->label_count_; }

private:
    void* binary_ = nullptr;
    Workspace workspace_;
    LabelBuffers buffers_;
    std::vector<std::int32_t> labels_;
    int label_count_ = 0;
};

/// Checks whether the GPU labels and the OpenCV labels describe the same partition.
///
/// Differences in the numbering scheme are allowed, so the decision rests on
/// whether a consistent mapping can be built in both directions. One direction
/// alone would miss the case where several components collapse into one.
::testing::AssertionResult same_partition(const cv::Mat& reference,
                                          const std::vector<std::int32_t>& labels) {
    std::map<int, std::int32_t> reference_to_gpu;
    std::map<std::int32_t, int> gpu_to_reference;
    for (int y = 0; y < reference.rows; ++y) {
        const auto* row = reference.ptr<std::int32_t>(y);
        for (int x = 0; x < reference.cols; ++x) {
            const std::size_t index =
                    (static_cast<std::size_t>(y) * static_cast<std::size_t>(reference.cols)) +
                    static_cast<std::size_t>(x);
            const int expected = row[x];
            const std::int32_t actual = labels[index];
            // OpenCV marks the background with 0; this implementation uses kBackgroundLabel.
            if ((expected == 0) != (actual == kBackgroundLabel)) {
                return ::testing::AssertionFailure()
                       << "foreground and background disagree at (" << x << ", " << y
                       << ") baseline " << expected << " target " << actual;
            }
            if (expected == 0) {
                continue;
            }
            const auto forward = reference_to_gpu.find(expected);
            if (forward == reference_to_gpu.end()) {
                reference_to_gpu.emplace(expected, actual);
            } else if (forward->second != actual) {
                return ::testing::AssertionFailure() << "one baseline component was split at (" << x
                                                     << ", " << y << ") baseline " << expected;
            }
            const auto backward = gpu_to_reference.find(actual);
            if (backward == gpu_to_reference.end()) {
                gpu_to_reference.emplace(actual, expected);
            } else if (backward->second != expected) {
                return ::testing::AssertionFailure() << "two baseline components were merged at ("
                                                     << x << ", " << y << ") target " << actual;
            }
        }
    }
    return ::testing::AssertionSuccess();
}

/// Builds the baseline with the OpenCV 8-connected labeling.
cv::Mat reference_labels(const cv::Mat& binary) {
    cv::Mat labels;
    static_cast<void>(cv::connectedComponents(binary, labels, 8, CV_32S));
    return labels;
}

/// Confirms that the GPU result is the same partition as the baseline and that the
/// labels are consecutive.
void expect_matches_reference(const cv::Mat& binary, const std::string& name) {
    LabelingRun run;
    ASSERT_TRUE(run.run(binary)) << name;
    const cv::Mat expected = reference_labels(binary);
    double minimum = 0.0;
    double maximum = 0.0;
    cv::minMaxLoc(expected, &minimum, &maximum);
    EXPECT_EQ(run.label_count(), static_cast<int>(maximum)) << name;
    EXPECT_TRUE(same_partition(expected, run.labels())) << name;
    for (const std::int32_t label : run.labels()) {
        if (label == kBackgroundLabel) {
            continue;
        }
        // The labels are consecutive starting at 0, because they are used directly as
        // indices into the statistics arrays.
        EXPECT_GE(label, 0) << name;
        EXPECT_LT(label, run.label_count()) << name;
    }
}

// Boundary: an image with no foreground yields zero labels.
TEST(LabelingTest, empty_image_has_no_label) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
    }
    const cv::Mat binary(37, 53, CV_8UC1, cv::Scalar(0));
    expect_matches_reference(binary, "empty");
}

// Boundary: an image whose pixels are all foreground yields one label.
TEST(LabelingTest, full_image_has_one_label) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
    }
    const cv::Mat binary(37, 53, CV_8UC1, cv::Scalar(255));
    expect_matches_reference(binary, "full");
}

// Boundary: a foreground of a single pixel.
TEST(LabelingTest, single_pixel_component) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
    }
    cv::Mat binary(17, 19, CV_8UC1, cv::Scalar(0));
    binary.at<std::uint8_t>(8, 9) = 255U;
    expect_matches_reference(binary, "single_pixel");
}

// Boundary: images of a single row and of a single column.
TEST(LabelingTest, degenerate_image_shapes) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
    }
    cv::Mat row(1, 64, CV_8UC1, cv::Scalar(0));
    for (int x = 0; x < 64; x += 3) {
        row.at<std::uint8_t>(0, x) = 255U;
    }
    expect_matches_reference(row, "single row");

    cv::Mat column(64, 1, CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < 64; y += 3) {
        column.at<std::uint8_t>(y, 0) = 255U;
    }
    expect_matches_reference(column, "single column");

    const cv::Mat single(1, 1, CV_8UC1, cv::Scalar(255));
    expect_matches_reference(single, "1x1");
}

// Nominal: rectangles that do not touch each become their own component.
TEST(LabelingTest, separate_rectangles_are_separate_components) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
    }
    cv::Mat binary(120, 200, CV_8UC1, cv::Scalar(0));
    cv::rectangle(binary, cv::Rect(10, 10, 30, 30), cv::Scalar(255), cv::FILLED);
    cv::rectangle(binary, cv::Rect(80, 20, 40, 25), cv::Scalar(255), cv::FILLED);
    cv::rectangle(binary, cv::Rect(150, 70, 20, 40), cv::Scalar(255), cv::FILLED);
    LabelingRun run;
    ASSERT_TRUE(run.run(binary));
    EXPECT_EQ(run.label_count(), 3);
    expect_matches_reference(binary, "three rectangles");
}

// Nominal: pixels that touch only diagonally form a single component under
// 8-connectivity.
//
// Under 4-connectivity they would be separate components. The OpenCV
// findContours walks the foreground as 8-connected, so splitting them here would
// make the candidates disagree with the CPU reference.
TEST(LabelingTest, diagonal_chain_is_one_component_with_eight_connectivity) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
    }
    cv::Mat binary(40, 40, CV_8UC1, cv::Scalar(0));
    for (int i = 0; i < 40; ++i) {
        binary.at<std::uint8_t>(i, i) = 255U;
    }
    LabelingRun run;
    ASSERT_TRUE(run.run(binary));
    EXPECT_EQ(run.label_count(), 1);
    expect_matches_reference(binary, "diagonal");
}

// Boundary: a checkerboard forms a single component under 8-connectivity.
TEST(LabelingTest, checkerboard_is_one_component) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
    }
    cv::Mat binary(33, 41, CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < binary.rows; ++y) {
        for (int x = 0; x < binary.cols; ++x) {
            binary.at<std::uint8_t>(y, x) = ((x + y) % 2 == 0) ? 255U : 0U;
        }
    }
    expect_matches_reference(binary, "checkerboard");
}

// Boundary: components that run into the image border are handled correctly too.
TEST(LabelingTest, components_touching_border) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
    }
    cv::Mat binary(50, 70, CV_8UC1, cv::Scalar(0));
    cv::rectangle(binary, cv::Rect(0, 0, 10, 10), cv::Scalar(255), cv::FILLED);
    cv::rectangle(binary, cv::Rect(60, 40, 10, 10), cv::Scalar(255), cv::FILLED);
    cv::rectangle(binary, cv::Rect(0, 45, 70, 5), cv::Scalar(255), cv::FILLED);
    expect_matches_reference(binary, "border");
}

// Nominal: a ring with a hole forms a single component. This is the case of the
// black border of a marker.
TEST(LabelingTest, ring_is_one_component) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
    }
    cv::Mat binary(80, 80, CV_8UC1, cv::Scalar(0));
    cv::rectangle(binary, cv::Rect(10, 10, 60, 60), cv::Scalar(255), cv::FILLED);
    cv::rectangle(binary, cv::Rect(25, 25, 30, 30), cv::Scalar(0), cv::FILLED);
    LabelingRun run;
    ASSERT_TRUE(run.run(binary));
    // One ring. The hole inside is background and forms a component distinct from the
    // outer background, but only the foreground is counted here, so the count is 1.
    EXPECT_EQ(run.label_count(), 1);
    expect_matches_reference(binary, "ring");
}

// Nominal: the partitions agree on random binary images as well.
TEST(LabelingTest, random_images_match_reference) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
    }
    // Vary the density. A sparse image has many components, while a dense one tends
    // to connect into a single component.
    const int densities[] = {5, 25, 50, 75, 95};
    for (const int density : densities) {
        std::mt19937 rng(20260827U + static_cast<unsigned int>(density));
        std::uniform_int_distribution<int> draw(0, 99);
        cv::Mat binary(97, 131, CV_8UC1);
        for (int y = 0; y < binary.rows; ++y) {
            for (int x = 0; x < binary.cols; ++x) {
                binary.at<std::uint8_t>(y, x) = (draw(rng) < density) ? 255U : 0U;
            }
        }
        expect_matches_reference(binary, "density " + std::to_string(density));
    }
}

// Nominal: larger images agree as well. This exercises the case where the scan
// spans more than one block.
TEST(LabelingTest, large_image_matches_reference) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
    }
    cv::Mat binary(1080, 1920, CV_8UC1, cv::Scalar(0));
    std::mt19937 rng(20260827U);
    std::uniform_int_distribution<int> x_draw(0, 1900);
    std::uniform_int_distribution<int> y_draw(0, 1060);
    for (int i = 0; i < 400; ++i) {
        cv::rectangle(binary, cv::Rect(x_draw(rng), y_draw(rng), 15, 15), cv::Scalar(255),
                      cv::FILLED);
    }
    expect_matches_reference(binary, "1920x1080");
}

// Nominal: the same input yields the same labels.
TEST(LabelingTest, labels_are_deterministic) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
    }
    std::mt19937 rng(4242U);
    std::uniform_int_distribution<int> draw(0, 99);
    cv::Mat binary(211, 307, CV_8UC1);
    for (int y = 0; y < binary.rows; ++y) {
        for (int x = 0; x < binary.cols; ++x) {
            binary.at<std::uint8_t>(y, x) = (draw(rng) < 40) ? 255U : 0U;
        }
    }
    LabelingRun first;
    LabelingRun second;
    ASSERT_TRUE(first.run(binary));
    ASSERT_TRUE(second.run(binary));
    EXPECT_EQ(first.label_count(), second.label_count());
    EXPECT_EQ(first.labels(), second.labels());
}

// Failure: nothing runs when the arguments are invalid.
TEST(LabelingTest, rejects_invalid_arguments) {
    Workspace workspace;
    LabelBuffers buffers;
    EXPECT_EQ(aruco3cuda::detail::reserve_labeling(4, 4, workspace, nullptr),
              Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::detail::reserve_labeling(0, 4, workspace, &buffers),
              Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::detail::reserve_labeling(4, -1, workspace, &buffers),
              Status::kInvalidArgument);
    // Nothing can be carved out of a workspace whose capacity was never reserved.
    EXPECT_NE(aruco3cuda::detail::reserve_labeling(4, 4, workspace, &buffers), Status::kOk);

    EXPECT_EQ(aruco3cuda::detail::build_labels_async({}, nullptr, nullptr),
              Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::detail::read_label_count(buffers, nullptr, nullptr),
              Status::kInvalidArgument);

    EXPECT_EQ(aruco3cuda::detail::labeling_workspace_bytes(0, 4), 0U);
    EXPECT_EQ(aruco3cuda::detail::labeling_workspace_bytes(4, 0), 0U);
    EXPECT_EQ(aruco3cuda::detail::labeling_workspace_bytes(-1, 4), 0U);
}

// Failure: an input whose size disagrees is rejected.
TEST(LabelingTest, rejects_size_mismatch) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
    }
    Workspace workspace;
    ASSERT_EQ(workspace.ensure_capacity(aruco3cuda::detail::labeling_workspace_bytes(16, 16),
                                        MemorySpace::kDevice, nullptr),
              Status::kOk);
    LabelBuffers buffers;
    ASSERT_EQ(aruco3cuda::detail::reserve_labeling(16, 16, workspace, &buffers), Status::kOk);

    aruco3cuda::detail::ImagePlaneU8 plane;
    std::uint8_t dummy = 0U;
    plane.data_ = &dummy;
    plane.width_px_ = 8;
    plane.height_px_ = 16;
    plane.pitch_bytes_ = 8U;
    EXPECT_EQ(aruco3cuda::detail::build_labels_async(plane, &buffers, nullptr),
              Status::kInvalidArgument);
}

/// Host-side statistics, used to cross-check the aggregation done on the GPU.
struct HostStatistics {
    int min_x_ = 0;
    int min_y_ = 0;
    int max_x_ = 0;
    int max_y_ = 0;
    int pixel_count_ = 0;
    double centroid_x_ = 0.0;
    double centroid_y_ = 0.0;
};

/// Copies the statistics aggregated on the GPU back to the host.
class StatisticsRun {
public:
    StatisticsRun() = default;
    StatisticsRun(const StatisticsRun&) = delete;
    StatisticsRun& operator=(const StatisticsRun&) = delete;
    ~StatisticsRun() {
        if (this->binary_ != nullptr) {
            static_cast<void>(cudaFree(this->binary_));
        }
    }

    bool run(const cv::Mat& binary) {
        const std::size_t pitch = static_cast<std::size_t>(binary.cols) + 32U;
        if (cudaMalloc(&this->binary_, pitch * static_cast<std::size_t>(binary.rows)) !=
            cudaSuccess) {
            return false;
        }
        if (cudaMemcpy2D(this->binary_, pitch, binary.data, static_cast<std::size_t>(binary.step),
                         static_cast<std::size_t>(binary.cols),
                         static_cast<std::size_t>(binary.rows),
                         cudaMemcpyHostToDevice) != cudaSuccess) {
            return false;
        }
        aruco3cuda::detail::ImagePlaneU8 plane;
        plane.data_ = static_cast<std::uint8_t*>(this->binary_);
        plane.width_px_ = binary.cols;
        plane.height_px_ = binary.rows;
        plane.pitch_bytes_ = pitch;

        const std::size_t bytes =
                aruco3cuda::detail::labeling_workspace_bytes(binary.cols, binary.rows) +
                aruco3cuda::detail::label_stats_workspace_bytes(binary.cols, binary.rows);
        if (bytes == 0U ||
            this->workspace_.ensure_capacity(bytes, MemorySpace::kDevice, nullptr) != Status::kOk) {
            return false;
        }
        this->workspace_.reset();
        aruco3cuda::detail::LabelStatisticsBuffers stats;
        if (aruco3cuda::detail::reserve_labeling(binary.cols, binary.rows, this->workspace_,
                                                 &this->buffers_) != Status::kOk ||
            aruco3cuda::detail::reserve_label_stats(binary.cols, binary.rows, this->workspace_,
                                                    &stats) != Status::kOk) {
            return false;
        }
        if (aruco3cuda::detail::build_labels_async(plane, &this->buffers_, nullptr) !=
                    Status::kOk ||
            aruco3cuda::detail::build_label_stats_async(this->buffers_, &stats, nullptr) !=
                    Status::kOk) {
            return false;
        }
        if (aruco3cuda::detail::read_label_count(this->buffers_, &this->label_count_, nullptr) !=
            Status::kOk) {
            return false;
        }
        this->labels_.resize(static_cast<std::size_t>(binary.cols) *
                             static_cast<std::size_t>(binary.rows));
        if (cudaMemcpy(this->labels_.data(), this->buffers_.labels_,
                       this->labels_.size() * sizeof(std::int32_t),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            return false;
        }
        return this->download(stats);
    }

    const std::vector<HostStatistics>& statistics() const { return this->statistics_; }
    const std::vector<std::int32_t>& labels() const { return this->labels_; }
    int label_count() const { return this->label_count_; }

private:
    bool download(const aruco3cuda::detail::LabelStatisticsBuffers& stats) {
        const auto count = static_cast<std::size_t>(this->label_count_);
        this->statistics_.assign(count, HostStatistics{});
        if (count == 0U) {
            return true;
        }
        std::vector<std::int32_t> min_x(count);
        std::vector<std::int32_t> min_y(count);
        std::vector<std::int32_t> max_x(count);
        std::vector<std::int32_t> max_y(count);
        std::vector<std::int32_t> pixels(count);
        std::vector<float> centroid_x(count);
        std::vector<float> centroid_y(count);
        const std::size_t int_bytes = count * sizeof(std::int32_t);
        const std::size_t float_bytes = count * sizeof(float);
        if (cudaMemcpy(min_x.data(), stats.min_x_, int_bytes, cudaMemcpyDeviceToHost) !=
                    cudaSuccess ||
            cudaMemcpy(min_y.data(), stats.min_y_, int_bytes, cudaMemcpyDeviceToHost) !=
                    cudaSuccess ||
            cudaMemcpy(max_x.data(), stats.max_x_, int_bytes, cudaMemcpyDeviceToHost) !=
                    cudaSuccess ||
            cudaMemcpy(max_y.data(), stats.max_y_, int_bytes, cudaMemcpyDeviceToHost) !=
                    cudaSuccess ||
            cudaMemcpy(pixels.data(), stats.pixel_count_, int_bytes, cudaMemcpyDeviceToHost) !=
                    cudaSuccess ||
            cudaMemcpy(centroid_x.data(), stats.centroid_x_, float_bytes, cudaMemcpyDeviceToHost) !=
                    cudaSuccess ||
            cudaMemcpy(centroid_y.data(), stats.centroid_y_, float_bytes, cudaMemcpyDeviceToHost) !=
                    cudaSuccess) {
            return false;
        }
        for (std::size_t i = 0; i < count; ++i) {
            this->statistics_[i] = HostStatistics{min_x[i],
                                                  min_y[i],
                                                  max_x[i],
                                                  max_y[i],
                                                  pixels[i],
                                                  static_cast<double>(centroid_x[i]),
                                                  static_cast<double>(centroid_y[i])};
        }
        return true;
    }

    void* binary_ = nullptr;
    Workspace workspace_;
    LabelBuffers buffers_;
    std::vector<HostStatistics> statistics_;
    std::vector<std::int32_t> labels_;
    int label_count_ = 0;
};

/// Computes the statistics on the host from a label image.
std::vector<HostStatistics> host_statistics(const cv::Mat& labels, int label_count) {
    std::vector<HostStatistics> result(static_cast<std::size_t>(label_count));
    std::vector<double> sum_x(static_cast<std::size_t>(label_count), 0.0);
    std::vector<double> sum_y(static_cast<std::size_t>(label_count), 0.0);
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i].min_x_ = labels.cols;
        result[i].min_y_ = labels.rows;
        result[i].max_x_ = -1;
        result[i].max_y_ = -1;
    }
    for (int y = 0; y < labels.rows; ++y) {
        const auto* row = labels.ptr<std::int32_t>(y);
        for (int x = 0; x < labels.cols; ++x) {
            if (row[x] == 0) {
                continue;
            }
            // OpenCV labels start at 1; the statistics indices are shifted to start at 0.
            const auto index = static_cast<std::size_t>(row[x] - 1);
            HostStatistics& item = result[index];
            item.min_x_ = std::min(item.min_x_, x);
            item.min_y_ = std::min(item.min_y_, y);
            item.max_x_ = std::max(item.max_x_, x);
            item.max_y_ = std::max(item.max_y_, y);
            item.pixel_count_ += 1;
            sum_x[index] += x;
            sum_y[index] += y;
        }
    }
    for (std::size_t i = 0; i < result.size(); ++i) {
        if (result[i].pixel_count_ > 0) {
            result[i].centroid_x_ = sum_x[i] / result[i].pixel_count_;
            result[i].centroid_y_ = sum_y[i] / result[i].pixel_count_;
        }
    }
    return result;
}

/// Confirms that the GPU statistics agree with the host statistics.
///
/// The two label numbering schemes differ, so the components are matched up first
/// and compared afterwards.
void expect_statistics_match(const cv::Mat& binary, const std::string& name) {
    StatisticsRun run;
    ASSERT_TRUE(run.run(binary)) << name;
    cv::Mat labels;
    const int expected_count = cv::connectedComponents(binary, labels, 8, CV_32S) - 1;
    ASSERT_EQ(run.label_count(), expected_count) << name;
    const std::vector<HostStatistics> expected = host_statistics(labels, expected_count);

    // The matching is done per pixel. The top-left corner of the bounding box can
    // coincide for several components, so keying on it would pair up the wrong ones.
    std::map<int, std::int32_t> reference_to_gpu;
    for (int y = 0; y < labels.rows; ++y) {
        const auto* row = labels.ptr<std::int32_t>(y);
        for (int x = 0; x < labels.cols; ++x) {
            if (row[x] == 0) {
                continue;
            }
            const std::size_t index =
                    (static_cast<std::size_t>(y) * static_cast<std::size_t>(labels.cols)) +
                    static_cast<std::size_t>(x);
            reference_to_gpu.emplace(row[x], run.labels()[index]);
        }
    }
    ASSERT_EQ(reference_to_gpu.size(), static_cast<std::size_t>(expected_count)) << name;

    for (const auto& entry : reference_to_gpu) {
        const HostStatistics& item = expected[static_cast<std::size_t>(entry.first - 1)];
        const HostStatistics& actual = run.statistics()[static_cast<std::size_t>(entry.second)];
        EXPECT_EQ(actual.min_x_, item.min_x_) << name << " label " << entry.first;
        EXPECT_EQ(actual.min_y_, item.min_y_) << name << " label " << entry.first;
        EXPECT_EQ(actual.max_x_, item.max_x_) << name << " label " << entry.first;
        EXPECT_EQ(actual.max_y_, item.max_y_) << name << " label " << entry.first;
        EXPECT_EQ(actual.pixel_count_, item.pixel_count_) << name << " label " << entry.first;
        // The centroid is kept as a float, so a difference of one rounding step is allowed.
        EXPECT_NEAR(actual.centroid_x_, item.centroid_x_, 1e-3) << name;
        EXPECT_NEAR(actual.centroid_y_, item.centroid_y_, 1e-3) << name;
    }
}

// Nominal: the bounding box, pixel count, and centroid of rectangles agree with the
// host aggregation.
TEST(LabelStatisticsTest, rectangles_match_host_aggregation) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
    }
    cv::Mat binary(150, 220, CV_8UC1, cv::Scalar(0));
    cv::rectangle(binary, cv::Rect(10, 12, 31, 17), cv::Scalar(255), cv::FILLED);
    cv::rectangle(binary, cv::Rect(90, 40, 44, 44), cv::Scalar(255), cv::FILLED);
    cv::rectangle(binary, cv::Rect(160, 100, 9, 40), cv::Scalar(255), cv::FILLED);
    expect_statistics_match(binary, "rectangles");
}

// Boundary: even a single-pixel component has a well-defined bounding box and centroid.
TEST(LabelStatisticsTest, single_pixel_statistics) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
    }
    cv::Mat binary(23, 29, CV_8UC1, cv::Scalar(0));
    binary.at<std::uint8_t>(7, 11) = 255U;
    StatisticsRun run;
    ASSERT_TRUE(run.run(binary));
    ASSERT_EQ(run.label_count(), 1);
    const auto& item = run.statistics()[0];
    EXPECT_EQ(item.min_x_, 11);
    EXPECT_EQ(item.max_x_, 11);
    EXPECT_EQ(item.min_y_, 7);
    EXPECT_EQ(item.max_y_, 7);
    EXPECT_EQ(item.pixel_count_, 1);
    EXPECT_NEAR(item.centroid_x_, 11.0, 1e-6);
    EXPECT_NEAR(item.centroid_y_, 7.0, 1e-6);
}

// Boundary: with no foreground there are no statistics either.
TEST(LabelStatisticsTest, empty_image_has_no_statistics) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
    }
    const cv::Mat binary(31, 41, CV_8UC1, cv::Scalar(0));
    StatisticsRun run;
    ASSERT_TRUE(run.run(binary));
    EXPECT_EQ(run.label_count(), 0);
    EXPECT_TRUE(run.statistics().empty());
}

// Nominal: the centroid of a ring with a hole sits at its center. This is the case
// of the black border of a marker.
TEST(LabelStatisticsTest, ring_centroid_is_at_center) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
    }
    cv::Mat binary(101, 101, CV_8UC1, cv::Scalar(0));
    cv::rectangle(binary, cv::Rect(20, 20, 61, 61), cv::Scalar(255), cv::FILLED);
    cv::rectangle(binary, cv::Rect(35, 35, 31, 31), cv::Scalar(0), cv::FILLED);
    StatisticsRun run;
    ASSERT_TRUE(run.run(binary));
    ASSERT_EQ(run.label_count(), 1);
    const auto& item = run.statistics()[0];
    EXPECT_EQ(item.min_x_, 20);
    EXPECT_EQ(item.min_y_, 20);
    EXPECT_EQ(item.max_x_, 80);
    EXPECT_EQ(item.max_y_, 80);
    EXPECT_EQ(item.pixel_count_, (61 * 61) - (31 * 31));
    EXPECT_NEAR(item.centroid_x_, 50.0, 1e-3);
    EXPECT_NEAR(item.centroid_y_, 50.0, 1e-3);
}

// Nominal: images with many components agree with the host aggregation as well.
TEST(LabelStatisticsTest, many_components_match_host_aggregation) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
    }
    // A layout with every other pixel set drives the label count close to its upper
    // bound, which also confirms that the reserved capacity is adequate.
    cv::Mat binary(81, 121, CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < binary.rows; y += 2) {
        for (int x = 0; x < binary.cols; x += 2) {
            binary.at<std::uint8_t>(y, x) = 255U;
        }
    }
    StatisticsRun run;
    ASSERT_TRUE(run.run(binary));
    EXPECT_EQ(run.label_count(), 41 * 61);
    EXPECT_EQ(run.label_count(), aruco3cuda::detail::max_label_count(binary.cols, binary.rows));
    expect_statistics_match(binary, "every other pixel");
}

// Nominal: random images agree with the host aggregation as well.
TEST(LabelStatisticsTest, random_image_matches_host_aggregation) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "no CUDA device available; skipping";
    }
    std::mt19937 rng(31415U);
    std::uniform_int_distribution<int> draw(0, 99);
    cv::Mat binary(89, 113, CV_8UC1);
    for (int y = 0; y < binary.rows; ++y) {
        for (int x = 0; x < binary.cols; ++x) {
            binary.at<std::uint8_t>(y, x) = (draw(rng) < 30) ? 255U : 0U;
        }
    }
    expect_statistics_match(binary, "random");
}

// Failure: nothing runs when the arguments are invalid.
TEST(LabelStatisticsTest, rejects_invalid_arguments) {
    Workspace workspace;
    aruco3cuda::detail::LabelStatisticsBuffers stats;
    EXPECT_EQ(aruco3cuda::detail::reserve_label_stats(4, 4, workspace, nullptr),
              Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::detail::reserve_label_stats(0, 4, workspace, &stats),
              Status::kInvalidArgument);
    EXPECT_NE(aruco3cuda::detail::reserve_label_stats(4, 4, workspace, &stats), Status::kOk);

    LabelBuffers labels;
    EXPECT_EQ(aruco3cuda::detail::build_label_stats_async(labels, nullptr, nullptr),
              Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::detail::build_label_stats_async(labels, &stats, nullptr),
              Status::kInvalidArgument);

    EXPECT_EQ(aruco3cuda::detail::max_label_count(0, 4), 0);
    EXPECT_EQ(aruco3cuda::detail::max_label_count(4, 0), 0);
    EXPECT_EQ(aruco3cuda::detail::label_stats_workspace_bytes(0, 4), 0U);
}

// Boundary: the upper bound on the label count follows from the layout with every
// other pixel set.
TEST(LabelStatisticsTest, max_label_count_follows_the_bound) {
    EXPECT_EQ(aruco3cuda::detail::max_label_count(1, 1), 1);
    EXPECT_EQ(aruco3cuda::detail::max_label_count(2, 2), 1);
    EXPECT_EQ(aruco3cuda::detail::max_label_count(3, 3), 4);
    EXPECT_EQ(aruco3cuda::detail::max_label_count(427, 240), 214 * 120);
}

}  // namespace
