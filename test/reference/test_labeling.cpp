// SPDX-License-Identifier: Apache-2.0
//
// 連結成分ラベリングを OpenCV の connectedComponents と突き合わせる。
//
// label の値そのものは実装ごとの採番規則で変わる。判断に必要なのは
// 「どの画素が同じ成分に属するか」であるため、画素の分割が一致するかを
// 検査する。あわせて、実行ごとに同じ label が得られることも固定する。
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

/// 二値化画像を device へ載せ、ラベリングを実行する道具立て。
///
/// pitch を幅と変えて確保し、非連続配置でも正しく読めることを併せて確かめる。
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

    /// 二値化画像を処理し、label 配列を host へ取り出す。成否を返す。
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

/// GPU の label と OpenCV の label が同じ分割を表すか調べる。
///
/// 採番規則の違いを許すため、両方向の写像が矛盾なく作れるかで判定する。
/// 片方向だけでは、複数の成分が 1 つへ潰れた場合を見逃す。
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
            // OpenCV は背景を 0 とし、こちらは kBackgroundLabel とする。
            if ((expected == 0) != (actual == kBackgroundLabel)) {
                return ::testing::AssertionFailure() << "前景と背景が食い違う (" << x << ", " << y
                                                     << ") 基準 " << expected << " 対象 " << actual;
            }
            if (expected == 0) {
                continue;
            }
            const auto forward = reference_to_gpu.find(expected);
            if (forward == reference_to_gpu.end()) {
                reference_to_gpu.emplace(expected, actual);
            } else if (forward->second != actual) {
                return ::testing::AssertionFailure()
                       << "基準の 1 成分が分かれた (" << x << ", " << y << ") 基準 " << expected;
            }
            const auto backward = gpu_to_reference.find(actual);
            if (backward == gpu_to_reference.end()) {
                gpu_to_reference.emplace(actual, expected);
            } else if (backward->second != expected) {
                return ::testing::AssertionFailure()
                       << "基準の別成分が 1 つになった (" << x << ", " << y << ") 対象 " << actual;
            }
        }
    }
    return ::testing::AssertionSuccess();
}

/// OpenCV の 8 連結ラベリングで基準を作る。
cv::Mat reference_labels(const cv::Mat& binary) {
    cv::Mat labels;
    static_cast<void>(cv::connectedComponents(binary, labels, 8, CV_32S));
    return labels;
}

/// GPU の結果が基準と同じ分割であり、label が連番であることを確かめる。
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
        // label は 0 から始まる連番である。統計配列の添字へそのまま使うため。
        EXPECT_GE(label, 0) << name;
        EXPECT_LT(label, run.label_count()) << name;
    }
}

// 境界値: 前景が無い画像では label が 0 個になる。
TEST(LabelingTest, empty_image_has_no_label) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const cv::Mat binary(37, 53, CV_8UC1, cv::Scalar(0));
    expect_matches_reference(binary, "empty");
}

// 境界値: 全画素が前景なら label は 1 個になる。
TEST(LabelingTest, full_image_has_one_label) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const cv::Mat binary(37, 53, CV_8UC1, cv::Scalar(255));
    expect_matches_reference(binary, "full");
}

// 境界値: 1 画素だけの前景。
TEST(LabelingTest, single_pixel_component) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    cv::Mat binary(17, 19, CV_8UC1, cv::Scalar(0));
    binary.at<std::uint8_t>(8, 9) = 255U;
    expect_matches_reference(binary, "single_pixel");
}

// 境界値: 1 行および 1 列の画像。
TEST(LabelingTest, degenerate_image_shapes) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    cv::Mat row(1, 64, CV_8UC1, cv::Scalar(0));
    for (int x = 0; x < 64; x += 3) {
        row.at<std::uint8_t>(0, x) = 255U;
    }
    expect_matches_reference(row, "1 行");

    cv::Mat column(64, 1, CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < 64; y += 3) {
        column.at<std::uint8_t>(y, 0) = 255U;
    }
    expect_matches_reference(column, "1 列");

    const cv::Mat single(1, 1, CV_8UC1, cv::Scalar(255));
    expect_matches_reference(single, "1x1");
}

// 正常系: 離れた矩形はそれぞれ別の成分になる。
TEST(LabelingTest, separate_rectangles_are_separate_components) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    cv::Mat binary(120, 200, CV_8UC1, cv::Scalar(0));
    cv::rectangle(binary, cv::Rect(10, 10, 30, 30), cv::Scalar(255), cv::FILLED);
    cv::rectangle(binary, cv::Rect(80, 20, 40, 25), cv::Scalar(255), cv::FILLED);
    cv::rectangle(binary, cv::Rect(150, 70, 20, 40), cv::Scalar(255), cv::FILLED);
    LabelingRun run;
    ASSERT_TRUE(run.run(binary));
    EXPECT_EQ(run.label_count(), 3);
    expect_matches_reference(binary, "3 矩形");
}

// 正常系: 対角にのみ接する画素は 8 近傍では 1 つの成分になる。
//
// 4 近傍だと別成分になる。OpenCV の findContours は前景を 8 連結として
// 辿るため、ここが分かれると CPU 基準と候補が食い違う。
TEST(LabelingTest, diagonal_chain_is_one_component_with_eight_connectivity) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    cv::Mat binary(40, 40, CV_8UC1, cv::Scalar(0));
    for (int i = 0; i < 40; ++i) {
        binary.at<std::uint8_t>(i, i) = 255U;
    }
    LabelingRun run;
    ASSERT_TRUE(run.run(binary));
    EXPECT_EQ(run.label_count(), 1);
    expect_matches_reference(binary, "対角");
}

// 境界値: 市松模様は 8 近傍では 1 つの成分になる。
TEST(LabelingTest, checkerboard_is_one_component) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    cv::Mat binary(33, 41, CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < binary.rows; ++y) {
        for (int x = 0; x < binary.cols; ++x) {
            binary.at<std::uint8_t>(y, x) = ((x + y) % 2 == 0) ? 255U : 0U;
        }
    }
    expect_matches_reference(binary, "市松");
}

// 境界値: 画像端に掛かる成分も正しく扱える。
TEST(LabelingTest, components_touching_border) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    cv::Mat binary(50, 70, CV_8UC1, cv::Scalar(0));
    cv::rectangle(binary, cv::Rect(0, 0, 10, 10), cv::Scalar(255), cv::FILLED);
    cv::rectangle(binary, cv::Rect(60, 40, 10, 10), cv::Scalar(255), cv::FILLED);
    cv::rectangle(binary, cv::Rect(0, 45, 70, 5), cv::Scalar(255), cv::FILLED);
    expect_matches_reference(binary, "端");
}

// 正常系: 穴を持つ枠は 1 つの成分になる。マーカーの黒枠に対応する。
TEST(LabelingTest, ring_is_one_component) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    cv::Mat binary(80, 80, CV_8UC1, cv::Scalar(0));
    cv::rectangle(binary, cv::Rect(10, 10, 60, 60), cv::Scalar(255), cv::FILLED);
    cv::rectangle(binary, cv::Rect(25, 25, 30, 30), cv::Scalar(0), cv::FILLED);
    LabelingRun run;
    ASSERT_TRUE(run.run(binary));
    // 枠が 1 つ。内側の穴は背景であり、外側の背景とは別の成分になるが、
    // ここでは前景のみを数えるため 1 になる。
    EXPECT_EQ(run.label_count(), 1);
    expect_matches_reference(binary, "枠");
}

// 正常系: 無作為な二値化画像でも分割が一致する。
TEST(LabelingTest, random_images_match_reference) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    // 密度を変える。疎な画像は成分が多く、密な画像は 1 つへ繋がりやすい。
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
        expect_matches_reference(binary, "密度 " + std::to_string(density));
    }
}

// 正常系: 大きめの画像でも一致する。scan が 1 block を超える条件を通す。
TEST(LabelingTest, large_image_matches_reference) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
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

// 正常系: 同じ入力からは同じ label が得られる。
TEST(LabelingTest, labels_are_deterministic) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
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

// 異常系: 引数が不正なら実行しない。
TEST(LabelingTest, rejects_invalid_arguments) {
    Workspace workspace;
    LabelBuffers buffers;
    EXPECT_EQ(aruco3cuda::detail::reserve_labeling(4, 4, workspace, nullptr),
              Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::detail::reserve_labeling(0, 4, workspace, &buffers),
              Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::detail::reserve_labeling(4, -1, workspace, &buffers),
              Status::kInvalidArgument);
    // 容量を確保していない workspace からは切り出せない。
    EXPECT_NE(aruco3cuda::detail::reserve_labeling(4, 4, workspace, &buffers), Status::kOk);

    EXPECT_EQ(aruco3cuda::detail::build_labels_async({}, nullptr, nullptr),
              Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::detail::read_label_count(buffers, nullptr, nullptr),
              Status::kInvalidArgument);

    EXPECT_EQ(aruco3cuda::detail::labeling_workspace_bytes(0, 4), 0U);
    EXPECT_EQ(aruco3cuda::detail::labeling_workspace_bytes(4, 0), 0U);
    EXPECT_EQ(aruco3cuda::detail::labeling_workspace_bytes(-1, 4), 0U);
}

// 異常系: 寸法の食い違う入力は拒否する。
TEST(LabelingTest, rejects_size_mismatch) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
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

/// host 側の統計。GPU の集計と突き合わせるために使う。
struct HostStatistics {
    int min_x_ = 0;
    int min_y_ = 0;
    int max_x_ = 0;
    int max_y_ = 0;
    int pixel_count_ = 0;
    double centroid_x_ = 0.0;
    double centroid_y_ = 0.0;
};

/// GPU が集計した統計を host へ取り出す。
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

/// label 画像から host 側で統計を求める。
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
            // OpenCV の label は 1 起点。統計の添字は 0 起点へ合わせる。
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

/// GPU と host の統計が一致することを確かめる。
///
/// label の採番規則が違うため、外接矩形の左上で対応付けてから比べる。
void expect_statistics_match(const cv::Mat& binary, const std::string& name) {
    StatisticsRun run;
    ASSERT_TRUE(run.run(binary)) << name;
    cv::Mat labels;
    const int expected_count = cv::connectedComponents(binary, labels, 8, CV_32S) - 1;
    ASSERT_EQ(run.label_count(), expected_count) << name;
    const std::vector<HostStatistics> expected = host_statistics(labels, expected_count);

    // 対応付けは画素単位で行う。外接矩形の左上は複数の成分で一致しうるため、
    // それを key にすると別の成分と突き合わせてしまう。
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
        // 重心は float へ落として保持するため、丸め分の差を許す。
        EXPECT_NEAR(actual.centroid_x_, item.centroid_x_, 1e-3) << name;
        EXPECT_NEAR(actual.centroid_y_, item.centroid_y_, 1e-3) << name;
    }
}

// 正常系: 矩形の外接矩形、画素数、重心が host 集計と一致する。
TEST(LabelStatisticsTest, rectangles_match_host_aggregation) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    cv::Mat binary(150, 220, CV_8UC1, cv::Scalar(0));
    cv::rectangle(binary, cv::Rect(10, 12, 31, 17), cv::Scalar(255), cv::FILLED);
    cv::rectangle(binary, cv::Rect(90, 40, 44, 44), cv::Scalar(255), cv::FILLED);
    cv::rectangle(binary, cv::Rect(160, 100, 9, 40), cv::Scalar(255), cv::FILLED);
    expect_statistics_match(binary, "矩形");
}

// 境界値: 1 画素の成分でも外接矩形と重心が定まる。
TEST(LabelStatisticsTest, single_pixel_statistics) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
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

// 境界値: 前景が無ければ統計も 0 件になる。
TEST(LabelStatisticsTest, empty_image_has_no_statistics) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const cv::Mat binary(31, 41, CV_8UC1, cv::Scalar(0));
    StatisticsRun run;
    ASSERT_TRUE(run.run(binary));
    EXPECT_EQ(run.label_count(), 0);
    EXPECT_TRUE(run.statistics().empty());
}

// 正常系: 穴を持つ枠の重心は中心になる。マーカーの黒枠に対応する。
TEST(LabelStatisticsTest, ring_centroid_is_at_center) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
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

// 正常系: 成分数が多い画像でも host 集計と一致する。
TEST(LabelStatisticsTest, many_components_match_host_aggregation) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    // 1 画素飛ばしの配置は label 数が上限に近づく。確保数の妥当性も確かめる。
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
    expect_statistics_match(binary, "1 画素飛ばし");
}

// 正常系: 無作為な画像でも host 集計と一致する。
TEST(LabelStatisticsTest, random_image_matches_host_aggregation) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    std::mt19937 rng(31415U);
    std::uniform_int_distribution<int> draw(0, 99);
    cv::Mat binary(89, 113, CV_8UC1);
    for (int y = 0; y < binary.rows; ++y) {
        for (int x = 0; x < binary.cols; ++x) {
            binary.at<std::uint8_t>(y, x) = (draw(rng) < 30) ? 255U : 0U;
        }
    }
    expect_statistics_match(binary, "無作為");
}

// 異常系: 引数が不正なら実行しない。
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

// 境界値: label 数の上限は 1 画素飛ばしの配置から決まる。
TEST(LabelStatisticsTest, max_label_count_follows_the_bound) {
    EXPECT_EQ(aruco3cuda::detail::max_label_count(1, 1), 1);
    EXPECT_EQ(aruco3cuda::detail::max_label_count(2, 2), 1);
    EXPECT_EQ(aruco3cuda::detail::max_label_count(3, 3), 4);
    EXPECT_EQ(aruco3cuda::detail::max_label_count(427, 240), 214 * 120);
}

}  // namespace
