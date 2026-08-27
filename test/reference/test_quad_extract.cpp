// SPDX-License-Identifier: Apache-2.0
//
// 極点探索による四隅推定を検証する。
//
// 四隅の並びは成分の形で決まるため、CPU 基準と同じ角から始まるとは限らない。
// 判断に必要なのは 4 点の集合が正しい位置にあることなので、対応付けてから
// 位置の差を測る。あわせて、四隅が定まらない成分を無効にできることを固定する。
#include "quad_extract.hpp"

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"
#include "aruco3cuda/workspace.hpp"
#include "labeling.hpp"
#include "preprocess.hpp"

namespace {

using aruco3cuda::MemorySpace;
using aruco3cuda::Status;
using aruco3cuda::Workspace;
using aruco3cuda::detail::kQuadCornerCount;
using aruco3cuda::detail::QuadBuffers;

bool has_cuda_device() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

/// label 1 つ分の四隅。
struct HostQuad {
    bool valid_ = false;
    std::vector<cv::Point2f> corners_;
};

/// 二値化画像から label、統計、四隅までを一度に求める。
class QuadRun {
public:
    QuadRun() = default;
    QuadRun(const QuadRun&) = delete;
    QuadRun& operator=(const QuadRun&) = delete;
    ~QuadRun() {
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
                aruco3cuda::detail::label_stats_workspace_bytes(binary.cols, binary.rows) +
                aruco3cuda::detail::quad_workspace_bytes(binary.cols, binary.rows);
        if (bytes == 0U ||
            this->workspace_.ensure_capacity(bytes, MemorySpace::kDevice, nullptr) != Status::kOk) {
            return false;
        }
        this->workspace_.reset();
        aruco3cuda::detail::LabelBuffers labels;
        aruco3cuda::detail::LabelStatisticsBuffers stats;
        QuadBuffers quads;
        if (aruco3cuda::detail::reserve_labeling(binary.cols, binary.rows, this->workspace_,
                                                 &labels) != Status::kOk ||
            aruco3cuda::detail::reserve_label_stats(binary.cols, binary.rows, this->workspace_,
                                                    &stats) != Status::kOk ||
            aruco3cuda::detail::reserve_quads(binary.cols, binary.rows, this->workspace_, &quads) !=
                    Status::kOk) {
            return false;
        }
        if (aruco3cuda::detail::build_labels_async(plane, &labels, nullptr) != Status::kOk ||
            aruco3cuda::detail::build_label_stats_async(labels, &stats, nullptr) != Status::kOk ||
            aruco3cuda::detail::build_quads_async(labels, stats, &quads, nullptr) != Status::kOk) {
            return false;
        }
        if (aruco3cuda::detail::read_label_count(labels, &this->label_count_, nullptr) !=
            Status::kOk) {
            return false;
        }
        return this->download(quads);
    }

    const std::vector<HostQuad>& quads() const { return this->quads_; }
    int label_count() const { return this->label_count_; }

private:
    bool download(const QuadBuffers& quads) {
        const auto count = static_cast<std::size_t>(this->label_count_);
        this->quads_.assign(count, HostQuad{});
        if (count == 0U) {
            return true;
        }
        // 確保数ではなく label 数だけを読む。label 数を超える範囲は書かれて
        // おらず、読むと未初期化 memory を触ることになる。
        const auto capacity = static_cast<std::size_t>(quads.capacity_);
        std::vector<std::int32_t> corner_x(count * kQuadCornerCount);
        std::vector<std::int32_t> corner_y(count * kQuadCornerCount);
        std::vector<std::uint8_t> valid(count);
        const std::size_t row_bytes = count * sizeof(std::int32_t);
        for (int corner = 0; corner < kQuadCornerCount; ++corner) {
            const std::size_t offset = static_cast<std::size_t>(corner) * capacity;
            const std::size_t destination = static_cast<std::size_t>(corner) * count;
            if (cudaMemcpy(corner_x.data() + destination, quads.corner_x_ + offset, row_bytes,
                           cudaMemcpyDeviceToHost) != cudaSuccess ||
                cudaMemcpy(corner_y.data() + destination, quads.corner_y_ + offset, row_bytes,
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
                return false;
            }
        }
        if (cudaMemcpy(valid.data(), quads.valid_, valid.size(), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
            return false;
        }
        for (std::size_t label = 0; label < count; ++label) {
            HostQuad& quad = this->quads_[label];
            quad.valid_ = valid[label] != 0U;
            quad.corners_.resize(kQuadCornerCount);
            for (int corner = 0; corner < kQuadCornerCount; ++corner) {
                const std::size_t index = (static_cast<std::size_t>(corner) * count) + label;
                quad.corners_[static_cast<std::size_t>(corner)] = cv::Point2f(
                        static_cast<float>(corner_x[index]), static_cast<float>(corner_y[index]));
            }
        }
        return true;
    }

    void* binary_ = nullptr;
    Workspace workspace_;
    std::vector<HostQuad> quads_;
    int label_count_ = 0;
};

/// 2 つの四隅を、開始位置と向きの違いを許して突き合わせた最大距離。
///
/// 極点探索が返す起点の角は成分の形で決まるため、期待値と同じ角から
/// 始まるとは限らない。巡回と反転の 8 通りを試し、最も近い対応を採る。
double quad_distance(const std::vector<cv::Point2f>& expected,
                     const std::vector<cv::Point2f>& actual) {
    double best = std::numeric_limits<double>::max();
    for (int reversed = 0; reversed < 2; ++reversed) {
        std::vector<cv::Point2f> candidate = actual;
        if (reversed == 1) {
            std::reverse(candidate.begin(), candidate.end());
        }
        for (int shift = 0; shift < kQuadCornerCount; ++shift) {
            double worst = 0.0;
            for (int c = 0; c < kQuadCornerCount; ++c) {
                const std::size_t index = static_cast<std::size_t>((c + shift) % kQuadCornerCount);
                const cv::Point2f delta = expected[static_cast<std::size_t>(c)] - candidate[index];
                worst = std::max(worst, std::sqrt(static_cast<double>((delta.x * delta.x) +
                                                                      (delta.y * delta.y))));
            }
            best = std::min(best, worst);
        }
    }
    return best;
}

/// 指定した四隅を塗りつぶした二値化画像を作る。
cv::Mat fill_quad(int width, int height, const std::vector<cv::Point2f>& corners) {
    cv::Mat binary(height, width, CV_8UC1, cv::Scalar(0));
    std::vector<cv::Point> points;
    points.reserve(corners.size());
    for (const cv::Point2f& corner : corners) {
        points.emplace_back(cvRound(corner.x), cvRound(corner.y));
    }
    cv::fillConvexPoly(binary, points, cv::Scalar(255), cv::LINE_8);
    return binary;
}

/// 中心と辺長と回転から正方形の四隅を作る。
std::vector<cv::Point2f> square_corners(double center_x, double center_y, double side,
                                        double degrees) {
    const double radians = degrees * CV_PI / 180.0;
    const double half = side / 2.0;
    const double offsets[4][2] = {{-half, -half}, {half, -half}, {half, half}, {-half, half}};
    std::vector<cv::Point2f> corners;
    corners.reserve(4);
    for (const auto& offset : offsets) {
        const double x =
                center_x + (offset[0] * std::cos(radians)) - (offset[1] * std::sin(radians));
        const double y =
                center_y + (offset[0] * std::sin(radians)) + (offset[1] * std::cos(radians));
        corners.emplace_back(static_cast<float>(x), static_cast<float>(y));
    }
    return corners;
}

// 正常系: 軸に平行な正方形の四隅を推定できる。
TEST(QuadExtractTest, axis_aligned_square) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const std::vector<cv::Point2f> expected = square_corners(120.0, 90.0, 80.0, 0.0);
    const cv::Mat binary = fill_quad(200, 180, expected);
    QuadRun run;
    ASSERT_TRUE(run.run(binary));
    ASSERT_EQ(run.label_count(), 1);
    ASSERT_TRUE(run.quads()[0].valid_);
    EXPECT_LE(quad_distance(expected, run.quads()[0].corners_), 1.5);
}

// 正常系: 回転した正方形でも四隅を推定できる。
TEST(QuadExtractTest, rotated_squares) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const double angles[] = {7.0, 15.0, 30.0, 45.0, 60.0, 80.0};
    double worst = 0.0;
    for (const double angle : angles) {
        const std::vector<cv::Point2f> expected = square_corners(150.0, 130.0, 110.0, angle);
        const cv::Mat binary = fill_quad(300, 260, expected);
        QuadRun run;
        ASSERT_TRUE(run.run(binary)) << angle;
        ASSERT_EQ(run.label_count(), 1) << angle;
        ASSERT_TRUE(run.quads()[0].valid_) << angle;
        const double distance = quad_distance(expected, run.quads()[0].corners_);
        worst = std::max(worst, distance);
        EXPECT_LE(distance, 1.5) << "角度 " << angle;
    }
    std::printf("[quad] 回転正方形の四隅の最大差 %.4f px\n", worst);
}

// 正常系: 射影で歪んだ四角形でも四隅を推定できる。
TEST(QuadExtractTest, perspective_distorted_quad) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const std::vector<cv::Point2f> expected = {
            cv::Point2f(40.0F, 50.0F), cv::Point2f(210.0F, 30.0F), cv::Point2f(240.0F, 190.0F),
            cv::Point2f(60.0F, 170.0F)};
    const cv::Mat binary = fill_quad(300, 240, expected);
    QuadRun run;
    ASSERT_TRUE(run.run(binary));
    ASSERT_EQ(run.label_count(), 1);
    ASSERT_TRUE(run.quads()[0].valid_);
    EXPECT_LE(quad_distance(expected, run.quads()[0].corners_), 1.5);
}

// 正常系: 穴を持つ枠でも外側の四隅を推定できる。マーカーの黒枠に対応する。
TEST(QuadExtractTest, marker_like_ring) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const std::vector<cv::Point2f> expected = square_corners(160.0, 140.0, 120.0, 22.0);
    cv::Mat binary = fill_quad(320, 280, expected);
    // 内側をくり抜き、枠だけを残す。極点探索は成分の全画素を見るため、
    // 穴があっても外側の四隅が得られる。
    const std::vector<cv::Point2f> inner = square_corners(160.0, 140.0, 90.0, 22.0);
    std::vector<cv::Point> inner_points;
    inner_points.reserve(inner.size());
    for (const cv::Point2f& corner : inner) {
        inner_points.emplace_back(cvRound(corner.x), cvRound(corner.y));
    }
    cv::fillConvexPoly(binary, inner_points, cv::Scalar(0), cv::LINE_8);

    QuadRun run;
    ASSERT_TRUE(run.run(binary));
    ASSERT_EQ(run.label_count(), 1);
    ASSERT_TRUE(run.quads()[0].valid_);
    EXPECT_LE(quad_distance(expected, run.quads()[0].corners_), 1.5);
}

// 正常系: 複数のマーカーが同時にあっても label ごとに四隅が得られる。
TEST(QuadExtractTest, multiple_components) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const std::vector<std::vector<cv::Point2f>> shapes = {square_corners(80.0, 80.0, 60.0, 0.0),
                                                          square_corners(240.0, 90.0, 70.0, 33.0),
                                                          square_corners(150.0, 240.0, 90.0, 12.0)};
    cv::Mat binary(320, 340, CV_8UC1, cv::Scalar(0));
    for (const auto& shape : shapes) {
        std::vector<cv::Point> points;
        points.reserve(shape.size());
        for (const cv::Point2f& corner : shape) {
            points.emplace_back(cvRound(corner.x), cvRound(corner.y));
        }
        cv::fillConvexPoly(binary, points, cv::Scalar(255), cv::LINE_8);
    }
    QuadRun run;
    ASSERT_TRUE(run.run(binary));
    ASSERT_EQ(run.label_count(), 3);
    // label の並びと図形の並びは対応しないため、最も近いものと突き合わせる。
    for (const auto& shape : shapes) {
        double best = std::numeric_limits<double>::max();
        for (const HostQuad& quad : run.quads()) {
            if (!quad.valid_) {
                continue;
            }
            best = std::min(best, quad_distance(shape, quad.corners_));
        }
        EXPECT_LE(best, 1.5);
    }
}

// 境界値: 1 画素の成分は四隅が定まらないため無効になる。
TEST(QuadExtractTest, single_pixel_is_invalid) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    cv::Mat binary(40, 40, CV_8UC1, cv::Scalar(0));
    binary.at<std::uint8_t>(20, 20) = 255U;
    QuadRun run;
    ASSERT_TRUE(run.run(binary));
    ASSERT_EQ(run.label_count(), 1);
    EXPECT_FALSE(run.quads()[0].valid_);
}

// 境界値: 直線状の成分は片側に点が無いため無効になる。
TEST(QuadExtractTest, straight_line_is_invalid) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    cv::Mat horizontal(40, 60, CV_8UC1, cv::Scalar(0));
    cv::line(horizontal, cv::Point(5, 20), cv::Point(54, 20), cv::Scalar(255), 1);
    QuadRun horizontal_run;
    ASSERT_TRUE(horizontal_run.run(horizontal));
    ASSERT_EQ(horizontal_run.label_count(), 1);
    EXPECT_FALSE(horizontal_run.quads()[0].valid_);

    cv::Mat diagonal(60, 60, CV_8UC1, cv::Scalar(0));
    for (int i = 5; i < 55; ++i) {
        diagonal.at<std::uint8_t>(i, i) = 255U;
    }
    QuadRun diagonal_run;
    ASSERT_TRUE(diagonal_run.run(diagonal));
    ASSERT_EQ(diagonal_run.label_count(), 1);
    EXPECT_FALSE(diagonal_run.quads()[0].valid_);
}

// 境界値: 前景が無ければ四隅も 0 件になる。
TEST(QuadExtractTest, empty_image_has_no_quad) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const cv::Mat binary(41, 53, CV_8UC1, cv::Scalar(0));
    QuadRun run;
    ASSERT_TRUE(run.run(binary));
    EXPECT_EQ(run.label_count(), 0);
    EXPECT_TRUE(run.quads().empty());
}

// 正常系: 四隅の並びは OpenCV と同じ向きへ揃う。
TEST(QuadExtractTest, corner_order_is_normalized) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const std::vector<cv::Point2f> expected = square_corners(120.0, 110.0, 90.0, 17.0);
    const cv::Mat binary = fill_quad(260, 240, expected);
    QuadRun run;
    ASSERT_TRUE(run.run(binary));
    ASSERT_EQ(run.label_count(), 1);
    const auto& corners = run.quads()[0].corners_;
    const double cross = (static_cast<double>(corners[1].x - corners[0].x) *
                          static_cast<double>(corners[2].y - corners[0].y)) -
                         (static_cast<double>(corners[1].y - corners[0].y) *
                          static_cast<double>(corners[2].x - corners[0].x));
    // OpenCV の _reorderCandidatesCorners と同じ符号になる。
    EXPECT_GE(cross, 0.0);
}

// 正常系: 同じ入力からは同じ四隅が得られる。
TEST(QuadExtractTest, quads_are_deterministic) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const std::vector<cv::Point2f> corners = square_corners(150.0, 130.0, 100.0, 41.0);
    const cv::Mat binary = fill_quad(300, 260, corners);
    QuadRun first;
    QuadRun second;
    ASSERT_TRUE(first.run(binary));
    ASSERT_TRUE(second.run(binary));
    ASSERT_EQ(first.label_count(), second.label_count());
    for (std::size_t i = 0; i < first.quads().size(); ++i) {
        EXPECT_EQ(first.quads()[i].valid_, second.quads()[i].valid_);
        EXPECT_EQ(first.quads()[i].corners_, second.quads()[i].corners_);
    }
}

// 異常系: 引数が不正なら実行しない。
TEST(QuadExtractTest, rejects_invalid_arguments) {
    Workspace workspace;
    QuadBuffers quads;
    EXPECT_EQ(aruco3cuda::detail::reserve_quads(4, 4, workspace, nullptr),
              Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::detail::reserve_quads(0, 4, workspace, &quads), Status::kInvalidArgument);
    EXPECT_NE(aruco3cuda::detail::reserve_quads(4, 4, workspace, &quads), Status::kOk);

    aruco3cuda::detail::LabelBuffers labels;
    aruco3cuda::detail::LabelStatisticsBuffers stats;
    EXPECT_EQ(aruco3cuda::detail::build_quads_async(labels, stats, nullptr, nullptr),
              Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::detail::build_quads_async(labels, stats, &quads, nullptr),
              Status::kInvalidArgument);

    EXPECT_EQ(aruco3cuda::detail::quad_workspace_bytes(0, 4), 0U);
    EXPECT_EQ(aruco3cuda::detail::quad_workspace_bytes(4, 0), 0U);
}

}  // namespace
