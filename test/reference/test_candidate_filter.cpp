// SPDX-License-Identifier: Apache-2.0
//
// 候補の篩、詰め込み、上限超過を検証する。
//
// 篩は CPU 経路の判定へ写像したものであり、周長と四角形らしさの測り方が
// 異なる。この違いが結果へどう出るかを、通す形と落とす形の両方で固定する。
// 上限超過は無言で捨てず Status で示すことも合わせて確かめる。
#include "candidate_filter.hpp"

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"
#include "aruco3cuda/workspace.hpp"
#include "labeling.hpp"
#include "preprocess.hpp"
#include "quad_extract.hpp"

namespace {

using aruco3cuda::DetectorConfig;
using aruco3cuda::MemorySpace;
using aruco3cuda::Status;
using aruco3cuda::Workspace;
using aruco3cuda::detail::kQuadCornerCount;

bool has_cuda_device() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

/// 候補 1 つ分。
struct HostCandidate {
    std::vector<cv::Point2f> corners_;
    int label_ = -1;
    int perimeter_ = 0;
};

/// 二値化画像から候補の詰め込みまでを一度に行う。
class CandidateRun {
public:
    CandidateRun() = default;
    CandidateRun(const CandidateRun&) = delete;
    CandidateRun& operator=(const CandidateRun&) = delete;
    ~CandidateRun() {
        if (this->binary_ != nullptr) {
            static_cast<void>(cudaFree(this->binary_));
        }
    }

    /// 処理して結果を取り出す。戻り値は read_candidate_count の Status。
    Status run(const cv::Mat& binary, const DetectorConfig& config) {
        const std::size_t pitch = static_cast<std::size_t>(binary.cols) + 32U;
        if (cudaMalloc(&this->binary_, pitch * static_cast<std::size_t>(binary.rows)) !=
            cudaSuccess) {
            return Status::kCudaError;
        }
        if (cudaMemcpy2D(this->binary_, pitch, binary.data, static_cast<std::size_t>(binary.step),
                         static_cast<std::size_t>(binary.cols),
                         static_cast<std::size_t>(binary.rows),
                         cudaMemcpyHostToDevice) != cudaSuccess) {
            return Status::kCudaError;
        }
        aruco3cuda::detail::ImagePlaneU8 plane;
        plane.data_ = static_cast<std::uint8_t*>(this->binary_);
        plane.width_px_ = binary.cols;
        plane.height_px_ = binary.rows;
        plane.pitch_bytes_ = pitch;

        const std::size_t bytes =
                aruco3cuda::detail::labeling_workspace_bytes(binary.cols, binary.rows) +
                aruco3cuda::detail::label_stats_workspace_bytes(binary.cols, binary.rows) +
                aruco3cuda::detail::quad_workspace_bytes(binary.cols, binary.rows) +
                aruco3cuda::detail::candidate_workspace_bytes(config, binary.cols, binary.rows);
        if (bytes == 0U ||
            this->workspace_.ensure_capacity(bytes, MemorySpace::kDevice, nullptr) != Status::kOk) {
            return Status::kInvalidConfig;
        }
        this->workspace_.reset();
        aruco3cuda::detail::LabelBuffers labels;
        aruco3cuda::detail::LabelStatisticsBuffers stats;
        aruco3cuda::detail::QuadBuffers quads;
        aruco3cuda::detail::CandidateFilterBuffers filter;
        aruco3cuda::detail::DeviceCandidates candidates;
        if (aruco3cuda::detail::reserve_labeling(binary.cols, binary.rows, this->workspace_,
                                                 &labels) != Status::kOk ||
            aruco3cuda::detail::reserve_label_stats(binary.cols, binary.rows, this->workspace_,
                                                    &stats) != Status::kOk ||
            aruco3cuda::detail::reserve_quads(binary.cols, binary.rows, this->workspace_, &quads) !=
                    Status::kOk ||
            aruco3cuda::detail::reserve_candidates(config, binary.cols, binary.rows,
                                                   this->workspace_, &filter,
                                                   &candidates) != Status::kOk) {
            return Status::kInvalidConfig;
        }
        if (aruco3cuda::detail::build_labels_async(plane, &labels, nullptr) != Status::kOk ||
            aruco3cuda::detail::build_label_stats_async(labels, &stats, nullptr) != Status::kOk ||
            aruco3cuda::detail::build_quads_async(labels, stats, &quads, nullptr) != Status::kOk ||
            aruco3cuda::detail::build_candidates_async(labels, stats, quads, config, &filter,
                                                       &candidates, false,
                                                       nullptr) != Status::kOk) {
            return Status::kCudaError;
        }
        const Status status =
                aruco3cuda::detail::read_candidate_count(candidates, &this->count_, nullptr);
        if (status != Status::kOk && status != Status::kCandidateOverflow) {
            return status;
        }
        return this->download(candidates) ? status : Status::kCudaError;
    }

    const std::vector<HostCandidate>& candidates() const { return this->candidates_; }
    int count() const { return this->count_; }

private:
    bool download(const aruco3cuda::detail::DeviceCandidates& candidates) {
        const auto count = static_cast<std::size_t>(this->count_);
        this->candidates_.assign(count, HostCandidate{});
        if (count == 0U) {
            return true;
        }
        // 確保数ではなく候補数だけを読む。候補数を超える範囲は書かれて
        // おらず、読むと未初期化 memory を触ることになる。
        const auto capacity = static_cast<std::size_t>(candidates.capacity_);
        std::vector<std::int32_t> corner_x(count * kQuadCornerCount);
        std::vector<std::int32_t> corner_y(count * kQuadCornerCount);
        std::vector<std::int32_t> label(count);
        std::vector<std::int32_t> perimeter(count);
        const std::size_t row_bytes = count * sizeof(std::int32_t);
        for (int corner = 0; corner < kQuadCornerCount; ++corner) {
            const std::size_t offset = static_cast<std::size_t>(corner) * capacity;
            const std::size_t destination = static_cast<std::size_t>(corner) * count;
            if (cudaMemcpy(corner_x.data() + destination, candidates.corner_x_ + offset, row_bytes,
                           cudaMemcpyDeviceToHost) != cudaSuccess ||
                cudaMemcpy(corner_y.data() + destination, candidates.corner_y_ + offset, row_bytes,
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
                return false;
            }
        }
        if (cudaMemcpy(label.data(), candidates.label_, row_bytes, cudaMemcpyDeviceToHost) !=
                    cudaSuccess ||
            cudaMemcpy(perimeter.data(), candidates.perimeter_, row_bytes,
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            return false;
        }
        for (std::size_t i = 0; i < count; ++i) {
            HostCandidate& item = this->candidates_[i];
            item.label_ = label[i];
            item.perimeter_ = perimeter[i];
            item.corners_.resize(kQuadCornerCount);
            for (int corner = 0; corner < kQuadCornerCount; ++corner) {
                const std::size_t index = (static_cast<std::size_t>(corner) * count) + i;
                item.corners_[static_cast<std::size_t>(corner)] = cv::Point2f(
                        static_cast<float>(corner_x[index]), static_cast<float>(corner_y[index]));
            }
        }
        return true;
    }

    void* binary_ = nullptr;
    Workspace workspace_;
    std::vector<HostCandidate> candidates_;
    int count_ = 0;
};

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

/// 四隅を塗りつぶす。
void fill(cv::Mat& image, const std::vector<cv::Point2f>& corners, int value) {
    std::vector<cv::Point> points;
    points.reserve(corners.size());
    for (const cv::Point2f& corner : corners) {
        points.emplace_back(cvRound(corner.x), cvRound(corner.y));
    }
    cv::fillConvexPoly(image, points, cv::Scalar(value), cv::LINE_8);
}

/// ArUco3 の下限を使わない設定。合成図形の大きさに合わせる。
DetectorConfig plain_config() {
    DetectorConfig config;
    config.use_aruco3_detection_ = false;
    config.min_side_length_canonical_img_px_ = 0;
    config.min_marker_length_ratio_original_img_ = 0.0F;
    return config;
}

// 正常系: 四角形は通り、四隅が詰めて並ぶ。
TEST(CandidateFilterTest, accepts_squares) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    cv::Mat binary(320, 400, CV_8UC1, cv::Scalar(0));
    fill(binary, square_corners(90.0, 90.0, 70.0, 0.0), 255);
    fill(binary, square_corners(260.0, 100.0, 80.0, 27.0), 255);
    fill(binary, square_corners(180.0, 240.0, 90.0, 13.0), 255);

    CandidateRun run;
    ASSERT_EQ(run.run(binary, plain_config()), Status::kOk);
    EXPECT_EQ(run.count(), 3);
    for (const HostCandidate& item : run.candidates()) {
        EXPECT_GE(item.label_, 0);
        EXPECT_GT(item.perimeter_, 0);
        EXPECT_EQ(item.corners_.size(), static_cast<std::size_t>(kQuadCornerCount));
    }
}

// 正常系: L 字は辺の裏付けの判定で落ちる。
//
// 内側比は 0.94 で通ってしまう。極点から引いた辺が成分の外を通ることを
// 別に見ないと落とせないことを実測で確かめている。
TEST(CandidateFilterTest, rejects_l_shape) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    cv::Mat binary(200, 200, CV_8UC1, cv::Scalar(0));
    cv::rectangle(binary, cv::Rect(40, 40, 30, 110), cv::Scalar(255), cv::FILLED);
    cv::rectangle(binary, cv::Rect(40, 120, 110, 30), cv::Scalar(255), cv::FILLED);
    CandidateRun run;
    ASSERT_EQ(run.run(binary, plain_config()), Status::kOk);
    EXPECT_EQ(run.count(), 0);
}

// 境界値: 周長が下限を下回る四角形は落ちる。
TEST(CandidateFilterTest, rejects_small_quad) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    cv::Mat binary(400, 400, CV_8UC1, cv::Scalar(0));
    // 既定の min_marker_perimeter_rate は 0.03。長辺 400 なら chain code 長
    // 12 が下限であり、辺 2 画素の四角形は届かない。
    fill(binary, square_corners(200.0, 200.0, 2.0, 0.0), 255);
    CandidateRun run;
    ASSERT_EQ(run.run(binary, plain_config()), Status::kOk);
    EXPECT_EQ(run.count(), 0);
}

// 境界値: 画像端に近すぎる四角形は落ちる。
TEST(CandidateFilterTest, rejects_quad_near_border) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    cv::Mat binary(200, 200, CV_8UC1, cv::Scalar(0));
    // 左上の角が (0, 0) に接する。既定の min_distance_to_border_px は 3。
    fill(binary,
         {cv::Point2f(0.0F, 0.0F), cv::Point2f(80.0F, 0.0F), cv::Point2f(80.0F, 80.0F),
          cv::Point2f(0.0F, 80.0F)},
         255);
    CandidateRun run;
    ASSERT_EQ(run.run(binary, plain_config()), Status::kOk);
    EXPECT_EQ(run.count(), 0);
}

// 正常系: 穴を持つ枠は通る。マーカーの黒枠に対応する。
TEST(CandidateFilterTest, accepts_marker_like_ring) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    cv::Mat binary(320, 320, CV_8UC1, cv::Scalar(0));
    fill(binary, square_corners(160.0, 160.0, 140.0, 19.0), 255);
    fill(binary, square_corners(160.0, 160.0, 105.0, 19.0), 0);
    CandidateRun run;
    ASSERT_EQ(run.run(binary, plain_config()), Status::kOk);
    ASSERT_EQ(run.count(), 1);
    // 枠の画素は全て推定四角形の内側にあるため、四角形らしさの判定を通る。
    EXPECT_GT(run.candidates()[0].perimeter_, 0);
}

// 異常系: 候補が上限を超えると打ち切り、kCandidateOverflow を返す。
TEST(CandidateFilterTest, reports_overflow_and_truncates) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    cv::Mat binary(400, 400, CV_8UC1, cv::Scalar(0));
    int drawn = 0;
    for (int y = 10; y + 24 < 400; y += 24) {
        for (int x = 10; x + 24 < 400; x += 24) {
            cv::rectangle(binary, cv::Rect(x, y, 18, 18), cv::Scalar(255), cv::FILLED);
            ++drawn;
        }
    }
    ASSERT_GT(drawn, 8);

    DetectorConfig config = plain_config();
    config.max_candidates_ = 8;
    CandidateRun run;
    EXPECT_EQ(run.run(binary, config), Status::kCandidateOverflow);
    // 打ち切っても、書けた分は正しく揃っている。
    EXPECT_EQ(run.count(), 8);
    EXPECT_EQ(run.candidates().size(), 8U);
    for (const HostCandidate& item : run.candidates()) {
        EXPECT_GE(item.label_, 0);
        EXPECT_GT(item.perimeter_, 0);
    }
}

// 境界値: 上限ちょうどでは打ち切りにならない。
TEST(CandidateFilterTest, exact_capacity_is_not_overflow) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    cv::Mat binary(320, 400, CV_8UC1, cv::Scalar(0));
    fill(binary, square_corners(90.0, 90.0, 70.0, 0.0), 255);
    fill(binary, square_corners(260.0, 100.0, 80.0, 27.0), 255);
    fill(binary, square_corners(180.0, 240.0, 90.0, 13.0), 255);

    DetectorConfig config = plain_config();
    config.max_candidates_ = 3;
    CandidateRun run;
    EXPECT_EQ(run.run(binary, config), Status::kOk);
    EXPECT_EQ(run.count(), 3);
}

// 正常系: 候補の並びは label の昇順で、実行ごとに変わらない。
TEST(CandidateFilterTest, order_is_deterministic) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    cv::Mat binary(320, 400, CV_8UC1, cv::Scalar(0));
    for (int i = 0; i < 6; ++i) {
        fill(binary, square_corners(60.0 + (i * 55), 80.0 + ((i % 3) * 90), 40.0, i * 7.0), 255);
    }
    CandidateRun first;
    CandidateRun second;
    ASSERT_EQ(first.run(binary, plain_config()), Status::kOk);
    ASSERT_EQ(second.run(binary, plain_config()), Status::kOk);
    ASSERT_EQ(first.count(), second.count());
    ASSERT_GT(first.count(), 1);
    for (std::size_t i = 0; i < first.candidates().size(); ++i) {
        EXPECT_EQ(first.candidates()[i].label_, second.candidates()[i].label_);
        EXPECT_EQ(first.candidates()[i].corners_, second.candidates()[i].corners_);
        if (i > 0) {
            EXPECT_LT(first.candidates()[i - 1].label_, first.candidates()[i].label_);
        }
    }
}

// 境界値: 前景が無ければ候補も 0 件になる。
TEST(CandidateFilterTest, empty_image_has_no_candidate) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const cv::Mat binary(120, 160, CV_8UC1, cv::Scalar(0));
    CandidateRun run;
    ASSERT_EQ(run.run(binary, plain_config()), Status::kOk);
    EXPECT_EQ(run.count(), 0);
}

// 異常系: 引数が不正なら実行しない。
TEST(CandidateFilterTest, rejects_invalid_arguments) {
    Workspace workspace;
    const DetectorConfig config;
    aruco3cuda::detail::CandidateFilterBuffers buffers;
    aruco3cuda::detail::DeviceCandidates candidates;
    EXPECT_EQ(aruco3cuda::detail::reserve_candidates(config, 4, 4, workspace, nullptr, &candidates),
              Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::detail::reserve_candidates(config, 4, 4, workspace, &buffers, nullptr),
              Status::kInvalidArgument);
    EXPECT_EQ(
            aruco3cuda::detail::reserve_candidates(config, 0, 4, workspace, &buffers, &candidates),
            Status::kInvalidArgument);

    aruco3cuda::detail::LabelBuffers labels;
    aruco3cuda::detail::LabelStatisticsBuffers stats;
    aruco3cuda::detail::QuadBuffers quads;
    EXPECT_EQ(aruco3cuda::detail::build_candidates_async(labels, stats, quads, config, nullptr,
                                                         &candidates, false, nullptr),
              Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::detail::build_candidates_async(labels, stats, quads, config, &buffers,
                                                         &candidates, false, nullptr),
              Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::detail::read_candidate_count(candidates, nullptr, nullptr),
              Status::kInvalidArgument);

    EXPECT_EQ(aruco3cuda::detail::candidate_workspace_bytes(config, 0, 4), 0U);
    DetectorConfig zero_capacity = config;
    zero_capacity.max_candidates_ = 0;
    EXPECT_EQ(aruco3cuda::detail::candidate_workspace_bytes(zero_capacity, 4, 4), 0U);
}

// 正常系: 円、楕円、六角形は内側比の判定で落ちる。
TEST(CandidateFilterTest, rejects_round_shapes) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    {
        cv::Mat binary(400, 400, CV_8UC1, cv::Scalar(0));
        cv::circle(binary, cv::Point(200, 200), 120, cv::Scalar(255), cv::FILLED);
        CandidateRun run;
        ASSERT_EQ(run.run(binary, plain_config()), Status::kOk);
        EXPECT_EQ(run.count(), 0) << "円";
    }
    {
        cv::Mat binary(400, 400, CV_8UC1, cv::Scalar(0));
        cv::ellipse(binary, cv::Point(200, 200), cv::Size(160, 70), 25.0, 0.0, 360.0,
                    cv::Scalar(255), cv::FILLED);
        CandidateRun run;
        ASSERT_EQ(run.run(binary, plain_config()), Status::kOk);
        EXPECT_EQ(run.count(), 0) << "楕円";
    }
    {
        cv::Mat binary(400, 400, CV_8UC1, cv::Scalar(0));
        std::vector<cv::Point> hexagon;
        hexagon.reserve(6);
        for (int i = 0; i < 6; ++i) {
            hexagon.emplace_back(cvRound(200.0 + (140.0 * std::cos(i * CV_PI / 3.0))),
                                 cvRound(200.0 + (140.0 * std::sin(i * CV_PI / 3.0))));
        }
        cv::fillConvexPoly(binary, hexagon, cv::Scalar(255));
        CandidateRun run;
        ASSERT_EQ(run.run(binary, plain_config()), Status::kOk);
        EXPECT_EQ(run.count(), 0) << "六角形";
    }
}

// 正常系: 十字は辺の裏付けの判定で落ちる。
TEST(CandidateFilterTest, rejects_cross_shape) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    cv::Mat binary(400, 400, CV_8UC1, cv::Scalar(0));
    const std::vector<cv::Point> cross = {{170, 50},  {230, 50},  {230, 170}, {350, 170},
                                          {350, 230}, {230, 230}, {230, 350}, {170, 350},
                                          {170, 230}, {50, 230},  {50, 170},  {170, 170}};
    cv::fillPoly(binary, cross, cv::Scalar(255));
    CandidateRun run;
    ASSERT_EQ(run.run(binary, plain_config()), Status::kOk);
    EXPECT_EQ(run.count(), 0);
}

// 正常系: 三角形は落ちる。
//
// 極点探索では、直線 c0c2 が三角形の 1 辺に重なる向きが選ばれる。片側に
// 点が残らないため四隅が定まらず、無効になる。向きを変えても同じである
// ことを確かめる。
TEST(CandidateFilterTest, rejects_triangles) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const std::vector<std::vector<cv::Point>> triangles = {{{60, 320}, {340, 320}, {200, 60}},
                                                           {{60, 60}, {340, 120}, {150, 330}},
                                                           {{200, 40}, {360, 200}, {40, 260}},
                                                           {{80, 300}, {320, 260}, {300, 70}}};
    for (std::size_t i = 0; i < triangles.size(); ++i) {
        cv::Mat binary(400, 400, CV_8UC1, cv::Scalar(0));
        cv::fillConvexPoly(binary, triangles[i], cv::Scalar(255));
        CandidateRun run;
        ASSERT_EQ(run.run(binary, plain_config()), Status::kOk) << i;
        EXPECT_EQ(run.count(), 0) << "三角形 " << i;
    }
}

// 既知の限界: 重なって 1 成分になった 2 枚は 1 つの四角形になる。
//
// CPU 経路では輪郭が 8 角形になり多角形近似で落ちる。案 A では外接する
// 四角形として受け入れられ、2 枚とも取りこぼす。この差は WP-2.6 の比較
// 対象であり、挙動を隠さず固定するためにこの test を置いている。
TEST(CandidateFilterTest, known_limitation_overlapping_markers_merge) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    cv::Mat binary(400, 400, CV_8UC1, cv::Scalar(0));
    // 横に少し重ねる。連結成分としては 1 つになる。
    fill(binary, square_corners(160.0, 200.0, 120.0, 0.0), 255);
    fill(binary, square_corners(250.0, 200.0, 120.0, 0.0), 255);
    CandidateRun run;
    ASSERT_EQ(run.run(binary, plain_config()), Status::kOk);
    // 2 枚ではなく 1 つの候補になる。
    EXPECT_EQ(run.count(), 1);
}

// 既知の限界: 縦にずらして重ねた 2 枚は、四角形でない形になり落ちる。
//
// 重なり方によって「1 つの四角形になる」か「落ちる」かが変わる。どちらも
// 2 枚を取りこぼす点は同じであり、CPU 基準との差として WP-2.6 で数える。
TEST(CandidateFilterTest, known_limitation_staggered_markers_are_lost) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    cv::Mat binary(400, 400, CV_8UC1, cv::Scalar(0));
    fill(binary, square_corners(150.0, 160.0, 120.0, 0.0), 255);
    fill(binary, square_corners(240.0, 250.0, 120.0, 0.0), 255);
    CandidateRun run;
    ASSERT_EQ(run.run(binary, plain_config()), Status::kOk);
    EXPECT_LT(run.count(), 2);
}

}  // namespace
