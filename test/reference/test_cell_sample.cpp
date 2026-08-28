// SPDX-License-Identifier: Apache-2.0
//
// 射影変換とセル sampling を OpenCV と突き合わせる。
//
// canonical 画像の 1 画素の違いは、セル 16 画素中の 1 画素、すなわち比の
// 0.0625 に相当する。判定の閾値近傍で ID が変わりうるため、byte 単位の
// 一致を求める。
//
// ただし完全一致を常に求めることはできない。OpenCV の warpPerspective は
// 経路が 3 つあり、aarch64 では NEON の v_muladd (積和融合) を使う SIMD 経路、
// x86_64 では融合を持たない SSE4.1 経路が選ばれる。同じ入力に対する出力の
// SHA256 が機種で異なることを実測で確認している。
//
// 本実装は融合しない側 (scalar と SSE4.1 の意味論) に合わせている。x86_64 の
// OpenCV とは完全に一致し、aarch64 では丸め境界のごく一部が異なる。実測では
// 40960 画素中 1 画素であった。上限を置いて監視する。
#include "cell_sample.hpp"

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"
#include "aruco3cuda/workspace.hpp"
#include "candidate_filter.hpp"
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

/// 決定的な試験画像。四角形の内側に模様を置く。
cv::Mat make_pattern(int width, int height, std::uint64_t seed) {
    cv::Mat image(height, width, CV_8UC1);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> noise(0, 255);
    for (int y = 0; y < height; ++y) {
        auto* row = image.ptr<std::uint8_t>(y);
        for (int x = 0; x < width; ++x) {
            row[x] = static_cast<std::uint8_t>(noise(rng));
        }
    }
    return image;
}

/// GPU 側の canonical 画像を作る。level 0 のみを使う単純な経路。
///
/// pyramid の段数を 1 に固定し、ArUco3 を無効にして level 選択を外す。
/// この test は射影変換と sampling だけを対象とする。
class CanonicalRun {
public:
    CanonicalRun() = default;
    CanonicalRun(const CanonicalRun&) = delete;
    CanonicalRun& operator=(const CanonicalRun&) = delete;
    ~CanonicalRun() {
        if (this->image_ != nullptr) {
            static_cast<void>(cudaFree(this->image_));
        }
    }

    bool run(const cv::Mat& source, const std::vector<std::vector<cv::Point2f>>& quads,
             int marker_size) {
        DetectorConfig config;
        config.use_aruco3_detection_ = false;
        config.min_side_length_canonical_img_px_ = 0;
        config.min_marker_length_ratio_original_img_ = 0.0F;
        config.max_candidates_ = static_cast<int>(quads.size());

        const std::size_t pitch = static_cast<std::size_t>(source.cols) + 48U;
        if (cudaMalloc(&this->image_, pitch * static_cast<std::size_t>(source.rows)) !=
            cudaSuccess) {
            return false;
        }
        if (cudaMemcpy2D(this->image_, pitch, source.data, static_cast<std::size_t>(source.step),
                         static_cast<std::size_t>(source.cols),
                         static_cast<std::size_t>(source.rows),
                         cudaMemcpyHostToDevice) != cudaSuccess) {
            return false;
        }

        aruco3cuda::detail::PreprocessBuffers preprocess;
        preprocess.level0_.data_ = static_cast<const std::uint8_t*>(this->image_);
        preprocess.level0_.width_px_ = source.cols;
        preprocess.level0_.height_px_ = source.rows;
        preprocess.level0_.pitch_bytes_ = pitch;
        preprocess.level0_.space_ = MemorySpace::kDevice;
        preprocess.level_count_ = 1;

        aruco3cuda::detail::ScalePlan plan;
        plan.segmentation_width_px_ = source.cols;
        plan.segmentation_height_px_ = source.rows;

        const std::size_t bytes =
                aruco3cuda::detail::candidate_workspace_bytes(config, 64, 64) +
                aruco3cuda::detail::canonical_workspace_bytes(config, marker_size);
        if (bytes == 0U ||
            this->workspace_.ensure_capacity(bytes, MemorySpace::kDevice, nullptr) != Status::kOk) {
            return false;
        }
        this->workspace_.reset();
        aruco3cuda::detail::CandidateFilterBuffers filter;
        aruco3cuda::detail::DeviceCandidates candidates;
        aruco3cuda::detail::CanonicalBuffers canonical;
        if (aruco3cuda::detail::reserve_candidates(config, 64, 64, this->workspace_, &filter,
                                                   &candidates) != Status::kOk ||
            aruco3cuda::detail::reserve_canonical(config, marker_size, this->workspace_,
                                                  &canonical) != Status::kOk) {
            return false;
        }
        if (!upload_quads(quads, candidates)) {
            return false;
        }
        if (aruco3cuda::detail::build_canonical_async(preprocess, plan, candidates, config,
                                                      &canonical, nullptr) != Status::kOk) {
            return false;
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            return false;
        }
        return download(canonical, quads.size());
    }

    const std::vector<cv::Mat>& images() const { return this->images_; }

private:
    static bool upload_quads(const std::vector<std::vector<cv::Point2f>>& quads,
                             const aruco3cuda::detail::DeviceCandidates& candidates) {
        const auto count = static_cast<int>(quads.size());
        const auto capacity = static_cast<std::size_t>(candidates.capacity_);
        std::vector<std::int32_t> corner_x(capacity * kQuadCornerCount, 0);
        std::vector<std::int32_t> corner_y(capacity * kQuadCornerCount, 0);
        std::vector<std::int32_t> perimeter(capacity, 0);
        for (std::size_t i = 0; i < quads.size(); ++i) {
            for (int c = 0; c < kQuadCornerCount; ++c) {
                const std::size_t index = (static_cast<std::size_t>(c) * capacity) + i;
                corner_x[index] =
                        static_cast<std::int32_t>(quads[i][static_cast<std::size_t>(c)].x);
                corner_y[index] =
                        static_cast<std::int32_t>(quads[i][static_cast<std::size_t>(c)].y);
            }
            perimeter[i] = 100;
        }
        const std::size_t corner_bytes = corner_x.size() * sizeof(std::int32_t);
        return cudaMemcpy(candidates.corner_x_, corner_x.data(), corner_bytes,
                          cudaMemcpyHostToDevice) == cudaSuccess &&
               cudaMemcpy(candidates.corner_y_, corner_y.data(), corner_bytes,
                          cudaMemcpyHostToDevice) == cudaSuccess &&
               cudaMemcpy(candidates.perimeter_, perimeter.data(),
                          perimeter.size() * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice) == cudaSuccess &&
               cudaMemcpy(candidates.count_, &count, sizeof(int), cudaMemcpyHostToDevice) ==
                       cudaSuccess;
    }

    bool download(const aruco3cuda::detail::CanonicalBuffers& canonical, std::size_t count) {
        const auto side = static_cast<std::size_t>(canonical.side_px_);
        std::vector<std::uint8_t> raw(count * side * side);
        if (cudaMemcpy(raw.data(), canonical.images_, raw.size(), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
            return false;
        }
        this->images_.clear();
        for (std::size_t i = 0; i < count; ++i) {
            cv::Mat image(canonical.side_px_, canonical.side_px_, CV_8UC1);
            std::copy(raw.begin() + static_cast<std::ptrdiff_t>(i * side * side),
                      raw.begin() + static_cast<std::ptrdiff_t>((i + 1) * side * side), image.data);
            this->images_.push_back(image);
        }
        return true;
    }

    void* image_ = nullptr;
    Workspace workspace_;
    std::vector<cv::Mat> images_;
};

/// OpenCV で同じ canonical 画像を作る。CPU 経路と同じ手順である。
cv::Mat reference_canonical(const cv::Mat& source, const std::vector<cv::Point2f>& quad, int side) {
    std::vector<cv::Point2f> destination(4);
    destination[0] = cv::Point2f(0.0F, 0.0F);
    destination[1] = cv::Point2f(static_cast<float>(side) - 1.0F, 0.0F);
    destination[2] = cv::Point2f(static_cast<float>(side) - 1.0F, static_cast<float>(side) - 1.0F);
    destination[3] = cv::Point2f(0.0F, static_cast<float>(side) - 1.0F);
    const cv::Mat transformation = cv::getPerspectiveTransform(quad, destination);
    cv::Mat canonical;
    cv::warpPerspective(source, canonical, transformation, cv::Size(side, side), cv::INTER_NEAREST);
    return canonical;
}

/// 中心と辺長と回転から四隅を作る。座標は整数へ丸める。
std::vector<cv::Point2f> square_quad(double cx, double cy, double side, double degrees) {
    const double radians = degrees * CV_PI / 180.0;
    const double half = side / 2.0;
    const double offsets[4][2] = {{-half, -half}, {half, -half}, {half, half}, {-half, half}};
    std::vector<cv::Point2f> quad;
    quad.reserve(4);
    for (const auto& offset : offsets) {
        const double x = cx + (offset[0] * std::cos(radians)) - (offset[1] * std::sin(radians));
        const double y = cy + (offset[0] * std::sin(radians)) + (offset[1] * std::cos(radians));
        quad.emplace_back(static_cast<float>(cvRound(x)), static_cast<float>(cvRound(y)));
    }
    return quad;
}

/// 不一致の割合が、機種差として説明できる範囲に収まることを確かめる。
///
/// 積和の融合を使う機種では、丸め境界のごく一部が異なる。0 を要求すると
/// aarch64 で常に落ちるため、実測に基づく上限を置く。上限を超えたら融合以外の
/// 原因があるということであり、調べる価値がある。
void expect_within_platform_tolerance(std::size_t mismatched, std::size_t total) {
    // 実測は 40960 画素中 1 画素 (0.0024%)。10 倍の余裕を持たせる。
    constexpr double kMaxRatio = 0.0003;
    EXPECT_LE(static_cast<double>(mismatched), kMaxRatio * static_cast<double>(total))
            << "不一致 " << mismatched << " / " << total;
}

/// 1 組を突き合わせ、不一致画素の数を返す。
std::size_t compare(const cv::Mat& expected, const cv::Mat& actual) {
    EXPECT_EQ(expected.size(), actual.size());
    std::size_t mismatched = 0;
    for (int y = 0; y < expected.rows; ++y) {
        const auto* e = expected.ptr<std::uint8_t>(y);
        const auto* a = actual.ptr<std::uint8_t>(y);
        for (int x = 0; x < expected.cols; ++x) {
            if (e[x] != a[x]) {
                ++mismatched;
            }
        }
    }
    return mismatched;
}

// 正常系: 回転した正方形で OpenCV と byte 単位で一致する。
TEST(CellSampleTest, matches_opencv_for_rotated_squares) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const cv::Mat source = make_pattern(640, 480, 20260828U);
    std::vector<std::vector<cv::Point2f>> quads;
    for (int angle = 0; angle < 90; angle += 7) {
        quads.push_back(square_quad(320.0, 240.0, 160.0, angle));
    }
    CanonicalRun run;
    ASSERT_TRUE(run.run(source, quads, 6));
    ASSERT_EQ(run.images().size(), quads.size());

    std::size_t total = 0;
    for (std::size_t i = 0; i < quads.size(); ++i) {
        const cv::Mat expected = reference_canonical(source, quads[i], 32);
        total += compare(expected, run.images()[i]);
    }
    const std::size_t pixels = quads.size() * static_cast<std::size_t>(32) * 32U;
    std::printf("[cell] 回転 %zu 通りの不一致画素 %zu / %zu (%.4f%%)\n", quads.size(), total,
                pixels, 100.0 * static_cast<double>(total) / static_cast<double>(pixels));
    expect_within_platform_tolerance(total, pixels);
}

// 正常系: 射影で強く歪んだ四角形でも一致する。
TEST(CellSampleTest, matches_opencv_for_perspective_quads) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const cv::Mat source = make_pattern(800, 600, 31415U);
    std::mt19937 rng(20260828U);
    std::uniform_int_distribution<int> jitter(-60, 60);
    std::vector<std::vector<cv::Point2f>> quads;
    for (int i = 0; i < 40; ++i) {
        std::vector<cv::Point2f> quad = square_quad(400.0, 300.0, 240.0, i * 3.0);
        for (auto& corner : quad) {
            corner.x += static_cast<float>(jitter(rng));
            corner.y += static_cast<float>(jitter(rng));
        }
        quads.push_back(quad);
    }
    CanonicalRun run;
    ASSERT_TRUE(run.run(source, quads, 6));

    std::size_t total = 0;
    std::size_t worst = 0;
    for (std::size_t i = 0; i < quads.size(); ++i) {
        const cv::Mat expected = reference_canonical(source, quads[i], 32);
        const std::size_t mismatched = compare(expected, run.images()[i]);
        total += mismatched;
        worst = std::max(worst, mismatched);
    }
    const std::size_t pixels = quads.size() * static_cast<std::size_t>(32) * 32U;
    std::printf("[cell] 射影歪み %zu 通りの不一致画素 %zu / %zu (最大 %zu / 1024)\n", quads.size(),
                total, pixels, worst);
    expect_within_platform_tolerance(total, pixels);
}

// 境界値: 画像の外へはみ出す四角形では、範囲外が 0 になる。
TEST(CellSampleTest, out_of_bounds_reads_zero) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const cv::Mat source = make_pattern(200, 200, 7U);
    const std::vector<std::vector<cv::Point2f>> quads = {square_quad(20.0, 20.0, 120.0, 0.0),
                                                         square_quad(190.0, 190.0, 100.0, 15.0)};
    CanonicalRun run;
    ASSERT_TRUE(run.run(source, quads, 6));
    for (std::size_t i = 0; i < quads.size(); ++i) {
        const cv::Mat expected = reference_canonical(source, quads[i], 32);
        expect_within_platform_tolerance(compare(expected, run.images()[i]),
                                         static_cast<std::size_t>(32) * 32U);
    }
}

// 境界値: cell 数と cell 辺長を変えても一致する。
TEST(CellSampleTest, matches_opencv_for_other_marker_sizes) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const cv::Mat source = make_pattern(500, 500, 99U);
    const std::vector<std::vector<cv::Point2f>> quads = {square_quad(250.0, 250.0, 200.0, 23.0)};
    for (const int marker_size : {4, 5, 7}) {
        CanonicalRun run;
        ASSERT_TRUE(run.run(source, quads, marker_size)) << marker_size;
        const int side = (marker_size + 2) * 4;
        const cv::Mat expected = reference_canonical(source, quads[0], side);
        expect_within_platform_tolerance(
                compare(expected, run.images()[0]),
                static_cast<std::size_t>(side) * static_cast<std::size_t>(side));
    }
}

// 異常系: 引数が不正なら実行しない。
TEST(CellSampleTest, rejects_invalid_arguments) {
    Workspace workspace;
    const DetectorConfig config;
    aruco3cuda::detail::CanonicalBuffers canonical;
    EXPECT_EQ(aruco3cuda::detail::reserve_canonical(config, 6, workspace, nullptr),
              Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::detail::reserve_canonical(config, 0, workspace, &canonical),
              Status::kInvalidArgument);
    EXPECT_NE(aruco3cuda::detail::reserve_canonical(config, 6, workspace, &canonical), Status::kOk);

    aruco3cuda::detail::PreprocessBuffers preprocess;
    aruco3cuda::detail::ScalePlan plan;
    aruco3cuda::detail::DeviceCandidates candidates;
    EXPECT_EQ(aruco3cuda::detail::build_canonical_async(preprocess, plan, candidates, config,
                                                        nullptr, nullptr),
              Status::kInvalidArgument);

    EXPECT_EQ(aruco3cuda::detail::canonical_side_px(config, 0), 0);
    EXPECT_EQ(aruco3cuda::detail::canonical_workspace_bytes(config, 0), 0U);
    // 既定設定と 36h12 では 32 になる。
    EXPECT_EQ(aruco3cuda::detail::canonical_side_px(config, 6), 32);
}

}  // namespace
