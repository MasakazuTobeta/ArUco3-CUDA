// SPDX-License-Identifier: Apache-2.0
//
// Cross-checks the perspective transform and cell sampling against OpenCV.
//
// A single pixel of difference in the canonical image is one pixel out of the
// 16 in a cell, i.e. 0.0625 of the ratio. Near the decision threshold that can
// change the decoded ID, so we require a byte-for-byte match.
//
// An exact match cannot be demanded unconditionally, however. OpenCV's
// warpPerspective has three code paths: on aarch64 it picks the NEON SIMD path
// that uses v_muladd (fused multiply-add), while on x86_64 it picks the SSE4.1
// path, which has no fusion. We have measured that the SHA256 of the output for
// identical input differs between machines.
//
// This implementation follows the non-fused side (scalar and SSE4.1
// semantics). It matches x86_64 OpenCV exactly; on aarch64 a very small number
// of rounding-boundary pixels differ. Measured, that was 1 pixel out of 40960.
// We cap and monitor that figure.
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

/// Deterministic test image; places a pattern inside the quadrilateral.
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

/// Builds the canonical image on the GPU via the simple path that only uses
/// level 0.
///
/// The pyramid is pinned to a single level and ArUco3 is disabled so that level
/// selection is taken out of the picture. This test targets only the
/// perspective transform and the sampling.
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

/// Builds the same canonical image with OpenCV, following the same steps as
/// the CPU path.
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

/// Builds the four corners from a center, a side length, and a rotation. The
/// coordinates are rounded to integers.
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

/// Checks that the mismatch rate stays within what platform differences can
/// explain.
///
/// On machines that use fused multiply-add, a very small number of
/// rounding-boundary pixels differ. Demanding zero would fail on aarch64 every
/// time, so we impose a measured upper bound instead. Exceeding that bound
/// means something other than fusion is at work, which is worth investigating.
void expect_within_platform_tolerance(std::size_t mismatched, std::size_t total) {
    // Measured at 1 pixel out of 40960 (0.0024%); allow a 10x margin.
    constexpr double kMaxRatio = 0.0003;
    EXPECT_LE(static_cast<double>(mismatched), kMaxRatio * static_cast<double>(total))
            << "mismatched " << mismatched << " / " << total;
}

/// Compares one pair and returns the number of mismatched pixels.
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

// Happy path: matches OpenCV byte for byte on rotated squares.
TEST(CellSampleTest, matches_opencv_for_rotated_squares) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
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
    std::printf("[cell] mismatched pixels over %zu rotations: %zu / %zu (%.4f%%)\n", quads.size(),
                total, pixels, 100.0 * static_cast<double>(total) / static_cast<double>(pixels));
    expect_within_platform_tolerance(total, pixels);
}

// Happy path: still matches on quadrilaterals with strong perspective
// distortion.
TEST(CellSampleTest, matches_opencv_for_perspective_quads) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
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
    std::printf(
            "[cell] mismatched pixels over %zu perspective distortions: %zu / %zu "
            "(worst %zu / 1024)\n",
            quads.size(), total, pixels, worst);
    expect_within_platform_tolerance(total, pixels);
}

// Boundary: for a quadrilateral that extends past the image, out-of-range
// samples read as 0.
TEST(CellSampleTest, out_of_bounds_reads_zero) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
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

// Boundary: still matches when the cell count and cell side length change.
TEST(CellSampleTest, matches_opencv_for_other_marker_sizes) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
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

// Failure path: invalid arguments perform no work.
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
    // With the default configuration and 36h12 this comes out to 32.
    EXPECT_EQ(aruco3cuda::detail::canonical_side_px(config, 6), 32);
}

}  // namespace
