// SPDX-License-Identifier: Apache-2.0
//
// Verifies candidate filtering, compaction, and capacity overflow.
//
// The filter mirrors the decisions of the CPU route, but it measures the perimeter
// and the squareness differently. How that difference shows up in the results is
// pinned down here with both shapes that pass and shapes that are rejected. These
// tests also confirm that an overflow is reported through a Status instead of being
// silently discarded.
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

/// A single candidate.
struct HostCandidate {
    std::vector<cv::Point2f> corners_;
    int label_ = -1;
    int perimeter_ = 0;
};

/// Runs everything from the binary image through to candidate compaction.
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

    /// Runs the pipeline and reads the results back. Returns the Status of
    /// read_candidate_count.
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
        // Read only as many entries as there are candidates, not the full capacity.
        // Nothing beyond the candidate count has been written, so reading it would
        // touch uninitialized memory.
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

/// Builds the four corners of a square from a center, an edge length, and a rotation.
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

/// Fills the polygon described by the corners.
void fill(cv::Mat& image, const std::vector<cv::Point2f>& corners, int value) {
    std::vector<cv::Point> points;
    points.reserve(corners.size());
    for (const cv::Point2f& corner : corners) {
        points.emplace_back(cvRound(corner.x), cvRound(corner.y));
    }
    cv::fillConvexPoly(image, points, cv::Scalar(value), cv::LINE_8);
}

/// Configuration that disables the ArUco3 lower bounds, matching the size of the
/// synthetic shapes.
DetectorConfig plain_config() {
    DetectorConfig config;
    config.use_aruco3_detection_ = false;
    config.min_side_length_canonical_img_px_ = 0;
    config.min_marker_length_ratio_original_img_ = 0.0F;
    return config;
}

// Happy path: quadrilaterals pass, and their corners come back compacted.
TEST(CandidateFilterTest, accepts_squares) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
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

// Happy path: an L shape is rejected by the edge support check.
//
// Its inside ratio is 0.94, which would pass. Measurement confirms that it can only
// be rejected by separately checking that an edge drawn between the extreme points
// runs outside the component.
TEST(CandidateFilterTest, rejects_l_shape) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    cv::Mat binary(200, 200, CV_8UC1, cv::Scalar(0));
    cv::rectangle(binary, cv::Rect(40, 40, 30, 110), cv::Scalar(255), cv::FILLED);
    cv::rectangle(binary, cv::Rect(40, 120, 110, 30), cv::Scalar(255), cv::FILLED);
    CandidateRun run;
    ASSERT_EQ(run.run(binary, plain_config()), Status::kOk);
    EXPECT_EQ(run.count(), 0);
}

// Boundary case: a quadrilateral whose perimeter is below the lower bound is
// rejected.
TEST(CandidateFilterTest, rejects_small_quad) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    cv::Mat binary(400, 400, CV_8UC1, cv::Scalar(0));
    // The default min_marker_perimeter_rate is 0.03. With a long side of 400 the
    // lower bound is a chain code length of 12, which a 2-pixel square cannot reach.
    fill(binary, square_corners(200.0, 200.0, 2.0, 0.0), 255);
    CandidateRun run;
    ASSERT_EQ(run.run(binary, plain_config()), Status::kOk);
    EXPECT_EQ(run.count(), 0);
}

// Boundary case: a quadrilateral too close to the image border is rejected.
TEST(CandidateFilterTest, rejects_quad_near_border) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    cv::Mat binary(200, 200, CV_8UC1, cv::Scalar(0));
    // The top-left corner touches (0, 0). The default min_distance_to_border_px is 3.
    fill(binary,
         {cv::Point2f(0.0F, 0.0F), cv::Point2f(80.0F, 0.0F), cv::Point2f(80.0F, 80.0F),
          cv::Point2f(0.0F, 80.0F)},
         255);
    CandidateRun run;
    ASSERT_EQ(run.run(binary, plain_config()), Status::kOk);
    EXPECT_EQ(run.count(), 0);
}

// Happy path: a ring with a hole passes. This corresponds to the black border of a
// marker.
TEST(CandidateFilterTest, accepts_marker_like_ring) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    cv::Mat binary(320, 320, CV_8UC1, cv::Scalar(0));
    fill(binary, square_corners(160.0, 160.0, 140.0, 19.0), 255);
    fill(binary, square_corners(160.0, 160.0, 105.0, 19.0), 0);
    CandidateRun run;
    ASSERT_EQ(run.run(binary, plain_config()), Status::kOk);
    ASSERT_EQ(run.count(), 1);
    // Every pixel of the ring lies inside the estimated quadrilateral, so it passes
    // the squareness check.
    EXPECT_GT(run.candidates()[0].perimeter_, 0);
}

// Failure path: once the candidates exceed the capacity, the run is truncated and
// kCandidateOverflow is returned.
TEST(CandidateFilterTest, reports_overflow_and_truncates) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
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
    // Even when truncated, the entries that were written are complete and consistent.
    EXPECT_EQ(run.count(), 8);
    EXPECT_EQ(run.candidates().size(), 8U);
    for (const HostCandidate& item : run.candidates()) {
        EXPECT_GE(item.label_, 0);
        EXPECT_GT(item.perimeter_, 0);
    }
}

// Boundary case: hitting the capacity exactly is not an overflow.
TEST(CandidateFilterTest, exact_capacity_is_not_overflow) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
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

// Happy path: candidates come out in ascending label order, identical across runs.
TEST(CandidateFilterTest, order_is_deterministic) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
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

// Boundary case: with no foreground there are no candidates either.
TEST(CandidateFilterTest, empty_image_has_no_candidate) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    const cv::Mat binary(120, 160, CV_8UC1, cv::Scalar(0));
    CandidateRun run;
    ASSERT_EQ(run.run(binary, plain_config()), Status::kOk);
    EXPECT_EQ(run.count(), 0);
}

// Failure path: nothing runs when the arguments are invalid.
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

// Happy path: circles, ellipses, and hexagons are rejected by the inside ratio
// check.
TEST(CandidateFilterTest, rejects_round_shapes) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    {
        cv::Mat binary(400, 400, CV_8UC1, cv::Scalar(0));
        cv::circle(binary, cv::Point(200, 200), 120, cv::Scalar(255), cv::FILLED);
        CandidateRun run;
        ASSERT_EQ(run.run(binary, plain_config()), Status::kOk);
        EXPECT_EQ(run.count(), 0) << "circle";
    }
    {
        cv::Mat binary(400, 400, CV_8UC1, cv::Scalar(0));
        cv::ellipse(binary, cv::Point(200, 200), cv::Size(160, 70), 25.0, 0.0, 360.0,
                    cv::Scalar(255), cv::FILLED);
        CandidateRun run;
        ASSERT_EQ(run.run(binary, plain_config()), Status::kOk);
        EXPECT_EQ(run.count(), 0) << "ellipse";
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
        EXPECT_EQ(run.count(), 0) << "hexagon";
    }
}

// Happy path: a cross is rejected by the edge support check.
TEST(CandidateFilterTest, rejects_cross_shape) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
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

// Happy path: triangles are rejected.
//
// The extreme-point search picks an orientation in which the line c0c2 lies along one
// edge of the triangle. No point is left on one side, so the four corners cannot be
// determined and the candidate becomes invalid. This test confirms that the same
// holds at other orientations.
TEST(CandidateFilterTest, rejects_triangles) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
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
        EXPECT_EQ(run.count(), 0) << "triangle " << i;
    }
}

// Known limitation: two markers that overlap into a single component become one
// quadrilateral.
//
// On the CPU route the contour becomes an octagon and is rejected by the polygon
// approximation. Approach A accepts it as the enclosing quadrilateral, so both markers
// are missed. This difference is part of what the candidate extraction comparison
// measures; this test exists to pin the behavior down rather than hide it.
TEST(CandidateFilterTest, known_limitation_overlapping_markers_merge) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    cv::Mat binary(400, 400, CV_8UC1, cv::Scalar(0));
    // Overlap them slightly along x, so they form a single connected component.
    fill(binary, square_corners(160.0, 200.0, 120.0, 0.0), 255);
    fill(binary, square_corners(250.0, 200.0, 120.0, 0.0), 255);
    CandidateRun run;
    ASSERT_EQ(run.run(binary, plain_config()), Status::kOk);
    // The result is one candidate rather than two.
    EXPECT_EQ(run.count(), 1);
}

// Known limitation: two markers overlapped with a vertical offset form a non-quad
// shape and are rejected.
//
// Depending on how they overlap, the outcome switches between "becomes one
// quadrilateral" and "is rejected". Either way both markers are missed, and that is
// counted as a difference from the CPU reference.
TEST(CandidateFilterTest, known_limitation_staggered_markers_are_lost) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    cv::Mat binary(400, 400, CV_8UC1, cv::Scalar(0));
    fill(binary, square_corners(150.0, 160.0, 120.0, 0.0), 255);
    fill(binary, square_corners(240.0, 250.0, 120.0, 0.0), 255);
    CandidateRun run;
    ASSERT_EQ(run.run(binary, plain_config()), Status::kOk);
    EXPECT_LT(run.count(), 2);
}

}  // namespace
