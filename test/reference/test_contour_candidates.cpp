// SPDX-License-Identifier: Apache-2.0
// Independent public-OpenCV oracles for the device contour and hole route.
#include <gtest/gtest.h>

#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/workspace.hpp"
#include "candidate_filter.hpp"
#include "contour_candidates.hpp"
#include "cpu_candidates.hpp"
#include "device_image.hpp"

namespace {

using aruco3cuda::DetectorConfig;
using aruco3cuda::Status;
using aruco3cuda::detail::ContourCandidateBuffers;
using aruco3cuda::detail::ContourPoint;
using Point = std::pair<int, int>;
using Path = std::vector<Point>;

bool has_cuda_device() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

void checked(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

void checked(Status status) {
    if (status != Status::kOk) {
        throw std::runtime_error(aruco3cuda::to_string(status));
    }
}

template <typename T>
std::vector<T> read_array(const T* source, std::size_t count) {
    std::vector<T> result(count);
    if (count != 0U) {
        checked(cudaMemcpy(result.data(), source, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
    return result;
}

// Contours may start at a different boundary edge. Rotation is immaterial, but
// direction is retained: reversing a hole into an outer contour must not pass.
Path rotate_canonically(const Path& path) {
    if (path.empty()) {
        return path;
    }
    Path best = path;
    for (std::size_t start = 1; start < path.size(); ++start) {
        if (path[start] != best[0]) {
            if (path[start] > best[0]) {
                continue;
            }
        }
        Path rotated;
        rotated.insert(rotated.end(), path.begin() + static_cast<std::ptrdiff_t>(start),
                       path.end());
        rotated.insert(rotated.end(), path.begin(),
                       path.begin() + static_cast<std::ptrdiff_t>(start));
        if (rotated < best) {
            best = std::move(rotated);
        }
    }
    return best;
}

struct Snapshot {
    int failure_ = 0;
    int count_ = 0;
    int accepted_total_ = 0;
    int point_total_ = 0;
    std::vector<Path> paths_;
    std::vector<Path> quads_;
};

// The pinned OpenCV stripe tracer can write into a neighboring stripe. Keep
// this independent contour oracle sequential, restoring the caller's setting.
class SequentialOpenCv {
public:
    SequentialOpenCv() : previous_(cv::getNumThreads()) { cv::setNumThreads(1); }
    ~SequentialOpenCv() { cv::setNumThreads(previous_); }
    SequentialOpenCv(const SequentialOpenCv&) = delete;
    SequentialOpenCv& operator=(const SequentialOpenCv&) = delete;

private:
    int previous_;
};

class Harness {
public:
    Harness(int width, int height, DetectorConfig config, int point_capacity = 0)
        : config_(config) {
        const int points = point_capacity == 0 ? 4 * width * height : point_capacity;
        const std::size_t contour_bytes =
                aruco3cuda::detail::contour_candidate_workspace_bytes(width, height, points);
        const std::size_t candidate_bytes =
                aruco3cuda::detail::candidate_workspace_bytes(config, width, height);
        checked(this->workspace_.ensure_capacity(contour_bytes + candidate_bytes,
                                                 aruco3cuda::MemorySpace::kDevice));
        checked(aruco3cuda::detail::reserve_contour_candidates(width, height, points,
                                                               this->workspace_, &this->buffers_));
        checked(aruco3cuda::detail::reserve_candidates(config, width, height, this->workspace_,
                                                       &this->filter_, &this->candidates_));
        checked(this->image_.reserve(width, height));
        const auto& view = this->image_.view();
        this->binary_ = {const_cast<std::uint8_t*>(view.data_), view.width_px_, view.height_px_,
                         view.pitch_bytes_};
        checked(cudaStreamCreateWithFlags(&this->stream_, cudaStreamNonBlocking));
    }

    ~Harness() {
        if (this->graph_exec_ != nullptr) {
            static_cast<void>(cudaGraphExecDestroy(this->graph_exec_));
        }
        if (this->graph_ != nullptr) {
            static_cast<void>(cudaGraphDestroy(this->graph_));
        }
        if (this->stream_ != nullptr) {
            static_cast<void>(cudaStreamDestroy(this->stream_));
        }
    }

    Harness(const Harness&) = delete;
    Harness& operator=(const Harness&) = delete;

    void upload(const cv::Mat& binary) {
        checked(this->image_.upload(binary.data, binary.cols, binary.rows, binary.step));
    }

    void submit(bool append = false) {
        checked(aruco3cuda::detail::build_contour_candidates_async(
                this->binary_, this->config_, &this->buffers_, &this->candidates_, append,
                this->stream_));
    }

    void capture() {
        checked(cudaStreamBeginCapture(this->stream_, cudaStreamCaptureModeThreadLocal));
        this->submit();
        checked(cudaStreamEndCapture(this->stream_, &this->graph_));
#if CUDART_VERSION >= 12000
        checked(cudaGraphInstantiate(&this->graph_exec_, this->graph_, 0));
#else
        checked(cudaGraphInstantiate(&this->graph_exec_, this->graph_, nullptr, nullptr, 0));
#endif
    }

    void launch_graph() { checked(cudaGraphLaunch(this->graph_exec_, this->stream_)); }

    Snapshot snapshot() {
        checked(cudaStreamSynchronize(this->stream_));
        Snapshot result;
        result.failure_ = read_array(this->buffers_.failure_, 1U)[0];
        result.count_ = read_array(this->candidates_.count_, 1U)[0];
        result.accepted_total_ = read_array(this->candidates_.accepted_total_, 1U)[0];
        result.point_total_ = read_array(this->buffers_.point_total_, 1U)[0];
        if (result.failure_ != 0) {
            return result;
        }
        if (result.count_ < 0 || result.count_ > this->candidates_.capacity_ ||
            result.point_total_ < 0 || result.point_total_ > this->buffers_.point_capacity_) {
            throw std::runtime_error("invalid device contour counters");
        }
        const int roots = read_array(this->buffers_.root_count_, 1U)[0];
        if (roots < 0 || roots > this->buffers_.root_capacity_) {
            throw std::runtime_error("invalid compact root count");
        }
        const auto lengths = read_array(this->buffers_.lengths_, roots);
        const auto offsets = read_array(this->buffers_.offsets_, roots);
        const auto points = read_array(this->buffers_.points_, result.point_total_);
        for (std::size_t i = 0; i < lengths.size(); ++i) {
            const int length = std::abs(lengths[i]);
            if (length == 0) {
                continue;
            }
            if (offsets[i] < 0 || length > result.point_total_ - offsets[i]) {
                throw std::runtime_error("contour path exceeds initialized point pool");
            }
            Path path;
            for (int j = 0; j < length; ++j) {
                const ContourPoint& point = points[static_cast<std::size_t>(offsets[i] + j)];
                path.emplace_back(point.x_, point.y_);
            }
            result.paths_.push_back(path);
        }
        std::vector<std::vector<std::int32_t>> xs;
        std::vector<std::vector<std::int32_t>> ys;
        for (int c = 0; c < 4; ++c) {
            const int offset = c * this->candidates_.capacity_;
            xs.push_back(read_array(this->candidates_.corner_x_ + offset, result.count_));
            ys.push_back(read_array(this->candidates_.corner_y_ + offset, result.count_));
        }
        for (int i = 0; i < result.count_; ++i) {
            Path quad;
            for (int c = 0; c < 4; ++c) {
                quad.emplace_back(xs[c][i], ys[c][i]);
            }
            result.quads_.push_back(rotate_canonically(quad));
        }
        std::sort(result.paths_.begin(), result.paths_.end());
        std::sort(result.quads_.begin(), result.quads_.end());
        return result;
    }

    const ContourCandidateBuffers& buffers() const { return this->buffers_; }

private:
    DetectorConfig config_;
    aruco3cuda::Workspace workspace_;
    aruco3cuda::hybrid::DeviceImage image_;
    ContourCandidateBuffers buffers_;
    aruco3cuda::detail::CandidateFilterBuffers filter_;
    aruco3cuda::detail::DeviceCandidates candidates_;
    aruco3cuda::detail::ImagePlaneU8 binary_;
    cudaStream_t stream_ = nullptr;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
};

DetectorConfig complete_path_config() {
    DetectorConfig config;
    config.use_aruco3_detection_ = false;
    config.corner_refine_method_ = aruco3cuda::CornerRefineMethod::kNone;
    config.min_marker_perimeter_rate_ = 0.000001;
    config.max_marker_perimeter_rate_ = 8.0;
    return config;
}

Snapshot oracle(const cv::Mat& binary, const DetectorConfig& config) {
    const SequentialOpenCv sequential;
    Snapshot result;
    std::vector<std::vector<cv::Point>> paths;
    // The pinned OpenCV fast finder is not an isolated-ROI oracle unless cloned.
    cv::findContours(binary.clone(), paths, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    const int longest = std::max(binary.cols, binary.rows);
    const std::size_t minimum =
            config.use_aruco3_detection_
                    ? 4U * config.min_side_length_canonical_img_px_
                    : static_cast<std::size_t>(config.min_marker_perimeter_rate_ * longest);
    const std::size_t maximum =
            static_cast<std::size_t>(config.max_marker_perimeter_rate_ * longest);
    for (const auto& source : paths) {
        if (source.size() < minimum || source.size() > maximum) {
            continue;
        }
        Path path;
        for (const cv::Point& point : source) {
            path.emplace_back(point.x, point.y);
        }
        result.paths_.push_back(path);
    }
    std::vector<std::vector<cv::Point2f>> quads;
    std::vector<std::vector<cv::Point>> accepted_paths;
    aruco3cuda::hybrid::find_quad_candidates(binary.clone(), config, quads, accepted_paths);
    for (auto& source : quads) {
        aruco3cuda::hybrid::reorder_corners(source);
        Path quad;
        for (const cv::Point2f& point : source) {
            quad.emplace_back(static_cast<int>(point.x), static_cast<int>(point.y));
        }
        result.quads_.push_back(rotate_canonically(quad));
    }
    std::sort(result.paths_.begin(), result.paths_.end());
    std::sort(result.quads_.begin(), result.quads_.end());
    result.count_ = static_cast<int>(quads.size());
    return result;
}

void expect_matches(Harness& harness, const cv::Mat& binary, const DetectorConfig& config) {
    harness.upload(binary);
    harness.submit();
    const Snapshot actual = harness.snapshot();
    const Snapshot expected = oracle(binary, config);
    ASSERT_EQ(actual.failure_, 0);
    EXPECT_EQ(actual.paths_, expected.paths_);
    EXPECT_EQ(actual.quads_, expected.quads_);
    EXPECT_EQ(actual.count_, expected.count_);
    EXPECT_EQ(actual.accepted_total_, expected.count_);
}

cv::Mat connected_rings() {
    cv::Mat binary(180, 300, CV_8UC1, cv::Scalar(0));
    cv::rectangle(binary, cv::Point(20, 20), cv::Point(100, 120), cv::Scalar(255), 1);
    cv::rectangle(binary, cv::Point(160, 40), cv::Point(260, 140), cv::Scalar(255), 1);
    cv::line(binary, cv::Point(100, 70), cv::Point(160, 70), cv::Scalar(255), 1);
    return binary;
}

TEST(ContourCandidatesTest, connected_component_preserves_both_hole_quads) {
    if (!has_cuda_device())
        GTEST_SKIP();
    const cv::Mat binary = connected_rings();
    const DetectorConfig config;
    const Snapshot expected = oracle(binary, config);
    ASSERT_EQ(expected.paths_.size(), 3U);
    ASSERT_EQ(expected.quads_.size(), 2U);
    Harness harness(binary.cols, binary.rows, config);
    expect_matches(harness, binary, config);
}

TEST(ContourCandidatesTest, tiny_lines_diagonal_contacts_nested_holes_and_strided_rois) {
    if (!has_cuda_device())
        GTEST_SKIP();
    const DetectorConfig config = complete_path_config();
    for (const cv::Size size :
         {cv::Size(1, 1), cv::Size(1, 17), cv::Size(19, 1), cv::Size(33, 31)}) {
        Harness harness(size.width, size.height, config);
        cv::Mat backing(size.height + 4, size.width + 9, CV_8UC1, cv::Scalar(73));
        cv::Mat binary = backing(cv::Rect(3, 2, size.width, size.height));
        binary.setTo(0);
        expect_matches(harness, binary, config);
        binary.setTo(255);
        expect_matches(harness, binary, config);
        binary.setTo(0);
        cv::line(binary, cv::Point(0, 0), cv::Point(size.width - 1, size.height - 1),
                 cv::Scalar(255));
        expect_matches(harness, binary, config);
        if (size.width > 20 && size.height > 20) {
            binary.setTo(0);
            cv::rectangle(binary, cv::Point(1, 1), cv::Point(29, 27), cv::Scalar(255), 1);
            cv::rectangle(binary, cv::Point(6, 6), cv::Point(24, 22), cv::Scalar(255), 1);
            cv::line(binary, cv::Point(1, 1), cv::Point(6, 6), cv::Scalar(255), 1);
            expect_matches(harness, binary, config);
        }
    }
}

TEST(ContourCandidatesTest, random_masks_preserve_full_paths_and_quad_decisions) {
    if (!has_cuda_device())
        GTEST_SKIP();
    const DetectorConfig config = complete_path_config();
    Harness harness(39, 37, config);
    cv::RNG random(20261117U);
    cv::Mat binary(37, 39, CV_8UC1);
    for (int i = 0; i < 64; ++i) {
        SCOPED_TRACE(i);
        random.fill(binary, cv::RNG::UNIFORM, 0, 256);
        cv::threshold(binary, binary, 30 + (i % 8) * 28, 255, cv::THRESH_BINARY);
        expect_matches(harness, binary, config);
    }
}

TEST(ContourCandidatesTest, perimeter_and_approximation_boundaries_match_public_opencv) {
    if (!has_cuda_device())
        GTEST_SKIP();
    const cv::Mat binary = connected_rings();
    for (double minimum : {355.0 / 300.0, 356.0 / 300.0, 357.0 / 300.0, 396.0 / 300.0}) {
        for (double epsilon : {0.01, 0.03, 0.10}) {
            DetectorConfig config = complete_path_config();
            config.min_marker_perimeter_rate_ = minimum;
            config.polygonal_approx_accuracy_rate_ = epsilon;
            Harness harness(binary.cols, binary.rows, config);
            expect_matches(harness, binary, config);
        }
    }
}

TEST(ContourCandidatesTest, append_and_fresh_calls_preserve_counts_without_stale_candidates) {
    if (!has_cuda_device())
        GTEST_SKIP();
    const cv::Mat binary = connected_rings();
    const DetectorConfig config;
    Harness harness(binary.cols, binary.rows, config);
    harness.upload(binary);
    harness.submit();
    const Snapshot once = harness.snapshot();
    ASSERT_EQ(once.failure_, 0);
    ASSERT_EQ(once.count_, 2);
    harness.submit(true);
    const Snapshot twice = harness.snapshot();
    ASSERT_EQ(twice.failure_, 0);
    EXPECT_EQ(twice.count_, 4);
    EXPECT_EQ(twice.accepted_total_, 4);
    std::vector<Path> doubled = once.quads_;
    doubled.insert(doubled.end(), once.quads_.begin(), once.quads_.end());
    std::sort(doubled.begin(), doubled.end());
    EXPECT_EQ(twice.quads_, doubled);
    expect_matches(harness, cv::Mat::zeros(binary.size(), CV_8UC1), config);
    expect_matches(harness, binary, config);
}

TEST(ContourCandidatesTest, captured_graph_uses_current_pixels_on_each_launch) {
    if (!has_cuda_device())
        GTEST_SKIP();
    const cv::Mat binary = connected_rings();
    const DetectorConfig config;
    Harness harness(binary.cols, binary.rows, config);
    harness.upload(binary);
    harness.capture();
    for (int frame = 0; frame < 6; ++frame) {
        const cv::Mat current = frame % 2 == 0 ? binary : cv::Mat::zeros(binary.size(), CV_8UC1);
        harness.upload(current);
        harness.launch_graph();
        const Snapshot actual = harness.snapshot();
        ASSERT_EQ(actual.failure_, 0);
        EXPECT_EQ(actual.paths_, oracle(current, config).paths_);
        EXPECT_EQ(actual.quads_, oracle(current, config).quads_);
    }
}

TEST(ContourCandidatesTest, point_overflow_is_sticky_across_append_and_clears_on_new_frame) {
    if (!has_cuda_device())
        GTEST_SKIP();
    const cv::Mat binary = connected_rings();
    const DetectorConfig config;
    Harness harness(binary.cols, binary.rows, config, 16);
    EXPECT_NE(harness.buffers().count_map_,
              reinterpret_cast<std::int32_t*>(harness.buffers().points_));
    EXPECT_NE(harness.buffers().count_area_,
              reinterpret_cast<long long*>(harness.buffers().approximation_));
    harness.upload(binary);
    harness.submit();
    const Snapshot overflow = harness.snapshot();
    EXPECT_NE(overflow.failure_ & aruco3cuda::detail::kContourPointOverflow, 0);
    EXPECT_EQ(overflow.count_, 0);
    harness.upload(cv::Mat::zeros(binary.size(), CV_8UC1));
    harness.submit(true);
    const Snapshot appended = harness.snapshot();
    EXPECT_NE(appended.failure_ & aruco3cuda::detail::kContourPointOverflow, 0);
    EXPECT_EQ(appended.count_, 0);
    harness.submit(false);
    const Snapshot fresh = harness.snapshot();
    EXPECT_EQ(fresh.failure_, 0);
    EXPECT_EQ(fresh.count_, 0);
}

TEST(ContourCandidatesTest, candidate_overflow_is_reported_instead_of_silent_loss) {
    if (!has_cuda_device())
        GTEST_SKIP();
    const cv::Mat binary = connected_rings();
    DetectorConfig config;
    config.max_candidates_ = 1;
    config.max_markers_ = 1;
    ASSERT_EQ(config.validate(nullptr), Status::kOk);
    Harness harness(binary.cols, binary.rows, config);
    harness.upload(binary);
    harness.submit();
    const Snapshot result = harness.snapshot();
    EXPECT_NE(result.failure_ & aruco3cuda::detail::kContourCandidateOverflow, 0);
    EXPECT_EQ(result.count_, 0);
    EXPECT_EQ(result.accepted_total_, 2);
}

TEST(ContourCandidatesTest, interface_bound_preserves_checkerboard_and_full_foreground_paths) {
    if (!has_cuda_device())
        GTEST_SKIP();
    const DetectorConfig config = complete_path_config();
    for (const cv::Size size :
         {cv::Size(1, 1), cv::Size(1, 73), cv::Size(79, 1), cv::Size(32, 32), cv::Size(33, 31)}) {
        const int bound = 2 * size.width * size.height + size.width + size.height;
        Harness harness(size.width, size.height, config);
        ASSERT_EQ(harness.buffers().point_capacity_, bound);
        ASSERT_EQ(harness.buffers().jump_capacity_, bound);
        cv::Mat binary(size, CV_8UC1, cv::Scalar(255));
        for (int pattern = 0; pattern < 3; ++pattern) {
            if (pattern != 0) {
                for (int y = 0; y < size.height; ++y) {
                    for (int x = 0; x < size.width; ++x) {
                        binary.at<std::uint8_t>(y, x) = ((x + y + pattern) & 1) != 0 ? 255 : 0;
                    }
                }
            }
            expect_matches(harness, binary, config);
            EXPECT_LE(harness.snapshot().point_total_, bound);
            const int prepared_edges = read_array(harness.buffers().jump_count_, 1U)[0];
            EXPECT_GE(prepared_edges, 0);
            EXPECT_LE(prepared_edges, bound);
        }
        Harness limited(size.width, size.height, config, 1);
        EXPECT_EQ(limited.buffers().point_capacity_, 1);
    }
}

TEST(ContourCandidatesTest, parallel_count_preserves_large_component_and_dense_holes) {
    if (!has_cuda_device())
        GTEST_SKIP();
    DetectorConfig config = complete_path_config();
    config.max_candidates_ = 32768;
    cv::Mat binary(129, 131, CV_8UC1, cv::Scalar(255));
    Harness harness(binary.cols, binary.rows, config);
    EXPECT_EQ(harness.buffers().count_map_,
              reinterpret_cast<std::int32_t*>(harness.buffers().points_));
    EXPECT_EQ(harness.buffers().count_area_,
              reinterpret_cast<long long*>(harness.buffers().approximation_));
    expect_matches(harness, binary, config);
    for (int y = 0; y < binary.rows; ++y) {
        for (int x = 0; x < binary.cols; ++x) {
            binary.at<std::uint8_t>(y, x) = ((x + y) & 1) == 0 ? 255U : 0U;
        }
    }
    expect_matches(harness, binary, config);
}

TEST(ContourCandidatesTest, materialization_preserves_warp_boundaries_and_crack_lengths) {
    if (!has_cuda_device())
        GTEST_SKIP();
    const DetectorConfig config = complete_path_config();
    for (const int length : {1, 15, 16, 31, 32, 63, 64, 65}) {
        for (const bool vertical : {false, true}) {
            cv::Mat binary(vertical ? length : 1, vertical ? 1 : length, CV_8UC1, cv::Scalar(255));
            Harness harness(binary.cols, binary.rows, config);
            expect_matches(harness, binary, config);
            ASSERT_EQ(read_array(harness.buffers().root_count_, 1U)[0], 1);
        }
    }
}

TEST(ContourCandidatesTest, sizing_rejects_invalid_dimensions_and_integer_overflow) {
    using aruco3cuda::detail::contour_candidate_workspace_bytes;
    EXPECT_EQ(contour_candidate_workspace_bytes(0, 1, 1), 0U);
    EXPECT_EQ(contour_candidate_workspace_bytes(1, -1, 1), 0U);
    EXPECT_EQ(contour_candidate_workspace_bytes(1, 1, 0), 0U);
    EXPECT_EQ(contour_candidate_workspace_bytes(std::numeric_limits<int>::max(), 2, 1), 0U);
    EXPECT_EQ(contour_candidate_workspace_bytes(std::numeric_limits<int>::max() / 4, 1, 1), 0U);
    EXPECT_NE(contour_candidate_workspace_bytes(1, 1, 4), 0U);
}

}  // namespace
