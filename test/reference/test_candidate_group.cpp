// SPDX-License-Identifier: Apache-2.0
//
// Verifies the merging of nearby candidates.
//
// Varying the binarization window yields slightly different candidates from the same
// marker. Which one is kept determines the corner positions, so keeping the one with
// the largest perimeter is pinned down here. The known difference from the CPU
// reference (how chained candidates are treated) is stated explicitly as well.
#include "candidate_group.hpp"

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"
#include "aruco3cuda/workspace.hpp"
#include "candidate_filter.hpp"
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

/// A single candidate for the tests. Its corners form an axis-aligned square.
struct InputQuad {
    int x_ = 0;
    int y_ = 0;
    int side_ = 0;
    int label_ = 0;
};

/// Holds the input and the output of one merge run together.
class GroupRun {
public:
    GroupRun() = default;
    GroupRun(const GroupRun&) = delete;
    GroupRun& operator=(const GroupRun&) = delete;

    /// Uploads the candidates to the device and merges them.
    bool run(const std::vector<InputQuad>& quads, const DetectorConfig& config) {
        const std::size_t bytes = aruco3cuda::detail::candidate_workspace_bytes(config, 64, 64) +
                                  aruco3cuda::detail::candidate_group_workspace_bytes(config);
        if (bytes == 0U ||
            this->workspace_.ensure_capacity(bytes, MemorySpace::kDevice, nullptr) != Status::kOk) {
            return false;
        }
        this->workspace_.reset();
        aruco3cuda::detail::CandidateFilterBuffers filter;
        aruco3cuda::detail::DeviceCandidates input;
        aruco3cuda::detail::CandidateGroupBuffers groups;
        aruco3cuda::detail::DeviceCandidates grouped;
        if (aruco3cuda::detail::reserve_candidates(config, 64, 64, this->workspace_, &filter,
                                                   &input) != Status::kOk ||
            aruco3cuda::detail::reserve_candidate_groups(config, this->workspace_, &groups,
                                                         &grouped) != Status::kOk) {
            return false;
        }
        if (!upload(quads, input)) {
            return false;
        }
        if (aruco3cuda::detail::build_candidate_groups_async(input, config, &groups, &grouped,
                                                             nullptr) != Status::kOk) {
            return false;
        }
        int count = 0;
        if (aruco3cuda::detail::read_candidate_count(grouped, &count, nullptr) != Status::kOk) {
            return false;
        }
        this->count_ = count;
        return download(grouped);
    }

    int count() const { return this->count_; }
    const std::vector<InputQuad>& results() const { return this->results_; }

private:
    /// Writes the candidates into the device arrays, expanding each into a square.
    static bool upload(const std::vector<InputQuad>& quads,
                       const aruco3cuda::detail::DeviceCandidates& input) {
        const auto count = static_cast<int>(quads.size());
        const auto capacity = static_cast<std::size_t>(input.capacity_);
        std::vector<std::int32_t> corner_x(capacity * kQuadCornerCount, 0);
        std::vector<std::int32_t> corner_y(capacity * kQuadCornerCount, 0);
        std::vector<std::int32_t> label(capacity, 0);
        std::vector<std::int32_t> perimeter(capacity, 0);
        for (std::size_t i = 0; i < quads.size(); ++i) {
            const InputQuad& quad = quads[i];
            const int xs[kQuadCornerCount] = {quad.x_, quad.x_ + quad.side_, quad.x_ + quad.side_,
                                              quad.x_};
            const int ys[kQuadCornerCount] = {quad.y_, quad.y_, quad.y_ + quad.side_,
                                              quad.y_ + quad.side_};
            for (int corner = 0; corner < kQuadCornerCount; ++corner) {
                const std::size_t index = (static_cast<std::size_t>(corner) * capacity) + i;
                corner_x[index] = xs[corner];
                corner_y[index] = ys[corner];
            }
            label[i] = quad.label_;
            perimeter[i] = 4 * quad.side_;
        }
        const std::size_t corner_bytes = corner_x.size() * sizeof(std::int32_t);
        const std::size_t plain_bytes = label.size() * sizeof(std::int32_t);
        return cudaMemcpy(input.corner_x_, corner_x.data(), corner_bytes, cudaMemcpyHostToDevice) ==
                       cudaSuccess &&
               cudaMemcpy(input.corner_y_, corner_y.data(), corner_bytes, cudaMemcpyHostToDevice) ==
                       cudaSuccess &&
               cudaMemcpy(input.label_, label.data(), plain_bytes, cudaMemcpyHostToDevice) ==
                       cudaSuccess &&
               cudaMemcpy(input.perimeter_, perimeter.data(), plain_bytes,
                          cudaMemcpyHostToDevice) == cudaSuccess &&
               cudaMemcpy(input.count_, &count, sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess;
    }

    bool download(const aruco3cuda::detail::DeviceCandidates& grouped) {
        const auto count = static_cast<std::size_t>(this->count_);
        this->results_.assign(count, InputQuad{});
        if (count == 0U) {
            return true;
        }
        const auto capacity = static_cast<std::size_t>(grouped.capacity_);
        std::vector<std::int32_t> corner_x(count * kQuadCornerCount);
        std::vector<std::int32_t> label(count);
        std::vector<std::int32_t> perimeter(count);
        const std::size_t row_bytes = count * sizeof(std::int32_t);
        for (int corner = 0; corner < kQuadCornerCount; ++corner) {
            const std::size_t offset = static_cast<std::size_t>(corner) * capacity;
            const std::size_t destination = static_cast<std::size_t>(corner) * count;
            if (cudaMemcpy(corner_x.data() + destination, grouped.corner_x_ + offset, row_bytes,
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
                return false;
            }
        }
        if (cudaMemcpy(label.data(), grouped.label_, row_bytes, cudaMemcpyDeviceToHost) !=
                    cudaSuccess ||
            cudaMemcpy(perimeter.data(), grouped.perimeter_, row_bytes, cudaMemcpyDeviceToHost) !=
                    cudaSuccess) {
            return false;
        }
        for (std::size_t i = 0; i < count; ++i) {
            this->results_[i].x_ = corner_x[i];
            this->results_[i].side_ = perimeter[i] / 4;
            this->results_[i].label_ = label[i];
        }
        return true;
    }

    Workspace workspace_;
    std::vector<InputQuad> results_;
    int count_ = 0;
};

/// Configuration with a small candidate limit, to keep the test allocations small.
DetectorConfig small_config() {
    DetectorConfig config;
    config.max_candidates_ = 64;
    return config;
}

// Happy path: nearby candidates merge into one, and the largest perimeter survives.
TEST(CandidateGroupTest, keeps_largest_perimeter_in_group) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    // Three candidates from the same marker at different windows. Their edge lengths
    // differ slightly.
    const std::vector<InputQuad> quads = {{100, 100, 60, 0}, {101, 101, 58, 1}, {99, 99, 62, 2}};
    GroupRun run;
    ASSERT_TRUE(run.run(quads, small_config()));
    ASSERT_EQ(run.count(), 1);
    EXPECT_EQ(run.results()[0].side_, 62);
    EXPECT_EQ(run.results()[0].label_, 2);
}

// Happy path: distant candidates are not merged.
TEST(CandidateGroupTest, keeps_distant_candidates_separate) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    const std::vector<InputQuad> quads = {
            {20, 20, 40, 0}, {200, 20, 40, 1}, {20, 200, 40, 2}, {200, 200, 40, 3}};
    GroupRun run;
    ASSERT_TRUE(run.run(quads, small_config()));
    EXPECT_EQ(run.count(), 4);
}

// Happy path: with several groups present, the representative of each one survives.
TEST(CandidateGroupTest, handles_multiple_groups) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    const std::vector<InputQuad> quads = {{20, 20, 40, 0},  {21, 21, 38, 1},   {200, 20, 50, 2},
                                          {201, 21, 48, 3}, {110, 200, 44, 4}, {111, 201, 42, 5}};
    GroupRun run;
    ASSERT_TRUE(run.run(quads, small_config()));
    ASSERT_EQ(run.count(), 3);
    std::vector<int> sides;
    sides.reserve(run.results().size());
    for (const InputQuad& item : run.results()) {
        sides.push_back(item.side_);
    }
    // The larger candidate of each group survives, ordered by descending perimeter.
    EXPECT_EQ(sides, (std::vector<int>{50, 44, 40}));
}

// Boundary case: a single candidate passes through unchanged.
TEST(CandidateGroupTest, single_candidate_passes_through) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    GroupRun run;
    ASSERT_TRUE(run.run({{50, 50, 40, 7}}, small_config()));
    ASSERT_EQ(run.count(), 1);
    EXPECT_EQ(run.results()[0].label_, 7);
}

// Boundary case: with no candidates, the result is empty as well.
TEST(CandidateGroupTest, empty_input_produces_no_group) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    GroupRun run;
    ASSERT_TRUE(run.run({}, small_config()));
    EXPECT_EQ(run.count(), 0);
}

// Known difference: three candidates chained by proximity merge into one.
//
// The CPU reference does not merge two candidates that already belong to different
// groups. In this arrangement, where the two ends are not directly close to each
// other, the CPU reference keeps two candidates while this implementation merges them
// into one. The size of that difference is measured in the comparison against the CPU
// reference.
TEST(CandidateGroupTest, known_difference_chained_candidates_merge) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    // An arrangement where each neighbor is close but the two ends are far apart.
    const std::vector<InputQuad> quads = {{100, 100, 60, 0}, {112, 100, 60, 1}, {124, 100, 60, 2}};
    GroupRun run;
    ASSERT_TRUE(run.run(quads, small_config()));
    EXPECT_EQ(run.count(), 1);
}

// Happy path: the same input yields the same result.
TEST(CandidateGroupTest, results_are_deterministic) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    std::vector<InputQuad> quads;
    quads.reserve(24);
    for (int i = 0; i < 12; ++i) {
        quads.push_back({20 + (i * 37), 30 + ((i % 4) * 45), 30 + i, i});
        quads.push_back({21 + (i * 37), 31 + ((i % 4) * 45), 29 + i, i + 100});
    }
    GroupRun first;
    GroupRun second;
    ASSERT_TRUE(first.run(quads, small_config()));
    ASSERT_TRUE(second.run(quads, small_config()));
    ASSERT_EQ(first.count(), second.count());
    for (std::size_t i = 0; i < first.results().size(); ++i) {
        EXPECT_EQ(first.results()[i].label_, second.results()[i].label_);
        EXPECT_EQ(first.results()[i].side_, second.results()[i].side_);
    }
}

// Failure path: nothing runs when the arguments are invalid.
TEST(CandidateGroupTest, rejects_invalid_arguments) {
    Workspace workspace;
    const DetectorConfig config;
    aruco3cuda::detail::CandidateGroupBuffers buffers;
    aruco3cuda::detail::DeviceCandidates grouped;
    EXPECT_EQ(aruco3cuda::detail::reserve_candidate_groups(config, workspace, nullptr, &grouped),
              Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::detail::reserve_candidate_groups(config, workspace, &buffers, nullptr),
              Status::kInvalidArgument);
    EXPECT_NE(aruco3cuda::detail::reserve_candidate_groups(config, workspace, &buffers, &grouped),
              Status::kOk);

    aruco3cuda::detail::DeviceCandidates input;
    EXPECT_EQ(aruco3cuda::detail::build_candidate_groups_async(input, config, nullptr, &grouped,
                                                               nullptr),
              Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::detail::build_candidate_groups_async(input, config, &buffers, &grouped,
                                                               nullptr),
              Status::kInvalidArgument);

    DetectorConfig zero_capacity = config;
    zero_capacity.max_candidates_ = 0;
    EXPECT_EQ(aruco3cuda::detail::candidate_group_workspace_bytes(zero_capacity), 0U);
}

}  // namespace
