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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"
#include "aruco3cuda/workspace.hpp"
#include "candidate_filter.hpp"
#include "candidate_retry.hpp"
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
    int contour_length_ = 0;
    std::vector<std::array<int, 2>> corners_ = {};
};

/// Holds the input and the output of one merge run together.
class GroupRun {
public:
    GroupRun() = default;
    GroupRun(const GroupRun&) = delete;
    GroupRun& operator=(const GroupRun&) = delete;

    /// Uploads the candidates to the device and merges them.
    bool run(const std::vector<InputQuad>& quads, const DetectorConfig& config,
             const std::vector<int>* primary_ids = nullptr,
             const std::vector<int>* retry_ids = nullptr, int width_px = 0, int height_px = 0) {
        const std::size_t bytes =
                aruco3cuda::detail::candidate_workspace_bytes(config, 64, 64) +
                aruco3cuda::detail::candidate_group_workspace_bytes(config) +
                (primary_ids == nullptr
                         ? 0U
                         : aruco3cuda::detail::candidate_retry_workspace_bytes(config) +
                                   aruco3cuda::detail::match_workspace_bytes(config));
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
                                                             nullptr, width_px,
                                                             height_px) != Status::kOk) {
            return false;
        }
        int count = 0;
        if (aruco3cuda::detail::read_candidate_count(grouped, &count, nullptr) != Status::kOk) {
            return false;
        }
        this->count_ = count;
        if (primary_ids != nullptr) {
            aruco3cuda::detail::CandidateRetryBuffers retries;
            aruco3cuda::detail::MatchBuffers primary;
            if (aruco3cuda::detail::reserve_candidate_retries(config, workspace_, &retries) !=
                        Status::kOk ||
                aruco3cuda::detail::reserve_matches(config, workspace_, &primary) != Status::kOk ||
                static_cast<int>(primary_ids->size()) != count ||
                !upload_matches(primary, *primary_ids)) {
                return false;
            }
            if (aruco3cuda::detail::prepare_candidate_retries_async(input, grouped, primary, groups,
                                                                    config, 6, &retries,
                                                                    nullptr) != Status::kOk) {
                return false;
            }
            int retry_count = 0;
            if (aruco3cuda::detail::read_candidate_count(retries.candidates_, &retry_count,
                                                         nullptr) != Status::kOk)
                return false;
            retry_labels_.resize(retry_count);
            retry_lengths_.resize(retry_count);
            if (retry_count > 0 &&
                (cudaMemcpy(retry_labels_.data(), retries.candidates_.label_,
                            retry_count * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess ||
                 cudaMemcpy(retry_lengths_.data(), retries.candidates_.perimeter_,
                            retry_count * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess))
                return false;
            if (retry_ids != nullptr) {
                if (static_cast<int>(retry_ids->size()) != retry_count ||
                    !upload_matches(retries.matches_, *retry_ids) ||
                    aruco3cuda::detail::resolve_candidate_retries_async(retries, &grouped, &primary,
                                                                        nullptr) != Status::kOk)
                    return false;
            }
            matched_ids_.resize(count);
            if (count > 0 && cudaMemcpy(matched_ids_.data(), primary.ids_, count * sizeof(int),
                                        cudaMemcpyDeviceToHost) != cudaSuccess)
                return false;
        }
        return download(grouped);
    }

    int count() const { return this->count_; }
    const std::vector<InputQuad>& results() const { return this->results_; }
    const std::vector<int>& retry_labels() const { return retry_labels_; }
    const std::vector<int>& retry_lengths() const { return retry_lengths_; }
    const std::vector<int>& matched_ids() const { return matched_ids_; }

private:
    static bool upload_matches(const aruco3cuda::detail::MatchBuffers& matches,
                               const std::vector<int>& ids) {
        if (ids.empty())
            return true;
        const std::vector<int> zeros(ids.size(), 0);
        const auto bytes = ids.size() * sizeof(int);
        return cudaMemcpy(matches.ids_, ids.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
               cudaMemcpy(matches.rotations_, zeros.data(), bytes, cudaMemcpyHostToDevice) ==
                       cudaSuccess &&
               cudaMemcpy(matches.distances_, zeros.data(), bytes, cudaMemcpyHostToDevice) ==
                       cudaSuccess;
    }
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
                corner_x[index] = quad.corners_.empty() ? xs[corner] : quad.corners_[corner][0];
                corner_y[index] = quad.corners_.empty() ? ys[corner] : quad.corners_[corner][1];
            }
            label[i] = quad.label_;
            perimeter[i] = quad.contour_length_ == 0 ? 4 * quad.side_ : quad.contour_length_;
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
            this->results_[i].side_ = corner_x[count + i] - corner_x[i];
            this->results_[i].label_ = label[i];
            this->results_[i].contour_length_ = perimeter[i];
        }
        return true;
    }

    Workspace workspace_;
    std::vector<InputQuad> results_;
    std::vector<int> retry_labels_;
    std::vector<int> retry_lengths_;
    std::vector<int> matched_ids_;
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

// Both paths have 202 points. Ranking contour length would keep the inner hole
// and make two refined projective_9 corners worse by several pixels.
TEST(CandidateGroupTest, equal_contour_lengths_choose_larger_projective_quad) {
    if (!has_cuda_device()) {
        GTEST_SKIP();
    }
    const InputQuad inner{0, 0, 0, 7, 202, {{159, 73}, {206, 118}, {160, 163}, {112, 129}}};
    const InputQuad outer{0, 0, 0, 9, 202, {{159, 71}, {207, 118}, {160, 164}, {111, 129}}};
    GroupRun run;
    ASSERT_TRUE(run.run({inner, outer}, small_config()));
    ASSERT_EQ(run.count(), 1);
    EXPECT_EQ(run.results()[0].label_, 9);
    EXPECT_EQ(run.results()[0].contour_length_, 202);
}

TEST(CandidateGroupTest, ranking_preserves_winning_contour_length_for_sampling) {
    if (!has_cuda_device()) {
        GTEST_SKIP();
    }
    GroupRun run;
    ASSERT_TRUE(run.run({{101, 101, 58, 0, 900}, {100, 100, 60, 1, 240}}, small_config()));
    ASSERT_EQ(run.count(), 1);
    EXPECT_EQ(run.results()[0].label_, 1);
    EXPECT_EQ(run.results()[0].side_, 60);
    EXPECT_EQ(run.results()[0].contour_length_, 240);
}

TEST(CandidateGroupTest, proximity_threshold_uses_quad_perimeter_not_contour_length) {
    if (!has_cuda_device()) {
        GTEST_SKIP();
    }
    // At the default distance rate .125, the quad threshold is 20 pixels.
    // The synthetic contour count would incorrectly raise it to 125 pixels.
    GroupRun run;
    ASSERT_TRUE(run.run({{20, 20, 40, 0, 1000}, {50, 20, 40, 1, 1000}}, small_config()));
    EXPECT_EQ(run.count(), 2);
}

TEST(CandidateGroupTest, quad_proximity_threshold_is_strict_at_float_boundary) {
    if (!has_cuda_device()) {
        GTEST_SKIP();
    }
    const std::vector<InputQuad> quads{{20, 20, 40, 0, 1000}, {40, 20, 40, 1, 1000}};
    DetectorConfig config = small_config();
    GroupRun run;
    config.min_marker_distance_rate_ = std::nextafter(0.125F, 0.0F);
    ASSERT_TRUE(run.run(quads, config));
    EXPECT_EQ(run.count(), 2);
    config.min_marker_distance_rate_ = 0.125;
    ASSERT_TRUE(run.run(quads, config));
    EXPECT_EQ(run.count(), 2);
    config.min_marker_distance_rate_ = std::nextafter(0.125F, 1.0F);
    ASSERT_TRUE(run.run(quads, config));
    EXPECT_EQ(run.count(), 1);
}

TEST(CandidateGroupTest, equal_quad_perimeters_preserve_input_order_despite_contour_lengths) {
    if (!has_cuda_device()) {
        GTEST_SKIP();
    }
    GroupRun run;
    ASSERT_TRUE(run.run({{20, 20, 40, 7, 160}, {21, 20, 40, 9, 1000}}, small_config()));
    ASSERT_EQ(run.count(), 1);
    EXPECT_EQ(run.results()[0].label_, 7);
    EXPECT_EQ(run.results()[0].contour_length_, 160);
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

TEST(CandidateGroupTest, dense_cycles_and_long_chains_have_deterministic_roots) {
    if (!has_cuda_device()) {
        GTEST_SKIP();
    }
    std::mt19937 random(20261117U);
    for (bool cycle_grid : {false, true}) {
        std::vector<InputQuad> quads;
        for (int i = 0; i < 64; ++i) {
            const int column = cycle_grid ? i % 8 : i;
            const int row = cycle_grid ? i / 8 : 0;
            quads.push_back({100 + column * 12, 100 + row * 12, 60, i, 240 + i});
        }
        GroupRun run;
        for (int repetition = 0; repetition < 32; ++repetition) {
            SCOPED_TRACE(cycle_grid);
            SCOPED_TRACE(repetition);
            std::shuffle(quads.begin(), quads.end(), random);
            ASSERT_TRUE(run.run(quads, small_config()));
            ASSERT_EQ(run.count(), 1);
            // Equal Euclidean perimeters choose the first input, independent
            // of atomic union order and the deliberately varying path counts.
            EXPECT_EQ(run.results()[0].label_, quads.front().label_);
            EXPECT_EQ(run.results()[0].contour_length_, quads.front().contour_length_);
        }
    }
}

// Failure path: nothing runs when the arguments are invalid.
TEST(CandidateGroupTest, retry_skips_successes_and_preserves_representative_level) {
    if (!has_cuda_device())
        GTEST_SKIP();
    const std::vector<InputQuad> quads{{100, 100, 100, 10, 900},
                                       {105, 105, 90, 11, 700},
                                       {110, 110, 80, 12, 500},
                                       {111, 111, 79, 13, 400}};
    const std::vector<int> success{7};
    const std::vector<int> empty;
    GroupRun run;
    ASSERT_TRUE(run.run(quads, small_config(), &success, &empty));
    EXPECT_TRUE(run.retry_labels().empty());
    EXPECT_EQ(run.matched_ids(), success);
    EXPECT_EQ(run.results()[0].side_, 100);
    const std::vector<int> failure{-1};
    const std::vector<int> alternatives{42, 43};
    ASSERT_TRUE(run.run(quads, small_config(), &failure, &alternatives));
    EXPECT_EQ(run.retry_labels(), (std::vector<int>{11, 12}));
    EXPECT_EQ(run.retry_lengths(), (std::vector<int>{900, 900}));
    EXPECT_EQ(run.matched_ids(), (std::vector<int>{42}));
    EXPECT_EQ(run.results()[0].x_, 105);
    EXPECT_EQ(run.results()[0].side_, 90);
}

TEST(CandidateGroupTest, retry_uses_first_success_and_leaves_all_failed_groups_unchanged) {
    if (!has_cuda_device())
        GTEST_SKIP();
    const std::vector<InputQuad> quads{{100, 100, 100, 10}, {105, 105, 90, 11}, {110, 110, 80, 12}};
    const std::vector<int> failure{-1};
    const std::vector<int> alternatives{-1, 43};
    GroupRun run;
    ASSERT_TRUE(run.run(quads, small_config(), &failure, &alternatives));
    EXPECT_EQ(run.matched_ids(), (std::vector<int>{43}));
    EXPECT_EQ(run.results()[0].side_, 80);
    const std::vector<int> failed_alternatives{-1, -1};
    ASSERT_TRUE(run.run(quads, small_config(), &failure, &failed_alternatives));
    EXPECT_EQ(run.matched_ids(), failure);
    EXPECT_EQ(run.results()[0].side_, 100);
}

TEST(CandidateGroupTest, retry_has_no_small_per_group_limit) {
    if (!has_cuda_device())
        GTEST_SKIP();
    std::vector<InputQuad> quads;
    for (int i = 0; i < 32; ++i)
        quads.push_back({100 + i * 6, 100 + i * 6, 400 - i * 12, i});
    DetectorConfig config = small_config();
    config.min_group_distance_ = 0.01;
    const std::vector<int> failure{-1};
    std::vector<int> alternatives(31, -1);
    alternatives.back() = 91;
    GroupRun run;
    ASSERT_TRUE(run.run(quads, config, &failure, &alternatives));
    ASSERT_EQ(run.retry_labels().size(), 31U);
    EXPECT_EQ(run.retry_labels().back(), 31);
    EXPECT_EQ(run.matched_ids(), (std::vector<int>{91}));
    EXPECT_EQ(run.results()[0].side_, 28);
}

TEST(CandidateGroupTest, retry_rejects_invalid_storage) {
    Workspace workspace;
    DetectorConfig config;
    aruco3cuda::detail::CandidateRetryBuffers retries;
    EXPECT_EQ(aruco3cuda::detail::reserve_candidate_retries(config, workspace, nullptr),
              Status::kInvalidArgument);
    EXPECT_NE(aruco3cuda::detail::reserve_candidate_retries(config, workspace, &retries),
              Status::kOk);
    config.max_candidates_ = 0;
    EXPECT_EQ(aruco3cuda::detail::candidate_retry_workspace_bytes(config), 0U);
    aruco3cuda::detail::DeviceCandidates grouped;
    aruco3cuda::detail::MatchBuffers primary;
    EXPECT_EQ(aruco3cuda::detail::resolve_candidate_retries_async(retries, &grouped, &primary,
                                                                  nullptr),
              Status::kInvalidArgument);
}

// These invalid calls must return before any CUDA submission. Host sentinel
// addresses make the boundary test usable on hosts without a CUDA device.
TEST(CandidateGroupTest, retry_rejects_each_missing_region_and_inconsistent_capacity) {
    using namespace aruco3cuda::detail;
    std::int32_t sentinel = 0;
    float perimeter = 0.0F;
    const DeviceCandidates candidate{&sentinel, &sentinel, &sentinel, &sentinel,
                                     &sentinel, &sentinel, 1};
    DeviceCandidates raw = candidate;
    DeviceCandidates grouped = candidate;
    MatchBuffers primary{&sentinel, &sentinel, &sentinel, 1};
    CandidateGroupBuffers groups;
    groups.quad_perimeter_ = &perimeter;
    groups.rank_ = groups.order_ = groups.parent_ = groups.selected_ = groups.offsets_ = &sentinel;
    groups.capacity_ = 1;
    CandidateRetryBuffers retries;
    retries.candidates_ = candidate;
    retries.matches_ = primary;
    retries.owners_ = retries.flags_ = retries.offsets_ = retries.chosen_ = &sentinel;
    retries.scan_ = {&sentinel, &sentinel, 1, 1};
    retries.capacity_ = 1;
    DetectorConfig config;
    auto prepare = [&]() {
        return prepare_candidate_retries_async(raw, grouped, primary, groups, config, 4, &retries,
                                               nullptr);
    };
    auto resolve = [&]() {
        return resolve_candidate_retries_async(retries, &grouped, &primary, nullptr);
    };
    std::int32_t** shared[] = {&grouped.corner_x_,
                               &grouped.corner_y_,
                               &grouped.label_,
                               &grouped.perimeter_,
                               &grouped.count_,
                               &grouped.accepted_total_,
                               &primary.ids_,
                               &primary.rotations_,
                               &primary.distances_,
                               &retries.owners_,
                               &retries.flags_,
                               &retries.offsets_,
                               &retries.chosen_,
                               &retries.candidates_.corner_x_,
                               &retries.candidates_.corner_y_,
                               &retries.candidates_.label_,
                               &retries.candidates_.perimeter_,
                               &retries.candidates_.count_,
                               &retries.candidates_.accepted_total_,
                               &retries.matches_.ids_,
                               &retries.matches_.rotations_,
                               &retries.matches_.distances_,
                               &retries.scan_.total_,
                               &retries.scan_.block_offsets_};
    for (auto** region : shared) {
        *region = nullptr;
        EXPECT_EQ(prepare(), Status::kInvalidArgument);
        EXPECT_EQ(resolve(), Status::kInvalidArgument);
        *region = &sentinel;
    }
    std::int32_t** prepare_only[] = {&raw.corner_x_,    &raw.corner_y_,  &raw.label_,
                                     &raw.perimeter_,   &raw.count_,     &raw.accepted_total_,
                                     &groups.rank_,     &groups.order_,  &groups.parent_,
                                     &groups.selected_, &groups.offsets_};
    for (auto** region : prepare_only) {
        *region = nullptr;
        EXPECT_EQ(prepare(), Status::kInvalidArgument);
        *region = &sentinel;
    }
    groups.quad_perimeter_ = nullptr;
    EXPECT_EQ(prepare(), Status::kInvalidArgument);
    groups.quad_perimeter_ = &perimeter;
    int* capacities[] = {&grouped.capacity_,          &primary.capacity_,
                         &retries.capacity_,          &retries.candidates_.capacity_,
                         &retries.matches_.capacity_, &retries.scan_.capacity_,
                         &retries.scan_.block_count_};
    for (int* capacity : capacities) {
        *capacity = 0;
        EXPECT_EQ(prepare(), Status::kInvalidArgument);
        EXPECT_EQ(resolve(), Status::kInvalidArgument);
        *capacity = 1;
    }
    grouped.capacity_ = 2;
    EXPECT_EQ(prepare(), Status::kInvalidArgument);
    EXPECT_EQ(resolve(), Status::kInvalidArgument);
    grouped.capacity_ = 1;
    raw.capacity_ = 0;
    EXPECT_EQ(prepare(), Status::kInvalidArgument);
    raw.capacity_ = 2;
    EXPECT_EQ(prepare(), Status::kInvalidArgument);
    raw.capacity_ = 1;
    groups.capacity_ = 0;
    EXPECT_EQ(prepare(), Status::kInvalidArgument);
    groups.capacity_ = 1;
    retries.candidates_.capacity_ = 2;
    EXPECT_EQ(prepare(), Status::kInvalidArgument);
    EXPECT_EQ(resolve(), Status::kInvalidArgument);
    retries.candidates_.capacity_ = 1;
    for (double value :
         {-1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::max()}) {
        config.min_group_distance_ = value;
        EXPECT_EQ(prepare(), Status::kInvalidArgument);
    }
}

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

TEST(CandidateGroupTest, border_margin_checks_all_edges_and_exact_boundary) {
    if (!has_cuda_device())
        GTEST_SKIP();
    DetectorConfig config = small_config();
    config.min_distance_to_border_px_ = 32;
    GroupRun run;
    for (const InputQuad& quad : std::vector<InputQuad>{
                 {32, 70, 40, 1}, {183, 70, 40, 2}, {70, 32, 40, 3}, {70, 119, 40, 4}}) {
        ASSERT_TRUE(run.run({quad}, config, nullptr, nullptr, 256, 192));
        ASSERT_EQ(run.count(), 1);
        InputQuad outside = quad;
        if (quad.label_ == 1)
            --outside.x_;
        if (quad.label_ == 2)
            ++outside.x_;
        if (quad.label_ == 3)
            --outside.y_;
        if (quad.label_ == 4)
            ++outside.y_;
        ASSERT_TRUE(run.run({outside}, config, nullptr, nullptr, 256, 192));
        EXPECT_EQ(run.count(), 0) << quad.label_;
    }
    config.min_distance_to_border_px_ = 0;
    ASSERT_TRUE(run.run({{0, 0, 40, 7}, {215, 151, 40, 8}}, config, nullptr, nullptr, 256, 192));
    EXPECT_EQ(run.count(), 2);
    config.min_distance_to_border_px_ = 128;
    ASSERT_TRUE(run.run({{100, 70, 40, 7}}, config, nullptr, nullptr, 256, 192));
    EXPECT_EQ(run.count(), 0);
}

TEST(CandidateGroupTest, border_margin_rejects_group_without_promoting_alternative) {
    if (!has_cuda_device())
        GTEST_SKIP();
    DetectorConfig config = small_config();
    config.min_distance_to_border_px_ = 32;
    GroupRun run;
    const std::vector<InputQuad> quads{{30, 70, 60, 1}, {33, 73, 54, 2}};
    ASSERT_TRUE(run.run(quads, config, nullptr, nullptr, 256, 192));
    EXPECT_EQ(run.count(), 0);
    ASSERT_TRUE(run.run({quads[1]}, config, nullptr, nullptr, 256, 192));
    EXPECT_EQ(run.count(), 1);
    ASSERT_TRUE(run.run(quads, config));
    ASSERT_EQ(run.count(), 1);
    EXPECT_EQ(run.results()[0].label_, 1);
    // Border rejection concerns the representative, not every alternative.
    ASSERT_TRUE(run.run({{32, 40, 60, 3}, {29, 43, 54, 4}}, config, nullptr, nullptr, 256, 192));
    ASSERT_EQ(run.count(), 1);
    EXPECT_EQ(run.results()[0].label_, 3);
}

TEST(CandidateGroupTest, border_margin_rejects_invalid_dimensions) {
    if (!has_cuda_device())
        GTEST_SKIP();
    GroupRun run;
    EXPECT_FALSE(run.run({}, small_config(), nullptr, nullptr, -1, 192));
    EXPECT_FALSE(run.run({}, small_config(), nullptr, nullptr, 256, 0));
    EXPECT_FALSE(run.run({}, small_config(), nullptr, nullptr, 0, 192));
}
