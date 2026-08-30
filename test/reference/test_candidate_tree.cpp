// SPDX-License-Identifier: Apache-2.0
//
// Cross-checks the containment tree and the identification cutoff against the CPU
// reference.
//
// The containment decision is compared against cv::pointPolygonTest itself. The cutoff
// is compared against a host rewrite of the while loop in the identifyCandidates of
// OpenCV. Values are injected directly instead of coming through the upstream stages,
// so exact agreement can be required here.
//
// detectInvertedMarker is out of scope because this project does not have it: there is
// no such field in DetectorConfig, and the CPU reference path has no branch for it.
#include "candidate_tree.hpp"

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
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
#include "dictionary_match.hpp"

namespace {

using aruco3cuda::DetectorConfig;
using aruco3cuda::MemorySpace;
using aruco3cuda::Status;
using aruco3cuda::Workspace;

bool has_cuda_device() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

/// The four corners of one candidate, ordered from corner 0 to corner 3.
struct Quad {
    std::int32_t x_[4] = {0, 0, 0, 0};
    std::int32_t y_[4] = {0, 0, 0, 0};
};

Quad axis_aligned(int left, int top, int right, int bottom) {
    Quad quad;
    const int xs[4] = {left, right, right, left};
    const int ys[4] = {top, top, bottom, bottom};
    for (int i = 0; i < 4; ++i) {
        quad.x_[i] = xs[i];
        quad.y_[i] = ys[i];
    }
    return quad;
}

std::vector<cv::Point2f> to_contour(const Quad& quad) {
    std::vector<cv::Point2f> points;
    points.reserve(4U);
    for (int i = 0; i < 4; ++i) {
        points.emplace_back(static_cast<float>(quad.x_[i]), static_cast<float>(quad.y_[i]));
    }
    return points;
}

/// The same decision as checkMarker1InMarker2 in OpenCV.
bool inside_reference(const Quad& inner, const Quad& outer) {
    const std::vector<cv::Point2f> polygon = to_contour(outer);
    for (int i = 0; i < 4; ++i) {
        const cv::Point2f point(static_cast<float>(inner.x_[i]), static_cast<float>(inner.y_[i]));
        if (cv::pointPolygonTest(polygon, point, false) < 0) {
            return false;
        }
    }
    return true;
}

/// The same procedure as the containment tree construction in OpenCV.
void build_tree_reference(const std::vector<Quad>& quads, std::vector<int>* parent,
                          std::vector<int>* depth) {
    const auto count = static_cast<int>(quads.size());
    parent->assign(quads.size(), -1);
    depth->assign(quads.size(), 0);
    for (int i = count - 1; i >= 0; --i) {
        for (int j = i - 1; j >= 0; --j) {
            if (inside_reference(quads[static_cast<std::size_t>(i)],
                                 quads[static_cast<std::size_t>(j)])) {
                (*parent)[static_cast<std::size_t>(i)] = j;
                (*depth)[static_cast<std::size_t>(j)] =
                        std::max((*depth)[static_cast<std::size_t>(j)],
                                 (*depth)[static_cast<std::size_t>(i)] + 1);
                break;
            }
        }
    }
}

/// The same traversal as the while loop in the identifyCandidates of OpenCV.
///
/// It matches even in how the reached count is computed, including the double count
/// where a candidate already counted as an ancestor is counted again at its own
/// depth.
void resolve_reference(const std::vector<int>& parent, const std::vector<int>& depth,
                       const std::vector<int>& ids, int* out_stop_depth, int* out_counter) {
    const auto total = static_cast<int>(parent.size());
    std::vector<std::uint8_t> was(parent.size(), 0U);
    int max_depth = 0;
    for (const int value : depth) {
        max_depth = std::max(max_depth, value);
    }
    std::vector<std::vector<int>> by_depth(static_cast<std::size_t>(max_depth) + 1U);
    for (int i = 0; i < total; ++i) {
        by_depth[static_cast<std::size_t>(depth[static_cast<std::size_t>(i)])].push_back(i);
    }

    int level = 0;
    int counter = 0;
    while (counter < total && level < static_cast<int>(by_depth.size())) {
        for (const int v : by_depth[static_cast<std::size_t>(level)]) {
            was[static_cast<std::size_t>(v)] = 1U;
        }
        for (const int v : by_depth[static_cast<std::size_t>(level)]) {
            if (ids[static_cast<std::size_t>(v)] >= 0) {
                int up = parent[static_cast<std::size_t>(v)];
                while (up != -1) {
                    if (was[static_cast<std::size_t>(up)] == 0U) {
                        was[static_cast<std::size_t>(up)] = 1U;
                        ++counter;
                    }
                    up = parent[static_cast<std::size_t>(up)];
                }
            }
            ++counter;
        }
        ++level;
    }
    *out_stop_depth = level;
    *out_counter = counter;
}

/// Injects corners and IDs directly, then computes the tree and the cutoff.
class TreeRun {
public:
    TreeRun() = default;
    TreeRun(const TreeRun&) = delete;
    TreeRun& operator=(const TreeRun&) = delete;

    bool run(const std::vector<Quad>& quads, const std::vector<int>& ids,
             const DetectorConfig& base) {
        DetectorConfig config = base;
        config.max_candidates_ = quads.empty() ? 1 : static_cast<int>(quads.size());
        config.max_markers_ = config.max_candidates_;

        const std::size_t bytes = aruco3cuda::detail::candidate_workspace_bytes(config, 64, 64) +
                                  aruco3cuda::detail::match_workspace_bytes(config) +
                                  aruco3cuda::detail::candidate_tree_workspace_bytes(config);
        if (bytes == 0U ||
            this->workspace_.ensure_capacity(bytes, MemorySpace::kDevice, nullptr) != Status::kOk) {
            return false;
        }
        this->workspace_.reset();

        aruco3cuda::detail::CandidateFilterBuffers filter;
        aruco3cuda::detail::DeviceCandidates candidates;
        aruco3cuda::detail::MatchBuffers matches;
        aruco3cuda::detail::CandidateTreeBuffers tree;
        if (aruco3cuda::detail::reserve_candidates(config, 64, 64, this->workspace_, &filter,
                                                   &candidates) != Status::kOk ||
            aruco3cuda::detail::reserve_matches(config, this->workspace_, &matches) !=
                    Status::kOk ||
            aruco3cuda::detail::reserve_candidate_tree(config, this->workspace_, &tree) !=
                    Status::kOk) {
            return false;
        }

        const auto count = static_cast<int>(quads.size());
        std::vector<std::int32_t> plane(quads.size());
        for (int corner = 0; corner < 4; ++corner) {
            for (std::size_t i = 0; i < quads.size(); ++i) {
                plane[i] = quads[i].x_[corner];
            }
            if (!quads.empty() &&
                cudaMemcpy(candidates.corner_x_ +
                                   (static_cast<std::ptrdiff_t>(corner) * candidates.capacity_),
                           plane.data(), plane.size() * sizeof(std::int32_t),
                           cudaMemcpyHostToDevice) != cudaSuccess) {
                return false;
            }
            for (std::size_t i = 0; i < quads.size(); ++i) {
                plane[i] = quads[i].y_[corner];
            }
            if (!quads.empty() &&
                cudaMemcpy(candidates.corner_y_ +
                                   (static_cast<std::ptrdiff_t>(corner) * candidates.capacity_),
                           plane.data(), plane.size() * sizeof(std::int32_t),
                           cudaMemcpyHostToDevice) != cudaSuccess) {
                return false;
            }
        }
        const std::vector<std::int32_t> id_values(ids.begin(), ids.end());
        if (cudaMemcpy(candidates.count_, &count, sizeof(int), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
            return false;
        }
        if (!id_values.empty() &&
            cudaMemcpy(matches.ids_, id_values.data(), id_values.size() * sizeof(std::int32_t),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            return false;
        }

        if (aruco3cuda::detail::build_candidate_tree_async(candidates, &tree, nullptr) !=
                    Status::kOk ||
            aruco3cuda::detail::resolve_suppression_async(candidates, matches, &tree, nullptr) !=
                    Status::kOk ||
            cudaDeviceSynchronize() != cudaSuccess) {
            return false;
        }

        this->parent_.resize(quads.size());
        this->depth_.resize(quads.size());
        const std::size_t plane_bytes = quads.size() * sizeof(std::int32_t);
        if (!quads.empty() && (cudaMemcpy(this->parent_.data(), tree.parent_, plane_bytes,
                                          cudaMemcpyDeviceToHost) != cudaSuccess ||
                               cudaMemcpy(this->depth_.data(), tree.depth_, plane_bytes,
                                          cudaMemcpyDeviceToHost) != cudaSuccess)) {
            return false;
        }
        return cudaMemcpy(&this->stop_depth_, tree.stop_depth_, sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost) == cudaSuccess &&
               cudaMemcpy(&this->counter_, tree.counter_, sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost) == cudaSuccess;
    }

    const std::vector<std::int32_t>& parent() const { return this->parent_; }
    const std::vector<std::int32_t>& depth() const { return this->depth_; }
    int stop_depth() const { return this->stop_depth_; }
    int counter() const { return this->counter_; }

private:
    Workspace workspace_;
    std::vector<std::int32_t> parent_;
    std::vector<std::int32_t> depth_;
    std::int32_t stop_depth_ = 0;
    std::int32_t counter_ = 0;
};

// Happy path: with three levels of nesting, the parents and depths match the CPU
// reference.
TEST(CandidateTreeTest, builds_tree_for_nested_quads) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    // Indices are in descending perimeter order, so 0 is the outermost quad.
    const std::vector<Quad> quads = {axis_aligned(0, 0, 300, 300), axis_aligned(50, 50, 250, 250),
                                     axis_aligned(100, 100, 200, 200)};
    const std::vector<int> ids = {-1, -1, 7};

    std::vector<int> expected_parent;
    std::vector<int> expected_depth;
    build_tree_reference(quads, &expected_parent, &expected_depth);
    ASSERT_EQ(expected_parent, (std::vector<int>{-1, 0, 1}));
    ASSERT_EQ(expected_depth, (std::vector<int>{2, 1, 0}));

    TreeRun run;
    ASSERT_TRUE(run.run(quads, ids, DetectorConfig()));
    for (std::size_t i = 0; i < quads.size(); ++i) {
        EXPECT_EQ(run.parent()[i], expected_parent[i]) << i;
        EXPECT_EQ(run.depth()[i], expected_depth[i]) << i;
    }
}

// Boundary case: when several quads enclose a candidate, the innermost one (the
// largest index) becomes the parent.
TEST(CandidateTreeTest, parent_is_the_innermost_container) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    // Both 0 and 1 enclose 2. Since 1 is the inner one, 1 becomes the parent.
    const std::vector<Quad> quads = {axis_aligned(0, 0, 400, 400), axis_aligned(10, 10, 390, 390),
                                     axis_aligned(100, 100, 200, 200)};
    TreeRun run;
    ASSERT_TRUE(run.run(quads, {-1, -1, 3}, DetectorConfig()));
    // Had the smallest index been selected, this would be 0 and the test would fail.
    EXPECT_EQ(run.parent()[2], 1);
    EXPECT_EQ(run.parent()[1], 0);
    EXPECT_EQ(run.depth()[0], 2);
}

// Boundary case: a quad that touches the boundary counts as inside.
TEST(CandidateTreeTest, boundary_touching_quad_counts_as_inside) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    struct Case {
        const char* name_;
        Quad inner_;
        Quad outer_;
    };
    const std::vector<Case> cases = {
            {"identical", axis_aligned(0, 0, 100, 100), axis_aligned(0, 0, 100, 100)},
            {"shares an edge", axis_aligned(0, 10, 50, 90), axis_aligned(0, 0, 100, 100)},
            {"touches at a vertex only", axis_aligned(100, 100, 200, 200),
             axis_aligned(0, 0, 100, 100)},
            // A corner that lands partway along the bottom edge. It is neither a vertex
            // nor on a slanted edge, so only the clause that handles horizontal edges
            // can classify it as inside. Drop that clause and it becomes outside.
            {"partway along the bottom edge", axis_aligned(25, 50, 75, 100),
             axis_aligned(0, 0, 100, 100)},
            {"partway along the top edge", axis_aligned(25, 0, 75, 50),
             axis_aligned(0, 0, 100, 100)},
            {"fully outside", axis_aligned(200, 200, 300, 300), axis_aligned(0, 0, 100, 100)},
            {"partly sticking out", axis_aligned(50, 50, 150, 150), axis_aligned(0, 0, 100, 100)},
    };
    for (const Case& item : cases) {
        const std::vector<Quad> quads = {item.outer_, item.inner_};
        TreeRun run;
        ASSERT_TRUE(run.run(quads, {-1, 1}, DetectorConfig())) << item.name_;
        const bool expected = inside_reference(item.inner_, item.outer_);
        EXPECT_EQ(run.parent()[1] == 0, expected) << item.name_;
    }
}

// Happy path: on random quads, the containment decision agrees with
// cv::pointPolygonTest.
TEST(CandidateTreeTest, matches_point_polygon_test_for_random_quads) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    std::mt19937_64 rng(20260828U);
    std::uniform_int_distribution<int> coordinate(0, 400);
    std::size_t mismatch = 0;
    std::size_t inside_cases = 0;
    std::size_t concave_cases = 0;

    // Concave quads are included. Replacing the decision with one that assumes
    // convexity would fail here.
    const auto is_convex = [](const Quad& quad) {
        int sign = 0;
        for (int i = 0; i < 4; ++i) {
            const int a = i;
            const int b = (i + 1) % 4;
            const int c = (i + 2) % 4;
            const long long cross =
                    (static_cast<long long>(quad.x_[b] - quad.x_[a]) * (quad.y_[c] - quad.y_[b])) -
                    (static_cast<long long>(quad.y_[b] - quad.y_[a]) * (quad.x_[c] - quad.x_[b]));
            const int current = (cross > 0) ? 1 : ((cross < 0) ? -1 : 0);
            if (current == 0) {
                continue;
            }
            if (sign == 0) {
                sign = current;
            } else if (sign != current) {
                return false;
            }
        }
        return true;
    };

    for (int trial = 0; trial < 400; ++trial) {
        Quad outer;
        Quad inner;
        for (int i = 0; i < 4; ++i) {
            outer.x_[i] = coordinate(rng);
            outer.y_[i] = coordinate(rng);
        }
        // Half of the cases are generated near the interior of the outer quad, so that
        // the inside case is actually exercised.
        for (int i = 0; i < 4; ++i) {
            if (trial % 2 == 0) {
                inner.x_[i] = (outer.x_[0] + outer.x_[1] + outer.x_[2] + outer.x_[3]) / 4 +
                              ((i % 2 == 0) ? -5 : 5);
                inner.y_[i] = (outer.y_[0] + outer.y_[1] + outer.y_[2] + outer.y_[3]) / 4 +
                              ((i / 2 == 0) ? -5 : 5);
            } else {
                inner.x_[i] = coordinate(rng);
                inner.y_[i] = coordinate(rng);
            }
        }
        if (!is_convex(outer)) {
            ++concave_cases;
        }
        const bool expected = inside_reference(inner, outer);
        if (expected) {
            ++inside_cases;
        }
        const std::vector<Quad> quads = {outer, inner};
        TreeRun run;
        ASSERT_TRUE(run.run(quads, {-1, 1}, DetectorConfig()));
        if ((run.parent()[1] == 0) != expected) {
            ++mismatch;
        }
    }
    std::printf("[tree] 400 pairs: mismatches %zu, inside %zu, concave outer %zu\n", mismatch,
                inside_cases, concave_cases);
    EXPECT_EQ(mismatch, 0U);
    EXPECT_GT(inside_cases, 0U);
    EXPECT_GT(concave_cases, 0U);
}

// Happy path: the cutoff depth matches the CPU reference.
TEST(CandidateTreeTest, stop_depth_matches_reference) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    const std::vector<Quad> quads = {axis_aligned(0, 0, 300, 300), axis_aligned(50, 50, 250, 250),
                                     axis_aligned(100, 100, 200, 200)};
    std::vector<int> parent;
    std::vector<int> depth;
    build_tree_reference(quads, &parent, &depth);

    struct Case {
        const char* name_;
        std::vector<int> ids_;
    };
    const std::vector<Case> cases = {
            {"innermost identified only", {-1, -1, 7}},
            {"none identified", {-1, -1, -1}},
            {"all identified", {1, 2, 3}},
            {"middle identified only", {-1, 5, -1}},
            {"outermost identified only", {9, -1, -1}},
    };
    for (const Case& item : cases) {
        int expected_stop = 0;
        int expected_counter = 0;
        resolve_reference(parent, depth, item.ids_, &expected_stop, &expected_counter);
        TreeRun run;
        ASSERT_TRUE(run.run(quads, item.ids_, DetectorConfig())) << item.name_;
        EXPECT_EQ(run.stop_depth(), expected_stop) << item.name_;
        EXPECT_EQ(run.counter(), expected_counter) << item.name_;
    }
}

// Happy path: on random trees too, the cutoff matches the CPU reference.
TEST(CandidateTreeTest, stop_depth_matches_reference_for_random_nesting) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    std::mt19937_64 rng(777U);
    std::uniform_int_distribution<int> identified(0, 2);
    std::size_t mismatch = 0;
    std::size_t suppressed_cases = 0;

    for (int trial = 0; trial < 60; ++trial) {
        // Build a nesting chain, 1 to 6 levels deep.
        const int levels = 1 + (trial % 6);
        std::vector<Quad> quads;
        quads.reserve(static_cast<std::size_t>(levels));
        for (int level = 0; level < levels; ++level) {
            const int margin = level * 20;
            quads.push_back(axis_aligned(margin, margin, 400 - margin, 400 - margin));
        }
        std::vector<int> ids(quads.size(), -1);
        for (std::size_t i = 0; i < ids.size(); ++i) {
            ids[i] = (identified(rng) == 0) ? static_cast<int>(i) : -1;
        }

        std::vector<int> parent;
        std::vector<int> depth;
        build_tree_reference(quads, &parent, &depth);
        int expected_stop = 0;
        int expected_counter = 0;
        resolve_reference(parent, depth, ids, &expected_stop, &expected_counter);
        if (expected_stop < levels) {
            ++suppressed_cases;
        }

        TreeRun run;
        ASSERT_TRUE(run.run(quads, ids, DetectorConfig()));
        if (run.stop_depth() != expected_stop || run.counter() != expected_counter) {
            ++mismatch;
        }
        for (std::size_t i = 0; i < quads.size(); ++i) {
            if (run.parent()[i] != parent[i] || run.depth()[i] != depth[i]) {
                ++mismatch;
                break;
            }
        }
    }
    std::printf("[tree] 60 cases: mismatches %zu, cutoff triggered %zu\n", mismatch,
                suppressed_cases);
    EXPECT_EQ(mismatch, 0U);
    EXPECT_GT(suppressed_cases, 0U);
}

// Happy path: on branching trees too, the cutoff matches the CPU reference.
//
// A single chain never produces the situation where a candidate already marked as an
// ancestor comes up again at its own depth, nor the one where siblings walk up to the
// same parent at the same time. Both the deduplication of ancestors and the double
// count in the counter only take effect once the tree branches, so a chain alone
// cannot pin them down.
TEST(CandidateTreeTest, stop_depth_matches_reference_for_branching_trees) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    // One outer quad holding two disjoint middle quads, each of which holds one
    // innermost quad. This gives parent = {-1, 0, 0, 1, 2} and depth = {2, 1, 1, 0, 0}.
    const std::vector<Quad> quads = {axis_aligned(0, 0, 1000, 1000), axis_aligned(10, 10, 480, 980),
                                     axis_aligned(520, 10, 990, 980),
                                     axis_aligned(60, 60, 430, 930),
                                     axis_aligned(570, 60, 940, 930)};

    std::vector<int> parent;
    std::vector<int> depth;
    build_tree_reference(quads, &parent, &depth);
    ASSERT_EQ(parent, (std::vector<int>{-1, 0, 0, 1, 2}));
    ASSERT_EQ(depth, (std::vector<int>{2, 1, 1, 0, 0}));

    struct Case {
        const char* name_;
        std::vector<int> ids_;
    };
    const std::vector<Case> cases = {
            {"one innermost identified", {-1, -1, -1, 4, -1}},
            {"both innermost identified", {-1, -1, -1, 4, 5}},
            {"none identified", {-1, -1, -1, -1, -1}},
            {"one middle identified", {-1, 6, -1, -1, -1}},
            {"outermost identified only", {9, -1, -1, -1, -1}},
    };
    for (const Case& item : cases) {
        int expected_stop = 0;
        int expected_counter = 0;
        resolve_reference(parent, depth, item.ids_, &expected_stop, &expected_counter);
        TreeRun run;
        ASSERT_TRUE(run.run(quads, item.ids_, DetectorConfig())) << item.name_;
        EXPECT_EQ(run.stop_depth(), expected_stop) << item.name_;
        EXPECT_EQ(run.counter(), expected_counter) << item.name_;
        for (std::size_t i = 0; i < quads.size(); ++i) {
            EXPECT_EQ(run.parent()[i], parent[i]) << item.name_ << " index=" << i;
            EXPECT_EQ(run.depth()[i], depth[i]) << item.name_ << " index=" << i;
        }
    }

    // In "one innermost identified" the walk goes up two ancestor levels and the
    // counter exceeds the total. Removing the double count would lower it here and
    // delay the cutoff by one level.
    int branch_stop = 0;
    int branch_counter = 0;
    resolve_reference(parent, depth, {-1, -1, -1, 4, -1}, &branch_stop, &branch_counter);
    EXPECT_GT(branch_counter, static_cast<int>(quads.size()));
}

/// Builds a nested forest. Children go into disjoint regions, so they become siblings.
///
/// No recursion is used: this project's conventions are based on MISRA C++, which
/// avoids recursion. The regions still to be processed are held in an explicit stack.
void grow_forest(int extent, int budget, std::mt19937_64& rng, std::vector<Quad>* out) {
    struct Region {
        int left_ = 0;
        int top_ = 0;
        int right_ = 0;
        int bottom_ = 0;
        int budget_ = 0;
    };
    std::uniform_int_distribution<int> children(0, 2);
    std::vector<Region> pending = {Region{0, 0, extent, extent, budget}};
    while (!pending.empty()) {
        const Region region = pending.back();
        pending.pop_back();
        out->push_back(axis_aligned(region.left_, region.top_, region.right_, region.bottom_));
        if (region.budget_ <= 0 || (region.right_ - region.left_) < 120 ||
            (region.bottom_ - region.top_) < 120) {
            continue;
        }
        const int count = children(rng);
        if (count == 0) {
            continue;
        }
        const int inner_left = region.left_ + 10;
        const int inner_right = region.right_ - 10;
        const int inner_top = region.top_ + 10;
        const int inner_bottom = region.bottom_ - 10;
        if (count == 1) {
            pending.push_back(
                    Region{inner_left, inner_top, inner_right, inner_bottom, region.budget_ - 1});
            continue;
        }
        const int middle = (inner_left + inner_right) / 2;
        pending.push_back(
                Region{inner_left, inner_top, middle - 5, inner_bottom, region.budget_ - 1});
        pending.push_back(
                Region{middle + 5, inner_top, inner_right, inner_bottom, region.budget_ - 1});
    }
}

// Happy path: on randomly generated nested forests too, the tree and the cutoff match
// the CPU reference.
TEST(CandidateTreeTest, matches_reference_for_random_forests) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    std::mt19937_64 rng(31415U);
    std::uniform_int_distribution<int> identified(0, 2);

    std::size_t mismatch = 0;
    std::size_t branching_cases = 0;
    std::size_t double_counted_cases = 0;
    std::size_t suppressed_cases = 0;
    for (int trial = 0; trial < 80; ++trial) {
        std::vector<Quad> quads;
        grow_forest(1200, 1 + (trial % 4), rng, &quads);
        // After merging, candidates are ordered by descending perimeter. Sort them the
        // same way before passing them in.
        std::stable_sort(quads.begin(), quads.end(), [](const Quad& a, const Quad& b) {
            const int a_perimeter = (a.x_[1] - a.x_[0]) + (a.y_[2] - a.y_[1]);
            const int b_perimeter = (b.x_[1] - b.x_[0]) + (b.y_[2] - b.y_[1]);
            return a_perimeter > b_perimeter;
        });
        std::vector<int> ids(quads.size(), -1);
        for (std::size_t i = 0; i < ids.size(); ++i) {
            ids[i] = (identified(rng) == 0) ? static_cast<int>(i) : -1;
        }

        std::vector<int> parent;
        std::vector<int> depth;
        build_tree_reference(quads, &parent, &depth);
        int expected_stop = 0;
        int expected_counter = 0;
        resolve_reference(parent, depth, ids, &expected_stop, &expected_counter);

        // Are there two or more quads at the same depth? Only then does the
        // deduplication of ancestors take effect.
        std::vector<int> per_depth(quads.size() + 1U, 0);
        int max_depth = 0;
        for (const int value : depth) {
            ++per_depth[static_cast<std::size_t>(value)];
            max_depth = std::max(max_depth, value);
        }
        for (const int value : per_depth) {
            if (value >= 2) {
                ++branching_cases;
                break;
            }
        }
        // The reached count can only exceed the total when a candidate counted as an
        // ancestor is counted again at its own depth. That is the sign that the double
        // count took effect.
        if (expected_counter > static_cast<int>(quads.size())) {
            ++double_counted_cases;
        }
        if (expected_stop <= max_depth) {
            ++suppressed_cases;
        }

        TreeRun run;
        ASSERT_TRUE(run.run(quads, ids, DetectorConfig()));
        if (run.stop_depth() != expected_stop || run.counter() != expected_counter) {
            ++mismatch;
            continue;
        }
        for (std::size_t i = 0; i < quads.size(); ++i) {
            if (run.parent()[i] != parent[i] || run.depth()[i] != depth[i]) {
                ++mismatch;
                break;
            }
        }
    }
    std::printf(
            "[tree] 80 forests: mismatches %zu, branching %zu, double counted %zu, "
            "cutoff %zu\n",
            mismatch, branching_cases, double_counted_cases, suppressed_cases);
    EXPECT_EQ(mismatch, 0U);
    EXPECT_GT(branching_cases, 0U);
    EXPECT_GT(double_counted_cases, 0U);
    EXPECT_GT(suppressed_cases, 0U);
}

// Boundary case: the traversal terminates with zero or one candidate as well.
TEST(CandidateTreeTest, handles_empty_and_single_candidate) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    TreeRun empty_run;
    ASSERT_TRUE(empty_run.run({}, {}, DetectorConfig()));
    EXPECT_EQ(empty_run.stop_depth(), 0);
    EXPECT_EQ(empty_run.counter(), 0);

    TreeRun single_run;
    ASSERT_TRUE(single_run.run({axis_aligned(0, 0, 100, 100)}, {5}, DetectorConfig()));
    EXPECT_EQ(single_run.parent()[0], -1);
    EXPECT_EQ(single_run.depth()[0], 0);
    EXPECT_EQ(single_run.stop_depth(), 1);
    EXPECT_EQ(single_run.counter(), 1);
}

// Happy path: running the same input twice yields the same result.
TEST(CandidateTreeTest, results_are_deterministic) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    const std::vector<Quad> quads = {axis_aligned(0, 0, 300, 300), axis_aligned(20, 20, 280, 280),
                                     axis_aligned(60, 60, 240, 240),
                                     axis_aligned(120, 120, 180, 180)};
    const std::vector<int> ids = {-1, 4, -1, 8};
    TreeRun first;
    TreeRun second;
    ASSERT_TRUE(first.run(quads, ids, DetectorConfig()));
    ASSERT_TRUE(second.run(quads, ids, DetectorConfig()));
    EXPECT_EQ(first.parent(), second.parent());
    EXPECT_EQ(first.depth(), second.depth());
    EXPECT_EQ(first.stop_depth(), second.stop_depth());
    EXPECT_EQ(first.counter(), second.counter());
}

// Failure path: nothing runs when the arguments are invalid.
TEST(CandidateTreeTest, rejects_invalid_arguments) {
    Workspace workspace;
    DetectorConfig config;
    aruco3cuda::detail::CandidateTreeBuffers tree;
    EXPECT_EQ(aruco3cuda::detail::reserve_candidate_tree(config, workspace, nullptr),
              Status::kInvalidArgument);
    EXPECT_NE(aruco3cuda::detail::reserve_candidate_tree(config, workspace, &tree), Status::kOk);

    aruco3cuda::detail::DeviceCandidates candidates;
    aruco3cuda::detail::MatchBuffers matches;
    EXPECT_EQ(aruco3cuda::detail::build_candidate_tree_async(candidates, nullptr, nullptr),
              Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::detail::build_candidate_tree_async(candidates, &tree, nullptr),
              Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::detail::resolve_suppression_async(candidates, matches, nullptr, nullptr),
              Status::kInvalidArgument);

    config.max_candidates_ = 0;
    EXPECT_EQ(aruco3cuda::detail::candidate_tree_workspace_bytes(config), 0U);
    config.max_candidates_ = 4096;
    // Pins the same value as the example output in the docs. If it drifts, the
    // workspace is no longer large enough.
    EXPECT_EQ(aruco3cuda::detail::candidate_tree_workspace_bytes(config), 49664U);
}

}  // namespace
