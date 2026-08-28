// SPDX-License-Identifier: Apache-2.0
//
// 包含木と識別の打ち切りを CPU 基準と突き合わせる。
//
// 包含判定は OpenCV の cv::pointPolygonTest そのものと比べる。打ち切りは
// OpenCV の identifyCandidates の while ループを host で書き直したものと比べる。
// 上流の段を通さず値を直接注入するため、ここでは完全一致を要求できる。
//
// detectInvertedMarker は本 project に無いため対象外である。DetectorConfig に
// 項目が無く、CPU 基準の経路にも分岐が無い。
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

/// 1 候補分の四隅。並びは corner 0 から 3。
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

/// OpenCV の checkMarker1InMarker2 と同じ判定。
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

/// OpenCV の包含木の構築と同じ手順。
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

/// OpenCV の identifyCandidates の while ループと同じ走査。
///
/// 到達数の数え方まで同じにする。祖先として数えた候補を自分の段でもう一度
/// 数える二重計上を含む。
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

/// 四隅と ID を直接注入して木と打ち切りを求める。
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

// 正常系: 3 段の入れ子で親と段数が CPU 基準と一致する。
TEST(CandidateTreeTest, builds_tree_for_nested_quads) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    // index は周長の降順。0 が最も外側。
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

// 境界値: 囲むものが複数あるとき、最も内側 (index が最大) を親にする。
TEST(CandidateTreeTest, parent_is_the_innermost_container) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    // 0 と 1 の両方が 2 を囲む。1 の方が内側なので 1 が親になる。
    const std::vector<Quad> quads = {axis_aligned(0, 0, 400, 400), axis_aligned(10, 10, 390, 390),
                                     axis_aligned(100, 100, 200, 200)};
    TreeRun run;
    ASSERT_TRUE(run.run(quads, {-1, -1, 3}, DetectorConfig()));
    // 最小の index を選んでいたらここで 0 になり落ちる。
    EXPECT_EQ(run.parent()[2], 1);
    EXPECT_EQ(run.parent()[1], 0);
    EXPECT_EQ(run.depth()[0], 2);
}

// 境界値: 境界に接する四角形は内側として扱う。
TEST(CandidateTreeTest, boundary_touching_quad_counts_as_inside) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    struct Case {
        const char* name_;
        Quad inner_;
        Quad outer_;
    };
    const std::vector<Case> cases = {
            {"完全に一致", axis_aligned(0, 0, 100, 100), axis_aligned(0, 0, 100, 100)},
            {"辺を共有", axis_aligned(0, 10, 50, 90), axis_aligned(0, 0, 100, 100)},
            {"頂点のみ一致", axis_aligned(100, 100, 200, 200), axis_aligned(0, 0, 100, 100)},
            // 下辺の途中に乗る隅。頂点でも斜辺上でもないため、水平な辺の上を
            // 拾う節でしか内側と判定できない。この節を落とすと外側になる。
            {"下辺の途中に乗る", axis_aligned(25, 50, 75, 100), axis_aligned(0, 0, 100, 100)},
            {"上辺の途中に乗る", axis_aligned(25, 0, 75, 50), axis_aligned(0, 0, 100, 100)},
            {"完全に外側", axis_aligned(200, 200, 300, 300), axis_aligned(0, 0, 100, 100)},
            {"一部がはみ出す", axis_aligned(50, 50, 150, 150), axis_aligned(0, 0, 100, 100)},
    };
    for (const Case& item : cases) {
        const std::vector<Quad> quads = {item.outer_, item.inner_};
        TreeRun run;
        ASSERT_TRUE(run.run(quads, {-1, 1}, DetectorConfig())) << item.name_;
        const bool expected = inside_reference(item.inner_, item.outer_);
        EXPECT_EQ(run.parent()[1] == 0, expected) << item.name_;
    }
}

// 正常系: 乱数の四角形で包含判定が cv::pointPolygonTest と一致する。
TEST(CandidateTreeTest, matches_point_polygon_test_for_random_quads) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    std::mt19937_64 rng(20260828U);
    std::uniform_int_distribution<int> coordinate(0, 400);
    std::size_t mismatch = 0;
    std::size_t inside_cases = 0;
    std::size_t concave_cases = 0;

    // 凹四角形を含める。凸性を仮定した判定に置き換えたらここで落ちる。
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
        // 半分は外側の内部に寄せて生成し、内側になる場合を実際に通す。
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
    std::printf("[tree] 400 組: 不一致 %zu、内側 %zu、凹な外側 %zu\n", mismatch, inside_cases,
                concave_cases);
    EXPECT_EQ(mismatch, 0U);
    EXPECT_GT(inside_cases, 0U);
    EXPECT_GT(concave_cases, 0U);
}

// 正常系: 打ち切りの段数が CPU 基準と一致する。
TEST(CandidateTreeTest, stop_depth_matches_reference) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
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
            {"最内だけ識別", {-1, -1, 7}}, {"どれも識別できない", {-1, -1, -1}},
            {"全て識別", {1, 2, 3}},       {"中間だけ識別", {-1, 5, -1}},
            {"最外だけ識別", {9, -1, -1}},
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

// 正常系: 乱数の木でも打ち切りが CPU 基準と一致する。
TEST(CandidateTreeTest, stop_depth_matches_reference_for_random_nesting) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    std::mt19937_64 rng(777U);
    std::uniform_int_distribution<int> identified(0, 2);
    std::size_t mismatch = 0;
    std::size_t suppressed_cases = 0;

    for (int trial = 0; trial < 60; ++trial) {
        // 入れ子の鎖を作る。段数は 1 から 6。
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
    std::printf("[tree] 60 通り: 不一致 %zu、打ち切りが起きた %zu\n", mismatch, suppressed_cases);
    EXPECT_EQ(mismatch, 0U);
    EXPECT_GT(suppressed_cases, 0U);
}

// 正常系: 枝分かれのある木でも打ち切りが CPU 基準と一致する。
//
// 一本鎖では「祖先として印を付けた候補が自分の段に回ってくる」状況も、
// 兄弟が同じ親へ同時に登る状況も起きない。祖先の重複排除と counter の
// 二重計上はどちらも枝分かれで初めて効くため、鎖だけでは固定できない。
TEST(CandidateTreeTest, stop_depth_matches_reference_for_branching_trees) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    // 外側 1 枚の中に互いに素な中間 2 枚、その各々に最内 1 枚。
    // parent = {-1, 0, 0, 1, 2}、depth = {2, 1, 1, 0, 0} になる。
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
            {"最内の片方だけ識別", {-1, -1, -1, 4, -1}},
            {"最内の両方を識別", {-1, -1, -1, 4, 5}},
            {"どれも識別できない", {-1, -1, -1, -1, -1}},
            {"中間の片方だけ識別", {-1, 6, -1, -1, -1}},
            {"最外だけ識別", {9, -1, -1, -1, -1}},
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

    // 「最内の片方だけ識別」では祖先を 2 段登って counter が総数を超える。
    // 二重計上を取り除くとここが減り、打ち切りが 1 段遅れる。
    int branch_stop = 0;
    int branch_counter = 0;
    resolve_reference(parent, depth, {-1, -1, -1, 4, -1}, &branch_stop, &branch_counter);
    EXPECT_GT(branch_counter, static_cast<int>(quads.size()));
}

/// 入れ子の森を組み立てる。子は互いに素な区画へ入れるため兄弟になる。
///
/// 再帰は使わない。本 project の規約は MISRA C++ を基にしており、再帰を
/// 避ける。処理待ちの区画を明示的な stack で持つ。
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

// 正常系: 乱数で作った入れ子の森でも木と打ち切りが CPU 基準と一致する。
TEST(CandidateTreeTest, matches_reference_for_random_forests) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
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
        // 統合後の候補は周長の降順に並ぶ。同じ並びにしてから渡す。
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

        // 同じ段に 2 件以上あるか。祖先の重複排除はここで初めて効く。
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
        // 到達数が総数を超えるのは、祖先として数えた候補を自分の段で
        // もう一度数えたときだけである。二重計上が効いた印になる。
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
    std::printf("[tree] 森 80 通り: 不一致 %zu、枝分かれ %zu、二重計上 %zu、打ち切り %zu\n",
                mismatch, branching_cases, double_counted_cases, suppressed_cases);
    EXPECT_EQ(mismatch, 0U);
    EXPECT_GT(branching_cases, 0U);
    EXPECT_GT(double_counted_cases, 0U);
    EXPECT_GT(suppressed_cases, 0U);
}

// 境界値: 候補が 0 件や 1 件でも走査が止まる。
TEST(CandidateTreeTest, handles_empty_and_single_candidate) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
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

// 正常系: 同じ入力を 2 度流すと同じ結果になる。
TEST(CandidateTreeTest, results_are_deterministic) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
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

// 異常系: 引数が不正なら実行しない。
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
    // doc の出力例と同じ値を固定する。ずれると workspace が足りなくなる。
    EXPECT_EQ(aruco3cuda::detail::candidate_tree_workspace_bytes(config), 49664U);
}

}  // namespace
