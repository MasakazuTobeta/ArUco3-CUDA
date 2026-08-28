// SPDX-License-Identifier: Apache-2.0
//
// 採用した候補の詰め直しと四隅の回転打ち消しを検証する。
//
// S9 の重複整理は「同じ ID を落とすこと」ではない。OpenCV は ID による
// 重複除去を持たない。重複が消えるのは包含木による打ち切りの結果である。
// この test は同じ ID が 2 件出ることを固定して、善意の ID 重複除去が
// 混入するのを防ぐ。
#include "detection_emit.hpp"

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"
#include "aruco3cuda/workspace.hpp"
#include "candidate_filter.hpp"
#include "candidate_tree.hpp"
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

/// 1 候補分の入力。
struct CandidateInput {
    std::int32_t x_[4] = {0, 0, 0, 0};
    std::int32_t y_[4] = {0, 0, 0, 0};
    int id_ = -1;
    int rotation_ = 0;
    int depth_ = 0;
};

CandidateInput make_candidate(int left, int top, int right, int bottom, int id, int rotation,
                              int depth) {
    CandidateInput input;
    const int xs[4] = {left, right, right, left};
    const int ys[4] = {top, top, bottom, bottom};
    for (int i = 0; i < 4; ++i) {
        input.x_[i] = xs[i];
        input.y_[i] = ys[i];
    }
    input.id_ = id;
    input.rotation_ = rotation;
    input.depth_ = depth;
    return input;
}

/// 1 件の検出結果。
struct Detection {
    int id_ = -1;
    int rotation_ = 0;
    int source_ = -1;
    float x_[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    float y_[4] = {0.0F, 0.0F, 0.0F, 0.0F};
};

/// 候補と段数を直接注入して検出結果を作る。
class EmitRun {
public:
    EmitRun() = default;
    EmitRun(const EmitRun&) = delete;
    EmitRun& operator=(const EmitRun&) = delete;

    bool run(const std::vector<CandidateInput>& inputs, int stop_depth, int max_markers,
             const DetectorConfig& base, int candidate_capacity = 0) {
        DetectorConfig config = base;
        config.max_candidates_ = (candidate_capacity > 0)
                                         ? candidate_capacity
                                         : (inputs.empty() ? 1 : static_cast<int>(inputs.size()));
        config.max_markers_ = max_markers;
        if (config.max_markers_ > config.max_candidates_) {
            return false;
        }

        const std::size_t bytes = aruco3cuda::detail::candidate_workspace_bytes(config, 64, 64) +
                                  aruco3cuda::detail::match_workspace_bytes(config) +
                                  aruco3cuda::detail::candidate_tree_workspace_bytes(config) +
                                  aruco3cuda::detail::detection_workspace_bytes(config);
        if (bytes == 0U ||
            this->workspace_.ensure_capacity(bytes, MemorySpace::kDevice, nullptr) != Status::kOk) {
            return false;
        }
        this->workspace_.reset();

        aruco3cuda::detail::CandidateFilterBuffers filter;
        aruco3cuda::detail::DeviceCandidates candidates;
        aruco3cuda::detail::MatchBuffers matches;
        aruco3cuda::detail::CandidateTreeBuffers tree;
        aruco3cuda::detail::DetectionEmitBuffers emit;
        aruco3cuda::detail::DeviceDetections detections;
        if (aruco3cuda::detail::reserve_candidates(config, 64, 64, this->workspace_, &filter,
                                                   &candidates) != Status::kOk ||
            aruco3cuda::detail::reserve_matches(config, this->workspace_, &matches) !=
                    Status::kOk ||
            aruco3cuda::detail::reserve_candidate_tree(config, this->workspace_, &tree) !=
                    Status::kOk ||
            aruco3cuda::detail::reserve_detections(config, this->workspace_, &emit, &detections) !=
                    Status::kOk) {
            return false;
        }

        if (!this->upload(inputs, stop_depth, candidates, matches, tree)) {
            return false;
        }
        if (aruco3cuda::detail::emit_detections_async(candidates, matches, tree, &emit, &detections,
                                                      nullptr) != Status::kOk) {
            return false;
        }
        this->count_status_ =
                aruco3cuda::detail::read_detection_count(detections, &this->count_, nullptr);
        return this->download(detections);
    }

    const std::vector<Detection>& detections() const { return this->detections_; }
    int count() const { return this->count_; }
    Status count_status() const { return this->count_status_; }

private:
    bool upload(const std::vector<CandidateInput>& inputs, int stop_depth,
                const aruco3cuda::detail::DeviceCandidates& candidates,
                const aruco3cuda::detail::MatchBuffers& matches,
                const aruco3cuda::detail::CandidateTreeBuffers& tree) {
        const auto count = static_cast<int>(inputs.size());
        if (cudaMemcpy(candidates.count_, &count, sizeof(int), cudaMemcpyHostToDevice) !=
                    cudaSuccess ||
            cudaMemcpy(tree.stop_depth_, &stop_depth, sizeof(int), cudaMemcpyHostToDevice) !=
                    cudaSuccess) {
            return false;
        }
        if (inputs.empty()) {
            return true;
        }
        std::vector<std::int32_t> plane(inputs.size());
        for (int corner = 0; corner < 4; ++corner) {
            for (std::size_t i = 0; i < inputs.size(); ++i) {
                plane[i] = inputs[i].x_[corner];
            }
            if (cudaMemcpy(candidates.corner_x_ +
                                   (static_cast<std::ptrdiff_t>(corner) * candidates.capacity_),
                           plane.data(), plane.size() * sizeof(std::int32_t),
                           cudaMemcpyHostToDevice) != cudaSuccess) {
                return false;
            }
            for (std::size_t i = 0; i < inputs.size(); ++i) {
                plane[i] = inputs[i].y_[corner];
            }
            if (cudaMemcpy(candidates.corner_y_ +
                                   (static_cast<std::ptrdiff_t>(corner) * candidates.capacity_),
                           plane.data(), plane.size() * sizeof(std::int32_t),
                           cudaMemcpyHostToDevice) != cudaSuccess) {
                return false;
            }
        }
        const auto copy_plane = [](std::int32_t* target, const std::vector<std::int32_t>& values) {
            return cudaMemcpy(target, values.data(), values.size() * sizeof(std::int32_t),
                              cudaMemcpyHostToDevice) == cudaSuccess;
        };
        std::vector<std::int32_t> ids(inputs.size());
        std::vector<std::int32_t> rotations(inputs.size());
        std::vector<std::int32_t> depths(inputs.size());
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            ids[i] = inputs[i].id_;
            rotations[i] = inputs[i].rotation_;
            depths[i] = inputs[i].depth_;
        }
        return copy_plane(matches.ids_, ids) && copy_plane(matches.rotations_, rotations) &&
               copy_plane(tree.depth_, depths);
    }

    bool download(const aruco3cuda::detail::DeviceDetections& detections) {
        this->detections_.clear();
        if (this->count_ <= 0) {
            return true;
        }
        // 検出数を超えた領域は kernel が書かない。初期化していない値を読むと
        // Compute Sanitizer の initcheck が指摘する。有効な範囲だけ写す。
        const auto valid = static_cast<std::size_t>(this->count_);
        const auto capacity = static_cast<std::ptrdiff_t>(detections.capacity_);
        std::vector<std::int32_t> ids(valid);
        std::vector<std::int32_t> rotations(valid);
        std::vector<std::int32_t> source(valid);
        std::vector<float> corner_x(valid * 4U);
        std::vector<float> corner_y(valid * 4U);
        const std::size_t plane = valid * sizeof(std::int32_t);
        if (cudaMemcpy(ids.data(), detections.ids_, plane, cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(rotations.data(), detections.rotations_, plane, cudaMemcpyDeviceToHost) !=
                    cudaSuccess ||
            cudaMemcpy(source.data(), detections.source_, plane, cudaMemcpyDeviceToHost) !=
                    cudaSuccess) {
            return false;
        }
        for (int corner = 0; corner < 4; ++corner) {
            const std::ptrdiff_t offset = static_cast<std::ptrdiff_t>(corner) * capacity;
            if (cudaMemcpy(corner_x.data() + (static_cast<std::size_t>(corner) * valid),
                           detections.corner_x_ + offset, valid * sizeof(float),
                           cudaMemcpyDeviceToHost) != cudaSuccess ||
                cudaMemcpy(corner_y.data() + (static_cast<std::size_t>(corner) * valid),
                           detections.corner_y_ + offset, valid * sizeof(float),
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
                return false;
            }
        }
        for (std::size_t index = 0; index < valid; ++index) {
            Detection item;
            item.id_ = ids[index];
            item.rotation_ = rotations[index];
            item.source_ = source[index];
            for (int corner = 0; corner < 4; ++corner) {
                item.x_[corner] = corner_x[(static_cast<std::size_t>(corner) * valid) + index];
                item.y_[corner] = corner_y[(static_cast<std::size_t>(corner) * valid) + index];
            }
            this->detections_.push_back(item);
        }
        return true;
    }

    Workspace workspace_;
    std::vector<Detection> detections_;
    int count_ = 0;
    Status count_status_ = Status::kOk;
};

// 正常系: 出力の並びが入力の候補の並びを保つ。
TEST(DetectionEmitTest, order_follows_candidate_index) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const std::vector<CandidateInput> inputs = {make_candidate(0, 0, 100, 100, 11, 0, 0),
                                                make_candidate(200, 0, 300, 100, -1, 0, 0),
                                                make_candidate(0, 200, 100, 300, 22, 0, 0),
                                                make_candidate(200, 200, 300, 300, 33, 0, 0)};
    EmitRun run;
    ASSERT_TRUE(run.run(inputs, 1, 4, DetectorConfig()));
    ASSERT_EQ(run.count(), 3);
    EXPECT_EQ(run.count_status(), Status::kOk);
    EXPECT_EQ(run.detections()[0].id_, 11);
    EXPECT_EQ(run.detections()[1].id_, 22);
    EXPECT_EQ(run.detections()[2].id_, 33);
    EXPECT_EQ(run.detections()[0].source_, 0);
    EXPECT_EQ(run.detections()[1].source_, 2);
    EXPECT_EQ(run.detections()[2].source_, 3);
}

// 正常系: 四隅が OpenCV の correctCornerPosition と同じ並びになる。
TEST(DetectionEmitTest, rotation_correction_matches_opencv) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    // 四隅を互いに違う座標にして、並び替えを見分けられるようにする。
    CandidateInput base;
    const int xs[4] = {10, 90, 80, 20};
    const int ys[4] = {11, 21, 91, 81};
    for (int i = 0; i < 4; ++i) {
        base.x_[i] = xs[i];
        base.y_[i] = ys[i];
    }
    base.id_ = 5;
    base.depth_ = 0;

    for (int rotation = 0; rotation < 4; ++rotation) {
        CandidateInput input = base;
        input.rotation_ = rotation;
        EmitRun run;
        ASSERT_TRUE(run.run({input}, 1, 1, DetectorConfig())) << rotation;
        ASSERT_EQ(run.count(), 1) << rotation;
        EXPECT_EQ(run.detections()[0].rotation_, rotation) << rotation;
        for (int corner = 0; corner < 4; ++corner) {
            // OpenCV の std::rotate(begin, begin + 4 - rot, end) と同じ。
            const int source = (corner + 4 - rotation) % 4;
            EXPECT_FLOAT_EQ(run.detections()[0].x_[corner], static_cast<float>(xs[source]))
                    << "rotation=" << rotation << " corner=" << corner;
            EXPECT_FLOAT_EQ(run.detections()[0].y_[corner], static_cast<float>(ys[source]))
                    << "rotation=" << rotation << " corner=" << corner;
        }
    }
}

// 正常系: 同じ ID の検出を 2 件とも出す。
TEST(DetectionEmitTest, duplicate_ids_are_both_emitted) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    // 離れた位置に同じ ID のマーカーが 2 枚ある場面。OpenCV は 2 件とも出す。
    const std::vector<CandidateInput> inputs = {make_candidate(0, 0, 100, 100, 42, 0, 0),
                                                make_candidate(300, 300, 400, 400, 42, 0, 0)};
    EmitRun run;
    ASSERT_TRUE(run.run(inputs, 1, 2, DetectorConfig()));
    // ID による重複除去を入れたらここで 1 になり落ちる。
    ASSERT_EQ(run.count(), 2);
    EXPECT_EQ(run.detections()[0].id_, 42);
    EXPECT_EQ(run.detections()[1].id_, 42);
    EXPECT_FLOAT_EQ(run.detections()[0].x_[0], 0.0F);
    EXPECT_FLOAT_EQ(run.detections()[1].x_[0], 300.0F);
}

// 正常系: 走査が届かなかった候補は ID があっても出さない。
TEST(DetectionEmitTest, suppressed_candidate_is_not_emitted) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    // 段数 1 の候補は ID があるが、打ち切りが 1 なので届かない。
    const std::vector<CandidateInput> inputs = {make_candidate(0, 0, 300, 300, 77, 0, 1),
                                                make_candidate(50, 50, 250, 250, 88, 0, 0)};
    EmitRun run;
    ASSERT_TRUE(run.run(inputs, 1, 2, DetectorConfig()));
    ASSERT_EQ(run.count(), 1);
    EXPECT_EQ(run.detections()[0].id_, 88);
    EXPECT_EQ(run.detections()[0].source_, 1);
}

// 境界値: 候補上限が候補数より大きくても、前回の値が漏れない。
//
// 実運用は候補上限 4096 に対し候補が数件である。述語 kernel が候補数の外を
// 0 で埋めないと、前の frame の採否が scan に混ざって検出数が壊れる。同じ
// EmitRun を 2 度使い、1 度目の値が残っている状態で 2 度目を測る。
TEST(DetectionEmitTest, stale_values_beyond_candidate_count_are_cleared) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    constexpr int kCapacity = 64;
    std::vector<CandidateInput> many;
    many.reserve(static_cast<std::size_t>(kCapacity));
    for (int i = 0; i < kCapacity; ++i) {
        const int left = (i % 8) * 40;
        const int top = (i / 8) * 40;
        many.push_back(make_candidate(left, top, left + 30, top + 30, i, 0, 0));
    }

    EmitRun run;
    ASSERT_TRUE(run.run(many, 1, kCapacity, DetectorConfig(), kCapacity));
    ASSERT_EQ(run.count(), kCapacity);

    // 同じ workspace を使い回して候補を 2 件に減らす。
    const std::vector<CandidateInput> few = {make_candidate(0, 0, 30, 30, 5, 0, 0),
                                             make_candidate(100, 100, 130, 130, -1, 0, 0)};
    ASSERT_TRUE(run.run(few, 1, kCapacity, DetectorConfig(), kCapacity));
    EXPECT_EQ(run.count(), 1);
    EXPECT_EQ(run.count_status(), Status::kOk);
    ASSERT_EQ(run.detections().size(), 1U);
    EXPECT_EQ(run.detections()[0].id_, 5);
    EXPECT_EQ(run.detections()[0].source_, 0);
}

// 境界値: 上限を超えたら打ち切り、kMarkerOverflow を返す。
TEST(DetectionEmitTest, reports_marker_overflow) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const std::vector<CandidateInput> inputs = {make_candidate(0, 0, 100, 100, 1, 0, 0),
                                                make_candidate(200, 0, 300, 100, 2, 0, 0),
                                                make_candidate(0, 200, 100, 300, 3, 0, 0)};
    EmitRun run;
    ASSERT_TRUE(run.run(inputs, 1, 2, DetectorConfig()));
    EXPECT_EQ(run.count(), 2);
    EXPECT_EQ(run.count_status(), Status::kMarkerOverflow);
    EXPECT_EQ(run.detections()[0].id_, 1);
    EXPECT_EQ(run.detections()[1].id_, 2);
}

// 境界値: 採用が 0 件でも成立する。
TEST(DetectionEmitTest, handles_no_accepted_candidate) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const std::vector<CandidateInput> inputs = {make_candidate(0, 0, 100, 100, -1, 0, 0),
                                                make_candidate(200, 0, 300, 100, -1, 0, 0)};
    EmitRun run;
    ASSERT_TRUE(run.run(inputs, 1, 2, DetectorConfig()));
    EXPECT_EQ(run.count(), 0);
    EXPECT_EQ(run.count_status(), Status::kOk);
}

// 正常系: 同じ入力を 2 度流すと同じ結果になる。
TEST(DetectionEmitTest, results_are_deterministic) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const std::vector<CandidateInput> inputs = {make_candidate(0, 0, 100, 100, 3, 2, 0),
                                                make_candidate(200, 0, 300, 100, -1, 0, 0),
                                                make_candidate(0, 200, 100, 300, 3, 1, 0)};
    EmitRun first;
    EmitRun second;
    ASSERT_TRUE(first.run(inputs, 1, 3, DetectorConfig()));
    ASSERT_TRUE(second.run(inputs, 1, 3, DetectorConfig()));
    ASSERT_EQ(first.count(), second.count());
    for (int i = 0; i < first.count(); ++i) {
        const auto index = static_cast<std::size_t>(i);
        EXPECT_EQ(first.detections()[index].id_, second.detections()[index].id_);
        EXPECT_EQ(first.detections()[index].source_, second.detections()[index].source_);
        for (int corner = 0; corner < 4; ++corner) {
            EXPECT_FLOAT_EQ(first.detections()[index].x_[corner],
                            second.detections()[index].x_[corner]);
        }
    }
}

// 異常系: 引数が不正なら実行しない。
TEST(DetectionEmitTest, rejects_invalid_arguments) {
    Workspace workspace;
    DetectorConfig config;
    aruco3cuda::detail::DetectionEmitBuffers buffers;
    aruco3cuda::detail::DeviceDetections detections;
    EXPECT_EQ(aruco3cuda::detail::reserve_detections(config, workspace, nullptr, &detections),
              Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::detail::reserve_detections(config, workspace, &buffers, nullptr),
              Status::kInvalidArgument);
    EXPECT_NE(aruco3cuda::detail::reserve_detections(config, workspace, &buffers, &detections),
              Status::kOk);

    aruco3cuda::detail::DeviceCandidates candidates;
    aruco3cuda::detail::MatchBuffers matches;
    aruco3cuda::detail::CandidateTreeBuffers tree;
    EXPECT_EQ(aruco3cuda::detail::emit_detections_async(candidates, matches, tree, nullptr,
                                                        &detections, nullptr),
              Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::detail::emit_detections_async(candidates, matches, tree, &buffers,
                                                        nullptr, nullptr),
              Status::kInvalidArgument);

    int count = 0;
    EXPECT_EQ(aruco3cuda::detail::read_detection_count(detections, nullptr, nullptr),
              Status::kInvalidArgument);
    aruco3cuda::detail::DeviceDetections empty;
    EXPECT_EQ(aruco3cuda::detail::read_detection_count(empty, &count, nullptr),
              Status::kInvalidArgument);

    // 検出上限が候補上限を超える設定は成立しない。
    config.max_markers_ = config.max_candidates_ + 1;
    EXPECT_EQ(aruco3cuda::detail::detection_workspace_bytes(config), 0U);
    config.max_markers_ = 1024;
    // doc の出力例と同じ値を固定する。ずれると workspace が足りなくなる。
    EXPECT_EQ(aruco3cuda::detail::detection_workspace_bytes(config), 62464U);
}

}  // namespace
