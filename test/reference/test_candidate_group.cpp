// SPDX-License-Identifier: Apache-2.0
//
// 近接候補の統合を検証する。
//
// 二値化 window を変えると同じマーカーから少しずつ違う候補が得られる。
// どれを残すかで四隅の位置が変わるため、最も周長が大きいものを残すことを
// 固定する。CPU 基準との既知の違い (数珠つなぎの扱い) も明示的に置く。
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

/// 試験用の候補 1 つ。四隅は軸に平行な正方形として与える。
struct InputQuad {
    int x_ = 0;
    int y_ = 0;
    int side_ = 0;
    int label_ = 0;
};

/// 統合の入力と出力をまとめて扱う。
class GroupRun {
public:
    GroupRun() = default;
    GroupRun(const GroupRun&) = delete;
    GroupRun& operator=(const GroupRun&) = delete;

    /// 候補を device へ載せて統合する。
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
    /// 候補を device の配列へ書き込む。四隅は正方形として展開する。
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

/// 候補上限を小さくした設定。試験の確保量を抑える。
DetectorConfig small_config() {
    DetectorConfig config;
    config.max_candidates_ = 64;
    return config;
}

// 正常系: 近接する候補は 1 つにまとまり、最も周長が大きいものが残る。
TEST(CandidateGroupTest, keeps_largest_perimeter_in_group) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    // 同じマーカーから window 違いで得た 3 つ。辺長がわずかに異なる。
    const std::vector<InputQuad> quads = {{100, 100, 60, 0}, {101, 101, 58, 1}, {99, 99, 62, 2}};
    GroupRun run;
    ASSERT_TRUE(run.run(quads, small_config()));
    ASSERT_EQ(run.count(), 1);
    EXPECT_EQ(run.results()[0].side_, 62);
    EXPECT_EQ(run.results()[0].label_, 2);
}

// 正常系: 離れた候補はまとまらない。
TEST(CandidateGroupTest, keeps_distant_candidates_separate) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const std::vector<InputQuad> quads = {
            {20, 20, 40, 0}, {200, 20, 40, 1}, {20, 200, 40, 2}, {200, 200, 40, 3}};
    GroupRun run;
    ASSERT_TRUE(run.run(quads, small_config()));
    EXPECT_EQ(run.count(), 4);
}

// 正常系: 複数の group が混在しても、それぞれの代表が残る。
TEST(CandidateGroupTest, handles_multiple_groups) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
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
    // 各 group で大きい方が残る。並びは周長の降順。
    EXPECT_EQ(sides, (std::vector<int>{50, 44, 40}));
}

// 境界値: 候補が 1 つならそのまま残る。
TEST(CandidateGroupTest, single_candidate_passes_through) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    GroupRun run;
    ASSERT_TRUE(run.run({{50, 50, 40, 7}}, small_config()));
    ASSERT_EQ(run.count(), 1);
    EXPECT_EQ(run.results()[0].label_, 7);
}

// 境界値: 候補が無ければ結果も 0 件になる。
TEST(CandidateGroupTest, empty_input_produces_no_group) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    GroupRun run;
    ASSERT_TRUE(run.run({}, small_config()));
    EXPECT_EQ(run.count(), 0);
}

// 既知の違い: 数珠つなぎに近接する 3 つは 1 つへまとまる。
//
// CPU 基準は、既に別々の group へ属している 2 つを統合しない。両端が
// 直接は近接していないこの配置では、CPU 基準が 2 つ残すのに対し本実装は
// 1 つにまとめる。差の大きさは WP-2.6 で実測する。
TEST(CandidateGroupTest, known_difference_chained_candidates_merge) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    // 隣とは近く、両端どうしは遠い配置。
    const std::vector<InputQuad> quads = {{100, 100, 60, 0}, {112, 100, 60, 1}, {124, 100, 60, 2}};
    GroupRun run;
    ASSERT_TRUE(run.run(quads, small_config()));
    EXPECT_EQ(run.count(), 1);
}

// 正常系: 同じ入力からは同じ結果が得られる。
TEST(CandidateGroupTest, results_are_deterministic) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
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

// 異常系: 引数が不正なら実行しない。
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
