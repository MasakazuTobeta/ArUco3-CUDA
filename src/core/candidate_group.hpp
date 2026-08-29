// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CORE_CANDIDATE_GROUP_HPP
#define ARUCO3CUDA_CORE_CANDIDATE_GROUP_HPP

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/workspace.hpp"
#include "candidate_filter.hpp"
#include "scan.hpp"

namespace aruco3cuda::detail {

/// 近接候補の統合に使う作業領域。
///
/// 所有権: 全ての pointer が指す領域の所有権は workspace にある。
/// 同期動作: 単なる参照の集合であり同期点を持たない。
///
/// 入力例: 候補上限 4096 での reserve_candidate_groups
/// 出力例: capacity_ = 4096 で各配列に pointer が入る
struct CandidateGroupBuffers {
    /// 周長の降順における順位。添字は候補 index。
    std::int32_t* rank_ = nullptr;
    /// 順位から候補 index への写像。rank_ の逆。
    std::int32_t* order_ = nullptr;
    /// 順位空間の union-find。根が group の代表になる。
    ///
    /// 順位が小さいほど周長が大きい。根を最小の順位に取れば、そのまま
    /// 「group 内で最大周長」を選んだことになる。
    std::int32_t* parent_ = nullptr;
    /// 代表なら 1。
    std::int32_t* selected_ = nullptr;
    /// 代表を詰めるための排他 scan の入出力。
    std::int32_t* offsets_ = nullptr;
    ScanBuffers scan_;
    int capacity_ = 0;
};

/// 統合に必要な workspace の容量を返す。
///
/// @param config 検出設定。候補上限を使う。
/// @return 必要な byte 数。引数が不正なら 0。
///
/// 所有権: 資源を保持しない。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: 候補上限 4096 の設定
/// 出力例: 作業領域と統合後の候補 4096 個分を収める byte 数
std::size_t candidate_group_workspace_bytes(const DetectorConfig& config);

/// workspace から統合用の領域を切り出す。
///
/// @param config 検出設定。候補上限を使う。
/// @param workspace 切り出し元。呼出側が所有する。
/// @param out_buffers 成功時に作業領域を格納する。nullptr は不可。
/// @param out_grouped 成功時に統合後の候補領域を格納する。nullptr は不可。
/// @return kOk。容量不足なら kInvalidConfig、引数が不正なら kInvalidArgument。
///
/// 所有権: 切り出した領域の所有権は workspace に残る。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: 既定設定と十分な容量の workspace
/// 出力例: capacity_ = 4096 で pointer が入る
Status reserve_candidate_groups(const DetectorConfig& config, Workspace& workspace,
                                CandidateGroupBuffers* out_buffers, DeviceCandidates* out_grouped);

/// 近接する候補を 1 つへまとめる。
///
/// 二値化 window を変えると同じマーカーから少しずつ違う候補が得られる。
/// 四隅の平均距離が周長に対する比より近いものを同じ group とし、group
/// 内で最も周長が大きい候補を残す。小さい候補を残すと四隅が内側へ寄る。
///
/// CPU 基準との違いが 1 つある。OpenCV は近接する 2 つが既に別々の group
/// へ属している場合、その 2 つを統合しない。本実装は近接関係の連結成分を
/// そのまま group とするため、この場合も統合する。3 つ以上が数珠つなぎに
/// 近接する配置でのみ差が出る。差の大きさは CPU 基準との比較で実測している。
///
/// @param input build_candidates_async が埋めた候補一式。
/// @param config 検出設定。
/// @param buffers reserve_candidate_groups が返した作業領域。nullptr は不可。
/// @param grouped reserve_candidate_groups が返した出力領域。nullptr は不可。
/// @param stream 発行先の stream。既定 stream を使う場合は nullptr。
/// @return kOk、または kInvalidArgument、kCudaError。
///
/// 所有権: 引数が指す領域の所有権は workspace に残る。
/// 同期動作: stream へ kernel を発行するだけで host 同期を行わない。
///
/// 入力例: 同じマーカーから得た周長違いの候補が 3 つ
/// 出力例: count_ = 1、最も周長が大きい候補だけが残る
Status build_candidate_groups_async(const DeviceCandidates& input, const DetectorConfig& config,
                                    CandidateGroupBuffers* buffers, DeviceCandidates* grouped,
                                    cudaStream_t stream);

}  // namespace aruco3cuda::detail

#endif  // ARUCO3CUDA_CORE_CANDIDATE_GROUP_HPP
