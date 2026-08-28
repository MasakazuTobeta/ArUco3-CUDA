// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CORE_DETECTION_EMIT_HPP
#define ARUCO3CUDA_CORE_DETECTION_EMIT_HPP

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/workspace.hpp"
#include "candidate_filter.hpp"
#include "candidate_tree.hpp"
#include "dictionary_match.hpp"
#include "scan.hpp"

namespace aruco3cuda::detail {

/// device 上に留まる検出結果。
///
/// 四隅は Dictionary 照合で得た回転を打ち消した後の並びである。OpenCV も
/// 識別の直後、subpixel 補正の前に打ち消す。順序を入れ替えると補正が別の
/// 隅に掛かる。
///
/// 所有権: 全ての pointer が指す領域の所有権は workspace にある。
/// 同期動作: 単なる参照の集合であり同期点を持たない。内容は発行済みの
///           kernel が完了するまで確定しない。
///
/// 入力例: max_markers_ = 1024 の設定
/// 出力例: ids_ が 1024 要素、corner_x_ が 4096 要素
struct DeviceDetections {
    /// 一致した Dictionary の ID。
    std::int32_t* ids_ = nullptr;
    /// 一致した回転。0 から 3。四隅は打ち消し済みだが、姿勢推定側が
    /// 元の向きを必要とする場合があるため値も残す。
    std::int32_t* rotations_ = nullptr;
    /// 四隅の x 座標。添字は (corner * capacity_) + detection。
    ///
    /// 単精度で持つ。入力は整数だが、S10 の subpixel 補正がここへ書き戻す。
    /// 2^24 未満の整数は単精度で厳密に表せるため、変換で値は変わらない。
    float* corner_x_ = nullptr;
    /// 四隅の y 座標。並びは corner_x_ と同じ。
    float* corner_y_ = nullptr;
    /// 由来した候補の index。S10 が周長や label を引くために残す。
    std::int32_t* source_ = nullptr;
    /// 詰めた検出数。要素数 1。上限で打ち切った後の値が入る。
    std::int32_t* count_ = nullptr;
    /// 打ち切る前の検出数。要素数 1。count_ より大きければ捨てている。
    std::int32_t* accepted_total_ = nullptr;
    int capacity_ = 0;
};

/// compaction の作業領域。
///
/// 所有権: 全ての pointer が指す領域の所有権は workspace にある。
/// 同期動作: 単なる参照の集合であり同期点を持たない。
///
/// 入力例: max_candidates_ = 4096 の設定
/// 出力例: offsets_ が 4096 要素
struct DetectionEmitBuffers {
    /// 候補ごとの採否 (0 か 1)。scan の後は書き出し先の index になる。
    std::int32_t* offsets_ = nullptr;
    ScanBuffers scan_;
    int capacity_ = 0;
};

/// 検出結果と作業領域に必要な workspace の大きさを返す。
///
/// 述語は候補の空間 (max_candidates_)、出力は検出の空間 (max_markers_) で
/// あり、要素数が違う。scan は候補の空間で行う。
///
/// @param config 候補上限と検出上限を含む設定。
/// @return 必要な byte 数。config が不正なら 0。
///
/// 所有権: 引数の領域を保持しない。
/// 同期動作: host 専用であり同期点を持たない。CUDA API を呼ばない。
///
/// 入力例: max_candidates_ = 4096、max_markers_ = 1024 の設定
/// 出力例: 62464
std::size_t detection_workspace_bytes(const DetectorConfig& config);

/// 検出結果と作業領域を確保する。
///
/// @param config 候補上限と検出上限を含む設定。
/// @param workspace device 空間の workspace。
/// @param out_buffers 成功時に作業領域を格納する。領域の所有権は呼出側にある。
/// @param out 成功時に検出結果の参照を格納する。領域の所有権は呼出側にある。
/// @return kOk。引数が不正なら kInvalidArgument、容量不足なら kInvalidConfig。
///
/// 所有権: 引数の領域を保持しない。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: 既定の設定、空きのある workspace
/// 出力例: kOk。out->capacity_ = 1024、out_buffers->capacity_ = 4096
Status reserve_detections(const DetectorConfig& config, Workspace& workspace,
                          DetectionEmitBuffers* out_buffers, DeviceDetections* out);

/// 採用した候補を詰めて検出結果を作る。
///
/// 採用の条件は次の 3 つを全て満たすことである。
///
/// 1. 候補数の範囲内にある
/// 2. 段数が打ち切りの段数未満である (走査が届いている)
/// 3. Dictionary 照合で ID が付いている
///
/// **同じ ID の検出は落とさない。** OpenCV も ID による重複除去を持たない。
/// 離れた位置に同じマーカーが 2 枚あれば 2 件とも出す。重複が消えるのは
/// 包含木による打ち切りの結果であり、ID の比較によるものではない。
///
/// 出力の並びは入力の候補の並びをそのまま保つ。統合後の候補は周長の降順に
/// 並んでいるため、検出も周長の降順になる。OpenCV の accepted と同じ順である。
///
/// @param grouped 統合後の候補。
/// @param matches 照合結果。
/// @param tree 段数と打ち切りの段数を格納済みの buffer。
/// @param buffers 作業領域。
/// @param out 出力先。
/// @param stream kernel を発行する stream。
/// @return kOk。引数が不正なら kInvalidArgument、kernel 起動に失敗したら
///         kCudaError。上限超過はここでは返さない。read_detection_count で
///         受け取る。
///
/// 所有権: 引数の領域を保持しない。
/// 同期動作: kernel を stream 上で非同期に発行する。呼出側が同期するまで
///           結果は確定しない。
///
/// 入力例: 候補 3 件、うち 2 件に ID あり
/// 出力例: kOk。count_ = 2
Status emit_detections_async(const DeviceCandidates& grouped, const MatchBuffers& matches,
                             const CandidateTreeBuffers& tree, DetectionEmitBuffers* buffers,
                             DeviceDetections* out, cudaStream_t stream);

/// 検出数を host へ読み出す。
///
/// @param detections 対象。
/// @param out_count 成功時に検出数を格納する。領域の所有権は呼出側にある。
/// @param stream 同期する stream。
/// @return kOk。上限で打ち切った場合は kMarkerOverflow。引数が不正なら
///         kInvalidArgument、CUDA API が失敗したら kCudaError。
///
/// 所有権: 引数の領域を保持しない。
/// 同期動作: **stream を同期する。** この関数だけが host 同期を伴う。
///
/// 入力例: 検出 2 件、上限 1024
/// 出力例: kOk。*out_count = 2
/// 入力例: 検出 3 件、上限 2
/// 出力例: kMarkerOverflow。*out_count = 2
Status read_detection_count(const DeviceDetections& detections, int* out_count,
                            cudaStream_t stream);

}  // namespace aruco3cuda::detail

#endif  // ARUCO3CUDA_CORE_DETECTION_EMIT_HPP
