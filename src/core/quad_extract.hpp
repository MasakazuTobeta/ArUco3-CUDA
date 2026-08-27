// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CORE_QUAD_EXTRACT_HPP
#define ARUCO3CUDA_CORE_QUAD_EXTRACT_HPP

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "aruco3cuda/status.hpp"
#include "aruco3cuda/workspace.hpp"
#include "labeling.hpp"

namespace aruco3cuda::detail {

/// 四隅の点数。
constexpr int kQuadCornerCount = 4;

/// label ごとの四隅と、その抽出に使う作業領域。
///
/// 四隅は極点探索で求める。重心から最も遠い点を c0、c0 から最も遠い点を
/// c2 とし、直線 c0c2 の左右それぞれで最も離れた点を c1 と c3 とする。
/// 輪郭を順に辿る必要がないため、label 単位で完全に並列化できる。
///
/// 所有権: 全ての pointer が指す領域の所有権は workspace にある。
/// 同期動作: 単なる参照の集合であり同期点を持たない。内容は発行済みの
///           kernel が完了するまで確定しない。
///
/// 入力例: 427x240 の label 画像
/// 出力例: capacity_ = 25680、corner_x_ が 4 * 25680 要素
struct QuadBuffers {
    /// 四隅の x 座標。添字は (corner * capacity_) + label。
    ///
    /// 角ごとに配列を分けて並べるのは、1 つの kernel が同じ角だけを
    /// 走査するためである。label 方向に連続していれば読み書きがまとまる。
    std::int32_t* corner_x_ = nullptr;
    /// 四隅の y 座標。並びは corner_x_ と同じ。
    std::int32_t* corner_y_ = nullptr;
    /// 四隅を取り出せたか。1 で有効、0 で無効。
    ///
    /// 1 画素だけの成分や直線状の成分は、直線 c0c2 の片側に点を持たない。
    /// この場合は四隅が定まらないため無効とする。
    std::uint8_t* valid_ = nullptr;
    /// 探索の途中結果。距離と画素 index を 1 語へ詰めたもの。
    ///
    /// 上位 32 bit に距離、下位 32 bit に画素の線形 index を置く。
    /// atomicMax 1 回で「最も遠い点」と「その位置」を同時に決められる。
    /// 距離が等しい場合は index の大きい方が残るため、結果は実行順に
    /// 依存しない。
    unsigned long long* best_ = nullptr;
    unsigned long long* best_positive_ = nullptr;
    unsigned long long* best_negative_ = nullptr;
    /// 確保した label 数。
    int capacity_ = 0;
};

/// 四隅抽出に必要な workspace の容量を返す。
///
/// @param width_px 画像の幅。
/// @param height_px 画像の高さ。
/// @return 必要な byte 数。桁溢れや引数が不正なら 0。
///
/// 所有権: 資源を保持しない。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: 427 と 240
/// 出力例: 25680 label 分の四隅と作業領域を収める byte 数
std::size_t quad_workspace_bytes(int width_px, int height_px);

/// workspace から四隅抽出用の領域を切り出す。
///
/// @param width_px 画像の幅。1 以上。
/// @param height_px 画像の高さ。1 以上。
/// @param workspace 切り出し元。呼出側が所有する。
/// @param out 成功時に buffer 一式を格納する。nullptr は不可。
/// @return kOk。容量不足なら kInvalidConfig、引数が不正なら kInvalidArgument。
///
/// 所有権: 切り出した領域の所有権は workspace に残る。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: 427 と 240 と十分な容量の workspace
/// 出力例: capacity_ = 25680 で各配列に pointer が入る
Status reserve_quads(int width_px, int height_px, Workspace& workspace, QuadBuffers* out);

/// label ごとに極点探索で四隅を求める。
///
/// 四隅の並びは OpenCV の `_reorderCandidatesCorners` と同じ向きへ揃える。
/// 起点の角は成分の形で決まるため、CPU 基準と同じ角から始まるとは限らない。
/// 起点の違いは Dictionary 照合が返す回転量に吸収される。
///
/// @param labels build_labels_async が埋めた label 一式。
/// @param stats build_label_stats_async が埋めた統計。重心を起点に使う。
/// @param quads reserve_quads が返した buffer 一式。nullptr は不可。
/// @param stream 発行先の stream。既定 stream を使う場合は nullptr。
/// @return kOk、または kInvalidArgument、kCudaError。
///
/// 所有権: 引数が指す領域の所有権は workspace に残る。
/// 同期動作: stream へ kernel を発行するだけで host 同期を行わない。
///           探索は 3 段階で、各段の結果を次段が使うため順に実行される。
///
/// 入力例: 1 つの正方形の枠が label 0 になっている label 画像
/// 出力例: valid_[0] = 1、corner_x_/corner_y_ が枠の 4 隅
Status build_quads_async(const LabelBuffers& labels, const LabelStatisticsBuffers& stats,
                         QuadBuffers* quads, cudaStream_t stream);

}  // namespace aruco3cuda::detail

#endif  // ARUCO3CUDA_CORE_QUAD_EXTRACT_HPP
