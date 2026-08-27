// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CORE_LABELING_HPP
#define ARUCO3CUDA_CORE_LABELING_HPP

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "aruco3cuda/status.hpp"
#include "aruco3cuda/workspace.hpp"
#include "preprocess.hpp"

namespace aruco3cuda::detail {

/// 背景画素に入る label。
///
/// 前景の label は 0 から始まる連番であるため、負値を背景へ割り当てる。
/// 0 を背景に使うと label の連番を 1 起点にする必要があり、統計配列の
/// 添字と label が 1 ずれる。ずれは取り違えの原因になるため避ける。
constexpr std::int32_t kBackgroundLabel = -1;

/// 連結成分ラベリングの出力と作業領域。
///
/// 所有権: 全ての pointer が指す領域の所有権は workspace にある。この構造体は
///         参照のみを持ち、複製も解放も行わない。workspace を reset() または
///         破棄すると全ての pointer が無効になる。
/// 同期動作: 単なる参照の集合であり同期点を持たない。内容は発行済みの
///           kernel が完了するまで確定しない。
///
/// 入力例: 427x240 の二値化画像
/// 出力例: labels_ が 102480 要素、label 数が label_count_ に入る
struct LabelBuffers {
    /// 画素ごとの label。要素数は width_px_ * height_px_。
    ///
    /// pitch を持たせず、行方向へ詰めて並べる。union-find は画素を線形
    /// index で辿るため、pitch があると添字から address への変換が
    /// 内側の loop へ入る。ラベリングでは 2 次元の近傍参照より線形走査の
    /// 方が支配的であり、pitch の利点より変換の costs が上回る。
    std::int32_t* labels_ = nullptr;
    /// root の線形 index から詰めた label への写像。要素数は labels_ と同じ。
    ///
    /// scan の入出力を兼ねる。root であることを示す 1/0 を書き込んでから
    /// 排他 scan を掛けると、そのまま詰めた label になる。
    std::int32_t* compact_ids_ = nullptr;
    /// scan の block ごとの開始位置。要素数は scan_block_count_。
    std::int32_t* block_offsets_ = nullptr;
    int scan_block_count_ = 0;
    /// device 側の label 数。要素数 1。
    std::int32_t* label_count_ = nullptr;
    int width_px_ = 0;
    int height_px_ = 0;
};

/// 必要な workspace の容量を返す。
///
/// @param width_px 二値化画像の幅。
/// @param height_px 二値化画像の高さ。
/// @return 必要な byte 数。桁溢れや引数が不正なら 0。
///
/// 所有権: 資源を保持しない。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: 427 と 240
/// 出力例: label 配列 2 枚と scan の作業領域を収める byte 数
std::size_t labeling_workspace_bytes(int width_px, int height_px);

/// workspace からラベリング用の領域を切り出す。
///
/// @param width_px 二値化画像の幅。1 以上。
/// @param height_px 二値化画像の高さ。1 以上。
/// @param workspace 切り出し元。呼出側が所有する。
/// @param out 成功時に buffer 一式を格納する。nullptr は不可。
/// @return kOk。容量不足なら kInvalidConfig、引数が不正なら kInvalidArgument。
///
/// 所有権: 切り出した領域の所有権は workspace に残る。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: 427 と 240 と十分な容量の workspace
/// 出力例: labels_ と compact_ids_ に pointer が入る
Status reserve_labeling(int width_px, int height_px, Workspace& workspace, LabelBuffers* out);

/// 二値化画像の前景を 8 近傍で連結成分へ分ける。
///
/// 8 近傍とするのは、OpenCV の `findContours` が前景を 8 連結として辿る
/// ためである。4 近傍にすると、対角にのみ接する前景が別成分になり、
/// CPU 基準が 1 つの候補とする形が 2 つに割れる。
///
/// label は 0 から始まる連番で、値は root の線形 index の昇順に振る。
/// 実行ごとに同じ入力から同じ label が得られるようにするためであり、
/// atomics の到着順に依存する採番は使わない。
///
/// @param binary 入力二値化画像。0 を背景、それ以外を前景とみなす。
///               reserve_labeling と同じ寸法である必要がある。
/// @param buffers reserve_labeling が返した buffer 一式。nullptr は不可。
/// @param stream 発行先の stream。既定 stream を使う場合は nullptr。
/// @return kOk、または kInvalidArgument、kCudaError。
///
/// 所有権: buffers が指す領域の所有権は workspace に残る。
/// 同期動作: stream へ kernel を発行するだけで host 同期を行わない。
///
/// 入力例: 中央に 1 つの正方形がある 427x240 の二値化画像
/// 出力例: 正方形の画素が label 0、他が kBackgroundLabel
Status build_labels_async(const ImagePlaneU8& binary, LabelBuffers* buffers, cudaStream_t stream);

/// label 数を host へ読み出す。
///
/// @param buffers build_labels_async へ渡した buffer 一式。
/// @param out_count 成功時に label 数を格納する。nullptr は不可。
/// @param stream 発行先の stream。転送の完了まで待つ。
/// @return kOk、または kInvalidArgument、kCudaError。
///
/// 所有権: 引数の領域を保持しない。
/// 同期動作: stream の完了を待つ。呼び出しから戻った時点で out_count は確定する。
///
/// 入力例: 正方形が 4 つある画像を処理した後の buffer
/// 出力例: out_count = 4
Status read_label_count(const LabelBuffers& buffers, int* out_count, cudaStream_t stream);

/// label ごとの統計。構造体の配列ではなく配列の構造体として持つ。
///
/// 集計は label 単位の atomics であり、同じ項目へ多数の thread が集まる。
/// 項目ごとに配列を分けると、1 つの atomics が隣接項目の cache line を
/// 巻き込まない。
///
/// 所有権: 全ての pointer が指す領域の所有権は workspace にある。
/// 同期動作: 単なる参照の集合であり同期点を持たない。内容は発行済みの
///           kernel が完了するまで確定しない。
///
/// 入力例: 427x240 の画像に対する reserve_label_stats
/// 出力例: capacity_ = 214 * 120 = 25680
struct LabelStatisticsBuffers {
    /// 外接矩形。min は包含、max も包含する。
    std::int32_t* min_x_ = nullptr;
    std::int32_t* min_y_ = nullptr;
    std::int32_t* max_x_ = nullptr;
    std::int32_t* max_y_ = nullptr;
    /// 成分に属する画素数。
    std::int32_t* pixel_count_ = nullptr;
    /// 座標の総和。重心を求めるために保持する。
    ///
    /// 型を unsigned long long とするのは atomicAdd の overload に合わせる
    /// ためである。LP64 では std::uint64_t は unsigned long であり、
    /// pointer 型が一致しない。
    unsigned long long* sum_x_ = nullptr;
    unsigned long long* sum_y_ = nullptr;
    /// 重心。総和を画素数で割ったもの。
    float* centroid_x_ = nullptr;
    float* centroid_y_ = nullptr;
    /// 確保した label 数。max_label_count() と等しい。
    int capacity_ = 0;
};

/// 画像寸法から label 数の上限を返す。
///
/// 8 近傍では、別成分どうしは縦横斜めのいずれでも接してはならない。
/// 成分を最も多く作る配置は 1 画素を 1 つ飛ばしに置いたものであり、
/// 上限は ceil(W/2) * ceil(H/2) になる。この値で確保すれば label の
/// 溢れは起こらないため、統計の側に溢れの経路を持たなくてよい。
///
/// @param width_px 画像の幅。
/// @param height_px 画像の高さ。
/// @return label 数の上限。引数が不正なら 0。
///
/// 所有権: 資源を保持しない。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: 427 と 240
/// 出力例: 214 * 120 = 25680
int max_label_count(int width_px, int height_px);

/// 統計に必要な workspace の容量を返す。
///
/// @param width_px 画像の幅。
/// @param height_px 画像の高さ。
/// @return 必要な byte 数。桁溢れや引数が不正なら 0。
///
/// 所有権: 資源を保持しない。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: 427 と 240
/// 出力例: 25680 label 分の統計を収める byte 数
std::size_t label_stats_workspace_bytes(int width_px, int height_px);

/// workspace から統計用の領域を切り出す。
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
Status reserve_label_stats(int width_px, int height_px, Workspace& workspace,
                           LabelStatisticsBuffers* out);

/// label ごとの外接矩形、画素数、重心を集計する。
///
/// 集計は整数の atomics のみで行う。加算の順序が変わっても結果が変わらず、
/// 実行ごとに同じ値が得られる。重心は総和を画素数で割った時点で 1 度だけ
/// 求めるため、割り算の順序による差も生じない。
///
/// @param labels build_labels_async が埋めた label 一式。
/// @param stats reserve_label_stats が返した buffer 一式。nullptr は不可。
/// @param stream 発行先の stream。既定 stream を使う場合は nullptr。
/// @return kOk、または kInvalidArgument、kCudaError。
///
/// 所有権: 引数が指す領域の所有権は workspace に残る。
/// 同期動作: stream へ kernel を発行するだけで host 同期を行わない。
///
/// 入力例: 中央の 30x30 の正方形が label 0 になっている label 画像
/// 出力例: pixel_count_[0] = 900、外接矩形が正方形の範囲、重心が中心
Status build_label_stats_async(const LabelBuffers& labels, LabelStatisticsBuffers* stats,
                               cudaStream_t stream);

}  // namespace aruco3cuda::detail

#endif  // ARUCO3CUDA_CORE_LABELING_HPP
