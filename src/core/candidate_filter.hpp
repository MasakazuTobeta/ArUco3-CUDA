// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CORE_CANDIDATE_FILTER_HPP
#define ARUCO3CUDA_CORE_CANDIDATE_FILTER_HPP

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/workspace.hpp"
#include "labeling.hpp"
#include "quad_extract.hpp"
#include "scan.hpp"

namespace aruco3cuda::detail {

/// 篩を通った候補を詰めて並べた出力。
///
/// label は画像全体に散らばるため、そのまま下流へ渡すと大半が空の
/// 添字になる。詰めておくと以降の段階が候補数だけを走査すればよい。
///
/// 所有権: 全ての pointer が指す領域の所有権は workspace にある。
/// 同期動作: 単なる参照の集合であり同期点を持たない。
///
/// 入力例: 候補上限 4096 での reserve_candidates
/// 出力例: capacity_ = 4096、corner_x_ が 4 * 4096 要素
struct DeviceCandidates {
    /// 四隅の x 座標。添字は (corner * capacity_) + candidate。
    std::int32_t* corner_x_ = nullptr;
    /// 四隅の y 座標。並びは corner_x_ と同じ。
    std::int32_t* corner_y_ = nullptr;
    /// 由来した label。どの成分から来た候補かを後から辿れるようにする。
    std::int32_t* label_ = nullptr;
    /// 四隅を結んだ折れ線の chain code 長。
    std::int32_t* perimeter_ = nullptr;
    /// 詰めた候補数。要素数 1。上限で打ち切った後の値が入る。
    std::int32_t* count_ = nullptr;
    /// 篩を通った候補数。要素数 1。上限を超えた場合はここが count_ より大きい。
    std::int32_t* accepted_total_ = nullptr;
    int capacity_ = 0;
};

/// 候補の篩と詰め込みに使う作業領域。
///
/// 所有権: 全ての pointer が指す領域の所有権は workspace にある。
/// 同期動作: 単なる参照の集合であり同期点を持たない。
///
/// 入力例: 427x240 と候補上限 4096
/// 出力例: label 25680 個分の判定結果と scan の作業領域
struct CandidateFilterBuffers {
    /// label ごとの合否。1 で合格。
    std::int32_t* accepted_ = nullptr;
    /// 合否を排他 scan したもの。合格した label の書き込み先になる。
    std::int32_t* offsets_ = nullptr;
    /// 推定四角形の内側にある成分画素の数。
    std::int32_t* inside_count_ = nullptr;
    /// 四隅を結んだ折れ線の chain code 長。label ごと。
    std::int32_t* perimeter_ = nullptr;
    /// 辺ごとの裏付け画素数。添字は (corner * capacity_) + label。
    std::int32_t* edge_support_ = nullptr;
    /// 書き込みを始める位置。要素数 1。
    ///
    /// 二値化 window ごとに候補を求めて 1 つの配列へ連ねるため、直前まで
    /// の件数を device 側で保持する。host へ戻して足すと window ごとに
    /// 同期が要る。
    std::int32_t* base_ = nullptr;
    ScanBuffers scan_;
    int capacity_ = 0;
};

/// 篩と詰め込みに必要な workspace の容量を返す。
///
/// @param config 検出設定。候補上限を使う。
/// @param width_px 画像の幅。
/// @param height_px 画像の高さ。
/// @return 必要な byte 数。桁溢れや引数が不正なら 0。
///
/// 所有権: 資源を保持しない。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: 既定設定と 427 と 240
/// 出力例: label 25680 個分の作業領域と候補 4096 個分を収める byte 数
std::size_t candidate_workspace_bytes(const DetectorConfig& config, int width_px, int height_px);

/// workspace から篩と詰め込み用の領域を切り出す。
///
/// @param config 検出設定。候補上限を使う。
/// @param width_px 画像の幅。1 以上。
/// @param height_px 画像の高さ。1 以上。
/// @param workspace 切り出し元。呼出側が所有する。
/// @param out_buffers 成功時に作業領域を格納する。nullptr は不可。
/// @param out_candidates 成功時に出力領域を格納する。nullptr は不可。
/// @return kOk。容量不足なら kInvalidConfig、引数が不正なら kInvalidArgument。
///
/// 所有権: 切り出した領域の所有権は workspace に残る。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: 既定設定と 427 と 240 と十分な容量の workspace
/// 出力例: capacity_ が label 数の上限と候補上限にそれぞれ入る
Status reserve_candidates(const DetectorConfig& config, int width_px, int height_px,
                          Workspace& workspace, CandidateFilterBuffers* out_buffers,
                          DeviceCandidates* out_candidates);

/// 四隅の推定結果を篩に掛け、通ったものを詰めて並べる。
///
/// 篩は CPU 経路の `_findMarkerContours` に対応する判定へ写像する。
/// 周長は輪郭の画素数ではなく、四隅を結んだ折れ線の chain code 長を使う。
/// 極点探索は輪郭を持たないため画素数を数えられないが、直線の chain code
/// 長は端点から一意に決まる。凸四角形では両者はほぼ一致する。
///
/// 四角形らしさは `polygonalApproxAccuracyRate` の代わりに 2 つの比で測る。
/// 1 つは成分画素が推定四角形の内側へ収まる割合で、円や楕円のように外へ
/// はみ出す形を落とす。もう 1 つは各辺の近くにある成分画素の数で、L 字や
/// 十字のように辺が成分の外を通る形を落とす。片方だけでは分離できない
/// ことを合成図形で実測して確かめている。
///
/// 極点探索は三角形と、接触して 1 成分になった 2 枚のマーカーを四角形と
/// して受け入れる。どちらの判定も通るためであり、案 A の既知の限界である。
/// 影響の大きさは CPU 基準との候補比較で評価している。
///
/// 書き込み先は排他 scan で決める。atomicAdd で場所を取ると到着順で
/// 並びが変わり、実行ごとに候補の順序が変わる。
///
/// @param labels build_labels_async が埋めた label 一式。
/// @param stats build_label_stats_async が埋めた統計。
/// @param quads build_quads_async が埋めた四隅。
/// @param config 検出設定。
/// @param buffers reserve_candidates が返した作業領域。nullptr は不可。
/// @param candidates reserve_candidates が返した出力領域。nullptr は不可。
/// @param append true なら既に入っている候補の後ろへ書き足す。false なら
///               先頭から書き直す。二値化 window ごとに呼ぶ場合、最初の
///               window だけ false にする。
/// @param stream 発行先の stream。既定 stream を使う場合は nullptr。
/// @return kOk、または kInvalidArgument、kCudaError。
///         上限超過は非同期のため、ここでは判定しない。
///
/// 所有権: 引数が指す領域の所有権は workspace に残る。
/// 同期動作: stream へ kernel を発行するだけで host 同期を行わない。
///
/// 入力例: マーカーの黒枠 4 つと noise の成分を含む label 画像
/// 出力例: count_ = 4、corner_x_/corner_y_ に 4 つ分の四隅
Status build_candidates_async(const LabelBuffers& labels, const LabelStatisticsBuffers& stats,
                              const QuadBuffers& quads, const DetectorConfig& config,
                              CandidateFilterBuffers* buffers, DeviceCandidates* candidates,
                              bool append, cudaStream_t stream);

/// 候補数を host へ読み出し、打ち切りの有無を返す。
///
/// @param candidates build_candidates_async へ渡した出力領域。
/// @param out_count 成功時に候補数を格納する。nullptr は不可。
/// @param stream 発行先の stream。転送の完了まで待つ。
/// @return kOk。上限で打ち切った場合は kCandidateOverflow。
///         引数が不正なら kInvalidArgument、転送の失敗は kCudaError。
///
/// 所有権: 引数の領域を保持しない。
/// 同期動作: stream の完了を待つ。呼び出しから戻った時点で out_count は確定する。
///
/// 入力例: 篩を 5000 件が通り、上限が 4096 の場合
/// 出力例: out_count = 4096、戻り値は kCandidateOverflow
Status read_candidate_count(const DeviceCandidates& candidates, int* out_count,
                            cudaStream_t stream);

}  // namespace aruco3cuda::detail

#endif  // ARUCO3CUDA_CORE_CANDIDATE_FILTER_HPP
