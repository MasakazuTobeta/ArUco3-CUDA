// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CORE_CELL_SAMPLE_HPP
#define ARUCO3CUDA_CORE_CELL_SAMPLE_HPP

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/workspace.hpp"
#include "candidate_filter.hpp"
#include "preprocess.hpp"

namespace aruco3cuda::detail {

/// 候補ごとの canonical 画像。射影変換で正方形へ直したもの。
///
/// 1 辺は (marker_size + 2 * marker_border_bits_) * perspective_remove_pixel_per_cell_
/// である。既定設定と DICT_ARUCO_MIP_36h12 では 32 になる。
///
/// 所有権: pointer が指す領域の所有権は workspace にある。
/// 同期動作: 単なる参照であり同期点を持たない。内容は発行済みの kernel が
///           完了するまで確定しない。
///
/// 入力例: 既定設定、marker_size = 6、候補上限 4096
/// 出力例: side_px_ = 32、images_ が 4096 * 32 * 32 byte
struct CanonicalBuffers {
    /// 候補ごとの canonical 画像。添字は (candidate * side_px_ * side_px_) + (y * side_px_) + x。
    std::uint8_t* images_ = nullptr;
    int side_px_ = 0;
    int capacity_ = 0;
};

/// canonical 画像の 1 辺を返す。
///
/// @param config 検出設定。
/// @param marker_size Dictionary のセル数。1 以上。
/// @return 1 辺の画素数。引数が不正なら 0。
///
/// 所有権: 資源を保持しない。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: 既定設定と marker_size = 6
/// 出力例: 32
int canonical_side_px(const DetectorConfig& config, int marker_size);

/// canonical 画像に必要な workspace の容量を返す。
///
/// @param config 検出設定。候補上限を使う。
/// @param marker_size Dictionary のセル数。
/// @return 必要な byte 数。桁溢れや引数が不正なら 0。
///
/// 所有権: 資源を保持しない。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: 既定設定と marker_size = 6
/// 出力例: 4096 * 32 * 32 を 256 境界へ整えた byte 数
std::size_t canonical_workspace_bytes(const DetectorConfig& config, int marker_size);

/// workspace から canonical 画像の領域を切り出す。
///
/// @param config 検出設定。候補上限を使う。
/// @param marker_size Dictionary のセル数。1 以上。
/// @param workspace 切り出し元。呼出側が所有する。
/// @param out 成功時に buffer を格納する。nullptr は不可。
/// @return kOk。容量不足なら kInvalidConfig、引数が不正なら kInvalidArgument。
///
/// 所有権: 切り出した領域の所有権は workspace に残る。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: 既定設定と marker_size = 6 と十分な容量の workspace
/// 出力例: side_px_ = 32、capacity_ = 4096
Status reserve_canonical(const DetectorConfig& config, int marker_size, Workspace& workspace,
                         CanonicalBuffers* out);

/// 候補ごとに射影変換を行い、canonical 画像を作る。
///
/// OpenCV の `getPerspectiveTransform` と `warpPerspective` を
/// `INTER_NEAREST` で呼んだ場合と同じ結果を目指す。両者の合成は
/// canonical から入力画像への逆写像であり、次を再現する。
///
/// 1. 8 元 1 次方程式を倍精度で組む。`a[i][6]` と `a[i][7]` は単精度の積を
///    倍精度へ広げる。倍精度で掛けると行列が相対 1.9e-3 までずれる。
/// 2. 部分ピボット付き Gauss 消去で解く。同値なら添字の小さい行を残す。
/// 3. 3x3 の余因子式で逆行列を作る。
/// 4. 逆数を先に取ってから掛け、最近接偶数丸めで画素番号を決める。
///
/// 積和の融合は行わない。融合すると丸め境界で参照画素が 1 つずれる。
/// この翻訳単位は `-fmad=false` で compile する。
///
/// 候補ごとに使う pyramid の level は、四隅を結ぶ折れ線の chain code 長から
/// 選ぶ。CPU 経路は輪郭の画素数を使うが、案 A に輪郭は無い。凸四角形では
/// ほぼ一致するが、境界付近では選ばれる level が 1 段ずれうる。
///
/// @param preprocess 前処理の buffer 一式。pyramid を参照する。
/// @param plan 縮小率の計画。segmentation の幅を level の選択に使う。
/// @param candidates 詰めた候補。四隅と周長を参照する。
/// @param config 検出設定。
/// @param canonical reserve_canonical が返した出力領域。nullptr は不可。
/// @param stream 発行先の stream。既定 stream を使う場合は nullptr。
/// @return kOk、または kInvalidArgument、kCudaError。
///
/// 所有権: 引数が指す領域の所有権は workspace に残る。
/// 同期動作: stream へ kernel を発行するだけで host 同期を行わない。
///           候補数は device 上で参照するため、host へ戻さない。
///
/// 入力例: マーカー 4 枚分の候補と 5 段の pyramid
/// 出力例: images_ の先頭 4 枚分が 32x32 の canonical 画像で埋まる
Status build_canonical_async(const PreprocessBuffers& preprocess, const ScalePlan& plan,
                             const DeviceCandidates& candidates, const DetectorConfig& config,
                             CanonicalBuffers* canonical, cudaStream_t stream);

}  // namespace aruco3cuda::detail

#endif  // ARUCO3CUDA_CORE_CELL_SAMPLE_HPP
