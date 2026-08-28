// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CORE_CELL_DECODE_HPP
#define ARUCO3CUDA_CORE_CELL_DECODE_HPP

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/workspace.hpp"
#include "candidate_filter.hpp"
#include "cell_sample.hpp"

namespace aruco3cuda::detail {

/// 候補ごとのセル比と border 検証の結果。
///
/// 所有権: 全ての pointer が指す領域の所有権は workspace にある。
/// 同期動作: 単なる参照の集合であり同期点を持たない。内容は発行済みの
///           kernel が完了するまで確定しない。
///
/// 入力例: 既定設定、marker_size = 6、候補上限 4096
/// 出力例: cells_per_side_ = 8、ratios_ が 4096 * 8 * 8 要素
struct CellRatioBuffers {
    /// セルごとの白画素比。添字は (candidate * cells * cells) + (row * cells) + col。
    ///
    /// bit へ潰さず比のまま持つ。既定の分母は 16 であり、比 0.5 は bit 0 とも
    /// bit 1 とも一致しない。bit 行列へ潰すと CPU 基準と結果が変わる。
    float* ratios_ = nullptr;
    /// 外周セルの誤り数。
    std::int32_t* border_errors_ = nullptr;
    /// border 検証を通ったか。1 で通過。
    std::uint8_t* accepted_ = nullptr;
    /// 1 辺のセル数。marker_size + 2 * marker_border_bits_ に等しい。
    int cells_per_side_ = 0;
    int capacity_ = 0;
};

/// 1 辺のセル数を返す。
///
/// @param config 検出設定。
/// @param marker_size Dictionary のセル数。1 以上。
/// @return 1 辺のセル数。引数が不正なら 0。
///
/// 所有権: 資源を保持しない。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: 既定設定と marker_size = 6
/// 出力例: 8
int cells_per_side(const DetectorConfig& config, int marker_size);

/// セル比に必要な workspace の容量を返す。
///
/// @param config 検出設定。候補上限を使う。
/// @param marker_size Dictionary のセル数。
/// @return 必要な byte 数。桁溢れや引数が不正なら 0。
///
/// 所有権: 資源を保持しない。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: 既定設定と marker_size = 6
/// 出力例: 比と誤り数と合否を収める byte 数
std::size_t cell_ratio_workspace_bytes(const DetectorConfig& config, int marker_size);

/// workspace からセル比の領域を切り出す。
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
/// 出力例: cells_per_side_ = 8、capacity_ = 4096
Status reserve_cell_ratios(const DetectorConfig& config, int marker_size, Workspace& workspace,
                           CellRatioBuffers* out);

/// canonical 画像からセル比を求め、外周セルの誤りを数える。
///
/// CPU 基準の `_extractCellPixelRatio` と `_getBorderErrors` に対応する。
/// 手順は次のとおりで、いずれも OpenCV の実装に合わせている。
///
/// 1. canonical の内側 (各辺を cell の半分だけ寄せた範囲) で平均と母標準偏差を
///    求める。和と二乗和は整数のまま積み、最後だけ倍精度で割る。平均は
///    `S * (1/N)` であって `S / N` ではない。
/// 2. 標準偏差が `min_otsu_std_dev_` を下回る場合、全セルの比を平均が 127 を
///    超えるかどうかで 1 か 0 に埋める。この場合も border 検証は行う。
/// 3. そうでなければ canonical の**全体**へ Otsu を掛ける。内側ではない。
///    二値化は `画素 > 閾値` で、等しい場合は 0 側とする。
/// 4. セルごとに白画素の比を求める。cell の余白は `(int)(rate * cell)` であり、
///    既定設定では 0 になる。分母は余白を除いた 1 辺の 2 乗である。
/// 5. 外周セルのうち比が `valid_bit_threshold_` を超えるものを数える。
///    `marker_size` の 2 乗に率を掛けた値を超えたら不合格とする。
///
/// 積和の融合は行わない。この翻訳単位は `-fmad=false` で compile する。
/// 融合すると分散の計算が 1 ULP ずれ、標準偏差の閾値付近で判定が変わる。
///
/// @param canonical build_canonical_async が埋めた canonical 画像。
/// @param candidates 詰めた候補。候補数を device 上で参照する。
/// @param config 検出設定。
/// @param marker_size Dictionary のセル数。1 以上。
/// @param ratios reserve_cell_ratios が返した出力領域。nullptr は不可。
/// @param stream 発行先の stream。既定 stream を使う場合は nullptr。
/// @return kOk、または kInvalidArgument、kCudaError。
///
/// 所有権: 引数が指す領域の所有権は workspace に残る。
/// 同期動作: stream へ kernel を発行するだけで host 同期を行わない。
///
/// 入力例: マーカー 4 枚分の canonical 画像
/// 出力例: ratios_ に 8x8 の比、accepted_ に 1 が 4 つ
Status build_cell_ratios_async(const CanonicalBuffers& canonical,
                               const DeviceCandidates& candidates, const DetectorConfig& config,
                               int marker_size, CellRatioBuffers* ratios, cudaStream_t stream);

}  // namespace aruco3cuda::detail

#endif  // ARUCO3CUDA_CORE_CELL_DECODE_HPP
