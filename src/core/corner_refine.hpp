// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CORE_CORNER_REFINE_HPP
#define ARUCO3CUDA_CORE_CORNER_REFINE_HPP

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "detection_emit.hpp"
#include "preprocess.hpp"

namespace aruco3cuda::detail {

/// 補正で使う窓の 1 辺の上限。
///
/// OpenCV は ArUco3 経路で窓の半径を 3 か 5 にする。cornerSubPix が実際に
/// 切り出すのは半径から求めた (2r+1)+2 辺であり、r = 5 のとき 13 になる。
inline constexpr int kMaxRefinePatchSide = 13;

/// 補正の経過を数える counter。
///
/// 反復解法は入力の 1 ULP が反復回数や打ち切りの採否を変える。CPU 基準と
/// 差が出たとき、どの分岐で分かれたのかを後から言えるようにする。
///
/// 所有権: 全ての pointer が指す領域の所有権は workspace にある。
/// 同期動作: 単なる参照の集合であり同期点を持たない。
///
/// 入力例: 検出上限 1024 の設定
/// 出力例: 5 要素の counter が 0 で初期化される
struct RefineDiagnostics {
    /// [0] 補正した隅の数
    /// [1] 窓が画像の外へ出て打ち切った数
    /// [2] 初期位置から離れすぎて初期位置へ戻した数
    /// [3] 行列式が 0 に近く打ち切った数
    /// [4] 反復回数の総和
    std::int32_t* counters_ = nullptr;
};

/// 補正が使う作業領域。
///
/// 所有権: 全ての pointer が指す領域の所有権は workspace にある。
/// 同期動作: 単なる参照の集合であり同期点を持たない。
///
/// 入力例: 既定の設定
/// 出力例: diagnostics_.counters_ が 5 要素
struct CornerRefineBuffers {
    RefineDiagnostics diagnostics_;
};

/// counter の要素数。
inline constexpr int kRefineCounterCount = 5;

/// 補正に必要な workspace の大きさを返す。
///
/// @param config 設定。現時点では大きさに影響しない。
/// @return 必要な byte 数。
///
/// 所有権: 引数の領域を保持しない。
/// 同期動作: host 専用であり同期点を持たない。CUDA API を呼ばない。
///
/// 入力例: 既定の設定
/// 出力例: 256
std::size_t corner_refine_workspace_bytes(const DetectorConfig& config);

/// 補正の作業領域を確保する。
///
/// @param config 設定。
/// @param workspace device 空間の workspace。
/// @param out 成功時に buffer 群を格納する。領域の所有権は呼出側にある。
/// @return kOk。out が nullptr なら kInvalidArgument、容量不足なら kInvalidConfig。
///
/// 所有権: 引数の領域を保持しない。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: 既定の設定、空きのある workspace
/// 出力例: kOk。out->diagnostics_.counters_ が有効
Status reserve_corner_refine(const DetectorConfig& config, Workspace& workspace,
                             CornerRefineBuffers* out);

/// 四隅を pyramid の段を登りながら補正し、原寸の座標へ戻す。
///
/// OpenCV の `findCornerInPyrImage` と `cv::cornerSubPix` を再現する。
///
/// 手順は次のとおりである。
///
/// 1. segmentation 座標の四隅へ `scale_init` を掛け、開始段の座標にする。
///    `scale_init` は `開始段の幅 / segmentation の幅` である。
/// 2. 段を 1 つ下げる (解像度は 2 倍) たびに四隅を 2 倍し、その段で
///    `cornerSubPix` を掛ける。段 0 まで繰り返す。
/// 3. 段 0 は原寸であるため、終わった時点で四隅は原寸の座標になる。
///    `fxfy` の逆数を別に掛けてはならない。
///
/// 窓の半径は設定ではなく段の大きさで決まる。OpenCV は
/// `max(段の幅, 段の高さ) > 1080 ? 5 : 3` とする。ArUco3 経路では
/// `corner_refinement_win_size_px_` と `relative_corner_refinement_win_size_`
/// は使われない。
///
/// 演算は CPU 基準と同じ順序で行い、積和を融合しない。反復解法であるため、
/// 融合による 1 ULP の差が反復回数や打ち切りの採否を変えうる。
///
/// @param pyramid 段ごとの画像。段 0 が原寸である。
/// @param plan 縮小の計画。開始段と segmentation の幅を使う。
/// @param config 反復回数と収束の閾値を含む設定。
/// @param block_count 起こす block 数。refine_block_count() で求める。
/// @param buffers 作業領域。counter を書く。
/// @param detections 四隅を **その場で書き換える**。ids_ と count_ を読む。
/// @param stream kernel を発行する stream。
/// @return kOk。引数が不正なら kInvalidArgument、設定が不整合なら
///         kInvalidConfig、kernel 起動に失敗したら kCudaError。
///
/// 所有権: 引数の領域を保持しない。
/// 同期動作: kernel を stream 上で非同期に発行する。呼出側が同期するまで
///           結果は確定しない。
///
/// 入力例: 1280x720 の pyramid、segmentation 427x240、検出 4 件
/// 出力例: kOk。corner_x_ と corner_y_ が原寸 1280x720 の座標になる
Status refine_corners_async(const PyramidRef& pyramid, const ScalePlan& plan,
                            const DetectorConfig& config, int block_count,
                            CornerRefineBuffers* buffers, DeviceDetections* detections,
                            cudaStream_t stream);

/// device の SM 数から、補正で起こす block 数を決める。
///
/// block 数は「device が同時に走らせられる数」と「仕事の量」の小さい方で
/// あるべきである。前者を SM 数の 2 倍で見積もり、後者 (評価計画の上限で
/// あるマーカー 16 枚 = 64 隅) を上限にする。1 block が共有 memory を
/// 約 5.5 KB 使うため、必要以上に起こすと検出が 0 件の frame でも損をする。
///
/// 固定値にすると、SM 数の桁が違う機で事故になる。実際に隅の上限 (4096) を
/// そのまま block 数にしていたとき、Jetson AGX Orin (16 SM) で検出 0 件でも
/// 7 倍遅くなった。
///
/// @param multi_processor_count device の SM 数。1 未満なら下限を返す。
/// @return 起こす block 数。32 以上 64 以下。
///
/// 所有権: 資源を保持しない。
/// 同期動作: host 専用であり同期点を持たない。CUDA API を呼ばない。
///
/// 入力例: 16 (Jetson AGX Orin)
/// 出力例: 32
int refine_block_count(int multi_processor_count);

}  // namespace aruco3cuda::detail

#endif  // ARUCO3CUDA_CORE_CORNER_REFINE_HPP
