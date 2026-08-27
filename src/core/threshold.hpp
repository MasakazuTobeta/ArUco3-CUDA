// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CORE_THRESHOLD_HPP
#define ARUCO3CUDA_CORE_THRESHOLD_HPP

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"
#include "aruco3cuda/workspace.hpp"
#include "preprocess.hpp"

namespace aruco3cuda::detail {

/// 適応的二値化の出力と作業領域。
///
/// 所有権: 全ての pointer が指す領域の所有権は workspace にある。この構造体は
///         参照のみを持ち、複製も解放も行わない。workspace を reset() または
///         破棄すると全ての pointer が無効になる。
/// 同期動作: 単なる参照の集合であり同期点を持たない。内容は発行済みの
///           kernel が完了するまで確定しない。
///
/// 入力例: 既定設定と 427x240 の segmentation 画像
/// 出力例: window_count_ = 3、window_sizes_px_ = {3, 13, 23}
struct ThresholdBuffers {
    /// window ごとの二値化結果。index は window_sizes_px_ に対応する。
    ImagePlaneU8 binary_[kMaxAdaptiveThresholdWindows];
    int window_count_ = 0;
    int window_sizes_px_[kMaxAdaptiveThresholdWindows] = {};
    /// 行方向の合計を保持する作業領域。window ごとに使い回す。
    std::int32_t* row_sums_ = nullptr;
    std::size_t row_sums_pitch_bytes_ = 0;
    int width_px_ = 0;
    int height_px_ = 0;
};

/// 設定から走査する window の一覧を求める。
///
/// OpenCV は偶数の window を奇数へ切り上げる。走査数は切り上げ前の値から
/// 決まるため、同じ window size が重複することがある。OpenCV はその場合も
/// 重複したまま処理するため、こちらも取り除かない。
///
/// @param config 検出設定。
/// @param out_sizes 成功時に window size を格納する。nullptr は不可。
/// @param capacity out_sizes の要素数。
/// @param out_count 成功時に window 数を格納する。nullptr は不可。
/// @return kOk、または kInvalidArgument、kInvalidConfig。
///
/// 所有権: 引数の領域を保持しない。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: 既定設定 (min 3、max 23、step 10)
/// 出力例: out_count = 3、out_sizes = {3, 13, 23}
Status threshold_window_sizes(const DetectorConfig& config, int* out_sizes, int capacity,
                              int* out_count);

/// 必要な workspace の容量を返す。
///
/// @param config 検出設定。
/// @param width_px segmentation 画像の幅。
/// @param height_px segmentation 画像の高さ。
/// @return 必要な byte 数。桁溢れや設定不正なら 0。
///
/// 所有権: 資源を保持しない。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: 既定設定と 427x240
/// 出力例: 二値化 3 枚と行合計 1 枚を収める byte 数
std::size_t threshold_workspace_bytes(const DetectorConfig& config, int width_px, int height_px);

/// workspace から二値化用の領域を切り出す。
///
/// @param config 検出設定。
/// @param width_px segmentation 画像の幅。1 以上。
/// @param height_px segmentation 画像の高さ。1 以上。
/// @param workspace 切り出し元。呼出側が所有する。
/// @param out 成功時に buffer 一式を格納する。nullptr は不可。
/// @return kOk。容量不足なら kInvalidConfig、引数が不正なら kInvalidArgument。
///
/// 所有権: 切り出した領域の所有権は workspace に残る。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: 既定設定と 427x240 と十分な容量の workspace
/// 出力例: binary_ に 3 枚分の pointer が入る
Status reserve_threshold(const DetectorConfig& config, int width_px, int height_px,
                         Workspace& workspace, ThresholdBuffers* out);

/// 適応的二値化を実行する。
///
/// OpenCV の `adaptiveThreshold` を `ADAPTIVE_THRESH_MEAN_C` と
/// `THRESH_BINARY_INV` で呼んだ場合と同じ結果を目指す。平均は
/// `boxFilter` を正規化ありで適用したものであり、境界は BORDER_REPLICATE、
/// 判定は (画素 - 平均) <= -floor(定数) で 255、そうでなければ 0 とする。
///
/// @param segmentation 入力画像。reserve_threshold と同じ寸法である必要がある。
/// @param buffers reserve_threshold が返した buffer 一式。nullptr は不可。
/// @param config 検出設定。
/// @param stream 発行先の stream。既定 stream を使う場合は nullptr。
/// @return kOk、または kInvalidArgument、kCudaError。
///
/// 所有権: buffers が指す領域の所有権は workspace に残る。
/// 同期動作: stream へ kernel を発行するだけで host 同期を行わない。
///           window ごとに行合計の作業領域を使い回すため、同じ stream 内で
///           順に実行される。複数 stream へ分けるには作業領域を分ける必要がある。
///
/// 入力例: 427x240 の segmentation 画像と既定設定
/// 出力例: binary_[0..2] が window 3、13、23 の二値化結果で埋まる
Status build_threshold_async(const ImageViewU8& segmentation, ThresholdBuffers* buffers,
                             const DetectorConfig& config, cudaStream_t stream);

}  // namespace aruco3cuda::detail

#endif  // ARUCO3CUDA_CORE_THRESHOLD_HPP
