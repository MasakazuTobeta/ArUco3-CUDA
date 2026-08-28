// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_DETECTIONS_HPP
#define ARUCO3CUDA_DETECTIONS_HPP

#include <cstdint>
#include <vector>

namespace aruco3cuda {

/// device 上に留まる検出結果。
///
/// 目的:
///   GPU 常駐の pipeline から host 同期なしで結果を参照する。姿勢推定などの
///   後段が同じ device 上にあるなら、host へ戻す必要が無い。
///
/// 四隅は Dictionary 照合で得た回転を**打ち消した後**の並びである。
/// rotations_ には打ち消す前の値が残るため、後段が rotations_ を見て
/// もう一度回すと 90 度ずれる。
///
/// 座標は入力画像の原寸である。縮小した segmentation 画像の座標ではない。
///
/// 所有権: 全ての pointer が指す領域の所有権は Detector にある。この構造体は
///         参照のみを持ち、複製も解放も行わない。Detector を破棄するか、
///         入力の寸法が変わる detect_async() を呼ぶと無効になる。
/// 同期動作: 単なる参照の集合であり同期点を持たない。内容は発行済みの
///           kernel が完了するまで確定しない。
///
/// 入力例: max_markers_ = 1024 の設定で initialize した Detector
/// 出力例: capacity_ = 1024、corner_x_ が 4096 要素
struct DeviceDetections {
    /// 一致した Dictionary の ID。
    std::int32_t* ids_ = nullptr;
    /// 一致した回転。0 から 3。四隅は打ち消し済みである。
    std::int32_t* rotations_ = nullptr;
    /// 四隅の x 座標。添字は (corner * capacity_) + detection。
    float* corner_x_ = nullptr;
    /// 四隅の y 座標。並びは corner_x_ と同じ。
    float* corner_y_ = nullptr;
    /// 由来した候補の index。
    std::int32_t* source_ = nullptr;
    /// 打ち切り後の検出数。要素数 1。device 上にあり、host から直接読めない。
    std::int32_t* count_ = nullptr;
    /// 打ち切る前の検出数。要素数 1。count_ より大きければ捨てている。
    std::int32_t* accepted_total_ = nullptr;
    int capacity_ = 0;
};

/// host 側へ取り出した検出結果。
///
/// 目的:
///   device 常駐の結果を host で扱う形へ写す。写す時点で 1 度だけ同期する。
///
/// 所有権: 値のみを持ち、外部の資源を参照しない。複製して保持してよい。
/// 同期動作: 単なる値の集合であり同期点を持たない。
///
/// 入力例: 検出 2 件を download() した結果
/// 出力例: ids_ が 2 要素、corners_ が 16 要素
struct HostDetections {
    std::vector<std::int32_t> ids_;
    /// 四隅を x0, y0, x1, y1, x2, y2, x3, y3 の順で 1 検出あたり 8 要素。
    std::vector<float> corners_;
    std::vector<std::int32_t> rotations_;
    /// 打ち切る前の検出数。ids_ の要素数より大きければ捨てている。
    std::int32_t accepted_total_ = 0;
    /// 上限で打ち切ったか。
    bool marker_overflow_ = false;
};

}  // namespace aruco3cuda

#endif  // ARUCO3CUDA_DETECTIONS_HPP
