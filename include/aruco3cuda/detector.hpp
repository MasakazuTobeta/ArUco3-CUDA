// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_DETECTOR_HPP
#define ARUCO3CUDA_DETECTOR_HPP

#include <cuda_runtime_api.h>

#include <memory>
#include <string>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/detections.hpp"
#include "aruco3cuda/dictionary.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"
#include "aruco3cuda/workspace.hpp"

namespace aruco3cuda {

/// ArUco3 検出戦略の CUDA 実装。
///
/// 目的:
///   前処理から四隅の補正までを 1 つの stream へ発行し、結果を device 上に
///   置いたまま返す。後段が同じ device 上にあるなら host へ戻す必要が無い。
///
/// 支持する設定の組み合わせは 2 つだけである。
///
/// | use_aruco3_detection_ | corner_refine_method_ | 四隅の座標 |
/// | --- | --- | --- |
/// | true | kSubpix | 原寸 (段を登って復元する) |
/// | false | kNone | 原寸 (縮小しないため) |
///
/// 他の 2 組は initialize() が拒否する。ArUco3 有効で補正を切ると四隅が
/// 縮小後の座標のまま残り、ArUco3 無効で補正を入れても段が 1 つしかないため
/// 補正が 1 度も走らないためである。
///
/// 所有権:
///   workspace と DeviceDetections が指す領域を本 class が所有する。入力画像と
///   stream の所有権は呼出側にある。DictionaryTable は initialize() の中で
///   device へ複製するため、呼出後に破棄してよい。
///   **この所有権は全ての public member 関数に適用される。**
/// 同期動作:
///   initialize() と download() だけが host 同期を伴う。detect_async() は
///   kernel の発行のみで同期しない。ただし入力の寸法か pitch が前回と変わった
///   frame では、buffer の配置を組み直す前に 1 度だけ stream を同期する。
///   1 つの instance を複数の thread から同時に使ってはならない。
///
/// 入力例: DICT_ARUCO_MIP_36h12 と既定設定で initialize
/// 出力例: kOk。以降 detect_async() が呼べる
class Detector {
public:
    Detector();
    ~Detector();
    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;
    Detector(Detector&&) noexcept;
    Detector& operator=(Detector&&) noexcept;

    /// workspace を確保し Dictionary を device へ転送する。
    ///
    /// workspace は設定の上限 (max_width_px_、max_height_px_、max_candidates_、
    /// max_markers_) から最悪値で確保する。以後 detect_async() が確保を
    /// 追加することはない。
    ///
    /// @param dictionary 照合に使う Dictionary。内容を device へ複製する。
    /// @param config 検出設定。validate() を通す。
    /// @param out_message 失敗時に理由を格納する。nullptr を渡してよい。
    /// @return kOk。Dictionary が不正なら kUnsupportedDictionary、設定が
    ///         不正なら kInvalidConfig、確保に失敗したら kInvalidConfig、
    ///         CUDA API が失敗したら kCudaError。
    ///
    /// 同期動作: 最後に device 全体を同期する。Dictionary の転送を、以後
    ///           どの stream から見ても完了した状態にするためである。
    ///
    /// 入力例: DICT_ARUCO_MIP_36h12 の table と既定設定
    /// 出力例: kOk。initialized() が true になる
    Status initialize(const DictionaryTable& dictionary, const DetectorConfig& config,
                      std::string* out_message = nullptr);

    /// 検出を stream へ発行する。
    ///
    /// **戻り値の kOk は上限で打ち切っていないことを意味しない。** 打ち切りの
    /// 有無は device 上の accepted_total_ と count_ の比較で分かる。host から
    /// 知るには download() を呼ぶ。
    ///
    /// @param image 入力画像。device か managed の空間である必要がある。
    ///              領域の所有権は呼出側にあり、kernel が完了するまで有効で
    ///              なければならない。
    /// @param stream kernel を発行する stream。nullptr は既定 stream を指す。
    /// @param out_message 失敗時に理由を格納する。nullptr を渡してよい。
    /// @return kOk。initialize() 前なら kNotInitialized、画像が不正なら
    ///         kInvalidImage、上限を超える寸法なら kInvalidArgument、
    ///         確保に失敗したら kInvalidConfig、kernel 起動に失敗したら
    ///         kCudaError。
    ///
    /// 同期動作: kernel を stream 上で非同期に発行する。入力の寸法か pitch が
    ///           前回と変わった frame でだけ、配置を組み直す前に 1 度
    ///           cudaStreamSynchronize を行う。
    ///
    /// 入力例: 1280x720 の device 画像
    /// 出力例: kOk。device_detections() が指す領域へ結果が書かれる
    Status detect_async(const ImageViewU8& image, cudaStream_t stream,
                        std::string* out_message = nullptr);

    /// 直近の detect_async() の device 常駐結果を写す。
    ///
    /// 同期は発生しない。指す領域の内容は、発行済みの kernel が完了するまで
    /// 確定しない。detect_async() のたびに呼び直すこと。入力の寸法が変わると
    /// 配置が変わり、pointer も変わる。
    ///
    /// @param out 成功時に参照を格納する。領域の所有権は呼出側にある。
    /// @return kOk。out が nullptr なら kInvalidArgument、detect_async() を
    ///         1 度も呼んでいなければ kNotInitialized。
    ///
    /// 同期動作: **同期しない。** CUDA API を呼ばない。
    ///
    /// 入力例: detect_async() の直後
    /// 出力例: kOk。out->count_ が device 上の検出数を指す
    Status device_detections(DeviceDetections* out) const;

    /// stream の完了を待って host 側の結果を得る。
    ///
    /// @param out 成功時に結果を格納する。領域の所有権は呼出側にある。
    /// @param stream 同期する stream。detect_async() に渡したものと同じにする。
    /// @param out_message 失敗時に理由を格納する。nullptr を渡してよい。
    /// @return kOk。上限で打ち切っていれば kMarkerOverflow (out は埋まる)。
    ///         out が nullptr なら kInvalidArgument、detect_async() を 1 度も
    ///         呼んでいなければ kNotInitialized、CUDA API が失敗したら
    ///         kCudaError。
    ///
    /// 同期動作: **stream を同期する。** 本 class で同期するのはここだけである。
    ///
    /// 入力例: 検出 2 件を書いた stream
    /// 出力例: kOk。out->ids_ が 2 要素、out->corners_ が 16 要素
    Status download(HostDetections* out, cudaStream_t stream, std::string* out_message = nullptr);

    /// device workspace の使用状況を返す。
    ///
    /// allocation_count_ が initialize() 後も 1 のままであることを確かめると、
    /// frame ごとに確保していないことが言える。
    ///
    /// @return 統計。initialize() 前は全て 0。
    ///
    /// 所有権: 戻り値は本 instance が所有する領域を指す。
    /// 同期動作: host 専用であり同期点を持たない。
    ///
    /// 入力例: initialize() 済みの instance
    /// 出力例: allocation_count_ = 1
    const WorkspaceStatistics& workspace_statistics() const;

    /// initialize() が成功しているか。
    ///
    /// @return 成功していれば true。
    ///
    /// 所有権: 資源を保持しない。
    /// 同期動作: host 専用であり同期点を持たない。
    ///
    /// 入力例: initialize() 済みの instance
    /// 出力例: true
    bool initialized() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace aruco3cuda

#endif  // ARUCO3CUDA_DETECTOR_HPP
