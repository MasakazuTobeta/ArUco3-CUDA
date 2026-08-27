// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_HYBRID_HYBRID_DETECTOR_HPP
#define ARUCO3CUDA_HYBRID_HYBRID_DETECTOR_HPP

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"
#include "aruco3cuda/workspace.hpp"

namespace aruco3cuda::hybrid {

/// 検出した 1 個のマーカー。
///
/// 所有権: 値のみを持ち、外部の資源を参照しない。
/// 同期動作: 単なる値の集合であり同期点を持たない。
///
/// 入力例: 合成画像に描いた ID 42 のマーカー
/// 出力例: id_ = 42、corners_ が原寸座標での四隅
struct HybridDetection {
    int id_ = -1;
    /// 四隅を x0, y0, ... y3 の順で保持する。入力画像の座標系。
    /// 並びは OpenCV の detectMarkers と同じく、Dictionary 照合で得た
    /// rotation を打ち消した後の順序である。
    std::array<double, 8> corners_{};
    /// Dictionary 照合で得た回転。0 から 3。
    int rotation_ = 0;
};

/// 1 フレームの検出結果。
///
/// 所有権: 値のみを持ち、外部の資源を参照しない。
/// 同期動作: 単なる値の集合であり同期点を持たない。
///
/// 入力例: マーカー 4 個の合成画像
/// 出力例: detections_ が 4 要素、candidate_count_ が二値化から得た候補数
struct HybridResult {
    std::vector<HybridDetection> detections_;
    /// 二値化画像から得た四角形候補の総数。window ごとの合計である。
    std::size_t candidate_count_ = 0;
    /// GPU 側の処理時間。単位は ms。同期を含む。
    double gpu_ms_ = 0.0;
    /// CPU 側の処理時間。単位は ms。
    double cpu_ms_ = 0.0;
};

/// GPU で前処理と二値化を行い、候補抽出と decode を CPU で行う検出器。
///
/// 位置付け:
///   [検出パイプライン設計](../docs/design/detector-pipeline.md) の案 C にあたる。
///   GPU 側の段階を 1 つずつ置き換えていく際の基準となり、案 A が期待どおりに
///   動かない場合の fallback でもある。Phase 2 以降も維持する。
///
/// 所有権:
///   workspace と OpenCV 側の一時 buffer をこの class が所有する。
///   入力画像の memory は呼出側が所有する。
///   **この所有権は全ての public member 関数に適用される。**
///
/// 同期動作:
///   detect() は内部で GPU の完了を待つ。二値化画像を host へ戻す必要が
///   あるためであり、この同期は案 C の構造上避けられない。
///   **この同期動作は全ての public member 関数に適用される。**
///   1 つの instance を複数 thread から同時に使用してはならない。
///
/// 入力例:
///   HybridDetector detector;
///   detector.initialize(config, "DICT_ARUCO_MIP_36h12", 1920, 1080, &message);
///   detector.detect(view, &result, &message);
/// 出力例:
///   result.detections_ に検出したマーカーの ID と四隅が入る
class HybridDetector {
public:
    HybridDetector();
    ~HybridDetector();

    HybridDetector(const HybridDetector&) = delete;
    HybridDetector& operator=(const HybridDetector&) = delete;
    HybridDetector(HybridDetector&&) noexcept;
    HybridDetector& operator=(HybridDetector&&) noexcept;

    /// 検出器を初期化する。
    ///
    /// workspace の容量を確保し、Dictionary を読み込む。フレームごとの確保を
    /// 避けるため、想定する最大解像度をここで与える。
    ///
    /// @param config 検出設定。validate() を通す。
    /// @param dictionary_name 定義済み Dictionary 名。
    /// @param max_width_px 想定する最大の幅。1 以上。
    /// @param max_height_px 想定する最大の高さ。1 以上。
    /// @param out_message 失敗時に理由を格納する。nullptr を渡してよい。
    /// @return kOk、または kInvalidConfig、kUnsupportedDictionary、kCudaError。
    ///
    /// 入力例: 既定設定、"DICT_ARUCO_MIP_36h12"、1920、1080
    /// 出力例: Status::kOk
    Status initialize(const DetectorConfig& config, const std::string& dictionary_name,
                      int max_width_px, int max_height_px, std::string* out_message = nullptr);

    /// 1 フレームを検出する。
    ///
    /// @param image 入力画像。device 常駐であること。
    /// @param out 成功時に結果を格納する。nullptr は不可。
    /// @param out_message 失敗時に理由を格納する。nullptr を渡してよい。
    /// @return kOk、または kNotInitialized、kInvalidImage、kCudaError。
    ///
    /// 入力例: device 上の 1280x720 grayscale
    /// 出力例: Status::kOk。out->detections_ に検出結果が入る
    Status detect(const ImageViewU8& image, HybridResult* out,
                  std::string* out_message = nullptr);

    /// workspace の使用状況を返す。
    ///
    /// @return 統計への参照。次の操作まで有効。
    ///
    /// 入力例: 100 フレーム処理した後
    /// 出力例: allocation_count_ が 1 のまま
    const WorkspaceStatistics& workspace_statistics() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace aruco3cuda::hybrid

#endif  // ARUCO3CUDA_HYBRID_HYBRID_DETECTOR_HPP
