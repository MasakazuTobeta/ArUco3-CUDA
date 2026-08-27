// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CONFIG_HPP
#define ARUCO3CUDA_CONFIG_HPP

#include <string>

#include "aruco3cuda/status.hpp"

namespace aruco3cuda {

/// 四隅の subpixel 補正の方式。
enum class CornerRefineMethod : int {
    kNone = 0,  ///< 補正しない
    kSubpix,    ///< 勾配法による subpixel 補正
};

/// CornerRefineMethod を識別子文字列へ変換する。
///
/// @param method 変換対象。列挙に無い値でも nullptr を返さない。
/// @return 静的記憶域を持つ文字列。
///
/// 所有権: 戻り値は静的記憶域を指す。呼出側は解放も変更もしない。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: CornerRefineMethod::kSubpix
/// 出力例: "kSubpix"
const char* to_string(CornerRefineMethod method);

/// 検出設定。
///
/// 既定値は OpenCV 4.x の `cv::aruco::DetectorParameters` に合わせる。ただし
/// ArUco3 検出戦略に関わる 2 項目は、本 project の評価目的に合わせて既定を変える。
/// OpenCV と同じ既定値が必要な場合は opencv_defaults() を使う。
///
/// 所有権:
///   値のみを持ち、外部の資源を参照しない。複製して保持してよい。
struct DetectorConfig {
    // --- 適応的二値化 ---
    /// 適応的二値化の window 辺長。単位は pixel。OpenCV は偶数を奇数へ切り上げる。
    int adaptive_thresh_win_size_min_px_ = 3;
    int adaptive_thresh_win_size_max_px_ = 23;
    int adaptive_thresh_win_size_step_px_ = 10;
    double adaptive_thresh_constant_ = 7.0;

    // --- 候補フィルタ ---
    double min_marker_perimeter_rate_ = 0.03;
    double max_marker_perimeter_rate_ = 4.0;
    double polygonal_approx_accuracy_rate_ = 0.03;
    double min_corner_distance_rate_ = 0.05;
    int min_distance_to_border_px_ = 3;
    double min_marker_distance_rate_ = 0.125;

    // --- ビット読取りと照合 ---
    int marker_border_bits_ = 1;
    int perspective_remove_pixel_per_cell_ = 4;
    double perspective_remove_ignored_margin_per_cell_ = 0.13;
    double max_erroneous_bits_in_border_rate_ = 0.35;
    double min_otsu_std_dev_ = 5.0;
    double error_correction_rate_ = 0.6;

    // --- 四隅補正 ---
    CornerRefineMethod corner_refine_method_ = CornerRefineMethod::kSubpix;
    int corner_refinement_win_size_px_ = 5;
    int corner_refinement_max_iterations_ = 30;
    double corner_refinement_min_accuracy_px_ = 0.1;

    // --- ArUco3 検出戦略 ---
    bool use_aruco3_detection_ = true;
    int min_side_length_canonical_img_px_ = 32;
    float min_marker_length_ratio_original_img_ = 0.05F;

    // --- CUDA 固有 ---
    /// 候補 buffer の上限。超過時は kCandidateOverflow を返し、結果を打ち切る。
    int max_candidates_ = 4096;
    /// 検出 buffer の上限。超過時は kMarkerOverflow を返し、結果を打ち切る。
    int max_markers_ = 1024;
    /// workspace を事前確保する基準の解像度。これを超える入力は再確保になる。
    int max_width_px_ = 3840;
    int max_height_px_ = 2160;
    /// 2 次元 kernel の block 1 辺の thread 数。
    ///
    /// 機種固有の最適化を source の固定値にせず設定から上書きできるようにする。
    /// 既定は多くの機種で無難な 16 とし、機種別の値は測定で決める。
    int cuda_block_dim_ = 16;

    /// 設定が有効な範囲にあり、相互に矛盾しないことを確認する。
    ///
    /// 検出器の生成前に呼び出せる。範囲外の値をそのまま CUDA kernel の
    /// 起動設定へ渡すと、無効な block 数や負の反復回数として現れ、
    /// 原因が設定であることが分かりにくくなる。
    ///
    /// @param out_message 失敗時に「項目名=値」を含む理由を格納する。
    ///                    nullptr を渡してよい。成功時は変更しない。
    /// @return 全て有効なら kOk、そうでなければ kInvalidConfig。
    ///
    /// 所有権: 引数の領域を保持しない。
    /// 同期動作: host 専用であり同期点を持たない。CUDA API を呼ばない。
    ///
    /// 入力例: 既定の DetectorConfig
    /// 出力例: Status::kOk
    /// 入力例: adaptive_thresh_win_size_min_px_ = 2
    /// 出力例: Status::kInvalidConfig。out_message に
    ///         "adaptive_thresh_win_size_min_px=2" を含む
    Status validate(std::string* out_message = nullptr) const;

    /// ArUco3 に関わる既定値を OpenCV と同じにした設定を返す。
    ///
    /// OpenCV は `useAruco3Detection = false`、
    /// `minMarkerLengthRatioOriginalImg = 0.0` を既定とする。互換性の確認では
    /// この既定に揃える必要がある。
    ///
    /// @return OpenCV の既定に合わせた設定。
    ///
    /// 所有権: 値を返す。呼出側が所有する。
    /// 同期動作: host 専用であり同期点を持たない。
    ///
    /// 入力例: 引数なし
    /// 出力例: use_aruco3_detection_ = false、
    ///         min_marker_length_ratio_original_img_ = 0.0F の設定
    static DetectorConfig opencv_defaults();
};

}  // namespace aruco3cuda

#endif  // ARUCO3CUDA_CONFIG_HPP
