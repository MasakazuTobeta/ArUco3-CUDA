// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_REFERENCE_RUNNER_HPP
#define ARUCO3CUDA_REFERENCE_RUNNER_HPP

#include <array>
#include <ostream>
#include <string>
#include <vector>

namespace aruco3cuda::reference {

/// CPU 基準実装の検出設定。
///
/// 既定値は OpenCV の DetectorParameters に合わせる。ArUco3 検出戦略に関わる
/// 項目のみ、評価の既定として本 project の値を用いる。
struct ReferenceConfig {
    std::string dictionary_name_ = "DICT_ARUCO_MIP_36h12";

    /// 適応的二値化の window 辺長。単位は pixel。
    int adaptive_thresh_win_size_min_px_ = 3;
    int adaptive_thresh_win_size_max_px_ = 23;
    int adaptive_thresh_win_size_step_px_ = 10;
    double adaptive_thresh_constant_ = 7.0;
    double min_marker_perimeter_rate_ = 0.03;
    double max_marker_perimeter_rate_ = 4.0;
    double polygonal_approx_accuracy_rate_ = 0.03;
    double min_corner_distance_rate_ = 0.05;
    int min_distance_to_border_px_ = 3;
    double min_marker_distance_rate_ = 0.125;
    int marker_border_bits_ = 1;
    int perspective_remove_pixel_per_cell_ = 4;
    double perspective_remove_ignored_margin_per_cell_ = 0.13;
    double max_erroneous_bits_in_border_rate_ = 0.35;
    double min_otsu_std_dev_ = 5.0;
    double error_correction_rate_ = 0.6;

    bool use_aruco3_detection_ = true;
    int min_side_length_canonical_img_px_ = 32;
    float min_marker_length_ratio_original_img_ = 0.05F;

    /// OpenCV の thread 数。測定の再現性のため既定で 1 に固定する。
    /// 0 を指定すると OpenCV の既定に従う。
    int num_threads_ = 1;

    /// 実行時間を JSON へ含めない。golden file との byte 単位比較に使用する。
    /// 時間は実行ごとに変動するため、決定的な出力が必要な場合は true にする。
    bool omit_timing_ = false;
};

/// 1 枚の画像に対する検出結果。
struct ReferenceDetection {
    int id_ = -1;
    /// 四隅を x0, y0, x1, y1, x2, y2, x3, y3 の順で保持する。
    /// 並びは OpenCV の detectMarkers が返す順序をそのまま用いる。
    std::array<double, 8> corners_{};
};

/// 1 枚の画像に対する実行結果。
struct ReferenceResult {
    std::string image_path_;
    std::string image_sha256_;
    int width_px_ = 0;
    int height_px_ = 0;

    /// ArUco3 の実効縮小率。評価計画が測定条件として記録を要求する。
    double fxfy_effective_ = 1.0;
    int segmentation_width_px_ = 0;
    int segmentation_height_px_ = 0;

    std::vector<ReferenceDetection> detections_;
    std::size_t rejected_count_ = 0;
    double detect_ms_ = 0.0;
};

/// 実行環境と OpenCV の version 情報。
struct ReferenceEnvironment {
    std::string opencv_version_;
    int opencv_threads_ = 0;
    std::string opencv_provenance_json_;  ///< image 内 provenance の生の JSON。空なら未取得
};

/// 検出設定が有効な範囲にあるかを検証する。
///
/// 範囲外の値をそのまま OpenCV へ渡すと cv::Exception が送出され、
/// bool と out_error で失敗を通知する契約を破って呼出側へ抜ける。
/// detect_image は最初にこの検証を行う。
///
/// @param config 検証対象。
/// @param out_error 失敗時に「項目名=値」を含む理由を格納する。nullptr は不可。
/// @return 全ての項目が有効なら true。
///
/// 入力例: adaptive_thresh_win_size_min_ = 2 の config
/// 出力例: false、out_error に "adaptive_thresh_win_size_min=2" を含む文字列
bool validate_config(const ReferenceConfig& config, std::string* out_error);

/// 名前から OpenCV の定義済み Dictionary を解決できるか確認する。
///
/// @param name 例: "DICT_ARUCO_MIP_36h12"
/// @return 解決できる場合は true。
bool is_known_dictionary(const std::string& name);

/// 対応している Dictionary 名の一覧を返す。
std::vector<std::string> known_dictionary_names();

/// 画像 1 枚を検出する。
///
/// @param image_path 8-bit grayscale として読み込む画像 file。
/// @param config 検出設定。
/// @param out_result 成功時に結果を格納する。
/// @param out_error 失敗時に理由を格納する。
/// @return 成功した場合は true。
///
/// 備考:
///   検出結果は id、次に最初の corner の x、y の順で安定に並べ替える。
///   OpenCV が返す順序は候補の抽出順に依存し、比較用途では扱いにくいため。
bool detect_image(const std::string& image_path, const ReferenceConfig& config,
                  ReferenceResult* out_result, std::string* out_error);

/// 実行環境の情報を収集する。
ReferenceEnvironment collect_environment(const ReferenceConfig& config);

/// 結果を JSON として書き出す。
void write_results_json(std::ostream& out, const ReferenceConfig& config,
                        const ReferenceEnvironment& environment,
                        const std::vector<ReferenceResult>& results);

}  // namespace aruco3cuda::reference

#endif  // ARUCO3CUDA_REFERENCE_RUNNER_HPP
