// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_REFERENCE_RUNNER_HPP
#define ARUCO3CUDA_REFERENCE_RUNNER_HPP

#include <array>
#include <memory>
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

    /// 四隅の subpixel 補正を行うか。OpenCV は ArUco3 有効時、この指定に
    /// かかわらず補正を行う。
    bool use_corner_subpix_refinement_ = false;
    int corner_refinement_win_size_px_ = 5;
    double relative_corner_refinement_win_size_ = 0.3;
    int corner_refinement_max_iterations_ = 30;
    double corner_refinement_min_accuracy_px_ = 0.1;

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
/// @param config 検証対象。参照するだけで保持しない。
/// @param out_error 失敗時に「項目名=値」を含む理由を格納する。nullptr は不可。
///                  領域の所有権は呼出側にある。
/// @return 全ての項目が有効なら true。
///
/// 所有権: 引数の領域を保持しない。
/// 同期動作: host 専用であり同期点を持たない。OpenCV も CUDA も呼ばない。
///
/// 入力例: adaptive_thresh_win_size_min_ = 2 の config
/// 出力例: false、out_error に "adaptive_thresh_win_size_min=2" を含む文字列
bool validate_config(const ReferenceConfig& config, std::string* out_error);

/// 名前から OpenCV の定義済み Dictionary を解決できるか確認する。
///
/// @param name 探す名前。参照するだけで保持しない。
/// @return 解決できる場合は true。
///
/// 所有権: 引数の領域を保持しない。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: "DICT_ARUCO_MIP_36h12"
/// 出力例: true
bool is_known_dictionary(const std::string& name);

/// 対応している Dictionary 名の一覧を返す。
///
/// @return 名前を昇順で並べた一覧。
///
/// 所有権: 戻り値は値であり、呼出側が所有する。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: 引数なし
/// 出力例: {"DICT_4X4_100", "DICT_4X4_1000", ..., "DICT_ARUCO_MIP_36h12"}
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

/// 画像と Dictionary を 1 度だけ用意し、検出だけを繰り返す。
///
/// `detect_image` は呼ぶたびに file の読み込みと checksum の計算を行う。
/// 測定でそれを繰り返すと、検出時間より読み込み時間の方が大きくなり、
/// 何を比べているのか分からなくなる。合成 corpus の 1280x720 PNG では
/// 読み込みが測定区間の 6 割から 8 割を占める。読み込みを初期化側へ寄せ、
/// 測定区間を検出だけにするためにこの型を用意する。
///
/// 所有権: 読み込んだ画像と OpenCV の検出器を自身で所有する。
/// 同期動作: 無し。1 つの instance を複数 thread から同時に使ってはならない。
///           `detect` は内部状態を変更しないが、OpenCV の検出器の thread 安全性を
///           前提にできないため const にしていない。
///
/// 入力例: initialize("scene.png", 既定設定) のあと detect を 200 回
/// 出力例: 毎回同じ検出結果。file 読み込みは 1 度だけ
class ReferenceDetector {
public:
    ReferenceDetector();
    ~ReferenceDetector();
    ReferenceDetector(const ReferenceDetector&) = delete;
    ReferenceDetector& operator=(const ReferenceDetector&) = delete;
    ReferenceDetector(ReferenceDetector&&) noexcept;
    ReferenceDetector& operator=(ReferenceDetector&&) noexcept;

    /// 画像を読み込み、Dictionary と検出器を用意する。
    ///
    /// @param image_path 8-bit grayscale として読み込む画像 file。
    /// @param config 検出設定。
    /// @param out_error 失敗時に理由を格納する。nullptr は不可。
    /// @return 成功した場合は true。
    ///
    /// 所有権: 読み込んだ画像を自身で所有する。引数は保持しない。
    /// 同期動作: 無し。file 入出力を伴う。
    ///
    /// 入力例: 1280x720 の PNG と既定設定
    /// 出力例: true。metadata() が寸法と checksum を返せるようになる
    bool initialize(const std::string& image_path, const ReferenceConfig& config,
                    std::string* out_error);

    /// 読み込み済みの画像を検出する。
    ///
    /// @param out_result 成功時に結果を格納する。nullptr は不可。
    /// @param out_error 失敗時に理由を格納する。nullptr は不可。
    /// @return 成功した場合は true。initialize 前に呼ぶと false。
    ///
    /// 備考:
    ///   検出結果の並べ替え規則は `detect_image` と同じである。
    ///
    /// 所有権: 引数を保持しない。
    /// 同期動作: 無し。file 入出力を行わない。
    ///
    /// 入力例: initialize 済みの instance
    /// 出力例: true。out_result に detect_image と同じ結果が入る
    bool detect(ReferenceResult* out_result, std::string* out_error);

    /// 画像の情報を返す。検出結果は含まない。
    ///
    /// @return path、checksum、寸法、fxfy 実効値を持つ結果。
    ///
    /// 所有権: 戻り値の参照先は本 instance が所有する。
    /// 同期動作: 無し。
    ///
    /// 入力例: initialize 済みの instance
    /// 出力例: image_sha256_ と width_px_ が埋まった結果
    const ReferenceResult& metadata() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/// 実行環境の情報を収集する。
///
/// @param config 検出設定。OpenCV の thread 数の設定にのみ使用する。
/// @return 収集した環境情報。取得できなかった項目は空文字列のままとする。
///
/// 所有権: 戻り値は値であり、内部の資源を参照しない。引数は保持しない。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 副作用: config.num_threads_ が 1 以上なら cv::setNumThreads() を呼び、
///         以降の OpenCV 処理の thread 数を変更する。測定条件を固定するための
///         意図的な副作用である。0 を指定した場合は変更しない。
///
/// 入力例: num_threads_ = 1 の ReferenceConfig
/// 出力例: opencv_version_ = "4.14.0"、opencv_threads_ = 1
ReferenceEnvironment collect_environment(const ReferenceConfig& config);

/// 結果を JSON として書き出す。
void write_results_json(std::ostream& out, const ReferenceConfig& config,
                        const ReferenceEnvironment& environment,
                        const std::vector<ReferenceResult>& results);

}  // namespace aruco3cuda::reference

#endif  // ARUCO3CUDA_REFERENCE_RUNNER_HPP
