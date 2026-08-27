// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CORPUSGEN_CORPUS_GENERATOR_HPP
#define ARUCO3CUDA_CORPUSGEN_CORPUS_GENERATOR_HPP

#include <array>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

namespace aruco3cuda::corpusgen {

/// 1 枚の合成画像の生成条件。
///
/// 全ての条件を明示的に持たせ、既定値へ依存しない。同じ spec と同じ seed から
/// 同じ画像が再生成できることを前提とする。
struct SceneSpec {
    std::string name_;
    int width_px_ = 1280;
    int height_px_ = 720;
    int marker_count_ = 1;
    /// マーカー 1 辺の目標長 (pixel)。射影歪みを加えると実際の辺長は前後する。
    double marker_side_px_ = 128.0;
    /// 画像平面内の回転 (度)。
    double rotation_deg_ = 0.0;
    /// 射影歪みの強さ。0 で歪みなし、1 で辺長の 25% まで四隅をずらす。
    double perspective_strength_ = 0.0;
    /// Gaussian ぼけの標準偏差。単位は pixel。0 で無効。
    double blur_sigma_px_ = 0.0;
    /// 加算 Gaussian noise の標準偏差。単位は階調 (0 から 255)。0 で無効。
    double noise_sigma_levels_ = 0.0;
    /// 照度勾配の強さ。0 で均一、1 で画像端が 50% まで暗くなる。
    double illumination_strength_ = 0.0;
    /// マーカーを覆う遮蔽の面積比。0 で遮蔽なし。
    double occlusion_ratio_ = 0.0;
    /// マーカーが画像境界からはみ出す配置を許すか。
    bool allow_border_clip_ = false;
};

/// 生成したマーカー 1 個の ground truth。
struct MarkerGroundTruth {
    int id_ = -1;
    /// 四隅を x0, y0, ... y3 の順で保持する。
    /// 並びはマーカー座標系の (0,0)、(S,0)、(S,S)、(0,S) に対応する。
    std::array<double, 8> corners_{};
    double side_px_ = 0.0;
    /// 辺長を画像の長辺で割った比。ArUco3 の minMarkerLengthRatioOriginalImg
    /// と直接比較でき、どの tau_i で検出対象になるかを判断できる。
    double side_ratio_ = 0.0;
    /// 四隅すべてが画像内に収まっているか。
    bool fully_inside_ = true;
    double occlusion_ratio_ = 0.0;
};

/// 1 枚の生成結果。
struct GeneratedScene {
    std::string name_;
    std::string path_;
    std::string sha256_;
    int width_px_ = 0;
    int height_px_ = 0;
    SceneSpec spec_;
    std::vector<MarkerGroundTruth> markers_;
};

/// corpus 全体の生成設定。
struct CorpusConfig {
    std::string dictionary_name_ = "DICT_ARUCO_MIP_36h12";
    std::string output_dir_ = "data/corpus";
    std::uint64_t seed_ = 20260827U;
    /// マーカー画像を描画する際の canonical 解像度。
    /// 実際の配置寸法より十分大きくし、縮小方向の再標本化になるようにする。
    int canonical_marker_px_ = 512;
    int marker_border_bits_ = 1;
};

/// corpus 設定と scene spec が有効な範囲にあるかを検証する。
///
/// 範囲外の値は OpenCV の assert を踏むか、巨大な memory 確保を招く。
/// generate_scene は最初にこの検証を行う。
///
/// @param config corpus 全体の設定。
/// @param spec 対象 scene の条件。
/// @param out_error 失敗時に「項目名=値」を含む理由を格納する。nullptr は不可。
/// @return 全ての項目が有効なら true。
///
/// 入力例: canonical_marker_px_ = 0 の config
/// 出力例: false、out_error に "canonical_marker_px=0" を含む文字列
bool validate_scene(const CorpusConfig& config, const SceneSpec& spec, std::string* out_error);

/// ArUco3 設定のもとで検出対象となる最小の辺長 (pixel) を返す。
///
/// 導出:
///   縮小率は fxfy = S / (S + L * tau_i)。縮小後の辺長が S 以上である必要があるため
///   side_px * fxfy >= S となり、整理すると side_px >= S + L * tau_i を得る。
///   辺長比で言えば side_ratio >= S / L + tau_i であり、tau_i そのものではない。
///
/// @param min_side_length_canonical_img_px S。OpenCV の minSideLengthCanonicalImg。
/// @param longest_image_side_px L。画像の長辺。
/// @param min_marker_length_ratio_original_img tau_i。
/// @return 下限となる辺長 (pixel)。境界値ちょうどでは再標本化の影響で
///         検出できないことがあるため、corpus では余裕を持たせる。
double minimum_detectable_side_px(int min_side_length_canonical_img_px, int longest_image_side_px,
                                  double min_marker_length_ratio_original_img);

/// preset 名から scene spec の一覧を作る。
///
/// @param preset "smoke"、"basic"、"full" のいずれか。
/// @param out_specs 成功時に spec 一覧を格納する。
/// @return 既知の preset なら true。
bool build_preset(const std::string& preset, std::vector<SceneSpec>* out_specs);

/// 対応する preset 名の一覧を返す。
std::vector<std::string> known_presets();

/// 1 枚の画像を生成して保存する。
///
/// @param config corpus 全体の設定。
/// @param spec この画像の条件。
/// @param scene_index 画像の通し番号。乱数種の導出に使用する。
/// @param out_scene 成功時に結果を格納する。
/// @param out_error 失敗時に理由を格納する。
/// @return 成功した場合は true。
///
/// 備考:
///   乱数種は seed と scene_index から導出する。scene を追加しても
///   既存 scene の内容が変わらないようにするため、通し番号で独立させる。
bool generate_scene(const CorpusConfig& config, const SceneSpec& spec, std::size_t scene_index,
                    GeneratedScene* out_scene, std::string* out_error);

/// manifest を JSON で書き出す。
///
/// 大容量の画像は repository へ commit せず、この manifest に保存先と
/// checksum を記録する。
void write_manifest_json(std::ostream& out, const CorpusConfig& config, const std::string& preset,
                         const std::vector<GeneratedScene>& scenes);

}  // namespace aruco3cuda::corpusgen

#endif  // ARUCO3CUDA_CORPUSGEN_CORPUS_GENERATOR_HPP
