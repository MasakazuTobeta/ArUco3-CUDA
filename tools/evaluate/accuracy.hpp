// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_TOOLS_EVALUATE_ACCURACY_HPP
#define ARUCO3CUDA_TOOLS_EVALUATE_ACCURACY_HPP

#include <cstddef>
#include <string>
#include <vector>

#include "geometry.hpp"

/// ground truth と検出結果を突き合わせ、正確性指標を算出する。
///
/// 目的:
///     [差分レポート](../report/report_diff.hpp) は CPU 基準実装を基準に据える。
///     しかし基準実装は互換性の oracle であって ground truth ではない。基準が
///     取りこぼしたマーカーは差分に現れず、「一致率 100%」でも検出できていない
///     ことがある。生成時に判っている真値と突き合わせ、precision と recall を
///     別に測る。
namespace aruco3cuda::evaluate {

using aruco3cuda::report::Quad;

/// 生成時に判っている真値 1 件。
///
/// 所有権: 値型。参照や pointer を持たない。
/// 同期動作: 無し。
///
/// 入力例: 中心 (100, 100)、1 辺 40 pixel に描かれた ID 7 のマーカー
/// 出力例: id_ = 7、side_px_ = 40.0、fully_inside_ = true
struct TruthMarker {
    int id_ = -1;
    /// 四隅。マーカー座標系の (0,0)、(S,0)、(S,S)、(0,S) に対応する。
    Quad corners_{};
    /// 1 辺の長さ (pixel)。
    double side_px_ = 0.0;
    /// 四隅すべてが画像内に収まっているか。
    bool fully_inside_ = true;
    /// マーカーを覆う遮蔽の面積比。
    double occlusion_ratio_ = 0.0;
};

/// 真値 1 件の結末。
enum class TruthOutcome {
    /// 位置が対応し ID も一致した。
    kDetected,
    /// 位置は対応したが ID を誤られた。
    kIdMismatched,
    /// 対応する検出が無かった。
    kMissed,
};

/// 評価対象が返した検出 1 件。
///
/// 所有権: 値型。
/// 同期動作: 無し。
///
/// 入力例: 検出器が返した ID 7 と四隅
/// 出力例: id_ = 7、corners_ に 8 要素
struct Observation {
    int id_ = -1;
    Quad corners_{};
};

/// 突き合わせの設定。
struct MatchConfig {
    /// 真値と検出を同じマーカーとみなす重心距離の上限。真値の 1 辺に対する比。
    double match_radius_ratio_ = 0.5;
};

/// 画像 1 枚分の突き合わせ結果。
///
/// 次の 2 項目は member 関数にも適用される。
/// 所有権: 値型。参照や pointer を持たない。
/// 同期動作: 無し。const member 関数は並行に呼べる。
///
/// 入力例: 真値 4 件に対し 3 件を正しい ID で検出し、1 件を取りこぼした画像
/// 出力例: truth_count_ = 4、true_positive_ = 3、false_negative_ = 1
struct ImageAccuracy {
    std::string image_name_;
    std::size_t truth_count_ = 0;
    std::size_t observed_count_ = 0;
    /// 位置が対応し ID も一致した件数。
    std::size_t true_positive_ = 0;
    /// 対応が付かない検出と、ID を誤った検出の合計。
    std::size_t false_positive_ = 0;
    /// 対応が付かない真値と、ID を誤られた真値の合計。
    std::size_t false_negative_ = 0;
    /// true_positive_ のうち四隅の並びまで一致した件数。
    std::size_t rotation_agreed_ = 0;
    /// true_positive_ の四隅距離の二乗和 (pixel^2)。
    double corner_squared_sum_px2_ = 0.0;
    /// 二乗和へ入れた四隅の数。true_positive_ の 4 倍になる。
    std::size_t corner_sample_count_ = 0;
    /// true_positive_ における四隅距離の最大値 (pixel)。
    double corner_max_px_ = 0.0;
    /// 真値 1 件ごとの結末。並びは compare_to_truth へ渡した truth に対応する。
    std::vector<TruthOutcome> truth_outcomes_;
    /// 真値 1 件ごとの四隅距離の二乗和 (pixel^2)。kDetected 以外では 0。
    std::vector<double> truth_squared_error_px2_;
    /// 真値 1 件ごとの四隅距離の最大値 (pixel)。kDetected 以外では 0。
    std::vector<double> truth_corner_max_px_;
    /// 真値 1 件ごとの四隅の並びの一致。kDetected 以外では false。
    std::vector<bool> truth_rotation_agreed_;
};

/// 複数画像をまとめた集計。
///
/// 所有権: 値型。
/// 同期動作: 無し。
///
/// 入力例: 真値 4 件のうち 3 件を正しく検出した画像 2 枚を足し込んだ状態
/// 出力例: image_count_ = 2、truth_count_ = 8、true_positive_ = 6
struct AccuracySummary {
    std::size_t image_count_ = 0;
    std::size_t truth_count_ = 0;
    std::size_t observed_count_ = 0;
    std::size_t true_positive_ = 0;
    std::size_t false_positive_ = 0;
    std::size_t false_negative_ = 0;
    std::size_t rotation_agreed_ = 0;
    double corner_squared_sum_px2_ = 0.0;
    std::size_t corner_sample_count_ = 0;
    double corner_max_px_ = 0.0;
};

/// 真値と検出を突き合わせ、1 枚分の指標を求める。
///
/// 対応付けは ID ではなく重心の近さで行う。ID で対応を取ると、ID を読み違えた
/// 場合に「未検出」と「過検出」が 1 件ずつ立ち、同じマーカーの ID を誤ったのか
/// 別の場所で誤検出したのかが区別できなくなる。
///
/// @param image_name 報告へ残す画像名。参照するだけで保持しない。
/// @param truth 生成時の真値。参照するだけで保持しない。
/// @param observed 評価対象の検出。参照するだけで保持しない。
/// @param config 対応付けの設定。参照するだけで保持しない。
/// @return 1 枚分の指標。
///
/// 所有権: 引数の領域を保持しない。戻り値は値として返る。
/// 同期動作: 無し。引数を変更しないため、同じ入力に対して並行に呼べる。
///
/// 入力例: 真値 1 件と、同じ位置・同じ ID の検出 1 件
/// 出力例: true_positive_ = 1、false_positive_ = 0、false_negative_ = 0
ImageAccuracy compare_to_truth(const std::string& image_name, const std::vector<TruthMarker>& truth,
                               const std::vector<Observation>& observed, const MatchConfig& config);

/// 1 枚分の指標を集計へ足し込む。
///
/// @param image 足し込む 1 枚分の指標。参照するだけで保持しない。
/// @param out_summary 足し込み先。領域の所有権は呼出側にある。nullptr は不可。
/// @return 無し。out_summary の内容だけが変化する。
///
/// 所有権: 引数の領域を保持しない。
/// 同期動作: 無し。同じ out_summary へ並行に足し込む場合は呼出側で排他する。
///
/// 入力例: true_positive_ = 1 の 1 枚を空の集計へ足す
/// 出力例: image_count_ = 1、true_positive_ = 1
void accumulate(const ImageAccuracy& image, AccuracySummary* out_summary);

/// 真値の部分集合だけを集計へ足し込む。
///
/// 条件別の recall を出すために使う。ArUco3 は縮小後の 1 辺が
/// `min_side_length_canonical_img_px` を下回るマーカーを原理上検出しない。
/// 検出できない大きさの真値を含めたまま recall を出すと、実装の取りこぼしと
/// 戦略上の下限が混ざり、どちらの数値なのか読み取れない。
///
/// 対応の付かない検出 (false positive) はどの真値にも属さないため足し込まない。
/// したがって求まる集計から precision は定義できない。recall と四隅の指標
/// だけを読むこと。
///
/// @param image 足し込む 1 枚分の指標。参照するだけで保持しない。
/// @param selected 真値の並びに対応する採否。要素数が truth_outcomes_ と
///                 異なる場合は何もしない。
/// @param out_summary 足し込み先。領域の所有権は呼出側にある。nullptr は不可。
/// @return 無し。out_summary の内容だけが変化する。
///
/// 所有権: 引数の領域を保持しない。
/// 同期動作: 無し。同じ out_summary へ並行に足し込む場合は呼出側で排他する。
///
/// 入力例: 真値 2 件のうち 1 件目だけを選んだ採否
/// 出力例: truth_count_ = 1。observed_count_ と false_positive_ は増えない
void accumulate_selected(const ImageAccuracy& image, const std::vector<bool>& selected,
                         AccuracySummary* out_summary);

/// precision を求める。
///
/// 検出が 1 件も無い場合は定義できない。0 除算を 1.0 とみなすと「検出しない
/// ほど precision が高い」ことになり、判断を誤らせる。
///
/// @param summary 対象の集計。参照するだけで保持しない。
/// @param out_value 求まった場合に格納する。領域の所有権は呼出側にある。nullptr は不可。
/// @return 定義できる場合は true。
///
/// 所有権: 引数の領域を保持しない。
/// 同期動作: 無し。再入可能。
///
/// 入力例: true_positive_ = 3、false_positive_ = 1 の集計
/// 出力例: true、out_value = 0.75
bool precision(const AccuracySummary& summary, double* out_value);

/// recall を求める。
///
/// 真値が 1 件も無い場合は定義できない。マーカー 0 個の場面が corpus に
/// 含まれるため、この場合は実際に起こる。
///
/// @param summary 対象の集計。参照するだけで保持しない。
/// @param out_value 求まった場合に格納する。領域の所有権は呼出側にある。nullptr は不可。
/// @return 定義できる場合は true。
///
/// 所有権: 引数の領域を保持しない。
/// 同期動作: 無し。再入可能。
///
/// 入力例: true_positive_ = 3、false_negative_ = 1 の集計
/// 出力例: true、out_value = 0.75
bool recall(const AccuracySummary& summary, double* out_value);

/// 四隅の RMSE を求める。
///
/// 最大値と別に持つ。1 隅だけ大きく外れた場合と 4 隅が一様にずれた場合を
/// 区別するためである。
///
/// @param summary 対象の集計。参照するだけで保持しない。
/// @param out_value 求まった場合に格納する。領域の所有権は呼出側にある。nullptr は不可。
/// @return 対応が 1 件以上あれば true。
///
/// 所有権: 引数の領域を保持しない。
/// 同期動作: 無し。再入可能。
///
/// 入力例: corner_squared_sum_px2_ = 4.0、corner_sample_count_ = 4 の集計
/// 出力例: true、out_value = 1.0
bool corner_rmse_px(const AccuracySummary& summary, double* out_value);

}  // namespace aruco3cuda::evaluate

#endif  // ARUCO3CUDA_TOOLS_EVALUATE_ACCURACY_HPP
