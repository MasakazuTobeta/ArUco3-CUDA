// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_TOOLS_REPORT_REPORT_DIFF_HPP
#define ARUCO3CUDA_TOOLS_REPORT_REPORT_DIFF_HPP

#include <array>
#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

/// 検出結果の差異を分類する。
///
/// 目的:
///     CPU 基準実装と評価対象の経路が返した検出結果を突き合わせ、差異を
///     種類ごとに数える。「一致率 98%」のような 1 つの数値では、四隅が
///     わずかにずれているのか、マーカーを取りこぼしているのかが区別
///     できない。判断に必要な粒度で残すため、種類を分けて記録する。
namespace aruco3cuda::report {

/// 突き合わせに使う 1 件の検出。
///
/// 経路によって内部表現が異なるため、比較に必要な項目だけを持つ共通形とする。
struct Detection {
    int id_ = -1;
    /// 四隅を x0, y0, x1, y1, x2, y2, x3, y3 の順で保持する。
    std::array<double, 8> corners_{};
};

/// 差異の種類。
enum class DiffKind {
    /// 基準にあり対象に無い。
    kMissed,
    /// 対象にあり基準に無い。
    kExtra,
    /// 同じ位置にあるが ID が異なる。
    kIdMismatch,
    /// ID は同じだが四隅の並びが巡回してずれている。
    kRotationMismatch,
    /// ID も並びも同じだが四隅の位置が許容差を超える。
    kCornerShift,
};

/// 差異 1 件。
///
/// 種類によって意味を持つ field が変わる。kMissed では target_id_ が -1、
/// kExtra では baseline_id_ が -1、rotation_steps_ は kRotationMismatch
/// でのみ 0 以外になる。
///
/// 所有権: 値型。参照や pointer を持たない。
/// 同期動作: 無し。単なる集約であり、同期は呼び出し側の責務。
///
/// 入力例: 基準 id=7 の位置に対象 id=8 がある
/// 出力例: kind_ = kIdMismatch、baseline_id_ = 7、target_id_ = 8
struct Diff {
    DiffKind kind_ = DiffKind::kMissed;
    /// 基準側の ID。kExtra では -1。
    int baseline_id_ = -1;
    /// 対象側の ID。kMissed では -1。
    int target_id_ = -1;
    /// 対象の四隅が基準に対して何段巡回しているか。kRotationMismatch でのみ意味を持つ。
    int rotation_steps_ = 0;
    /// 四隅の最大距離 (pixel)。対応が付かない場合は 0。
    double corner_error_px_ = 0.0;
    /// 基準側の重心 (pixel)。差異の位置を示すために持つ。
    double center_x_px_ = 0.0;
    double center_y_px_ = 0.0;
};

/// 画像 1 枚分の比較結果。
///
/// 次の 2 項目は member 関数に適用される。
/// 所有権: 値型。内部の vector と string を自身で所有する。
/// 同期動作: 無し。const member 関数は並行に呼べる。
///
/// 入力例: 基準 4 件、対象 3 件で 1 件が取りこぼされた画像
/// 出力例: baseline_count_ = 4、target_count_ = 3、agreed_count_ = 3、
///         diffs_ に kMissed が 1 件
struct ImageComparison {
    std::string image_path_;
    std::size_t baseline_count_ = 0;
    std::size_t target_count_ = 0;
    /// 種類を問わず差異が無かった検出の数。
    std::size_t agreed_count_ = 0;
    /// 対応が付いた検出における四隅の最大距離 (pixel)。
    double worst_corner_error_px_ = 0.0;
    std::vector<Diff> diffs_;

    /// 差異が 1 件も無いか。
    ///
    /// @return 差異が 1 件も無ければ true。
    ///
    /// 入力例: diffs_ が空
    /// 出力例: true
    bool agrees() const { return this->diffs_.empty(); }
};

/// 比較の設定。
struct CompareConfig {
    /// 四隅の距離がこの値以下なら一致とみなす。単位は pixel。
    double corner_tolerance_px_ = 1.0;
    /// 2 つの検出を同じマーカーとみなす重心距離の上限。マーカー 1 辺に対する比。
    ///
    /// 絶対値ではなく比とするのは、画像の解像度とマーカーの大きさによって
    /// 妥当な距離が変わるためである。
    double match_radius_ratio_ = 0.5;
};

/// 種類を日本語名で返す。報告の表示に使う。
///
/// @param kind 名前を求める差異の種類。
/// @return 静的記憶域にある文字列。列挙に無い値では "不明" を返す。
///
/// 所有権: 戻り値の所有権は移らない。解放してはならない。
/// 同期動作: 無し。再入可能。
///
/// 入力例: DiffKind::kMissed
/// 出力例: "未検出"
const char* diff_kind_name(DiffKind kind);

/// 基準と対象を突き合わせ、差異を分類する。
///
/// 対応付けは ID ではなく重心の近さで行う。ID で対応を取ると、ID を
/// 読み違えた場合に「未検出」と「過検出」が 1 件ずつ計上され、実際に
/// 起きたこと (同じマーカーの ID を誤った) が読み取れなくなる。
///
/// 入力例: 基準 4 件、対象 4 件で 1 件の ID が異なる
/// 出力例: diffs_ に kIdMismatch が 1 件、agreed_count_ は 3
ImageComparison compare_detections(const std::string& image_path,
                                   const std::vector<Detection>& baseline,
                                   const std::vector<Detection>& target,
                                   const CompareConfig& config);

/// 複数画像の比較結果をまとめた集計。
///
/// 所有権: 値型。参照や pointer を持たない。
/// 同期動作: 無し。
///
/// 入力例: 3 枚のうち 1 枚が一致し、他の 2 枚に未検出と過検出が 1 件ずつ
/// 出力例: image_count_ = 3、agreed_image_count_ = 1、
///         kind_counts_ の kMissed と kExtra がそれぞれ 1
struct Summary {
    std::size_t image_count_ = 0;
    std::size_t agreed_image_count_ = 0;
    std::size_t baseline_detection_count_ = 0;
    std::size_t target_detection_count_ = 0;
    std::size_t agreed_detection_count_ = 0;
    /// 種類ごとの件数。DiffKind の並びに対応する。
    std::array<std::size_t, 5> kind_counts_{};
    double worst_corner_error_px_ = 0.0;
};

/// 画像ごとの比較結果を集計する。
///
/// @param comparisons 集計対象。順序は結果に影響しない。
/// @return 種類ごとの件数と最大差を含む集計。
///
/// 所有権: 引数を保持しない。戻り値は値として返る。
/// 同期動作: 無し。引数を変更しないため、同じ入力に対して並行に呼べる。
///
/// 入力例: 一致 1 枚と未検出 1 件を含む 1 枚
/// 出力例: image_count_ = 2、agreed_image_count_ = 1
Summary summarize(const std::vector<ImageComparison>& comparisons);

/// 人が読む形式で報告を書き出す。
///
/// 有利な結果だけを選ばないよう、差異のある画像は全て列挙する。
///
/// @param out 出力先。呼び出し中のみ使用する。
/// @param comparisons 画像ごとの比較結果。
/// @param summary comparisons を集計したもの。
/// @return 無し。out の状態のみが変化する。
///
/// 所有権: out の所有権は移らない。
/// 同期動作: 無し。同じ out へ並行に書く場合は呼び出し側で排他する。
///
/// 入力例: 差異の無い比較結果 1 件
/// 出力例: "差異は無い。" を含む複数行
void write_text_report(std::ostream& out, const std::vector<ImageComparison>& comparisons,
                       const Summary& summary);

/// 機械可読形式で報告を書き出す。
///
/// @param out 出力先。呼び出し中のみ使用する。
/// @param comparisons 画像ごとの比較結果。
/// @param summary comparisons を集計したもの。
/// @param config 比較に使った設定。判断の根拠として出力へ含める。
/// @return 無し。out の状態のみが変化する。
///
/// 所有権: out の所有権は移らない。
/// 同期動作: 無し。同じ out へ並行に書く場合は呼び出し側で排他する。
///
/// 入力例: 未検出 1 件を含む比較結果 1 件
/// 出力例: summary.kindCounts.missed が 1 の JSON
void write_json_report(std::ostream& out, const std::vector<ImageComparison>& comparisons,
                       const Summary& summary, const CompareConfig& config);

}  // namespace aruco3cuda::report

#endif  // ARUCO3CUDA_TOOLS_REPORT_REPORT_DIFF_HPP
