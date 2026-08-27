// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_HYBRID_CPU_CANDIDATES_HPP
#define ARUCO3CUDA_HYBRID_CPU_CANDIDATES_HPP

#include <opencv2/core.hpp>

#include <vector>

#include "aruco3cuda/config.hpp"

/// CPU 経路の四角形候補抽出。
///
/// 目的:
///     OpenCV の ArUco 検出器と同じ手順で二値化画像から候補を求める。
///     ハイブリッド経路が使うほか、GPU 経路との差異を測る比較の基準にも
///     使う。同じ code を両方から呼ぶことで、比較の対象が実装の違いだけ
///     になるようにする。
namespace aruco3cuda::hybrid {

/// 候補 1 つ分の四隅・輪郭・周長。
///
/// 周長は輪郭長ではなく四隅を結んだ四角形の辺長和とする。OpenCV の
/// MarkerCandidate と同じ定義であり、グループ内で採用する候補の選択に使う。
///
/// 所有権: 値型。内部の vector を自身で所有する。
/// 同期動作: 無し。単なる集約である。
///
/// 入力例: 辺 60 の正方形の輪郭
/// 出力例: perimeter_ = 240
struct MarkerCandidate {
    std::vector<cv::Point2f> corners_;
    std::vector<cv::Point> contour_;
    float perimeter_ = 0.0F;
};

/// 候補を包含関係の木として保持する。OpenCV の MarkerCandidateTree と同じ。
///
/// parent_ は自分を内側に含む候補の index、depth_ は自分より内側にある
/// 候補の最大段数である。識別は depth_ の小さい順に行い、マーカーが確定
/// したらその親を識別対象から外す。
///
/// 所有権: 値型。内部の vector を自身で所有する。
/// 同期動作: 無し。単なる集約である。
///
/// 入力例: 黒枠の外周と内周が候補になった場合
/// 出力例: 内周の parent_ が外周の index、外周の depth_ が 1
struct CandidateNode : MarkerCandidate {
    int parent_ = -1;
    int depth_ = 0;
    std::vector<MarkerCandidate> close_contours_;
};

/// 四隅を結んだ四角形の辺長和。
///
/// @param corners 四隅。要素数は 4。
/// @return 辺長の和。
///
/// 所有権: 引数を保持しない。
/// 同期動作: 無し。再入可能。
///
/// 入力例: 辺 10 の正方形
/// 出力例: 40
float quad_perimeter(const std::vector<cv::Point2f>& corners);

/// 2 つの候補の四隅間平均距離。開始頂点の 4 通りの対応のうち最小を採る。
///
/// OpenCV の getAverageDistance と同じ。頂点の並びが 1 つずれていても
/// 同じマーカーだと判定できるようにするため、対応を総当たりする。
///
/// @param first 一方の四隅。要素数は 4。
/// @param second もう一方の四隅。要素数は 4。
/// @return 平均距離。
///
/// 所有権: 引数を保持しない。
/// 同期動作: 無し。再入可能。
///
/// 入力例: 同じ位置の 2 つの正方形
/// 出力例: 0
float average_quad_distance(const std::vector<cv::Point2f>& first,
                            const std::vector<cv::Point2f>& second);

/// 四隅から求めたセル 1 辺の平均画素数。OpenCV の getAverageModuleSize と同じ。
///
/// @param corners 四隅。要素数は 4。
/// @param marker_size Dictionary のセル数。
/// @param border_bits 外枠のセル数。
/// @return セル 1 辺の平均画素数。
///
/// 所有権: 引数を保持しない。
/// 同期動作: 無し。再入可能。
///
/// 入力例: 辺 80 の正方形、marker_size 6、border_bits 1
/// 出力例: 10
float average_module_size(const std::vector<cv::Point2f>& corners, int marker_size,
                          int border_bits);

/// first の四隅が全て second の内側にあるか。OpenCV の checkMarker1InMarker2 と同じ。
///
/// @param first 内側にあるかを調べる四隅。要素数は 4。
/// @param second 外側の四隅。要素数は 4。
/// @return 全ての点が内側または境界上なら true。
///
/// 所有権: 引数を保持しない。
/// 同期動作: 無し。再入可能。
///
/// 入力例: 小さい正方形と、それを囲む大きい正方形
/// 出力例: true
bool quad_inside_quad(const std::vector<cv::Point2f>& first,
                      const std::vector<cv::Point2f>& second);

/// 二値化画像から四角形候補を抽出する。
///
/// OpenCV の _findMarkerContours と同じ判定を行う。ArUco3 有効時は
/// 周長の下限を minSideLengthCanonicalImg * 4 へ置き換える。
///
/// @param binary 入力二値化画像。0 を背景、それ以外を前景とみなす。
/// @param config 検出設定。
/// @param candidates 見つけた四隅を追加する。既存の要素は保持する。
/// @param contours_out 対応する輪郭を追加する。candidates と同じ数だけ増える。
/// @return 無し。引数の vector へ追加する。
///
/// 所有権: 引数の vector は呼出側が所有する。
/// 同期動作: 無し。再入可能。
///
/// 入力例: 正方形が 3 つある二値化画像
/// 出力例: candidates が 3 要素増える
void find_quad_candidates(const cv::Mat& binary, const DetectorConfig& config,
                          std::vector<std::vector<cv::Point2f>>& candidates,
                          std::vector<std::vector<cv::Point>>& contours_out);

/// 四隅を時計回りへ揃える。OpenCV の _reorderCandidatesCorners と同じ。
///
/// @param candidate 並べ替える四隅。要素数は 4。その場で書き換える。
/// @return 無し。引数を書き換える。
///
/// 所有権: 引数は呼出側が所有する。
/// 同期動作: 無し。再入可能。
///
/// 入力例: 反時計回りの四隅
/// 出力例: 1 番目と 3 番目が入れ替わる
void reorder_corners(std::vector<cv::Point2f>& candidate);

/// 近接候補をグループ化し、各グループの代表を選んで包含関係の木を作る。
///
/// 二値化 window を変えると同じマーカーから少しずつ違う候補が得られる。
/// OpenCV の filterTooCloseCandidates と同じく、周長の降順に並べたうえで
/// 近接するものを 1 グループとし、グループ内で最大周長の候補を代表に採る。
/// 代表以外のうち代表から離れているものは close_contours_ として残し、
/// 代表の識別が失敗した場合の代替に使う。
///
/// 代表が画像端に近すぎる場合はグループごと捨てる。OpenCV も同じ扱いで、
/// 端に掛かったマーカーは四隅が信用できないためである。
///
/// @param image_size 二値化画像の寸法。端からの距離の判定に使う。
/// @param candidates 候補の四隅。中身は move で取り出されるため呼出後は空になる。
/// @param contours 対応する輪郭。同じく move で取り出される。
/// @param config 検出設定。
/// @param marker_size Dictionary のセル数。
/// @return 代表候補の一覧。周長の降順に並ぶ。
///
/// 所有権: 引数の vector の要素は戻り値へ move される。
/// 同期動作: 無し。再入可能。
///
/// 入力例: 同じマーカーから得た周長違いの候補が 3 つ
/// 出力例: 要素数 1。最大周長の候補が残る
std::vector<CandidateNode> filter_too_close_candidates(
        const cv::Size& image_size, std::vector<std::vector<cv::Point2f>>& candidates,
        std::vector<std::vector<cv::Point>>& contours, const DetectorConfig& config,
        int marker_size);

}  // namespace aruco3cuda::hybrid

#endif  // ARUCO3CUDA_HYBRID_CPU_CANDIDATES_HPP
