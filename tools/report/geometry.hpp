// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_TOOLS_REPORT_GEOMETRY_HPP
#define ARUCO3CUDA_TOOLS_REPORT_GEOMETRY_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <utility>

/// 四隅を持つ検出どうしを突き合わせるための幾何計算。
///
/// 目的:
///     基準との差分分類 (report_diff) と ground truth との照合 (evaluate) は、
///     同じ規則で対応付けを行う必要がある。片方だけ規則が変わると 2 つの報告が
///     食い違い、どちらが正しいか判断できなくなる。共有する形で 1 か所に置く。
namespace aruco3cuda::report {

/// 四隅の並び。x0, y0, x1, y1, x2, y2, x3, y3 の順。
using Quad = std::array<double, 8>;

/// 四隅の数。
constexpr std::size_t kQuadCorners = 4U;

/// 四隅の重心を返す。
///
/// @param quad 四隅。参照するだけで保持しない。
/// @return (x, y) の組。単位は pixel。
///
/// 所有権: 引数を保持しない。戻り値は値として返る。
/// 同期動作: 無し。再入可能。
///
/// 入力例: {0,0, 2,0, 2,2, 0,2}
/// 出力例: (1.0, 1.0)
inline std::pair<double, double> quad_centroid(const Quad& quad) {
    double x = 0.0;
    double y = 0.0;
    for (std::size_t c = 0; c < kQuadCorners; ++c) {
        x += quad[c * 2U];
        y += quad[(c * 2U) + 1U];
    }
    return {x / static_cast<double>(kQuadCorners), y / static_cast<double>(kQuadCorners)};
}

/// 四隅を結んだ四角形の平均辺長を返す。
///
/// 対応付けの半径を絶対値ではなく辺長比で決めるために使う。解像度と
/// マーカーの大きさによって妥当な距離が変わるためである。
///
/// @param quad 四隅。参照するだけで保持しない。
/// @return 平均辺長。単位は pixel。
///
/// 所有権: 引数を保持しない。
/// 同期動作: 無し。再入可能。
///
/// 入力例: {0,0, 2,0, 2,2, 0,2}
/// 出力例: 2.0
inline double quad_average_side(const Quad& quad) {
    double total = 0.0;
    for (std::size_t c = 0; c < kQuadCorners; ++c) {
        const std::size_t next = (c + 1U) % kQuadCorners;
        const double dx = quad[c * 2U] - quad[next * 2U];
        const double dy = quad[(c * 2U) + 1U] - quad[(next * 2U) + 1U];
        total += std::sqrt((dx * dx) + (dy * dy));
    }
    return total / static_cast<double>(kQuadCorners);
}

/// 対象の四隅を steps 段巡回させたときの、対応する四隅の最大距離を返す。
///
/// @param baseline 基準の四隅。参照するだけで保持しない。
/// @param target 対象の四隅。参照するだけで保持しない。
/// @param steps 巡回段数。0 から 3。
/// @return 4 隅の距離の最大値。単位は pixel。
///
/// 所有権: 引数を保持しない。
/// 同期動作: 無し。再入可能。
///
/// 入力例: 同じ四隅どうし、steps = 0
/// 出力例: 0.0
inline double quad_corner_error(const Quad& baseline, const Quad& target, int steps) {
    double worst = 0.0;
    for (std::size_t c = 0; c < kQuadCorners; ++c) {
        const std::size_t shifted = (c + static_cast<std::size_t>(steps)) % kQuadCorners;
        const double dx = baseline[c * 2U] - target[shifted * 2U];
        const double dy = baseline[(c * 2U) + 1U] - target[(shifted * 2U) + 1U];
        worst = std::max(worst, std::sqrt((dx * dx) + (dy * dy)));
    }
    return worst;
}

/// 対象の四隅を steps 段巡回させたときの、対応する四隅の距離の二乗和を返す。
///
/// RMSE は最大値と別の情報を持つ。1 隅だけ大きく外れた場合と 4 隅が一様に
/// ずれた場合を区別するため、二乗和も取れるようにする。
///
/// @param baseline 基準の四隅。参照するだけで保持しない。
/// @param target 対象の四隅。参照するだけで保持しない。
/// @param steps 巡回段数。0 から 3。
/// @return 4 隅の距離の二乗和。単位は pixel^2。
///
/// 所有権: 引数を保持しない。
/// 同期動作: 無し。再入可能。
///
/// 入力例: 1 隅だけ 2 pixel ずれた四隅、steps = 0
/// 出力例: 4.0
inline double quad_squared_error(const Quad& baseline, const Quad& target, int steps) {
    double total = 0.0;
    for (std::size_t c = 0; c < kQuadCorners; ++c) {
        const std::size_t shifted = (c + static_cast<std::size_t>(steps)) % kQuadCorners;
        const double dx = baseline[c * 2U] - target[shifted * 2U];
        const double dy = baseline[(c * 2U) + 1U] - target[(shifted * 2U) + 1U];
        total += (dx * dx) + (dy * dy);
    }
    return total;
}

}  // namespace aruco3cuda::report

#endif  // ARUCO3CUDA_TOOLS_REPORT_GEOMETRY_HPP
