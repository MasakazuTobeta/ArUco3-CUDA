// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_UTIL_STATISTICS_HPP
#define ARUCO3CUDA_UTIL_STATISTICS_HPP

#include <cstddef>
#include <vector>

namespace aruco3cuda::util {

/// 測定標本の要約統計。
///
/// 評価計画は平均値だけでなく中央値と裾の分位点を求める。外れ値は除去せず、
/// 全標本から算出する。
struct SampleStatistics {
    std::size_t count_ = 0;
    double min_ = 0.0;
    double max_ = 0.0;
    double mean_ = 0.0;
    /// 標本標準偏差。count_ が 1 の場合は 0 とする。
    double stddev_ = 0.0;
    double p50_ = 0.0;
    double p95_ = 0.0;
    double p99_ = 0.0;
};

/// 分位点を nearest-rank 法で求める。
///
/// 昇順に並べた標本に対し、順位 ceil(percentile / 100 * count) の値を返す。
/// 補間を行わないため、返る値は必ず実測値のいずれかである。集計方法を
/// 実装依存にしないよう、この方法を本 project の標準とする。
///
/// @param sorted_samples 昇順に並んだ標本。空であってはならない。
/// @param percentile 0 より大きく 100 以下。
/// @return 対応する標本値。引数が不正な場合は 0 を返す。
///
/// 入力例: {1, 2, 3, 4, 5}、percentile = 50
/// 出力例: 3
double percentile_nearest_rank(const std::vector<double>& sorted_samples, double percentile);

/// 標本から要約統計を求める。
///
/// 外れ値の除去は行わない。有利な結果だけが残ることを避けるため、
/// 集計は常に全標本に対して行う。
///
/// @param samples 標本。順序は問わない。空の場合は false を返す。
/// @param out 成功時に結果を格納する。失敗時は変更しない。
/// @return 成功した場合は true。
bool compute_statistics(const std::vector<double>& samples, SampleStatistics* out);

}  // namespace aruco3cuda::util

#endif  // ARUCO3CUDA_UTIL_STATISTICS_HPP
