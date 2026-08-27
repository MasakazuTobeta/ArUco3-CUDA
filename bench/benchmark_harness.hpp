// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_BENCH_BENCHMARK_HARNESS_HPP
#define ARUCO3CUDA_BENCH_BENCHMARK_HARNESS_HPP

#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

#include "aruco3cuda/util/statistics.hpp"
#include "reference_runner.hpp"

namespace aruco3cuda::bench {

/// 評価計画が定める比較経路。
///
/// 現時点で実装があるのは kCpu のみである。CUDA 経路は Phase 1 以降で追加するが、
/// 結果 schema を後から変えずに済むよう識別子を先に定義しておく。
enum class Route : int {
    kCpu = 0,        ///< OpenCV ArUco3。cv::Mat 入力から結果取得まで
    kCudaEndToEnd,   ///< host 入力 CUDA。upload、検出、download、同期を含む
    kCudaResident,   ///< device 入力 CUDA。GPU 常駐画像から device 結果まで
    kHybrid,         ///< CUDA と CPU の組み合わせ
};

/// 入力 buffer の memory 種別。
///
/// DGX Spark と Jetson Orin はいずれも統合 GPU であり、明示的な copy の費用が
/// discrete GPU と大きく異なる。経路と独立した測定軸として記録する。
enum class MemoryMode : int {
    kNotApplicable = 0, ///< CPU 経路
    kHostPageable,
    kHostPinned,
    kManaged,
    kDevice,
};

/// Route を評価計画の表記へ変換する。結果 JSONL の識別子に使用する。
///
/// @param route 変換対象。列挙に無い値でも nullptr を返さない。
/// @return 静的記憶域を持つ文字列。
///
/// 所有権: 戻り値は静的記憶域を指す。呼出側は解放も変更もしない。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: Route::kCudaEndToEnd
/// 出力例: "CUDA-E2E"
const char* to_string(Route route);

/// MemoryMode を評価計画の表記へ変換する。
///
/// @param mode 変換対象。列挙に無い値でも nullptr を返さない。
/// @return 静的記憶域を持つ文字列。
///
/// 所有権: 戻り値は静的記憶域を指す。呼出側は解放も変更もしない。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: MemoryMode::kHostPinned
/// 出力例: "M-Pinned"
const char* to_string(MemoryMode mode);

/// 測定条件。
struct BenchmarkConfig {
    Route route_ = Route::kCpu;
    MemoryMode memory_mode_ = MemoryMode::kNotApplicable;

    /// 測定区間へ含めない準備実行の回数。
    int warmup_iterations_ = 20;
    /// 単一フレーム遅延の測定回数。
    int latency_iterations_ = 200;
    /// throughput 測定で連続処理するフレーム数。0 で測定しない。
    int throughput_frames_ = 100;
    /// 全標本を結果へ含めるか。分布を保存する必要がある場合に使用する。
    bool save_all_samples_ = false;

    /// 測定に使用する CPU 番号の一覧。空なら OS の割り当てに任せる。
    ///
    /// DGX Spark GB10 のように性能 core と効率 core が混在する機では、
    /// 割り当て先の core 種別で CPU 基準値が 1.6 倍変わる。core を固定しないと
    /// 測定値が実行ごとに二極化し、crossover point の判断を誤る。
    std::vector<int> cpu_affinity_;

    aruco3cuda::reference::ReferenceConfig detector_;
};

/// 1 つの入力に対する測定結果。
struct MeasurementRecord {
    std::string image_path_;
    std::string image_sha256_;
    int width_px_ = 0;
    int height_px_ = 0;
    std::size_t detection_count_ = 0;

    /// ArUco3 の実効縮小率。測定条件として必ず記録する。
    double fxfy_effective_ = 1.0;

    /// 入力準備から host 結果取得までの wall-clock。単位は ms。
    aruco3cuda::util::SampleStatistics end_to_end_ms_;
    /// CUDA event で測定する検出処理時間。CPU 経路では未測定とする。
    bool kernel_time_available_ = false;
    aruco3cuda::util::SampleStatistics kernel_ms_;

    /// 連続処理時の frame/s。throughput_frames_ が 0 の場合は未測定。
    bool throughput_available_ = false;
    double throughput_fps_ = 0.0;

    /// save_all_samples_ が true の場合のみ格納する。
    std::vector<double> end_to_end_samples_ms_;
};

/// 実行環境の記録。
struct EnvironmentRecord {
    std::string hostname_;
    std::string os_;
    std::string kernel_;
    std::string architecture_;
    std::string opencv_version_;
    int opencv_threads_ = 0;
    std::string cuda_toolkit_version_;
    std::string gpu_name_;
    std::string gpu_compute_capability_;
    std::string driver_version_;
    /// Jetson の L4T release。Jetson には nvidia-smi が無く driver version を
    /// 取得できないため、対応する情報としてこちらを記録する。
    std::string platform_release_;
    /// 基板名。device tree から取得する。
    std::string platform_model_;
    /// 電力モード。評価計画が測定条件として記録を要求する。
    std::string power_mode_;
    /// CPU の core 構成。性能 core と効率 core が混在する機では、
    /// どの種別で測ったかが分からないと測定値を比較できない。
    std::string cpu_topology_;
    /// 実際に使用した CPU 番号。固定しなかった場合は "unpinned"。
    std::string cpu_affinity_;
    /// address space 配置の無作為化 (ASLR) の状態。
    ///
    /// 全解像度の CPU 経路では、ASLR による memory 配置の違いだけで
    /// p50 が 9% 変動する。無効化すると実行間で完全に一致する。
    /// どちらで測ったかが分からないと測定値を比較できない。
    std::string address_randomization_;
    /// GPU の最大 clock (MHz)。取得できない場合は 0 ではなく未設定とする。
    bool gpu_clock_available_ = false;
    int gpu_max_clock_mhz_ = 0;
    int gpu_current_clock_mhz_ = 0;
    bool gpu_integrated_ = false;
    int cpu_online_cores_ = 0;
    /// GPU 情報を取得できなかった場合の理由。空なら取得に成功している。
    /// 無言で項目が欠けることを防ぎ、後から原因を追えるようにする。
    std::string gpu_probe_error_;
};

/// 実行環境の情報を収集する。
///
/// GPU 情報は CUDA device が利用できる場合のみ埋まる。取得できない項目は
/// 空文字列のままとし、推測で埋めない。
///
/// @param config 検出設定。OpenCV の thread 数の設定にのみ使用する。
/// @return 収集した環境情報。取得できなかった項目は空文字列または未設定を示す flag を持つ。
///
/// 所有権: 戻り値は値であり、内部の資源を参照しない。
/// 同期動作: CUDA device の性質取得で CUDA runtime を呼ぶが device 同期は行わない。
///           power mode と driver version の取得のため外部 command を起動する。
///
/// 副作用: config.detector_.num_threads_ が 1 以上なら cv::setNumThreads() を呼び、
///         以降の OpenCV 処理の thread 数を変更する。測定条件を固定するための意図的な副作用である。
///
/// 入力例: 既定の BenchmarkConfig
/// 出力例: opencv_version_ = "4.14.0"、gpu_name_ = "Orin"、power_mode_ = "MAXN (0)"
EnvironmentRecord collect_environment(const BenchmarkConfig& config);

/// 1 つの画像を測定する。
///
/// @param image_path 対象画像。
/// @param config 測定条件。
/// @param out_record 成功時に結果を格納する。
/// @param out_error 失敗時に理由を格納する。
/// @return 成功した場合は true。
///
/// 備考:
///   現時点で実装があるのは CPU 経路のみである。他の経路を指定した場合は
///   未実装であることを明示して失敗する。無言で CPU 経路へ読み替えない。
bool measure_image(const std::string& image_path, const BenchmarkConfig& config,
                   MeasurementRecord* out_record, std::string* out_error);

/// 環境情報を JSONL の 1 行として書き出す。
///
/// 末尾へ改行を 1 つ書く。JSONL は 1 行 1 record であり、この行が結果 file の先頭になる。
///
/// @param out 出力先。所有権は呼出側が保持し、この関数は解放も flush も行わない。
///            書き込み失敗の確認は呼出側の責務である。
/// @param environment 書き出す環境情報。取得できなかった clock は 0 ではなく null になる。
/// @return 無し。
///
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: collect_environment() の戻り値
/// 出力例: {"type":"environment","schema_version":1,...}
void write_environment_line(std::ostream& out, const EnvironmentRecord& environment);

/// 測定結果を JSONL の 1 行として書き出す。
///
/// 末尾へ改行を 1 つ書く。CPU 経路には kernel 時間が存在しないため、
/// kernel_time_available_ が false の場合は 0 で埋めず null を出力する。
/// 0 を書くと集計時に「非常に速い kernel」と誤読されるためである。
///
/// @param out 出力先。所有権は呼出側が保持し、この関数は解放も flush も行わない。
///            書き込み失敗の確認は呼出側の責務である。
/// @param config 測定条件。経路、memory 種別、検出設定を条件として記録する。
/// @param record 測定結果。
/// @return 無し。
///
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: measure_image() の戻り値と、その測定に使用した BenchmarkConfig
/// 出力例: {"type":"measurement","route":"CPU",...,"kernel":null,...}
void write_measurement_line(std::ostream& out, const BenchmarkConfig& config,
                            const MeasurementRecord& record);

}  // namespace aruco3cuda::bench

#endif  // ARUCO3CUDA_BENCH_BENCHMARK_HARNESS_HPP
