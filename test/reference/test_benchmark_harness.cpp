// SPDX-License-Identifier: Apache-2.0
//
// 測定 harness を検証する。
//
// 測定値そのものは環境依存で再現しないため、検証対象は次に限る。
//   - 測定条件と環境情報が結果へ確実に残ること
//   - 未実装の経路を無言で CPU へ読み替えないこと
//   - 統計の整合性 (min <= p50 <= p95 <= p99 <= max)
#include "benchmark_harness.hpp"
#include "reference_runner.hpp"

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <ratio>
#include <sstream>
#include <string>

#include <unistd.h>
#include <vector>

namespace {

bool has_cuda_device() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

using aruco3cuda::bench::BenchmarkConfig;
using aruco3cuda::bench::MeasurementRecord;
using aruco3cuda::bench::Route;

class BenchmarkHarnessTest : public ::testing::Test {
protected:
    void SetUp() override {
        const cv::aruco::Dictionary dictionary =
                cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_MIP_36h12);
        cv::Mat marker;
        dictionary.generateImageMarker(17, 160, marker, 1);
        cv::Mat scene(480, 640, CV_8UC1, cv::Scalar(230));
        marker.copyTo(scene(cv::Rect(200, 150, 160, 160)));
        // 画像の path を test ごとに分ける。ctest -j で同時に走る他の test の
        // TearDown に消されないようにする。
        const ::testing::TestInfo* info = ::testing::UnitTest::GetInstance()->current_test_info();
        this->image_path_ = std::string("/tmp/aruco3cuda_bench_test_") +
                            (info != nullptr ? info->name() : "unknown") + "_" +
                            std::to_string(static_cast<long>(::getpid())) + ".png";
        ASSERT_TRUE(cv::imwrite(this->image_path_, scene));

        // test を短時間で終えるため回数を絞る。既定値の妥当性は別途評価する。
        this->config_.warmup_iterations_ = 2;
        this->config_.latency_iterations_ = 5;
        this->config_.throughput_frames_ = 3;
    }
    void TearDown() override { std::remove(this->image_path_.c_str()); }

    std::string image_path_;
    BenchmarkConfig config_;
};

// 正常系: CPU 経路を測定でき、統計が整合する。
TEST_F(BenchmarkHarnessTest, measures_cpu_route) {
    MeasurementRecord record;
    std::string error;
    ASSERT_TRUE(aruco3cuda::bench::measure_image(this->image_path_, this->config_, &record, &error))
            << error;
    EXPECT_EQ(record.width_px_, 640);
    EXPECT_EQ(record.height_px_, 480);
    EXPECT_EQ(record.detection_count_, 1U);
    EXPECT_FALSE(record.image_sha256_.empty());

    EXPECT_EQ(record.end_to_end_ms_.count_, 5U);
    EXPECT_LE(record.end_to_end_ms_.min_, record.end_to_end_ms_.p50_);
    EXPECT_LE(record.end_to_end_ms_.p50_, record.end_to_end_ms_.p95_);
    EXPECT_LE(record.end_to_end_ms_.p95_, record.end_to_end_ms_.p99_);
    EXPECT_LE(record.end_to_end_ms_.p99_, record.end_to_end_ms_.max_);
    EXPECT_GT(record.end_to_end_ms_.min_, 0.0);

    // CPU 経路には kernel 時間が存在しない。0 で埋めず未測定として扱う。
    EXPECT_FALSE(record.kernel_time_available_);
    EXPECT_TRUE(record.throughput_available_);
    EXPECT_GT(record.throughput_fps_, 0.0);
}

// 異常系: 未実装の経路を無言で CPU へ読み替えない。
TEST_F(BenchmarkHarnessTest, refuses_unimplemented_routes) {
    const std::vector<Route> routes = {Route::kCudaEndToEnd, Route::kCudaResident};
    for (const Route route : routes) {
        BenchmarkConfig config = this->config_;
        config.route_ = route;
        MeasurementRecord record;
        std::string error;
        EXPECT_FALSE(aruco3cuda::bench::measure_image(this->image_path_, config, &record, &error))
                << aruco3cuda::bench::to_string(route);
        EXPECT_NE(error.find("未実装"), std::string::npos);
    }
}

// 正常系: 起動の費用が経路ごとに記録される。
//
// warm-up 後の分位点には現れないため、別に記録しないと単発の検出での
// 比較ができない。CUDA 経路は文脈の生成と kernel の読み込みで、定常状態の
// 数百倍になる。
TEST_F(BenchmarkHarnessTest, records_startup_cost) {
    MeasurementRecord record;
    std::string error;
    ASSERT_TRUE(aruco3cuda::bench::measure_image(this->image_path_, this->config_, &record, &error))
            << error;
    EXPECT_GT(record.first_frame_ms_, 0.0);
    // 1 枚目までの時間は 1 枚目の検出を含むため、それ以上になる。
    EXPECT_GE(record.time_to_first_result_ms_, record.first_frame_ms_);
    // 「1 枚目は cache が冷えているため定常より遅い」は主張しない。CPU 経路の
    // 起動の費用は数 ms しかなく、機の負荷で容易に逆転する。実際 Jetson で
    // ctest -j 8 のとき 6 回中 3 回この向きが崩れた。負荷の影響を受けない
    // 定義上の関係だけを検査し、値そのものは下に表示して目視できるようにする。
    EXPECT_GT(record.end_to_end_ms_.p50_, 0.0);
    std::printf("[bench] CPU 1 枚目まで %.3f ms (検出 %.3f ms) 定常 %.3f ms\n",
                record.time_to_first_result_ms_, record.first_frame_ms_,
                record.end_to_end_ms_.p50_);
}

// 正常系: hybrid 経路を測定でき、段階時間が記録される。
TEST_F(BenchmarkHarnessTest, measures_hybrid_route) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    BenchmarkConfig config = this->config_;
    config.route_ = Route::kHybrid;
    config.memory_mode_ = aruco3cuda::bench::MemoryMode::kDevice;
    config.cuda_detector_ = aruco3cuda::bench::cuda_config_from_reference(config.detector_);
    MeasurementRecord record;
    std::string error;
    ASSERT_TRUE(aruco3cuda::bench::measure_image(this->image_path_, config, &record, &error))
            << error;
    EXPECT_GT(record.end_to_end_ms_.p50_, 0.0);
    ASSERT_TRUE(record.stage_times_available_);
    EXPECT_GT(record.gpu_stage_ms_.p50_, 0.0);
    EXPECT_GT(record.cpu_stage_ms_.p50_, 0.0);
    EXPECT_EQ(record.gpu_stage_ms_.count_, record.cpu_stage_ms_.count_);
    // 段階の標本数は遅延測定の反復数と一致する。warm-up と throughput の分が
    // 混ざっていれば、end_to_end の分位点と直接は比べられない。
    EXPECT_EQ(record.gpu_stage_ms_.count_, static_cast<std::size_t>(config.latency_iterations_));
    // 段階の中央値の和と end-to-end の中央値は比べない。中央値の和は和の
    // 中央値と一致せず、段階が互いに逆方向へ振れると大小が入れ替わる。
    // kernel 時間は CUDA event 由来のみとする。段階時間で埋めない。
    EXPECT_FALSE(record.kernel_time_available_);

    // CUDA の文脈生成と kernel 読み込みは定常状態より桁違いに大きい。
    EXPECT_GT(record.time_to_first_result_ms_, record.end_to_end_ms_.p50_);
    std::printf("[bench] Hybrid 1 枚目まで %.3f ms (検出 %.3f ms) 定常 %.3f ms\n",
                record.time_to_first_result_ms_, record.first_frame_ms_,
                record.end_to_end_ms_.p50_);
}

// 正常系: host 入力の memory 種別でも測定でき、検出結果は同じになる。
//
// 転送を測定区間へ含めた分だけ時間は長くなるはずだが、その大小をここで
// 主張しない。640x480 の転送は数十 us であり、ctest を並列実行したときの
// 測定ばらつきに埋もれる。時間の比較は benchmark の仕事であり、test の
// 仕事は両方の経路が成立することの確認である。
TEST_F(BenchmarkHarnessTest, hybrid_supports_both_memory_modes) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    BenchmarkConfig config = this->config_;
    config.route_ = Route::kHybrid;
    config.cuda_detector_ = aruco3cuda::bench::cuda_config_from_reference(config.detector_);

    MeasurementRecord resident;
    MeasurementRecord pageable;
    std::string error;
    config.memory_mode_ = aruco3cuda::bench::MemoryMode::kDevice;
    ASSERT_TRUE(aruco3cuda::bench::measure_image(this->image_path_, config, &resident, &error))
            << error;
    config.memory_mode_ = aruco3cuda::bench::MemoryMode::kHostPageable;
    ASSERT_TRUE(aruco3cuda::bench::measure_image(this->image_path_, config, &pageable, &error))
            << error;

    // 入力の置き場所を変えても検出結果は変わらない。
    EXPECT_EQ(pageable.detection_count_, resident.detection_count_);
    EXPECT_EQ(pageable.image_sha256_, resident.image_sha256_);
    EXPECT_TRUE(pageable.stage_times_available_);
    EXPECT_GT(pageable.end_to_end_ms_.p50_, 0.0);
}

// 異常系: hybrid が対応しない memory 種別は拒否する。
TEST_F(BenchmarkHarnessTest, hybrid_rejects_unsupported_memory_mode) {
    BenchmarkConfig config = this->config_;
    config.route_ = Route::kHybrid;
    config.memory_mode_ = aruco3cuda::bench::MemoryMode::kHostPinned;
    MeasurementRecord record;
    std::string error;
    EXPECT_FALSE(aruco3cuda::bench::measure_image(this->image_path_, config, &record, &error));
    EXPECT_NE(error.find("memory 種別"), std::string::npos);
}

// 正常系: CPU 経路の測定区間に画像の読み込みが入らない。
//
// 読み込みを含めると、合成 corpus の 1280x720 PNG では測定区間の 6 割以上が
// PNG の復号になる。検出時間の比較にならないため、区間から外している。
// 同じ画像を detect_image で回した場合との差でそれを確かめる。
TEST_F(BenchmarkHarnessTest, cpu_route_excludes_image_loading) {
    // 読み込みを含む経路を測る。測定の前後で 2 度取り、harness の測定を時刻で
    // 挟む。Jetson のように負荷で clock が大きく動く機では、先に取った標本と
    // 後に取った標本の速さが 2 倍ほど違う。片側だけで比べると、経路の違いでは
    // なく clock の違いを見てしまう。挟んだ両方の最小値を基準にすれば、
    // どちらへ動いても取りこぼさない。
    BenchmarkConfig config = this->config_;
    // 標本 5 個では最小値が偶然に左右される。この test だけ数を増やす。
    config.latency_iterations_ = 15;

    const auto measure_with_loading = [&]() {
        double smallest = std::numeric_limits<double>::max();
        for (int i = 0; i < config.latency_iterations_; ++i) {
            aruco3cuda::reference::ReferenceResult result;
            std::string loop_error;
            const auto start = std::chrono::steady_clock::now();
            const bool ok = aruco3cuda::reference::detect_image(this->image_path_, config.detector_,
                                                                &result, &loop_error);
            const auto finish = std::chrono::steady_clock::now();
            EXPECT_TRUE(ok) << loop_error;
            smallest = std::min(smallest,
                                std::chrono::duration<double, std::milli>(finish - start).count());
        }
        return smallest;
    };

    const double before = measure_with_loading();
    MeasurementRecord record;
    std::string error;
    ASSERT_TRUE(aruco3cuda::bench::measure_image(this->image_path_, config, &record, &error))
            << error;
    const double after = measure_with_loading();

    const double loading_min = std::min(before, after);
    std::printf("[bench] 検出のみ 最小 %.3f ms / 読み込み込み 最小 %.3f ms (前 %.3f 後 %.3f)\n",
                record.end_to_end_ms_.min_, loading_min, before, after);
    // 最小値で比べる。中央値は負荷の山を拾い、経路の差より大きく振れる。
    // PNG の復号は必ず加算されるため、同じ clock で比べる限り向きは変わらない。
    EXPECT_LT(record.end_to_end_ms_.min_, loading_min);
}

// 異常系: 不正な入力と引数を拒否する。
TEST_F(BenchmarkHarnessTest, rejects_invalid_input) {
    MeasurementRecord record;
    std::string error;
    EXPECT_FALSE(
            aruco3cuda::bench::measure_image("/nonexistent.png", this->config_, &record, &error));
    EXPECT_FALSE(error.empty());

    BenchmarkConfig zero_iterations = this->config_;
    zero_iterations.latency_iterations_ = 0;
    EXPECT_FALSE(
            aruco3cuda::bench::measure_image(this->image_path_, zero_iterations, &record, &error));
    EXPECT_FALSE(
            aruco3cuda::bench::measure_image(this->image_path_, this->config_, nullptr, &error));
    EXPECT_FALSE(
            aruco3cuda::bench::measure_image(this->image_path_, this->config_, &record, nullptr));
}

// 境界値: throughput を無効にすると未測定として記録される。
TEST_F(BenchmarkHarnessTest, throughput_can_be_disabled) {
    BenchmarkConfig config = this->config_;
    config.throughput_frames_ = 0;
    MeasurementRecord record;
    std::string error;
    ASSERT_TRUE(aruco3cuda::bench::measure_image(this->image_path_, config, &record, &error))
            << error;
    EXPECT_FALSE(record.throughput_available_);
}

// 正常系: 全標本を保存できる。
TEST_F(BenchmarkHarnessTest, can_save_all_samples) {
    BenchmarkConfig config = this->config_;
    config.save_all_samples_ = true;
    MeasurementRecord record;
    std::string error;
    ASSERT_TRUE(aruco3cuda::bench::measure_image(this->image_path_, config, &record, &error))
            << error;
    EXPECT_EQ(record.end_to_end_samples_ms_.size(), 5U);
}

// 正常系: 測定条件と縮小率が結果 JSONL へ残る。
TEST_F(BenchmarkHarnessTest, jsonl_contains_conditions_and_environment) {
    MeasurementRecord record;
    std::string error;
    ASSERT_TRUE(aruco3cuda::bench::measure_image(this->image_path_, this->config_, &record, &error))
            << error;

    std::ostringstream out;
    aruco3cuda::bench::write_environment_line(
            out, aruco3cuda::bench::collect_environment(this->config_));
    aruco3cuda::bench::write_measurement_line(out, this->config_, record);
    const std::string jsonl = out.str();

    // JSONL は 1 行 1 record。改行が 2 つある。
    std::size_t newline_count = 0;
    for (const char c : jsonl) {
        if (c == '\n') {
            ++newline_count;
        }
    }
    EXPECT_EQ(newline_count, 2U);

    EXPECT_NE(jsonl.find("\"type\":\"environment\""), std::string::npos);
    EXPECT_NE(jsonl.find("\"type\":\"measurement\""), std::string::npos);
    EXPECT_NE(jsonl.find("\"route\":\"CPU\""), std::string::npos);
    EXPECT_NE(jsonl.find("\"fxfy_effective\""), std::string::npos);
    EXPECT_NE(jsonl.find("\"warmup_iterations\""), std::string::npos);
    EXPECT_NE(jsonl.find("\"p99_ms\""), std::string::npos);
    // CPU 経路の kernel は null であり 0 ではない。
    EXPECT_NE(jsonl.find("\"kernel\":null"), std::string::npos);
    EXPECT_NE(jsonl.find("\"startup\""), std::string::npos);
    EXPECT_NE(jsonl.find("\"time_to_first_result_ms\""), std::string::npos);
    EXPECT_NE(jsonl.find(record.image_sha256_), std::string::npos);
}

// 正常系: 測定条件の再現に必要な環境情報が記録される。
// CPU の core 種別と親和性、ASLR の状態が分からないと測定値を比較できない。
TEST(BenchmarkEnvironmentTest, records_cpu_topology_and_affinity) {
    aruco3cuda::bench::BenchmarkConfig config;
    const aruco3cuda::bench::EnvironmentRecord unpinned =
            aruco3cuda::bench::collect_environment(config);
    EXPECT_FALSE(unpinned.cpu_topology_.empty());
    EXPECT_EQ(unpinned.cpu_affinity_, "unpinned");
    EXPECT_FALSE(unpinned.address_randomization_.empty());

    config.cpu_affinity_ = {0};
    const aruco3cuda::bench::EnvironmentRecord pinned =
            aruco3cuda::bench::collect_environment(config);
    EXPECT_EQ(pinned.cpu_affinity_, "0");
}

// 異常系: 範囲外の CPU 番号は固定せず、その旨を記録する。
TEST(BenchmarkEnvironmentTest, invalid_cpu_number_is_reported) {
    aruco3cuda::bench::BenchmarkConfig config;
    config.cpu_affinity_ = {-1};
    const aruco3cuda::bench::EnvironmentRecord environment =
            aruco3cuda::bench::collect_environment(config);
    EXPECT_NE(environment.cpu_affinity_.find("invalid-cpu"), std::string::npos)
            << environment.cpu_affinity_;
}

// 正常系: 経路と memory 種別の識別子が評価計画の表記と一致する。
// 集計 script が識別子で経路を区別するため、表記のずれは比較を壊す。
TEST(BenchmarkRouteTest, identifiers_match_evaluation_plan) {
    using aruco3cuda::bench::MemoryMode;
    EXPECT_STREQ(aruco3cuda::bench::to_string(Route::kCpu), "CPU");
    EXPECT_STREQ(aruco3cuda::bench::to_string(Route::kCudaEndToEnd), "CUDA-E2E");
    EXPECT_STREQ(aruco3cuda::bench::to_string(Route::kCudaResident), "CUDA-Resident");
    EXPECT_STREQ(aruco3cuda::bench::to_string(Route::kHybrid), "Hybrid");
    EXPECT_STREQ(aruco3cuda::bench::to_string(MemoryMode::kNotApplicable), "N/A");
    EXPECT_STREQ(aruco3cuda::bench::to_string(MemoryMode::kHostPageable), "M-Pageable");
    EXPECT_STREQ(aruco3cuda::bench::to_string(MemoryMode::kHostPinned), "M-Pinned");
    EXPECT_STREQ(aruco3cuda::bench::to_string(MemoryMode::kManaged), "M-Managed");
    EXPECT_STREQ(aruco3cuda::bench::to_string(MemoryMode::kDevice), "M-Device");
}

// 異常系: 列挙に無い値でも nullptr を返さない。
TEST(BenchmarkRouteTest, unknown_enum_values_are_named) {
    EXPECT_STREQ(aruco3cuda::bench::to_string(static_cast<Route>(999)), "Unknown");
    EXPECT_STREQ(aruco3cuda::bench::to_string(static_cast<aruco3cuda::bench::MemoryMode>(999)),
                 "Unknown");
}

// 正常系: kernel 時間がある場合は JSON へ統計を出力する。
// CUDA 経路が入るまで実測では通らない経路のため、record を直接組み立てて確認する。
TEST(BenchmarkOutputTest, kernel_statistics_are_written_when_available) {
    const aruco3cuda::bench::BenchmarkConfig config;
    aruco3cuda::bench::MeasurementRecord record;
    record.image_path_ = "/tmp/x.png";
    record.kernel_time_available_ = true;
    record.kernel_ms_.p50_ = 1.5;
    record.kernel_ms_.p95_ = 2.5;
    record.kernel_ms_.p99_ = 3.5;
    record.end_to_end_ms_.count_ = 1;

    std::ostringstream out;
    aruco3cuda::bench::write_measurement_line(out, config, record);
    const std::string json = out.str();
    EXPECT_EQ(json.find("\"kernel\":null"), std::string::npos) << json;
    EXPECT_NE(json.find("1.5000"), std::string::npos) << json;
    EXPECT_NE(json.find("3.5000"), std::string::npos) << json;
}

// 境界値: throughput が未測定なら null を出力する。0 で埋めない。
TEST(BenchmarkOutputTest, unavailable_throughput_is_null) {
    const aruco3cuda::bench::BenchmarkConfig config;
    aruco3cuda::bench::MeasurementRecord record;
    record.throughput_available_ = false;
    std::ostringstream out;
    aruco3cuda::bench::write_measurement_line(out, config, record);
    EXPECT_NE(out.str().find("\"throughput_fps\":null"), std::string::npos) << out.str();
}

// 境界値: clock を取得できない環境では null を出力する。
// 0 を書くと「clock が 0」と誤読されるため、未取得と 0 を区別する。
TEST(BenchmarkOutputTest, unavailable_clock_is_null) {
    aruco3cuda::bench::EnvironmentRecord environment;
    environment.hostname_ = "test-host";
    environment.gpu_clock_available_ = false;
    environment.gpu_current_clock_mhz_ = 0;

    std::ostringstream out;
    aruco3cuda::bench::write_environment_line(out, environment);
    const std::string json = out.str();
    EXPECT_NE(json.find("\"gpu_max_clock_mhz\":null"), std::string::npos) << json;
    EXPECT_NE(json.find("\"gpu_current_clock_mhz\":null"), std::string::npos) << json;
}

// 正常系: clock を取得できる環境では数値を出力する。
TEST(BenchmarkOutputTest, available_clock_is_written) {
    aruco3cuda::bench::EnvironmentRecord environment;
    environment.gpu_clock_available_ = true;
    environment.gpu_max_clock_mhz_ = 1300;
    environment.gpu_current_clock_mhz_ = 306;

    std::ostringstream out;
    aruco3cuda::bench::write_environment_line(out, environment);
    const std::string json = out.str();
    EXPECT_NE(json.find("\"gpu_max_clock_mhz\":1300"), std::string::npos) << json;
    EXPECT_NE(json.find("\"gpu_current_clock_mhz\":306"), std::string::npos) << json;
}

}  // namespace
