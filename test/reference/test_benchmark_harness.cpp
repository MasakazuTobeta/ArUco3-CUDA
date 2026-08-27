// SPDX-License-Identifier: Apache-2.0
//
// 測定 harness を検証する。
//
// 測定値そのものは環境依存で再現しないため、検証対象は次に限る。
//   - 測定条件と環境情報が結果へ確実に残ること
//   - 未実装の経路を無言で CPU へ読み替えないこと
//   - 統計の整合性 (min <= p50 <= p95 <= p99 <= max)
#include "benchmark_harness.hpp"

#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

namespace {

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
        this->image_path_ = "/tmp/aruco3cuda_bench_test.png";
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
    const std::vector<Route> routes = {Route::kCudaEndToEnd, Route::kCudaResident, Route::kHybrid};
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
    EXPECT_NE(jsonl.find(record.image_sha256_), std::string::npos);
}

// 正常系: 経路と memory 種別の識別子が評価計画の表記と一致する。
TEST(BenchmarkRouteTest, identifiers_match_evaluation_plan) {
    EXPECT_STREQ(aruco3cuda::bench::to_string(Route::kCpu), "CPU");
    EXPECT_STREQ(aruco3cuda::bench::to_string(Route::kCudaEndToEnd), "CUDA-E2E");
    EXPECT_STREQ(aruco3cuda::bench::to_string(Route::kCudaResident), "CUDA-Resident");
    EXPECT_STREQ(aruco3cuda::bench::to_string(Route::kHybrid), "Hybrid");
    EXPECT_STREQ(aruco3cuda::bench::to_string(aruco3cuda::bench::MemoryMode::kHostPinned),
                 "M-Pinned");
    EXPECT_STREQ(aruco3cuda::bench::to_string(aruco3cuda::bench::MemoryMode::kDevice), "M-Device");
}

}  // namespace
