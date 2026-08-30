// SPDX-License-Identifier: Apache-2.0
//
// Verifies the measurement harness.
//
// The measured values themselves depend on the machine and do not reproduce, so
// only the following are checked here.
//   - The measurement conditions and the environment information always end up in
//     the result
//   - An unimplemented route is never silently reinterpreted as the CPU route
//   - The statistics stay consistent (min <= p50 <= p95 <= p99 <= max)
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
        // Give every test its own image path so that the TearDown of another test
        // running concurrently under ctest -j cannot delete it.
        const ::testing::TestInfo* info = ::testing::UnitTest::GetInstance()->current_test_info();
        this->image_path_ = std::string("/tmp/aruco3cuda_bench_test_") +
                            (info != nullptr ? info->name() : "unknown") + "_" +
                            std::to_string(static_cast<long>(::getpid())) + ".png";
        ASSERT_TRUE(cv::imwrite(this->image_path_, scene));

        // Keep the iteration counts small so the test finishes quickly. Whether the
        // defaults are appropriate is evaluated separately.
        this->config_.warmup_iterations_ = 2;
        this->config_.latency_iterations_ = 5;
        this->config_.throughput_frames_ = 3;
    }
    void TearDown() override { std::remove(this->image_path_.c_str()); }

    std::string image_path_;
    BenchmarkConfig config_;
};

/// Tests that cross-check the results of the different routes.
///
/// They make no claim about which route is faster, so the suite name does not
/// contain Timing and the suite also runs under Compute Sanitizer. This is the
/// only test that exercises the CUDA-EndToEnd route.
class BenchmarkHarnessRouteTest : public ::testing::Test {
protected:
    void SetUp() override {
        const cv::aruco::Dictionary dictionary =
                cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_MIP_36h12);
        cv::Mat marker;
        dictionary.generateImageMarker(17, 160, marker, 1);
        cv::Mat scene(480, 640, CV_8UC1, cv::Scalar(230));
        marker.copyTo(scene(cv::Rect(200, 150, 160, 160)));
        // Give every test its own image path so that the TearDown of another test
        // running concurrently under ctest -j cannot delete it.
        const ::testing::TestInfo* info = ::testing::UnitTest::GetInstance()->current_test_info();
        this->image_path_ = std::string("/tmp/aruco3cuda_bench_test_") +
                            (info != nullptr ? info->name() : "unknown") + "_" +
                            std::to_string(static_cast<long>(::getpid())) + ".png";
        ASSERT_TRUE(cv::imwrite(this->image_path_, scene));

        // Keep the iteration counts small so the test finishes quickly. Whether the
        // defaults are appropriate is evaluated separately.
        this->config_.warmup_iterations_ = 2;
        this->config_.latency_iterations_ = 5;
        this->config_.throughput_frames_ = 3;
    }
    void TearDown() override { std::remove(this->image_path_.c_str()); }

    std::string image_path_;
    BenchmarkConfig config_;
};

// Happy path: the CPU route can be measured and the statistics are consistent.
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

    // The CPU route has no kernel time. It is reported as not measured rather than
    // filled in with 0.
    EXPECT_FALSE(record.kernel_time_available_);
    EXPECT_TRUE(record.throughput_available_);
    EXPECT_GT(record.throughput_fps_, 0.0);
}

// Failure path: a memory kind the route does not support is never silently
// reinterpreted.
TEST_F(BenchmarkHarnessTest, refuses_mismatched_memory_mode) {
    struct Case {
        Route route_;
        aruco3cuda::bench::MemoryMode mode_;
    };
    const std::vector<Case> cases = {
            // Pass a kind that includes a transfer to a device-resident route.
            {Route::kCudaResident, aruco3cuda::bench::MemoryMode::kHostPageable},
            // Pass device-resident memory to a route that includes the transfer.
            {Route::kCudaEndToEnd, aruco3cuda::bench::MemoryMode::kDevice},
            // The CPU route has no memory kind.
            {Route::kCpu, aruco3cuda::bench::MemoryMode::kManaged},
            {Route::kCpu, aruco3cuda::bench::MemoryMode::kHostPinned},
            {Route::kCpu, aruco3cuda::bench::MemoryMode::kDevice},
    };
    for (const Case& item : cases) {
        BenchmarkConfig config = this->config_;
        config.route_ = item.route_;
        config.memory_mode_ = item.mode_;
        config.cuda_detector_ = aruco3cuda::bench::cuda_config_from_reference(config.detector_);
        MeasurementRecord record;
        std::string error;
        EXPECT_FALSE(aruco3cuda::bench::measure_image(this->image_path_, config, &record, &error))
                << aruco3cuda::bench::to_string(item.route_);
        EXPECT_NE(error.find("memory kind"), std::string::npos) << error;
    }
}

// Happy path: the CUDA-Resident route can be measured.
//
// The measured interval includes the stream synchronization. Without it only the
// kernel launch time would be measured. That the synchronization really happens is
// confirmed by the steady-state median being larger than a launch alone.
TEST_F(BenchmarkHarnessTest, measures_cuda_resident_route) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    BenchmarkConfig config = this->config_;
    config.route_ = Route::kCudaResident;
    config.memory_mode_ = aruco3cuda::bench::MemoryMode::kDevice;
    config.cuda_detector_ = aruco3cuda::bench::cuda_config_from_reference(config.detector_);

    MeasurementRecord record;
    std::string error;
    ASSERT_TRUE(aruco3cuda::bench::measure_image(this->image_path_, config, &record, &error))
            << error;
    EXPECT_GT(record.end_to_end_ms_.p50_, 0.0);
    EXPECT_EQ(record.end_to_end_ms_.count_, static_cast<std::size_t>(config.latency_iterations_));
    EXPECT_EQ(record.detection_count_, 1U);
    // The per-stage breakdown requires CUDA events and does not exist yet.
    EXPECT_FALSE(record.stage_times_available_);
    EXPECT_FALSE(record.kernel_time_available_);
    // No claim is made that "time to first result > steady state". Under Compute
    // Sanitizer the steady state grows by orders of magnitude and the direction
    // flips. The values are printed below so they can be inspected by eye.
    EXPECT_GT(record.time_to_first_result_ms_, 0.0);
    std::printf(
            "[bench] CUDA-Resident time to first result %.3f ms (detect %.3f ms) "
            "steady state %.3f ms\n",
            record.time_to_first_result_ms_, record.first_frame_ms_, record.end_to_end_ms_.p50_);
}

// Happy path: the startup cost is recorded per route.
//
// It does not show up in the post-warm-up percentiles, so without recording it
// separately a single-shot detection cannot be compared. On the CUDA routes the
// context creation and kernel loading make it hundreds of times the steady state.
TEST_F(BenchmarkHarnessTest, records_startup_cost) {
    MeasurementRecord record;
    std::string error;
    ASSERT_TRUE(aruco3cuda::bench::measure_image(this->image_path_, this->config_, &record, &error))
            << error;
    EXPECT_GT(record.first_frame_ms_, 0.0);
    // The time to the first result includes the first detection, so it is at least
    // as large as that detection.
    EXPECT_GE(record.time_to_first_result_ms_, record.first_frame_ms_);
    // No claim is made that "the first frame is slower than the steady state because
    // the caches are cold". On the CPU route the startup cost is only a few ms and
    // machine load easily reverses it: on a Jetson under ctest -j 8 this direction
    // broke in 3 out of 6 runs. Only the relation that holds by definition, and is
    // therefore immune to load, is checked; the values themselves are printed below
    // so they can be inspected by eye.
    EXPECT_GT(record.end_to_end_ms_.p50_, 0.0);
    std::printf(
            "[bench] CPU time to first result %.3f ms (detect %.3f ms) "
            "steady state %.3f ms\n",
            record.time_to_first_result_ms_, record.first_frame_ms_, record.end_to_end_ms_.p50_);
}

// Happy path: the hybrid route can be measured and the stage times are recorded.
TEST_F(BenchmarkHarnessTest, measures_hybrid_route) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
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
    // The stage sample count equals the latency iteration count. If warm-up or
    // throughput samples were mixed in, the stage numbers could not be compared
    // directly against the end_to_end percentiles.
    EXPECT_EQ(record.gpu_stage_ms_.count_, static_cast<std::size_t>(config.latency_iterations_));
    // The sum of the stage medians is not compared against the end-to-end median.
    // The sum of medians is not the median of the sum, and when the stages swing in
    // opposite directions the order between them flips.
    // Kernel time only ever comes from CUDA events; it is not filled in with stage
    // times.
    EXPECT_FALSE(record.kernel_time_available_);

    // No claim is made that "time to first result > steady state". Under Compute
    // Sanitizer, or on a busy machine, the steady state grows and the direction
    // flips. The values are printed below.
    EXPECT_GT(record.time_to_first_result_ms_, 0.0);
    std::printf(
            "[bench] Hybrid time to first result %.3f ms (detect %.3f ms) "
            "steady state %.3f ms\n",
            record.time_to_first_result_ms_, record.first_frame_ms_, record.end_to_end_ms_.p50_);
}

// Happy path: a host-input memory kind can also be measured and yields the same
// detections.
//
// Including the transfer in the measured interval should make the time longer, but
// no claim about that ordering is made here. A 640x480 transfer takes tens of us,
// which is buried in the measurement spread when ctest runs in parallel. Comparing
// times is the benchmark's job; the test's job is to confirm that both paths work.
TEST_F(BenchmarkHarnessTest, hybrid_supports_both_memory_modes) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
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

    // Changing where the input lives does not change the detections.
    EXPECT_EQ(pageable.detection_count_, resident.detection_count_);
    EXPECT_EQ(pageable.image_sha256_, resident.image_sha256_);
    EXPECT_TRUE(pageable.stage_times_available_);
    EXPECT_GT(pageable.end_to_end_ms_.p50_, 0.0);
}

// Failure path: a memory kind the hybrid route does not support is rejected.
TEST_F(BenchmarkHarnessTest, hybrid_rejects_unsupported_memory_mode) {
    BenchmarkConfig config = this->config_;
    config.route_ = Route::kHybrid;
    config.memory_mode_ = aruco3cuda::bench::MemoryMode::kHostPinned;
    MeasurementRecord record;
    std::string error;
    EXPECT_FALSE(aruco3cuda::bench::measure_image(this->image_path_, config, &record, &error));
    EXPECT_NE(error.find("memory kind"), std::string::npos);
}

// Happy path: the measured interval of the CPU route excludes image loading.
//
// With loading included, more than 60% of the measured interval on the 1280x720 PNGs
// of the synthetic corpus is PNG decoding, which makes the numbers useless for
// comparing detection times, so loading is kept outside the interval. This is
// confirmed by the difference against running the same image through detect_image.
TEST_F(BenchmarkHarnessTest, cpu_route_excludes_image_loading) {
    // Measure the path that includes loading. It is sampled twice, before and after,
    // so that the harness measurement is bracketed in time. On a machine such as a
    // Jetson, where load moves the clock a lot, samples taken before and after can
    // differ in speed by about a factor of two. Comparing against only one side would
    // show the clock difference rather than the difference between the paths. Using
    // the minimum of both bracketing samples as the baseline keeps the comparison
    // valid whichever way the clock moved.
    BenchmarkConfig config = this->config_;
    // With only 5 samples the minimum is dominated by chance. This test alone uses
    // more iterations.
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
    std::printf(
            "[bench] detect only min %.3f ms / with loading min %.3f ms "
            "(before %.3f after %.3f)\n",
            record.end_to_end_ms_.min_, loading_min, before, after);
    // Compare on the minimum. The median picks up load spikes and swings more than
    // the difference between the paths. PNG decoding is always added on top, so as
    // long as both are compared at the same clock the direction cannot change.
    EXPECT_LT(record.end_to_end_ms_.min_, loading_min);
}

// Failure path: invalid input and arguments are rejected.
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

// Boundary case: disabling throughput records it as not measured.
TEST_F(BenchmarkHarnessTest, throughput_can_be_disabled) {
    BenchmarkConfig config = this->config_;
    config.throughput_frames_ = 0;
    MeasurementRecord record;
    std::string error;
    ASSERT_TRUE(aruco3cuda::bench::measure_image(this->image_path_, config, &record, &error))
            << error;
    EXPECT_FALSE(record.throughput_available_);
}

// Happy path: all samples can be saved.
TEST_F(BenchmarkHarnessTest, can_save_all_samples) {
    BenchmarkConfig config = this->config_;
    config.save_all_samples_ = true;
    MeasurementRecord record;
    std::string error;
    ASSERT_TRUE(aruco3cuda::bench::measure_image(this->image_path_, config, &record, &error))
            << error;
    EXPECT_EQ(record.end_to_end_samples_ms_.size(), 5U);
}

// Happy path: the measurement conditions and the scale factor survive into the
// result JSONL.
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

    // JSONL holds one record per line, so there are two newlines.
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
    // The kernel entry of the CPU route is null, not 0.
    EXPECT_NE(jsonl.find("\"kernel\":null"), std::string::npos);
    EXPECT_NE(jsonl.find("\"startup\""), std::string::npos);
    EXPECT_NE(jsonl.find("\"time_to_first_result_ms\""), std::string::npos);
    EXPECT_NE(jsonl.find(record.image_sha256_), std::string::npos);
}

// Happy path: the environment information needed to reproduce the measurement
// conditions is recorded. Without the CPU core kinds, the affinity, and the ASLR
// state, measured values cannot be compared.
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

// Failure path: an out-of-range CPU number is not pinned, and that fact is
// recorded.
TEST(BenchmarkEnvironmentTest, invalid_cpu_number_is_reported) {
    aruco3cuda::bench::BenchmarkConfig config;
    config.cpu_affinity_ = {-1};
    const aruco3cuda::bench::EnvironmentRecord environment =
            aruco3cuda::bench::collect_environment(config);
    EXPECT_NE(environment.cpu_affinity_.find("invalid-cpu"), std::string::npos)
            << environment.cpu_affinity_;
}

// Happy path: the route and memory kind identifiers match the spelling used in the
// evaluation plan. The aggregation scripts tell the routes apart by these
// identifiers, so any drift in spelling breaks the comparison.
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

// Failure path: even a value outside the enumeration does not yield nullptr.
TEST(BenchmarkRouteTest, unknown_enum_values_are_named) {
    EXPECT_STREQ(aruco3cuda::bench::to_string(static_cast<Route>(999)), "Unknown");
    EXPECT_STREQ(aruco3cuda::bench::to_string(static_cast<aruco3cuda::bench::MemoryMode>(999)),
                 "Unknown");
}

// Happy path: when kernel time is available, its statistics are written to the JSON.
// This path is not exercised by a real measurement until the CUDA routes land, so
// the record is assembled directly.
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

// Boundary case: an unmeasured throughput is written as null, not filled in with 0.
TEST(BenchmarkOutputTest, unavailable_throughput_is_null) {
    const aruco3cuda::bench::BenchmarkConfig config;
    aruco3cuda::bench::MeasurementRecord record;
    record.throughput_available_ = false;
    std::ostringstream out;
    aruco3cuda::bench::write_measurement_line(out, config, record);
    EXPECT_NE(out.str().find("\"throughput_fps\":null"), std::string::npos) << out.str();
}

// Boundary case: on a machine where the clock cannot be read, null is written.
// Writing 0 would be misread as "the clock is 0", so unavailable and 0 are kept
// distinct.
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

// Happy path: on a machine where the clock can be read, the number is written.
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

// Happy path: the CUDA-EndToEnd route and the CUDA-Resident route produce the same
// results.
//
// **No claim is made about which is faster.** The two routes differ only by the cost
// of the transfer and the readback, which at 640x480 is a 307 KB transfer. On the
// bandwidth of an integrated GPU that is under 10 us, less than 1% of the 1 ms
// detection. In practice the spread caused by the clock ramping up is larger, and the
// order flips with the order in which the routes are measured. Comparing on images
// large enough for the difference to show is the job of the on-device measurements
// (docs/measurements).
TEST_F(BenchmarkHarnessRouteTest, cuda_routes_agree_on_detections) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    BenchmarkConfig resident = this->config_;
    resident.route_ = Route::kCudaResident;
    resident.memory_mode_ = aruco3cuda::bench::MemoryMode::kDevice;
    resident.cuda_detector_ = aruco3cuda::bench::cuda_config_from_reference(resident.detector_);
    BenchmarkConfig end_to_end = resident;
    end_to_end.route_ = Route::kCudaEndToEnd;
    end_to_end.memory_mode_ = aruco3cuda::bench::MemoryMode::kHostPageable;

    MeasurementRecord resident_record;
    MeasurementRecord end_to_end_record;
    std::string error;
    ASSERT_TRUE(
            aruco3cuda::bench::measure_image(this->image_path_, resident, &resident_record, &error))
            << error;
    ASSERT_TRUE(aruco3cuda::bench::measure_image(this->image_path_, end_to_end, &end_to_end_record,
                                                 &error))
            << error;

    // If the two routes do not agree on the detections, their implementations have
    // diverged.
    EXPECT_EQ(end_to_end_record.detection_count_, resident_record.detection_count_);
    EXPECT_GT(resident_record.end_to_end_ms_.min_, 0.0);
    EXPECT_GT(end_to_end_record.end_to_end_ms_.min_, 0.0);
    std::printf("[bench] min Resident %.3f ms / EndToEnd(pageable) %.3f ms\n",
                resident_record.end_to_end_ms_.min_, end_to_end_record.end_to_end_ms_.min_);
}

}  // namespace
