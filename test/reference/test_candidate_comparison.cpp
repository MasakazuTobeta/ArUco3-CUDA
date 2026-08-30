// SPDX-License-Identifier: Apache-2.0
//
// Cross-checks plan A (device-resident candidate extraction) against plan C (contour
// tracing on the CPU).
//
// This covers the comparison of the two candidate extraction approaches. Both derive
// candidates from the same binary image, but they determine the four corners
// differently: plan A uses the extreme points of the connected component, plan C the
// polygon approximation of the contour. These tests measure where the differences
// appear and how large they are, as material for the ADR that records which approach
// is chosen. So that no one looks only at the favorable results, every scene is
// printed as a table.
#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <unistd.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <ratio>
#include <string>
#include <utility>
#include <vector>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"
#include "aruco3cuda/workspace.hpp"
#include "candidate_filter.hpp"
#include "candidate_group.hpp"
#include "corpus_generator.hpp"
#include "cpu_candidates.hpp"
#include "labeling.hpp"
#include "preprocess.hpp"
#include "quad_extract.hpp"
#include "threshold.hpp"

namespace {

using aruco3cuda::DetectorConfig;
using aruco3cuda::ImageViewU8;
using aruco3cuda::MemorySpace;
using aruco3cuda::Status;
using aruco3cuda::Workspace;
using aruco3cuda::detail::kQuadCornerCount;

/// Warm-up iterations before a measurement. The first run includes kernel loading and
/// page faults.
constexpr int kWarmupIterations = 5;
/// Measured iterations when timing. Odd so that a median can be taken.
constexpr int kTimingIterations = 21;

bool has_cuda_device() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

/// Comparison result for a single scene.
struct SceneOutcome {
    std::string name_;
    std::size_t cpu_count_ = 0;
    std::size_t gpu_count_ = 0;
    std::size_t matched_ = 0;
    /// Number of candidates produced by plan C only.
    std::size_t missed_ = 0;
    /// Number of candidates produced by plan A only.
    std::size_t extra_ = 0;
    double worst_corner_px_ = 0.0;
    std::size_t markers_ = 0;
    std::size_t markers_found_by_gpu_ = 0;
    double worst_marker_px_ = 0.0;
    double gpu_ms_ = 0.0;
    double cpu_ms_ = 0.0;
};

/// Uploads an image to the device.
class DeviceImage {
public:
    DeviceImage() = default;
    DeviceImage(const DeviceImage&) = delete;
    DeviceImage& operator=(const DeviceImage&) = delete;
    ~DeviceImage() {
        if (this->data_ != nullptr) {
            static_cast<void>(cudaFree(this->data_));
        }
    }

    bool upload(const cv::Mat& image) {
        this->pitch_bytes_ = static_cast<std::size_t>(image.cols) + 32U;
        if (cudaMalloc(&this->data_, this->pitch_bytes_ * static_cast<std::size_t>(image.rows)) !=
            cudaSuccess) {
            return false;
        }
        if (cudaMemcpy2D(this->data_, this->pitch_bytes_, image.data,
                         static_cast<std::size_t>(image.step), static_cast<std::size_t>(image.cols),
                         static_cast<std::size_t>(image.rows),
                         cudaMemcpyHostToDevice) != cudaSuccess) {
            return false;
        }
        this->view_.data_ = static_cast<const std::uint8_t*>(this->data_);
        this->view_.width_px_ = image.cols;
        this->view_.height_px_ = image.rows;
        this->view_.pitch_bytes_ = this->pitch_bytes_;
        this->view_.space_ = MemorySpace::kDevice;
        return true;
    }

    const ImageViewU8& view() const { return this->view_; }

private:
    void* data_ = nullptr;
    std::size_t pitch_bytes_ = 0;
    ImageViewU8 view_;
};

/// Reads a device plane back to the host.
cv::Mat download_plane(const aruco3cuda::detail::ImagePlaneU8& plane) {
    cv::Mat result(plane.height_px_, plane.width_px_, CV_8UC1);
    const cudaError_t error =
            cudaMemcpy2D(result.data, static_cast<std::size_t>(result.step), plane.data_,
                         plane.pitch_bytes_, static_cast<std::size_t>(plane.width_px_),
                         static_cast<std::size_t>(plane.height_px_), cudaMemcpyDeviceToHost);
    EXPECT_EQ(error, cudaSuccess) << cudaGetErrorString(error);
    return result;
}

/// Largest corner distance between two quads, allowing for a different starting
/// corner and winding direction.
double quad_distance(const std::vector<cv::Point2f>& expected,
                     const std::vector<cv::Point2f>& actual) {
    double best = 1e18;
    for (int reversed = 0; reversed < 2; ++reversed) {
        std::vector<cv::Point2f> candidate = actual;
        if (reversed == 1) {
            std::reverse(candidate.begin(), candidate.end());
        }
        for (int shift = 0; shift < kQuadCornerCount; ++shift) {
            double worst = 0.0;
            for (int c = 0; c < kQuadCornerCount; ++c) {
                const auto index = static_cast<std::size_t>((c + shift) % kQuadCornerCount);
                const cv::Point2f delta = expected[static_cast<std::size_t>(c)] - candidate[index];
                worst = std::max(worst, std::sqrt(static_cast<double>((delta.x * delta.x) +
                                                                      (delta.y * delta.y))));
            }
            best = std::min(best, worst);
        }
    }
    return best;
}

/// Runs both paths on one scene and compares them.
class ComparisonRun {
public:
    ComparisonRun() = default;
    ComparisonRun(const ComparisonRun&) = delete;
    ComparisonRun& operator=(const ComparisonRun&) = delete;

    /// @param measure true warms up, repeats, and takes the median time. false runs
    ///                once only: verifying correctness does not need timings, and
    ///                repeating makes the run time impractical under Compute
    ///                Sanitizer.
    bool run(const cv::Mat& image, const DetectorConfig& config, bool measure, SceneOutcome* out) {
        const int warmup = measure ? kWarmupIterations : 0;
        const int iterations = measure ? kTimingIterations : 1;
        if (!this->device_.upload(image)) {
            return false;
        }
        aruco3cuda::detail::ScalePlan plan;
        if (aruco3cuda::detail::plan_scales(config, image.cols, image.rows, &plan) != Status::kOk) {
            return false;
        }
        const int width = plan.segmentation_width_px_;
        const int height = plan.segmentation_height_px_;

        const std::size_t bytes =
                aruco3cuda::detail::preprocess_workspace_bytes(plan, image.cols, image.rows) +
                aruco3cuda::detail::threshold_workspace_bytes(config, width, height) +
                aruco3cuda::detail::labeling_workspace_bytes(width, height) +
                aruco3cuda::detail::label_stats_workspace_bytes(width, height) +
                aruco3cuda::detail::quad_workspace_bytes(width, height) +
                aruco3cuda::detail::candidate_workspace_bytes(config, width, height) +
                aruco3cuda::detail::candidate_group_workspace_bytes(config);
        if (bytes == 0U ||
            this->workspace_.ensure_capacity(bytes, MemorySpace::kDevice, nullptr) != Status::kOk) {
            return false;
        }
        this->workspace_.reset();

        aruco3cuda::detail::PreprocessBuffers preprocess;
        aruco3cuda::detail::ThresholdBuffers threshold;
        aruco3cuda::detail::LabelBuffers labels;
        aruco3cuda::detail::LabelStatisticsBuffers stats;
        aruco3cuda::detail::QuadBuffers quads;
        aruco3cuda::detail::CandidateFilterBuffers filter;
        aruco3cuda::detail::DeviceCandidates candidates;
        aruco3cuda::detail::CandidateGroupBuffers groups;
        aruco3cuda::detail::DeviceCandidates grouped;
        if (aruco3cuda::detail::reserve_preprocess(plan, this->device_.view(), this->workspace_,
                                                   &preprocess) != Status::kOk ||
            aruco3cuda::detail::reserve_threshold(config, width, height, this->workspace_,
                                                  &threshold) != Status::kOk ||
            aruco3cuda::detail::reserve_labeling(width, height, this->workspace_, &labels) !=
                    Status::kOk ||
            aruco3cuda::detail::reserve_label_stats(width, height, this->workspace_, &stats) !=
                    Status::kOk ||
            aruco3cuda::detail::reserve_quads(width, height, this->workspace_, &quads) !=
                    Status::kOk ||
            aruco3cuda::detail::reserve_candidates(config, width, height, this->workspace_, &filter,
                                                   &candidates) != Status::kOk ||
            aruco3cuda::detail::reserve_candidate_groups(config, this->workspace_, &groups,
                                                         &grouped) != Status::kOk) {
            return false;
        }

        // Preprocessing and binarization produce the input shared by both paths. They
        // are not part of the comparison, so they stay outside the measured region.
        if (aruco3cuda::detail::build_pyramid_async(&preprocess, config, nullptr) != Status::kOk ||
            aruco3cuda::detail::build_segmentation_async(plan, &preprocess, config, nullptr) !=
                    Status::kOk) {
            return false;
        }
        const ImageViewU8 segmentation{preprocess.segmentation_.data_,
                                       preprocess.segmentation_.width_px_,
                                       preprocess.segmentation_.height_px_,
                                       preprocess.segmentation_.pitch_bytes_, MemorySpace::kDevice};
        if (aruco3cuda::detail::build_threshold_async(segmentation, &threshold, config, nullptr) !=
                    Status::kOk ||
            cudaDeviceSynchronize() != cudaSuccess) {
            return false;
        }

        // Plan A: derive candidates per window, append them into one array, and merge
        // them all at once.
        //
        // A single timed run includes the first kernel load and cannot be used to
        // compare the paths, so it warms up, repeats, and takes the median.
        std::vector<double> gpu_samples;
        gpu_samples.reserve(static_cast<std::size_t>(iterations));
        int gpu_count = 0;
        for (int iteration = 0; iteration < warmup + iterations; ++iteration) {
            const auto gpu_start = std::chrono::steady_clock::now();
            for (int window = 0; window < threshold.window_count_; ++window) {
                if (aruco3cuda::detail::build_labels_async(threshold.binary_[window], &labels,
                                                           nullptr) != Status::kOk ||
                    aruco3cuda::detail::build_label_stats_async(labels, &stats, nullptr) !=
                            Status::kOk ||
                    aruco3cuda::detail::build_quads_async(labels, stats, &quads, nullptr) !=
                            Status::kOk ||
                    aruco3cuda::detail::build_candidates_async(labels, stats, quads, config,
                                                               &filter, &candidates, window != 0,
                                                               nullptr) != Status::kOk) {
                    return false;
                }
            }
            if (aruco3cuda::detail::build_candidate_groups_async(
                        candidates, config, &groups, &grouped, nullptr) != Status::kOk) {
                return false;
            }
            const Status gpu_status =
                    aruco3cuda::detail::read_candidate_count(grouped, &gpu_count, nullptr);
            if (gpu_status != Status::kOk && gpu_status != Status::kCandidateOverflow) {
                return false;
            }
            const auto gpu_end = std::chrono::steady_clock::now();
            if (iteration >= kWarmupIterations) {
                gpu_samples.push_back(
                        std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count());
            }
        }
        std::sort(gpu_samples.begin(), gpu_samples.end());
        out->gpu_ms_ = gpu_samples[gpu_samples.size() / 2];

        std::vector<std::vector<cv::Point2f>> gpu_quads;
        if (!download_candidates(grouped, gpu_count, &gpu_quads)) {
            return false;
        }

        // Plan C: bring the same binary images back to the host and derive candidates
        // by contour tracing.
        std::vector<cv::Mat> binaries;
        binaries.reserve(static_cast<std::size_t>(threshold.window_count_));
        for (int window = 0; window < threshold.window_count_; ++window) {
            binaries.push_back(download_plane(threshold.binary_[window]));
        }
        std::vector<double> cpu_samples;
        cpu_samples.reserve(static_cast<std::size_t>(iterations));
        std::vector<aruco3cuda::hybrid::CandidateNode> cpu_nodes;
        for (int iteration = 0; iteration < warmup + iterations; ++iteration) {
            const auto cpu_start = std::chrono::steady_clock::now();
            std::vector<std::vector<cv::Point2f>> cpu_candidates;
            std::vector<std::vector<cv::Point>> cpu_contours;
            for (const cv::Mat& binary : binaries) {
                aruco3cuda::hybrid::find_quad_candidates(binary, config, cpu_candidates,
                                                         cpu_contours);
            }
            for (std::vector<cv::Point2f>& candidate : cpu_candidates) {
                aruco3cuda::hybrid::reorder_corners(candidate);
            }
            cpu_nodes = aruco3cuda::hybrid::filter_too_close_candidates(
                    cv::Size(width, height), cpu_candidates, cpu_contours, config, 6);
            const auto cpu_end = std::chrono::steady_clock::now();
            if (iteration >= warmup) {
                cpu_samples.push_back(
                        std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count());
            }
        }
        std::sort(cpu_samples.begin(), cpu_samples.end());
        out->cpu_ms_ = cpu_samples[cpu_samples.size() / 2];

        // Pairing is done on the corner distance. The starting corner and the winding
        // direction differ between the paths, so the distance allows for a cyclic
        // shift and a reversal. Counting a different starting corner as a difference
        // would turn every candidate that actually agrees into a mismatch.
        std::vector<bool> gpu_used(gpu_quads.size(), false);
        std::vector<std::pair<double, std::pair<std::size_t, std::size_t>>> pairs;
        for (std::size_t i = 0; i < cpu_nodes.size(); ++i) {
            for (std::size_t j = 0; j < gpu_quads.size(); ++j) {
                const double distance = quad_distance(cpu_nodes[i].corners_, gpu_quads[j]);
                // The pairing radius is half the edge length. It is an upper bound
                // that keeps a candidate from being paired with a different marker,
                // not a test for agreement.
                const double radius = static_cast<double>(cpu_nodes[i].perimeter_) / 8.0;
                if (distance <= radius) {
                    pairs.emplace_back(distance, std::make_pair(i, j));
                }
            }
        }
        std::sort(pairs.begin(), pairs.end());
        std::vector<bool> cpu_used(cpu_nodes.size(), false);
        for (const auto& pair : pairs) {
            const std::size_t i = pair.second.first;
            const std::size_t j = pair.second.second;
            if (cpu_used[i] || gpu_used[j]) {
                continue;
            }
            cpu_used[i] = true;
            gpu_used[j] = true;
            ++out->matched_;
            out->worst_corner_px_ = std::max(out->worst_corner_px_, pair.first);
        }
        out->cpu_count_ = cpu_nodes.size();
        out->gpu_count_ = gpu_quads.size();
        out->missed_ =
                static_cast<std::size_t>(std::count(cpu_used.begin(), cpu_used.end(), false));
        out->extra_ = static_cast<std::size_t>(std::count(gpu_used.begin(), gpu_used.end(), false));
        this->gpu_quads_ = std::move(gpu_quads);
        return true;
    }

    const std::vector<std::vector<cv::Point2f>>& gpu_quads() const { return this->gpu_quads_; }

private:
    static bool download_candidates(const aruco3cuda::detail::DeviceCandidates& candidates,
                                    int count, std::vector<std::vector<cv::Point2f>>* out) {
        out->assign(static_cast<std::size_t>(count), {});
        if (count <= 0) {
            return true;
        }
        const auto size = static_cast<std::size_t>(count);
        const auto capacity = static_cast<std::size_t>(candidates.capacity_);
        std::vector<std::int32_t> corner_x(size * kQuadCornerCount);
        std::vector<std::int32_t> corner_y(size * kQuadCornerCount);
        const std::size_t row_bytes = size * sizeof(std::int32_t);
        for (int corner = 0; corner < kQuadCornerCount; ++corner) {
            const std::size_t offset = static_cast<std::size_t>(corner) * capacity;
            const std::size_t destination = static_cast<std::size_t>(corner) * size;
            if (cudaMemcpy(corner_x.data() + destination, candidates.corner_x_ + offset, row_bytes,
                           cudaMemcpyDeviceToHost) != cudaSuccess ||
                cudaMemcpy(corner_y.data() + destination, candidates.corner_y_ + offset, row_bytes,
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
                return false;
            }
        }
        for (std::size_t i = 0; i < size; ++i) {
            (*out)[i].resize(kQuadCornerCount);
            for (int corner = 0; corner < kQuadCornerCount; ++corner) {
                const std::size_t index = (static_cast<std::size_t>(corner) * size) + i;
                (*out)[i][static_cast<std::size_t>(corner)] = cv::Point2f(
                        static_cast<float>(corner_x[index]), static_cast<float>(corner_y[index]));
            }
        }
        return true;
    }

    DeviceImage device_;
    Workspace workspace_;
    std::vector<std::vector<cv::Point2f>> gpu_quads_;
};

/// Generates one image of the synthetic corpus.
cv::Mat make_scene(const aruco3cuda::corpusgen::SceneSpec& spec,
                   aruco3cuda::corpusgen::GeneratedScene* out) {
    aruco3cuda::corpusgen::CorpusConfig config;
    // Give every process its own output directory. The two tests in this file
    // generate the same scenes, so under a parallel ctest run their writes and reads
    // would race.
    config.output_dir_ = std::string("/tmp/aruco3cuda_compare_test_") + std::to_string(::getpid());
    config.seed_ = 20260827U;
    std::string error;
    EXPECT_TRUE(aruco3cuda::corpusgen::generate_scene(config, spec, 0, out, &error)) << error;
    return cv::imread(out->path_, cv::IMREAD_GRAYSCALE);
}

/// The list of scenes used for the comparison.
std::vector<aruco3cuda::corpusgen::SceneSpec> comparison_cases() {
    std::vector<aruco3cuda::corpusgen::SceneSpec> cases;
    {
        aruco3cuda::corpusgen::SceneSpec base;
        base.width_px_ = 1280;
        base.height_px_ = 720;
        base.marker_count_ = 4;
        base.marker_side_px_ = 128.0;

        aruco3cuda::corpusgen::SceneSpec spec = base;
        spec.name_ = "cmp_n1";
        spec.marker_count_ = 1;
        spec.marker_side_px_ = 160.0;
        cases.push_back(spec);

        spec = base;
        spec.name_ = "cmp_n4";
        cases.push_back(spec);

        spec = base;
        spec.name_ = "cmp_n9";
        spec.marker_count_ = 9;
        cases.push_back(spec);

        spec = base;
        spec.name_ = "cmp_rot";
        spec.rotation_deg_ = 23.0;
        cases.push_back(spec);

        spec = base;
        spec.name_ = "cmp_persp";
        spec.perspective_strength_ = 0.5;
        cases.push_back(spec);

        spec = base;
        spec.name_ = "cmp_blur";
        spec.blur_sigma_px_ = 1.2;
        cases.push_back(spec);

        spec = base;
        spec.name_ = "cmp_noise";
        spec.noise_sigma_levels_ = 6.0;
        cases.push_back(spec);

        spec = base;
        spec.name_ = "cmp_illum";
        spec.illumination_strength_ = 0.6;
        cases.push_back(spec);

        spec = base;
        spec.name_ = "cmp_1080p";
        spec.width_px_ = 1920;
        spec.height_px_ = 1080;
        spec.marker_side_px_ = 160.0;
        cases.push_back(spec);
    }

    return cases;
}

/// Two configurations. With ArUco3 enabled, candidate extraction runs on the reduced
/// image (long side 1/3); with it disabled, on the full-size image. The pixel count
/// changes by almost a factor of ten, so which approach wins can depend on the scale.
struct ConfigCase {
    const char* label;
    DetectorConfig config;
};

/// Runs every scene x every configuration and collects the results.
///
/// @param measure whether to time the runs. false runs each scene once only.
/// @param out_outcomes the per-configuration results are appended here.
/// @return the pair of the ground truth marker count and the count plan A found.
std::pair<std::size_t, std::size_t> run_all(bool measure,
                                            std::vector<std::vector<SceneOutcome>>* out) {
    const ConfigCase config_cases[] = {{"aruco3", DetectorConfig{}},
                                       {"plain", DetectorConfig::opencv_defaults()}};
    const std::vector<aruco3cuda::corpusgen::SceneSpec> cases = comparison_cases();

    std::size_t grand_markers = 0;
    std::size_t grand_found = 0;
    for (const ConfigCase& config_case : config_cases) {
        const DetectorConfig& config = config_case.config;
        std::vector<SceneOutcome> outcomes;
        for (const auto& spec : cases) {
            aruco3cuda::corpusgen::GeneratedScene scene;
            const cv::Mat image = make_scene(spec, &scene);
            if (image.empty()) {
                ADD_FAILURE() << spec.name_;
                return {0U, 1U};
            }

            SceneOutcome outcome;
            outcome.name_ = spec.name_;
            ComparisonRun run;
            if (!run.run(image, config, measure, &outcome)) {
                ADD_FAILURE() << spec.name_;
                return {0U, 1U};
            }

            // Map the true corners into segmentation coordinates and check whether
            // plan A covers them.
            aruco3cuda::detail::ScalePlan plan;
            EXPECT_EQ(aruco3cuda::detail::plan_scales(config, image.cols, image.rows, &plan),
                      Status::kOk);
            const auto scale = static_cast<float>(plan.fxfy_);
            outcome.markers_ = scene.markers_.size();
            for (const auto& marker : scene.markers_) {
                std::vector<cv::Point2f> truth(kQuadCornerCount);
                for (std::size_t c = 0; c < static_cast<std::size_t>(kQuadCornerCount); ++c) {
                    truth[c] =
                            cv::Point2f(static_cast<float>(marker.corners_[c * 2U]) * scale,
                                        static_cast<float>(marker.corners_[(c * 2U) + 1U]) * scale);
                }
                double best = 1e18;
                for (const auto& quad : run.gpu_quads()) {
                    best = std::min(best, quad_distance(truth, quad));
                }
                if (best <= 2.5) {
                    ++outcome.markers_found_by_gpu_;
                    outcome.worst_marker_px_ = std::max(outcome.worst_marker_px_, best);
                }
            }
            outcomes.push_back(outcome);
        }

        std::printf("\n=== plan A versus plan C candidates (%s, segmentation coordinates) ===\n",
                    config_case.label);
        // "C only" counts candidates plan C produced and plan A did not. Most of them
        // are the inner edge of the black border or the contours of the inner cells,
        // which identification would reject anyway. They do not mean a marker was
        // missed; the truth column shows that.
        if (measure) {
            std::printf("%-12s %5s %5s %5s %7s %7s %8s %9s %8s %8s\n", "scene", "planC", "planA",
                        "match", "C only", "A only", "corner", "truth", "GPU ms", "CPU ms");
        } else {
            std::printf("%-12s %5s %5s %5s %7s %7s %8s %9s\n", "scene", "planC", "planA", "match",
                        "C only", "A only", "corner", "truth");
        }
        std::size_t total_markers = 0;
        std::size_t total_found = 0;
        double worst_marker = 0.0;
        for (const SceneOutcome& outcome : outcomes) {
            if (measure) {
                std::printf("%-12s %5zu %5zu %5zu %7zu %7zu %8.3f %4zu/%-4zu %8.3f %8.3f\n",
                            outcome.name_.c_str(), outcome.cpu_count_, outcome.gpu_count_,
                            outcome.matched_, outcome.missed_, outcome.extra_,
                            outcome.worst_corner_px_, outcome.markers_found_by_gpu_,
                            outcome.markers_, outcome.gpu_ms_, outcome.cpu_ms_);
            } else {
                std::printf("%-12s %5zu %5zu %5zu %7zu %7zu %8.3f %4zu/%-4zu\n",
                            outcome.name_.c_str(), outcome.cpu_count_, outcome.gpu_count_,
                            outcome.matched_, outcome.missed_, outcome.extra_,
                            outcome.worst_corner_px_, outcome.markers_found_by_gpu_,
                            outcome.markers_);
            }
            total_markers += outcome.markers_;
            total_found += outcome.markers_found_by_gpu_;
            worst_marker = std::max(worst_marker, outcome.worst_marker_px_);
        }
        std::printf("found against truth %zu/%zu, largest corner difference %.3f px\n", total_found,
                    total_markers, worst_marker);
        grand_markers += total_markers;
        grand_found += total_found;
        out->push_back(std::move(outcomes));
    }
    return {grand_markers, grand_found};
}

// Happy path: the plan A candidates cover the true corners within tolerance. The
// differences from plan C are counted alongside.
TEST(CandidateComparisonTest, plan_a_versus_plan_c) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    std::vector<std::vector<SceneOutcome>> outcomes;
    const auto totals = run_all(false, &outcomes);
    // Require that plan A misses no true marker. The difference between the candidate
    // sets of the two plans is left in the table as a measurement; the decision is
    // made in the ADR.
    EXPECT_EQ(totals.second, totals.first);
}

// Measurement: compares how long candidate extraction takes for plan A and plan C.
//
// It warms up and repeats, so it runs long, and it is excluded from Compute Sanitizer.
// The correctness of the same paths is covered by the test above, and repeating them
// exercises no new code path.
TEST(CandidateTimingTest, plan_a_versus_plan_c) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    std::printf(
            "\n[note] the times below are only meaningful when run alone with a pinned "
            "core. Under a parallel ctest run they grow because the GPU and the CPU are "
            "contended.\n");
    std::vector<std::vector<SceneOutcome>> outcomes;
    const auto totals = run_all(true, &outcomes);
    EXPECT_EQ(totals.second, totals.first);
}

}  // namespace
