// SPDX-License-Identifier: Apache-2.0
//
// 案 A (GPU 常駐の候補抽出) と案 C (CPU の輪郭追跡) を突き合わせる。
//
// 候補抽出の 2 方式の比較に対応する。両者は同じ二値化画像から候補を求めるが、
// 四隅の決め方が異なる。案 A は連結成分の極点、案 C は輪郭の多角形近似で
// ある。差がどこにどれだけ出るかを実測し、主案の選択を ADR へ残す材料に
// する。有利な結果だけを見ないよう、全ての場面の結果を表として出す。
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

/// 計測の暖機回数。初回は kernel の読み込みと page fault を含む。
constexpr int kWarmupIterations = 5;
/// 時間を測る場合の本測定回数。中央値を採るため奇数にする。
constexpr int kTimingIterations = 21;

bool has_cuda_device() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

/// 1 つの場面についての比較結果。
struct SceneOutcome {
    std::string name_;
    std::size_t cpu_count_ = 0;
    std::size_t gpu_count_ = 0;
    std::size_t matched_ = 0;
    /// 案 C だけが出した候補の数。
    std::size_t missed_ = 0;
    /// 案 A だけが出した候補の数。
    std::size_t extra_ = 0;
    double worst_corner_px_ = 0.0;
    std::size_t markers_ = 0;
    std::size_t markers_found_by_gpu_ = 0;
    double worst_marker_px_ = 0.0;
    double gpu_ms_ = 0.0;
    double cpu_ms_ = 0.0;
};

/// 画像を device へ載せる。
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

/// device の平面を host へ取り出す。
cv::Mat download_plane(const aruco3cuda::detail::ImagePlaneU8& plane) {
    cv::Mat result(plane.height_px_, plane.width_px_, CV_8UC1);
    const cudaError_t error =
            cudaMemcpy2D(result.data, static_cast<std::size_t>(result.step), plane.data_,
                         plane.pitch_bytes_, static_cast<std::size_t>(plane.width_px_),
                         static_cast<std::size_t>(plane.height_px_), cudaMemcpyDeviceToHost);
    EXPECT_EQ(error, cudaSuccess) << cudaGetErrorString(error);
    return result;
}

/// 2 つの四隅を、開始位置と向きの違いを許して突き合わせた最大距離。
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

/// 1 場面分の両経路を実行して比較する。
class ComparisonRun {
public:
    ComparisonRun() = default;
    ComparisonRun(const ComparisonRun&) = delete;
    ComparisonRun& operator=(const ComparisonRun&) = delete;

    /// @param measure true なら暖機して繰り返し、時間の中央値を求める。
    ///                false なら 1 回だけ実行する。正しさの検証には時間が
    ///                不要であり、繰り返すと Compute Sanitizer の下で
    ///                実行時間が現実的でなくなる。
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

        // 前処理と二値化は両経路に共通の入力を作る。比較の対象から外すため
        // 計測の外へ置く。
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

        // 案 A: window ごとに候補を求めて 1 つの配列へ連ね、まとめて統合する。
        //
        // 1 回だけの計測は初回の kernel 読み込みを含み、経路の比較に使えない。
        // 暖機してから繰り返し、中央値を採る。
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

        // 案 C: 同じ二値化画像を host へ戻し、輪郭追跡で候補を求める。
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

        // 突き合わせは四隅の距離で行う。開始する角と回り方が経路ごとに
        // 違うため、巡回と反転を許した距離を使う。開始角の違いを差として
        // 数えると、実際には一致している候補が全て不一致になる。
        std::vector<bool> gpu_used(gpu_quads.size(), false);
        std::vector<std::pair<double, std::pair<std::size_t, std::size_t>>> pairs;
        for (std::size_t i = 0; i < cpu_nodes.size(); ++i) {
            for (std::size_t j = 0; j < gpu_quads.size(); ++j) {
                const double distance = quad_distance(cpu_nodes[i].corners_, gpu_quads[j]);
                // 対応付けの半径は辺長の半分とする。別のマーカーへ誤って
                // 対応させないための上限であり、一致の判定ではない。
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

/// 合成 corpus の 1 枚を生成する。
cv::Mat make_scene(const aruco3cuda::corpusgen::SceneSpec& spec,
                   aruco3cuda::corpusgen::GeneratedScene* out) {
    aruco3cuda::corpusgen::CorpusConfig config;
    // 出力先を process ごとに分ける。この file の 2 つの test は同じ場面を
    // 生成するため、ctest の並列実行では書き込みと読み出しが競合する。
    config.output_dir_ = std::string("/tmp/aruco3cuda_compare_test_") + std::to_string(::getpid());
    config.seed_ = 20260827U;
    std::string error;
    EXPECT_TRUE(aruco3cuda::corpusgen::generate_scene(config, spec, 0, out, &error)) << error;
    return cv::imread(out->path_, cv::IMREAD_GRAYSCALE);
}

/// 比較に使う場面の一覧。
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

/// 設定 2 つ。ArUco3 有効では候補抽出が縮小画像 (長辺 1/3) で行われ、
/// 無効では原寸で行われる。画素数が 10 倍近く変わるため、どちらが有利かは
/// 規模によって変わりうる。
struct ConfigCase {
    const char* label;
    DetectorConfig config;
};

/// 全場面 x 全設定を実行し、結果を集める。
///
/// @param measure 時間を測るか。false なら 1 回だけ実行する。
/// @param out_outcomes 設定ごとの結果を追加する。
/// @return 真値のマーカー数と、案 A が見つけた数の組。
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

            // 真の四隅を segmentation 座標へ写し、案 A が覆えているかを見る。
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

        std::printf("\n=== 案 A と案 C の候補比較 (%s、segmentation 座標) ===\n",
                    config_case.label);
        // 「案 C のみ」は案 C が出して案 A が出さない候補である。多くは黒枠の
        // 内周や内部セルの輪郭であり、識別で落ちるものである。マーカーを
        // 取りこぼしたことを意味しない。真値一致の列がそれを示す。
        if (measure) {
            std::printf("%-12s %5s %5s %5s %7s %7s %8s %9s %8s %8s\n", "場面", "案C", "案A", "一致",
                        "案Cのみ", "案Aのみ", "四隅差", "真値一致", "GPU ms", "CPU ms");
        } else {
            std::printf("%-12s %5s %5s %5s %7s %7s %8s %9s\n", "場面", "案C", "案A", "一致",
                        "案Cのみ", "案Aのみ", "四隅差", "真値一致");
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
        std::printf("真値に対する検出 %zu/%zu、四隅の最大差 %.3f px\n", total_found, total_markers,
                    worst_marker);
        grand_markers += total_markers;
        grand_found += total_found;
        out->push_back(std::move(outcomes));
    }
    return {grand_markers, grand_found};
}

// 正常系: 案 A の候補が、真の四隅を許容差内で覆う。あわせて案 C との差を数える。
TEST(CandidateComparisonTest, plan_a_versus_plan_c) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    std::vector<std::vector<SceneOutcome>> outcomes;
    const auto totals = run_all(false, &outcomes);
    // 案 A が真のマーカーを取りこぼさないことを求める。案 C との候補集合の
    // 差は測定値として表に残し、判断は ADR で行う。
    EXPECT_EQ(totals.second, totals.first);
}

// 計測: 案 A と案 C の候補抽出にかかる時間を比べる。
//
// 暖機したうえで繰り返すため実行時間が長い。Compute Sanitizer からは
// 除外する。同じ経路の正しさは上の test が確かめており、繰り返しても
// 新しい実行経路は通らない。
TEST(CandidateTimingTest, plan_a_versus_plan_c) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    std::printf(
            "\n[注意] 以下の時間は単独実行かつ core 固定でのみ意味を持つ。"
            "ctest の並列実行では GPU と CPU の取り合いで値が伸びる。\n");
    std::vector<std::vector<SceneOutcome>> outcomes;
    const auto totals = run_all(true, &outcomes);
    EXPECT_EQ(totals.second, totals.first);
}

}  // namespace
