// SPDX-License-Identifier: Apache-2.0
#include "hybrid_detector.hpp"

#include <cuda_runtime_api.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <ratio>
#include <string>
#include <utility>
#include <vector>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"
#include "aruco3cuda/workspace.hpp"
#include "cpu_candidates.hpp"
#include "preprocess.hpp"
#include "threshold.hpp"

namespace aruco3cuda::hybrid {
namespace {

const std::map<std::string, int>& dictionary_table() {
    static const std::map<std::string, int> kTable = {
            {"DICT_4X4_50", cv::aruco::DICT_4X4_50},
            {"DICT_4X4_100", cv::aruco::DICT_4X4_100},
            {"DICT_4X4_250", cv::aruco::DICT_4X4_250},
            {"DICT_4X4_1000", cv::aruco::DICT_4X4_1000},
            {"DICT_5X5_50", cv::aruco::DICT_5X5_50},
            {"DICT_5X5_100", cv::aruco::DICT_5X5_100},
            {"DICT_5X5_250", cv::aruco::DICT_5X5_250},
            {"DICT_5X5_1000", cv::aruco::DICT_5X5_1000},
            {"DICT_6X6_50", cv::aruco::DICT_6X6_50},
            {"DICT_6X6_100", cv::aruco::DICT_6X6_100},
            {"DICT_6X6_250", cv::aruco::DICT_6X6_250},
            {"DICT_6X6_1000", cv::aruco::DICT_6X6_1000},
            {"DICT_7X7_50", cv::aruco::DICT_7X7_50},
            {"DICT_7X7_100", cv::aruco::DICT_7X7_100},
            {"DICT_7X7_250", cv::aruco::DICT_7X7_250},
            {"DICT_7X7_1000", cv::aruco::DICT_7X7_1000},
            {"DICT_ARUCO_ORIGINAL", cv::aruco::DICT_ARUCO_ORIGINAL},
            {"DICT_ARUCO_MIP_36h12", cv::aruco::DICT_ARUCO_MIP_36h12},
    };
    return kTable;
}

cv::aruco::DetectorParameters to_detector_parameters(const DetectorConfig& config) {
    cv::aruco::DetectorParameters params;
    params.adaptiveThreshWinSizeMin = config.adaptive_thresh_win_size_min_px_;
    params.adaptiveThreshWinSizeMax = config.adaptive_thresh_win_size_max_px_;
    params.adaptiveThreshWinSizeStep = config.adaptive_thresh_win_size_step_px_;
    params.adaptiveThreshConstant = config.adaptive_thresh_constant_;
    params.minMarkerPerimeterRate = config.min_marker_perimeter_rate_;
    params.maxMarkerPerimeterRate = config.max_marker_perimeter_rate_;
    params.polygonalApproxAccuracyRate = config.polygonal_approx_accuracy_rate_;
    params.minCornerDistanceRate = config.min_corner_distance_rate_;
    params.minDistanceToBorder = config.min_distance_to_border_px_;
    params.markerBorderBits = config.marker_border_bits_;
    params.perspectiveRemovePixelPerCell = config.perspective_remove_pixel_per_cell_;
    params.perspectiveRemoveIgnoredMarginPerCell =
            config.perspective_remove_ignored_margin_per_cell_;
    params.maxErroneousBitsInBorderRate = config.max_erroneous_bits_in_border_rate_;
    params.minOtsuStdDev = config.min_otsu_std_dev_;
    params.errorCorrectionRate = config.error_correction_rate_;
    params.useAruco3Detection = config.use_aruco3_detection_;
    params.minSideLengthCanonicalImg = config.min_side_length_canonical_img_px_;
    params.minMarkerLengthRatioOriginalImg = config.min_marker_length_ratio_original_img_;
    params.minMarkerDistanceRate = config.min_marker_distance_rate_;
    params.minGroupDistance = static_cast<float>(config.min_group_distance_);
    params.cornerRefinementWinSize = config.corner_refinement_win_size_px_;
    params.relativeCornerRefinmentWinSize =
            static_cast<float>(config.relative_corner_refinement_win_size_);
    params.cornerRefinementMaxIterations = config.corner_refinement_max_iterations_;
    params.cornerRefinementMinAccuracy = config.corner_refinement_min_accuracy_px_;
    return params;
}


/// 候補の周長から、識別に使う pyramid level を選ぶ。
///
/// OpenCV の _findOptPyrImageForCanonicalImg と同じ。距離が正のものだけを
/// 対象とし、より大きい level を優先する。
std::size_t find_optimal_level(const std::vector<cv::Mat>& pyramid, int scaled_width,
                               int contour_size, int min_perimeter) {
    std::size_t optimal = 0;
    float distance = std::numeric_limits<float>::max();
    for (std::size_t i = 0; i < pyramid.size(); ++i) {
        const float scale = static_cast<float>(pyramid[i].cols) / static_cast<float>(scaled_width);
        const float scaled_perimeter = static_cast<float>(contour_size) * scale;
        const float new_distance = scaled_perimeter - static_cast<float>(min_perimeter);
        if (new_distance < distance && new_distance > 0.0F) {
            distance = new_distance;
            optimal = i;
        }
    }
    return optimal;
}

/// セルごとの白画素比を求める。OpenCV の _extractCellPixelRatio と同じ。
cv::Mat extract_cell_pixel_ratio(const cv::Mat& image, const std::vector<cv::Point2f>& corners,
                                 int marker_size, int border_bits, int cell_size,
                                 double cell_margin_rate, double min_std_dev) {
    const int size_with_borders = marker_size + 2 * border_bits;
    const int cell_margin = static_cast<int>(cell_margin_rate * cell_size);
    const int result_size = size_with_borders * cell_size;

    std::vector<cv::Point2f> destination(4);
    destination[0] = cv::Point2f(0.0F, 0.0F);
    destination[1] = cv::Point2f(static_cast<float>(result_size) - 1.0F, 0.0F);
    destination[2] = cv::Point2f(static_cast<float>(result_size) - 1.0F,
                                 static_cast<float>(result_size) - 1.0F);
    destination[3] = cv::Point2f(0.0F, static_cast<float>(result_size) - 1.0F);

    const cv::Mat transformation = cv::getPerspectiveTransform(corners, destination);
    cv::Mat canonical;
    cv::warpPerspective(image, canonical, transformation, cv::Size(result_size, result_size),
                        cv::INTER_NEAREST);

    cv::Mat ratio(size_with_borders, size_with_borders, CV_32FC1, cv::Scalar::all(0));
    cv::Mat mean;
    cv::Mat stddev;
    const cv::Mat inner = canonical.colRange(cell_size / 2, canonical.cols - cell_size / 2)
                                  .rowRange(cell_size / 2, canonical.rows - cell_size / 2);
    cv::meanStdDev(inner, mean, stddev);
    if (stddev.ptr<double>(0)[0] < min_std_dev) {
        // 分散が小さい候補は全て白か全て黒とみなす。Otsu が意味を持たない。
        ratio.setTo(mean.ptr<double>(0)[0] > 127.0 ? 1.0 : 0.0);
        return ratio;
    }
    cv::threshold(canonical, canonical, 125, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    for (int y = 0; y < size_with_borders; ++y) {
        for (int x = 0; x < size_with_borders; ++x) {
            const cv::Rect cell(x * cell_size + cell_margin, y * cell_size + cell_margin,
                                cell_size - 2 * cell_margin, cell_size - 2 * cell_margin);
            const cv::Mat square = canonical(cell);
            ratio.at<float>(y, x) = static_cast<float>(cv::countNonZero(square)) /
                                    static_cast<float>(square.total());
        }
    }
    return ratio;
}

/// 外周セルの誤り数を数える。OpenCV の _getBorderErrors と同じ。
int count_border_errors(const cv::Mat& ratio, int marker_size, int border_bits,
                        float valid_bit_threshold) {
    const int size_with_borders = marker_size + 2 * border_bits;
    int errors = 0;
    for (int y = 0; y < size_with_borders; ++y) {
        const auto* row = ratio.ptr<float>(y);
        for (int k = 0; k < border_bits; ++k) {
            if (row[k] > valid_bit_threshold) {
                ++errors;
            }
            if (row[size_with_borders - 1 - k] > valid_bit_threshold) {
                ++errors;
            }
        }
    }
    for (int x = border_bits; x < size_with_borders - border_bits; ++x) {
        for (int k = 0; k < border_bits; ++k) {
            if (ratio.ptr<float>(k)[x] > valid_bit_threshold) {
                ++errors;
            }
            if (ratio.ptr<float>(size_with_borders - 1 - k)[x] > valid_bit_threshold) {
                ++errors;
            }
        }
    }
    return errors;
}


/// host 側の受け取りに必要な pinned memory の量を求める。
///
/// 二値化画像は window ごとに segmentation の大きさ、pyramid は level ごとに
/// 半分ずつ小さくなる。level 0 は原寸であり、ここが最大を占める。
/// 桁溢れや設定不正では 0 を返す。
std::size_t host_receive_bytes(const DetectorConfig& config, const detail::ScalePlan& plan,
                               int max_width_px, int max_height_px) {
    if (max_width_px < 1 || max_height_px < 1) {
        return 0U;
    }
    int window_sizes[kMaxAdaptiveThresholdWindows] = {};
    int window_count = 0;
    if (detail::threshold_window_sizes(config, window_sizes, kMaxAdaptiveThresholdWindows,
                                       &window_count) != Status::kOk) {
        return 0U;
    }
    // 行ごとに整列させるため、1 平面あたり少し余裕を積む。
    constexpr std::size_t kRowAlignment = 512U;
    const auto plane_bytes = [](int width, int height) {
        return align_up(static_cast<std::size_t>(width), kRowAlignment) *
               static_cast<std::size_t>(height);
    };
    std::size_t total = plane_bytes(plan.segmentation_width_px_, plan.segmentation_height_px_) *
                        static_cast<std::size_t>(window_count);
    int level_width = max_width_px;
    int level_height = max_height_px;
    for (int level = 0; level < plan.level_count_; ++level) {
        total += plane_bytes(level_width, level_height);
        level_width = std::max(1, (level_width + 1) / 2);
        level_height = std::max(1, (level_height + 1) / 2);
    }
    // 端数の整列で不足しないよう 1 平面分の余裕を持たせる。
    return total + plane_bytes(max_width_px, 1);
}

/// pinned arena から 1 平面を切り出し、cv::Mat として参照する。
///
/// cv::Mat は外部 memory を指すだけで所有しない。arena を reset すると
/// 参照先が無効になるため、frame の途中で reset しないこと。
Status reserve_host_plane(Workspace& workspace, int width_px, int height_px, cv::Mat* out) {
    constexpr std::size_t kRowAlignment = 512U;
    const std::size_t step = align_up(static_cast<std::size_t>(width_px), kRowAlignment);
    void* pointer = nullptr;
    const Status status =
            workspace.allocate(step * static_cast<std::size_t>(height_px), kRowAlignment,
                               &pointer);
    if (status != Status::kOk) {
        return status;
    }
    *out = cv::Mat(height_px, width_px, CV_8UC1, pointer, step);
    return Status::kOk;
}

}  // namespace

/// 実装本体。公開 header が OpenCV へ依存しないよう pimpl とする。
class HybridDetector::Impl {
public:
    Impl() = default;
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
    ~Impl() {
        if (this->stream_ != nullptr) {
            static_cast<void>(cudaStreamDestroy(this->stream_));
        }
    }

    Status initialize(const DetectorConfig& config, const std::string& dictionary_name,
                      int max_width_px, int max_height_px, std::string* out_message);
    Status detect(const ImageViewU8& image, HybridResult* out, std::string* out_message);
    const WorkspaceStatistics& workspace_statistics() const {
        return this->workspace_.statistics();
    }

private:
    Status run_gpu_stages(const ImageViewU8& image, std::string* out_message);
    void run_cpu_stages(const ImageViewU8& image, HybridResult* out);

    bool initialized_ = false;
    DetectorConfig config_;
    cv::aruco::Dictionary dictionary_;
    cv::aruco::DetectorParameters parameters_;
    Workspace workspace_;
    /// host 側の受け取り先。pinned memory を arena として確保する。
    ///
    /// 非同期転送は pinned memory を要求する。pageable memory を指定すると
    /// driver が内部で一時 buffer へ複製するため、同期転送と変わらない。
    Workspace host_workspace_;
    /// 転送と kernel を載せる stream。
    ///
    /// 既定 stream を使うと、同一 process 内の他の作業と直列化する。
    /// また転送を 1 つずつ blocking すると、8 回分の呼び出し費用が積み上がる。
    cudaStream_t stream_ = nullptr;
    detail::ScalePlan plan_;
    detail::PreprocessBuffers preprocess_;
    detail::ThresholdBuffers threshold_;
    std::vector<cv::Mat> binary_images_;
    std::vector<cv::Mat> pyramid_;
};

Status HybridDetector::Impl::initialize(const DetectorConfig& config,
                                        const std::string& dictionary_name, int max_width_px,
                                        int max_height_px, std::string* out_message) {
    const Status config_status = config.validate(out_message);
    if (config_status != Status::kOk) {
        return config_status;
    }
    if (max_width_px < 1 || max_height_px < 1) {
        if (out_message != nullptr) {
            *out_message = "最大解像度は 1 以上である必要がある";
        }
        return Status::kInvalidArgument;
    }
    const auto entry = dictionary_table().find(dictionary_name);
    if (entry == dictionary_table().end()) {
        if (out_message != nullptr) {
            *out_message = "未対応の Dictionary: " + dictionary_name;
        }
        return Status::kUnsupportedDictionary;
    }

    this->config_ = config;
    this->dictionary_ = cv::aruco::getPredefinedDictionary(entry->second);
    this->parameters_ = to_detector_parameters(config);

    // 最大解像度から必要な容量を求め、初期化時に一括で確保する。
    detail::ScalePlan plan;
    const Status plan_status = detail::plan_scales(config, max_width_px, max_height_px, &plan);
    if (plan_status != Status::kOk) {
        return plan_status;
    }
    const std::size_t preprocess_bytes =
            detail::preprocess_workspace_bytes(plan, max_width_px, max_height_px);
    const std::size_t threshold_bytes = detail::threshold_workspace_bytes(
            config, plan.segmentation_width_px_, plan.segmentation_height_px_);
    if (preprocess_bytes == 0U || threshold_bytes == 0U) {
        if (out_message != nullptr) {
            *out_message = "workspace の必要量を算出できない";
        }
        return Status::kInvalidConfig;
    }
    const Status capacity_status = this->workspace_.ensure_capacity(
            preprocess_bytes + threshold_bytes, MemorySpace::kDevice, out_message);
    if (capacity_status != Status::kOk) {
        return capacity_status;
    }

    // host 側の受け取り先も初期化時に確保する。frame ごとの確保を避けるため。
    const std::size_t host_bytes = host_receive_bytes(config, plan, max_width_px, max_height_px);
    if (host_bytes == 0U) {
        if (out_message != nullptr) {
            *out_message = "host 側 buffer の必要量を算出できない";
        }
        return Status::kInvalidConfig;
    }
    const Status host_status =
            this->host_workspace_.ensure_capacity(host_bytes, MemorySpace::kHostPinned,
                                                  out_message);
    if (host_status != Status::kOk) {
        return host_status;
    }

    if (this->stream_ == nullptr) {
        const cudaError_t created = cudaStreamCreate(&this->stream_);
        if (created != cudaSuccess) {
            if (out_message != nullptr) {
                *out_message = std::string("stream を作れない: ") + cudaGetErrorString(created);
            }
            return Status::kCudaError;
        }
    }
    this->initialized_ = true;
    return Status::kOk;
}

Status HybridDetector::Impl::run_gpu_stages(const ImageViewU8& image, std::string* out_message) {
    const Status plan_status =
            detail::plan_scales(this->config_, image.width_px_, image.height_px_, &this->plan_);
    if (plan_status != Status::kOk) {
        return plan_status;
    }
    // フレームの先頭で切り出し位置を戻す。容量は保持されるため確保は起きない。
    this->workspace_.reset();

    const Status reserve_status = detail::reserve_preprocess(this->plan_, image, this->workspace_,
                                                             &this->preprocess_);
    if (reserve_status != Status::kOk) {
        if (out_message != nullptr) {
            *out_message = "前処理の領域を確保できない。初期化時の最大解像度を超えている";
        }
        return reserve_status;
    }
    const Status threshold_reserve = detail::reserve_threshold(
            this->config_, this->plan_.segmentation_width_px_, this->plan_.segmentation_height_px_,
            this->workspace_, &this->threshold_);
    if (threshold_reserve != Status::kOk) {
        if (out_message != nullptr) {
            *out_message = "二値化の領域を確保できない。初期化時の最大解像度を超えている";
        }
        return threshold_reserve;
    }

    Status status = detail::build_pyramid_async(&this->preprocess_, this->config_, this->stream_);
    if (status != Status::kOk) {
        return status;
    }
    status = detail::build_segmentation_async(this->plan_, &this->preprocess_, this->config_,
                                              this->stream_);
    if (status != Status::kOk) {
        return status;
    }
    const ImageViewU8 segmentation{this->preprocess_.segmentation_.data_,
                                   this->preprocess_.segmentation_.width_px_,
                                   this->preprocess_.segmentation_.height_px_,
                                   this->preprocess_.segmentation_.pitch_bytes_, image.space_};
    status = detail::build_threshold_async(segmentation, &this->threshold_, this->config_,
                                           this->stream_);
    if (status != Status::kOk) {
        return status;
    }

    // 転送を stream へ積んでから 1 度だけ同期する。1 つずつ blocking すると
    // 8 回分の呼び出し費用が積み上がり、DGX Spark では複製そのものの 3 倍以上に
    // なる。受け取り先は pinned memory であり、非同期転送が成立する。
    this->host_workspace_.reset();
    this->binary_images_.resize(static_cast<std::size_t>(this->threshold_.window_count_));
    for (int i = 0; i < this->threshold_.window_count_; ++i) {
        const detail::ImagePlaneU8& plane = this->threshold_.binary_[i];
        cv::Mat& destination = this->binary_images_[static_cast<std::size_t>(i)];
        const Status reserved = reserve_host_plane(this->host_workspace_, plane.width_px_,
                                                   plane.height_px_, &destination);
        if (reserved != Status::kOk) {
            if (out_message != nullptr) {
                *out_message = "二値化画像の受け取り先を確保できない";
            }
            return reserved;
        }
        if (cudaMemcpy2DAsync(destination.data, static_cast<std::size_t>(destination.step),
                              plane.data_, plane.pitch_bytes_,
                              static_cast<std::size_t>(plane.width_px_),
                              static_cast<std::size_t>(plane.height_px_),
                              cudaMemcpyDeviceToHost, this->stream_) != cudaSuccess) {
            if (out_message != nullptr) {
                *out_message = "二値化画像を host へ戻せない";
            }
            return Status::kCudaError;
        }
    }

    this->pyramid_.resize(static_cast<std::size_t>(this->preprocess_.level_count_));
    for (int level = 0; level < this->preprocess_.level_count_; ++level) {
        const ImageViewU8 view = detail::level_view(this->preprocess_, level);
        cv::Mat& destination = this->pyramid_[static_cast<std::size_t>(level)];
        const Status reserved = reserve_host_plane(this->host_workspace_, view.width_px_,
                                                   view.height_px_, &destination);
        if (reserved != Status::kOk) {
            if (out_message != nullptr) {
                *out_message = "pyramid の受け取り先を確保できない";
            }
            return reserved;
        }
        if (cudaMemcpy2DAsync(destination.data, static_cast<std::size_t>(destination.step),
                              view.data_, view.pitch_bytes_,
                              static_cast<std::size_t>(view.width_px_),
                              static_cast<std::size_t>(view.height_px_),
                              cudaMemcpyDeviceToHost, this->stream_) != cudaSuccess) {
            if (out_message != nullptr) {
                *out_message = "pyramid を host へ戻せない";
            }
            return Status::kCudaError;
        }
    }

    // kernel と転送の完了をここで 1 度だけ待つ。
    const cudaError_t sync = cudaStreamSynchronize(this->stream_);
    if (sync != cudaSuccess) {
        if (out_message != nullptr) {
            *out_message = std::string("GPU の同期に失敗した: ") + cudaGetErrorString(sync);
        }
        return Status::kCudaError;
    }
    return Status::kOk;
}

void HybridDetector::Impl::run_cpu_stages(const ImageViewU8& image, HybridResult* out) {
    const int segmentation_width = this->plan_.segmentation_width_px_;
    const int segmentation_height = this->plan_.segmentation_height_px_;
    const int min_perimeter = this->config_.min_side_length_canonical_img_px_ * 4;
    const int cell_size = this->config_.perspective_remove_pixel_per_cell_;
    const int marker_size = this->dictionary_.markerSize;
    const int border_bits = this->config_.marker_border_bits_;
    const int max_border_errors = static_cast<int>(
            marker_size * marker_size * this->config_.max_erroneous_bits_in_border_rate_);

    std::vector<std::vector<cv::Point2f>> candidates;
    std::vector<std::vector<cv::Point>> contours;
    for (const cv::Mat& binary : this->binary_images_) {
        find_quad_candidates(binary, this->config_, candidates, contours);
    }
    out->candidate_count_ = candidates.size();
    for (std::vector<cv::Point2f>& candidate : candidates) {
        reorder_corners(candidate);
    }

    // 二値化 window ごとに同じマーカーが複数回検出されるため、近接候補を
    // まとめて 1 つに絞る。ここで最大周長の候補を選ばないと四隅が内側へ
    // 寄り、原寸へ戻したときに数 px の誤差になる。
    std::vector<CandidateNode> nodes = filter_too_close_candidates(
            cv::Size(segmentation_width, segmentation_height), candidates, contours,
            this->config_, marker_size);

    // 1 つの候補を識別する。成功したら id と rotation を書き込む。
    const auto identify_one = [&](const std::vector<cv::Point2f>& corners, const cv::Mat& level,
                                  float scale, int* out_id, int* out_rotation) {
        std::vector<cv::Point2f> scaled(4);
        for (int c = 0; c < 4; ++c) {
            scaled[static_cast<std::size_t>(c)] = corners[static_cast<std::size_t>(c)] * scale;
        }
        const cv::Mat ratio = extract_cell_pixel_ratio(
                level, scaled, marker_size, border_bits, cell_size,
                this->config_.perspective_remove_ignored_margin_per_cell_,
                this->config_.min_otsu_std_dev_);
        if (count_border_errors(ratio, marker_size, border_bits,
                                this->parameters_.validBitIdThreshold) > max_border_errors) {
            return false;
        }
        const cv::Mat inner =
                ratio(cv::Rect(border_bits, border_bits, ratio.cols - (2 * border_bits),
                               ratio.rows - (2 * border_bits)));
        return this->dictionary_.identify(inner, *out_id, *out_rotation,
                                          this->config_.error_correction_rate_,
                                          this->parameters_.validBitIdThreshold);
    };

    const std::size_t node_count = nodes.size();
    std::vector<int> ids(node_count, -1);
    std::vector<int> rotations(node_count, 0);
    std::vector<std::uint8_t> valid(node_count, 0U);
    std::vector<std::uint8_t> visited(node_count, 0U);

    int max_depth = 0;
    for (const CandidateNode& node : nodes) {
        max_depth = std::max(max_depth, node.depth_);
    }
    std::vector<std::vector<std::size_t>> by_depth(static_cast<std::size_t>(max_depth) + 1U);
    for (std::size_t i = 0; i < node_count; ++i) {
        by_depth[static_cast<std::size_t>(nodes[i].depth_)].push_back(i);
    }

    // 内側の候補から識別し、マーカーが確定したらその外側 (親) を数え上げ
    // 済みとして扱う。マーカーの黒枠は外側の輪郭も候補になるため、内側が
    // 当たった時点で外側を識別する必要がない。
    std::size_t counter = 0;
    std::size_t depth = 0;
    while (counter < node_count && depth < by_depth.size()) {
        for (const std::size_t v : by_depth[depth]) {
            visited[v] = 1U;
            std::size_t level = 0;
            float scale = 1.0F;
            if (this->config_.use_aruco3_detection_) {
                level = find_optimal_level(this->pyramid_, segmentation_width,
                                           static_cast<int>(nodes[v].contour_.size()),
                                           min_perimeter);
                scale = static_cast<float>(this->pyramid_[level].cols) /
                        static_cast<float>(segmentation_width);
            }
            const cv::Mat& level_image = this->pyramid_[level];
            if (identify_one(nodes[v].corners_, level_image, scale, &ids[v], &rotations[v])) {
                valid[v] = 1U;
                continue;
            }
            // 代表で失敗した場合、同じ group の離れた候補を試す。
            for (const MarkerCandidate& close : nodes[v].close_contours_) {
                if (identify_one(close.corners_, level_image, scale, &ids[v], &rotations[v])) {
                    nodes[v].corners_ = close.corners_;
                    nodes[v].contour_ = close.contour_;
                    valid[v] = 1U;
                    break;
                }
            }
        }
        for (const std::size_t v : by_depth[depth]) {
            if (valid[v] != 0U) {
                int parent = nodes[v].parent_;
                while (parent != -1) {
                    const auto index = static_cast<std::size_t>(parent);
                    if (visited[index] == 0U) {
                        visited[index] = 1U;
                        ++counter;
                    }
                    parent = nodes[index].parent_;
                }
            }
            ++counter;
        }
        ++depth;
    }

    // ArUco3 は縮小画像で輪郭を取るため、原寸へ戻すには subpixel 補正が要る。
    // OpenCV も useAruco3Detection が有効なら補正方法の指定を上書きする。
    const bool use_subpix = this->config_.use_aruco3_detection_ ||
                            this->config_.corner_refine_method_ == CornerRefineMethod::kSubpix;
    for (std::size_t v = 0; v < node_count; ++v) {
        if (valid[v] == 0U) {
            continue;
        }
        std::vector<cv::Point2f> corners = nodes[v].corners_;
        // Dictionary 照合で得た回転を打ち消す。OpenCV の correctCornerPosition と同じ。
        std::rotate(corners.begin(),
                    corners.begin() + (4 - static_cast<std::ptrdiff_t>(rotations[v])),
                    corners.end());

        // 四隅を原寸座標へ戻す。ArUco3 は pyramid を 1 段ずつ上げながら
        // 各 level で subpixel 補正する。
        if (this->config_.use_aruco3_detection_) {
            const int closest = this->plan_.closest_level_index_;
            const float initial_scale =
                    static_cast<float>(
                            this->pyramid_[static_cast<std::size_t>(closest)].cols) /
                    static_cast<float>(segmentation_width);
            cv::Mat corner_matrix(corners);
            if (initial_scale != 1.0F) {
                corner_matrix *= static_cast<double>(initial_scale);
            }
            for (int idx = closest - 1; idx >= 0; --idx) {
                corner_matrix *= 2.0;
                const cv::Mat& level_image = this->pyramid_[static_cast<std::size_t>(idx)];
                const int window = std::max(level_image.cols, level_image.rows) > 1080 ? 5 : 3;
                cv::cornerSubPix(level_image, corner_matrix, cv::Size(window, window),
                                 cv::Size(-1, -1),
                                 cv::TermCriteria(cv::TermCriteria::MAX_ITER |
                                                          cv::TermCriteria::EPS,
                                                  this->config_.corner_refinement_max_iterations_,
                                                  this->config_.corner_refinement_min_accuracy_px_));
            }
        } else if (use_subpix) {
            // ArUco3 無効時は原寸画像で 1 度だけ補正する。window はセル 1 辺
            // から決め、上限で頭打ちにする。OpenCV の非 ArUco3 経路と同じ。
            const float module_size =
                    average_module_size(corners, marker_size, border_bits);
            int window = std::max(
                    1, cvRound(static_cast<float>(
                                       this->config_.relative_corner_refinement_win_size_) *
                               module_size));
            window = std::min(window, this->config_.corner_refinement_win_size_px_);
            cv::Mat corner_matrix(corners);
            cv::cornerSubPix(this->pyramid_[0], corner_matrix, cv::Size(window, window),
                             cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
                                              this->config_.corner_refinement_max_iterations_,
                                              this->config_.corner_refinement_min_accuracy_px_));
        }
        // ArUco3 が無効なら縮小しないため、原寸へ戻す拡大は不要である。

        HybridDetection detection;
        detection.id_ = ids[v];
        detection.rotation_ = rotations[v];
        for (int c = 0; c < 4; ++c) {
            const std::size_t index = static_cast<std::size_t>(c);
            detection.corners_[index * 2U] = static_cast<double>(corners[index].x);
            detection.corners_[(index * 2U) + 1U] = static_cast<double>(corners[index].y);
        }
        out->detections_.push_back(detection);
    }

    // 結果の順序を入力順序に依存させないため、ID と位置で整列する。
    std::sort(out->detections_.begin(), out->detections_.end(),
              [](const HybridDetection& a, const HybridDetection& b) {
                  if (a.id_ != b.id_) {
                      return a.id_ < b.id_;
                  }
                  if (a.corners_[0] != b.corners_[0]) {
                      return a.corners_[0] < b.corners_[0];
                  }
                  return a.corners_[1] < b.corners_[1];
              });
    (void)image;
}

Status HybridDetector::Impl::detect(const ImageViewU8& image, HybridResult* out,
                                    std::string* out_message) {
    if (out == nullptr) {
        return Status::kInvalidArgument;
    }
    if (!this->initialized_) {
        if (out_message != nullptr) {
            *out_message = "initialize() が呼ばれていない";
        }
        return Status::kNotInitialized;
    }
    const Status image_status = validate_image_view(image, out_message);
    if (image_status != Status::kOk) {
        return image_status;
    }

    HybridResult result;
    const auto gpu_start = std::chrono::steady_clock::now();
    const Status gpu_status = this->run_gpu_stages(image, out_message);
    const auto gpu_finish = std::chrono::steady_clock::now();
    if (gpu_status != Status::kOk) {
        return gpu_status;
    }
    result.gpu_ms_ = std::chrono::duration<double, std::milli>(gpu_finish - gpu_start).count();

    const auto cpu_start = std::chrono::steady_clock::now();
    this->run_cpu_stages(image, &result);
    const auto cpu_finish = std::chrono::steady_clock::now();
    result.cpu_ms_ = std::chrono::duration<double, std::milli>(cpu_finish - cpu_start).count();

    *out = std::move(result);
    return Status::kOk;
}

HybridDetector::HybridDetector() : impl_(std::make_unique<Impl>()) {}
HybridDetector::~HybridDetector() = default;
HybridDetector::HybridDetector(HybridDetector&&) noexcept = default;
HybridDetector& HybridDetector::operator=(HybridDetector&&) noexcept = default;

Status HybridDetector::initialize(const DetectorConfig& config, const std::string& dictionary_name,
                                  int max_width_px, int max_height_px, std::string* out_message) {
    return this->impl_->initialize(config, dictionary_name, max_width_px, max_height_px,
                                   out_message);
}

Status HybridDetector::detect(const ImageViewU8& image, HybridResult* out,
                              std::string* out_message) {
    return this->impl_->detect(image, out, out_message);
}

const WorkspaceStatistics& HybridDetector::workspace_statistics() const {
    return this->impl_->workspace_statistics();
}

}  // namespace aruco3cuda::hybrid
