// SPDX-License-Identifier: Apache-2.0
#include "aruco3cuda/detector.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/detections.hpp"
#include "aruco3cuda/device_probe.hpp"
#include "aruco3cuda/dictionary.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"
#include "aruco3cuda/workspace.hpp"
#include "candidate_filter.hpp"
#include "candidate_group.hpp"
#include "candidate_tree.hpp"
#include "cell_decode.hpp"
#include "cell_sample.hpp"
#include "corner_refine.hpp"
#include "cuda_check.hpp"
#include "detection_emit.hpp"
#include "dictionary_match.hpp"
#include "labeling.hpp"
#include "preprocess.hpp"
#include "quad_extract.hpp"
#include "threshold.hpp"

namespace aruco3cuda {
namespace {

constexpr int kQuadCornerCount = 4;

void set_message(std::string* out_message, const std::string& text) {
    if (out_message != nullptr) {
        *out_message = text;
    }
}

/// 確保する segmentation の上限を求める。
///
/// 縮小率は fxfy = S / (S + max(W, H) * tau) であり、分母に長辺が入る。
/// そのため上限の寸法で計算した segmentation は上界にならない。長辺が
/// 短い入力ほど縮小が緩く、segmentation が大きくなるためである。
///
/// 例: 上限 1000x4000 の設定で 1000x1000 を入れると、上限で計算した
/// 138x552 より大きい 390x390 になる。
///
/// fxfy * W は W について単調増加であるため、正方形として計算した幅と
/// 高さがそれぞれの上界になる。
Status sizing_segmentation(const DetectorConfig& config, int* out_width, int* out_height) {
    detail::ScalePlan by_width;
    detail::ScalePlan by_height;
    Status status =
            detail::plan_scales(config, config.max_width_px_, config.max_width_px_, &by_width);
    if (status != Status::kOk) {
        return status;
    }
    status = detail::plan_scales(config, config.max_height_px_, config.max_height_px_, &by_height);
    if (status != Status::kOk) {
        return status;
    }
    *out_width = by_width.segmentation_width_px_;
    *out_height = by_height.segmentation_height_px_;
    return Status::kOk;
}

}  // namespace

/// Detector の実体。
class Detector::Impl {
public:
    Impl() = default;
    ~Impl() { this->release_graph(); }
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    Status initialize(const DictionaryTable& dictionary, const DetectorConfig& config,
                      std::string* out_message);
    Status detect_async(const ImageViewU8& image, cudaStream_t stream, std::string* out_message);
    Status device_detections(DeviceDetections* out) const;
    Status download(HostDetections* out, cudaStream_t stream, std::string* out_message);

    const WorkspaceStatistics& workspace_statistics() const { return this->frame_.statistics(); }
    bool initialized() const { return this->initialized_; }

private:
    Status reserve_all(const ImageViewU8& image, std::string* out_message);
    Status dispatch(const ImageViewU8& image, cudaStream_t stream);
    Status capture_graph(const ImageViewU8& image, cudaStream_t stream, std::string* out_message);
    void release_graph();

    bool initialized_ = false;
    bool has_result_ = false;
    DetectorConfig config_;
    int marker_size_ = 0;
    /// 補正で起こす block 数。device の SM 数から initialize 時に決める。
    int refine_blocks_ = 0;
    /// 捕らえた発行列。以降は 1 回の起動で済む。
    ///
    /// graph は kernel の引数を焼き込む。workspace の切り出しや設定が変わった
    /// のに使い回すと、**静かに古い pointer と古い値で走る**。破棄の契機を
    /// 取りこぼさないよう、捕らえたときの stream と配置を併せて持つ。
    cudaGraphExec_t graph_ = nullptr;
    cudaStream_t graph_stream_ = nullptr;
    /// 前回の入力。配置を組み直す必要があるかの判定に使う。
    const std::uint8_t* last_data_ = nullptr;
    int last_width_px_ = 0;
    int last_height_px_ = 0;
    std::size_t last_pitch_bytes_ = 0;

    /// Dictionary 専用。frame をまたいで生きるため reset しない。
    Workspace persistent_;
    /// frame ごとに reset して切り出し直す。
    Workspace frame_;

    detail::DeviceDictionary dictionary_;
    detail::ScalePlan plan_;
    detail::PreprocessBuffers preprocess_;
    detail::ThresholdBuffers threshold_;
    detail::LabelBuffers labels_;
    detail::LabelStatisticsBuffers stats_;
    detail::QuadBuffers quads_;
    detail::CandidateFilterBuffers filter_;
    detail::DeviceCandidates candidates_;
    detail::CandidateGroupBuffers groups_;
    detail::DeviceCandidates grouped_;
    detail::CanonicalBuffers canonical_;
    detail::CellRatioBuffers ratios_;
    detail::MatchBuffers matches_;
    detail::CandidateTreeBuffers tree_;
    detail::DetectionEmitBuffers emit_;
    detail::DeviceDetections detections_;
    detail::CornerRefineBuffers refine_;
};

Status Detector::Impl::initialize(const DictionaryTable& dictionary, const DetectorConfig& config,
                                  std::string* out_message) {
    if (dictionary.codes_ == nullptr || dictionary.code_count_ <= 0 ||
        dictionary.marker_size_ < 1) {
        set_message(out_message, "Dictionary が不正");
        return Status::kUnsupportedDictionary;
    }
    Status status = config.validate(out_message);
    if (status != Status::kOk) {
        return status;
    }
    // 支持しない組み合わせを拒否する。ArUco3 有効で補正を切ると四隅が縮小後の
    // 座標のまま残り、ArUco3 無効で補正を入れても段が 1 つしかないため補正が
    // 1 度も走らない。どちらも黙って通すと座標系が食い違う。
    if (config.use_aruco3_detection_ && config.corner_refine_method_ == CornerRefineMethod::kNone) {
        set_message(out_message,
                    "use_aruco3_detection が有効な場合、corner_refine_method に kNone は "
                    "使えない。四隅が縮小後の座標のまま残る");
        return Status::kInvalidConfig;
    }
    if (!config.use_aruco3_detection_ &&
        config.corner_refine_method_ == CornerRefineMethod::kSubpix) {
        set_message(out_message,
                    "use_aruco3_detection が無効な場合、corner_refine_method に kSubpix は "
                    "使えない。pyramid の段が 1 つしかなく補正が走らない");
        return Status::kInvalidConfig;
    }

    int sizing_width = 0;
    int sizing_height = 0;
    status = sizing_segmentation(config, &sizing_width, &sizing_height);
    if (status != Status::kOk) {
        set_message(out_message, "縮小率を決められない");
        return status;
    }
    detail::ScalePlan sizing_plan;
    status = detail::plan_scales(config, config.max_width_px_, config.max_height_px_, &sizing_plan);
    if (status != Status::kOk) {
        return status;
    }
    // 段数は長辺が大きいほど増える。上限の寸法で数えたものが上界になる。
    sizing_plan.segmentation_width_px_ = sizing_width;
    sizing_plan.segmentation_height_px_ = sizing_height;

    const int marker_size = dictionary.marker_size_;
    const std::size_t sizes[] = {
            detail::preprocess_workspace_bytes(sizing_plan, config.max_width_px_,
                                               config.max_height_px_),
            detail::threshold_workspace_bytes(config, sizing_width, sizing_height),
            detail::labeling_workspace_bytes(sizing_width, sizing_height),
            detail::label_stats_workspace_bytes(sizing_width, sizing_height),
            detail::quad_workspace_bytes(sizing_width, sizing_height),
            detail::candidate_workspace_bytes(config, sizing_width, sizing_height),
            detail::candidate_group_workspace_bytes(config),
            detail::canonical_workspace_bytes(config, marker_size),
            detail::cell_ratio_workspace_bytes(config, marker_size),
            detail::match_workspace_bytes(config),
            detail::candidate_tree_workspace_bytes(config),
            detail::detection_workspace_bytes(config),
            detail::corner_refine_workspace_bytes(config),
    };
    std::size_t frame_bytes = 0;
    for (const std::size_t value : sizes) {
        if (value == 0U) {
            set_message(out_message, "設定から workspace の必要量を決められない");
            return Status::kInvalidConfig;
        }
        frame_bytes += value;
    }
    const std::size_t persistent_bytes = detail::device_dictionary_workspace_bytes(dictionary);
    if (persistent_bytes == 0U) {
        set_message(out_message, "Dictionary の必要量を決められない");
        return Status::kUnsupportedDictionary;
    }

    status = this->persistent_.ensure_capacity(persistent_bytes, MemorySpace::kDevice, out_message);
    if (status != Status::kOk) {
        return status;
    }
    status = this->frame_.ensure_capacity(frame_bytes, MemorySpace::kDevice, out_message);
    if (status != Status::kOk) {
        return status;
    }
    this->persistent_.reset();
    status = detail::upload_dictionary(dictionary, this->persistent_, &this->dictionary_, nullptr);
    if (status != Status::kOk) {
        set_message(out_message, "Dictionary を device へ転送できない");
        return status;
    }
    // 転送元は host の静的記憶域であり pageable である。ここで同期しておかないと、
    // detect_async が別の stream を使ったときに転送との順序が保証されない。
    status = detail::check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize",
                                "detector.initialize", -1, nullptr);
    if (status != Status::kOk) {
        set_message(out_message, "Dictionary の転送を待てない");
        return status;
    }

    // 設定と Dictionary が変わった。焼き込んだ値は無効になった。
    //
    // これは冗長な防御である。has_result_ を false にするため、次の
    // detect_async が必ず reserve_all を通り、そこで破棄される。実際に
    // この行を落としても test は落ちない (変異で確認済み)。取り違えを
    // 防ぐため残す。
    this->release_graph();
    // device の SM 数から補正の block 数を決める。取得できない場合は
    // 下限が使われる。frame ごとに問い合わせない。
    DeviceProbeResult probe;
    this->refine_blocks_ = detail::refine_block_count(
            (probe_device(0, &probe) == Status::kOk) ? probe.multi_processor_count_ : 0);

    this->config_ = config;
    this->marker_size_ = marker_size;
    this->has_result_ = false;
    this->last_data_ = nullptr;
    this->last_width_px_ = 0;
    this->last_height_px_ = 0;
    this->last_pitch_bytes_ = 0;
    this->initialized_ = true;
    return Status::kOk;
}

Status Detector::Impl::reserve_all(const ImageViewU8& image, std::string* out_message) {
    const int width = this->plan_.segmentation_width_px_;
    const int height = this->plan_.segmentation_height_px_;
    this->frame_.reset();

    const Status steps[] = {
            detail::reserve_preprocess(this->plan_, image, this->frame_, &this->preprocess_),
            detail::reserve_threshold(this->config_, width, height, this->frame_,
                                      &this->threshold_),
            detail::reserve_labeling(width, height, this->frame_, &this->labels_),
            detail::reserve_label_stats(width, height, this->frame_, &this->stats_),
            detail::reserve_quads(width, height, this->frame_, &this->quads_),
            detail::reserve_candidates(this->config_, width, height, this->frame_, &this->filter_,
                                       &this->candidates_),
            detail::reserve_candidate_groups(this->config_, this->frame_, &this->groups_,
                                             &this->grouped_),
            detail::reserve_canonical(this->config_, this->marker_size_, this->frame_,
                                      &this->canonical_),
            detail::reserve_cell_ratios(this->config_, this->marker_size_, this->frame_,
                                        &this->ratios_),
            detail::reserve_matches(this->config_, this->frame_, &this->matches_),
            detail::reserve_candidate_tree(this->config_, this->frame_, &this->tree_),
            detail::reserve_detections(this->config_, this->frame_, &this->emit_,
                                       &this->detections_),
            detail::reserve_corner_refine(this->config_, this->frame_, &this->refine_),
    };
    for (const Status status : steps) {
        if (status != Status::kOk) {
            set_message(out_message,
                        "workspace を切り出せない。max_width_px か max_height_px を "
                        "入力に合わせて上げること");
            return status;
        }
    }
    return Status::kOk;
}

Status Detector::Impl::dispatch(const ImageViewU8& image, cudaStream_t stream) {
    (void)image;
    Status status = detail::build_pyramid_async(&this->preprocess_, this->config_, stream);
    if (status != Status::kOk) {
        return status;
    }
    status = detail::build_segmentation_async(this->plan_, &this->preprocess_, this->config_,
                                              stream);
    if (status != Status::kOk) {
        return status;
    }

    ImageViewU8 segmentation;
    segmentation.data_ = this->preprocess_.segmentation_.data_;
    segmentation.width_px_ = this->preprocess_.segmentation_.width_px_;
    segmentation.height_px_ = this->preprocess_.segmentation_.height_px_;
    segmentation.pitch_bytes_ = this->preprocess_.segmentation_.pitch_bytes_;
    segmentation.space_ = MemorySpace::kDevice;

    status = detail::build_threshold_async(segmentation, &this->threshold_, this->config_, stream);
    if (status != Status::kOk) {
        return status;
    }

    // window ごとの走査は同じ stream で直列に行う。labels_、stats_、quads_ を
    // window 間で共有しており、別 stream へ散らすと互いに上書きする。
    for (int window = 0; window < this->threshold_.window_count_; ++window) {
        status = detail::build_labels_async(this->threshold_.binary_[window], &this->labels_,
                                            stream);
        if (status != Status::kOk) {
            return status;
        }
        status = detail::build_label_stats_async(this->labels_, &this->stats_, stream);
        if (status != Status::kOk) {
            return status;
        }
        status = detail::build_quads_async(this->labels_, this->stats_, &this->quads_, stream);
        if (status != Status::kOk) {
            return status;
        }
        status = detail::build_candidates_async(this->labels_, this->stats_, this->quads_,
                                                this->config_, &this->filter_, &this->candidates_,
                                                window != 0, stream);
        if (status != Status::kOk) {
            return status;
        }
    }

    status = detail::build_candidate_groups_async(this->candidates_, this->config_, &this->groups_,
                                                  &this->grouped_, stream);
    if (status != Status::kOk) {
        return status;
    }
    status = detail::build_canonical_async(this->preprocess_, this->plan_, this->grouped_,
                                           this->config_, &this->canonical_, stream);
    if (status != Status::kOk) {
        return status;
    }
    status = detail::build_cell_ratios_async(this->canonical_, this->grouped_, this->config_,
                                             this->marker_size_, &this->ratios_, stream);
    if (status != Status::kOk) {
        return status;
    }
    status = detail::match_candidates_async(this->ratios_, this->grouped_, this->dictionary_,
                                            this->config_, &this->matches_, stream);
    if (status != Status::kOk) {
        return status;
    }
    status = detail::build_candidate_tree_async(this->grouped_, &this->tree_, stream);
    if (status != Status::kOk) {
        return status;
    }
    status =
            detail::resolve_suppression_async(this->grouped_, this->matches_, &this->tree_, stream);
    if (status != Status::kOk) {
        return status;
    }
    status = detail::emit_detections_async(this->grouped_, this->matches_, this->tree_,
                                           &this->emit_, &this->detections_, stream);
    if (status != Status::kOk) {
        return status;
    }
    if (this->config_.corner_refine_method_ != CornerRefineMethod::kSubpix) {
        return Status::kOk;
    }
    detail::PyramidRef pyramid;
    status = detail::make_pyramid_ref(this->preprocess_, &pyramid);
    if (status != Status::kOk) {
        return status;
    }
    return detail::refine_corners_async(pyramid, this->plan_, this->config_, this->refine_blocks_,
                                        &this->refine_, &this->detections_, stream);
}

void Detector::Impl::release_graph() {
    if (this->graph_ != nullptr) {
        // 破棄の失敗は報告しない。ここで返せる先が無く、次の capture が
        // 新しい実体を作るため状態は矛盾しない。
        static_cast<void>(cudaGraphExecDestroy(this->graph_));
        this->graph_ = nullptr;
        this->graph_stream_ = nullptr;
    }
}

Status Detector::Impl::capture_graph(const ImageViewU8& image, cudaStream_t stream,
                                     std::string* out_message) {
    this->release_graph();
    // 捕らえる間は発行だけが記録され、kernel は動かない。
    Status status =
            detail::check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
                               "cudaStreamBeginCapture", "detector.capture_graph", -1, stream);
    if (status != Status::kOk) {
        set_message(out_message, "発行列を捕らえられない");
        return status;
    }
    const Status dispatched = this->dispatch(image, stream);
    cudaGraph_t graph = nullptr;
    const cudaError_t ended = cudaStreamEndCapture(stream, &graph);
    if (dispatched != Status::kOk) {
        // 捕らえるのを止めてから返す。止めないと stream が捕獲状態に残る。
        if (graph != nullptr) {
            static_cast<void>(cudaGraphDestroy(graph));
        }
        set_message(out_message, "発行列の記録中に失敗した");
        return dispatched;
    }
    status =
            detail::check_cuda(ended, "cudaStreamEndCapture", "detector.capture_graph", -1, stream);
    if (status != Status::kOk) {
        set_message(out_message, "発行列を閉じられない");
        return status;
    }
    // cudaGraphInstantiate の署名は CUDA 12 で変わった。11.x (Jetson の L4T
    // R35 が載せる 11.4) は error node と log buffer を取る 5 引数版である。
#if CUDART_VERSION >= 12000
    const cudaError_t instantiated = cudaGraphInstantiate(&this->graph_, graph, 0);
#else
    const cudaError_t instantiated =
            cudaGraphInstantiate(&this->graph_, graph, nullptr, nullptr, 0);
#endif
    status = detail::check_cuda(instantiated, "cudaGraphInstantiate", "detector.capture_graph", -1,
                                stream);
    static_cast<void>(cudaGraphDestroy(graph));
    if (status != Status::kOk) {
        this->graph_ = nullptr;
        set_message(out_message, "発行列を実体化できない");
        return status;
    }
    this->graph_stream_ = stream;
    return Status::kOk;
}

Status Detector::Impl::detect_async(const ImageViewU8& image, cudaStream_t stream,
                                    std::string* out_message) {
    if (!this->initialized_) {
        set_message(out_message, "initialize が済んでいない");
        return Status::kNotInitialized;
    }
    Status status = validate_image_view(image, out_message);
    if (status != Status::kOk) {
        return status;
    }
    // kernel が直接読むため host 側の memory は受け取れない。
    if (image.space_ != MemorySpace::kDevice && image.space_ != MemorySpace::kManaged) {
        set_message(out_message, "入力画像は device か managed の空間である必要がある");
        return Status::kInvalidImage;
    }
    if (image.width_px_ > this->config_.max_width_px_ ||
        image.height_px_ > this->config_.max_height_px_) {
        set_message(out_message, "入力が max_width_px または max_height_px を超えている");
        return Status::kInvalidArgument;
    }

    const bool layout_changed = image.data_ != this->last_data_ ||
                                image.width_px_ != this->last_width_px_ ||
                                image.height_px_ != this->last_height_px_ ||
                                image.pitch_bytes_ != this->last_pitch_bytes_;
    if (layout_changed && this->has_result_) {
        // 配置を組み直すため、前 frame の kernel が同じ領域を触り終えるのを待つ。
        // 同じ入力を流し続ける定常状態では 1 度も通らない。
        status = detail::check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize",
                                    "detector.detect_async", -1, stream);
        if (status != Status::kOk) {
            return status;
        }
    }

    status = detail::plan_scales(this->config_, image.width_px_, image.height_px_, &this->plan_);
    if (status != Status::kOk) {
        return status;
    }
    if (layout_changed || !this->has_result_) {
        status = this->reserve_all(image, out_message);
        if (status != Status::kOk) {
            return status;
        }
        // 切り出しが変わったので、焼き込んだ pointer は無効になった。
        // **この破棄は落とせない。** 落とすと古い pointer を静かに叩き、
        // graph_is_rebuilt_when_layout_changes が落ちる (変異で確認済み)。
        this->release_graph();
    }
    // stream が変わったら捕らえ直す。
    //
    // これも冗長な防御である。graph は捕らえた stream 以外へ起動しても内部の
    // 順序を保つため、落としても test は落ちない (変異で確認済み)。ただし
    // 前の stream に同じ領域を触る仕事が残っている場合、2 本の stream の間に
    // 順序の保証は無い。呼出側が stream を渡し替える使い方を安全にするため
    // 捕らえ直す。
    if (this->graph_ != nullptr && this->graph_stream_ != stream) {
        this->release_graph();
    }
    // **既定 stream は捕らえられない。** CUDA は legacy default stream の
    // 捕獲を許さないため、nullptr を渡された場合は従来どおり 1 段ずつ発行する。
    // 明示的な stream を渡せば発行列を 1 回の起動へ畳める。
    if (stream == nullptr) {
        this->release_graph();
        status = this->dispatch(image, stream);
        if (status != Status::kOk) {
            set_message(out_message, "kernel の発行に失敗した");
            return status;
        }
    } else {
        if (this->graph_ == nullptr) {
            status = this->capture_graph(image, stream, out_message);
            if (status != Status::kOk) {
                return status;
            }
        }
        status = detail::check_cuda(cudaGraphLaunch(this->graph_, stream), "cudaGraphLaunch",
                                    "detector.detect_async", -1, stream);
        if (status != Status::kOk) {
            set_message(out_message, "発行列を起動できない");
            return status;
        }
    }

    this->last_data_ = image.data_;
    this->last_width_px_ = image.width_px_;
    this->last_height_px_ = image.height_px_;
    this->last_pitch_bytes_ = image.pitch_bytes_;
    this->has_result_ = true;
    return Status::kOk;
}

Status Detector::Impl::device_detections(DeviceDetections* out) const {
    if (out == nullptr) {
        return Status::kInvalidArgument;
    }
    if (!this->has_result_) {
        return Status::kNotInitialized;
    }
    *out = this->detections_;
    return Status::kOk;
}

Status Detector::Impl::download(HostDetections* out, cudaStream_t stream,
                                std::string* out_message) {
    if (out == nullptr) {
        return Status::kInvalidArgument;
    }
    if (!this->has_result_) {
        set_message(out_message, "detect_async が済んでいない");
        return Status::kNotInitialized;
    }

    int count = 0;
    const Status count_status = detail::read_detection_count(this->detections_, &count, stream);
    if (count_status != Status::kOk && count_status != Status::kMarkerOverflow) {
        set_message(out_message, "検出数を読めない");
        return count_status;
    }

    HostDetections result;
    result.marker_overflow_ = (count_status == Status::kMarkerOverflow);
    const auto total = static_cast<std::size_t>(count);
    result.ids_.resize(total);
    result.rotations_.resize(total);
    result.corners_.resize(total * kQuadCornerCount * 2U);
    std::int32_t accepted_total = 0;

    if (total > 0U) {
        std::vector<float> corner_x(total * kQuadCornerCount);
        std::vector<float> corner_y(total * kQuadCornerCount);
        const std::size_t plane = total * sizeof(std::int32_t);
        Status status =
                detail::check_cuda(cudaMemcpyAsync(result.ids_.data(), this->detections_.ids_,
                                                   plane, cudaMemcpyDeviceToHost, stream),
                                   "cudaMemcpyAsync", "detector.download.ids", -1, stream);
        if (status != Status::kOk) {
            return status;
        }
        status = detail::check_cuda(
                cudaMemcpyAsync(result.rotations_.data(), this->detections_.rotations_, plane,
                                cudaMemcpyDeviceToHost, stream),
                "cudaMemcpyAsync", "detector.download.rotations", -1, stream);
        if (status != Status::kOk) {
            return status;
        }
        for (int corner = 0; corner < kQuadCornerCount; ++corner) {
            const auto offset = static_cast<std::ptrdiff_t>(corner) * this->detections_.capacity_;
            const std::size_t slot = static_cast<std::size_t>(corner) * total;
            status = detail::check_cuda(
                    cudaMemcpyAsync(corner_x.data() + slot, this->detections_.corner_x_ + offset,
                                    total * sizeof(float), cudaMemcpyDeviceToHost, stream),
                    "cudaMemcpyAsync", "detector.download.corner_x", -1, stream);
            if (status != Status::kOk) {
                return status;
            }
            status = detail::check_cuda(
                    cudaMemcpyAsync(corner_y.data() + slot, this->detections_.corner_y_ + offset,
                                    total * sizeof(float), cudaMemcpyDeviceToHost, stream),
                    "cudaMemcpyAsync", "detector.download.corner_y", -1, stream);
            if (status != Status::kOk) {
                return status;
            }
        }
        status = detail::check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize",
                                    "detector.download", -1, stream);
        if (status != Status::kOk) {
            return status;
        }
        // device の面ごとの並びから、検出ごとに 8 要素へ詰め替える。
        for (std::size_t index = 0; index < total; ++index) {
            for (int corner = 0; corner < kQuadCornerCount; ++corner) {
                const std::size_t source = (static_cast<std::size_t>(corner) * total) + index;
                const std::size_t destination =
                        (index * kQuadCornerCount * 2U) + (static_cast<std::size_t>(corner) * 2U);
                result.corners_[destination] = corner_x[source];
                result.corners_[destination + 1U] = corner_y[source];
            }
        }
    }

    const Status status =
            detail::check_cuda(cudaMemcpy(&accepted_total, this->detections_.accepted_total_,
                                          sizeof(std::int32_t), cudaMemcpyDeviceToHost),
                               "cudaMemcpy", "detector.download.total", -1, stream);
    if (status != Status::kOk) {
        return status;
    }
    result.accepted_total_ = accepted_total;
    *out = std::move(result);
    return count_status;
}

Detector::Detector() : impl_(std::make_unique<Impl>()) {}
Detector::~Detector() = default;
Detector::Detector(Detector&&) noexcept = default;
Detector& Detector::operator=(Detector&&) noexcept = default;

Status Detector::initialize(const DictionaryTable& dictionary, const DetectorConfig& config,
                            std::string* out_message) {
    if (this->impl_ == nullptr) {
        return Status::kNotInitialized;
    }
    return this->impl_->initialize(dictionary, config, out_message);
}

Status Detector::detect_async(const ImageViewU8& image, cudaStream_t stream,
                              std::string* out_message) {
    if (this->impl_ == nullptr) {
        return Status::kNotInitialized;
    }
    return this->impl_->detect_async(image, stream, out_message);
}

Status Detector::device_detections(DeviceDetections* out) const {
    if (this->impl_ == nullptr) {
        return Status::kNotInitialized;
    }
    return this->impl_->device_detections(out);
}

Status Detector::download(HostDetections* out, cudaStream_t stream, std::string* out_message) {
    if (this->impl_ == nullptr) {
        return Status::kNotInitialized;
    }
    return this->impl_->download(out, stream, out_message);
}

const WorkspaceStatistics& Detector::workspace_statistics() const {
    static const WorkspaceStatistics kEmpty;
    if (this->impl_ == nullptr) {
        return kEmpty;
    }
    return this->impl_->workspace_statistics();
}

bool Detector::initialized() const {
    return this->impl_ != nullptr && this->impl_->initialized();
}

}  // namespace aruco3cuda
