// SPDX-License-Identifier: Apache-2.0
#include "aruco3cuda/detector.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>
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
#include "candidate_retry.hpp"
#include "candidate_tree.hpp"
#include "cell_decode.hpp"
#include "cell_sample.hpp"
#include "contour_candidates.hpp"
#include "corner_refine.hpp"
#include "cuda_check.hpp"
#include "detection_emit.hpp"
#include "dictionary_match.hpp"
#include "labeling.hpp"
#include "preprocess.hpp"
#include "quad_extract.hpp"
#include "threshold.hpp"

#ifndef ARUCO3CUDA_EXPERIMENT_CONTOUR_CANDIDATES
#define ARUCO3CUDA_EXPERIMENT_CONTOUR_CANDIDATES 0
#endif

namespace aruco3cuda {
namespace {

constexpr int kQuadCornerCount = 4;

static_assert(static_cast<std::int32_t>(DeviceFailure::kInvalidContourCycle) ==
                      detail::kContourInvalidCycle,
              "contour failure bit drift");
static_assert(static_cast<std::int32_t>(DeviceFailure::kContourPointOverflow) ==
                      detail::kContourPointOverflow,
              "contour failure bit drift");
static_assert(static_cast<std::int32_t>(DeviceFailure::kCandidateOverflow) ==
                      detail::kContourCandidateOverflow,
              "contour failure bit drift");
static_assert(static_cast<std::int32_t>(DeviceFailure::kContourApproximationOverflow) ==
                      detail::kContourApproximationOverflow,
              "contour failure bit drift");

#if ARUCO3CUDA_EXPERIMENT_CONTOUR_CANDIDATES
/// Four directed edges per pixel bound the complete current-window point pool.
/// Keep the bound representable by the signed scan/path indices before casting.
int contour_point_capacity(int width, int height) {
    if (width <= 0 || height <= 0) {
        return 0;
    }
    const std::uint64_t pixels =
            static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height);
    return pixels <= static_cast<std::uint64_t>(std::numeric_limits<int>::max() / 4)
                   ? static_cast<int>(pixels * 4U)
                   : 0;
}
#endif

void set_message(std::string* out_message, const std::string& text) {
    if (out_message != nullptr) {
        *out_message = text;
    }
}

/// Computes the upper bound of the segmentation image that has to be allocated.
///
/// The downscaling factor is fxfy = S / (S + max(W, H) * tau), so the longer side appears in the
/// denominator. A segmentation computed from the maximum dimensions is therefore not an upper
/// bound: the shorter the longer side of the input, the gentler the downscaling and the larger
/// the segmentation image.
///
/// Example: with a configuration capped at 1000x4000, an input of 1000x1000 yields 390x390, which
/// is larger than the 138x552 computed from the caps.
///
/// fxfy * W increases monotonically in W, so the width and height computed as if the input were
/// square are each an upper bound.
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

/// The implementation behind Detector.
class Detector::Impl {
public:
    Impl() = default;
    ~Impl() {
        // The caller may already have destroyed its stream. The event remains
        // detector-owned and covers the last successful or partial submission.
        static_cast<void>(this->wait_for_completion(nullptr));
        this->release_graph();
        if (this->completion_ != nullptr) {
            static_cast<void>(cudaEventDestroy(this->completion_));
        }
    }
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
    Status wait_for_completion(std::string* out_message);
    Status enqueue_completion_wait(cudaStream_t stream, std::string* out_message);
    Status record_completion(cudaStream_t stream, std::string* out_message);

    bool initialized_ = false;
    bool has_result_ = false;
    /// Owned independently of the caller's stream lifetime or recycled handle.
    cudaEvent_t completion_ = nullptr;
    bool completion_recorded_ = false;
    /// An event-record failure could leave work without a usable completion marker.
    bool untracked_work_ = false;
    DetectorConfig config_;
    int marker_size_ = 0;
    /// Number of blocks the refinement launches. Decided from the SM count of the device during
    /// initialize.
    int refine_blocks_ = 0;
    /// The captured launch sequence. From then on a single launch is enough.
    ///
    /// A graph bakes in the kernel arguments. Reusing one after the workspace layout or the
    /// configuration has changed makes it **silently run with stale pointers and stale values**.
    /// The stream and the layout it was captured with are kept alongside it so that no reason to
    /// destroy it is missed.
    cudaGraphExec_t graph_ = nullptr;
    cudaStream_t graph_stream_ = nullptr;
    /// The previous input, used to decide whether the layout has to be rebuilt.
    const std::uint8_t* last_data_ = nullptr;
    int last_width_px_ = 0;
    int last_height_px_ = 0;
    std::size_t last_pitch_bytes_ = 0;

    /// Reserved for the dictionary. It lives across frames and is therefore never reset.
    Workspace persistent_;
    /// Reset and re-carved every frame.
    Workspace frame_;

    detail::DeviceDictionary dictionary_;
    detail::ScalePlan plan_;
    detail::PreprocessBuffers preprocess_;
    detail::ThresholdBuffers threshold_;
#if ARUCO3CUDA_EXPERIMENT_CONTOUR_CANDIDATES
    detail::ContourCandidateBuffers contours_;
#else
    detail::LabelBuffers labels_;
    detail::LabelStatisticsBuffers stats_;
    detail::QuadBuffers quads_;
#endif
    detail::CandidateFilterBuffers filter_;
    detail::DeviceCandidates candidates_;
    detail::CandidateGroupBuffers groups_;
    detail::DeviceCandidates grouped_;
#if ARUCO3CUDA_EXPERIMENT_CONTOUR_CANDIDATES
    detail::CandidateRetryBuffers retries_;
#endif
    detail::CanonicalBuffers canonical_;
    detail::CellRatioBuffers ratios_;
    detail::MatchBuffers matches_;
    detail::CandidateTreeBuffers tree_;
    detail::DetectionEmitBuffers emit_;
    detail::DeviceDetections detections_;
    detail::CornerRefineBuffers refine_;
};

Status Detector::Impl::wait_for_completion(std::string* out_message) {
    Status status = Status::kOk;
    if (this->untracked_work_) {
        status = detail::check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize",
                                    "detector.untracked_work", -1, nullptr);
    } else if (this->completion_recorded_) {
        status = detail::check_cuda(cudaEventSynchronize(this->completion_), "cudaEventSynchronize",
                                    "detector.completion", -1, nullptr);
    }
    if (status != Status::kOk) {
        this->has_result_ = false;
        this->initialized_ = false;
        set_message(out_message, "cannot wait for previous detector work");
        return status;
    }
    this->completion_recorded_ = false;
    this->untracked_work_ = false;
    return Status::kOk;
}

Status Detector::Impl::enqueue_completion_wait(cudaStream_t stream, std::string* out_message) {
    if (!this->completion_recorded_) {
        return Status::kOk;
    }
    // Always wait, even when handle values compare equal: a caller may destroy
    // a stream and CUDA may recycle its handle while its work is still pending.
    const Status status =
            detail::check_cuda(cudaStreamWaitEvent(stream, this->completion_, 0),
                               "cudaStreamWaitEvent", "detector.completion", -1, stream);
    if (status != Status::kOk) {
        set_message(out_message, "cannot order the stream after previous detector work");
    }
    return status;
}

Status Detector::Impl::record_completion(cudaStream_t stream, std::string* out_message) {
    const Status status = detail::check_cuda(cudaEventRecord(this->completion_, stream),
                                             "cudaEventRecord", "detector.completion", -1, stream);
    if (status == Status::kOk) {
        this->completion_recorded_ = true;
        return Status::kOk;
    }
    this->has_result_ = false;
    this->initialized_ = false;
    // A submission may already have queued work. Drain that stream before any
    // buffer can be reused. If even the drain fails, subsequent initialization
    // and destruction must attempt a device drain rather than trust an old event.
    const cudaError_t drained = cudaStreamSynchronize(stream);
    this->completion_recorded_ = false;
    this->untracked_work_ = drained != cudaSuccess;
    set_message(out_message, this->untracked_work_
                                     ? "cannot record detector completion or drain submitted work"
                                     : "cannot record detector completion; submitted work drained");
    return status;
}

Status Detector::Impl::initialize(const DictionaryTable& dictionary, const DetectorConfig& config,
                                  std::string* out_message) {
    if (dictionary.codes_ == nullptr || dictionary.code_count_ <= 0 ||
        dictionary.marker_size_ < 1) {
        set_message(out_message, "invalid dictionary");
        return Status::kUnsupportedDictionary;
    }
    Status status = config.validate(out_message);
    if (status != Status::kOk) {
        return status;
    }
    // Reject the combinations that are not supported. With ArUco3 enabled and refinement off,
    // the corners stay in downscaled coordinates; with ArUco3 disabled and refinement on, the
    // pyramid has a single level and the refinement never runs. Letting either through silently
    // would leave the coordinate systems inconsistent.
    if (config.use_aruco3_detection_ && config.corner_refine_method_ == CornerRefineMethod::kNone) {
        set_message(out_message,
                    "corner_refine_method must not be kNone while use_aruco3_detection is "
                    "enabled: the corners would stay in downscaled coordinates");
        return Status::kInvalidConfig;
    }
    if (!config.use_aruco3_detection_ &&
        config.corner_refine_method_ == CornerRefineMethod::kSubpix) {
        set_message(out_message,
                    "corner_refine_method must not be kSubpix while use_aruco3_detection is "
                    "disabled: the pyramid has a single level and the refinement never runs");
        return Status::kInvalidConfig;
    }

    int sizing_width = 0;
    int sizing_height = 0;
    status = sizing_segmentation(config, &sizing_width, &sizing_height);
    if (status != Status::kOk) {
        set_message(out_message, "cannot determine the downscaling factor");
        return status;
    }
    detail::ScalePlan sizing_plan;
    status = detail::plan_scales(config, config.max_width_px_, config.max_height_px_, &sizing_plan);
    if (status != Status::kOk) {
        return status;
    }
    // The level count grows with the longer side, so counting it from the maximum dimensions
    // gives an upper bound.
    sizing_plan.segmentation_width_px_ = sizing_width;
    sizing_plan.segmentation_height_px_ = sizing_height;

    const int marker_size = dictionary.marker_size_;
#if ARUCO3CUDA_EXPERIMENT_CONTOUR_CANDIDATES
    const int max_contour_points = contour_point_capacity(sizing_width, sizing_height);
    if (max_contour_points == 0) {
        set_message(out_message,
                    "experimental contour point budget 4 * width * height exceeds signed "
                    "index capacity");
        return Status::kInvalidConfig;
    }
#endif
    const std::size_t sizes[] = {
            detail::preprocess_workspace_bytes(sizing_plan, config.max_width_px_,
                                               config.max_height_px_),
            detail::threshold_workspace_bytes(config, sizing_width, sizing_height),
#if ARUCO3CUDA_EXPERIMENT_CONTOUR_CANDIDATES
            detail::contour_candidate_workspace_bytes(sizing_width, sizing_height,
                                                      max_contour_points),
#else
            detail::labeling_workspace_bytes(sizing_width, sizing_height),
            detail::label_stats_workspace_bytes(sizing_width, sizing_height),
            detail::quad_workspace_bytes(sizing_width, sizing_height),
#endif
            detail::candidate_workspace_bytes(config, sizing_width, sizing_height),
            detail::candidate_group_workspace_bytes(config),
#if ARUCO3CUDA_EXPERIMENT_CONTOUR_CANDIDATES
            detail::candidate_retry_workspace_bytes(config),
#endif
            detail::canonical_workspace_bytes(config, marker_size),
            detail::cell_ratio_workspace_bytes(config, marker_size),
            detail::match_workspace_bytes(config),
            detail::candidate_tree_workspace_bytes(config),
            detail::detection_workspace_bytes(config),
            detail::corner_refine_workspace_bytes(config),
    };
    std::size_t frame_bytes = 0;
    for (const std::size_t value : sizes) {
        if (value == 0U || frame_bytes > std::numeric_limits<std::size_t>::max() - value) {
            set_message(out_message,
                        "cannot determine the required workspace size from the configuration");
            return Status::kInvalidConfig;
        }
        frame_bytes += value;
    }
    const std::size_t persistent_bytes = detail::device_dictionary_workspace_bytes(dictionary);
    if (persistent_bytes == 0U) {
        set_message(out_message, "cannot determine the size required by the dictionary");
        return Status::kUnsupportedDictionary;
    }

    // Host validation above leaves an existing detector intact. Once resource
    // changes begin, failure leaves it explicitly uninitialized, never pointing
    // at an earlier dictionary that a successful arena growth may have freed.
    status = this->wait_for_completion(out_message);
    if (status != Status::kOk) {
        return status;
    }
    if (this->completion_ == nullptr) {
        status = detail::check_cuda(
                cudaEventCreateWithFlags(&this->completion_, cudaEventDisableTiming),
                "cudaEventCreateWithFlags", "detector.initialize", -1, nullptr);
        if (status != Status::kOk) {
            return status;
        }
    }
    this->initialized_ = false;
    this->has_result_ = false;
    this->release_graph();
    this->dictionary_ = {};
    this->detections_ = {};
    this->last_data_ = nullptr;
    this->last_width_px_ = 0;
    this->last_height_px_ = 0;
    this->last_pitch_bytes_ = 0;

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
        set_message(out_message, "cannot transfer the dictionary to the device");
        const Status tracked = this->record_completion(nullptr, out_message);
        if (tracked != Status::kOk) {
            return tracked;
        }
        // initialize borrows host dictionary storage only for this call, even
        // when an asynchronous upload reports failure after queuing work.
        const Status drained = this->wait_for_completion(out_message);
        return drained == Status::kOk ? status : drained;
    }
    // The transfer source is host static storage and therefore pageable. Without synchronizing
    // here, nothing orders the transfer against a detect_async that runs on a different stream.
    status = detail::check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize",
                                "detector.initialize", -1, nullptr);
    if (status != Status::kOk) {
        set_message(out_message, "cannot wait for the dictionary transfer");
        const Status tracked = this->record_completion(nullptr, out_message);
        if (tracked != Status::kOk) {
            return tracked;
        }
        // initialize borrows host dictionary storage only for this call, even
        // when an asynchronous upload reports failure after queuing work.
        const Status drained = this->wait_for_completion(out_message);
        return drained == Status::kOk ? status : drained;
    }

    // Decide the refinement block count from the SM count of the device. When it cannot be
    // read, the lower bound is used. This is not queried per frame.
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
#if ARUCO3CUDA_EXPERIMENT_CONTOUR_CANDIDATES
    const int max_contour_points = contour_point_capacity(width, height);
    if (max_contour_points == 0) {
        set_message(out_message, "experimental contour point budget exceeds signed index capacity");
        return Status::kInvalidConfig;
    }
#endif
    this->frame_.reset();

    const Status steps[] = {
            detail::reserve_preprocess(this->plan_, image, this->frame_, &this->preprocess_),
            detail::reserve_threshold(this->config_, width, height, this->frame_,
                                      &this->threshold_),
#if ARUCO3CUDA_EXPERIMENT_CONTOUR_CANDIDATES
            detail::reserve_contour_candidates(width, height, max_contour_points, this->frame_,
                                               &this->contours_),
#else
            detail::reserve_labeling(width, height, this->frame_, &this->labels_),
            detail::reserve_label_stats(width, height, this->frame_, &this->stats_),
            detail::reserve_quads(width, height, this->frame_, &this->quads_),
#endif
            detail::reserve_candidates(this->config_, width, height, this->frame_, &this->filter_,
                                       &this->candidates_),
            detail::reserve_candidate_groups(this->config_, this->frame_, &this->groups_,
                                             &this->grouped_),
#if ARUCO3CUDA_EXPERIMENT_CONTOUR_CANDIDATES
            detail::reserve_candidate_retries(this->config_, this->frame_, &this->retries_),
#endif
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
                        "cannot carve the workspace: raise max_width_px or max_height_px to "
                        "match the input");
            return status;
        }
    }
    return Status::kOk;
}

Status Detector::Impl::dispatch(const ImageViewU8& image, cudaStream_t stream) {
    (void)image;
    Status status = Status::kOk;
#if !defined(ARUCO3CUDA_EXPERIMENT_DEMAND_PYRAMID) || !ARUCO3CUDA_EXPERIMENT_DEMAND_PYRAMID
    status = detail::build_pyramid_async(&this->preprocess_, this->config_, stream);
    if (status != Status::kOk) {
        return status;
    }
#endif
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

    // Per-window scratch is shared, so the extraction sweeps remain ordered on
    // this stream. The contour route keeps failure bits across appended windows.
    for (int window = 0; window < this->threshold_.window_count_; ++window) {
#if ARUCO3CUDA_EXPERIMENT_CONTOUR_CANDIDATES
        status = detail::build_contour_candidates_async(this->threshold_.binary_[window],
                                                        this->config_, &this->contours_,
                                                        &this->candidates_, window != 0, stream);
        if (status != Status::kOk) {
            return status;
        }
#else
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
#endif
    }

    status = detail::build_candidate_groups_async(this->candidates_, this->config_, &this->groups_,
                                                  &this->grouped_, stream);
    if (status != Status::kOk) {
        return status;
    }
#if defined(ARUCO3CUDA_EXPERIMENT_DEMAND_PYRAMID) && ARUCO3CUDA_EXPERIMENT_DEMAND_PYRAMID
    // Segmentation uses level zero directly. Deferring the separate decode/refinement
    // pyramid lets the current frame's device candidate count determine its work.
    status = detail::build_demanded_pyramid_async(&this->preprocess_, this->plan_, this->grouped_,
                                                  this->config_, stream);
    if (status != Status::kOk) {
        return status;
    }
#endif
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
#if ARUCO3CUDA_EXPERIMENT_CONTOUR_CANDIDATES
    // Containment is already built from the original representative geometry.
    // Retry only failed representatives using this frame's grouped alternatives.
    // Their inherited contour lengths select levels already covered by the demand
    // pyramid. Canonical/ratio scratch is reused; primary matches stay separate.
    status = detail::prepare_candidate_retries_async(this->candidates_, this->grouped_,
                                                     this->matches_, this->groups_, this->config_,
                                                     this->marker_size_, &this->retries_, stream);
    if (status != Status::kOk) {
        return status;
    }
    status = detail::build_canonical_async(this->preprocess_, this->plan_,
                                           this->retries_.candidates_, this->config_,
                                           &this->canonical_, stream);
    if (status != Status::kOk) {
        return status;
    }
    status = detail::build_cell_ratios_async(this->canonical_, this->retries_.candidates_,
                                             this->config_, this->marker_size_, &this->ratios_,
                                             stream);
    if (status != Status::kOk) {
        return status;
    }
    status = detail::match_candidates_async(this->ratios_, this->retries_.candidates_,
                                            this->dictionary_, this->config_,
                                            &this->retries_.matches_, stream);
    if (status != Status::kOk) {
        return status;
    }
    status = detail::resolve_candidate_retries_async(this->retries_, &this->grouped_,
                                                     &this->matches_, stream);
    if (status != Status::kOk) {
        return status;
    }
#endif
    status =
            detail::resolve_suppression_async(this->grouped_, this->matches_, &this->tree_, stream);
    if (status != Status::kOk) {
        return status;
    }
    const std::int32_t* upstream_failure = nullptr;
#if ARUCO3CUDA_EXPERIMENT_CONTOUR_CANDIDATES
    upstream_failure = this->contours_.failure_;
#endif
    status = detail::emit_detections_async(this->grouped_, this->matches_, this->tree_,
                                           &this->emit_, &this->detections_, stream,
                                           &this->candidates_, upstream_failure);
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
        // A failed destruction is not reported: there is nowhere to return it to here, and the
        // next capture creates a new instance, so the state stays consistent.
        static_cast<void>(cudaGraphExecDestroy(this->graph_));
        this->graph_ = nullptr;
        this->graph_stream_ = nullptr;
    }
}

Status Detector::Impl::capture_graph(const ImageViewU8& image, cudaStream_t stream,
                                     std::string* out_message) {
    this->release_graph();
    // While capturing, only the launches are recorded and no kernel actually runs.
    Status status =
            detail::check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
                               "cudaStreamBeginCapture", "detector.capture_graph", -1, stream);
    if (status != Status::kOk) {
        set_message(out_message, "cannot begin capturing the launch sequence");
        return status;
    }
    const Status dispatched = this->dispatch(image, stream);
    cudaGraph_t graph = nullptr;
    const cudaError_t ended = cudaStreamEndCapture(stream, &graph);
    if (dispatched != Status::kOk) {
        // End the capture before returning; otherwise the stream stays in capture mode.
        if (graph != nullptr) {
            static_cast<void>(cudaGraphDestroy(graph));
        }
        set_message(out_message, "failed while recording the launch sequence");
        return dispatched;
    }
    status =
            detail::check_cuda(ended, "cudaStreamEndCapture", "detector.capture_graph", -1, stream);
    if (status != Status::kOk) {
        set_message(out_message, "cannot end the capture of the launch sequence");
        return status;
    }
    // The signature of cudaGraphInstantiate changed in CUDA 12. On 11.x (11.4, which the Jetson
    // L4T R35 ships) it is the five-argument form taking an error node and a log buffer.
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
        set_message(out_message, "cannot instantiate the launch sequence");
        return status;
    }
    this->graph_stream_ = stream;
    return Status::kOk;
}

Status Detector::Impl::detect_async(const ImageViewU8& image, cudaStream_t stream,
                                    std::string* out_message) {
    if (!this->initialized_) {
        set_message(out_message, "initialize has not been called");
        return Status::kNotInitialized;
    }
    Status status = validate_image_view(image, out_message);
    if (status != Status::kOk) {
        return status;
    }
    // The kernels read the image directly, so host memory cannot be accepted.
    if (image.space_ != MemorySpace::kDevice && image.space_ != MemorySpace::kManaged) {
        set_message(out_message, "the input image must live in device or managed space");
        return Status::kInvalidImage;
    }
    if (image.width_px_ > this->config_.max_width_px_ ||
        image.height_px_ > this->config_.max_height_px_) {
        set_message(out_message, "the input exceeds max_width_px or max_height_px");
        return Status::kInvalidArgument;
    }

    const bool layout_changed = image.data_ != this->last_data_ ||
                                image.width_px_ != this->last_width_px_ ||
                                image.height_px_ != this->last_height_px_ ||
                                image.pitch_bytes_ != this->last_pitch_bytes_;
    const bool rebuild_layout = layout_changed || !this->has_result_;
    if (rebuild_layout) {
        status = this->wait_for_completion(out_message);
    } else {
        status = this->enqueue_completion_wait(stream, out_message);
    }
    if (status != Status::kOk) {
        return status;
    }
    // After this point a failed attempt must not publish the previous frame as
    // the result of this invocation; partial dispatches are tracked below.
    this->has_result_ = false;

    status = detail::plan_scales(this->config_, image.width_px_, image.height_px_, &this->plan_);
    if (status != Status::kOk) {
        return status;
    }
    if (rebuild_layout) {
        status = this->reserve_all(image, out_message);
        if (status != Status::kOk) {
            return status;
        }
#if defined(ARUCO3CUDA_PRECOMPUTED_RESIZE_COEFFICIENTS) && \
        ARUCO3CUDA_PRECOMPUTED_RESIZE_COEFFICIENTS
        // Immutable Kleidi resize coefficients depend only on layout/configuration.
        // Preparation synchronizes once here, before graph capture; no frame
        // pixels or detections are reused by subsequent submissions.
        status = detail::prepare_segmentation_resize(&this->preprocess_, this->config_, stream);
        if (status != Status::kOk) {
            this->has_result_ = false;
            this->release_graph();
            set_message(out_message, "cannot prepare segmentation resize coefficients");
            const Status tracked = this->record_completion(stream, out_message);
            return tracked == Status::kOk ? status : tracked;
        }
#endif
        // The carving changed, so the baked-in pointers are stale.
        // **This destruction cannot be dropped.** Dropping it silently dereferences the old
        // pointers and makes graph_is_rebuilt_when_layout_changes fail (confirmed by mutation).
        this->release_graph();
    }
    // Recapture preserves the graph's stream association. The owned completion
    // event above, not recapture itself, orders access to reusable frame storage.
    if (this->graph_ != nullptr && this->graph_stream_ != stream) {
        this->release_graph();
    }
    // **The default stream cannot be captured.** CUDA does not allow capturing the legacy
    // default stream, so when nullptr is passed the work is issued step by step as before.
    // Passing an explicit stream folds the launch sequence into a single launch.
    if (stream == nullptr) {
        this->release_graph();
        status = this->dispatch(image, stream);
        if (status != Status::kOk) {
            set_message(out_message, "failed to issue the kernels");
            const Status tracked = this->record_completion(stream, out_message);
            return tracked == Status::kOk ? status : tracked;
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
            set_message(out_message, "cannot launch the launch sequence");
            const Status tracked = this->record_completion(stream, out_message);
            return tracked == Status::kOk ? status : tracked;
        }
    }

    status = this->record_completion(stream, out_message);
    if (status != Status::kOk) {
        return status;
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
        set_message(out_message, "detect_async has not been called");
        return Status::kNotInitialized;
    }

    const Status ordered = this->enqueue_completion_wait(stream, out_message);
    if (ordered != Status::kOk) {
        return ordered;
    }

    // Queue the device status before read_detection_count's existing stream
    // synchronization. Never publish a partial result after an asynchronous failure.
    std::int32_t device_failure = 0;
    const Status diagnostic_status = detail::check_cuda(
            cudaMemcpyAsync(&device_failure, this->detections_.failure_bits_, sizeof(std::int32_t),
                            cudaMemcpyDeviceToHost, stream),
            "cudaMemcpyAsync", "detector.download.failure_bits", -1, stream);
    if (diagnostic_status != Status::kOk) {
        set_message(out_message, "cannot read GPU detection diagnostics");
        // Keep local host storage alive until all queued copies have drained.
        // Use the raw API here so the original checked diagnostic is preserved.
        static_cast<void>(cudaStreamSynchronize(stream));
        return diagnostic_status;
    }

    int count = 0;
    const Status count_status = detail::read_detection_count(this->detections_, &count, stream);
    if (count_status != Status::kOk && count_status != Status::kMarkerOverflow) {
        set_message(out_message, "cannot read the detection count");
        static_cast<void>(cudaStreamSynchronize(stream));
        return count_status;
    }
    if (device_failure != 0) {
        std::string message = "GPU detection failure bits=" + std::to_string(device_failure) + ":";
        if ((device_failure & detail::kContourInvalidCycle) != 0) {
            message += " invalid cycle (GPU contour consistency failure);";
        }
        if ((device_failure & detail::kContourPointOverflow) != 0) {
            message += " contour point budget exceeded;";
        }
        if ((device_failure & detail::kContourCandidateOverflow) != 0) {
            message += " candidate budget exceeded;";
        }
        if ((device_failure & detail::kContourApproximationOverflow) != 0) {
            message += " contour approximation stack budget exceeded;";
        }
        if ((device_failure &
             static_cast<std::int32_t>(DeviceFailure::kInvalidCandidateCounters)) != 0) {
            message += " inconsistent GPU candidate counters;";
        }
        set_message(out_message, message);
        constexpr std::int32_t kKnownFailures =
                detail::kContourInvalidCycle | detail::kContourPointOverflow |
                detail::kContourCandidateOverflow | detail::kContourApproximationOverflow |
                static_cast<std::int32_t>(DeviceFailure::kInvalidCandidateCounters);
        if ((device_failure &
             (detail::kContourInvalidCycle |
              static_cast<std::int32_t>(DeviceFailure::kInvalidCandidateCounters))) != 0 ||
            (device_failure & ~kKnownFailures) != 0) {
            return Status::kCudaError;
        }
        if ((device_failure &
             (detail::kContourPointOverflow | detail::kContourApproximationOverflow)) != 0) {
            return Status::kInvalidConfig;
        }
        return Status::kCandidateOverflow;
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
            static_cast<void>(cudaStreamSynchronize(stream));
            return status;
        }
        status = detail::check_cuda(
                cudaMemcpyAsync(result.rotations_.data(), this->detections_.rotations_, plane,
                                cudaMemcpyDeviceToHost, stream),
                "cudaMemcpyAsync", "detector.download.rotations", -1, stream);
        if (status != Status::kOk) {
            static_cast<void>(cudaStreamSynchronize(stream));
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
                static_cast<void>(cudaStreamSynchronize(stream));
                return status;
            }
            status = detail::check_cuda(
                    cudaMemcpyAsync(corner_y.data() + slot, this->detections_.corner_y_ + offset,
                                    total * sizeof(float), cudaMemcpyDeviceToHost, stream),
                    "cudaMemcpyAsync", "detector.download.corner_y", -1, stream);
            if (status != Status::kOk) {
                static_cast<void>(cudaStreamSynchronize(stream));
                return status;
            }
        }
        status = detail::check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize",
                                    "detector.download", -1, stream);
        if (status != Status::kOk) {
            static_cast<void>(cudaStreamSynchronize(stream));
            return status;
        }
        // Repack from the per-plane device layout into 8 elements per detection.
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
