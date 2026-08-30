// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_CORE_PREPROCESS_HPP
#define ARUCO3CUDA_CORE_PREPROCESS_HPP

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"
#include "aruco3cuda/workspace.hpp"

namespace aruco3cuda::detail {

/// Upper bound on the number of pyramid levels.
///
/// The level count is determined by log2, so even a 65536x65536 input fits
/// within 16 levels. A fixed-length bound avoids a dynamic allocation per level.
inline constexpr int kMaxPyramidLevels = 24;

/// Plan for the ArUco3 downscaling.
///
/// Implements the observed OpenCV 4.x behavior exactly as recorded in
/// [the detection pipeline design](../../docs/design/detector-pipeline.md).
///
/// Ownership: holds values only and references no external resource. It may be
///            copied and retained freely.
/// Synchronization: a plain set of values, so it carries no synchronization point.
///
/// Example input: pass 1280x720 to plan_scales() with the default configuration
/// Example output: fxfy_ = 0.3333, segmentation 427x240, level_count_ = 5
struct ScalePlan {
    /// Downscaling ratio. 1 when ArUco3 is disabled.
    double fxfy_ = 1.0;
    int segmentation_width_px_ = 0;
    int segmentation_height_px_ = 0;
    /// Total number of pyramid levels, including level 0.
    int level_count_ = 1;
    /// Level at which corner upsampling starts.
    int closest_level_index_ = 0;
};

/// An 8-bit image plane held in the workspace.
///
/// Ownership: the region data_ points at is owned by the workspace. This struct
///            holds a reference only; it neither copies nor frees. data_ becomes
///            invalid once the workspace is reset() or destroyed.
/// Synchronization: a plain set of references, so it carries no synchronization
///                  point. The contents data_ points at are not settled until the
///                  already-issued kernels complete.
///
/// Example input: the level 1 plane carved out by reserve_preprocess()
/// Example output: width_px_ = 640, height_px_ = 360, pitch_bytes_ = 640
struct ImagePlaneU8 {
    std::uint8_t* data_ = nullptr;
    int width_px_ = 0;
    int height_px_ = 0;
    std::size_t pitch_bytes_ = 0;
};

/// Reference used to hand each pyramid level to a kernel.
///
/// The pitch differs per level, so the levels must not be treated uniformly.
/// Level 0 points at the caller's input image, and its pitch is the caller's too.
///
/// Kept a POD that can be passed by value, so it can sit directly in a kernel
/// argument list.
///
/// Ownership: the regions data_ points at are owned by the caller and by the
///            workspace. This struct holds references only; it neither copies nor
///            frees.
/// Synchronization: a plain set of references, so it carries no synchronization point.
///
/// Example input: the PreprocessBuffers for a 1280x720 image
/// Example output: level_count_ = 5, width_[0] = 1280, width_[1] = 640
struct PyramidRef {
    const std::uint8_t* data_[kMaxPyramidLevels] = {};
    int width_[kMaxPyramidLevels] = {};
    int height_[kMaxPyramidLevels] = {};
    std::size_t pitch_[kMaxPyramidLevels] = {};
    int level_count_ = 0;
};

/// The full set of buffers used by preprocessing.
///
/// Level 0 points at the input itself and is not copied. A copy would cost a
/// W*H byte transfer per frame, which is not negligible even on an integrated GPU.
///
/// Ownership: the region level0_ points at is owned by the caller, and the regions
///            levels_ and segmentation_ point at are owned by the workspace. This
///            struct holds references only to both and frees neither. levels_ and
///            segmentation_ become invalid once the workspace is reset() or destroyed.
/// Synchronization: a plain set of references, so it carries no synchronization point.
///
/// Example input: the output of reserve_preprocess() for a 1280x720 input
/// Example output: level_count_ = 5, segmentation_ at 427x240
struct PreprocessBuffers {
    ImageViewU8 level0_;
    /// Level 1 and above. Index 0 corresponds to level 1.
    ImagePlaneU8 levels_[kMaxPyramidLevels];
    int level_count_ = 1;
    ImagePlaneU8 segmentation_;
};

/// Computes the downscaling ratio and the pyramid level count.
///
/// @param config Detection configuration. fxfy becomes 1 when
///               use_aruco3_detection_ is false.
/// @param width_px Input width. At least 1.
/// @param height_px Input height. At least 1.
/// @param out Receives the plan on success; left unchanged on failure. Must not
///            be nullptr.
/// @return kOk, or kInvalidArgument.
///
/// Ownership: retains none of the regions passed as arguments.
/// Synchronization: host only, so it carries no synchronization point. It calls
///                  no CUDA API.
///
/// Example input: default configuration, 1280x720
/// Example output: fxfy_ = 0.3333, segmentation 427x240
Status plan_scales(const DetectorConfig& config, int width_px, int height_px, ScalePlan* out);

/// Returns the workspace capacity the plan requires.
///
/// @param plan Downscaling plan.
/// @param width_px Input width.
/// @param height_px Input height.
/// @return Required byte count. Returns 0 on overflow.
///
/// Ownership: retains no resource.
/// Synchronization: host only, so it carries no synchronization point.
///
/// Example input: the default plan for 1280x720
/// Example output: the byte count that holds the pyramid and the segmentation image
std::size_t preprocess_workspace_bytes(const ScalePlan& plan, int width_px, int height_px);

/// Carves the preprocessing regions out of the workspace.
///
/// @param plan Downscaling plan.
/// @param input Input image. Referenced as level 0.
/// @param workspace Source of the carve-out. Owned by the caller.
/// @param out Receives the full set of buffers on success. Must not be nullptr.
/// @return kOk. kInvalidConfig when the capacity is insufficient,
///         kInvalidArgument when an argument is invalid.
///
/// Ownership: the carved-out regions stay owned by the workspace. out holds
///            references only.
/// Synchronization: host only, so it carries no synchronization point. It calls
///                  no CUDA API.
///
/// Example input: a 1280x720 input and a workspace with sufficient capacity
/// Example output: levels_ and segmentation_ receive pointers into the workspace
Status reserve_preprocess(const ScalePlan& plan, const ImageViewU8& input, Workspace& workspace,
                          PreprocessBuffers* out);

/// Builds the image pyramid.
///
/// Level 0 references the input, so it is never written. Levels 1 and above are
/// generated in order. Uses the same separable [1,4,6,4,1] kernel as OpenCV
/// `pyrDown`, with BORDER_REFLECT_101 at the borders and (sum + 128) >> 8 rounding.
///
/// @param buffers The set of buffers returned by reserve_preprocess. Must not be
///                nullptr.
/// @param config Detection configuration. Used for the block dimensions.
/// @param stream Stream to issue on. Pass nullptr to use the default stream.
/// @return kOk, or kInvalidArgument, kCudaError.
///
/// Ownership: the regions buffers points at stay owned by the workspace.
/// Synchronization: only issues kernels on the stream and performs no host
///                  synchronization. The caller synchronizes at the point where
///                  it needs the results.
///
/// Example input: buffers with level_count_ = 4
/// Example output: levels_[0] through levels_[2] are generated
Status build_pyramid_async(PreprocessBuffers* buffers, const DetectorConfig& config,
                           cudaStream_t stream);

/// Builds the segmentation image.
///
/// Candidate extraction runs on this image alone. Even when the downscaling ratio
/// is 1 the image is copied, so that the later stages can reference the same buffer.
///
/// @param plan Downscaling plan.
/// @param buffers The set of buffers returned by reserve_preprocess. Must not be
///                nullptr.
/// @param config Detection configuration. Used for the block dimensions.
/// @param stream Stream to issue on. Pass nullptr to use the default stream.
/// @return kOk, or kInvalidArgument, kCudaError.
///
/// Ownership: the regions buffers points at stay owned by the workspace.
/// Synchronization: only issues kernels on the stream and performs no host
///                  synchronization.
///
/// Example input: a plan with fxfy_ = 0.3333 and a 1280x720 input
/// Example output: segmentation_ is filled at 427x240
Status build_segmentation_async(const ScalePlan& plan, PreprocessBuffers* buffers,
                                const DetectorConfig& config, cudaStream_t stream);

/// Returns the requested level as a read-only view.
///
/// @param buffers The set of buffers.
/// @param level At least 0 and below level_count_. Out of range, a view whose
///              data_ is nullptr is returned.
/// @return The view for that level. Ownership does not transfer.
///
/// Ownership: returns a reference only. The caller does not free it.
/// Synchronization: host only, so it carries no synchronization point.
///
/// Example input: level = 0
/// Example output: a view pointing at the input itself
ImageViewU8 level_view(const PreprocessBuffers& buffers, int level);

/// Assembles the reference handed to kernels from a PreprocessBuffers.
///
/// It merely calls level_view() per level and repacks the results. Writing the
/// same assembly in every kernel that uses the levels would scatter the contract
/// that level 0 points at the input itself, so it is collected in one place.
///
/// @param buffers The preprocessing set of buffers.
/// @param out Receives the reference on success. The region stays owned by the
///            caller.
/// @return kOk. kInvalidArgument when out is nullptr or the level count is out of
///         range.
///
/// Ownership: retains none of the regions buffers points at. out holds references
///            only.
/// Synchronization: host only, so it carries no synchronization point. It calls
///                  no CUDA API.
///
/// Example input: a PreprocessBuffers with level_count_ = 5
/// Example output: kOk. out->level_count_ = 5
Status make_pyramid_ref(const PreprocessBuffers& buffers, PyramidRef* out);

}  // namespace aruco3cuda::detail

#endif  // ARUCO3CUDA_CORE_PREPROCESS_HPP
