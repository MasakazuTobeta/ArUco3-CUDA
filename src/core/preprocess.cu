// SPDX-License-Identifier: Apache-2.0
#include "preprocess.hpp"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"
#include "aruco3cuda/workspace.hpp"
#include "cuda_check.hpp"

namespace aruco3cuda::detail {
namespace {

/// Alignment that fits each workspace plane to the CUDA requirements.
constexpr std::size_t kPlaneAlignment = 256U;

/// BORDER_REFLECT_101 index reflection.
///
/// Folds back without duplicating the edge pixel. This is the default border of
/// OpenCV pyrDown; without matching it, the outermost 2 columns and 2 rows differ.
///
/// Example input: index = -1, size = 8 gives 1
/// Example input: index = 8, size = 8 gives 6
__device__ inline int reflect101(int index, int size) {
    if (size == 1) {
        return 0;
    }
    // The fold can happen repeatedly, so keep going until the index lands in range.
    while (index < 0 || index >= size) {
        if (index < 0) {
            index = -index;
        }
        if (index >= size) {
            index = 2 * (size - 1) - index;
        }
    }
    return index;
}

__device__ inline std::uint8_t load_pixel(const std::uint8_t* data, std::size_t pitch_bytes, int x,
                                          int y) {
    return data[static_cast<std::size_t>(y) * pitch_bytes + static_cast<std::size_t>(x)];
}

/// Performs one level of downscaling.
///
/// Thread-to-data mapping:
///   Thread (x, y) writes exactly one output pixel. It reads the 5x5 input
///   neighborhood centered at (2x, 2y). Since no two threads write the same
///   destination, there is no race, and neither atomics nor block-level
///   synchronization are needed.
/// Boundary conditions:
///   Threads outside the output extent do nothing. Input reads are folded back
///   with reflect101.
/// Rounding:
///   The weights sum to 256. Rounds with (sum + 128) >> 8, the same as OpenCV.
///   Intermediate values are not rounded, so the separable form and a 2D
///   convolution give identical results.
__global__ void pyr_down_kernel(const std::uint8_t* __restrict__ src, std::size_t src_pitch,
                                int src_width, int src_height, std::uint8_t* __restrict__ dst,
                                std::size_t dst_pitch, int dst_width, int dst_height) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= dst_width || y >= dst_height) {
        return;
    }
    constexpr int kWeights[5] = {1, 4, 6, 4, 1};
    const int center_x = x * 2;
    const int center_y = y * 2;

    int sum = 0;
    for (int ky = 0; ky < 5; ++ky) {
        const int sy = reflect101(center_y + ky - 2, src_height);
        int row_sum = 0;
        for (int kx = 0; kx < 5; ++kx) {
            const int sx = reflect101(center_x + kx - 2, src_width);
            row_sum += kWeights[kx] * static_cast<int>(load_pixel(src, src_pitch, sx, sy));
        }
        sum += kWeights[ky] * row_sum;
    }
    dst[static_cast<std::size_t>(y) * dst_pitch + static_cast<std::size_t>(x)] =
            static_cast<std::uint8_t>((sum + 128) >> 8);
}

/// Downscales by an arbitrary ratio with bilinear interpolation.
///
/// Reproduces OpenCV `resize` with `INTER_LINEAR`, down to the 8-bit fixed-point
/// arithmetic. Computing in single precision shifts the rounding by one gray
/// level, which can flip a pixel between black and white in the downstream
/// adaptive threshold. Keeping the preprocessing difference at zero narrows any
/// difference that appears in the later stages down to the design differences in
/// candidate extraction alone.
///
/// Fixed-point convention:
///   The coefficients are 11-bit shorts (scaled by 2048). The horizontal pass
///   accumulates in int; the vertical pass accumulates once more and then rounds
///   with (v + (1 << 21)) >> 22.
///   OpenCV rounds 1 - fx and fx independently, so the coefficients do not always
///   sum to 2048. This behavior is matched as well.
///
/// Thread-to-data mapping:
///   Thread (x, y) writes exactly one output pixel and reads a 2x2 input
///   neighborhood. Since destinations never overlap, there is no race and no
///   synchronization is needed.
/// Coordinate convention:
///   Half-pixel centers: src = (dst + 0.5) * scale - 0.5.
/// Boundary conditions:
///   Threads outside the output extent do nothing. When a read would cross an
///   edge it is clamped to the edge and the matching coefficient is set to 0. No
///   extrapolation is performed.
__global__ void resize_bilinear_kernel(const std::uint8_t* __restrict__ src, std::size_t src_pitch,
                                       int src_width, int src_height,
                                       std::uint8_t* __restrict__ dst, std::size_t dst_pitch,
                                       int dst_width, int dst_height, double scale_x,
                                       double scale_y) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= dst_width || y >= dst_height) {
        return;
    }
    // Coefficient scale. 2^11, the same as OpenCV INTER_RESIZE_COEF_SCALE.
    constexpr float kCoefScale = 2048.0F;

    // OpenCV computes (dx + 0.5) * scale - 0.5 in double precision and then
    // narrows to single. Staying in single precision shifts the rounding by one
    // gray level, so the order and the types of the operations are matched too.
    float fx = static_cast<float>((static_cast<double>(x) + 0.5) * scale_x - 0.5);
    int ix = static_cast<int>(floorf(fx));
    fx -= static_cast<float>(ix);
    if (ix < 0) {
        ix = 0;
        fx = 0.0F;
    }
    if (ix >= src_width - 1) {
        ix = src_width - 1;
        fx = 0.0F;
    }

    float fy = static_cast<float>((static_cast<double>(y) + 0.5) * scale_y - 0.5);
    int iy = static_cast<int>(floorf(fy));
    fy -= static_cast<float>(iy);
    if (iy < 0) {
        iy = 0;
        fy = 0.0F;
    }
    if (iy >= src_height - 1) {
        iy = src_height - 1;
        fy = 0.0F;
    }

    // OpenCV rounds to nearest with saturate_cast<short>.
    const int alpha1 = __float2int_rn(fx * kCoefScale);
    const int alpha0 = __float2int_rn((1.0F - fx) * kCoefScale);
    const int beta1 = __float2int_rn(fy * kCoefScale);
    const int beta0 = __float2int_rn((1.0F - fy) * kCoefScale);

    const int ix1 = (ix + 1 < src_width) ? (ix + 1) : ix;
    const int iy1 = (iy + 1 < src_height) ? (iy + 1) : iy;

    const int top = static_cast<int>(load_pixel(src, src_pitch, ix, iy)) * alpha0 +
                    static_cast<int>(load_pixel(src, src_pitch, ix1, iy)) * alpha1;
    const int bottom = static_cast<int>(load_pixel(src, src_pitch, ix, iy1)) * alpha0 +
                       static_cast<int>(load_pixel(src, src_pitch, ix1, iy1)) * alpha1;

    // The vertical combination follows the same order as the OpenCV
    // VResizeLinear<uchar, ...> specialization. Instead of the general
    // (v + (1 << 21)) >> 22, that specialization rounds in the order >> 4,
    // multiply by beta, >> 16, add, +2, >> 2. It is written for SIMD, but the
    // result differs, so failing to reproduce the order costs one gray level.
    const int value = ((beta0 * (top >> 4)) >> 16) + ((beta1 * (bottom >> 4)) >> 16) + 2;
    const int rounded = value >> 2;
    dst[static_cast<std::size_t>(y) * dst_pitch + static_cast<std::size_t>(x)] =
            static_cast<std::uint8_t>(rounded);
}

/// Plain copy. Used when the downscaling ratio is 1.
///
/// Thread-to-data mapping:
///   Thread (x, y) writes exactly one output pixel. No race occurs.
/// Boundary conditions:
///   Threads outside the output extent do nothing.
__global__ void copy_plane_kernel(const std::uint8_t* __restrict__ src, std::size_t src_pitch,
                                  std::uint8_t* __restrict__ dst, std::size_t dst_pitch, int width,
                                  int height) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) {
        return;
    }
    dst[static_cast<std::size_t>(y) * dst_pitch + static_cast<std::size_t>(x)] =
            load_pixel(src, src_pitch, x, y);
}

/// Byte count for a single plane. Returns 0 on overflow.
std::size_t plane_bytes(int width_px, int height_px) {
    if (width_px <= 0 || height_px <= 0) {
        return 0U;
    }
    const std::size_t pitch = align_up(static_cast<std::size_t>(width_px), kPlaneAlignment);
    if (pitch == 0U) {
        return 0U;
    }
    const auto height = static_cast<std::size_t>(height_px);
    if (pitch > std::numeric_limits<std::size_t>::max() / height) {
        return 0U;
    }
    return align_up(pitch * height, kPlaneAlignment);
}

/// Output size of pyrDown. Rounds up, the same as the OpenCV default.
int halve(int size) {
    return (size + 1) / 2;
}

dim3 block_dim_of(const DetectorConfig& config) {
    const auto side = static_cast<unsigned int>(config.cuda_block_dim_);
    return dim3(side, side, 1U);
}

dim3 grid_dim_for(int width, int height, dim3 block) {
    return dim3((static_cast<unsigned int>(width) + block.x - 1U) / block.x,
                (static_cast<unsigned int>(height) + block.y - 1U) / block.y, 1U);
}

}  // namespace

Status plan_scales(const DetectorConfig& config, int width_px, int height_px, ScalePlan* out) {
    if (out == nullptr || width_px < 1 || height_px < 1) {
        return Status::kInvalidArgument;
    }
    ScalePlan plan;
    if (!config.use_aruco3_detection_) {
        plan.fxfy_ = 1.0;
        plan.segmentation_width_px_ = width_px;
        plan.segmentation_height_px_ = height_px;
        plan.level_count_ = 1;
        plan.closest_level_index_ = 0;
        *out = plan;
        return Status::kOk;
    }

    // OpenCV computes fxfy in single precision. Computing in double precision can
    // make the segmentation image size differ by 1 pixel at a rounding boundary,
    // so the arithmetic types are matched for compatibility.
    const float side = static_cast<float>(config.min_side_length_canonical_img_px_);
    const float longest = static_cast<float>(width_px > height_px ? width_px : height_px);
    const float denominator = side + longest * config.min_marker_length_ratio_original_img_;
    // validate() rejects the combination where both are 0, so this cannot be 0 here.
    if (!(denominator > 0.0F)) {
        return Status::kInvalidArgument;
    }
    const float fxfy = side / denominator;
    plan.fxfy_ = static_cast<double>(fxfy);

    // OpenCV cvRound rounds half to even. std::lround rounds away from zero, so it
    // differs by 1 pixel at exactly 0.5. Use lrint instead.
    plan.segmentation_width_px_ =
            static_cast<int>(std::lrint(static_cast<double>(fxfy * static_cast<float>(width_px))));
    plan.segmentation_height_px_ =
            static_cast<int>(std::lrint(static_cast<double>(fxfy * static_cast<float>(height_px))));
    if (plan.segmentation_width_px_ < 1) {
        plan.segmentation_width_px_ = 1;
    }
    if (plan.segmentation_height_px_ < 1) {
        plan.segmentation_height_px_ = 1;
    }

    // OpenCV passes num_levels = (int)(log2(W*H / S^2) / 2) as the maxlevel of
    // buildPyramid. The level count is that value plus 1.
    const double image_area = static_cast<double>(width_px) * static_cast<double>(height_px);
    const double min_area = side * side;
    int max_level = 0;
    if (min_area > 0.0 && image_area > min_area) {
        max_level = static_cast<int>(std::log2(image_area / min_area) / 2.0);
    }
    if (max_level < 0) {
        max_level = 0;
    }
    if (max_level > kMaxPyramidLevels - 1) {
        max_level = kMaxPyramidLevels - 1;
    }
    plan.level_count_ = max_level + 1;

    const double scaled_area = image_area * plan.fxfy_ * plan.fxfy_;
    int closest = 0;
    if (scaled_area > 0.0) {
        // Here too, use the same round-half-to-even as cvRound.
        closest = static_cast<int>(std::lrint(std::log2(image_area / scaled_area) / 2.0));
    }
    if (closest < 0) {
        closest = 0;
    }
    if (closest > plan.level_count_ - 1) {
        closest = plan.level_count_ - 1;
    }
    plan.closest_level_index_ = closest;

    *out = plan;
    return Status::kOk;
}

std::size_t preprocess_workspace_bytes(const ScalePlan& plan, int width_px, int height_px) {
    if (width_px < 1 || height_px < 1 || plan.level_count_ < 1) {
        return 0U;
    }
    std::size_t total = 0U;
    int level_width = width_px;
    int level_height = height_px;
    // Level 0 references the input, so nothing is allocated for it.
    for (int level = 1; level < plan.level_count_; ++level) {
        level_width = halve(level_width);
        level_height = halve(level_height);
        const std::size_t bytes = plane_bytes(level_width, level_height);
        if (bytes == 0U || total > std::numeric_limits<std::size_t>::max() - bytes) {
            return 0U;
        }
        total += bytes;
    }
    const std::size_t segmentation =
            plane_bytes(plan.segmentation_width_px_, plan.segmentation_height_px_);
    if (segmentation == 0U || total > std::numeric_limits<std::size_t>::max() - segmentation) {
        return 0U;
    }
    return total + segmentation;
}

Status reserve_preprocess(const ScalePlan& plan, const ImageViewU8& input, Workspace& workspace,
                          PreprocessBuffers* out) {
    if (out == nullptr || plan.level_count_ < 1 || plan.level_count_ > kMaxPyramidLevels) {
        return Status::kInvalidArgument;
    }
    const Status image_status = validate_image_view(input, nullptr);
    if (image_status != Status::kOk) {
        return image_status;
    }

    PreprocessBuffers buffers;
    buffers.level0_ = input;
    buffers.level_count_ = plan.level_count_;

    int level_width = input.width_px_;
    int level_height = input.height_px_;
    for (int level = 1; level < plan.level_count_; ++level) {
        level_width = halve(level_width);
        level_height = halve(level_height);
        const std::size_t pitch = align_up(static_cast<std::size_t>(level_width), kPlaneAlignment);
        void* pointer = nullptr;
        const Status status = workspace.allocate(pitch * static_cast<std::size_t>(level_height),
                                                 kPlaneAlignment, &pointer);
        if (status != Status::kOk) {
            return status;
        }
        ImagePlaneU8& plane = buffers.levels_[level - 1];
        plane.data_ = static_cast<std::uint8_t*>(pointer);
        plane.width_px_ = level_width;
        plane.height_px_ = level_height;
        plane.pitch_bytes_ = pitch;
    }

    const std::size_t segmentation_pitch =
            align_up(static_cast<std::size_t>(plan.segmentation_width_px_), kPlaneAlignment);
    void* segmentation_pointer = nullptr;
    const Status segmentation_status = workspace.allocate(
            segmentation_pitch * static_cast<std::size_t>(plan.segmentation_height_px_),
            kPlaneAlignment, &segmentation_pointer);
    if (segmentation_status != Status::kOk) {
        return segmentation_status;
    }
    buffers.segmentation_.data_ = static_cast<std::uint8_t*>(segmentation_pointer);
    buffers.segmentation_.width_px_ = plan.segmentation_width_px_;
    buffers.segmentation_.height_px_ = plan.segmentation_height_px_;
    buffers.segmentation_.pitch_bytes_ = segmentation_pitch;

    *out = buffers;
    return Status::kOk;
}

ImageViewU8 level_view(const PreprocessBuffers& buffers, int level) {
    ImageViewU8 view;
    if (level < 0 || level >= buffers.level_count_) {
        return view;
    }
    if (level == 0) {
        return buffers.level0_;
    }
    const ImagePlaneU8& plane = buffers.levels_[level - 1];
    view.data_ = plane.data_;
    view.width_px_ = plane.width_px_;
    view.height_px_ = plane.height_px_;
    view.pitch_bytes_ = plane.pitch_bytes_;
    view.space_ = buffers.level0_.space_;
    return view;
}

Status make_pyramid_ref(const PreprocessBuffers& buffers, PyramidRef* out) {
    if (out == nullptr || buffers.level_count_ < 1 || buffers.level_count_ > kMaxPyramidLevels) {
        return Status::kInvalidArgument;
    }
    PyramidRef pyramid{};
    pyramid.level_count_ = buffers.level_count_;
    for (int level = 0; level < buffers.level_count_; ++level) {
        const ImageViewU8 view = level_view(buffers, level);
        pyramid.data_[level] = view.data_;
        pyramid.width_[level] = view.width_px_;
        pyramid.height_[level] = view.height_px_;
        pyramid.pitch_[level] = view.pitch_bytes_;
    }
    *out = pyramid;
    return Status::kOk;
}

Status build_pyramid_async(PreprocessBuffers* buffers, const DetectorConfig& config,
                           cudaStream_t stream) {
    if (buffers == nullptr || buffers->level_count_ < 1 ||
        buffers->level_count_ > kMaxPyramidLevels) {
        return Status::kInvalidArgument;
    }
    if (buffers->level0_.data_ == nullptr) {
        return Status::kInvalidArgument;
    }
    const dim3 block = block_dim_of(config);

    for (int level = 1; level < buffers->level_count_; ++level) {
        const ImageViewU8 source = level_view(*buffers, level - 1);
        ImagePlaneU8& destination = buffers->levels_[level - 1];
        if (source.data_ == nullptr || destination.data_ == nullptr) {
            return Status::kInvalidArgument;
        }
        const dim3 grid = grid_dim_for(destination.width_px_, destination.height_px_, block);
        pyr_down_kernel<<<grid, block, 0, stream>>>(source.data_, source.pitch_bytes_,
                                                    source.width_px_, source.height_px_,
                                                    destination.data_, destination.pitch_bytes_,
                                                    destination.width_px_, destination.height_px_);
        // Check for a launch failure per level. Checking them all at once would
        // hide which level failed. Synchronization is the caller's responsibility
        // and is not done here.
        const Status status =
                detail::check_kernel_launch("preprocess.pyr_down_kernel", -1, false, stream);
        if (status != Status::kOk) {
            return status;
        }
    }
    return Status::kOk;
}

Status build_segmentation_async(const ScalePlan& plan, PreprocessBuffers* buffers,
                                const DetectorConfig& config, cudaStream_t stream) {
    if (buffers == nullptr || buffers->segmentation_.data_ == nullptr ||
        buffers->level0_.data_ == nullptr) {
        return Status::kInvalidArgument;
    }
    const ImageViewU8 source = buffers->level0_;
    const ImagePlaneU8& destination = buffers->segmentation_;
    const dim3 block = block_dim_of(config);
    const dim3 grid = grid_dim_for(destination.width_px_, destination.height_px_, block);

    if (destination.width_px_ == source.width_px_ && destination.height_px_ == source.height_px_) {
        // Downscaling ratio of 1. Copy so that the later stages can reference the
        // same buffer.
        copy_plane_kernel<<<grid, block, 0, stream>>>(
                source.data_, source.pitch_bytes_, destination.data_, destination.pitch_bytes_,
                destination.width_px_, destination.height_px_);
    } else {
        // OpenCV derives inv_scale = dst / src and then takes scale = 1 / inv_scale.
        // Computing src / dst directly does not match in floating point, and some
        // pixels end up with an interpolation coefficient off by one unit, so the
        // order of the divisions is matched as well.
        const double inverse_scale_x =
                static_cast<double>(destination.width_px_) / static_cast<double>(source.width_px_);
        const double inverse_scale_y = static_cast<double>(destination.height_px_) /
                                       static_cast<double>(source.height_px_);
        const double scale_x = 1.0 / inverse_scale_x;
        const double scale_y = 1.0 / inverse_scale_y;
        resize_bilinear_kernel<<<grid, block, 0, stream>>>(
                source.data_, source.pitch_bytes_, source.width_px_, source.height_px_,
                destination.data_, destination.pitch_bytes_, destination.width_px_,
                destination.height_px_, scale_x, scale_y);
    }
    (void)plan;
    return detail::check_kernel_launch("preprocess.segmentation", -1, false, stream);
}

}  // namespace aruco3cuda::detail
