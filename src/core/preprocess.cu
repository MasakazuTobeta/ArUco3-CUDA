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
#include "candidate_filter.hpp"
#include "cuda_check.hpp"
#include "pyramid_sampling.cuh"

namespace aruco3cuda::detail {
namespace {

/// Alignment that fits each workspace plane to the CUDA requirements.
constexpr std::size_t kPlaneAlignment = 256U;

int arm_resize_mode(int source_width, int source_height, int width, int height) {
    if (source_width < 1 || source_height < 1 || width < 1 || height < 1 ||
        (source_width == width && source_height == height))
        return 0;
#if ARUCO3CUDA_RESIZE_PROFILE_OPENCV_ARM_HAL
    const double scale_x = 1.0 / (static_cast<double>(width) / source_width);
    const double scale_y = 1.0 / (static_cast<double>(height) / source_height);
#endif
    int arm_mode = 0;
#if ARUCO3CUDA_RESIZE_PROFILE_OPENCV_ARM_HAL
    const bool quarter = static_cast<long long>(source_width) == 2LL * width &&
                         static_cast<long long>(source_height) == 2LL * height;
    const bool kleidi =
            static_cast<long long>(width) * 3 >= source_width && width < source_width &&
            height < source_height && width >= 8 &&
            source_width >= (static_cast<long long>(width) * 2 >= source_width ? 16 : 32);
    if (!quarter && kleidi) {
        arm_mode = 1;
    } else if (!quarter && source_width >= 8 && width >= 8 && height >= 8) {
        arm_mode = source_width >= 16 && static_cast<float>(scale_x) <= 2.0F &&
                                   static_cast<float>(scale_y) <= 2.0F
                           ? 3
                           : 2;
    }
#endif
    return arm_mode;
}

bool resize_tables_match(const PreprocessBuffers& b, int mode) {
    return (mode == 1 || b.resize_all_modes_) && b.resize_coefficients_ready_ &&
           b.resize_x_ != nullptr && b.resize_y_ != nullptr &&
           b.resize_x_capacity_ >= b.segmentation_.width_px_ &&
           b.resize_y_capacity_ >= b.segmentation_.height_px_ &&
           b.resize_cached_source_width_ == b.level0_.width_px_ &&
           b.resize_cached_source_height_ == b.level0_.height_px_ &&
           b.resize_cached_width_ == b.segmentation_.width_px_ &&
           b.resize_cached_height_ == b.segmentation_.height_px_ && b.resize_cached_mode_ == mode &&
           b.resize_cached_sve_bytes_ == b.resize_sve_vector_bytes_ &&
           b.resize_cached_carotene_x_fma_ == b.resize_carotene_x_fma_;
}

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
                                std::size_t dst_pitch, int dst_width, int dst_height,
                                const std::int32_t* required_level, int level) {
    if (required_level != nullptr && level > *required_level) {
        return;
    }
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

constexpr int kDemandThreads = 256;

/// One block reduces current candidate demand, then replaces the previous frame's
/// scalar. Refinement can read levels start-1 down to zero, but only if candidates
/// exist. Every thread participates in the reduction, including an empty frame.
__global__ void pyramid_demand_kernel(PyramidRef pyramid, const std::int32_t* perimeter,
                                      const std::int32_t* count, int capacity,
                                      int segmentation_width, int min_perimeter,
                                      int refinement_level, bool use_aruco3,
                                      std::int32_t* required_level) {
    __shared__ int demand[kDemandThreads];
    const int candidates = min(max(*count, 0), capacity);
    int required = (candidates > 0) ? refinement_level : 0;
    for (int candidate = static_cast<int>(threadIdx.x); candidate < candidates;
         candidate += kDemandThreads) {
        if (use_aruco3) {
            required = max(required, choose_canonical_level(pyramid, segmentation_width,
                                                            perimeter[candidate], min_perimeter));
        }
    }
    demand[threadIdx.x] = required;
    __syncthreads();
    for (int offset = kDemandThreads / 2; offset > 0; offset /= 2) {
        if (static_cast<int>(threadIdx.x) < offset) {
            demand[threadIdx.x] = max(demand[threadIdx.x], demand[threadIdx.x + offset]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0U) {
        *required_level = demand[0];
    }
}

// Scalar equivalents of the pinned OpenCV ARM HAL interpolation arithmetic.
// See preprocess.md and NOTICE for Carotene BSD-3-Clause / KleidiCV Apache-2.0 provenance.
__device__ unsigned long long rounded_divide(unsigned long long numerator, int denominator) {
    return (numerator + static_cast<unsigned int>(denominator) / 2U) /
           static_cast<unsigned int>(denominator);
}

__device__ unsigned long long aligned_fixed_coordinate(int position, int source, int destination) {
    // Split before scaling: ((2*position+1)*source)<<15 can overflow even when
    // each image dimension is representable. Quotient/remainder preserves the
    // exact rounded fixed-point coordinate without that intermediate overflow.
    const auto numerator =
            (2ULL * static_cast<unsigned int>(position) + 1ULL) * static_cast<unsigned int>(source);
    const auto denominator = static_cast<unsigned int>(destination);
    return (numerator / denominator) * 32768ULL +
           rounded_divide((numerator % denominator) * 32768ULL, destination) - 32768ULL + 128ULL;
}

/// Match the vector-coordinate rounding schedule, including NEON's overlapping tail.
__device__ unsigned long long kleidi_x_coordinate(int x, int source, int destination,
                                                  int sve_vector_bytes) {
    const int lanes = sve_vector_bytes > 0 ? sve_vector_bytes : 16;
    const auto step =
            rounded_divide((static_cast<unsigned long long>(source) * lanes) << 16U, destination);
    const int full = (destination / 32) * 32;
    if (sve_vector_bytes == 0 && destination % 32 != 0 && x >= min(full, destination - 8)) {
        const int halves = (destination - full + 7) / 8;
        const int half = x >= destination - 8 ? halves - 1 : (x - full) / 8;
        const int nominal = full + half * 8;
        const int start = min(nominal, destination - 8);
        const auto half_step =
                rounded_divide((static_cast<unsigned long long>(source) * 8U) << 16U, destination);
        const auto pixel_step =
                rounded_divide(static_cast<unsigned long long>(source) << 16U, destination);
        return aligned_fixed_coordinate(full, source, destination) + half * half_step -
               (nominal - start) * pixel_step +
               rounded_divide((static_cast<unsigned long long>(x - start) * source) << 16U,
                              destination);
    }
    const int block = x / lanes;
    const int recalibrated = (block / 512) * 512 * lanes;
    return aligned_fixed_coordinate(recalibrated, source, destination) + (block % 512) * step +
           rounded_divide((static_cast<unsigned long long>(x % lanes) * source) << 16U,
                          destination);
}

/// One thread writes one destination pixel; all source reads clamp to the image.
/// Threads outside the destination return before reading, and no workspace is shared.
/// mode 1: KleidiCV rounded 8-bit vertical/horizontal interpolation.
/// mode 2: Carotene rounded 7-bit vertical/horizontal interpolation.
/// mode 3: Carotene 11-bit downscale with the generic OpenCV vertical truncation order.
__global__ void resize_arm_hal_kernel(const std::uint8_t* src, std::size_t src_pitch, int src_width,
                                      int src_height, std::uint8_t* dst, std::size_t dst_pitch,
                                      int dst_width, int dst_height, float scale_x, float scale_y,
                                      int mode, int sve_vector_bytes, bool carotene_x_fma) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= dst_width || y >= dst_height)
        return;
    int ix = 0;
    int iy = 0;
    int wx = 0;
    int wy = 0;
    int denominator = 0;
    if (mode == 1) {
        const auto fx = kleidi_x_coordinate(x, src_width, dst_width, sve_vector_bytes);
        const auto fy = aligned_fixed_coordinate(y, src_height, dst_height);
        ix = static_cast<int>(fx >> 16U);
        iy = static_cast<int>(fy >> 16U);
        wx = static_cast<int>((fx & 65535U) >> 8U);
        wy = static_cast<int>((fy & 65535U) >> 8U);
        denominator = 256;
    } else {
        // The pinned Carotene build determines whether its horizontal NEON
        // intrinsic contracts. Scalar vertical and fast coordinates use FMA.
        const float offset_x = __fadd_rn(__fmul_rn(scale_x, 0.5F), -0.5F);
        const float fx = mode == 2 && !carotene_x_fma
                                 ? __fadd_rn(__fmul_rn(static_cast<float>(x), scale_x), offset_x)
                                 : __fmaf_rn(static_cast<float>(x), scale_x, offset_x);
        const float fy = __fmaf_rn(static_cast<float>(y), scale_y, scale_y * 0.5F - 0.5F);
        ix = static_cast<int>(floorf(fx));
        iy = static_cast<int>(floorf(fy));
        denominator = mode == 2 ? 128 : 2048;
        const float round = mode == 2 ? 0.0F : 0.5F;
        const float left =
                __fmul_rn(static_cast<float>(ix + 1) - fx, static_cast<float>(denominator));
        const float top =
                __fmul_rn(static_cast<float>(iy + 1) - fy, static_cast<float>(denominator));
        wx = denominator - static_cast<int>(floorf(__fadd_rn(left, round)));
        wy = denominator - static_cast<int>(floorf(__fadd_rn(top, round)));
    }
    const int x0 = max(0, min(ix, src_width - 1));
    const int x1 = max(0, min(ix + 1, src_width - 1));
    const int y0 = max(0, min(iy, src_height - 1));
    const int y1 = max(0, min(iy + 1, src_height - 1));
    const int a = load_pixel(src, src_pitch, x0, y0);
    const int b = load_pixel(src, src_pitch, x1, y0);
    const int c = load_pixel(src, src_pitch, x0, y1);
    const int d = load_pixel(src, src_pitch, x1, y1);
    int value = 0;
    if (mode == 3) {
        const int top = a * (2048 - wx) + b * wx;
        const int bottom = c * (2048 - wx) + d * wx;
        value = ((((2048 - wy) * (top >> 4)) >> 16) + ((wy * (bottom >> 4)) >> 16) + 2) >> 2;
    } else {
        const int left = (a * (denominator - wy) + c * wy + denominator / 2) / denominator;
        const int right = (b * (denominator - wy) + d * wy + denominator / 2) / denominator;
        value = (left * (denominator - wx) + right * wx + denominator / 2) / denominator;
    }
    dst[static_cast<std::size_t>(y) * dst_pitch + static_cast<std::size_t>(x)] =
            static_cast<std::uint8_t>(value);
}

// Geometry only: no image address is passed to this initialization kernel.
__device__ ResizeAxisCoefficients resize_axis_coefficients(int position, int source, int target,
                                                           float scale, int mode, bool horizontal,
                                                           int sve_bytes, bool x_fma) {
    int index = 0;
    int weight1 = 0;
    const int denominator = mode == 1 ? 256 : (mode == 2 ? 128 : 2048);
    if (mode == 1) {
        const auto fixed = horizontal ? kleidi_x_coordinate(position, source, target, sve_bytes)
                                      : aligned_fixed_coordinate(position, source, target);
        index = static_cast<int>(fixed >> 16U);
        weight1 = static_cast<int>((fixed & 65535U) >> 8U);
    } else {
        const float offset = __fadd_rn(__fmul_rn(scale, 0.5F), -0.5F);
        const float coordinate =
                mode == 2 && horizontal && !x_fma
                        ? __fadd_rn(__fmul_rn(static_cast<float>(position), scale), offset)
                        : __fmaf_rn(static_cast<float>(position), scale, offset);
        index = static_cast<int>(floorf(coordinate));
        const float complement = __fmul_rn(static_cast<float>(index + 1) - coordinate,
                                           static_cast<float>(denominator));
        weight1 = denominator -
                  static_cast<int>(floorf(__fadd_rn(complement, mode == 2 ? 0.0F : 0.5F)));
    }
    return {max(0, min(index, source - 1)), max(0, min(index + 1, source - 1)),
            denominator - weight1, weight1};
}

// Each thread owns one element in each axis array; there are no shared writes.
__global__ void initialize_resize_coefficients_kernel(ResizeAxisCoefficients* x,
                                                      ResizeAxisCoefficients* y, int source_width,
                                                      int source_height, int width, int height,
                                                      float scale_x, float scale_y, int mode,
                                                      int sve_bytes, bool x_fma) {
    const unsigned int position = blockIdx.x * blockDim.x + threadIdx.x;
    if (position < static_cast<unsigned int>(width))
        x[position] = resize_axis_coefficients(static_cast<int>(position), source_width, width,
                                               scale_x, mode, true, sve_bytes, x_fma);
    if (position < static_cast<unsigned int>(height))
        y[position] = resize_axis_coefficients(static_cast<int>(position), source_height, height,
                                               scale_y, mode, false, sve_bytes, x_fma);
}

// One thread reads immutable geometry and four CURRENT-frame pixels, then writes
// one output pixel. Coefficients contain no image values or frame-dependent state.
__global__ void resize_from_coefficients_kernel(const std::uint8_t* source,
                                                std::size_t source_pitch, std::uint8_t* destination,
                                                std::size_t destination_pitch, int width,
                                                int height, const ResizeAxisCoefficients* xs,
                                                const ResizeAxisCoefficients* ys, int mode) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height)
        return;
    const ResizeAxisCoefficients horizontal = xs[x];
    const ResizeAxisCoefficients vertical = ys[y];
    const int a = load_pixel(source, source_pitch, horizontal.source0_, vertical.source0_);
    const int b = load_pixel(source, source_pitch, horizontal.source1_, vertical.source0_);
    const int c = load_pixel(source, source_pitch, horizontal.source0_, vertical.source1_);
    const int d = load_pixel(source, source_pitch, horizontal.source1_, vertical.source1_);
    int value = 0;
    if (mode == 3) {
        const int top = a * horizontal.weight0_ + b * horizontal.weight1_;
        const int bottom = c * horizontal.weight0_ + d * horizontal.weight1_;
        value = (((vertical.weight0_ * (top >> 4)) >> 16) +
                 ((vertical.weight1_ * (bottom >> 4)) >> 16) + 2) >>
                2;
    } else {
        const int shift = mode == 1 ? 8 : 7;
        const int rounding = 1 << (shift - 1);
        const int left = (a * vertical.weight0_ + c * vertical.weight1_ + rounding) >> shift;
        const int right = (b * vertical.weight0_ + d * vertical.weight1_ + rounding) >> shift;
        value = (left * horizontal.weight0_ + right * horizontal.weight1_ + rounding) >> shift;
    }
    destination[static_cast<std::size_t>(y) * destination_pitch + x] =
            static_cast<std::uint8_t>(value);
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
///   accumulates in int. The specialized vertical path shifts horizontal sums
///   by 4, multiplies each beta, shifts by 16, adds 2 and shifts by 2.
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
    std::size_t total = kPlaneAlignment;
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
    total += segmentation;
#if ARUCO3CUDA_RESIZE_PROFILE_OPENCV_ARM_HAL
    // Maximum-layout sizing must also cover a nearby smaller image whose HAL
    // dispatch changes from the table-free quarter/copy case to a table path.
    {
        for (int length : {plan.segmentation_width_px_, plan.segmentation_height_px_}) {
            const auto size = static_cast<std::size_t>(length);
            if (size > (std::numeric_limits<std::size_t>::max() - kPlaneAlignment) /
                               sizeof(ResizeAxisCoefficients))
                return 0U;
            const auto bytes = align_up(size * sizeof(ResizeAxisCoefficients), kPlaneAlignment);
            if (total > std::numeric_limits<std::size_t>::max() - bytes)
                return 0U;
            total += bytes;
        }
    }
#endif
    return total;
}

Status reserve_preprocess(const ScalePlan& plan, const ImageViewU8& input, Workspace& workspace,
                          PreprocessBuffers* out, bool all_resize_modes) {
    if (out == nullptr || plan.level_count_ < 1 || plan.level_count_ > kMaxPyramidLevels) {
        return Status::kInvalidArgument;
    }
    const Status image_status = validate_image_view(input, nullptr);
    if (image_status != Status::kOk) {
        return image_status;
    }

    PreprocessBuffers buffers;
    buffers.level0_ = input;
    buffers.resize_all_modes_ = all_resize_modes;
    buffers.resize_sve_vector_bytes_ = aruco3cuda::resize_sve_vector_bytes();
    buffers.resize_carotene_x_fma_ = aruco3cuda::resize_carotene_x_fma();
    buffers.level_count_ = plan.level_count_;
    void* required_level_pointer = nullptr;
    const Status demand_status =
            workspace.allocate(kPlaneAlignment, kPlaneAlignment, &required_level_pointer);
    if (demand_status != Status::kOk) {
        return demand_status;
    }
    buffers.required_level_ = static_cast<std::int32_t*>(required_level_pointer);

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
    const int resize_mode =
            arm_resize_mode(input.width_px_, input.height_px_, plan.segmentation_width_px_,
                            plan.segmentation_height_px_);
    if (resize_mode == 1 || (all_resize_modes && resize_mode != 0)) {
        void* x = nullptr;
        void* y = nullptr;
        Status status = workspace.allocate(static_cast<std::size_t>(plan.segmentation_width_px_) *
                                                   sizeof(ResizeAxisCoefficients),
                                           kPlaneAlignment, &x);
        if (status != Status::kOk)
            return status;
        status = workspace.allocate(static_cast<std::size_t>(plan.segmentation_height_px_) *
                                            sizeof(ResizeAxisCoefficients),
                                    kPlaneAlignment, &y);
        if (status != Status::kOk)
            return status;
        buffers.resize_x_ = static_cast<ResizeAxisCoefficients*>(x);
        buffers.resize_y_ = static_cast<ResizeAxisCoefficients*>(y);
        buffers.resize_x_capacity_ = plan.segmentation_width_px_;
        buffers.resize_y_capacity_ = plan.segmentation_height_px_;
    }

    *out = buffers;
    return Status::kOk;
}

Status prepare_segmentation_resize(PreprocessBuffers* buffers, const DetectorConfig& config,
                                   cudaStream_t stream) {
    if (buffers == nullptr)
        return Status::kInvalidArgument;
    const int width = buffers->segmentation_.width_px_;
    const int height = buffers->segmentation_.height_px_;
    if (buffers->level0_.width_px_ < 1 || buffers->level0_.height_px_ < 1 || width < 1 ||
        height < 1 || config.cuda_block_dim_ < 1 || config.cuda_block_dim_ > 32) {
        buffers->resize_coefficients_ready_ = false;
        return Status::kInvalidArgument;
    }
    const int mode =
            arm_resize_mode(buffers->level0_.width_px_, buffers->level0_.height_px_, width, height);
    if (mode == 0 || (mode != 1 && !buffers->resize_all_modes_)) {
        buffers->resize_coefficients_ready_ = false;
        return Status::kOk;
    }
    if (resize_tables_match(*buffers, mode))
        return Status::kOk;
    buffers->resize_coefficients_ready_ = false;
    if (buffers->resize_x_ == nullptr || buffers->resize_y_ == nullptr ||
        buffers->resize_x_capacity_ < width || buffers->resize_y_capacity_ < height)
        return Status::kInvalidArgument;
    cudaStreamCaptureStatus capture;
    Status status = check_cuda(cudaStreamIsCapturing(stream, &capture), "cudaStreamIsCapturing",
                               "preprocess.prepare_resize", -1, stream);
    if (status != Status::kOk)
        return status;
    if (capture != cudaStreamCaptureStatusNone)
        return Status::kInvalidArgument;
    const unsigned int threads =
            static_cast<unsigned int>(config.cuda_block_dim_ * config.cuda_block_dim_);
    const unsigned int length = static_cast<unsigned int>(width > height ? width : height);
    const unsigned int blocks = (length + threads - 1U) / threads;
    const float scale_x =
            static_cast<float>(1.0 / (static_cast<double>(width) / buffers->level0_.width_px_));
    const float scale_y =
            static_cast<float>(1.0 / (static_cast<double>(height) / buffers->level0_.height_px_));
    initialize_resize_coefficients_kernel<<<blocks, threads, 0, stream>>>(
            buffers->resize_x_, buffers->resize_y_, buffers->level0_.width_px_,
            buffers->level0_.height_px_, width, height, scale_x, scale_y, mode,
            buffers->resize_sve_vector_bytes_, buffers->resize_carotene_x_fma_);
    status = check_kernel_launch("preprocess.initialize_resize_coefficients", -1, false, stream);
    if (status != Status::kOk)
        return status;
    status = check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize",
                        "preprocess.prepare_resize", -1, stream);
    if (status != Status::kOk)
        return status;
    buffers->resize_cached_source_width_ = buffers->level0_.width_px_;
    buffers->resize_cached_source_height_ = buffers->level0_.height_px_;
    buffers->resize_cached_width_ = width;
    buffers->resize_cached_height_ = height;
    buffers->resize_cached_mode_ = mode;
    buffers->resize_cached_sve_bytes_ = buffers->resize_sve_vector_bytes_;
    buffers->resize_cached_carotene_x_fma_ = buffers->resize_carotene_x_fma_;
    buffers->resize_coefficients_ready_ = true;
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
        pyr_down_kernel<<<grid, block, 0, stream>>>(
                source.data_, source.pitch_bytes_, source.width_px_, source.height_px_,
                destination.data_, destination.pitch_bytes_, destination.width_px_,
                destination.height_px_, nullptr, level);
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

Status build_demanded_pyramid_async(PreprocessBuffers* buffers, const ScalePlan& plan,
                                    const DeviceCandidates& candidates,
                                    const DetectorConfig& config, cudaStream_t stream) {
    if (buffers == nullptr || buffers->level_count_ < 1 ||
        buffers->level_count_ > kMaxPyramidLevels || buffers->level0_.data_ == nullptr ||
        buffers->required_level_ == nullptr || candidates.perimeter_ == nullptr ||
        candidates.count_ == nullptr || candidates.capacity_ < 1 ||
        plan.segmentation_width_px_ < 1) {
        return Status::kInvalidArgument;
    }
    if (plan.closest_level_index_ < 0 || plan.closest_level_index_ >= buffers->level_count_) {
        return Status::kInvalidConfig;
    }
    if (buffers->level_count_ == 1) {
        // There is no generated pyramid to read, so no demand reduction is needed.
        return Status::kOk;
    }
    PyramidRef pyramid{};
    Status status = make_pyramid_ref(*buffers, &pyramid);
    if (status != Status::kOk) {
        return status;
    }
    const int refinement_level =
            config.corner_refine_method_ == CornerRefineMethod::kSubpix
                    ? ((plan.closest_level_index_ > 0) ? plan.closest_level_index_ - 1 : 0)
                    : 0;
    pyramid_demand_kernel<<<1U, static_cast<unsigned int>(kDemandThreads), 0, stream>>>(
            pyramid, candidates.perimeter_, candidates.count_, candidates.capacity_,
            plan.segmentation_width_px_, 4 * config.min_side_length_canonical_img_px_,
            refinement_level, config.use_aruco3_detection_, buffers->required_level_);
    status = check_kernel_launch("preprocess.pyramid_demand_kernel", -1, false, stream);
    if (status != Status::kOk) {
        return status;
    }
    const dim3 block = block_dim_of(config);
    for (int level = 1; level < buffers->level_count_; ++level) {
        const ImageViewU8 source = level_view(*buffers, level - 1);
        ImagePlaneU8& destination = buffers->levels_[level - 1];
        if (source.data_ == nullptr || destination.data_ == nullptr) {
            return Status::kInvalidArgument;
        }
        const dim3 grid = grid_dim_for(destination.width_px_, destination.height_px_, block);
        pyr_down_kernel<<<grid, block, 0, stream>>>(
                source.data_, source.pitch_bytes_, source.width_px_, source.height_px_,
                destination.data_, destination.pitch_bytes_, destination.width_px_,
                destination.height_px_, buffers->required_level_, level);
        status = check_kernel_launch("preprocess.demanded_pyr_down_kernel", -1, false, stream);
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
        const int arm_mode = arm_resize_mode(source.width_px_, source.height_px_,
                                             destination.width_px_, destination.height_px_);
        if (arm_mode == 0) {
            resize_bilinear_kernel<<<grid, block, 0, stream>>>(
                    source.data_, source.pitch_bytes_, source.width_px_, source.height_px_,
                    destination.data_, destination.pitch_bytes_, destination.width_px_,
                    destination.height_px_, scale_x, scale_y);
        } else if (resize_tables_match(*buffers, arm_mode)) {
            resize_from_coefficients_kernel<<<grid, block, 0, stream>>>(
                    source.data_, source.pitch_bytes_, destination.data_, destination.pitch_bytes_,
                    destination.width_px_, destination.height_px_, buffers->resize_x_,
                    buffers->resize_y_, arm_mode);
        } else {
            resize_arm_hal_kernel<<<grid, block, 0, stream>>>(
                    source.data_, source.pitch_bytes_, source.width_px_, source.height_px_,
                    destination.data_, destination.pitch_bytes_, destination.width_px_,
                    destination.height_px_, static_cast<float>(scale_x),
                    static_cast<float>(scale_y), arm_mode, buffers->resize_sve_vector_bytes_,
                    buffers->resize_carotene_x_fma_);
        }
    }
    (void)plan;
    return detail::check_kernel_launch("preprocess.segmentation", -1, false, stream);
}

}  // namespace aruco3cuda::detail
