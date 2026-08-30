// SPDX-License-Identifier: Apache-2.0
#include "corner_refine.hpp"

#include <cuda_runtime_api.h>

#include <cfloat>
#include <cstddef>
#include <cstdint>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/workspace.hpp"
#include "cuda_check.hpp"
#include "detection_emit.hpp"
#include "preprocess.hpp"
#include "quad_extract.hpp"

namespace aruco3cuda::detail {
namespace {

constexpr std::size_t kPlaneAlignment = 256U;
/// Thread count of the block that handles one corner.
///
/// One iteration touches at most 13x13 = 169 patch elements and 11x11 = 121 gradient elements.
/// This is the smallest multiple of the warp size that covers 169 in a single pass.
constexpr int kRefineThreads = 192;
/// Lower and upper bound on the number of blocks launched.
///
/// The block count should be the smaller of "how many the device can run concurrently" and "how
/// much work there is"; the upper bound comes from the latter. Covering 16 markers = 64 corners,
/// the upper bound of the evaluation plan, in a single wave is enough, and launching more only
/// adds blocks with no corner to process. One block consumes about 5.5 KB of shared memory, so
/// the launch cost of every block is paid even on a frame with no detections.
constexpr int kMinRefineBlocks = 32;
constexpr int kMaxRefineBlocks = 64;
/// Upper bound OpenCV applies to the iteration count inside cornerSubPix.
constexpr int kMaxIterations = 100;
/// Level size at which the window radius switches.
constexpr int kLargeLevelSidePx = 1080;
/// Window radii used for large and small levels.
constexpr int kLargeWinRadius = 5;
constexpr int kSmallWinRadius = 3;
/// Upper bound on the window side length. 11 when the radius is 5.
constexpr int kMaxWinSide = (kLargeWinRadius * 2) + 1;

/// Window weights and the dimensions derived from them.
///
/// OpenCV evaluates exp inside cornerSubPix on every call. There are only two radii, 3 and 5, so
/// the weights are built on the host and passed by value.
struct RefineMask {
    float weight_[kMaxRefinePatchSide * kMaxRefinePatchSide] = {};
    int radius_ = 0;
    /// Window side length. 2 * radius_ + 1.
    int win_side_ = 0;
    /// Side length of the patch that is cut out. win_side_ + 2.
    int patch_side_ = 0;
};

/// Holds the two window variants used across the levels.
struct RefineMasks {
    RefineMask small_;
    RefineMask large_;
};

/// Counter indices.
enum RefineCounter : int {
    kRefinedCorners = 0,
    kOutOfImageBreaks = 1,
    kPoorConvergence = 2,
    kSingularBreaks = 3,
    kIterationTotal = 4,
};

/// Builds the same weights as OpenCV's cornerSubPix.
///
/// zeroZone is passed as Size(-1, -1), so the branch that zeroes out the center is never taken.
RefineMask make_mask(int radius) {
    RefineMask mask;
    mask.radius_ = radius;
    mask.win_side_ = (radius * 2) + 1;
    mask.patch_side_ = mask.win_side_ + 2;
    for (int i = 0; i < mask.win_side_; ++i) {
        const float y = static_cast<float>(i - radius) / static_cast<float>(radius);
        const float vy = expf(-y * y);
        for (int j = 0; j < mask.win_side_; ++j) {
            const float x = static_cast<float>(j - radius) / static_cast<float>(radius);
            mask.weight_[(i * mask.win_side_) + j] = vy * expf(-x * x);
        }
    }
    return mask;
}

/// The position-dependent coefficients and rectangle needed to cut out a patch.
///
/// This bundles the values the sequential version computed once outside the loop. The elements
/// are computed in parallel, so every thread has to hold the same values.
struct PatchSetup {
    bool inside_ = false;
    float a_ = 0.0F;
    float b_ = 0.0F;
    int ip_x_ = 0;
    int ip_y_ = 0;
    // Rectangle and start position that adjustRect determines on the boundary path.
    long long offset_ = 0;
    int rect_x_ = 0;
    int rect_width_ = 0;
    int rect_y_ = 0;
    int rect_height_ = 0;
};

/// Computes the coefficients and the rectangle getRectSubPix uses.
///
/// Whether the interior path or the boundary path is taken is decided here as well. The condition
/// is the same as the one guarding the fast path of OpenCV's getRectSubPix_8u32f.
__device__ PatchSetup prepare_patch(std::size_t pitch, int width, int height, float center_x,
                                    float center_y, int patch_side) {
    PatchSetup setup;
    const float shift = static_cast<float>(patch_side - 1) * 0.5F;
    const float cx = center_x - shift;
    const float cy = center_y - shift;
    setup.ip_x_ = static_cast<int>(floorf(cx));
    setup.ip_y_ = static_cast<int>(floorf(cy));
    setup.a_ = cx - static_cast<float>(setup.ip_x_);
    setup.b_ = cy - static_cast<float>(setup.ip_y_);
    setup.inside_ = setup.ip_x_ >= 0 && (setup.ip_x_ + patch_side) < width && setup.ip_y_ >= 0 &&
                    (setup.ip_y_ + patch_side) < height;
    if (setup.inside_) {
        // OpenCV clamps here to avoid a division by zero.
        setup.a_ = fmaxf(setup.a_, 0.0001F);
        return setup;
    }

    // Compute the same rectangle and start position as adjustRect.
    //
    // Three of these branches are unreachable from cornerSubPix. The corners are checked to lie
    // inside the image before the call, so the patch origin never falls completely outside it.
    // Concretely those branches are `rect_x > patch_side` (the origin is more than 13 px to the
    // left), `rect_width < 0` and `rect_height < 0` (the origin lies entirely past the right or
    // bottom edge). Mutating them does not make any test fail. They are kept as a faithful copy
    // of adjustRect, but note explicitly that no test pins them down.
    if (setup.ip_x_ >= 0) {
        setup.offset_ += setup.ip_x_;
    } else {
        setup.rect_x_ = -setup.ip_x_;
        if (setup.rect_x_ > patch_side) {
            setup.rect_x_ = patch_side;
        }
    }
    if (setup.ip_x_ < width - patch_side) {
        setup.rect_width_ = patch_side;
    } else {
        setup.rect_width_ = width - setup.ip_x_ - 1;
        if (setup.rect_width_ < 0) {
            setup.offset_ += setup.rect_width_;
            setup.rect_width_ = 0;
        }
    }
    if (setup.ip_y_ >= 0) {
        setup.offset_ += static_cast<long long>(setup.ip_y_) * static_cast<long long>(pitch);
    } else {
        setup.rect_y_ = -setup.ip_y_;
    }
    if (setup.ip_y_ < height - patch_side) {
        setup.rect_height_ = patch_side;
    } else {
        setup.rect_height_ = height - setup.ip_y_ - 1;
        if (setup.rect_height_ < 0) {
            setup.offset_ +=
                    static_cast<long long>(setup.rect_height_) * static_cast<long long>(pitch);
            setup.rect_height_ = 0;
        }
    }
    setup.offset_ -= setup.rect_x_;
    return setup;
}

/// Computes one element of the patch.
///
/// Written with the **same expression and the same operand order** as the sequential version. The
/// interior path walks a row from left to right, but the carried state is only one element deep,
/// so the expression stays closed per element. On the boundary path the row advance depends on
/// the rectangle, so the start position is derived directly from i.
__device__ float patch_element(const std::uint8_t* image, std::size_t pitch, const PatchSetup& s,
                               int patch_side, int row, int column) {
    const float a = s.a_;
    const float b = s.b_;
    if (s.inside_) {
        const float a12 = a * (1.0F - b);
        const float a22 = a * b;
        const float b1 = 1.0F - b;
        const float b2 = b;
        const double scale = (1.0 - static_cast<double>(a)) / static_cast<double>(a);
        const std::uint8_t* source = image + (static_cast<std::size_t>(s.ip_y_ + row) * pitch) +
                                     static_cast<std::size_t>(s.ip_x_);
        const std::uint8_t* next = source + pitch;
        const float t = (a12 * static_cast<float>(source[column + 1])) +
                        (a22 * static_cast<float>(next[column + 1]));
        // previous is the previous t multiplied by scale. Only column 0 uses a different
        // expression.
        const float previous =
                (column == 0)
                        ? ((1.0F - a) * ((b1 * static_cast<float>(source[0])) +
                                         (b2 * static_cast<float>(next[0]))))
                        : static_cast<float>(
                                  static_cast<double>((a12 * static_cast<float>(source[column])) +
                                                      (a22 * static_cast<float>(next[column]))) *
                                  scale);
        return previous + t;
    }

    const float a11 = (1.0F - a) * (1.0F - b);
    const float a12 = a * (1.0F - b);
    const float a21 = (1.0F - a) * b;
    const float a22 = a * b;
    const float b1 = 1.0F - b;
    const float b2 = b;

    // The sequential version advances source by one row only for the rows between rect_y and
    // rect_height. Count how many times it advanced directly from row.
    const int advanced = max(0, min(row, s.rect_height_) - s.rect_y_);
    const std::uint8_t* source = image + s.offset_ + (static_cast<long long>(advanced) * pitch);
    const std::uint8_t* second =
            (row < s.rect_y_ || row >= s.rect_height_) ? source : source + pitch;

    // The sequential version writes in the order left-edge fill, right-edge fill, bilinear. The
    // later write is the one that survives, so the checks run starting from the right edge.
    if (column >= s.rect_width_) {
        return (b1 * static_cast<float>(source[s.rect_width_])) +
               (b2 * static_cast<float>(second[s.rect_width_]));
    }
    if (column < s.rect_x_) {
        return (b1 * static_cast<float>(source[s.rect_x_])) +
               (b2 * static_cast<float>(second[s.rect_x_]));
    }
    return (static_cast<float>(source[column]) * a11) +
           (static_cast<float>(source[column + 1]) * a12) +
           (static_cast<float>(second[column]) * a21) +
           (static_cast<float>(second[column + 1]) * a22);
}

/// Scratch space shared within a block.
struct RefineShared {
    float patch_[kMaxRefinePatchSide * kMaxRefinePatchSide];
    // The five accumulated quantities, stored per element. The accumulation keeps ascending
    // index order.
    double part_[5][kMaxWinSide * kMaxWinSide];
    double sum_[5];
    float x_;
    float y_;
    int stop_;
    int iteration_;
};

/// Reproduces cv::cornerSubPix for one corner. One block handles one corner.
///
/// The five accumulated quantities are closed per element and carry no dependency between
/// elements. Double-precision arithmetic is a deterministic function of its operands, so the bit
/// pattern is the same no matter which thread computes it. **The only thing that must not change
/// is the order of accumulation**, so the sums are built as five chains that keep ascending index
/// order. Those five are five independent variables in the source as well, and spreading them
/// over different lanes does not change the order within any single sum.
__device__ void corner_sub_pix_block(const std::uint8_t* image, std::size_t pitch, int width,
                                     int height, const RefineMask& mask, int max_iterations,
                                     double eps, RefineShared& shared, std::int32_t* counters) {
    const int tid = static_cast<int>(threadIdx.x);
    const int win_side = mask.win_side_;
    const int patch_side = mask.patch_side_;
    const int patch_count = patch_side * patch_side;
    const int element_count = win_side * win_side;
    const float start_x = shared.x_;
    const float start_y = shared.y_;

    if (tid == 0) {
        shared.stop_ = 0;
        shared.iteration_ = 0;
        // The CPU reference throws when the starting point lies outside the image. A GPU
        // cannot throw, so the original position is kept unrefined and the case is counted.
        if (!(start_x >= 0.0F && start_x < static_cast<float>(width) && start_y >= 0.0F &&
              start_y < static_cast<float>(height))) {
            atomicAdd(&counters[kOutOfImageBreaks], 1);
            shared.stop_ = 2;
        }
    }
    __syncthreads();
    if (shared.stop_ == 2) {
        return;
    }

    while (true) {
        const PatchSetup setup =
                prepare_patch(pitch, width, height, shared.x_, shared.y_, patch_side);
        for (int index = tid; index < patch_count; index += kRefineThreads) {
            shared.patch_[index] = patch_element(image, pitch, setup, patch_side,
                                                 index / patch_side, index % patch_side);
        }
        __syncthreads();

        // The traversal starts one row and one column inside the patch. The index mapping is
        // the same as in the sequential version: k = i * win_side + j.
        for (int k = tid; k < element_count; k += kRefineThreads) {
            const int i = k / win_side;
            const int j = k % win_side;
            const float* row = shared.patch_ + ((i + 1) * patch_side) + 1;
            const double m = static_cast<double>(mask.weight_[k]);
            const double tgx = static_cast<double>(row[j + 1]) - static_cast<double>(row[j - 1]);
            const double tgy = static_cast<double>(row[j + patch_side]) -
                               static_cast<double>(row[j - patch_side]);
            const double gxx = tgx * tgx * m;
            const double gxy = tgx * tgy * m;
            const double gyy = tgy * tgy * m;
            const double px = static_cast<double>(j - mask.radius_);
            const double py = static_cast<double>(i - mask.radius_);
            shared.part_[0][k] = gxx;
            shared.part_[1][k] = gxy;
            shared.part_[2][k] = gyy;
            shared.part_[3][k] = (gxx * px) + (gxy * py);
            shared.part_[4][k] = (gxy * px) + (gyy * py);
        }
        __syncthreads();

        if (tid < 5) {
            double accumulator = 0.0;
            for (int k = 0; k < element_count; ++k) {
                accumulator += shared.part_[tid][k];
            }
            shared.sum_[tid] = accumulator;
        }
        __syncthreads();

        if (tid == 0) {
            const double a = shared.sum_[0];
            const double b = shared.sum_[1];
            const double c = shared.sum_[2];
            const double bb1 = shared.sum_[3];
            const double bb2 = shared.sum_[4];
            const double determinant = (a * c) - (b * b);
            if (fabs(determinant) <= (DBL_EPSILON * DBL_EPSILON)) {
                atomicAdd(&counters[kSingularBreaks], 1);
                shared.stop_ = 1;
            } else {
                const double scale = 1.0 / determinant;
                const float next_x = static_cast<float>(static_cast<double>(shared.x_) +
                                                        (c * scale * bb1) - (b * scale * bb2));
                const float next_y = static_cast<float>(static_cast<double>(shared.y_) -
                                                        (b * scale * bb1) + (a * scale * bb2));
                // The error is computed in single precision and then widened to double, the
                // same order as the CPU reference.
                const float dx = next_x - shared.x_;
                const float dy = next_y - shared.y_;
                const double error = static_cast<double>((dx * dx) + (dy * dy));
                if (!(next_x >= 0.0F && next_x < static_cast<float>(width) && next_y >= 0.0F &&
                      next_y < static_cast<float>(height))) {
                    atomicAdd(&counters[kOutOfImageBreaks], 1);
                    shared.stop_ = 1;
                } else {
                    shared.x_ = next_x;
                    shared.y_ = next_y;
                    ++shared.iteration_;
                    if (!(shared.iteration_ < max_iterations && error > eps)) {
                        shared.stop_ = 1;
                    }
                }
            }
        }
        __syncthreads();
        if (shared.stop_ != 0) {
            break;
        }
    }

    if (tid == 0) {
        atomicAdd(&counters[kIterationTotal], shared.iteration_);
        // Drifting farther from the initial position than the window radius counts as poor
        // convergence, so the corner is reset.
        if (fabsf(shared.x_ - start_x) > static_cast<float>(mask.radius_) ||
            fabsf(shared.y_ - start_y) > static_cast<float>(mask.radius_)) {
            atomicAdd(&counters[kPoorConvergence], 1);
            shared.x_ = start_x;
            shared.y_ = start_y;
        }
    }
    __syncthreads();
}

/// Refines the four corners while climbing the levels. One block handles one corner.
__global__ void refine_kernel(PyramidRef pyramid, DeviceDetections detections, RefineMasks masks,
                              float scale_init, int start_level, int max_iterations, double eps,
                              std::int32_t* counters) {
    const int total = *detections.count_ * kQuadCornerCount;
    __shared__ RefineShared shared;

    // Launching one block per corner slot would start thousands of blocks even when there are
    // only a few detections. Shared memory limits how many blocks fit on one SM, so the launch
    // splits into waves and turns into waiting time; a Jetson AGX Orin was 7 times slower even
    // with no detections. Launch a modest number of blocks instead and stride across corners.
    for (int index = static_cast<int>(blockIdx.x); index < total;
         index += static_cast<int>(gridDim.x)) {
        const int detection = index / kQuadCornerCount;
        const int corner = index % kQuadCornerCount;
        const std::size_t slot = (static_cast<std::size_t>(corner) *
                                  static_cast<std::size_t>(detections.capacity_)) +
                                 static_cast<std::size_t>(detection);

        __syncthreads();
        if (threadIdx.x == 0U) {
            float x = detections.corner_x_[slot];
            float y = detections.corner_y_[slot];
            // Move from segmentation coordinates to the coordinates of the starting level.
            if (scale_init != 1.0F) {
                x *= scale_init;
                y *= scale_init;
            }
            shared.x_ = x;
            shared.y_ = y;
        }
        __syncthreads();

        for (int level = start_level - 1; level >= 0; --level) {
            if (threadIdx.x == 0U) {
                // Dropping one level down doubles the resolution.
                shared.x_ *= 2.0F;
                shared.y_ *= 2.0F;
            }
            __syncthreads();
            const int side = max(pyramid.width_[level], pyramid.height_[level]);
            const RefineMask& mask = (side > kLargeLevelSidePx) ? masks.large_ : masks.small_;
            corner_sub_pix_block(pyramid.data_[level], pyramid.pitch_[level], pyramid.width_[level],
                                 pyramid.height_[level], mask, max_iterations, eps, shared,
                                 counters);
        }
        if (threadIdx.x == 0U) {
            detections.corner_x_[slot] = shared.x_;
            detections.corner_y_[slot] = shared.y_;
            atomicAdd(&counters[kRefinedCorners], 1);
        }
    }
}

}  // namespace

std::size_t corner_refine_workspace_bytes(const DetectorConfig& config) {
    if (config.max_markers_ <= 0) {
        return 0U;
    }
    return align_up(static_cast<std::size_t>(kRefineCounterCount) * sizeof(std::int32_t),
                    kPlaneAlignment);
}

Status reserve_corner_refine(const DetectorConfig& config, Workspace& workspace,
                             CornerRefineBuffers* out) {
    if (out == nullptr) {
        return Status::kInvalidArgument;
    }
    if (corner_refine_workspace_bytes(config) == 0U) {
        return Status::kInvalidConfig;
    }
    void* pointer = nullptr;
    const Status status =
            workspace.allocate(static_cast<std::size_t>(kRefineCounterCount) * sizeof(std::int32_t),
                               kPlaneAlignment, &pointer);
    if (status != Status::kOk) {
        return status;
    }
    CornerRefineBuffers buffers;
    buffers.diagnostics_.counters_ = static_cast<std::int32_t*>(pointer);
    *out = buffers;
    return Status::kOk;
}

int refine_block_count(int multi_processor_count) {
    // Twice the SM count is the target. On a machine with few SMs this lands below the cap of
    // 64 and the launch cost drops.
    //
    // Measured by alternating the two versions within one session (comparing across sessions is
    // meaningless: the difference drowns in the state of the clocks):
    //   Jetson AGX Orin (16 SM -> 32 blocks): no detections -6.0%,
    //     4 markers -3.3%, 16 markers -1.3%
    //   DGX Spark (20 SM -> 40 blocks): the difference sits inside the noise. Even over 8
    //     repetitions the direction of the median flips from run to run, and each version's
    //     spread (0.59 to 0.71 ms) is wider than the gap between the versions
    //   RTX 5070 Ti (70 SM): pinned to the cap of 64, so nothing changes
    //
    // The gain is modest, but dropping the fixed value matters in itself. When the corner cap
    // (4096) was used directly as the block count, the Jetson was 7 times slower. The same
    // accident will not repeat when a machine with an order-of-magnitude different SM count is
    // added.
    const long long doubled = 2LL * static_cast<long long>(multi_processor_count);
    if (doubled < kMinRefineBlocks) {
        return kMinRefineBlocks;
    }
    if (doubled > kMaxRefineBlocks) {
        return kMaxRefineBlocks;
    }
    return static_cast<int>(doubled);
}

Status refine_corners_async(const PyramidRef& pyramid, const ScalePlan& plan,
                            const DetectorConfig& config, int block_count,
                            CornerRefineBuffers* buffers, DeviceDetections* detections,
                            cudaStream_t stream) {
    if (buffers == nullptr || detections == nullptr || buffers->diagnostics_.counters_ == nullptr ||
        detections->corner_x_ == nullptr || detections->corner_y_ == nullptr ||
        detections->count_ == nullptr) {
        return Status::kInvalidArgument;
    }
    if (pyramid.level_count_ < 1 || pyramid.level_count_ > kMaxPyramidLevels) {
        return Status::kInvalidArgument;
    }
    if (plan.closest_level_index_ < 0 || plan.closest_level_index_ >= pyramid.level_count_) {
        return Status::kInvalidConfig;
    }
    if (plan.segmentation_width_px_ <= 0 || config.corner_refinement_max_iterations_ <= 0) {
        return Status::kInvalidConfig;
    }

    RefineMasks masks;
    masks.small_ = make_mask(kSmallWinRadius);
    masks.large_ = make_mask(kLargeWinRadius);

    const auto scale_init = static_cast<float>(pyramid.width_[plan.closest_level_index_]) /
                            static_cast<float>(plan.segmentation_width_px_);
    // OpenCV clamps the iteration count to between 1 and 100 and squares the convergence
    // threshold before use.
    int max_iterations = config.corner_refinement_max_iterations_;
    max_iterations = (max_iterations < 1) ? 1 : max_iterations;
    max_iterations = (max_iterations > kMaxIterations) ? kMaxIterations : max_iterations;
    const double accuracy = (config.corner_refinement_min_accuracy_px_ > 0.0)
                                    ? config.corner_refinement_min_accuracy_px_
                                    : 0.0;
    const double eps = accuracy * accuracy;

    Status status = check_cuda(
            cudaMemsetAsync(buffers->diagnostics_.counters_, 0,
                            static_cast<std::size_t>(kRefineCounterCount) * sizeof(std::int32_t),
                            stream),
            "cudaMemsetAsync", "corner_refine.reset", -1, stream);
    if (status != Status::kOk) {
        return status;
    }

    // One block handles one corner, and corners beyond the block count are processed by
    // striding. Given the SM count of the target machines and the number of blocks per SM that
    // the shared memory allows, the count is chosen to cover the upper bound of the evaluation
    // plan (16 markers = 64 corners) in a single wave.
    if (block_count < 1) {
        return Status::kInvalidArgument;
    }
    const auto corner_count = detections->capacity_ * kQuadCornerCount;
    const auto blocks =
            static_cast<unsigned int>((corner_count < block_count) ? corner_count : block_count);
    refine_kernel<<<blocks, static_cast<unsigned int>(kRefineThreads), 0, stream>>>(
            pyramid, *detections, masks, scale_init, plan.closest_level_index_, max_iterations, eps,
            buffers->diagnostics_.counters_);
    return check_kernel_launch("corner_refine.refine_kernel", -1, false, stream);
}

}  // namespace aruco3cuda::detail
