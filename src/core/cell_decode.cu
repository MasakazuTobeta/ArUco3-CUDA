// SPDX-License-Identifier: Apache-2.0
#include "cell_decode.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/workspace.hpp"
#include "candidate_filter.hpp"
#include "cell_sample.hpp"
#include "cuda_check.hpp"

namespace aruco3cuda::detail {
namespace {

constexpr std::size_t kPlaneAlignment = 256U;
/// Thread count of the block that handles one candidate.
constexpr int kDecodeThreads = 256;
/// Number of intensity levels. 256 because the image is 8-bit.
constexpr int kHistogramBins = 256;
constexpr int kWarpSize = 32;
/// Warp count of one block. It caps the shared memory used by the reductions.
constexpr int kDecodeWarps = kDecodeThreads / kWarpSize;
/// Value OpenCV uses when excluding Otsu bins. The same as FLT_EPSILON.
constexpr double kOtsuGuard = 1.1920928955078125e-07;

/// Bundles the constants the decision needs, to keep the argument count down.
struct DecodeParams {
    int canonical_side_ = 0;
    int cells_ = 0;
    int cell_size_ = 0;
    int cell_margin_ = 0;
    int border_bits_ = 0;
    int max_border_errors_ = 0;
    float valid_bit_threshold_ = 0.0F;
    double min_otsu_std_dev_ = 0.0;
};

/// State the Otsu recurrence leaves behind for each bin.
///
/// Only the recurrence itself has to run sequentially. Everything before and after it is
/// independent per element, so computing it in parallel does not change the rounding.
struct OtsuState {
    double q1_[kHistogramBins];
    double mu1_[kHistogramBins];
    /// Whether the bin passed the guard. A bin that did not is never a candidate.
    bool candidate_[kHistogramBins];
};

/// The values for a single candidate in the threshold search.
struct OtsuCandidate {
    double sigma_;
    int index_;
};

/// Computes the centroid of the histogram in parallel.
///
/// Every term i * histogram[i] and every partial sum is an integer bounded by
/// 255 * pixel_count. The canonical side length is limited to 32767, so pixel_count stays at or
/// below 1.07e9 and even multiplied by 255 it reaches only 2.7e11, which fits within 2^53.
/// **Every partial sum is therefore exactly representable in double precision, no rounding ever
/// occurs, and changing the order of the additions leaves the result bit-identical.**
__device__ double otsu_center(const int* histogram, double scale, double* warp_partial) {
    double partial = 0.0;
    for (int i = static_cast<int>(threadIdx.x); i < kHistogramBins;
         i += static_cast<int>(blockDim.x)) {
        partial += static_cast<double>(i) * static_cast<double>(histogram[i]);
    }
    // Reduce within the warp with shuffles, which avoids spending shared memory per thread.
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        partial += __shfl_down_sync(0xffffffffU, partial, offset);
    }
    const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    if (lane == 0) {
        warp_partial[warp] = partial;
    }
    __syncthreads();
    double total = 0.0;
    for (int w = 0; w < kDecodeWarps; ++w) {
        total += warp_partial[w];
    }
    return total * scale;
}

/// Runs the Otsu recurrence sequentially and records the state of every bin.
///
/// **This part must never be parallelized.** p_i is a rounded double, so accumulating q1 depends
/// on the order. On top of that, an iteration skipped by the guard carries mu1 forward in its
/// unnormalized state (the previous mu1 multiplied by the previous q1), and the next iteration
/// multiplies it by yet another q1. mu1 is thus a path-dependent quantity that depends on how
/// many iterations in a row were skipped, and any scheme that builds q1 with a parallel scan and
/// then reconstructs mu1 is guaranteed to break.
__device__ void otsu_recurrence(const int* histogram, double scale, OtsuState* state) {
    double mu1 = 0.0;
    double q1 = 0.0;
    for (int i = 0; i < kHistogramBins; ++i) {
        const double p_i = static_cast<double>(histogram[i]) * scale;
        mu1 *= q1;
        q1 += p_i;
        const double q2 = 1.0 - q1;
        state->q1_[i] = q1;
        if (fmin(q1, q2) < kOtsuGuard || fmax(q1, q2) > 1.0 - kOtsuGuard) {
            state->candidate_[i] = false;
            continue;
        }
        mu1 = (mu1 + (static_cast<double>(i) * p_i)) / q1;
        state->mu1_[i] = mu1;
        state->candidate_[i] = true;
    }
}

/// Computes the separability of every bin in parallel and picks the level that maximizes it.
///
/// The sequential version starts max_sigma at 0.0 and updates on a strict greater-than. Three
/// properties follow from that.
///
/// 1. A bin whose sigma is 0 or less is never selected
/// 2. On a tie the lower intensity level wins
/// 3. If no update ever happens the result is 0
///
/// The reduction honors all three exactly. The comparison is lexicographic on
/// (larger sigma, lower level).
__device__ int otsu_argmax(const OtsuState& state, double mu, OtsuCandidate* warp_best) {
    double best_sigma = 0.0;
    int best_index = kHistogramBins;
    for (int i = static_cast<int>(threadIdx.x); i < kHistogramBins;
         i += static_cast<int>(blockDim.x)) {
        if (!state.candidate_[i]) {
            continue;
        }
        const double q1 = state.q1_[i];
        const double mu1 = state.mu1_[i];
        const double q2 = 1.0 - q1;
        const double mu2 = (mu - (q1 * mu1)) / q2;
        const double sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2);
        // Values of 0 or less are not candidates, matching the sequential version whose
        // max_sigma starts at 0.0 and updates only on a strict greater-than.
        if (sigma <= 0.0) {
            continue;
        }
        if (sigma > best_sigma || (sigma == best_sigma && i < best_index)) {
            best_sigma = sigma;
            best_index = i;
        }
    }
    // Reduce within the warp with shuffles, lexicographic on (larger sigma, lower level).
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        const double other_sigma = __shfl_down_sync(0xffffffffU, best_sigma, offset);
        const int other_index = __shfl_down_sync(0xffffffffU, best_index, offset);
        if (other_sigma > best_sigma || (other_sigma == best_sigma && other_index < best_index)) {
            best_sigma = other_sigma;
            best_index = other_index;
        }
    }
    const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    if (lane == 0) {
        warp_best[warp].sigma_ = best_sigma;
        warp_best[warp].index_ = best_index;
    }
    __syncthreads();
    double sigma = 0.0;
    int index = kHistogramBins;
    for (int w = 0; w < kDecodeWarps; ++w) {
        if (warp_best[w].sigma_ > sigma ||
            (warp_best[w].sigma_ == sigma && warp_best[w].index_ < index)) {
            sigma = warp_best[w].sigma_;
            index = warp_best[w].index_;
        }
    }
    // Return 0 when there was never a candidate.
    return (index >= kHistogramBins) ? 0 : index;
}

/// Computes the cell ratios per candidate and counts the erroneous border cells. One block
/// handles one candidate.
__global__ void decode_cells_kernel(const std::uint8_t* canonical, const std::int32_t* count,
                                    float* ratios, std::int32_t* border_errors,
                                    std::uint8_t* accepted, std::int32_t* thresholds,
                                    DecodeParams params) {
    const int candidate = static_cast<int>(blockIdx.x);
    if (candidate >= *count) {
        return;
    }

    __shared__ int histogram[kHistogramBins];
    // Shared memory for the three Otsu phases: the state left by the recurrence and the scratch
    // space of the reductions.
    __shared__ OtsuState otsu_state;
    __shared__ double reduction[kDecodeWarps];
    __shared__ OtsuCandidate best[kDecodeWarps];
    __shared__ unsigned long long inner_sum;
    __shared__ unsigned long long inner_square_sum;
    __shared__ int threshold;
    __shared__ float uniform_ratio;
    __shared__ bool uniform;

    const int side = params.canonical_side_;
    const int pixels = side * side;
    const std::uint8_t* image =
            canonical + (static_cast<std::size_t>(candidate) * static_cast<std::size_t>(side) *
                         static_cast<std::size_t>(side));

    for (int i = static_cast<int>(threadIdx.x); i < kHistogramBins;
         i += static_cast<int>(blockDim.x)) {
        histogram[i] = 0;
    }
    if (threadIdx.x == 0U) {
        inner_sum = 0ULL;
        inner_square_sum = 0ULL;
    }
    __syncthreads();

    // The interior range. The CPU reference pulls each side in by half a cell, using integer
    // division.
    const int margin = params.cell_size_ / 2;
    const int inner_begin = margin;
    const int inner_end = side - margin;

    unsigned long long local_sum = 0ULL;
    unsigned long long local_square = 0ULL;
    for (int index = static_cast<int>(threadIdx.x); index < pixels;
         index += static_cast<int>(blockDim.x)) {
        const int value = static_cast<int>(image[index]);
        // Otsu is applied to the whole canonical image, not to its interior.
        atomicAdd(&histogram[value], 1);
        const int x = index % side;
        const int y = index / side;
        if (x >= inner_begin && x < inner_end && y >= inner_begin && y < inner_end) {
            local_sum += static_cast<unsigned long long>(value);
            local_square += static_cast<unsigned long long>(value * value);
        }
    }
    atomicAdd(&inner_sum, local_sum);
    atomicAdd(&inner_square_sum, local_square);
    __syncthreads();

    if (threadIdx.x == 0U) {
        const int inner_side = inner_end - inner_begin;
        const int inner_count = inner_side * inner_side;
        // The sum and the sum of squares were accumulated as integers; rounding starts here.
        const double scale = 1.0 / static_cast<double>(inner_count);
        const double mean = static_cast<double>(inner_sum) * scale;
        const double variance = (static_cast<double>(inner_square_sum) * scale) - (mean * mean);
        const double deviation = sqrt(fmax(variance, 0.0));
        uniform = deviation < params.min_otsu_std_dev_;
        uniform_ratio = (mean > 127.0) ? 1.0F : 0.0F;
    }
    __syncthreads();

    // Determine the Otsu threshold in three phases. Accumulating the centroid and searching for
    // the separability are independent per element and therefore run in parallel; only the
    // recurrence runs sequentially on thread 0.
    if (!uniform) {
        const double pixel_scale = 1.0 / static_cast<double>(pixels);
        const double center = otsu_center(histogram, pixel_scale, reduction);
        if (threadIdx.x == 0U) {
            otsu_recurrence(histogram, pixel_scale, &otsu_state);
        }
        __syncthreads();
        const int found = otsu_argmax(otsu_state, center, best);
        if (threadIdx.x == 0U) {
            threshold = found;
        }
    } else if (threadIdx.x == 0U) {
        threshold = 0;
    }
    __syncthreads();
    if (threadIdx.x == 0U) {
        thresholds[candidate] = threshold;
    }

    // Ratio per cell: count the white pixels in the range that excludes the margin.
    const int cells = params.cells_;
    const int cell_size = params.cell_size_;
    const int cell_margin = params.cell_margin_;
    const int inner_cell = cell_size - (2 * cell_margin);
    const float denominator = static_cast<float>(inner_cell) * static_cast<float>(inner_cell);
    float* cell_ratios =
            ratios + (static_cast<std::size_t>(candidate) * static_cast<std::size_t>(cells) *
                      static_cast<std::size_t>(cells));
    const int cell_count = cells * cells;
    for (int cell = static_cast<int>(threadIdx.x); cell < cell_count;
         cell += static_cast<int>(blockDim.x)) {
        if (uniform) {
            cell_ratios[cell] = uniform_ratio;
            continue;
        }
        const int cell_x = cell % cells;
        const int cell_y = cell / cells;
        const int origin_x = (cell_x * cell_size) + cell_margin;
        const int origin_y = (cell_y * cell_size) + cell_margin;
        int white = 0;
        for (int y = 0; y < inner_cell; ++y) {
            for (int x = 0; x < inner_cell; ++x) {
                const int value = static_cast<int>(image[((origin_y + y) * side) + origin_x + x]);
                // Binarization is "pixel > threshold", so an equal value falls on the 0 side.
                if (value > threshold) {
                    ++white;
                }
            }
        }
        cell_ratios[cell] = static_cast<float>(white) / denominator;
    }
    __syncthreads();

    if (threadIdx.x == 0U) {
        // Count the erroneous border cells: first the full left and right columns, then the
        // middle part of the top and bottom rows. This is the same traversal as the CPU
        // reference and never counts a corner cell twice.
        int errors = 0;
        for (int y = 0; y < cells; ++y) {
            for (int k = 0; k < params.border_bits_; ++k) {
                if (cell_ratios[(y * cells) + k] > params.valid_bit_threshold_) {
                    ++errors;
                }
                if (cell_ratios[(y * cells) + (cells - 1 - k)] > params.valid_bit_threshold_) {
                    ++errors;
                }
            }
        }
        for (int x = params.border_bits_; x < cells - params.border_bits_; ++x) {
            for (int k = 0; k < params.border_bits_; ++k) {
                if (cell_ratios[(k * cells) + x] > params.valid_bit_threshold_) {
                    ++errors;
                }
                if (cell_ratios[((cells - 1 - k) * cells) + x] > params.valid_bit_threshold_) {
                    ++errors;
                }
            }
        }
        border_errors[candidate] = errors;
        accepted[candidate] = (errors > params.max_border_errors_) ? 0U : 1U;
    }
}

}  // namespace

int cells_per_side(const DetectorConfig& config, int marker_size) {
    if (marker_size < 1 || config.marker_border_bits_ < 1) {
        return 0;
    }
    const long long cells =
            static_cast<long long>(marker_size) + (2LL * config.marker_border_bits_);
    if (cells > 4096LL) {
        return 0;
    }
    return static_cast<int>(cells);
}

std::size_t cell_ratio_workspace_bytes(const DetectorConfig& config, int marker_size) {
    const int cells = cells_per_side(config, marker_size);
    if (cells == 0 || config.max_candidates_ <= 0) {
        return 0U;
    }
    const auto capacity = static_cast<std::size_t>(config.max_candidates_);
    const auto per_candidate = static_cast<std::size_t>(cells) * static_cast<std::size_t>(cells);
    if (per_candidate > std::numeric_limits<std::size_t>::max() / capacity) {
        return 0U;
    }
    const std::size_t ratios = align_up(per_candidate * capacity * sizeof(float), kPlaneAlignment);
    const std::size_t errors = align_up(capacity * sizeof(std::int32_t), kPlaneAlignment);
    const std::size_t flags = align_up(capacity * sizeof(std::uint8_t), kPlaneAlignment);
    const std::size_t thresholds = align_up(capacity * sizeof(std::int32_t), kPlaneAlignment);
    if (ratios == 0U || errors == 0U || flags == 0U || thresholds == 0U) {
        return 0U;
    }
    return ratios + errors + flags + thresholds;
}

Status reserve_cell_ratios(const DetectorConfig& config, int marker_size, Workspace& workspace,
                           CellRatioBuffers* out) {
    if (out == nullptr) {
        return Status::kInvalidArgument;
    }
    const int cells = cells_per_side(config, marker_size);
    if (cells == 0) {
        return Status::kInvalidArgument;
    }
    if (cell_ratio_workspace_bytes(config, marker_size) == 0U) {
        return Status::kInvalidConfig;
    }
    const auto capacity = static_cast<std::size_t>(config.max_candidates_);
    const auto per_candidate = static_cast<std::size_t>(cells) * static_cast<std::size_t>(cells);

    CellRatioBuffers buffers;
    buffers.cells_per_side_ = cells;
    buffers.capacity_ = config.max_candidates_;

    void* pointer = nullptr;
    Status status =
            workspace.allocate(per_candidate * capacity * sizeof(float), kPlaneAlignment, &pointer);
    if (status != Status::kOk) {
        return status;
    }
    buffers.ratios_ = static_cast<float*>(pointer);

    status = workspace.allocate(capacity * sizeof(std::int32_t), kPlaneAlignment, &pointer);
    if (status != Status::kOk) {
        return status;
    }
    buffers.border_errors_ = static_cast<std::int32_t*>(pointer);

    status = workspace.allocate(capacity * sizeof(std::uint8_t), kPlaneAlignment, &pointer);
    if (status != Status::kOk) {
        return status;
    }
    buffers.accepted_ = static_cast<std::uint8_t*>(pointer);

    status = workspace.allocate(capacity * sizeof(std::int32_t), kPlaneAlignment, &pointer);
    if (status != Status::kOk) {
        return status;
    }
    buffers.thresholds_ = static_cast<std::int32_t*>(pointer);

    *out = buffers;
    return Status::kOk;
}

Status build_cell_ratios_async(const CanonicalBuffers& canonical,
                               const DeviceCandidates& candidates, const DetectorConfig& config,
                               int marker_size, CellRatioBuffers* ratios, cudaStream_t stream) {
    if (ratios == nullptr || ratios->ratios_ == nullptr || ratios->thresholds_ == nullptr ||
        canonical.images_ == nullptr || candidates.count_ == nullptr) {
        return Status::kInvalidArgument;
    }
    const int cells = cells_per_side(config, marker_size);
    if (cells == 0 || ratios->cells_per_side_ != cells) {
        return Status::kInvalidArgument;
    }
    if (ratios->capacity_ < candidates.capacity_) {
        return Status::kInvalidArgument;
    }

    DecodeParams params;
    params.canonical_side_ = canonical.side_px_;
    params.cells_ = cells;
    params.cell_size_ = config.perspective_remove_pixel_per_cell_;
    // The margin truncates toward zero. Under the default configuration it becomes 0 and the
    // whole cell is counted.
    params.cell_margin_ = static_cast<int>(config.perspective_remove_ignored_margin_per_cell_ *
                                           config.perspective_remove_pixel_per_cell_);
    params.border_bits_ = config.marker_border_bits_;
    // The rate multiplies the square of marker_size, not the number of border cells, matching
    // the CPU reference.
    params.max_border_errors_ = static_cast<int>(static_cast<double>(marker_size) * marker_size *
                                                 config.max_erroneous_bits_in_border_rate_);
    params.valid_bit_threshold_ = static_cast<float>(config.valid_bit_threshold_);
    params.min_otsu_std_dev_ = config.min_otsu_std_dev_;
    if (params.cell_size_ - (2 * params.cell_margin_) < 1) {
        return Status::kInvalidConfig;
    }

    decode_cells_kernel<<<static_cast<unsigned int>(candidates.capacity_),
                          static_cast<unsigned int>(kDecodeThreads), 0, stream>>>(
            canonical.images_, candidates.count_, ratios->ratios_, ratios->border_errors_,
            ratios->accepted_, ratios->thresholds_, params);
    return check_kernel_launch("cell_decode.decode_cells_kernel", -1, false, stream);
}

}  // namespace aruco3cuda::detail
