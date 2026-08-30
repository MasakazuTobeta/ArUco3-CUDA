// SPDX-License-Identifier: Apache-2.0
#include "cell_sample.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/workspace.hpp"
#include "candidate_filter.hpp"
#include "cuda_check.hpp"
#include "preprocess.hpp"
#include "quad_extract.hpp"

namespace aruco3cuda::detail {
namespace {

constexpr std::size_t kPlaneAlignment = 256U;
/// Thread count of the block handling one candidate. The threads split the
/// canonical pixels between them.
constexpr int kCanonicalThreads = 256;
/// Threshold for the LU singularity test. Same as OpenCV's DBL_EPSILON * 100.
constexpr double kLuEpsilon = 2.2204460492503131e-14;

/// Builds the coefficient matrix that solves the map from a candidate's four
/// corners to canonical.
///
/// The layout matches OpenCV's getPerspectiveTransform: src is the canonical
/// corners and dst is the candidate corners. The CPU path solves the opposite
/// direction and then inverts, while this solves the inverse map from the
/// start. The two describe the same map, but the rounding differs, so the later
/// steps reproduce the inversion procedure as well.
__device__ void build_system(const float src_x[kQuadCornerCount],
                             const float src_y[kQuadCornerCount],
                             const float dst_x[kQuadCornerCount],
                             const float dst_y[kQuadCornerCount], double a[8][8], double b[8]) {
    for (int i = 0; i < 4; ++i) {
        a[i][0] = static_cast<double>(src_x[i]);
        a[i][1] = static_cast<double>(src_y[i]);
        a[i][2] = 1.0;
        a[i][3] = 0.0;
        a[i][4] = 0.0;
        a[i][5] = 0.0;
        a[i + 4][0] = 0.0;
        a[i + 4][1] = 0.0;
        a[i + 4][2] = 0.0;
        a[i + 4][3] = static_cast<double>(src_x[i]);
        a[i + 4][4] = static_cast<double>(src_y[i]);
        a[i + 4][5] = 1.0;
        // Widen a single-precision product to double. To round the way OpenCV
        // does, these must not be multiplied in double precision: the matrix
        // shifts by up to 1.9e-3 relative.
        a[i][6] = static_cast<double>(-src_x[i] * dst_x[i]);
        a[i][7] = static_cast<double>(-src_y[i] * dst_x[i]);
        a[i + 4][6] = static_cast<double>(-src_x[i] * dst_y[i]);
        a[i + 4][7] = static_cast<double>(-src_y[i] * dst_y[i]);
        b[i] = static_cast<double>(dst_x[i]);
        b[i + 4] = static_cast<double>(dst_y[i]);
    }
}

/// Solves the 8-unknown linear system by Gaussian elimination with partial pivoting.
///
/// Follows the same steps as OpenCV's LUImpl: on a tie keep the row with the
/// smaller index, and elimination forms d = -1/A[i][i] once and multiplies by
/// it. Writing alpha = -A[j][i]/A[i][i] instead changes the rounding.
///
/// @return true when solved. false when the pivot falls below the threshold.
__device__ bool solve_lu8(double a[8][8], double b[8]) {
    constexpr int kMatrixSize = 8;
    for (int i = 0; i < kMatrixSize; ++i) {
        int k = i;
        for (int j = i + 1; j < kMatrixSize; ++j) {
            if (fabs(a[j][i]) > fabs(a[k][i])) {
                k = j;
            }
        }
        if (fabs(a[k][i]) < kLuEpsilon) {
            return false;
        }
        if (k != i) {
            for (int j = i; j < kMatrixSize; ++j) {
                const double swap = a[i][j];
                a[i][j] = a[k][j];
                a[k][j] = swap;
            }
            const double swap = b[i];
            b[i] = b[k];
            b[k] = swap;
        }
        const double d = -1.0 / a[i][i];
        for (int j = i + 1; j < kMatrixSize; ++j) {
            const double alpha = a[j][i] * d;
            for (int c = i + 1; c < kMatrixSize; ++c) {
                a[j][c] += alpha * a[i][c];
            }
            b[j] += alpha * b[i];
        }
    }
    for (int i = kMatrixSize - 1; i >= 0; --i) {
        double s = b[i];
        for (int k = i + 1; k < kMatrixSize; ++k) {
            s -= a[i][k] * b[k];
        }
        b[i] = s / a[i][i];
    }
    return true;
}

/// Converts a double to an int with round-to-nearest-even. Same as OpenCV's cvRound.
__device__ int round_to_nearest_even(double value) {
    // Clamp into int range before rounding. OpenCV does it in the same order.
    const double clamped = fmin(fmax(value, -2147483648.0), 2147483647.0);
    return __double2int_rn(clamped);
}

/// Chooses the pyramid level used for identification, from a candidate's perimeter.
///
/// Same rule as the CPU path's _findOptPyrImageForCanonicalImg: consider only
/// positive distances and take the level with the smallest one. The comparison
/// is done in single precision.
__device__ int choose_level(const PyramidRef& pyramid, int segmentation_width, int perimeter,
                            int min_perimeter) {
    int optimal = 0;
    float distance = 3.402823466e+38F;
    for (int i = 0; i < pyramid.level_count_; ++i) {
        const float scale =
                static_cast<float>(pyramid.width_[i]) / static_cast<float>(segmentation_width);
        const float scaled = static_cast<float>(perimeter) * scale;
        const float value = scaled - static_cast<float>(min_perimeter);
        if (value < distance && value > 0.0F) {
            distance = value;
            optimal = i;
        }
    }
    return optimal;
}

/// Builds the canonical image per candidate. One block handles one candidate.
__global__ void warp_canonical_kernel(PyramidRef pyramid, const std::int32_t* corner_x,
                                      const std::int32_t* corner_y, const std::int32_t* perimeter,
                                      int candidate_capacity, const std::int32_t* count,
                                      std::uint8_t* images, int side, int segmentation_width,
                                      int min_perimeter, bool use_aruco3) {
    const int candidate = static_cast<int>(blockIdx.x);
    if (candidate >= *count) {
        return;
    }

    __shared__ double matrix[9];
    __shared__ const std::uint8_t* level_data;
    __shared__ int level_width;
    __shared__ int level_height;
    __shared__ std::size_t level_pitch;

    if (threadIdx.x == 0U) {
        const int level = use_aruco3 ? choose_level(pyramid, segmentation_width,
                                                    perimeter[candidate], min_perimeter)
                                     : 0;
        level_data = pyramid.data_[level];
        level_width = pyramid.width_[level];
        level_height = pyramid.height_[level];
        level_pitch = pyramid.pitch_[level];

        // Scale the four corners into the level's coordinates. The CPU path
        // computes this as a cv::Point2f times a float, so single precision is
        // used here too.
        const float scale =
                use_aruco3
                        ? (static_cast<float>(level_width) / static_cast<float>(segmentation_width))
                        : 1.0F;
        float quad_x[kQuadCornerCount];
        float quad_y[kQuadCornerCount];
        for (int c = 0; c < kQuadCornerCount; ++c) {
            const int index = (c * candidate_capacity) + candidate;
            quad_x[c] = static_cast<float>(corner_x[index]) * scale;
            quad_y[c] = static_cast<float>(corner_y[index]) * scale;
        }
        // Corners of the canonical image. The edge is side - 1, not side.
        const float edge = static_cast<float>(side) - 1.0F;
        const float square_x[kQuadCornerCount] = {0.0F, edge, edge, 0.0F};
        const float square_y[kQuadCornerCount] = {0.0F, 0.0F, edge, edge};

        // The CPU path solves the map from quad to square and warpPerspective
        // builds the inverse. The same order is used here to match the rounding.
        double a[8][8];
        double b[8];
        build_system(quad_x, quad_y, square_x, square_y, a, b);
        double forward[9];
        if (solve_lu8(a, b)) {
            for (int i = 0; i < 8; ++i) {
                forward[i] = b[i];
            }
            forward[8] = 1.0;
        } else {
            // Degenerate corners. OpenCV falls back to SVD, but this does not
            // happen for candidates that passed the filter. Here the canonical
            // image is left filled with 0 instead.
            for (int i = 0; i < 9; ++i) {
                forward[i] = 0.0;
            }
        }

        // Build the inverse with the 3x3 cofactor formula, in the same order
        // as OpenCV's invert.
        const double determinant =
                forward[0] * (forward[4] * forward[8] - forward[5] * forward[7]) -
                forward[1] * (forward[3] * forward[8] - forward[5] * forward[6]) +
                forward[2] * (forward[3] * forward[7] - forward[4] * forward[6]);
        if (determinant != 0.0) {
            const double r = 1.0 / determinant;
            matrix[0] = (forward[4] * forward[8] - forward[5] * forward[7]) * r;
            matrix[1] = (forward[2] * forward[7] - forward[1] * forward[8]) * r;
            matrix[2] = (forward[1] * forward[5] - forward[2] * forward[4]) * r;
            matrix[3] = (forward[5] * forward[6] - forward[3] * forward[8]) * r;
            matrix[4] = (forward[0] * forward[8] - forward[2] * forward[6]) * r;
            matrix[5] = (forward[2] * forward[3] - forward[0] * forward[5]) * r;
            matrix[6] = (forward[3] * forward[7] - forward[4] * forward[6]) * r;
            matrix[7] = (forward[1] * forward[6] - forward[0] * forward[7]) * r;
            matrix[8] = (forward[0] * forward[4] - forward[1] * forward[3]) * r;
        } else {
            // OpenCV's invert returns without touching the matrix when det is 0.
            for (int i = 0; i < 9; ++i) {
                matrix[i] = forward[i];
            }
        }
    }
    __syncthreads();

    std::uint8_t* destination =
            images + (static_cast<std::size_t>(candidate) * static_cast<std::size_t>(side) *
                      static_cast<std::size_t>(side));
    const int pixels = side * side;
    for (int index = static_cast<int>(threadIdx.x); index < pixels;
         index += static_cast<int>(blockDim.x)) {
        const int x = index % side;
        const int y = index / side;
        // OpenCV forms the partial sums at the block origin. When the
        // canonical side is 32 or less, the block width equals the side, so the
        // origin is always 0 and the partial sum equals M[1] * y + M[2].
        const double x0 = matrix[1] * static_cast<double>(y) + matrix[2];
        const double y0 = matrix[4] * static_cast<double>(y) + matrix[5];
        const double w0 = matrix[7] * static_cast<double>(y) + matrix[8];

        const double wd = w0 + matrix[6] * static_cast<double>(x);
        // Take the reciprocal first and then multiply. Writing it as a division differs by 1 ULP.
        const double w = (wd != 0.0) ? (1.0 / wd) : 0.0;
        const int sx = round_to_nearest_even((x0 + matrix[0] * static_cast<double>(x)) * w);
        const int sy = round_to_nearest_even((y0 + matrix[3] * static_cast<double>(x)) * w);

        std::uint8_t value = 0U;
        if (static_cast<unsigned int>(sx) < static_cast<unsigned int>(level_width) &&
            static_cast<unsigned int>(sy) < static_cast<unsigned int>(level_height)) {
            value = level_data[(static_cast<std::size_t>(sy) * level_pitch) +
                               static_cast<std::size_t>(sx)];
        }
        destination[(static_cast<std::size_t>(y) * static_cast<std::size_t>(side)) +
                    static_cast<std::size_t>(x)] = value;
    }
}

}  // namespace

int canonical_side_px(const DetectorConfig& config, int marker_size) {
    if (marker_size < 1 || config.marker_border_bits_ < 1 ||
        config.perspective_remove_pixel_per_cell_ < 1) {
        return 0;
    }
    const long long cells =
            static_cast<long long>(marker_size) + (2LL * config.marker_border_bits_);
    const long long side = cells * config.perspective_remove_pixel_per_cell_;
    if (side > 32767LL) {
        // OpenCV's warpPerspective itself requires less than 32767.
        return 0;
    }
    return static_cast<int>(side);
}

std::size_t canonical_workspace_bytes(const DetectorConfig& config, int marker_size) {
    const int side = canonical_side_px(config, marker_size);
    if (side == 0 || config.max_candidates_ <= 0) {
        return 0U;
    }
    const auto pixels = static_cast<std::size_t>(side) * static_cast<std::size_t>(side);
    const auto capacity = static_cast<std::size_t>(config.max_candidates_);
    if (pixels > std::numeric_limits<std::size_t>::max() / capacity) {
        return 0U;
    }
    return align_up(pixels * capacity, kPlaneAlignment);
}

Status reserve_canonical(const DetectorConfig& config, int marker_size, Workspace& workspace,
                         CanonicalBuffers* out) {
    if (out == nullptr) {
        return Status::kInvalidArgument;
    }
    const int side = canonical_side_px(config, marker_size);
    if (side == 0) {
        return Status::kInvalidArgument;
    }
    const std::size_t bytes = canonical_workspace_bytes(config, marker_size);
    if (bytes == 0U) {
        return Status::kInvalidConfig;
    }
    void* pointer = nullptr;
    const Status status =
            workspace.allocate(static_cast<std::size_t>(side) * static_cast<std::size_t>(side) *
                                       static_cast<std::size_t>(config.max_candidates_),
                               kPlaneAlignment, &pointer);
    if (status != Status::kOk) {
        return status;
    }
    CanonicalBuffers buffers;
    buffers.images_ = static_cast<std::uint8_t*>(pointer);
    buffers.side_px_ = side;
    buffers.capacity_ = config.max_candidates_;
    *out = buffers;
    return Status::kOk;
}

Status build_canonical_async(const PreprocessBuffers& preprocess, const ScalePlan& plan,
                             const DeviceCandidates& candidates, const DetectorConfig& config,
                             CanonicalBuffers* canonical, cudaStream_t stream) {
    if (canonical == nullptr || canonical->images_ == nullptr || candidates.corner_x_ == nullptr ||
        candidates.count_ == nullptr) {
        return Status::kInvalidArgument;
    }
    if (canonical->capacity_ < candidates.capacity_) {
        return Status::kInvalidArgument;
    }
    if (preprocess.level_count_ < 1 || preprocess.level_count_ > kMaxPyramidLevels) {
        return Status::kInvalidArgument;
    }

    PyramidRef pyramid{};
    const Status assembled = make_pyramid_ref(preprocess, &pyramid);
    if (assembled != Status::kOk) {
        return assembled;
    }

    const int min_perimeter = 4 * config.min_side_length_canonical_img_px_;
    warp_canonical_kernel<<<static_cast<unsigned int>(candidates.capacity_),
                            static_cast<unsigned int>(kCanonicalThreads), 0, stream>>>(
            pyramid, candidates.corner_x_, candidates.corner_y_, candidates.perimeter_,
            candidates.capacity_, candidates.count_, canonical->images_, canonical->side_px_,
            plan.segmentation_width_px_, min_perimeter, config.use_aruco3_detection_);
    return check_kernel_launch("cell_sample.warp_canonical_kernel", -1, false, stream);
}

}  // namespace aruco3cuda::detail
