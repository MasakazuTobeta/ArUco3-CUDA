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
constexpr int kRefineThreads = 64;
/// OpenCV が cornerSubPix 内で反復回数へ掛ける上限。
constexpr int kMaxIterations = 100;
/// 窓の半径を切り替える段の大きさ。
constexpr int kLargeLevelSidePx = 1080;
/// 大きい段と小さい段で使う窓の半径。
constexpr int kLargeWinRadius = 5;
constexpr int kSmallWinRadius = 3;

/// 窓の重みと、そこから決まる寸法。
///
/// OpenCV は cornerSubPix の中で毎回 exp を評価する。半径は 3 と 5 の
/// 2 通りしかないため、host で作って値渡しする。
struct RefineMask {
    float weight_[kMaxRefinePatchSide * kMaxRefinePatchSide] = {};
    int radius_ = 0;
    /// 窓の 1 辺。2 * radius_ + 1。
    int win_side_ = 0;
    /// 切り出す patch の 1 辺。win_side_ + 2。
    int patch_side_ = 0;
};

/// 段ごとの窓を 2 通り持つ。
struct RefineMasks {
    RefineMask small_;
    RefineMask large_;
};

/// counter の添字。
enum RefineCounter : int {
    kRefinedCorners = 0,
    kOutOfImageBreaks = 1,
    kPoorConvergence = 2,
    kSingularBreaks = 3,
    kIterationTotal = 4,
};

/// OpenCV の cornerSubPix と同じ重みを作る。
///
/// zeroZone は Size(-1, -1) で呼ばれるため、中央を 0 にする分岐へは入らない。
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

/// getRectSubPix の内側の経路。走る位置が画像に完全に収まる場合に使う。
///
/// 1 行を左から右へ辿り、直前の項を持ち回る。数学的には双一次補間だが、
/// 持ち回りの丸めが積み上がるため、素直な双一次で置き換えると値が変わる。
__device__ void sample_patch_inside(const std::uint8_t* image, std::size_t pitch, int ip_x,
                                    int ip_y, float a, float b, int patch_side, float* patch) {
    const float a12 = a * (1.0F - b);
    const float a22 = a * b;
    const float b1 = 1.0F - b;
    const float b2 = b;
    const double s = (1.0 - static_cast<double>(a)) / static_cast<double>(a);

    for (int row = 0; row < patch_side; ++row) {
        const std::uint8_t* source = image + (static_cast<std::size_t>(ip_y + row) * pitch) +
                                     static_cast<std::size_t>(ip_x);
        const std::uint8_t* next = source + pitch;
        float previous = (1.0F - a) * ((b1 * static_cast<float>(source[0])) +
                                       (b2 * static_cast<float>(next[0])));
        float* destination = patch + (row * patch_side);
        for (int j = 0; j < patch_side; ++j) {
            const float t = (a12 * static_cast<float>(source[j + 1])) +
                            (a22 * static_cast<float>(next[j + 1]));
            destination[j] = previous + t;
            previous = static_cast<float>(static_cast<double>(t) * s);
        }
    }
}

/// getRectSubPix の境界の経路。走る位置が画像からはみ出す場合に使う。
///
/// OpenCV の adjustRect が決める矩形の内側は双一次で、外側は端の値を
/// 縦方向にだけ補間して埋める。行の進み方も OpenCV と同じにする。
__device__ void sample_patch_border(const std::uint8_t* image, std::size_t pitch, int width,
                                    int height, int ip_x, int ip_y, float a, float b,
                                    int patch_side, float* patch) {
    const float a11 = (1.0F - a) * (1.0F - b);
    const float a12 = a * (1.0F - b);
    const float a21 = (1.0F - a) * b;
    const float a22 = a * b;
    const float b1 = 1.0F - b;
    const float b2 = b;

    // adjustRect と同じ矩形と開始位置を求める。
    long long offset = 0;
    int rect_x = 0;
    int rect_width = 0;
    int rect_y = 0;
    int rect_height = 0;
    if (ip_x >= 0) {
        offset += ip_x;
        rect_x = 0;
    } else {
        rect_x = -ip_x;
        if (rect_x > patch_side) {
            rect_x = patch_side;
        }
    }
    if (ip_x < width - patch_side) {
        rect_width = patch_side;
    } else {
        rect_width = width - ip_x - 1;
        if (rect_width < 0) {
            offset += rect_width;
            rect_width = 0;
        }
    }
    if (ip_y >= 0) {
        offset += static_cast<long long>(ip_y) * static_cast<long long>(pitch);
        rect_y = 0;
    } else {
        rect_y = -ip_y;
    }
    if (ip_y < height - patch_side) {
        rect_height = patch_side;
    } else {
        rect_height = height - ip_y - 1;
        if (rect_height < 0) {
            offset += static_cast<long long>(rect_height) * static_cast<long long>(pitch);
            rect_height = 0;
        }
    }
    offset -= rect_x;

    const std::uint8_t* source = image + offset;
    for (int i = 0; i < patch_side; ++i) {
        const std::uint8_t* second = source + pitch;
        if (i < rect_y || i >= rect_height) {
            second -= pitch;
        }
        float* destination = patch + (i * patch_side);

        float edge = (b1 * static_cast<float>(source[rect_x])) +
                     (b2 * static_cast<float>(second[rect_x]));
        for (int j = 0; j < rect_x; ++j) {
            destination[j] = edge;
        }
        edge = (b1 * static_cast<float>(source[rect_width])) +
               (b2 * static_cast<float>(second[rect_width]));
        for (int j = rect_width; j < patch_side; ++j) {
            destination[j] = edge;
        }
        for (int j = rect_x; j < rect_width; ++j) {
            destination[j] = (static_cast<float>(source[j]) * a11) +
                             (static_cast<float>(source[j + 1]) * a12) +
                             (static_cast<float>(second[j]) * a21) +
                             (static_cast<float>(second[j + 1]) * a22);
        }
        if (i < rect_height) {
            source = second;
        }
    }
}

/// getRectSubPix と同じ patch を切り出す。
__device__ void sample_patch(const std::uint8_t* image, std::size_t pitch, int width, int height,
                             float center_x, float center_y, int patch_side, float* patch) {
    const float shift = static_cast<float>(patch_side - 1) * 0.5F;
    const float cx = center_x - shift;
    const float cy = center_y - shift;
    const int ip_x = static_cast<int>(floorf(cx));
    const int ip_y = static_cast<int>(floorf(cy));

    if (ip_x >= 0 && (ip_x + patch_side) < width && ip_y >= 0 && (ip_y + patch_side) < height) {
        float a = cx - static_cast<float>(ip_x);
        const float b = cy - static_cast<float>(ip_y);
        // OpenCV は 0 除算を避けるためここで下限を置く。
        a = fmaxf(a, 0.0001F);
        sample_patch_inside(image, pitch, ip_x, ip_y, a, b, patch_side, patch);
        return;
    }
    const float a = cx - static_cast<float>(ip_x);
    const float b = cy - static_cast<float>(ip_y);
    sample_patch_border(image, pitch, width, height, ip_x, ip_y, a, b, patch_side, patch);
}

/// cv::cornerSubPix を 1 点について再現する。
///
/// 累積は CPU 基準と同じ順序で行う。row-major に 1 要素ずつ足す。並列に
/// 畳み込むと丸めが変わり、反復回数や打ち切りの採否まで変わりうる。
__device__ void corner_sub_pix(const std::uint8_t* image, std::size_t pitch, int width, int height,
                               const RefineMask& mask, int max_iterations, double eps, float* out_x,
                               float* out_y, std::int32_t* counters) {
    const float start_x = *out_x;
    const float start_y = *out_y;
    float current_x = start_x;
    float current_y = start_y;
    // 画像の外から始まる場合、CPU 基準は例外を投げる。GPU では投げられない
    // ため、補正せずに元の位置を残し counter で数える。
    if (!(start_x >= 0.0F && start_x < static_cast<float>(width) && start_y >= 0.0F &&
          start_y < static_cast<float>(height))) {
        atomicAdd(&counters[kOutOfImageBreaks], 1);
        return;
    }

    float patch[kMaxRefinePatchSide * kMaxRefinePatchSide];
    const int win_side = mask.win_side_;
    const int patch_side = mask.patch_side_;
    int iteration = 0;
    double error = 0.0;
    do {
        double a = 0.0;
        double b = 0.0;
        double c = 0.0;
        double bb1 = 0.0;
        double bb2 = 0.0;

        sample_patch(image, pitch, width, height, current_x, current_y, patch_side, patch);

        // 走査は patch の 1 行 1 列だけ内側から始める。
        int k = 0;
        for (int i = 0; i < win_side; ++i) {
            const float* row = patch + ((i + 1) * patch_side) + 1;
            const double py = static_cast<double>(i - mask.radius_);
            for (int j = 0; j < win_side; ++j, ++k) {
                const double m = static_cast<double>(mask.weight_[k]);
                const double tgx =
                        static_cast<double>(row[j + 1]) - static_cast<double>(row[j - 1]);
                const double tgy = static_cast<double>(row[j + patch_side]) -
                                   static_cast<double>(row[j - patch_side]);
                const double gxx = tgx * tgx * m;
                const double gxy = tgx * tgy * m;
                const double gyy = tgy * tgy * m;
                const double px = static_cast<double>(j - mask.radius_);

                a += gxx;
                b += gxy;
                c += gyy;

                bb1 += (gxx * px) + (gxy * py);
                bb2 += (gxy * px) + (gyy * py);
            }
        }

        const double determinant = (a * c) - (b * b);
        if (fabs(determinant) <= (DBL_EPSILON * DBL_EPSILON)) {
            atomicAdd(&counters[kSingularBreaks], 1);
            break;
        }

        const double scale = 1.0 / determinant;
        const float next_x = static_cast<float>(static_cast<double>(current_x) + (c * scale * bb1) -
                                                (b * scale * bb2));
        const float next_y = static_cast<float>(static_cast<double>(current_y) - (b * scale * bb1) +
                                                (a * scale * bb2));
        // 誤差は単精度で求めてから倍精度へ広げる。CPU 基準と同じ順序である。
        const float dx = next_x - current_x;
        const float dy = next_y - current_y;
        error = static_cast<double>((dx * dx) + (dy * dy));
        if (!(next_x >= 0.0F && next_x < static_cast<float>(width) && next_y >= 0.0F &&
              next_y < static_cast<float>(height))) {
            atomicAdd(&counters[kOutOfImageBreaks], 1);
            break;
        }
        current_x = next_x;
        current_y = next_y;
    } while (++iteration < max_iterations && error > eps);

    atomicAdd(&counters[kIterationTotal], iteration);
    // 初期位置から窓の半径より遠ければ、収束が悪いとみなして戻す。
    if (fabsf(current_x - start_x) > static_cast<float>(mask.radius_) ||
        fabsf(current_y - start_y) > static_cast<float>(mask.radius_)) {
        atomicAdd(&counters[kPoorConvergence], 1);
        current_x = start_x;
        current_y = start_y;
    }
    *out_x = current_x;
    *out_y = current_y;
}

/// 段を登りながら四隅を補正する。1 thread が 1 隅を担当する。
__global__ void refine_kernel(PyramidRef pyramid, DeviceDetections detections, RefineMasks masks,
                              float scale_init, int start_level, int max_iterations, double eps,
                              std::int32_t* counters) {
    const int index = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    const int total = *detections.count_ * kQuadCornerCount;
    if (index >= total) {
        return;
    }
    const int detection = index / kQuadCornerCount;
    const int corner = index % kQuadCornerCount;
    const std::size_t slot =
            (static_cast<std::size_t>(corner) * static_cast<std::size_t>(detections.capacity_)) +
            static_cast<std::size_t>(detection);

    float x = detections.corner_x_[slot];
    float y = detections.corner_y_[slot];
    // segmentation の座標から開始段の座標へ移す。
    if (scale_init != 1.0F) {
        x *= scale_init;
        y *= scale_init;
    }
    for (int level = start_level - 1; level >= 0; --level) {
        // 段を 1 つ下げると解像度は 2 倍になる。
        x *= 2.0F;
        y *= 2.0F;
        const int side = max(pyramid.width_[level], pyramid.height_[level]);
        const RefineMask& mask = (side > kLargeLevelSidePx) ? masks.large_ : masks.small_;
        corner_sub_pix(pyramid.data_[level], pyramid.pitch_[level], pyramid.width_[level],
                       pyramid.height_[level], mask, max_iterations, eps, &x, &y, counters);
    }
    detections.corner_x_[slot] = x;
    detections.corner_y_[slot] = y;
    atomicAdd(&counters[kRefinedCorners], 1);
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

Status refine_corners_async(const PyramidRef& pyramid, const ScalePlan& plan,
                            const DetectorConfig& config, CornerRefineBuffers* buffers,
                            DeviceDetections* detections, cudaStream_t stream) {
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
    // OpenCV は反復回数を 1 以上 100 以下に丸め、収束の閾値を 2 乗して使う。
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

    const auto capacity = static_cast<unsigned int>(detections->capacity_ * kQuadCornerCount);
    const unsigned int blocks = (capacity + static_cast<unsigned int>(kRefineThreads) - 1U) /
                                static_cast<unsigned int>(kRefineThreads);
    refine_kernel<<<blocks, static_cast<unsigned int>(kRefineThreads), 0, stream>>>(
            pyramid, *detections, masks, scale_init, plan.closest_level_index_, max_iterations, eps,
            buffers->diagnostics_.counters_);
    return check_kernel_launch("corner_refine.refine_kernel", -1, false, stream);
}

}  // namespace aruco3cuda::detail
