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
/// 1 隅を担当する block の thread 数。
///
/// 1 反復で触る要素は patch が最大 13x13 = 169、勾配が最大 11x11 = 121 で
/// ある。169 を 1 巡で覆える最小の warp の倍数にする。
constexpr int kRefineThreads = 192;
/// 起こす block 数の上限。
///
/// 1 block が共有 memory を約 5.5 KB 使うため、起こすだけでも費用がかかる。
/// 検出が 0 件の frame でも block 数だけの起動費用は払うため、必要以上に
/// 起こさない。評価計画の上限であるマーカー 16 枚 = 64 隅を 1 波で覆う数に
/// する。これを超える隅は block が跨ぎながら処理する。
constexpr int kRefineBlocks = 64;
/// OpenCV が cornerSubPix 内で反復回数へ掛ける上限。
constexpr int kMaxIterations = 100;
/// 窓の半径を切り替える段の大きさ。
constexpr int kLargeLevelSidePx = 1080;
/// 大きい段と小さい段で使う窓の半径。
constexpr int kLargeWinRadius = 5;
constexpr int kSmallWinRadius = 3;
/// 窓の 1 辺の上限。半径 5 のとき 11。
constexpr int kMaxWinSide = (kLargeWinRadius * 2) + 1;

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

/// patch の切り出しに必要な、位置から決まる係数と矩形。
///
/// 逐次版が loop の外で 1 度だけ求めていた値をまとめたもの。要素ごとに
/// 並列で計算するため、すべての thread が同じ値を持つ必要がある。
struct PatchSetup {
    bool inside_ = false;
    float a_ = 0.0F;
    float b_ = 0.0F;
    int ip_x_ = 0;
    int ip_y_ = 0;
    // 境界の経路で adjustRect が決める矩形と開始位置。
    long long offset_ = 0;
    int rect_x_ = 0;
    int rect_width_ = 0;
    int rect_y_ = 0;
    int rect_height_ = 0;
};

/// getRectSubPix が使う係数と矩形を求める。
///
/// 内側の経路と境界の経路のどちらへ入るかも、ここで決まる。条件は OpenCV の
/// getRectSubPix_8u32f の速い経路の条件と同じである。
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
        // OpenCV は 0 除算を避けるためここで下限を置く。
        setup.a_ = fmaxf(setup.a_, 0.0001F);
        return setup;
    }

    // adjustRect と同じ矩形と開始位置を求める。
    //
    // このうち 3 つの分岐は cornerSubPix からは到達しない。四隅が画像の内側に
    // あることが呼出前に確かめられているため、patch の原点が画像を完全に
    // 外れることが無い。具体的には `rect_x > patch_side` (原点が左へ 13 px
    // より遠い)、`rect_width < 0` と `rect_height < 0` (原点が右または下へ
    // はみ出しきる) である。実際に変異を入れても test は落ちない。
    // adjustRect の写しとして残すが、test で固定できないことを明記する。
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

/// patch の 1 要素を求める。
///
/// 逐次版と**同じ式、同じ被演算子の順序**で書く。内側の経路では 1 行を
/// 左から右へ辿るが、持ち回りの深さは 1 なので要素ごとの閉じた式になる。
/// 境界の経路では行の進み方が矩形に依存するため、開始位置を i から直接求める。
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
        // previous は 1 つ前の t に scale を掛けたものである。column 0 だけ
        // 別の式になる。
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

    // 逐次版は rect_y から rect_height の行でだけ source を 1 行進める。
    // 進んだ回数を row から直接数える。
    const int advanced = max(0, min(row, s.rect_height_) - s.rect_y_);
    const std::uint8_t* source = image + s.offset_ + (static_cast<long long>(advanced) * pitch);
    const std::uint8_t* second =
            (row < s.rect_y_ || row >= s.rect_height_) ? source : source + pitch;

    // 逐次版の書き込み順は 左端の埋め、右端の埋め、双一次である。後の書き込みが
    // 残るため、判定は右端から順に行う。
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

/// block 内で共有する作業領域。
struct RefineShared {
    float patch_[kMaxRefinePatchSide * kMaxRefinePatchSide];
    // 累積する 5 つの量を要素ごとに置く。累積の順序は添字の昇順を保つ。
    double part_[5][kMaxWinSide * kMaxWinSide];
    double sum_[5];
    float x_;
    float y_;
    int stop_;
    int iteration_;
};

/// cv::cornerSubPix を 1 隅について再現する。1 block が 1 隅を担当する。
///
/// 累積される 5 つの量は要素ごとに閉じており、要素間に依存が無い。倍精度の
/// 演算は被演算子の決定的な関数なので、どの thread が計算しても bit 表現は
/// 変わらない。**変えられないのは累積の順序だけ**であり、そこは添字の昇順を
/// 保って 5 本の鎖として足す。5 本は source でも独立な 5 変数であり、
/// 別の lane へ割っても同一の和の中の順序は変わらない。
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
        // 画像の外から始まる場合、CPU 基準は例外を投げる。GPU では投げられない
        // ため、補正せずに元の位置を残し counter で数える。
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

        // 走査は patch の 1 行 1 列だけ内側から始める。添字の対応は逐次版と
        // 同じで、k = i * win_side + j である。
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
                // 誤差は単精度で求めてから倍精度へ広げる。CPU 基準と同じ順序。
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
        // 初期位置から窓の半径より遠ければ、収束が悪いとみなして戻す。
        if (fabsf(shared.x_ - start_x) > static_cast<float>(mask.radius_) ||
            fabsf(shared.y_ - start_y) > static_cast<float>(mask.radius_)) {
            atomicAdd(&counters[kPoorConvergence], 1);
            shared.x_ = start_x;
            shared.y_ = start_y;
        }
    }
    __syncthreads();
}

/// 段を登りながら四隅を補正する。1 block が 1 隅を担当する。
__global__ void refine_kernel(PyramidRef pyramid, DeviceDetections detections, RefineMasks masks,
                              float scale_init, int start_level, int max_iterations, double eps,
                              std::int32_t* counters) {
    const int total = *detections.count_ * kQuadCornerCount;
    __shared__ RefineShared shared;

    // block を隅の上限だけ起こすと、検出が数件でも数千 block を起動すること
    // になる。共有 memory を使うため 1 SM に載る block 数が限られ、起動が
    // 波に分かれて待ち時間になる。Jetson AGX Orin では検出 0 件でも 7 倍
    // 遅くなった。控えめな block 数で起こして隅を跨ぎながら走る。
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
            // segmentation の座標から開始段の座標へ移す。
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
                // 段を 1 つ下げると解像度は 2 倍になる。
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

    // 1 block が 1 隅を担当し、block 数を超える隅は跨ぎながら処理する。
    // 対象機の SM 数と、共有 memory から決まる 1 SM あたりの block 数を
    // 踏まえ、評価計画の上限 (マーカー 16 枚 = 64 隅) を 1 波で覆える数にする。
    const auto corner_count = detections->capacity_ * kQuadCornerCount;
    const auto blocks = static_cast<unsigned int>((corner_count < kRefineBlocks) ? corner_count
                                                                                 : kRefineBlocks);
    refine_kernel<<<blocks, static_cast<unsigned int>(kRefineThreads), 0, stream>>>(
            pyramid, *detections, masks, scale_init, plan.closest_level_index_, max_iterations, eps,
            buffers->diagnostics_.counters_);
    return check_kernel_launch("corner_refine.refine_kernel", -1, false, stream);
}

}  // namespace aruco3cuda::detail
