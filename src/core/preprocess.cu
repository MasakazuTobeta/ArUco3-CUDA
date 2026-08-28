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

/// workspace 上の各面を CUDA の要求へ合わせる境界。
constexpr std::size_t kPlaneAlignment = 256U;

/// BORDER_REFLECT_101 の index 反射。
///
/// 端の画素を重複させずに折り返す。OpenCV の pyrDown の既定境界であり、
/// 一致させないと端 2 列と 2 行が食い違う。
///
/// 入力例: index = -1、size = 8 のとき 1
/// 入力例: index = 8、size = 8 のとき 6
__device__ inline int reflect101(int index, int size) {
    if (size == 1) {
        return 0;
    }
    // 折り返しは繰り返し起こり得るため、範囲へ収まるまで続ける。
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

/// 1 段の縮小を行う。
///
/// thread と data の対応:
///   thread (x, y) が出力の 1 画素だけを書き込む。入力は (2x, 2y) を中心とする
///   5x5 を読む。書き込み先が thread 間で重複しないため競合は発生せず、
///   atomic も block 内同期も不要である。
/// 境界条件:
///   出力範囲外の thread は何もしない。入力の参照は reflect101 で折り返す。
/// 丸め:
///   重みの総和は 256 である。OpenCV と同じ (sum + 128) >> 8 で丸める。
///   中間値を丸めないため、分離型と 2 次元畳み込みの結果は一致する。
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

/// 双線形補間で任意倍率へ縮小する。
///
/// OpenCV の `resize` の `INTER_LINEAR` を 8-bit 向けの固定小数点まで含めて
/// 再現する。単精度で計算すると丸めが 1 階調ずれ、下流の適応的二値化で
/// 画素の白黒が入れ替わり得る。前処理の差を 0 にしておくことで、以降の
/// 段階で現れる差異を候補抽出の設計差だけに絞り込める。
///
/// 固定小数点の規約:
///   係数は 11 bit (2048 倍) の short。水平方向は int で積和し、垂直方向で
///   もう一度積和したうえで (v + (1 << 21)) >> 22 で丸める。
///   OpenCV は 1 - fx と fx をそれぞれ独立に丸めるため、係数の和が 2048 に
///   ならないことがある。この振る舞いまで揃える。
///
/// thread と data の対応:
///   thread (x, y) が出力の 1 画素だけを書き込む。入力は 2x2 を読む。
///   書き込み先が重複しないため競合は発生せず、同期も不要である。
/// 座標規約:
///   src = (dst + 0.5) * scale - 0.5 の half-pixel 中心。
/// 境界条件:
///   出力範囲外の thread は何もしない。参照が端を超える場合は端へ寄せ、
///   対応する係数を 0 にする。外挿は行わない。
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
    // 係数の倍率。OpenCV の INTER_RESIZE_COEF_SCALE と同じ 2^11 である。
    constexpr float kCoefScale = 2048.0F;

    // OpenCV は (dx + 0.5) * scale - 0.5 を倍精度で計算してから単精度へ落とす。
    // 単精度のまま計算すると丸めが 1 階調ずれる。演算の順序と型まで揃える。
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

    // OpenCV は saturate_cast<short> で最近接へ丸める。
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

    // 垂直方向の合成は OpenCV の VResizeLinear<uchar, ...> 特殊化と同じ順序で
    // 行う。この特殊化は一般形の (v + (1 << 21)) >> 22 ではなく、
    // >> 4、beta 倍、>> 16、加算、+2、>> 2 という順序で丸める。SIMD 向けの
    // 実装だが結果が異なるため、順序まで再現しないと 1 階調ずれる。
    const int value = ((beta0 * (top >> 4)) >> 16) + ((beta1 * (bottom >> 4)) >> 16) + 2;
    const int rounded = value >> 2;
    dst[static_cast<std::size_t>(y) * dst_pitch + static_cast<std::size_t>(x)] =
            static_cast<std::uint8_t>(rounded);
}

/// 単純な複製。縮小率が 1 の場合に使う。
///
/// thread と data の対応:
///   thread (x, y) が出力の 1 画素だけを書き込む。競合は発生しない。
/// 境界条件:
///   出力範囲外の thread は何もしない。
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

/// 面 1 枚分の byte 数。桁溢れする場合は 0 を返す。
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

/// pyrDown の出力寸法。OpenCV の既定と同じ切り上げ。
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

    // OpenCV は fxfy を単精度で計算する。倍精度で計算すると、丸めの境界で
    // segmentation 画像の寸法が 1 pixel 食い違うことがある。互換性のため
    // 演算の型まで揃える。
    const float side = static_cast<float>(config.min_side_length_canonical_img_px_);
    const float longest = static_cast<float>(width_px > height_px ? width_px : height_px);
    const float denominator = side + longest * config.min_marker_length_ratio_original_img_;
    // validate() が両方 0 の組み合わせを拒否するため、ここで 0 にはならない。
    if (!(denominator > 0.0F)) {
        return Status::kInvalidArgument;
    }
    const float fxfy = side / denominator;
    plan.fxfy_ = static_cast<double>(fxfy);

    // OpenCV の cvRound は最近接偶数丸めである。std::lround は 0 から遠い側へ
    // 丸めるため、ちょうど 0.5 の場合に 1 pixel 食い違う。lrint を使う。
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

    // OpenCV は num_levels = (int)(log2(W*H / S^2) / 2) を buildPyramid の
    // maxlevel として渡す。level 数はこれに 1 を足した値になる。
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
        // ここも cvRound と同じ最近接偶数丸めを使う。
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
    // level 0 は入力を参照するため確保しない。
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
        // 段ごとに起動の失敗を確認する。まとめて確認すると、どの段で
        // 失敗したかが分からなくなる。同期は呼出側の責務であり行わない。
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
        // 縮小率が 1 の場合。以降の段階が同じ buffer を参照できるよう複製する。
        copy_plane_kernel<<<grid, block, 0, stream>>>(
                source.data_, source.pitch_bytes_, destination.data_, destination.pitch_bytes_,
                destination.width_px_, destination.height_px_);
    } else {
        // OpenCV は inv_scale = dst / src を求めてから scale = 1 / inv_scale と
        // する。src / dst を直接計算すると浮動小数点で一致せず、補間係数が
        // 1 単位ずれる画素が出る。除算の順序まで揃える。
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
