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

/// pyramid の段数の上限。
///
/// 段数は log2 で決まるため、65536x65536 の入力でも 16 段に収まる。
/// 固定長にすることで、段ごとの動的確保を避ける。
inline constexpr int kMaxPyramidLevels = 24;

/// ArUco3 の縮小に関する計画。
///
/// [検出パイプライン設計](../../docs/design/detector-pipeline.md) に記録した
/// OpenCV 4.x の観測仕様をそのまま実装する。
///
/// 所有権: 値のみを持ち、外部の資源を参照しない。複製して保持してよい。
/// 同期動作: 単なる値の集合であり同期点を持たない。
///
/// 入力例: 既定設定で 1280x720 を plan_scales() へ渡す
/// 出力例: fxfy_ = 0.3333、segmentation 427x240、level_count_ = 5
struct ScalePlan {
    /// 縮小率。ArUco3 が無効なら 1。
    double fxfy_ = 1.0;
    int segmentation_width_px_ = 0;
    int segmentation_height_px_ = 0;
    /// level 0 を含む pyramid の総段数。
    int level_count_ = 1;
    /// 四隅の upsampling を開始する level。
    int closest_level_index_ = 0;
};

/// workspace 上の 8-bit 画像面。
///
/// 所有権: data_ が指す領域の所有権は workspace にある。この構造体は
///         参照のみを持ち、複製も解放も行わない。workspace が reset() または
///         破棄されると data_ は無効になる。
/// 同期動作: 単なる参照の集合であり同期点を持たない。data_ が指す内容は
///           発行済みの kernel が完了するまで確定しない。
///
/// 入力例: reserve_preprocess() が切り出した level 1 の面
/// 出力例: width_px_ = 640、height_px_ = 360、pitch_bytes_ = 640
struct ImagePlaneU8 {
    std::uint8_t* data_ = nullptr;
    int width_px_ = 0;
    int height_px_ = 0;
    std::size_t pitch_bytes_ = 0;
};

/// pyramid の各 level を kernel へ渡すための参照。
///
/// level ごとに pitch が異なるため、一律に扱ってはならない。level 0 は
/// 呼出側の入力画像を指し、pitch も呼出側のものである。
///
/// 値渡しできる POD にしてある。kernel の引数へそのまま置ける。
///
/// 所有権: data_ が指す領域の所有権は呼出側と workspace にある。この構造体は
///         参照のみを持ち、複製も解放も行わない。
/// 同期動作: 単なる参照の集合であり同期点を持たない。
///
/// 入力例: 1280x720 に対する PreprocessBuffers
/// 出力例: level_count_ = 5、width_[0] = 1280、width_[1] = 640
struct PyramidRef {
    const std::uint8_t* data_[kMaxPyramidLevels] = {};
    int width_[kMaxPyramidLevels] = {};
    int height_[kMaxPyramidLevels] = {};
    std::size_t pitch_[kMaxPyramidLevels] = {};
    int level_count_ = 0;
};

/// 前処理が使用する buffer 一式。
///
/// level 0 は入力そのものを指し、複製しない。複製は 1 フレームあたり
/// W*H byte の転送になり、統合 GPU でも無視できない。
///
/// 所有権: level0_ が指す領域の所有権は呼出側にあり、levels_ と
///         segmentation_ が指す領域の所有権は workspace にある。この構造体は
///         いずれについても参照のみを持ち、解放しない。workspace を reset()
///         または破棄すると levels_ と segmentation_ は無効になる。
/// 同期動作: 単なる参照の集合であり同期点を持たない。
///
/// 入力例: 1280x720 の入力に対する reserve_preprocess() の出力
/// 出力例: level_count_ = 5、segmentation_ が 427x240
struct PreprocessBuffers {
    ImageViewU8 level0_;
    /// level 1 以降。index 0 が level 1 に対応する。
    ImagePlaneU8 levels_[kMaxPyramidLevels];
    int level_count_ = 1;
    ImagePlaneU8 segmentation_;
};

/// 縮小率と pyramid 段数を求める。
///
/// @param config 検出設定。use_aruco3_detection_ が false なら fxfy は 1 になる。
/// @param width_px 入力の幅。1 以上。
/// @param height_px 入力の高さ。1 以上。
/// @param out 成功時に計画を格納する。失敗時は変更しない。nullptr は不可。
/// @return kOk、または kInvalidArgument。
///
/// 所有権: 引数の領域を保持しない。
/// 同期動作: host 専用であり同期点を持たない。CUDA API を呼ばない。
///
/// 入力例: 既定設定、1280x720
/// 出力例: fxfy_ = 0.3333、segmentation 427x240
Status plan_scales(const DetectorConfig& config, int width_px, int height_px, ScalePlan* out);

/// 計画に必要な workspace の容量を返す。
///
/// @param plan 縮小計画。
/// @param width_px 入力の幅。
/// @param height_px 入力の高さ。
/// @return 必要な byte 数。桁溢れする場合は 0 を返す。
///
/// 所有権: 資源を保持しない。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: 1280x720 の既定計画
/// 出力例: pyramid と segmentation を収める byte 数
std::size_t preprocess_workspace_bytes(const ScalePlan& plan, int width_px, int height_px);

/// workspace から前処理用の領域を切り出す。
///
/// @param plan 縮小計画。
/// @param input 入力画像。level 0 として参照する。
/// @param workspace 切り出し元。呼出側が所有する。
/// @param out 成功時に buffer 一式を格納する。nullptr は不可。
/// @return kOk。容量不足なら kInvalidConfig、引数が不正なら kInvalidArgument。
///
/// 所有権: 切り出した領域の所有権は workspace に残る。out は参照のみを持つ。
/// 同期動作: host 専用であり同期点を持たない。CUDA API を呼ばない。
///
/// 入力例: 1280x720 の入力と十分な容量の workspace
/// 出力例: levels_ と segmentation_ に workspace 内の pointer が入る
Status reserve_preprocess(const ScalePlan& plan, const ImageViewU8& input, Workspace& workspace,
                          PreprocessBuffers* out);

/// 画像 pyramid を構築する。
///
/// level 0 は入力を参照するため書き込まない。level 1 以降を順に生成する。
/// OpenCV の `pyrDown` と同じ [1,4,6,4,1] の分離型 kernel、境界は
/// BORDER_REFLECT_101、丸めは (sum + 128) >> 8 とする。
///
/// @param buffers reserve_preprocess が返した buffer 一式。nullptr は不可。
/// @param config 検出設定。block 寸法に使用する。
/// @param stream 発行先の stream。既定 stream を使う場合は nullptr。
/// @return kOk、または kInvalidArgument、kCudaError。
///
/// 所有権: buffers が指す領域の所有権は workspace に残る。
/// 同期動作: stream へ kernel を発行するだけで host 同期を行わない。
///           結果が必要な時点で呼出側が同期する。
///
/// 入力例: level_count_ = 4 の buffers
/// 出力例: levels_[0] から levels_[2] が生成される
Status build_pyramid_async(PreprocessBuffers* buffers, const DetectorConfig& config,
                           cudaStream_t stream);

/// segmentation 画像を生成する。
///
/// 候補抽出はこの画像だけで行う。縮小率が 1 の場合も、以降の段階が
/// 同じ buffer を参照できるよう複製する。
///
/// @param plan 縮小計画。
/// @param buffers reserve_preprocess が返した buffer 一式。nullptr は不可。
/// @param config 検出設定。block 寸法に使用する。
/// @param stream 発行先の stream。既定 stream を使う場合は nullptr。
/// @return kOk、または kInvalidArgument、kCudaError。
///
/// 所有権: buffers が指す領域の所有権は workspace に残る。
/// 同期動作: stream へ kernel を発行するだけで host 同期を行わない。
///
/// 入力例: fxfy_ = 0.3333 の計画と 1280x720 の入力
/// 出力例: segmentation_ が 427x240 で埋まる
Status build_segmentation_async(const ScalePlan& plan, PreprocessBuffers* buffers,
                                const DetectorConfig& config, cudaStream_t stream);

/// 指定 level を読み取り用の view として返す。
///
/// @param buffers buffer 一式。
/// @param level 0 以上 level_count_ 未満。範囲外では data_ が nullptr の view を返す。
/// @return level に対応する view。所有権は移らない。
///
/// 所有権: 参照のみを返す。呼出側は解放しない。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: level = 0
/// 出力例: 入力そのものを指す view
ImageViewU8 level_view(const PreprocessBuffers& buffers, int level);

/// PreprocessBuffers から kernel へ渡す参照を組み立てる。
///
/// 段ごとに level_view() を呼んで詰め直すだけである。同じ組み立てを段を
/// 使う kernel ごとに書くと、level 0 が入力そのものを指すという約束が
/// 散らばるため、1 箇所にまとめる。
///
/// @param buffers 前処理の buffer 一式。
/// @param out 成功時に参照を格納する。領域の所有権は呼出側にある。
/// @return kOk。out が nullptr、または段数が範囲外なら kInvalidArgument。
///
/// 所有権: buffers が指す領域を保持しない。out は参照だけを持つ。
/// 同期動作: host 専用であり同期点を持たない。CUDA API を呼ばない。
///
/// 入力例: level_count_ = 5 の PreprocessBuffers
/// 出力例: kOk。out->level_count_ = 5
Status make_pyramid_ref(const PreprocessBuffers& buffers, PyramidRef* out);

}  // namespace aruco3cuda::detail

#endif  // ARUCO3CUDA_CORE_PREPROCESS_HPP
