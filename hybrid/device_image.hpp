// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_HYBRID_DEVICE_IMAGE_HPP
#define ARUCO3CUDA_HYBRID_DEVICE_IMAGE_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"

/// host の画像を device memory へ載せる補助。
///
/// 目的:
///     検出器は device から読める memory を指す ImageViewU8 を受け取り、
///     転送は行わない。呼出側が毎回 cudaMalloc と cudaMemcpy2D を書くと
///     同じ code が散らばり、pitch の扱いを取り違える箇所が増える。
namespace aruco3cuda::hybrid {

/// 画像 1 枚分の device buffer。
///
/// 所有権: 確保した device memory を自身で所有し、破棄時に解放する。
/// 同期動作: upload() は同期版の転送を行い、戻った時点で転送は完了している。
///           1 つの instance を複数 thread から同時に使ってはならない。
///
/// 入力例: reserve(1280, 720) のあと upload() を毎 frame
/// 出力例: view() が同じ device pointer を指し続ける。確保は 1 度だけ
class DeviceImage {
public:
    DeviceImage();
    ~DeviceImage();
    DeviceImage(const DeviceImage&) = delete;
    DeviceImage& operator=(const DeviceImage&) = delete;
    DeviceImage(DeviceImage&&) noexcept;
    DeviceImage& operator=(DeviceImage&&) noexcept;

    /// device buffer を確保する。既に十分な大きさなら何もしない。
    ///
    /// @param width_px 画像の幅。1 以上。
    /// @param height_px 画像の高さ。1 以上。
    /// @param out_message 失敗時に理由を格納する。nullptr 可。
    /// @return kOk、または kInvalidArgument、kCudaError。
    ///
    /// 所有権: 確保した領域は本 instance が所有する。
    /// 同期動作: cudaMalloc を呼ぶ。frame ごとの確保を避けるため、
    ///           測定や実時間処理では初期化時に 1 度だけ呼ぶこと。
    ///
    /// 入力例: 1280 と 720
    /// 出力例: kOk。view() が有効になる
    Status reserve(int width_px, int height_px, std::string* out_message = nullptr);

    /// memory 種別を指定して領域を確保する。
    ///
    /// 種別ごとに確保のしかたと upload() の意味が変わる。
    ///
    /// | space | 確保 | upload() |
    /// | --- | --- | --- |
    /// | kDevice | cudaMallocPitch | host から device へ転送する |
    /// | kManaged | cudaMallocManaged | 同じ領域を host と device が見る。転送は起きない |
    ///
    /// **転送元が page-locked かどうかはここでは扱いません。** 渡された
    /// pointer をそのまま読みます。page-locked な入力を用意するのは呼出側の
    /// 責務です。この class が中継へ写すと、「入力 buffer の種別」ではなく
    /// 「写しの費用」を測ることになります。
    ///
    /// @param space 確保する空間。kDevice、kHostPageable、kHostPinned は
    ///              いずれも device 側へ確保する。kManaged だけが別扱い。
    /// @param width_px 幅。1 以上。
    /// @param height_px 高さ。1 以上。
    /// @param out_message 失敗時に理由を格納する。nullptr を渡してよい。
    /// @return kOk、または kInvalidArgument、kCudaError。
    ///
    /// 所有権: 確保した領域は本 instance が所有する。
    /// 同期動作: 確保のみで同期点を持たない。
    ///
    /// 入力例: MemorySpace::kManaged、1280、720
    /// 出力例: kOk。view() の space_ が kManaged になる
    Status reserve(MemorySpace space, int width_px, int height_px,
                   std::string* out_message = nullptr);

    /// host の 8-bit grayscale 画像を device へ転送する。
    ///
    /// @param data host 側の先頭 pointer。nullptr は不可。
    /// @param width_px 画像の幅。reserve した幅以下である必要がある。
    /// @param height_px 画像の高さ。reserve した高さ以下である必要がある。
    /// @param source_pitch_bytes host 側の 1 行の byte 数。
    /// @param out_message 失敗時に理由を格納する。nullptr 可。
    /// @return kOk、または kInvalidArgument、kCudaError。
    ///
    /// 所有権: 引数の host memory は呼出側が所有する。転送後は参照しない。
    /// 同期動作: 同期版の cudaMemcpy2D を使う。戻った時点で転送は完了している。
    ///
    /// 入力例: cv::Mat の data、cols、rows、step
    /// 出力例: kOk。view() の指す領域が更新される
    Status upload(const std::uint8_t* data, int width_px, int height_px,
                  std::size_t source_pitch_bytes, std::string* out_message = nullptr);

    /// 現在の内容を指す view を返す。
    ///
    /// @return device memory を指す view。reserve 前は data_ が nullptr。
    ///
    /// 所有権: 参照先は本 instance が所有する。
    /// 同期動作: 無し。
    ///
    /// 入力例: reserve と upload 済みの instance
    /// 出力例: space_ が kDevice の view
    const ImageViewU8& view() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace aruco3cuda::hybrid

#endif  // ARUCO3CUDA_HYBRID_DEVICE_IMAGE_HPP
