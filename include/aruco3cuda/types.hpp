// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_TYPES_HPP
#define ARUCO3CUDA_TYPES_HPP

#include <cstddef>
#include <cstdint>
#include <string>

#include "aruco3cuda/status.hpp"

namespace aruco3cuda {

/// 入力 buffer が存在する memory 空間。
///
/// DGX Spark GB10 と Jetson Orin はいずれも統合 GPU であり、host と device が
/// 同一物理 memory を共有する。それでも明示的な copy、page lock、managed の
/// 移送費用は異なるため、経路とは独立した測定軸として型で区別する。
enum class MemorySpace : int {
    kHostPageable = 0,  ///< 通常の host memory。転送に page lock が発生する
    kHostPinned,        ///< page-locked host memory
    kManaged,           ///< managed memory。明示的な copy なし
    kDevice,            ///< device 常駐
};

/// MemorySpace を評価計画の表記へ変換する。
///
/// @param space 変換対象。列挙に無い値でも nullptr を返さない。
/// @return 静的記憶域を持つ文字列。
///
/// 所有権: 戻り値は静的記憶域を指す。呼出側は解放も変更もしない。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: MemorySpace::kHostPinned
/// 出力例: "M-Pinned"
const char* to_string(MemorySpace space);

/// 8-bit grayscale 画像の非所有 view。
///
/// 目的:
///   呼出側が確保した画像を、所有権を移さずに検出器へ渡す。ROI と
///   非連続な行配置を最初から扱えるよう、幅とは独立に行間隔を持たせる。
///
/// 所有権:
///   data_ が指す領域の所有権は呼出側にある。この構造体は複製も解放も行わない。
///   検出の間、領域は有効であり続ける必要がある。
struct ImageViewU8 {
    /// 画像先頭の pointer。space_ が示す memory 空間に属する必要がある。
    const std::uint8_t* data_ = nullptr;
    int width_px_ = 0;
    int height_px_ = 0;
    /// 行間隔。単位は byte。width_px_ と等しいとは限らない。
    std::size_t pitch_bytes_ = 0;
    MemorySpace space_ = MemorySpace::kDevice;
};

/// 扱える画像寸法の上限。
///
/// 外部入力を信頼しないための上限であり、性能上の制約ではない。
/// 評価計画の最大解像度 3840x2160 に対して十分な余裕を取る。
inline constexpr int kMaxImageWidthPx = 65536;
inline constexpr int kMaxImageHeightPx = 65536;

/// 画像 view を境界で検証する。
///
/// 検出器の公開 API は最初にこの検証を行う。不正な view をそのまま
/// CUDA へ渡すと、範囲外 access が非同期の失敗として離れた場所で現れ、
/// 原因の特定が難しくなる。
///
/// 検証項目:
///   - data_ が nullptr でない
///   - width_px_ と height_px_ が 1 以上で上限以下
///   - pitch_bytes_ が 1 行分の byte 数以上
///   - pitch_bytes_ と height_px_ の積が size_t で表現できる
///   - space_ が列挙のいずれかである
///
/// @param image 検証対象。
/// @param out_message 失敗時に「項目名=値」を含む理由を格納する。nullptr を渡してよい。
///                    成功時は変更しない。領域の所有権は呼出側にある。
/// @return 全て有効なら kOk、そうでなければ kInvalidImage。
///
/// 所有権: 引数の領域を保持しない。
/// 同期動作: host 専用であり同期点を持たない。CUDA API を呼ばないため、
///           device pointer の実在は検証しない。
///
/// 入力例: data_ = 有効な device pointer、1920x1080、pitch_bytes_ = 1920
/// 出力例: Status::kOk
/// 入力例: pitch_bytes_ = 1000 の 1920x1080
/// 出力例: Status::kInvalidImage。out_message に "pitch_bytes=1000" を含む
Status validate_image_view(const ImageViewU8& image, std::string* out_message = nullptr);

}  // namespace aruco3cuda

#endif  // ARUCO3CUDA_TYPES_HPP
