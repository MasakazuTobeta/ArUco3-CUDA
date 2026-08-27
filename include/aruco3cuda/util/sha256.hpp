// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_UTIL_SHA256_HPP
#define ARUCO3CUDA_UTIL_SHA256_HPP

#include <cstddef>
#include <cstdint>
#include <string>

namespace aruco3cuda::util {

/// SHA-256 を逐次計算する。FIPS 180-4 の仕様に基づく独自実装。
///
/// 用途:
///   評価に使用した入力画像と結果 JSON を対応付けるための checksum。
///   暗号用途ではなく、同一性の確認に使用する。
///
/// 所有権:
///   内部状態は instance が所有し、destructor で解放される。引数として
///   受け取る領域は読み取るだけで、複製も保持もしない。
///   **この所有権は以下の全ての public member 関数に適用される。**
///
/// 同期動作:
///   host 専用であり同期点を持たない。内部に計算途中の状態を持つため、
///   1 つの instance を複数 thread から同時に使用してはならない。
///   **この同期動作は以下の全ての public member 関数に適用される。**
///
/// 入力例:
///   Sha256 hasher;
///   hasher.update("abc", 3);
///   const std::string digest = hasher.finalize();
/// 出力例:
///   digest == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
class Sha256 {
public:
    /// 初期 hash 値を設定した状態で構築する。
    ///
    /// 同期動作: host 専用であり同期点を持たない。
    /// 所有権: 内部 buffer は instance が所有し、destructor で解放される。
    Sha256();

    /// data の先頭 size byte を取り込む。複数回呼び出せる。
    ///
    /// 分割して呼んでも 1 度に呼んでも同じ結果になる。
    ///
    /// @param data 取り込む領域の先頭。nullptr を渡してよく、その場合は何もしない。
    ///             領域の所有権は呼出側にあり、この関数は複製も保持もしない。
    /// @param size 取り込む byte 数。0 を渡してよく、その場合は何もしない。
    /// @return 無し。
    ///
    /// 入力例: update("ab", 2) の後 update("c", 1)
    /// 出力例: finalize() が "abc" の hash を返す
    void update(const void* data, std::size_t size);

    /// 取り込んだ内容の hash を返す。
    ///
    /// padding を適用して計算を確定させるため、呼出後に同じ instance を
    /// 再利用してはならない。再利用すると誤った値を返す。
    ///
    /// @return 64 文字の小文字 16 進文字列。
    ///
    /// 入力例: update("abc", 3) の後に呼ぶ
    /// 出力例: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    std::string finalize();

private:
    void process_block(const std::uint8_t* block);

    std::uint32_t state_[8];
    std::uint64_t total_bits_;
    std::uint8_t buffer_[64];
    std::size_t buffer_size_;
};

/// file 全体の SHA-256 を計算する。
///
/// @param path 対象 file。
/// @param out_hex 成功時に 64 文字の 16 進文字列を格納する。
/// @return 読み取れた場合は true。開けない場合は false で out_hex は変更しない。
bool sha256_file(const std::string& path, std::string* out_hex);

/// memory 上の byte 列の SHA-256 を計算する。
///
/// @param data 対象領域の先頭。nullptr を渡してよく、その場合は空入力として扱う。
///             領域の所有権は呼出側にあり、この関数は複製も保持もしない。
/// @param size 対象の byte 数。
/// @return 64 文字の小文字 16 進文字列。
///
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: "abc" の 3 byte
/// 出力例: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
std::string sha256_bytes(const void* data, std::size_t size);

}  // namespace aruco3cuda::util

#endif  // ARUCO3CUDA_UTIL_SHA256_HPP
