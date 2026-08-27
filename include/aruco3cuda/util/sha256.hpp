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
class Sha256 {
public:
    Sha256();

    /// data の先頭 size byte を取り込む。複数回呼び出せる。
    void update(const void* data, std::size_t size);

    /// 取り込んだ内容の hash を 64 文字の小文字 16 進文字列で返す。
    /// 呼出後は再利用しない。
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
/// 入力例: "abc" の 3 byte
/// 出力例: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
std::string sha256_bytes(const void* data, std::size_t size);

}  // namespace aruco3cuda::util

#endif  // ARUCO3CUDA_UTIL_SHA256_HPP
