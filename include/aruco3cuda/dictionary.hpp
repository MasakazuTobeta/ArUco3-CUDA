// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_DICTIONARY_HPP
#define ARUCO3CUDA_DICTIONARY_HPP

#include <cstddef>
#include <cstdint>

#include "aruco3cuda/status.hpp"

namespace aruco3cuda {

/// 1 marker の bit 配列を格納する型。
///
/// 対応する最大 marker size は 7x7 = 49 bit であり 64 bit に収まる。
/// bit 0 が (row 0, col 0)、以降 row-major に並ぶ。未使用の上位 bit は 0 とする。
using MarkerCode = std::uint64_t;

/// 定義済み Dictionary の packed 表現。
///
/// 目的:
///   CUDA から constant memory または global memory へそのまま転送でき、
///   4 回転分の照合を分岐なしに行える形で codeword を保持する。
///
/// 所有権:
///   codes_ は静的記憶域を指す。この構造体は所有権を持たない。
struct DictionaryTable {
    const char* name_ = nullptr;
    int marker_size_ = 0;          ///< 1 辺の bit 数。6x6 なら 6
    int max_correction_bits_ = 0;  ///< OpenCV の maxCorrectionBits
    int code_count_ = 0;           ///< 収録 ID 数
    /// [code_count_ * 4]。index = id * 4 + rotation。
    /// rotation は反時計回りに 90 度ずつ進む。
    const MarkerCode* codes_ = nullptr;

    /// 収録 bit 数。marker_size_ * marker_size_ に等しい。
    int bit_count() const { return this->marker_size_ * this->marker_size_; }
};

/// Dictionary 照合の結果。
struct DictionaryMatch {
    int id_ = -1;
    int rotation_ = 0;  ///< 0 から 3。codes_ の rotation index に対応する
    int distance_ = 0;  ///< 最小 Hamming 距離
};

/// 収録されている定義済み Dictionary の数を返す。
std::size_t builtin_dictionary_count();

/// index 番目の定義済み Dictionary を返す。範囲外では nullptr を返す。
const DictionaryTable* builtin_dictionary_at(std::size_t index);

/// 名前から定義済み Dictionary を探す。見つからない場合は nullptr を返す。
///
/// 入力例: "DICT_ARUCO_MIP_36h12"
const DictionaryTable* find_builtin_dictionary(const char* name);

/// 候補の bit 列を Dictionary と照合する。
///
/// 全 ID の 4 回転に対する Hamming 距離の最小値を求め、
/// max_correction_errors 以下であれば一致とみなす。
///
/// @param table 対象 Dictionary。
/// @param candidate 候補の bit 列。table の bit_count() を超える bit は 0 とする。
/// @param max_correction_errors 許容する最大の Hamming 距離。
/// @param out 照合結果を格納する。一致しなかった場合は id_ に -1 を入れる。
/// @return 照合を実行できた場合は kOk。引数が不正な場合は kInvalidArgument。
///
/// 備考:
///   一致しなかったことは失敗ではないため戻り値では区別しない。
///   呼出側は out->id_ が -1 かどうかで判断する。
///
/// 入力例: 6x6 Dictionary、candidate = ID 42 の正しい codeword
/// 出力例: out->id_ = 42、out->rotation_ = 0、out->distance_ = 0
/// 入力例: どの codeword とも遠い candidate
/// 出力例: out->id_ = -1
Status match_dictionary(const DictionaryTable& table, MarkerCode candidate,
                        int max_correction_errors, DictionaryMatch* out);

/// Dictionary 内の最小 Hamming 距離を計算する。
///
/// 全 ID の組み合わせについて、4 回転を含めた最小距離を求める。
/// 公称値との一致確認に使用する。code_count が大きい Dictionary では
/// O(n^2) となるため、test での使用を想定する。
///
/// @param table 対象 Dictionary。
/// @param out_distance 成功時に最小距離を格納する。
/// @return kOk または kInvalidArgument。
Status minimum_hamming_distance(const DictionaryTable& table, int* out_distance);

/// bit 配列から MarkerCode を組み立てる。
///
/// @param bits 各要素が 0 または 1。長さは marker_size * marker_size。
/// @param marker_size 1 辺の bit 数。
/// @param out 成功時に packed 値を格納する。
/// @return kOk または kInvalidArgument。
Status pack_marker_code(const std::uint8_t* bits, int marker_size, MarkerCode* out);

/// MarkerCode を bit 配列へ展開する。pack_marker_code の逆変換。
Status unpack_marker_code(MarkerCode code, int marker_size, std::uint8_t* out_bits);

/// bit 配列を反時計回りに 90 度回転した MarkerCode を返す。
///
/// @param code 元の値。
/// @param marker_size 1 辺の bit 数。
/// @param out 成功時に回転後の値を格納する。
/// @return kOk または kInvalidArgument。
Status rotate_marker_code(MarkerCode code, int marker_size, MarkerCode* out);

}  // namespace aruco3cuda

#endif  // ARUCO3CUDA_DICTIONARY_HPP
