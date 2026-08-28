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

    /// 収録 bit 数を返す。
    ///
    /// @return marker_size_ * marker_size_。marker_size_ が 0 なら 0。
    ///
    /// 所有権: 資源を保持しない。
    /// 同期動作: host と device のいずれからも呼べる純粋な計算であり同期点を持たない。
    ///
    /// 入力例: marker_size_ = 6 の table
    /// 出力例: 36
    int bit_count() const { return this->marker_size_ * this->marker_size_; }
};

/// Dictionary 照合の結果。
struct DictionaryMatch {
    int id_ = -1;
    int rotation_ = 0;  ///< 0 から 3。codes_ の rotation index に対応する
    int distance_ = 0;  ///< 最小 Hamming 距離
};

/// セル比から作る 2 つの bit mask。
///
/// 目的:
///   セル比は 0 か 1 とは限らない。OpenCV は 1 つの bit 列ではなく、
///   「黒ではない」と「白ではない」の 2 つの mask で候補を表す。閾値の
///   近くにある比はどちらの mask にも立ち、bit が 0 でも 1 でも誤りとして
///   数えられる。1 つの bit 列に潰すとこの区別が失われる。
///
/// 所有権:
///   値のみを持つ。
/// 同期動作:
///   値のみを持ち同期点を持たない。device 側へそのまま渡してよい。
///
/// 入力例: 閾値 0.49、比が {1.0, 0.0, 0.5, 1.0} の 2x2
/// 出力例: not_black_ = 0b1101、not_white_ = 0b0110
struct CellMasks {
    /// 比が閾値を超えたセル。bit 0 が (row 0, col 0) で row-major。
    MarkerCode not_black_ = 0;
    /// 比が 1 から閾値を引いた値を下回ったセル。並びは not_black_ と同じ。
    MarkerCode not_white_ = 0;
};

/// 収録されている定義済み Dictionary の数を返す。
///
/// @return 収録数。0 にはならない。
///
/// 所有権: 資源を保持しない。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: 引数なし
/// 出力例: 1
std::size_t builtin_dictionary_count();

/// index 番目の定義済み Dictionary を返す。
///
/// @param index 0 以上 builtin_dictionary_count() 未満。範囲外では nullptr を返す。
/// @return 静的記憶域を持つ table への pointer。範囲外では nullptr。
///
/// 所有権: 戻り値は静的記憶域を指す。呼出側は解放も変更もしない。
///         table が指す codes_ も同じく静的記憶域である。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: 0
/// 出力例: name_ が "DICT_ARUCO_MIP_36h12" の table
const DictionaryTable* builtin_dictionary_at(std::size_t index);

/// 名前から定義済み Dictionary を探す。
///
/// @param name 探す名前。nullptr を渡してよく、その場合は nullptr を返す。
/// @return 静的記憶域を持つ table への pointer。見つからない場合は nullptr。
///
/// 所有権: 戻り値は静的記憶域を指す。呼出側は解放も変更もしない。
///         引数の name は複製も保持もしない。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: "DICT_ARUCO_MIP_36h12"
/// 出力例: marker_size_ = 6、code_count_ = 250 の table
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
///            領域の所有権は呼出側にある。
/// @return 照合を実行できた場合は kOk。引数が不正な場合は kInvalidArgument。
///
/// 所有権: table が指す codes_ は参照するだけで保持しない。
/// 同期動作: host 専用であり同期点を持たない。CUDA 実装では同じ規則を
///           device 側の関数として再実装する。
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

/// セル比から CellMasks を組み立てる。
///
/// OpenCV の `CellBitMasks` と同じ規則で、比を 2 つの mask へ振り分ける。
/// 比が閾値を超えれば not_black_ に、1 から閾値を引いた値を下回れば
/// not_white_ に bit を立てる。既定の閾値 0.49 では 0.49 より大きく
/// 0.51 より小さい比が両方に立ち、bit が 0 でも 1 でも誤りになる。
///
/// @param ratios セル比。長さは marker_size * marker_size 以上で row-major。
///               領域の所有権は呼出側にあり、この関数は複製も保持もしない。
/// @param marker_size 1 辺のセル数。1 以上 7 以下。border は含まない。
/// @param valid_bit_threshold 比を bit とみなす閾値。OpenCV の
///                            `validBitIdThreshold` に対応する。
/// @param out 成功時に mask を格納する。領域の所有権は呼出側にある。
/// @return kOk。ratios か out が nullptr、または marker_size が範囲外なら
///         kInvalidArgument。
///
/// 所有権: 引数の領域を保持しない。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: ratios = {1.0, 0.0, 0.0, 1.0}、marker_size = 2、閾値 0.49
/// 出力例: out->not_black_ = 0b1001、out->not_white_ = 0b0110
Status build_cell_masks(const float* ratios, int marker_size, float valid_bit_threshold,
                        CellMasks* out);

/// OpenCV の `Dictionary::identify` と同じ規則で候補を識別する。
///
/// match_dictionary との違いは 2 つある。第 1 に、比の曖昧さを 2 つの mask で
/// 扱う。第 2 に、全 ID の最小距離を求めるのではなく、**ID の昇順に見て許容
/// 距離を満たした最初の ID で打ち切る**。収録間の最小距離が許容距離の 2 倍を
/// 超える Dictionary では両者は一致するが、そうでない場合は結果が変わりうる。
/// OpenCV との一致を優先してこちらを検出経路で使う。
///
/// 許容距離は `int(max_correction_bits_ * error_correction_rate)` である。
/// 小数は 0 方向へ切り捨てる。
///
/// @param table 対象 Dictionary。
/// @param masks 候補のセル比から作った mask。
/// @param error_correction_rate OpenCV の `errorCorrectionRate`。0 以上 1 以下。
/// @param out 照合結果を格納する。一致しなかった場合は id_ に -1 を入れ、
///            distance_ には全 ID を通した最小距離を入れる。
///            領域の所有権は呼出側にある。
/// @return 照合を実行できた場合は kOk。引数が不正な場合は kInvalidArgument。
///
/// 所有権: table が指す codes_ は参照するだけで保持しない。
/// 同期動作: host 専用であり同期点を持たない。CUDA 実装は同じ規則を
///           device 側で再実装する。
///
/// 入力例: 6x6 Dictionary、ID 42 の codeword そのままの比から作った mask
/// 出力例: out->id_ = 42、out->rotation_ = 0、out->distance_ = 0
/// 入力例: どの codeword とも遠い mask
/// 出力例: out->id_ = -1
Status identify_marker(const DictionaryTable& table, const CellMasks& masks,
                       double error_correction_rate, DictionaryMatch* out);

/// Dictionary 内の最小 Hamming 距離を計算する。
///
/// 全 ID の組み合わせについて、4 回転を含めた最小距離を求める。
/// 公称値との一致確認に使用する。code_count が大きい Dictionary では
/// O(n^2) となるため、test での使用を想定する。
///
/// @param table 対象 Dictionary。
/// @param out_distance 成功時に最小距離を格納する。領域の所有権は呼出側にある。
/// @return kOk または kInvalidArgument。
///
/// 所有権: table が指す codes_ は参照するだけで保持しない。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: DICT_ARUCO_MIP_36h12 の table
/// 出力例: *out_distance = 12
Status minimum_hamming_distance(const DictionaryTable& table, int* out_distance);

/// bit 配列から MarkerCode を組み立てる。
///
/// @param bits 各要素が 0 または 1。長さは marker_size * marker_size 以上。
///             領域の所有権は呼出側にあり、この関数は複製も保持もしない。
/// @param marker_size 1 辺の bit 数。1 以上 7 以下。
/// @param out 成功時に packed 値を格納する。領域の所有権は呼出側にある。
/// @return kOk。bits か out が nullptr、または marker_size が範囲外なら kInvalidArgument。
///
/// 所有権: 引数の領域を保持しない。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: bits = {1, 0, 1, 0}、marker_size = 2
/// 出力例: *out = 0b0101
Status pack_marker_code(const std::uint8_t* bits, int marker_size, MarkerCode* out);

/// MarkerCode を bit 配列へ展開する。pack_marker_code の逆変換。
///
/// @param code 展開対象。
/// @param marker_size 1 辺の bit 数。1 以上 7 以下。
/// @param out_bits 展開先。marker_size * marker_size 要素以上の領域が必要であり、
///                 領域の確保と解放は呼出側の責務である。各要素へ 0 または 1 を書く。
/// @return kOk。out_bits が nullptr、または marker_size が範囲外なら kInvalidArgument。
///
/// 所有権: 引数の領域を保持しない。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: code = 0b101、marker_size = 2
/// 出力例: out_bits = {1, 0, 1, 0}
Status unpack_marker_code(MarkerCode code, int marker_size, std::uint8_t* out_bits);

/// bit 配列を反時計回りに 90 度回転した MarkerCode を返す。
///
/// @param code 元の値。
/// @param marker_size 1 辺の bit 数。1 以上 7 以下。
/// @param out 成功時に回転後の値を格納する。領域の所有権は呼出側にある。
/// @return kOk。out が nullptr、または marker_size が範囲外なら kInvalidArgument。
///
/// 所有権: 引数の領域を保持しない。
/// 同期動作: host 専用であり同期点を持たない。
///
/// 入力例: 2x2 の {1, 0, 0, 0}、marker_size = 2
/// 出力例: 反時計回りに 90 度回した {0, 0, 1, 0}
Status rotate_marker_code(MarkerCode code, int marker_size, MarkerCode* out);

}  // namespace aruco3cuda

#endif  // ARUCO3CUDA_DICTIONARY_HPP
