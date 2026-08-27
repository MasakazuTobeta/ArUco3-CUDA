// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_UTIL_JSON_WRITER_HPP
#define ARUCO3CUDA_UTIL_JSON_WRITER_HPP

#include <ostream>
#include <string>
#include <vector>

namespace aruco3cuda::util {

/// 決定的な JSON を出力する最小の writer。
///
/// 目的:
///   評価結果を機械可読形式で保存する。同じ入力からは byte 単位で同じ出力を
///   得られる必要があるため、浮動小数点の桁数を呼出側が明示的に指定する。
///
/// 制約:
///   非有限値 (NaN、無限大) は JSON で表現できないため null として出力する。
class JsonWriter {
public:
    /// @param out 出力先。writer の生存期間中に有効である必要がある。
    /// @param indent_width 入れ子 1 段あたりの空白数。0 で 1 行出力になる。
    explicit JsonWriter(std::ostream& out, int indent_width = 2);

    void begin_object();
    void end_object();
    void begin_array();
    void end_array();

    /// object の member 名を書く。直後に値を 1 つ書く。
    void key(const std::string& name);

    void value_string(const std::string& value);
    void value_int(long long value);
    void value_bool(bool value);
    void value_null();

    /// @param precision 小数点以下の桁数。出力を安定させるため呼出側が指定する。
    void value_double(double value, int precision);

    void member_string(const std::string& name, const std::string& value);
    void member_int(const std::string& name, long long value);
    void member_bool(const std::string& name, bool value);
    void member_double(const std::string& name, double value, int precision);

private:
    void write_separator();
    void write_newline_indent();

    std::ostream& out_;
    int indent_width_;
    int depth_;
    /// 各階層で既に要素を書いたか。区切りの , を判断するために保持する。
    std::vector<bool> has_element_;
    /// 直前が key であれば、値の前に区切りと改行を入れない。
    bool after_key_;
};

/// JSON 文字列として安全な形へ escape する。制御文字は \u 形式にする。
std::string escape_json_string(const std::string& value);

}  // namespace aruco3cuda::util

#endif  // ARUCO3CUDA_UTIL_JSON_WRITER_HPP
