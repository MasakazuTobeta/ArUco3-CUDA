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
/// 所有権:
///   出力先 ostream は非所有の参照として保持する。解放は行わない。
///   呼出側は writer より長く ostream を生存させる必要がある。
///   引数として受け取る文字列は複製して出力し、保持しない。
///   **この所有権は以下の全ての public member 関数に適用される。**
///
/// 同期動作:
///   host 専用であり同期点を持たない。内部に書式状態を持つため、
///   1 つの instance を複数 thread から同時に使用してはならない。
///   **この同期動作は以下の全ての public member 関数に適用される。**
///
/// 制約:
///   非有限値 (NaN、無限大) は JSON で表現できないため null として出力する。
///
/// 入力例:
///   std::ostringstream out;
///   JsonWriter writer(out, 0);
///   writer.begin_object();
///   writer.member_int("a", 1);
///   writer.end_object();
/// 出力例:
///   {"a":1}
class JsonWriter {
public:
    /// @param out 出力先。writer の生存期間中に有効である必要がある。所有権は移らない。
    /// @param indent_width 入れ子 1 段あたりの空白数。0 で改行と字下げを行わず 1 行出力になる。
    ///
    /// 入力例: JsonWriter(out, 0) と JsonWriter(out, 2) で同じ呼び出し列を実行する
    /// 出力例: 前者は {"a":1}、後者は改行と 2 空白の字下げを伴う複数行
    explicit JsonWriter(std::ostream& out, int indent_width = 2);

    /// object を開始する。直前が key でなければ区切りと字下げを自動で入れる。
    ///
    /// @return 無し。出力先 ostream の状態のみが変化する。
    ///
    /// 入力例: indent_width = 0 で begin_object() の後に end_object()
    /// 出力例: {}
    void begin_object();

    /// object を終了する。要素を 1 つ以上書いていれば閉じ括弧の前で改行する。
    ///
    /// @return 無し。
    ///
    /// 入力例: begin_object(); member_int("a", 1); end_object();
    /// 出力例: {"a":1}
    void end_object();

    /// array を開始する。直前が key でなければ区切りと字下げを自動で入れる。
    ///
    /// @return 無し。
    ///
    /// 入力例: begin_array(); value_int(1); value_int(2); end_array();
    /// 出力例: [1,2]
    void begin_array();

    /// array を終了する。
    ///
    /// @return 無し。
    ///
    /// 入力例: begin_array(); end_array();
    /// 出力例: []
    void end_array();

    /// object の member 名を書く。直後に値を 1 つ書く必要がある。
    ///
    /// @param name member 名。JSON 文字列として escape してから出力する。
    ///             引数は複製され、writer は保持しない。
    /// @return 無し。
    ///
    /// 入力例: key("a\"b"); value_int(1);
    /// 出力例: "a\"b":1
    void key(const std::string& name);

    /// 文字列を値として書く。JSON 文字列として escape する。
    ///
    /// @param value 出力する文字列。複製され、writer は保持しない。
    /// @return 無し。
    ///
    /// 入力例: value_string("a\"b")
    /// 出力例: "a\"b"
    void value_string(const std::string& value);

    /// 整数を値として書く。
    ///
    /// @param value 出力する値。
    /// @return 無し。
    ///
    /// 入力例: value_int(-7)
    /// 出力例: -7
    void value_int(long long value);

    /// 真偽値を書く。
    ///
    /// @param value 出力する値。
    /// @return 無し。
    ///
    /// 入力例: value_bool(true)
    /// 出力例: true
    void value_bool(bool value);

    /// null を書く。値が未取得であることと 0 であることを区別するために使う。
    ///
    /// @return 無し。
    ///
    /// 入力例: value_null()
    /// 出力例: null
    void value_null();

    /// 浮動小数点数を固定小数点表記で書く。
    ///
    /// 出力を安定させるため桁数を呼出側が指定する。非有限値 (NaN、無限大) は
    /// JSON で表現できないため、値を捏造せず null を出力する。
    ///
    /// @param value 出力する値。
    /// @param precision 小数点以下の桁数。
    /// @return 無し。
    ///
    /// 入力例: value_double(1.0 / 3.0, 4)
    /// 出力例: 0.3333
    void value_double(double value, int precision);

    /// key() と value_string() をまとめて行う。
    ///
    /// @param name member 名。escape される。
    /// @param value 値。escape される。いずれも複製され writer は保持しない。
    /// @return 無し。
    ///
    /// 入力例: member_string("k", "v")
    /// 出力例: "k":"v"
    void member_string(const std::string& name, const std::string& value);

    /// key() と value_int() をまとめて行う。
    ///
    /// @param name member 名。
    /// @param value 値。
    /// @return 無し。
    ///
    /// 入力例: member_int("k", 3)
    /// 出力例: "k":3
    void member_int(const std::string& name, long long value);

    /// key() と value_bool() をまとめて行う。
    ///
    /// @param name member 名。
    /// @param value 値。
    /// @return 無し。
    ///
    /// 入力例: member_bool("k", false)
    /// 出力例: "k":false
    void member_bool(const std::string& name, bool value);

    /// key() と value_double() をまとめて行う。
    ///
    /// @param name member 名。
    /// @param value 値。
    /// @param precision 小数点以下の桁数。value_double と同じ意味を持つ。
    /// @return 無し。
    ///
    /// 入力例: member_double("k", 0.5, 2)
    /// 出力例: "k":0.50
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

/// JSON 文字列として安全な形へ escape する。
///
/// 引用符と逆斜線を escape し、制御文字は \\u 形式にする。0x20 以上の byte は
/// そのまま通すため、UTF-8 の多 byte 文字は元の並びを保つ。
///
/// @param value 変換対象。複製され、呼出後に元の値は変化しない。
/// @return escape 済みの文字列。引用符は含まない。
///
/// 入力例: "四隅\n"
/// 出力例: 四隅\\n
std::string escape_json_string(const std::string& value);

}  // namespace aruco3cuda::util

#endif  // ARUCO3CUDA_UTIL_JSON_WRITER_HPP
