// SPDX-License-Identifier: Apache-2.0
//
// JSON writer の出力が決定的で、escape と非有限値を正しく扱うことを検証する。
#include "aruco3cuda/util/json_writer.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <sstream>
#include <string>

namespace {

using aruco3cuda::util::JsonWriter;

// 正常系: 入れ子構造を区切りと共に正しく出力する。
TEST(JsonWriterTest, writes_nested_structure_compactly) {
    std::ostringstream out;
    JsonWriter writer(out, 0);
    writer.begin_object();
    writer.member_int("a", 1);
    writer.key("b");
    writer.begin_array();
    writer.value_int(2);
    writer.value_int(3);
    writer.end_array();
    writer.member_bool("c", true);
    writer.end_object();
    EXPECT_EQ(out.str(), "{\"a\":1,\"b\":[2,3],\"c\":true}");
}

// 境界値: 空の object と array を出力できる。
TEST(JsonWriterTest, writes_empty_containers) {
    std::ostringstream out;
    JsonWriter writer(out, 0);
    writer.begin_object();
    writer.key("empty_object");
    writer.begin_object();
    writer.end_object();
    writer.key("empty_array");
    writer.begin_array();
    writer.end_array();
    writer.end_object();
    EXPECT_EQ(out.str(), "{\"empty_object\":{},\"empty_array\":[]}");
}

// 正常系: 制御文字と記号を escape する。
TEST(JsonWriterTest, escapes_special_characters) {
    EXPECT_EQ(aruco3cuda::util::escape_json_string("a\"b\\c"), "a\\\"b\\\\c");
    EXPECT_EQ(aruco3cuda::util::escape_json_string("line\ntab\t"), "line\\ntab\\t");
    EXPECT_EQ(aruco3cuda::util::escape_json_string(std::string(1, '\x01')), "\\u0001");
    // UTF-8 の多 byte 文字はそのまま通す。
    EXPECT_EQ(aruco3cuda::util::escape_json_string("四隅"), "四隅");
}

// 正常系: 指定した桁数で安定して出力する。
TEST(JsonWriterTest, formats_doubles_with_requested_precision) {
    std::ostringstream out;
    JsonWriter writer(out, 0);
    writer.begin_array();
    writer.value_double(1.0 / 3.0, 4);
    writer.value_double(-0.5, 2);
    writer.end_array();
    EXPECT_EQ(out.str(), "[0.3333,-0.50]");
}

// 異常系: JSON で表現できない値は null にする。値を捏造しない。
TEST(JsonWriterTest, writes_null_for_non_finite_values) {
    std::ostringstream out;
    JsonWriter writer(out, 0);
    writer.begin_array();
    writer.value_double(std::nan(""), 4);
    writer.value_double(std::numeric_limits<double>::infinity(), 4);
    writer.end_array();
    EXPECT_EQ(out.str(), "[null,null]");
}

// 正常系: 同じ呼び出し列からは同じ byte 列が得られる。
TEST(JsonWriterTest, output_is_deterministic) {
    auto build = []() {
        std::ostringstream out;
        JsonWriter writer(out, 2);
        writer.begin_object();
        writer.member_string("name", "検出結果");
        writer.member_double("value", 0.1 + 0.2, 6);
        writer.end_object();
        return out.str();
    };
    EXPECT_EQ(build(), build());
}

}  // namespace
