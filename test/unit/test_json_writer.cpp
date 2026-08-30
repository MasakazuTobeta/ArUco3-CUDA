// SPDX-License-Identifier: Apache-2.0
//
// Verifies that the JSON writer produces deterministic output and handles escaping
// and non-finite values correctly.
#include "aruco3cuda/util/json_writer.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <sstream>
#include <string>

namespace {

using aruco3cuda::util::JsonWriter;

// Nominal: nested structures are written with the correct separators.
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

// Boundary: empty objects and arrays can be written.
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

// Nominal: control characters and reserved symbols are escaped.
TEST(JsonWriterTest, escapes_special_characters) {
    EXPECT_EQ(aruco3cuda::util::escape_json_string("a\"b\\c"), "a\\\"b\\\\c");
    EXPECT_EQ(aruco3cuda::util::escape_json_string("line\ntab\t"), "line\\ntab\\t");
    EXPECT_EQ(aruco3cuda::util::escape_json_string(std::string(1, '\x01')), "\\u0001");
    // Multi-byte UTF-8 characters pass through unchanged. The literal below stays
    // non-ASCII on purpose: it is the input under test, and replacing it with ASCII
    // would remove the very property this line checks.
    EXPECT_EQ(aruco3cuda::util::escape_json_string("四隅"), "四隅");
}

// Nominal: doubles are written stably at the requested precision.
TEST(JsonWriterTest, formats_doubles_with_requested_precision) {
    std::ostringstream out;
    JsonWriter writer(out, 0);
    writer.begin_array();
    writer.value_double(1.0 / 3.0, 4);
    writer.value_double(-0.5, 2);
    writer.end_array();
    EXPECT_EQ(out.str(), "[0.3333,-0.50]");
}

// Error case: values JSON cannot represent become null. No value is invented.
TEST(JsonWriterTest, writes_null_for_non_finite_values) {
    std::ostringstream out;
    JsonWriter writer(out, 0);
    writer.begin_array();
    writer.value_double(std::nan(""), 4);
    writer.value_double(std::numeric_limits<double>::infinity(), 4);
    writer.end_array();
    EXPECT_EQ(out.str(), "[null,null]");
}

// Nominal: the same sequence of calls yields the same byte sequence.
// The member value stays non-ASCII on purpose, so that the multi-byte path through
// member_string is covered as well.
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

// Nominal: each value_* function can be called directly, so that no path is only
// ever reached through the member_* wrappers.
// The string value is deliberately non-ASCII: it checks that value_string passes
// multi-byte UTF-8 through to the output byte for byte.
TEST(JsonWriterTest, value_functions_can_be_called_directly) {
    std::ostringstream out;
    JsonWriter writer(out, 0);
    writer.begin_array();
    writer.value_string("文字列");
    writer.value_int(-7);
    writer.value_bool(true);
    writer.value_bool(false);
    writer.value_null();
    writer.value_double(0.5, 2);
    writer.end_array();
    EXPECT_EQ(out.str(), "[\"文字列\",-7,true,false,null,0.50]");
}

// Boundary: when the requested precision does not fit the buffer, the value is
// written as null rather than truncated. Writing a truncated number would record
// a value different from the actual one.
TEST(JsonWriterTest, writes_null_when_precision_does_not_fit) {
    std::ostringstream out;
    JsonWriter writer(out, 0);
    writer.begin_array();
    writer.value_double(1.0, 200);
    writer.end_array();
    EXPECT_EQ(out.str(), "[null]");
}

}  // namespace
