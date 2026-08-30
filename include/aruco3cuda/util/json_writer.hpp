// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_UTIL_JSON_WRITER_HPP
#define ARUCO3CUDA_UTIL_JSON_WRITER_HPP

#include <ostream>
#include <string>
#include <vector>

namespace aruco3cuda::util {

/// A minimal writer that emits deterministic JSON.
///
/// Purpose:
///   Stores evaluation results in a machine-readable form. The same input has to
///   produce byte-identical output, so the caller specifies the number of digits for
///   floating point values explicitly.
///
/// Ownership:
///   The destination ostream is held as a non-owning reference and is never freed. The
///   caller must keep the ostream alive longer than the writer. Strings taken as
///   arguments are copied into the output and not retained.
///   **This ownership applies to all public member functions below.**
///
/// Synchronization:
///   Host only, with no synchronization point. It carries formatting state internally,
///   so a single instance must not be used from several threads at once.
///   **This synchronization applies to all public member functions below.**
///
/// Limitations:
///   Non-finite values (NaN, infinity) cannot be represented in JSON and are emitted as
///   null.
///
/// Example input:
///   std::ostringstream out;
///   JsonWriter writer(out, 0);
///   writer.begin_object();
///   writer.member_int("a", 1);
///   writer.end_object();
/// Example output:
///   {"a":1}
class JsonWriter {
public:
    /// @param out The destination. It must stay valid for the lifetime of the writer.
    ///            Ownership does not transfer.
    /// @param indent_width Spaces per nesting level. 0 emits a single line with no
    ///                     newlines or indentation.
    ///
    /// Example input: the same call sequence run through JsonWriter(out, 0) and
    ///                JsonWriter(out, 2)
    /// Example output: the former gives {"a":1}, the latter several lines with newlines
    ///                 and two-space indentation
    explicit JsonWriter(std::ostream& out, int indent_width = 2);

    /// Begins an object. Unless a key was just written, the separator and the
    /// indentation are inserted automatically.
    ///
    /// @return Nothing. Only the state of the destination ostream changes.
    ///
    /// Example input: begin_object() followed by end_object() with indent_width = 0
    /// Example output: {}
    void begin_object();

    /// Ends an object. If at least one element was written, a newline precedes the
    /// closing brace.
    ///
    /// @return Nothing.
    ///
    /// Example input: begin_object(); member_int("a", 1); end_object();
    /// Example output: {"a":1}
    void end_object();

    /// Begins an array. Unless a key was just written, the separator and the
    /// indentation are inserted automatically.
    ///
    /// @return Nothing.
    ///
    /// Example input: begin_array(); value_int(1); value_int(2); end_array();
    /// Example output: [1,2]
    void begin_array();

    /// Ends an array.
    ///
    /// @return Nothing.
    ///
    /// Example input: begin_array(); end_array();
    /// Example output: []
    void end_array();

    /// Writes an object member name. Exactly one value must be written right after it.
    ///
    /// @param name The member name. It is escaped as a JSON string before being written.
    ///             The argument is copied and not retained by the writer.
    /// @return Nothing.
    ///
    /// Example input: key("a\"b"); value_int(1);
    /// Example output: "a\"b":1
    void key(const std::string& name);

    /// Writes a string as a value, escaped as a JSON string.
    ///
    /// @param value The string to write. It is copied and not retained by the writer.
    /// @return Nothing.
    ///
    /// Example input: value_string("a\"b")
    /// Example output: "a\"b"
    void value_string(const std::string& value);

    /// Writes an integer as a value.
    ///
    /// @param value The value to write.
    /// @return Nothing.
    ///
    /// Example input: value_int(-7)
    /// Example output: -7
    void value_int(long long value);

    /// Writes a boolean.
    ///
    /// @param value The value to write.
    /// @return Nothing.
    ///
    /// Example input: value_bool(true)
    /// Example output: true
    void value_bool(bool value);

    /// Writes null. Used to distinguish a value that was not obtained from a value of 0.
    ///
    /// @return Nothing.
    ///
    /// Example input: value_null()
    /// Example output: null
    void value_null();

    /// Writes a floating point number in fixed-point notation.
    ///
    /// The caller specifies the number of digits so the output stays stable. Non-finite
    /// values (NaN, infinity) cannot be represented in JSON, so null is written rather
    /// than a fabricated value.
    ///
    /// @param value The value to write.
    /// @param precision The number of digits after the decimal point.
    /// @return Nothing.
    ///
    /// Example input: value_double(1.0 / 3.0, 4)
    /// Example output: 0.3333
    void value_double(double value, int precision);

    /// Performs key() and value_string() together.
    ///
    /// @param name The member name. It is escaped.
    /// @param value The value. It is escaped. Both are copied and not retained by the
    ///              writer.
    /// @return Nothing.
    ///
    /// Example input: member_string("k", "v")
    /// Example output: "k":"v"
    void member_string(const std::string& name, const std::string& value);

    /// Performs key() and value_int() together.
    ///
    /// @param name The member name.
    /// @param value The value.
    /// @return Nothing.
    ///
    /// Example input: member_int("k", 3)
    /// Example output: "k":3
    void member_int(const std::string& name, long long value);

    /// Performs key() and value_bool() together.
    ///
    /// @param name The member name.
    /// @param value The value.
    /// @return Nothing.
    ///
    /// Example input: member_bool("k", false)
    /// Example output: "k":false
    void member_bool(const std::string& name, bool value);

    /// Performs key() and value_double() together.
    ///
    /// @param name The member name.
    /// @param value The value.
    /// @param precision The number of digits after the decimal point, with the same
    ///                  meaning as in value_double.
    /// @return Nothing.
    ///
    /// Example input: member_double("k", 0.5, 2)
    /// Example output: "k":0.50
    void member_double(const std::string& name, double value, int precision);

private:
    void write_separator();
    void write_newline_indent();

    std::ostream& out_;
    int indent_width_;
    int depth_;
    /// Whether an element has already been written at each nesting level. Held to decide
    /// where the separating , goes.
    std::vector<bool> has_element_;
    /// If a key was just written, no separator and no newline precede the value.
    bool after_key_;
};

/// Escapes a string into a form that is safe as JSON.
///
/// Quotation marks and backslashes are escaped and control characters become \\u
/// sequences. Bytes of 0x20 and above pass through unchanged, so a multi-byte UTF-8
/// character keeps its original byte sequence.
///
/// @param value The string to convert. It is copied, and the original is unchanged after
///              the call.
/// @return The escaped string, without the surrounding quotation marks.
///
/// The example below deliberately keeps a non-ASCII string: it is what documents the
/// UTF-8 pass-through above, and it mirrors the case pinned in
/// test/unit/test_json_writer.cpp. An ASCII example would no longer show that behavior.
///
/// Example input: "四隅\n"
/// Example output: 四隅\\n
std::string escape_json_string(const std::string& value);

}  // namespace aruco3cuda::util

#endif  // ARUCO3CUDA_UTIL_JSON_WRITER_HPP
