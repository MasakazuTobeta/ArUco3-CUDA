// SPDX-License-Identifier: Apache-2.0
#include "aruco3cuda/util/json_writer.hpp"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <ostream>
#include <string>

namespace aruco3cuda::util {

std::string escape_json_string(const std::string& value) {
    std::string result;
    result.reserve(value.size() + 2);
    for (const char raw : value) {
        const auto c = static_cast<unsigned char>(raw);
        switch (c) {
            case '"':
                result += "\\\"";
                break;
            case '\\':
                result += "\\\\";
                break;
            case '\b':
                result += "\\b";
                break;
            case '\f':
                result += "\\f";
                break;
            case '\n':
                result += "\\n";
                break;
            case '\r':
                result += "\\r";
                break;
            case '\t':
                result += "\\t";
                break;
            default:
                if (c < 0x20U) {
                    char buffer[8];
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x", c);
                    result += buffer;
                } else {
                    // 0x20 以上はそのまま通す。UTF-8 の多 byte 文字も維持する。
                    result += raw;
                }
                break;
        }
    }
    return result;
}

JsonWriter::JsonWriter(std::ostream& out, int indent_width)
    : out_(out), indent_width_(indent_width), depth_(0), after_key_(false) {}

void JsonWriter::write_newline_indent() {
    if (this->indent_width_ <= 0) {
        return;
    }
    this->out_ << '\n';
    for (int i = 0; i < this->depth_ * this->indent_width_; ++i) {
        this->out_ << ' ';
    }
}

void JsonWriter::write_separator() {
    if (this->after_key_) {
        this->after_key_ = false;
        return;
    }
    if (!this->has_element_.empty()) {
        if (this->has_element_.back()) {
            this->out_ << ',';
        }
        this->has_element_.back() = true;
    }
    this->write_newline_indent();
}

void JsonWriter::begin_object() {
    this->write_separator();
    this->out_ << '{';
    ++this->depth_;
    this->has_element_.push_back(false);
}

void JsonWriter::end_object() {
    --this->depth_;
    const bool had_element = !this->has_element_.empty() && this->has_element_.back();
    if (!this->has_element_.empty()) {
        this->has_element_.pop_back();
    }
    if (had_element) {
        this->write_newline_indent();
    }
    this->out_ << '}';
}

void JsonWriter::begin_array() {
    this->write_separator();
    this->out_ << '[';
    ++this->depth_;
    this->has_element_.push_back(false);
}

void JsonWriter::end_array() {
    --this->depth_;
    const bool had_element = !this->has_element_.empty() && this->has_element_.back();
    if (!this->has_element_.empty()) {
        this->has_element_.pop_back();
    }
    if (had_element) {
        this->write_newline_indent();
    }
    this->out_ << ']';
}

void JsonWriter::key(const std::string& name) {
    this->write_separator();
    this->out_ << '"' << escape_json_string(name) << "\":";
    if (this->indent_width_ > 0) {
        this->out_ << ' ';
    }
    this->after_key_ = true;
}

void JsonWriter::value_string(const std::string& value) {
    this->write_separator();
    this->out_ << '"' << escape_json_string(value) << '"';
}

void JsonWriter::value_int(long long value) {
    this->write_separator();
    this->out_ << value;
}

void JsonWriter::value_bool(bool value) {
    this->write_separator();
    this->out_ << (value ? "true" : "false");
}

void JsonWriter::value_null() {
    this->write_separator();
    this->out_ << "null";
}

void JsonWriter::value_double(double value, int precision) {
    this->write_separator();
    if (!std::isfinite(value)) {
        // JSON は NaN と無限大を表現できない。値を捏造せず null とする。
        this->out_ << "null";
        return;
    }
    // 桁数が大きいと固定長 buffer へ収まらない。切り詰めた値を書くと
    // 実際の値と異なる数値を出力することになるため、収まらない場合は null とする。
    char buffer[64];
    const int written = std::snprintf(buffer, sizeof(buffer), "%.*f", precision, value);
    if (written < 0 || static_cast<std::size_t>(written) >= sizeof(buffer)) {
        this->out_ << "null";
        return;
    }
    this->out_ << buffer;
}

void JsonWriter::member_string(const std::string& name, const std::string& value) {
    this->key(name);
    this->value_string(value);
}

void JsonWriter::member_int(const std::string& name, long long value) {
    this->key(name);
    this->value_int(value);
}

void JsonWriter::member_bool(const std::string& name, bool value) {
    this->key(name);
    this->value_bool(value);
}

void JsonWriter::member_double(const std::string& name, double value, int precision) {
    this->key(name);
    this->value_double(value, precision);
}

}  // namespace aruco3cuda::util
