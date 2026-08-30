// SPDX-License-Identifier: Apache-2.0
//
// Implementation of the PGM helpers used by the samples.
//
// The parser is written out by hand rather than leaning on operator>> so that
// every rejection carries a reason. A sample whose only failure mode is "could
// not read the image" teaches nothing about what a valid input looks like.
#include "pgm.hpp"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <istream>
#include <limits>
#include <string>

namespace aruco3cuda_examples {
namespace {

/// Reports the reason through out_message when it is not nullptr.
///
/// @param out_message Destination. Nothing happens when it is nullptr.
/// @param text The reason.
/// @return Always false, so a caller can write `return fail(...)`.
bool fail(std::string* out_message, const std::string& text) {
    if (out_message != nullptr) {
        *out_message = text;
    }
    return false;
}

/// Whether the character is one of the whitespace characters PGM allows.
///
/// std::isspace is not used because it depends on the locale and has undefined
/// behavior for a negative argument, both of which are hazards when the input is
/// arbitrary bytes.
///
/// @param character The character to test, as read from an istream.
/// @return true if it is a space, tab, carriage return, line feed, vertical tab,
///         or form feed.
bool is_pgm_space(int character) {
    return character == ' ' || character == '\t' || character == '\r' || character == '\n' ||
           character == '\v' || character == '\f';
}

/// Reads one whitespace-separated token from a PGM header.
///
/// Comments run from '#' to the end of the line and may appear between any two
/// tokens. The whitespace character that terminates the token is consumed, which
/// is what the format requires: exactly one whitespace character separates the
/// maximum value from the start of the binary payload.
///
/// @param in The stream to read from.
/// @param out_token Receives the token. Cleared first.
/// @return true if a token was read, false at end of input.
bool next_header_token(std::istream& in, std::string* out_token) {
    out_token->clear();
    int character = in.get();
    for (;;) {
        if (character == std::istream::traits_type::eof()) {
            return false;
        }
        if (character == '#') {
            while (character != '\n' && character != std::istream::traits_type::eof()) {
                character = in.get();
            }
            continue;
        }
        if (!is_pgm_space(character)) {
            break;
        }
        character = in.get();
    }
    while (character != std::istream::traits_type::eof() && !is_pgm_space(character)) {
        out_token->push_back(static_cast<char>(character));
        character = in.get();
    }
    return !out_token->empty();
}

/// Parses a token as a non-negative decimal integer.
///
/// std::stoi is avoided because it throws and because it accepts a sign and
/// trailing characters. A header field is untrusted input, so anything that is not
/// a plain run of digits is a rejection rather than something to interpret.
///
/// @param token The token to parse.
/// @param out On success, receives the value.
/// @return true if the token is one or more digits and the value fits in int.
bool parse_non_negative_int(const std::string& token, int* out) {
    if (token.empty() || token.size() > 10U) {
        return false;
    }
    long long value = 0;
    for (const char digit : token) {
        if (digit < '0' || digit > '9') {
            return false;
        }
        value = value * 10 + (digit - '0');
        if (value > std::numeric_limits<int>::max()) {
            return false;
        }
    }
    *out = static_cast<int>(value);
    return true;
}

}  // namespace

bool read_pgm(const std::string& path, GrayImage* out, std::string* out_message) {
    if (out == nullptr) {
        return fail(out_message, "out is nullptr");
    }
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        return fail(out_message, "cannot open the file: " + path);
    }

    std::string token;
    if (!next_header_token(in, &token)) {
        return fail(out_message, "the file is empty: " + path);
    }
    if (token != "P5") {
        if (token == "P2") {
            return fail(out_message,
                        "ASCII PGM (P2) is not supported. Save it as binary PGM (P5): " + path);
        }
        return fail(out_message, "not a PGM: the magic number is \"" + token + "\"");
    }

    int width_px = 0;
    int height_px = 0;
    int max_value = 0;
    if (!next_header_token(in, &token) || !parse_non_negative_int(token, &width_px) ||
        !next_header_token(in, &token) || !parse_non_negative_int(token, &height_px) ||
        !next_header_token(in, &token) || !parse_non_negative_int(token, &max_value)) {
        return fail(out_message, "the header is malformed: " + path);
    }
    if (width_px < 1 || height_px < 1 || width_px > kMaxSidePx || height_px > kMaxSidePx) {
        return fail(out_message,
                    "the dimensions are out of range: width=" + std::to_string(width_px) +
                            " height=" + std::to_string(height_px));
    }
    if (max_value != 255) {
        return fail(out_message, "only a maximum value of 255 is supported: max_value=" +
                                         std::to_string(max_value));
    }

    // The dimensions have been bounded, so the product cannot overflow here.
    const std::size_t pixel_count =
            static_cast<std::size_t>(width_px) * static_cast<std::size_t>(height_px);
    out->pixels_.assign(pixel_count, 0U);
    in.read(reinterpret_cast<char*>(out->pixels_.data()),
            static_cast<std::streamsize>(pixel_count));
    if (static_cast<std::size_t>(in.gcount()) != pixel_count) {
        out->pixels_.clear();
        return fail(out_message, "the payload is short: expected " + std::to_string(pixel_count) +
                                         " bytes but read " + std::to_string(in.gcount()));
    }
    out->width_px_ = width_px;
    out->height_px_ = height_px;
    return true;
}

bool write_pgm(const std::string& path, const GrayImage& image, std::string* out_message) {
    if (image.width_px_ < 1 || image.height_px_ < 1 || image.width_px_ > kMaxSidePx ||
        image.height_px_ > kMaxSidePx) {
        return fail(out_message,
                    "the dimensions are out of range: width=" + std::to_string(image.width_px_) +
                            " height=" + std::to_string(image.height_px_));
    }
    const std::size_t pixel_count =
            static_cast<std::size_t>(image.width_px_) * static_cast<std::size_t>(image.height_px_);
    if (image.pixels_.size() != pixel_count) {
        return fail(out_message, "the pixel count does not match the dimensions: expected " +
                                         std::to_string(pixel_count) + " but held " +
                                         std::to_string(image.pixels_.size()));
    }

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        return fail(out_message, "cannot open the file for writing: " + path);
    }
    out << "P5\n" << image.width_px_ << " " << image.height_px_ << "\n255\n";
    out.write(reinterpret_cast<const char*>(image.pixels_.data()),
              static_cast<std::streamsize>(pixel_count));
    out.flush();
    if (!out.good()) {
        return fail(out_message, "writing failed: " + path);
    }
    return true;
}

}  // namespace aruco3cuda_examples
