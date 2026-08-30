// SPDX-License-Identifier: Apache-2.0
//
// Sample: render a marker from a built-in dictionary and write it as a PGM.
//
// Purpose:
//   Gives detect_image.cpp something to detect without requiring an external
//   asset, a camera, or an image library, and shows how the packed dictionary in
//   include/aruco3cuda/dictionary.hpp turns back into a bit pattern.
//
// Provenance:
//   The rendering rule is the one OpenCV 4.x uses in
//   cv::aruco::Dictionary::generateImageMarker (Apache-2.0): the border cells are
//   black, an inner bit of 1 is white and 0 is black, and the cell grid is scaled
//   up by nearest neighbour. Only the observable rule is reproduced; no OpenCV
//   code is copied, and nothing here derives from the GPLv3 ArUco distribution.
//   See docs/ip-and-licensing.md.
//
// Usage:
//   generate_marker --id 42 --size 200 --margin 40 --output marker.pgm
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "aruco3cuda/dictionary.hpp"
#include "aruco3cuda/status.hpp"

#include "pgm.hpp"

namespace {

using aruco3cuda::DictionaryTable;
using aruco3cuda_examples::GrayImage;
using aruco3cuda_examples::kMaxSidePx;

/// Value written for a white pixel. A marker is rendered as pure black and white,
/// because any intermediate level would be an artifact of this sample rather than
/// a property of the marker.
constexpr std::uint8_t kWhite = 255U;
constexpr std::uint8_t kBlack = 0U;

void print_usage(std::ostream& out) {
    out << "Usage: generate_marker --output <path> [option]...\n"
        << "\n"
        << "  --output <path>        Destination PGM file. Required\n"
        << "  --dictionary <name>    Dictionary name. Default DICT_ARUCO_MIP_36h12\n"
        << "  --id <n>               Marker id. Default 0\n"
        << "  --size <px>            Side of the marker itself, in pixels. Default 200\n"
        << "  --margin <px>          White quiet zone on each side, in pixels. Default 40\n"
        << "  --border-bits <n>      Width of the black border, in cells. Default 1\n"
        << "  --list-dictionaries    List the built-in dictionaries and exit\n"
        << "  --help                 Print this help and exit\n"
        << "\n"
        << "The quiet zone is not decoration. A marker whose black border touches\n"
        << "the image edge is rejected by min_distance_to_border_px_, so a margin of\n"
        << "0 produces an image that detect_image will find nothing in.\n";
}

/// Take the next argument. Returns false when it is missing.
bool take_value(int argc, char** argv, int* index, const char* option, std::string* out) {
    if (*index + 1 >= argc) {
        std::cerr << "missing argument: " << option << '\n';
        return false;
    }
    ++(*index);
    *out = argv[*index];
    return true;
}

bool parse_int(const std::string& text, int* out) {
    try {
        std::size_t consumed = 0;
        const int value = std::stoi(text, &consumed);
        if (consumed != text.size()) {
            return false;
        }
        *out = value;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

/// Confirm that a parsed integer lies within the given range.
///
/// argv is external input and is not trusted, so an out-of-range value never
/// reaches the rendering code. Validating on the CLI side lets the message name
/// the option that was wrong.
bool check_range(const std::string& option, int value, int minimum, int maximum) {
    if (value < minimum || value > maximum) {
        std::cerr << option << " must be between " << minimum << " and " << maximum
                  << " inclusive: " << value << '\n';
        return false;
    }
    return true;
}

void list_dictionaries(std::ostream& out) {
    for (std::size_t index = 0; index < aruco3cuda::builtin_dictionary_count(); ++index) {
        const DictionaryTable* table = aruco3cuda::builtin_dictionary_at(index);
        if (table == nullptr) {
            continue;
        }
        out << table->name_ << "  " << table->marker_size_ << "x" << table->marker_size_
            << " bits, " << table->code_count_ << " ids\n";
    }
}

/// Render one marker at the requested size, without the quiet zone.
///
/// The cell grid is (marker_size + 2 * border_bits) on a side. Each output pixel
/// takes the cell that covers it, which is nearest-neighbour scaling: cell
/// boundaries stay exactly on pixel boundaries only when side_px is a multiple of
/// the cell count, and otherwise some cells are one pixel wider than others. That
/// is the same behaviour as an INTER_NEAREST resize, and it is why a size that is
/// a multiple of the cell count gives the cleanest marker.
bool render_marker(const DictionaryTable& table, int id, int border_bits, int side_px,
                   GrayImage* out) {
    const int cells = table.marker_size_ + 2 * border_bits;
    if (side_px < cells) {
        std::cerr << "--size must be at least the cell count (" << cells << "): " << side_px
                  << '\n';
        return false;
    }

    // Rotation 0 is the codeword as it is drawn. The table stores all four
    // rotations, and index id * 4 + rotation selects one of them.
    const aruco3cuda::MarkerCode code = table.codes_[static_cast<std::size_t>(id) * 4U];
    std::vector<std::uint8_t> bits(static_cast<std::size_t>(table.bit_count()), 0U);
    if (aruco3cuda::unpack_marker_code(code, table.marker_size_, bits.data()) !=
        aruco3cuda::Status::kOk) {
        std::cerr << "failed to unpack the codeword: id=" << id << '\n';
        return false;
    }

    out->width_px_ = side_px;
    out->height_px_ = side_px;
    out->pixels_.assign(static_cast<std::size_t>(side_px) * static_cast<std::size_t>(side_px),
                        kBlack);
    for (int row_px = 0; row_px < side_px; ++row_px) {
        const int cell_row = row_px * cells / side_px;
        for (int col_px = 0; col_px < side_px; ++col_px) {
            const int cell_col = col_px * cells / side_px;
            const int inner_row = cell_row - border_bits;
            const int inner_col = cell_col - border_bits;
            std::uint8_t value = kBlack;
            if (inner_row >= 0 && inner_col >= 0 && inner_row < table.marker_size_ &&
                inner_col < table.marker_size_) {
                const std::size_t bit_index = static_cast<std::size_t>(inner_row) *
                                                      static_cast<std::size_t>(table.marker_size_) +
                                              static_cast<std::size_t>(inner_col);
                value = bits[bit_index] != 0U ? kWhite : kBlack;
            }
            out->pixels_[static_cast<std::size_t>(row_px) * static_cast<std::size_t>(side_px) +
                         static_cast<std::size_t>(col_px)] = value;
        }
    }
    return true;
}

/// Place the rendered marker at the centre of a white canvas.
void add_quiet_zone(const GrayImage& marker, int margin_px, GrayImage* out) {
    const int side_px = marker.width_px_ + 2 * margin_px;
    out->width_px_ = side_px;
    out->height_px_ = side_px;
    out->pixels_.assign(static_cast<std::size_t>(side_px) * static_cast<std::size_t>(side_px),
                        kWhite);
    for (int row_px = 0; row_px < marker.height_px_; ++row_px) {
        for (int col_px = 0; col_px < marker.width_px_; ++col_px) {
            const std::size_t source =
                    static_cast<std::size_t>(row_px) * static_cast<std::size_t>(marker.width_px_) +
                    static_cast<std::size_t>(col_px);
            const std::size_t destination = static_cast<std::size_t>(row_px + margin_px) *
                                                    static_cast<std::size_t>(side_px) +
                                            static_cast<std::size_t>(col_px + margin_px);
            out->pixels_[destination] = marker.pixels_[source];
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string dictionary_name = "DICT_ARUCO_MIP_36h12";
    std::string output_path;
    int id = 0;
    int side_px = 200;
    int margin_px = 40;
    int border_bits = 1;

    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        std::string value;
        if (option == "--help") {
            print_usage(std::cout);
            return EXIT_SUCCESS;
        }
        if (option == "--list-dictionaries") {
            list_dictionaries(std::cout);
            return EXIT_SUCCESS;
        }
        if (option == "--output") {
            if (!take_value(argc, argv, &i, "--output", &output_path)) {
                return EXIT_FAILURE;
            }
        } else if (option == "--dictionary") {
            if (!take_value(argc, argv, &i, "--dictionary", &dictionary_name)) {
                return EXIT_FAILURE;
            }
        } else if (option == "--id" || option == "--size" || option == "--margin" ||
                   option == "--border-bits") {
            if (!take_value(argc, argv, &i, option.c_str(), &value)) {
                return EXIT_FAILURE;
            }
            int parsed = 0;
            if (!parse_int(value, &parsed)) {
                std::cerr << option << " is not an integer: " << value << '\n';
                return EXIT_FAILURE;
            }
            if (option == "--id") {
                id = parsed;
            } else if (option == "--size") {
                side_px = parsed;
            } else if (option == "--margin") {
                margin_px = parsed;
            } else {
                border_bits = parsed;
            }
        } else {
            std::cerr << "unknown option: " << option << '\n';
            print_usage(std::cerr);
            return EXIT_FAILURE;
        }
    }

    if (output_path.empty()) {
        std::cerr << "--output was not specified\n";
        return EXIT_FAILURE;
    }
    if (!check_range("--size", side_px, 1, kMaxSidePx) ||
        !check_range("--margin", margin_px, 0, kMaxSidePx) ||
        !check_range("--border-bits", border_bits, 1, 16)) {
        return EXIT_FAILURE;
    }
    if (side_px + 2 * margin_px > kMaxSidePx) {
        std::cerr << "--size plus twice --margin exceeds " << kMaxSidePx << '\n';
        return EXIT_FAILURE;
    }

    const aruco3cuda::DictionaryTable* table =
            aruco3cuda::find_builtin_dictionary(dictionary_name.c_str());
    if (table == nullptr) {
        std::cerr << "unknown dictionary: " << dictionary_name << '\n';
        std::cerr << "the built-in dictionaries are:\n";
        list_dictionaries(std::cerr);
        return EXIT_FAILURE;
    }
    if (!check_range("--id", id, 0, table->code_count_ - 1)) {
        return EXIT_FAILURE;
    }

    GrayImage marker;
    if (!render_marker(*table, id, border_bits, side_px, &marker)) {
        return EXIT_FAILURE;
    }
    GrayImage canvas;
    add_quiet_zone(marker, margin_px, &canvas);

    std::string message;
    if (!aruco3cuda_examples::write_pgm(output_path, canvas, &message)) {
        std::cerr << message << '\n';
        return EXIT_FAILURE;
    }
    std::cout << "wrote " << output_path << ": " << canvas.width_px_ << "x" << canvas.height_px_
              << ", " << table->name_ << " id=" << id << ", marker " << side_px << " px, margin "
              << margin_px << " px\n";
    return EXIT_SUCCESS;
}
