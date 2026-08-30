// SPDX-License-Identifier: Apache-2.0
//
// Purpose:
//   Generates the packed codewords used on the CUDA side from the predefined
//   Dictionaries of OpenCV 4.x and writes them out as C++ source.
//
// Policy:
//   core does not depend on OpenCV. The generated files are therefore committed to
//   the repository, and their agreement with OpenCV is verified continuously by
//   tests. Generating them from OpenCV at build time would make OpenCV a requirement
//   for building core, which contradicts the architecture policy.
//
// Provenance:
//   OpenCV is Apache-2.0, and the source version and commit are recorded in the
//   generated file. Nothing is extracted from the GPLv3 distribution of the official
//   ArUco.
#include <opencv2/core.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <ios>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "aruco3cuda/dictionary.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/util/sha256.hpp"

namespace {

const std::map<std::string, int>& dictionary_table() {
    static const std::map<std::string, int> kTable = {
            {"DICT_4X4_50", cv::aruco::DICT_4X4_50},
            {"DICT_4X4_100", cv::aruco::DICT_4X4_100},
            {"DICT_4X4_250", cv::aruco::DICT_4X4_250},
            {"DICT_4X4_1000", cv::aruco::DICT_4X4_1000},
            {"DICT_5X5_50", cv::aruco::DICT_5X5_50},
            {"DICT_5X5_100", cv::aruco::DICT_5X5_100},
            {"DICT_5X5_250", cv::aruco::DICT_5X5_250},
            {"DICT_5X5_1000", cv::aruco::DICT_5X5_1000},
            {"DICT_6X6_50", cv::aruco::DICT_6X6_50},
            {"DICT_6X6_100", cv::aruco::DICT_6X6_100},
            {"DICT_6X6_250", cv::aruco::DICT_6X6_250},
            {"DICT_6X6_1000", cv::aruco::DICT_6X6_1000},
            {"DICT_7X7_50", cv::aruco::DICT_7X7_50},
            {"DICT_7X7_100", cv::aruco::DICT_7X7_100},
            {"DICT_7X7_250", cv::aruco::DICT_7X7_250},
            {"DICT_7X7_1000", cv::aruco::DICT_7X7_1000},
            {"DICT_ARUCO_ORIGINAL", cv::aruco::DICT_ARUCO_ORIGINAL},
            {"DICT_ARUCO_MIP_36h12", cv::aruco::DICT_ARUCO_MIP_36h12},
    };
    return kTable;
}

/// Extracts the bit array for the given ID and rotation from the OpenCV bytesList.
///
/// The memory layout of bytesList is `bytesList.ptr(i)[k*nbytes + j]`: a contiguous
/// block per rotation, not channel interleaving. To avoid reading it incorrectly,
/// the byte sequence is not unpacked by hand; the OpenCV accessor is used instead.
std::vector<std::uint8_t> bits_from_bytes_list(const cv::aruco::Dictionary& dictionary, int id,
                                               int rotation) {
    const cv::Mat bits = cv::aruco::Dictionary::getBitsFromByteList(
            dictionary.bytesList.rowRange(id, id + 1), dictionary.markerSize, rotation);
    std::vector<std::uint8_t> result;
    result.reserve(static_cast<std::size_t>(bits.total()));
    for (int r = 0; r < bits.rows; ++r) {
        for (int c = 0; c < bits.cols; ++c) {
            result.push_back(static_cast<std::uint8_t>(bits.at<std::uint8_t>(r, c) != 0 ? 1 : 0));
        }
    }
    return result;
}

/// Confirms that our own rotation rule agrees with the OpenCV rotation definition.
///
/// Rather than deriving the rotations from the rotation-0 codeword alone, the CUDA
/// side uses a table with all 4 rotations expanded in advance. Comparing the rotation
/// of a match against OpenCV presupposes that the rotation order of that table is the
/// same as OpenCV's.
bool verify_rotation_convention(const cv::aruco::Dictionary& dictionary,
                                const std::vector<aruco3cuda::MarkerCode>& codes_for_id) {
    aruco3cuda::MarkerCode rotated = codes_for_id[0];
    for (int rotation = 1; rotation < 4; ++rotation) {
        if (aruco3cuda::rotate_marker_code(rotated, dictionary.markerSize, &rotated) !=
            aruco3cuda::Status::kOk) {
            return false;
        }
        if (rotated != codes_for_id[static_cast<std::size_t>(rotation)]) {
            return false;
        }
    }
    // Rotating four times returns to the original.
    aruco3cuda::MarkerCode full_turn = rotated;
    if (aruco3cuda::rotate_marker_code(full_turn, dictionary.markerSize, &full_turn) !=
        aruco3cuda::Status::kOk) {
        return false;
    }
    return full_turn == codes_for_id[0];
}

/// Converts a Dictionary name to the constant naming convention of CONTRIBUTING.md.
///
/// Constants are kPascalCase, so each `_`-separated word is capitalized on its first
/// letter and the words are joined.
///
/// Example input: "DICT_ARUCO_MIP_36h12"
/// Example output: "DictArucoMip36h12"
std::string identifier_from_name(const std::string& name) {
    std::string result;
    result.reserve(name.size());
    bool at_word_start = true;
    for (const char c : name) {
        if (c == '_') {
            at_word_start = true;
            continue;
        }
        const char lowered = static_cast<char>((c >= 'A' && c <= 'Z') ? (c - 'A' + 'a') : c);
        if (at_word_start && lowered >= 'a' && lowered <= 'z') {
            result += static_cast<char>(lowered - 'a' + 'A');
        } else {
            result += lowered;
        }
        at_word_start = false;
    }
    return result;
}

void print_usage(std::ostream& out) {
    out << "usage: aruco3cuda_dictgen --dictionary <name> --output <path.cpp>\n"
        << "\n"
        << "  --dictionary <name>  Dictionary to generate; default DICT_ARUCO_MIP_36h12\n"
        << "  --output <path>      C++ source to write\n"
        << "  --check <path>       generate nothing; only check that an existing file "
           "matches\n"
        << "  --list               print the supported Dictionary names and exit\n"
        << "  --help               print this help and exit\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string dictionary_name = "DICT_ARUCO_MIP_36h12";
    std::string output_path;
    std::string check_path;

    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--help") {
            print_usage(std::cout);
            return EXIT_SUCCESS;
        }
        if (option == "--list") {
            for (const auto& entry : dictionary_table()) {
                std::cout << entry.first << '\n';
            }
            return EXIT_SUCCESS;
        }
        if (i + 1 >= argc) {
            std::cerr << "missing argument for: " << option << '\n';
            return EXIT_FAILURE;
        }
        if (option == "--dictionary") {
            dictionary_name = argv[++i];
        } else if (option == "--output") {
            output_path = argv[++i];
        } else if (option == "--check") {
            check_path = argv[++i];
        } else {
            std::cerr << "unknown option: " << option << '\n';
            print_usage(std::cerr);
            return EXIT_FAILURE;
        }
    }

    const auto entry = dictionary_table().find(dictionary_name);
    if (entry == dictionary_table().end()) {
        std::cerr << "unsupported Dictionary: " << dictionary_name << '\n';
        return EXIT_FAILURE;
    }
    if (output_path.empty() && check_path.empty()) {
        std::cerr << "specify either --output or --check\n";
        return EXIT_FAILURE;
    }

    const cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(entry->second);
    const int code_count = dictionary.bytesList.rows;
    const int marker_size = dictionary.markerSize;

    std::vector<aruco3cuda::MarkerCode> codes;
    codes.reserve(static_cast<std::size_t>(code_count) * 4U);
    for (int id = 0; id < code_count; ++id) {
        std::vector<aruco3cuda::MarkerCode> codes_for_id;
        codes_for_id.reserve(4U);
        for (int rotation = 0; rotation < 4; ++rotation) {
            const std::vector<std::uint8_t> bits = bits_from_bytes_list(dictionary, id, rotation);
            if (static_cast<int>(bits.size()) != marker_size * marker_size) {
                std::cerr << "unexpected number of bits: id=" << id << " rotation=" << rotation
                          << '\n';
                return EXIT_FAILURE;
            }
            aruco3cuda::MarkerCode code = 0U;
            if (aruco3cuda::pack_marker_code(bits.data(), marker_size, &code) !=
                aruco3cuda::Status::kOk) {
                std::cerr << "packing failed: id=" << id << '\n';
                return EXIT_FAILURE;
            }
            codes_for_id.push_back(code);
        }
        if (!verify_rotation_convention(dictionary, codes_for_id)) {
            std::cerr << "the rotation rule does not agree with OpenCV: id=" << id << '\n';
            return EXIT_FAILURE;
        }
        codes.insert(codes.end(), codes_for_id.begin(), codes_for_id.end());
    }

    const std::string identifier = identifier_from_name(dictionary_name);
    std::string generated;
    generated += "// SPDX-License-Identifier: Apache-2.0\n";
    generated += "//\n";
    generated += "// This file was generated by tools/dictgen. Do not edit by hand.\n";
    generated += "//\n";
    generated += "// Source: OpenCV " + std::string(CV_VERSION) +
                 " cv::aruco::getPredefinedDictionary()\n";
    generated += "// License: Apache-2.0\n";
    generated += "// Dictionary: " + dictionary_name + "\n";
    generated += "// markerSize: " + std::to_string(marker_size) + "\n";
    generated += "// codes: " + std::to_string(code_count) + "\n";
    generated += "// maxCorrectionBits: " + std::to_string(dictionary.maxCorrectionBits) + "\n";
    generated += "//\n";
    generated += "// Regenerate: aruco3cuda_dictgen --dictionary " + dictionary_name +
                 " --output <path>\n";
    generated += "#include \"aruco3cuda/dictionary.hpp\"\n\n";
    generated += "namespace aruco3cuda::generated {\n\n";
    generated +=
            "// index = id * 4 + rotation. Rotation advances counterclockwise in 90-degree "
            "steps.\n";
    generated += "//\n";
    generated += "// A namespace-scope const has internal linkage by default, so extern is\n";
    generated += "// stated explicitly to let the registry refer to it.\n";
    generated += "extern const MarkerCode k" + identifier + "Codes[];\n";
    generated += "extern const MarkerCode k" + identifier + "Codes[] = {\n";
    for (std::size_t i = 0; i < codes.size(); ++i) {
        if (i % 4U == 0U) {
            generated += "    ";
        }
        char buffer[32];
        std::snprintf(buffer, sizeof(buffer), "0x%016llxULL",
                      static_cast<unsigned long long>(codes[i]));
        generated += buffer;
        generated += (i + 1U == codes.size()) ? "\n" : ",";
        if (i % 4U == 3U && i + 1U != codes.size()) {
            generated += "\n";
        } else if (i + 1U != codes.size()) {
            generated += " ";
        }
    }
    generated += "};\n\n";
    generated += "extern const DictionaryTable k" + identifier + "Table;\n";
    generated += "extern const DictionaryTable k" + identifier + "Table = {\n";
    generated += "    \"" + dictionary_name + "\",\n";
    generated += "    " + std::to_string(marker_size) + ",\n";
    generated += "    " + std::to_string(dictionary.maxCorrectionBits) + ",\n";
    generated += "    " + std::to_string(code_count) + ",\n";
    generated += "    k" + identifier + "Codes,\n";
    generated += "};\n\n";
    generated += "}  // namespace aruco3cuda::generated\n";

    if (!check_path.empty()) {
        std::ifstream existing(check_path, std::ios::binary);
        if (!existing) {
            std::cerr << "cannot open the file to check: " << check_path << '\n';
            return EXIT_FAILURE;
        }
        // The file being compared is known to be a few tens of KB. A cap keeps an
        // arbitrary path from being read without limit; exceeding it is treated as a
        // failure before the comparison.
        constexpr std::streamsize kMaxCheckBytes = 16 << 20;  // 16 MiB
        existing.seekg(0, std::ios::end);
        const std::streamsize existing_size = existing.tellg();
        if (existing_size < 0 || existing_size > kMaxCheckBytes) {
            std::cerr << "the file to check is too large or cannot be read: " << check_path << '\n';
            return EXIT_FAILURE;
        }
        existing.seekg(0, std::ios::beg);
        // istreambuf_iterator triggers a false -Wnull-dereference inside libstdc++, so
        // the file is read through rdbuf instead.
        std::ostringstream buffer;
        buffer << existing.rdbuf();
        const std::string current = buffer.str();
        if (current == generated) {
            std::cout << "matches: " << check_path << '\n';
            return EXIT_SUCCESS;
        }
        std::cerr << "the generated result does not match " << check_path << ".\n"
                  << "The OpenCV version may have changed. Regenerate and inspect the "
                     "difference.\n";
        return EXIT_FAILURE;
    }

    std::ofstream output(output_path, std::ios::binary);
    if (!output) {
        std::cerr << "cannot open the output file: " << output_path << '\n';
        return EXIT_FAILURE;
    }
    output << generated;
    // Never report a failed write as a success. This keeps a file whose generation
    // failed part way from being committed as if it were correct.
    output.close();
    if (!output) {
        std::cerr << "writing the output file failed: " << output_path << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "generated: " << output_path << '\n'
              << "  dictionary=" << dictionary_name << " codes=" << code_count
              << " markerSize=" << marker_size
              << " maxCorrectionBits=" << dictionary.maxCorrectionBits << '\n'
              << "  sha256=" << aruco3cuda::util::sha256_bytes(generated.data(), generated.size())
              << '\n';
    return EXIT_SUCCESS;
}
