// SPDX-License-Identifier: Apache-2.0
//
// 目的:
//   OpenCV 4.x の定義済み Dictionary から、CUDA 側で使用する packed codeword を
//   生成し、C++ source として出力する。
//
// 方針:
//   core は OpenCV へ依存しない。したがって生成物を repository へ commit し、
//   OpenCV との一致は test で継続的に検証する。build 時に OpenCV から生成すると
//   core の build に OpenCV が必要になり、architecture の方針と矛盾する。
//
// 由来:
//   OpenCV は Apache-2.0 であり、取得元 version と commit を生成物へ記録する。
//   公式 ArUco の GPLv3 配布物からは抽出しない。
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

/// OpenCV の bytesList から、指定 ID と回転の bit 配列を取り出す。
///
/// bytesList の memory 配置は `bytesList.ptr(i)[k*nbytes + j]` であり、
/// channel の interleave ではなく回転ごとの連続 block である。誤った読み方を
/// 避けるため、byte 列を自前で展開せず OpenCV の accessor を使用する。
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

/// 自前の回転規則が OpenCV の rotation 定義と一致することを確認する。
///
/// CUDA 側は rotation 0 の codeword だけを持って回転を導出するのではなく、
/// 4 回転を事前展開した table を使う。その table の回転順序が OpenCV と
/// 同じであることは、照合結果の rotation を比較する前提になる。
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
    // 4 回まわすと元へ戻る。
    aruco3cuda::MarkerCode full_turn = rotated;
    if (aruco3cuda::rotate_marker_code(full_turn, dictionary.markerSize, &full_turn) !=
        aruco3cuda::Status::kOk) {
        return false;
    }
    return full_turn == codes_for_id[0];
}

/// Dictionary 名を CONTRIBUTING.md の定数命名規則へ変換する。
///
/// 定数は kPascalCase とするため、`_` 区切りの各語を先頭だけ大文字にして連結する。
///
/// 入力例: "DICT_ARUCO_MIP_36h12"
/// 出力例: "DictArucoMip36h12"
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
    out << "使用方法: aruco3cuda_dictgen --dictionary <name> --output <path.cpp>\n"
        << "\n"
        << "  --dictionary <name>  生成対象。既定 DICT_ARUCO_MIP_36h12\n"
        << "  --output <path>      出力する C++ source\n"
        << "  --check <path>       生成せず、既存 file と一致するかだけ確認する\n"
        << "  --list               対応する Dictionary 名を表示して終了\n"
        << "  --help               この説明を表示して終了\n";
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
            std::cerr << "引数が不足している: " << option << '\n';
            return EXIT_FAILURE;
        }
        if (option == "--dictionary") {
            dictionary_name = argv[++i];
        } else if (option == "--output") {
            output_path = argv[++i];
        } else if (option == "--check") {
            check_path = argv[++i];
        } else {
            std::cerr << "未知の option: " << option << '\n';
            print_usage(std::cerr);
            return EXIT_FAILURE;
        }
    }

    const auto entry = dictionary_table().find(dictionary_name);
    if (entry == dictionary_table().end()) {
        std::cerr << "未対応の Dictionary: " << dictionary_name << '\n';
        return EXIT_FAILURE;
    }
    if (output_path.empty() && check_path.empty()) {
        std::cerr << "--output または --check を指定すること\n";
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
                std::cerr << "bit 数が想定と異なる: id=" << id << " rotation=" << rotation << '\n';
                return EXIT_FAILURE;
            }
            aruco3cuda::MarkerCode code = 0U;
            if (aruco3cuda::pack_marker_code(bits.data(), marker_size, &code) !=
                aruco3cuda::Status::kOk) {
                std::cerr << "packing に失敗した: id=" << id << '\n';
                return EXIT_FAILURE;
            }
            codes_for_id.push_back(code);
        }
        if (!verify_rotation_convention(dictionary, codes_for_id)) {
            std::cerr << "回転規則が OpenCV と一致しない: id=" << id << '\n';
            return EXIT_FAILURE;
        }
        codes.insert(codes.end(), codes_for_id.begin(), codes_for_id.end());
    }

    const std::string identifier = identifier_from_name(dictionary_name);
    std::string generated;
    generated += "// SPDX-License-Identifier: Apache-2.0\n";
    generated += "//\n";
    generated += "// このファイルは tools/dictgen が生成した。手で編集しない。\n";
    generated += "//\n";
    generated += "// 取得元: OpenCV " + std::string(CV_VERSION) +
                 " の cv::aruco::getPredefinedDictionary()\n";
    generated += "// License: Apache-2.0\n";
    generated += "// Dictionary: " + dictionary_name + "\n";
    generated += "// markerSize: " + std::to_string(marker_size) + "\n";
    generated += "// codes: " + std::to_string(code_count) + "\n";
    generated += "// maxCorrectionBits: " + std::to_string(dictionary.maxCorrectionBits) + "\n";
    generated += "//\n";
    generated +=
            "// 再生成: aruco3cuda_dictgen --dictionary " + dictionary_name + " --output <path>\n";
    generated += "#include \"aruco3cuda/dictionary.hpp\"\n\n";
    generated += "namespace aruco3cuda::generated {\n\n";
    generated += "// index = id * 4 + rotation。rotation は反時計回りに 90 度ずつ進む。\n";
    generated += "//\n";
    generated += "// namespace scope の const は既定で internal linkage になるため、\n";
    generated += "// registry から参照できるよう extern を明示する。\n";
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
            std::cerr << "確認対象の file を開けない: " << check_path << '\n';
            return EXIT_FAILURE;
        }
        // 比較対象は数十 KB と既知である。任意 path を無制限に読み込まないため
        // 上限を設ける。上限超過は比較の前に失敗として扱う。
        constexpr std::streamsize kMaxCheckBytes = 16 << 20;  // 16 MiB
        existing.seekg(0, std::ios::end);
        const std::streamsize existing_size = existing.tellg();
        if (existing_size < 0 || existing_size > kMaxCheckBytes) {
            std::cerr << "確認対象の file が大きすぎるか読み取れない: " << check_path << '\n';
            return EXIT_FAILURE;
        }
        existing.seekg(0, std::ios::beg);
        // istreambuf_iterator は libstdc++ の内部で -Wnull-dereference の
        // 誤検知を招くため、rdbuf 経由で読み込む。
        std::ostringstream buffer;
        buffer << existing.rdbuf();
        const std::string current = buffer.str();
        if (current == generated) {
            std::cout << "一致: " << check_path << '\n';
            return EXIT_SUCCESS;
        }
        std::cerr << "生成結果が " << check_path << " と一致しない。\n"
                  << "OpenCV の version が変わった可能性がある。再生成して差分を確認すること。\n";
        return EXIT_FAILURE;
    }

    std::ofstream output(output_path, std::ios::binary);
    if (!output) {
        std::cerr << "出力 file を開けない: " << output_path << '\n';
        return EXIT_FAILURE;
    }
    output << generated;
    // 書き込み失敗を成功として報告しない。途中で失敗した生成物を
    // 正しいものとして commit してしまうことを防ぐ。
    output.close();
    if (!output) {
        std::cerr << "出力 file への書き込みに失敗した: " << output_path << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "生成: " << output_path << '\n'
              << "  dictionary=" << dictionary_name << " codes=" << code_count
              << " markerSize=" << marker_size
              << " maxCorrectionBits=" << dictionary.maxCorrectionBits << '\n'
              << "  sha256=" << aruco3cuda::util::sha256_bytes(generated.data(), generated.size())
              << '\n';
    return EXIT_SUCCESS;
}
