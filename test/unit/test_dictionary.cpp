// SPDX-License-Identifier: Apache-2.0
//
// Dictionary の packed 表現と照合処理を検証する。
// OpenCV へ依存しない範囲の検証であり、OpenCV との一致は
// test/reference/test_dictionary_conformance.cpp が担当する。
#include "aruco3cuda/dictionary.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include "aruco3cuda/status.hpp"

namespace {

constexpr int kMaxBits = 49;

const aruco3cuda::DictionaryTable& mip_table() {
    const aruco3cuda::DictionaryTable* table =
            aruco3cuda::find_builtin_dictionary("DICT_ARUCO_MIP_36h12");
    EXPECT_NE(table, nullptr);
    return *table;
}

// 正常系: 収録 Dictionary の metadata が公称値と一致する。
TEST(DictionaryTest, builtin_table_metadata_matches_specification) {
    const aruco3cuda::DictionaryTable& table = mip_table();
    EXPECT_STREQ(table.name_, "DICT_ARUCO_MIP_36h12");
    EXPECT_EQ(table.marker_size_, 6);
    EXPECT_EQ(table.code_count_, 250);
    EXPECT_EQ(table.max_correction_bits_, 5);
    EXPECT_EQ(table.bit_count(), 36);
    ASSERT_NE(table.codes_, nullptr);
}

// 正常系: 収録数と index 参照が一致する。
TEST(DictionaryTest, registry_lists_all_builtin_tables) {
    ASSERT_GE(aruco3cuda::builtin_dictionary_count(), 1U);
    for (std::size_t i = 0; i < aruco3cuda::builtin_dictionary_count(); ++i) {
        ASSERT_NE(aruco3cuda::builtin_dictionary_at(i), nullptr);
    }
    EXPECT_EQ(aruco3cuda::builtin_dictionary_at(aruco3cuda::builtin_dictionary_count()), nullptr);
    EXPECT_EQ(aruco3cuda::find_builtin_dictionary("DICT_NOPE"), nullptr);
    EXPECT_EQ(aruco3cuda::find_builtin_dictionary(nullptr), nullptr);
}

// 正常系: pack と unpack が往復する。
TEST(DictionaryTest, pack_and_unpack_roundtrip) {
    const aruco3cuda::DictionaryTable& table = mip_table();
    std::uint8_t bits[kMaxBits] = {};
    for (int id = 0; id < table.code_count_; ++id) {
        const aruco3cuda::MarkerCode original = table.codes_[static_cast<std::size_t>(id) * 4U];
        ASSERT_EQ(aruco3cuda::unpack_marker_code(original, table.marker_size_, bits),
                  aruco3cuda::Status::kOk);
        aruco3cuda::MarkerCode repacked = 0U;
        ASSERT_EQ(aruco3cuda::pack_marker_code(bits, table.marker_size_, &repacked),
                  aruco3cuda::Status::kOk);
        ASSERT_EQ(repacked, original) << "id=" << id;
    }
}

// 正常系: 4 回転すると元へ戻る。
TEST(DictionaryTest, four_rotations_return_to_original) {
    const aruco3cuda::DictionaryTable& table = mip_table();
    for (int id = 0; id < table.code_count_; ++id) {
        aruco3cuda::MarkerCode code = table.codes_[static_cast<std::size_t>(id) * 4U];
        const aruco3cuda::MarkerCode original = code;
        for (int i = 0; i < 4; ++i) {
            ASSERT_EQ(aruco3cuda::rotate_marker_code(code, table.marker_size_, &code),
                      aruco3cuda::Status::kOk);
        }
        ASSERT_EQ(code, original) << "id=" << id;
    }
}

// 正常系: table の 4 回転が rotate_marker_code の連鎖と一致する。
// CUDA 側は事前展開した table を使うため、この一致が rotation 比較の前提になる。
TEST(DictionaryTest, table_rotations_match_rotate_function) {
    const aruco3cuda::DictionaryTable& table = mip_table();
    for (int id = 0; id < table.code_count_; ++id) {
        aruco3cuda::MarkerCode code = table.codes_[static_cast<std::size_t>(id) * 4U];
        for (int rotation = 1; rotation < 4; ++rotation) {
            ASSERT_EQ(aruco3cuda::rotate_marker_code(code, table.marker_size_, &code),
                      aruco3cuda::Status::kOk);
            ASSERT_EQ(code, table.codes_[static_cast<std::size_t>(id) * 4U +
                                         static_cast<std::size_t>(rotation)])
                    << "id=" << id << " rotation=" << rotation;
        }
    }
}

// 正常系: 正しい codeword は距離 0 で一致する。
TEST(DictionaryTest, exact_codeword_matches_with_zero_distance) {
    const aruco3cuda::DictionaryTable& table = mip_table();
    for (int id = 0; id < table.code_count_; ++id) {
        for (int rotation = 0; rotation < 4; ++rotation) {
            aruco3cuda::DictionaryMatch match;
            ASSERT_EQ(aruco3cuda::match_dictionary(table,
                                                   table.codes_[static_cast<std::size_t>(id) * 4U +
                                                                static_cast<std::size_t>(rotation)],
                                                   3, &match),
                      aruco3cuda::Status::kOk);
            EXPECT_EQ(match.id_, id) << "id=" << id << " rotation=" << rotation;
            EXPECT_EQ(match.rotation_, rotation);
            EXPECT_EQ(match.distance_, 0);
        }
    }
}

// 境界値: 許容誤り数の内側では一致し、外側では一致しない。
TEST(DictionaryTest, matches_within_correction_limit_only) {
    const aruco3cuda::DictionaryTable& table = mip_table();
    constexpr int kId = 42;
    constexpr int kAllowedErrors = 3;
    const aruco3cuda::MarkerCode original = table.codes_[static_cast<std::size_t>(kId) * 4U];

    for (int flips = 0; flips <= kAllowedErrors; ++flips) {
        aruco3cuda::MarkerCode corrupted = original;
        for (int bit = 0; bit < flips; ++bit) {
            corrupted ^= (static_cast<aruco3cuda::MarkerCode>(1) << bit);
        }
        aruco3cuda::DictionaryMatch match;
        ASSERT_EQ(aruco3cuda::match_dictionary(table, corrupted, kAllowedErrors, &match),
                  aruco3cuda::Status::kOk);
        EXPECT_EQ(match.id_, kId) << "flips=" << flips;
        EXPECT_EQ(match.distance_, flips);
    }

    // 許容数を 0 にすると 1 bit 反転でも一致しない。
    aruco3cuda::MarkerCode one_flip = original ^ 1U;
    aruco3cuda::DictionaryMatch strict_match;
    ASSERT_EQ(aruco3cuda::match_dictionary(table, one_flip, 0, &strict_match),
              aruco3cuda::Status::kOk);
    EXPECT_EQ(strict_match.id_, -1);
    EXPECT_EQ(strict_match.distance_, 1);
}

// 正常系: 最小 Hamming 距離が公称値と一致する。
// DICT_ARUCO_MIP_36h12 の公称値は 12。
TEST(DictionaryTest, minimum_hamming_distance_matches_nominal_value) {
    const aruco3cuda::DictionaryTable& table = mip_table();
    int distance = 0;
    ASSERT_EQ(aruco3cuda::minimum_hamming_distance(table, &distance), aruco3cuda::Status::kOk);
    EXPECT_EQ(distance, 12);
}

// 異常系: 不正な引数を拒否する。
TEST(DictionaryTest, rejects_invalid_arguments) {
    const aruco3cuda::DictionaryTable& table = mip_table();
    std::uint8_t bits[kMaxBits] = {};
    aruco3cuda::MarkerCode code = 0U;
    aruco3cuda::DictionaryMatch match;
    int distance = 0;

    EXPECT_EQ(aruco3cuda::pack_marker_code(nullptr, 6, &code),
              aruco3cuda::Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::pack_marker_code(bits, 6, nullptr), aruco3cuda::Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::pack_marker_code(bits, 0, &code), aruco3cuda::Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::pack_marker_code(bits, 8, &code), aruco3cuda::Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::unpack_marker_code(0U, 6, nullptr), aruco3cuda::Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::rotate_marker_code(0U, 6, nullptr), aruco3cuda::Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::match_dictionary(table, 0U, 3, nullptr),
              aruco3cuda::Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::match_dictionary(table, 0U, -1, &match),
              aruco3cuda::Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::minimum_hamming_distance(table, nullptr),
              aruco3cuda::Status::kInvalidArgument);

    // codes_ が無い table は照合できない。
    aruco3cuda::DictionaryTable empty;
    EXPECT_EQ(aruco3cuda::match_dictionary(empty, 0U, 3, &match),
              aruco3cuda::Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::minimum_hamming_distance(empty, &distance),
              aruco3cuda::Status::kInvalidArgument);
}

// 境界値: marker_size の有効範囲は全関数で一致する。
// pack_marker_code だけを検査すると、他の関数の範囲外検査漏れに気付けない。
TEST(DictionaryTest, marker_size_bounds_are_consistent_across_functions) {
    std::uint8_t bits[kMaxBits] = {};
    aruco3cuda::MarkerCode code = 0U;
    const int invalid_sizes[] = {-1, 0, 8, 64};
    for (const int size : invalid_sizes) {
        EXPECT_EQ(aruco3cuda::pack_marker_code(bits, size, &code),
                  aruco3cuda::Status::kInvalidArgument)
                << "marker_size=" << size;
        EXPECT_EQ(aruco3cuda::unpack_marker_code(0U, size, bits),
                  aruco3cuda::Status::kInvalidArgument)
                << "marker_size=" << size;
        EXPECT_EQ(aruco3cuda::rotate_marker_code(0U, size, &code),
                  aruco3cuda::Status::kInvalidArgument)
                << "marker_size=" << size;
    }
    // 有効範囲の両端は受理される。
    for (const int size : {1, 7}) {
        EXPECT_EQ(aruco3cuda::pack_marker_code(bits, size, &code), aruco3cuda::Status::kOk)
                << "marker_size=" << size;
        EXPECT_EQ(aruco3cuda::unpack_marker_code(0U, size, bits), aruco3cuda::Status::kOk)
                << "marker_size=" << size;
        EXPECT_EQ(aruco3cuda::rotate_marker_code(0U, size, &code), aruco3cuda::Status::kOk)
                << "marker_size=" << size;
    }
}

// 境界値: match_dictionary は marker_size が範囲外の table を拒否する。
TEST(DictionaryTest, match_rejects_table_with_invalid_marker_size) {
    const aruco3cuda::MarkerCode codes[4] = {0U, 0U, 0U, 0U};
    aruco3cuda::DictionaryMatch match;
    for (const int size : {0, 8}) {
        aruco3cuda::DictionaryTable table;
        table.name_ = "invalid";
        table.marker_size_ = size;
        table.max_correction_bits_ = 1;
        table.code_count_ = 1;
        table.codes_ = codes;
        EXPECT_EQ(aruco3cuda::match_dictionary(table, 0U, 0, &match),
                  aruco3cuda::Status::kInvalidArgument)
                << "marker_size=" << size;
        int distance = 0;
        table.code_count_ = 2;
        EXPECT_EQ(aruco3cuda::minimum_hamming_distance(table, &distance),
                  aruco3cuda::Status::kInvalidArgument)
                << "marker_size=" << size;
    }
}

}  // namespace
