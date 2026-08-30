// SPDX-License-Identifier: Apache-2.0
//
// Verifies the packed dictionary representation and the matching routine.
// These checks stay within what can be verified without OpenCV; agreement with
// OpenCV is covered by test/reference/test_dictionary_conformance.cpp.
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

// Nominal: the metadata of the bundled dictionary matches its nominal values.
TEST(DictionaryTest, builtin_table_metadata_matches_specification) {
    const aruco3cuda::DictionaryTable& table = mip_table();
    EXPECT_STREQ(table.name_, "DICT_ARUCO_MIP_36h12");
    EXPECT_EQ(table.marker_size_, 6);
    EXPECT_EQ(table.code_count_, 250);
    EXPECT_EQ(table.max_correction_bits_, 5);
    EXPECT_EQ(table.bit_count(), 36);
    ASSERT_NE(table.codes_, nullptr);
}

// Nominal: the registry count and index-based lookup agree.
TEST(DictionaryTest, registry_lists_all_builtin_tables) {
    ASSERT_GE(aruco3cuda::builtin_dictionary_count(), 1U);
    for (std::size_t i = 0; i < aruco3cuda::builtin_dictionary_count(); ++i) {
        ASSERT_NE(aruco3cuda::builtin_dictionary_at(i), nullptr);
    }
    EXPECT_EQ(aruco3cuda::builtin_dictionary_at(aruco3cuda::builtin_dictionary_count()), nullptr);
    EXPECT_EQ(aruco3cuda::find_builtin_dictionary("DICT_NOPE"), nullptr);
    EXPECT_EQ(aruco3cuda::find_builtin_dictionary(nullptr), nullptr);
}

// Nominal: pack and unpack round-trip.
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

// Nominal: four rotations return the code to its original value.
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

// Nominal: the four rotations stored in the table agree with repeated calls to
// rotate_marker_code. The CUDA side uses the pre-expanded table, so this agreement
// is the precondition for comparing rotations there.
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

// Nominal: an exact codeword matches at distance 0.
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

// Boundary: a code matches within the allowed error count and not beyond it.
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

    // With zero errors allowed, even a single flipped bit no longer matches.
    aruco3cuda::MarkerCode one_flip = original ^ 1U;
    aruco3cuda::DictionaryMatch strict_match;
    ASSERT_EQ(aruco3cuda::match_dictionary(table, one_flip, 0, &strict_match),
              aruco3cuda::Status::kOk);
    EXPECT_EQ(strict_match.id_, -1);
    EXPECT_EQ(strict_match.distance_, 1);
}

// Nominal: the minimum Hamming distance matches the nominal value.
// For DICT_ARUCO_MIP_36h12 the nominal value is 12.
TEST(DictionaryTest, minimum_hamming_distance_matches_nominal_value) {
    const aruco3cuda::DictionaryTable& table = mip_table();
    int distance = 0;
    ASSERT_EQ(aruco3cuda::minimum_hamming_distance(table, &distance), aruco3cuda::Status::kOk);
    EXPECT_EQ(distance, 12);
}

// Error case: invalid arguments are rejected.
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

    // A table without codes_ cannot be matched against.
    aruco3cuda::DictionaryTable empty;
    EXPECT_EQ(aruco3cuda::match_dictionary(empty, 0U, 3, &match),
              aruco3cuda::Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::minimum_hamming_distance(empty, &distance),
              aruco3cuda::Status::kInvalidArgument);
}

// Boundary: the valid range of marker_size is the same across all functions.
// Checking only pack_marker_code would hide a missing range check in the others.
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
    // Both ends of the valid range are accepted.
    for (const int size : {1, 7}) {
        EXPECT_EQ(aruco3cuda::pack_marker_code(bits, size, &code), aruco3cuda::Status::kOk)
                << "marker_size=" << size;
        EXPECT_EQ(aruco3cuda::unpack_marker_code(0U, size, bits), aruco3cuda::Status::kOk)
                << "marker_size=" << size;
        EXPECT_EQ(aruco3cuda::rotate_marker_code(0U, size, &code), aruco3cuda::Status::kOk)
                << "marker_size=" << size;
    }
}

// Boundary: match_dictionary rejects a table whose marker_size is out of range.
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
