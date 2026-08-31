// SPDX-License-Identifier: Apache-2.0
//
// Implements checks 1 through 5 that docs/dictionaries.md requires for every
// dictionary. The generated table is cross-checked against the OpenCV
// dictionary obtained on the spot.
#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "aruco3cuda/dictionary.hpp"
#include "aruco3cuda/status.hpp"

namespace {

/// One dictionary under test: our name and the OpenCV enumerator for it.
struct DictionaryCase {
    const char* name_;
    int opencv_id_;
    /// Minimum Hamming distance over every pair of ids and rotations. OpenCV
    /// does not publish it, so it is recomputed by check 5 and compared against
    /// the value recorded here, which was itself obtained by that computation.
    /// It is kept as a literal so that a change in the table shows up as a
    /// failure rather than as a silently different number.
    int minimum_distance_;
};

/// The dictionaries exercised here.
///
/// Not all eighteen. Every marker size is represented, plus the two 6x6
/// dictionaries that differ only in their codewords, and the largest of each
/// size. What this file checks beyond the tables themselves is the decode and
/// identify paths, and those depend on the marker size rather than on which
/// dictionary of that size is used. Table correctness for all eighteen is
/// covered mechanically by `dictgen --check` in test/cli, which compares every
/// committed table against OpenCV.
const DictionaryCase kCases[] = {
        {"DICT_4X4_50", cv::aruco::DICT_4X4_50, 4},
        {"DICT_5X5_1000", cv::aruco::DICT_5X5_1000, 5},
        {"DICT_6X6_250", cv::aruco::DICT_6X6_250, 11},
        {"DICT_7X7_1000", cv::aruco::DICT_7X7_1000, 14},
        {"DICT_ARUCO_MIP_36h12", cv::aruco::DICT_ARUCO_MIP_36h12, 12},
};

class DictionaryConformanceTest : public ::testing::TestWithParam<DictionaryCase> {
protected:
    const aruco3cuda::DictionaryTable& generated_table() const {
        const aruco3cuda::DictionaryTable* table =
                aruco3cuda::find_builtin_dictionary(GetParam().name_);
        EXPECT_NE(table, nullptr) << GetParam().name_;
        return *table;
    }

    cv::aruco::Dictionary opencv_dictionary() const {
        return cv::aruco::getPredefinedDictionary(GetParam().opencv_id_);
    }

    /// Ids to exercise where walking every id would be wasteful. Clamped and
    /// deduplicated, because the smallest dictionaries hold fewer ids than the
    /// fixed sample would name.
    std::vector<int> sample_ids() const {
        const int last = generated_table().code_count_ - 1;
        std::vector<int> ids;
        for (const int candidate : {0, 1, 42, 128, last}) {
            const int clamped = candidate > last ? last : candidate;
            if (std::find(ids.begin(), ids.end(), clamped) == ids.end()) {
                ids.push_back(clamped);
            }
        }
        return ids;
    }
};

/// Converts an OpenCV bits Mat into the packed representation.
aruco3cuda::MarkerCode pack_from_mat(const cv::Mat& bits, int marker_size) {
    std::vector<std::uint8_t> flat;
    flat.reserve(static_cast<std::size_t>(bits.total()));
    for (int r = 0; r < bits.rows; ++r) {
        for (int c = 0; c < bits.cols; ++c) {
            flat.push_back(static_cast<std::uint8_t>(bits.at<std::uint8_t>(r, c) != 0 ? 1 : 0));
        }
    }
    aruco3cuda::MarkerCode code = 0U;
    EXPECT_EQ(aruco3cuda::pack_marker_code(flat.data(), marker_size, &code),
              aruco3cuda::Status::kOk);
    return code;
}

// Check 1: the ID count, the marker size, and the maximum number of
// correctable bits match the OpenCV reference.
TEST_P(DictionaryConformanceTest, metadata_matches_opencv) {
    const cv::aruco::Dictionary dictionary = opencv_dictionary();
    const aruco3cuda::DictionaryTable& table = generated_table();
    EXPECT_EQ(table.code_count_, dictionary.bytesList.rows);
    EXPECT_EQ(table.marker_size_, dictionary.markerSize);
    EXPECT_EQ(table.max_correction_bits_, dictionary.maxCorrectionBits);
}

// Check 2: the packed codewords for every ID and all four rotations match
// OpenCV's bytesList.
TEST_P(DictionaryConformanceTest, all_codewords_match_opencv_bytes_list) {
    const cv::aruco::Dictionary dictionary = opencv_dictionary();
    const aruco3cuda::DictionaryTable& table = generated_table();
    ASSERT_EQ(table.code_count_, dictionary.bytesList.rows);

    for (int id = 0; id < table.code_count_; ++id) {
        for (int rotation = 0; rotation < 4; ++rotation) {
            const cv::Mat bits = cv::aruco::Dictionary::getBitsFromByteList(
                    dictionary.bytesList.rowRange(id, id + 1), dictionary.markerSize, rotation);
            const aruco3cuda::MarkerCode expected = pack_from_mat(bits, table.marker_size_);
            ASSERT_EQ(table.codes_[static_cast<std::size_t>(id) * 4U +
                                   static_cast<std::size_t>(rotation)],
                      expected)
                    << "id=" << id << " rotation=" << rotation;
        }
    }
}

// Check 3: decoding the marker image of every ID recovers the original ID and
// rotation.
TEST_P(DictionaryConformanceTest, decodes_generated_marker_images) {
    const cv::aruco::Dictionary dictionary = opencv_dictionary();
    const aruco3cuda::DictionaryTable& table = generated_table();

    for (int id = 0; id < table.code_count_; ++id) {
        // generateImageMarker requires borderBits > 0. Ask for the side length
        // at which one cell is one pixel including the border, then crop out
        // the inner bit grid.
        constexpr int kBorderBits = 1;
        cv::Mat image;
        dictionary.generateImageMarker(id, dictionary.markerSize + 2 * kBorderBits, image,
                                       kBorderBits);
        ASSERT_EQ(image.rows, dictionary.markerSize + 2 * kBorderBits);
        const cv::Mat payload = image(
                cv::Rect(kBorderBits, kBorderBits, dictionary.markerSize, dictionary.markerSize));

        std::vector<std::uint8_t> bits;
        bits.reserve(static_cast<std::size_t>(payload.total()));
        for (int r = 0; r < payload.rows; ++r) {
            for (int c = 0; c < payload.cols; ++c) {
                // White (255) is bit 1; black (0) is bit 0.
                bits.push_back(
                        static_cast<std::uint8_t>(payload.at<std::uint8_t>(r, c) > 127 ? 1 : 0));
            }
        }
        aruco3cuda::MarkerCode code = 0U;
        ASSERT_EQ(aruco3cuda::pack_marker_code(bits.data(), table.marker_size_, &code),
                  aruco3cuda::Status::kOk);

        aruco3cuda::DictionaryMatch match;
        ASSERT_EQ(aruco3cuda::match_dictionary(table, code, 0, &match), aruco3cuda::Status::kOk);
        EXPECT_EQ(match.id_, id);
        EXPECT_EQ(match.rotation_, 0);
        EXPECT_EQ(match.distance_, 0);

        // A rotated bit pattern also decodes, reported as the corresponding
        // rotation.
        aruco3cuda::MarkerCode rotated = code;
        for (int rotation = 1; rotation < 4; ++rotation) {
            ASSERT_EQ(aruco3cuda::rotate_marker_code(rotated, table.marker_size_, &rotated),
                      aruco3cuda::Status::kOk);
            aruco3cuda::DictionaryMatch rotated_match;
            ASSERT_EQ(aruco3cuda::match_dictionary(table, rotated, 0, &rotated_match),
                      aruco3cuda::Status::kOk);
            EXPECT_EQ(rotated_match.id_, id) << "rotation=" << rotation;
            EXPECT_EQ(rotated_match.rotation_, rotation);
        }
    }
}

// Check 4: accept / reject decisions under bit flips match OpenCV.
TEST_P(DictionaryConformanceTest, accept_reject_matches_opencv_for_bit_flips) {
    const cv::aruco::Dictionary dictionary = opencv_dictionary();
    const aruco3cuda::DictionaryTable& table = generated_table();
    // OpenCV uses maxCorrectionBits * maxCorrectionRate as the number of
    // tolerated errors.
    constexpr double kMaxCorrectionRate = 0.6;
    const int allowed_errors = static_cast<int>(dictionary.maxCorrectionBits * kMaxCorrectionRate);

    const std::vector<int> ids_under_test = sample_ids();
    for (const int id : ids_under_test) {
        for (int flips = 0; flips <= dictionary.maxCorrectionBits + 1; ++flips) {
            cv::Mat bits = cv::aruco::Dictionary::getBitsFromByteList(
                    dictionary.bytesList.rowRange(id, id + 1), dictionary.markerSize, 0);
            // Flip bits in a fixed order. No randomness, so that the OpenCV
            // side receives exactly the same input.
            for (int flip = 0; flip < flips; ++flip) {
                const int row = flip / dictionary.markerSize;
                const int col = flip % dictionary.markerSize;
                bits.at<std::uint8_t>(row, col) =
                        static_cast<std::uint8_t>(bits.at<std::uint8_t>(row, col) != 0 ? 0 : 1);
            }

            int opencv_id = -1;
            int opencv_rotation = -1;
            const bool opencv_accepted =
                    dictionary.identify(bits, opencv_id, opencv_rotation, kMaxCorrectionRate);

            const aruco3cuda::MarkerCode code = pack_from_mat(bits, table.marker_size_);
            aruco3cuda::DictionaryMatch match;
            ASSERT_EQ(aruco3cuda::match_dictionary(table, code, allowed_errors, &match),
                      aruco3cuda::Status::kOk);
            const bool ours_accepted = match.id_ >= 0;

            EXPECT_EQ(ours_accepted, opencv_accepted) << "id=" << id << " flips=" << flips;
            if (opencv_accepted && ours_accepted) {
                EXPECT_EQ(match.id_, opencv_id) << "id=" << id << " flips=" << flips;
                EXPECT_EQ(match.rotation_, opencv_rotation) << "id=" << id << " flips=" << flips;
            }
        }
    }
}

// Check 4b: matching directly from cell ratios also agrees with OpenCV.
//
// Check 4 feeds in a bit pattern (0 or 1). In real detection the ratios take
// intermediate values, and very close to the threshold they land in a third
// state that is "neither black nor white". That state cannot be expressed once
// it is collapsed into a single bit pattern, so here we cross-check on the
// ratios themselves.
TEST_P(DictionaryConformanceTest, identify_from_cell_ratios_matches_opencv) {
    const cv::aruco::Dictionary dictionary = opencv_dictionary();
    const aruco3cuda::DictionaryTable& table = generated_table();
    constexpr double kMaxCorrectionRate = 0.6;
    constexpr float kValidBitThreshold = 0.49F;

    // Candidate ratios, covering both sides of the 0.49 threshold and its 0.51
    // upper counterpart, as well as the boundaries themselves.
    const std::vector<float> ratio_values = {0.0F,  0.25F, 0.48F, 0.49F, 0.50F,
                                             0.51F, 0.52F, 0.75F, 1.0F};
    std::mt19937_64 rng(20260828U);
    std::uniform_int_distribution<std::size_t> pick(0U, ratio_values.size() - 1U);

    const std::vector<int> ids_under_test = sample_ids();
    std::size_t ambiguous_cases = 0;
    for (const int id : ids_under_test) {
        for (int trial = 0; trial < 200; ++trial) {
            // Start from the ID's codeword and disturb the ratio of only a few
            // cells. Fully random ratios would land far from every ID and would
            // never take the accepting path.
            cv::Mat bits = cv::aruco::Dictionary::getBitsFromByteList(
                    dictionary.bytesList.rowRange(id, id + 1), dictionary.markerSize, 0);
            cv::Mat ratios(dictionary.markerSize, dictionary.markerSize, CV_32FC1);
            const int perturbed = trial % 5;
            for (int r = 0; r < dictionary.markerSize; ++r) {
                for (int c = 0; c < dictionary.markerSize; ++c) {
                    ratios.at<float>(r, c) = bits.at<std::uint8_t>(r, c) != 0 ? 1.0F : 0.0F;
                }
            }
            for (int k = 0; k < perturbed; ++k) {
                const int cell = static_cast<int>(pick(rng) * 7U + static_cast<std::size_t>(k)) %
                                 (dictionary.markerSize * dictionary.markerSize);
                ratios.at<float>(cell / dictionary.markerSize, cell % dictionary.markerSize) =
                        ratio_values[pick(rng)];
            }

            int opencv_id = -1;
            int opencv_rotation = -1;
            const bool opencv_accepted = dictionary.identify(
                    ratios, opencv_id, opencv_rotation, kMaxCorrectionRate, kValidBitThreshold);

            aruco3cuda::CellMasks masks;
            ASSERT_EQ(aruco3cuda::build_cell_masks(ratios.ptr<float>(0), table.marker_size_,
                                                   kValidBitThreshold, &masks),
                      aruco3cuda::Status::kOk);
            // A bit set in both masks marks a cell that is neither black nor
            // white.
            if ((masks.not_black_ & masks.not_white_) != 0U) {
                ++ambiguous_cases;
            }
            aruco3cuda::DictionaryMatch match;
            ASSERT_EQ(aruco3cuda::identify_marker(table, masks, kMaxCorrectionRate, &match),
                      aruco3cuda::Status::kOk);

            EXPECT_EQ(match.id_ >= 0, opencv_accepted) << "id=" << id << " trial=" << trial;
            if (opencv_accepted && match.id_ >= 0) {
                EXPECT_EQ(match.id_, opencv_id) << "id=" << id << " trial=" << trial;
                EXPECT_EQ(match.rotation_, opencv_rotation) << "id=" << id << " trial=" << trial;
            }
        }
    }
    // Confirm that cases with ambiguous cells were actually exercised. If they
    // were not, this test only covers the bit-pattern case and adds nothing over
    // check 4.
    std::printf("[dict] cases containing an ambiguous cell: %zu\n", ambiguous_cases);
    EXPECT_GT(ambiguous_cases, 0U);
}

// Check 5: recomputing the minimum Hamming distance matches the nominal value.
TEST_P(DictionaryConformanceTest, minimum_hamming_distance_matches_nominal) {
    const aruco3cuda::DictionaryTable& table = generated_table();
    int distance = 0;
    ASSERT_EQ(aruco3cuda::minimum_hamming_distance(table, &distance), aruco3cuda::Status::kOk);
    EXPECT_EQ(distance, GetParam().minimum_distance_);

    // A code of minimum distance d can correct at most (d - 1) / 2 errors.
    // Checking that against the maxCorrectionBits OpenCV publishes ties our
    // recomputation to a number obtained independently, which comparing against
    // a literal alone would not do. It holds for every bundled dictionary.
    // DICT_ARUCO_ORIGINAL, which is not bundled, is where it breaks: its minimum
    // distance is 1 while OpenCV declares one correctable bit. See
    // docs/dictionaries.md for why that dictionary is left out.
    EXPECT_EQ(table.max_correction_bits_, (distance - 1) / 2);
}

INSTANTIATE_TEST_SUITE_P(AllMarkerSizes, DictionaryConformanceTest, ::testing::ValuesIn(kCases),
                         [](const ::testing::TestParamInfo<DictionaryCase>& parameter) {
                             return std::string(parameter.param.name_);
                         });

}  // namespace
