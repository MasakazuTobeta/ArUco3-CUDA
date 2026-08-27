// SPDX-License-Identifier: Apache-2.0
//
// docs/dictionaries.md が Dictionary ごとに必須と定める検証 1 から 5 を実装する。
// 生成済み table を、その場で取得した OpenCV の Dictionary と突き合わせる。
#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "aruco3cuda/dictionary.hpp"
#include "aruco3cuda/status.hpp"

namespace {

const aruco3cuda::DictionaryTable& generated_table() {
    const aruco3cuda::DictionaryTable* table =
            aruco3cuda::find_builtin_dictionary("DICT_ARUCO_MIP_36h12");
    EXPECT_NE(table, nullptr);
    return *table;
}

cv::aruco::Dictionary opencv_dictionary() {
    return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_MIP_36h12);
}

/// OpenCV の bits Mat を packed 表現へ変換する。
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

// 検証 1: ID 数、marker size、最大訂正 bit 数が OpenCV 基準と一致する。
TEST(DictionaryConformanceTest, metadata_matches_opencv) {
    const cv::aruco::Dictionary dictionary = opencv_dictionary();
    const aruco3cuda::DictionaryTable& table = generated_table();
    EXPECT_EQ(table.code_count_, dictionary.bytesList.rows);
    EXPECT_EQ(table.marker_size_, dictionary.markerSize);
    EXPECT_EQ(table.max_correction_bits_, dictionary.maxCorrectionBits);
}

// 検証 2: 全 ID、全 4 回転の packed codeword が OpenCV の bytesList と一致する。
TEST(DictionaryConformanceTest, all_codewords_match_opencv_bytes_list) {
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

// 検証 3: 全 ID の marker image を decode し、元の ID と回転を得られる。
TEST(DictionaryConformanceTest, decodes_generated_marker_images) {
    const cv::aruco::Dictionary dictionary = opencv_dictionary();
    const aruco3cuda::DictionaryTable& table = generated_table();

    for (int id = 0; id < table.code_count_; ++id) {
        // generateImageMarker は borderBits > 0 を要求する。border 込みで
        // 1 cell = 1 pixel になる辺長を指定し、内側の bit 格子を切り出す。
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
                // 白 (255) が bit 1、黒 (0) が bit 0。
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

        // 回転させた bit 列も、対応する rotation として復号できる。
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

// 検証 4: bit 反転に対する accept / reject が OpenCV と一致する。
TEST(DictionaryConformanceTest, accept_reject_matches_opencv_for_bit_flips) {
    const cv::aruco::Dictionary dictionary = opencv_dictionary();
    const aruco3cuda::DictionaryTable& table = generated_table();
    // OpenCV は maxCorrectionBits * maxCorrectionRate を許容誤り数として使う。
    constexpr double kMaxCorrectionRate = 0.6;
    const int allowed_errors = static_cast<int>(dictionary.maxCorrectionBits * kMaxCorrectionRate);
    ASSERT_EQ(allowed_errors, 3);

    const std::vector<int> ids_under_test = {0, 1, 42, 128, table.code_count_ - 1};
    for (const int id : ids_under_test) {
        for (int flips = 0; flips <= dictionary.maxCorrectionBits + 1; ++flips) {
            cv::Mat bits = cv::aruco::Dictionary::getBitsFromByteList(
                    dictionary.bytesList.rowRange(id, id + 1), dictionary.markerSize, 0);
            // 決まった順序で bit を反転する。OpenCV 側と同じ入力を作るため
            // 乱数を使わない。
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

// 検証 5: 最小 Hamming 距離を再計算し、公称値と一致する。
TEST(DictionaryConformanceTest, minimum_hamming_distance_matches_nominal) {
    const aruco3cuda::DictionaryTable& table = generated_table();
    int distance = 0;
    ASSERT_EQ(aruco3cuda::minimum_hamming_distance(table, &distance), aruco3cuda::Status::kOk);
    EXPECT_EQ(distance, 12);
}

}  // namespace
