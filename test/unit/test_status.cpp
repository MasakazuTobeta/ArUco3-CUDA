// SPDX-License-Identifier: Apache-2.0
//
// Status と version の正常系および境界値を検証する。
#include "aruco3cuda/status.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <set>
#include <string>

#include "aruco3cuda/version.hpp"

namespace {

// 正常系: 全ての列挙子が固有かつ空でない識別子を返す。
TEST(StatusTest, to_string_returns_unique_identifier_for_each_value) {
    const aruco3cuda::Status values[] = {
            aruco3cuda::Status::kOk,
            aruco3cuda::Status::kInvalidArgument,
            aruco3cuda::Status::kInvalidConfig,
            aruco3cuda::Status::kUnsupportedDictionary,
            aruco3cuda::Status::kCandidateOverflow,
            aruco3cuda::Status::kMarkerOverflow,
            aruco3cuda::Status::kCudaError,
            aruco3cuda::Status::kNotInitialized,
    };
    std::set<std::string> seen;
    for (const aruco3cuda::Status value : values) {
        const char* text = aruco3cuda::to_string(value);
        ASSERT_NE(text, nullptr);
        EXPECT_NE(std::strlen(text), 0U);
        EXPECT_TRUE(seen.insert(text).second) << "識別子が重複している: " << text;
    }
}

// 異常系: 列挙に無い値が渡されても nullptr を返さない。
TEST(StatusTest, to_string_handles_out_of_range_value) {
    const auto invalid = static_cast<aruco3cuda::Status>(9999);
    const char* text = aruco3cuda::to_string(invalid);
    ASSERT_NE(text, nullptr);
    EXPECT_STREQ(text, "kUnknown");
}

// 境界値: CUDA を呼ぶ前でも有効な文字列を返す。
TEST(StatusTest, last_cuda_error_message_is_valid_before_any_cuda_call) {
    const char* message = aruco3cuda::last_cuda_error_message();
    ASSERT_NE(message, nullptr);
}

// 正常系: version 文字列が構成要素と一致する。
TEST(VersionTest, version_string_matches_components) {
    const std::string expected = std::to_string(aruco3cuda::kVersionMajor) + "." +
                                 std::to_string(aruco3cuda::kVersionMinor) + "." +
                                 std::to_string(aruco3cuda::kVersionPatch);
    EXPECT_EQ(std::string(aruco3cuda::version_string()), expected);
}

}  // namespace
