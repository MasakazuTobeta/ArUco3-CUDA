// SPDX-License-Identifier: Apache-2.0
//
// Verifies the nominal and boundary behavior of Status and of the version constants.
#include "aruco3cuda/status.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <set>
#include <string>

#include "aruco3cuda/version.hpp"

namespace {

// Nominal: every enumerator returns a distinct, non-empty identifier.
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
        EXPECT_TRUE(seen.insert(text).second) << "duplicate identifier: " << text;
    }
}

// Error case: a value outside the enumeration must not yield nullptr.
TEST(StatusTest, to_string_handles_out_of_range_value) {
    const auto invalid = static_cast<aruco3cuda::Status>(9999);
    const char* text = aruco3cuda::to_string(invalid);
    ASSERT_NE(text, nullptr);
    EXPECT_STREQ(text, "kUnknown");
}

// Boundary: a valid string is returned even before any CUDA call has been made.
TEST(StatusTest, last_cuda_error_message_is_valid_before_any_cuda_call) {
    const char* message = aruco3cuda::last_cuda_error_message();
    ASSERT_NE(message, nullptr);
}

// Nominal: the version string matches its individual components.
TEST(VersionTest, version_string_matches_components) {
    const std::string expected = std::to_string(aruco3cuda::kVersionMajor) + "." +
                                 std::to_string(aruco3cuda::kVersionMinor) + "." +
                                 std::to_string(aruco3cuda::kVersionPatch);
    EXPECT_EQ(std::string(aruco3cuda::version_string()), expected);
}

}  // namespace
