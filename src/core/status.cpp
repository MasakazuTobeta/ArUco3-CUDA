// SPDX-License-Identifier: Apache-2.0
#include "aruco3cuda/status.hpp"

#include <cstdio>

#include "aruco3cuda/version.hpp"

namespace aruco3cuda {
namespace {

/// Buffer holding the description of the last CUDA error.
///
/// Keeping it per thread prevents records from being mixed together when several streams are
/// driven from different threads.
constexpr int kCudaErrorMessageCapacity = 512;
thread_local char g_cuda_error_message[kCudaErrorMessageCapacity] = {};

}  // namespace

const char* version_string() {
    // The number of digits is known in advance, so no truncation can occur.
    static char buffer[32] = {};
    if (buffer[0] == '\0') {
        std::snprintf(buffer, sizeof(buffer), "%d.%d.%d", kVersionMajor, kVersionMinor,
                      kVersionPatch);
    }
    return buffer;
}

const char* to_string(Status status) {
    switch (status) {
        case Status::kOk:
            return "kOk";
        case Status::kInvalidArgument:
            return "kInvalidArgument";
        case Status::kInvalidImage:
            return "kInvalidImage";
        case Status::kInvalidConfig:
            return "kInvalidConfig";
        case Status::kUnsupportedDictionary:
            return "kUnsupportedDictionary";
        case Status::kCandidateOverflow:
            return "kCandidateOverflow";
        case Status::kMarkerOverflow:
            return "kMarkerOverflow";
        case Status::kCudaError:
            return "kCudaError";
        case Status::kNotInitialized:
            return "kNotInitialized";
    }
    // Never return nullptr, not even when a value outside the enum is passed in.
    return "kUnknown";
}

const char* last_cuda_error_message() {
    return g_cuda_error_message;
}

namespace detail {

/// Records the description of a CUDA error. Used from cuda_check.cpp.
void store_cuda_error_message(const char* api_name, const char* stage, int device_index,
                              const void* stream, const char* cuda_error_name,
                              const char* cuda_error_string) {
    // The stream is part of the context as well. Once several streams run concurrently, the
    // cause cannot be narrowed down without knowing which stream failed.
    std::snprintf(g_cuda_error_message, kCudaErrorMessageCapacity,
                  "api=%s stage=%s device=%d stream=%p error=%s (%s)", api_name, stage,
                  device_index, stream, cuda_error_name, cuda_error_string);
}

}  // namespace detail
}  // namespace aruco3cuda
