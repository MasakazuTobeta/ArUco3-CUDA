// SPDX-License-Identifier: Apache-2.0
#include "aruco3cuda/status.hpp"

#include <cstdio>

#include "aruco3cuda/version.hpp"

namespace aruco3cuda {
namespace {

/// CUDA エラーの説明を保持する buffer。
///
/// thread ごとに独立させることで、複数 stream を別 thread から扱う場合でも
/// 記録が混ざらないようにする。
constexpr int kCudaErrorMessageCapacity = 512;
thread_local char g_cuda_error_message[kCudaErrorMessageCapacity] = {};

}  // namespace

const char* version_string() {
    // 桁数は既知であり切り詰めは発生しない。
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
    // enum に無い値が渡された場合も nullptr を返さない。
    return "kUnknown";
}

const char* last_cuda_error_message() {
    return g_cuda_error_message;
}

namespace detail {

/// CUDA エラーの説明を記録する。cuda_check.cpp から使用する。
void store_cuda_error_message(const char* api_name, const char* stage, int device_index,
                              const void* stream, const char* cuda_error_name,
                              const char* cuda_error_string) {
    // stream も文脈へ含める。複数 stream を並行して使う段階で、
    // どの stream の失敗かを特定できないと原因を絞り込めない。
    std::snprintf(g_cuda_error_message, kCudaErrorMessageCapacity,
                  "api=%s stage=%s device=%d stream=%p error=%s (%s)", api_name, stage,
                  device_index, stream, cuda_error_name, cuda_error_string);
}

}  // namespace detail
}  // namespace aruco3cuda
