// SPDX-License-Identifier: Apache-2.0
#include "aruco3cuda/config.hpp"

#if defined(__linux__) && defined(__aarch64__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#include <sys/prctl.h>
#endif

#ifndef ARUCO3CUDA_RESIZE_ARM_SVE_BYTES
#define ARUCO3CUDA_RESIZE_ARM_SVE_BYTES -1
#endif

namespace aruco3cuda {

const char* compiled_resize_profile() noexcept {
#if defined(ARUCO3CUDA_RESIZE_PROFILE_OPENCV_ARM_HAL) && ARUCO3CUDA_RESIZE_PROFILE_OPENCV_ARM_HAL
    return "OPENCV_ARM_HAL";
#else
    return "GENERIC";
#endif
}

int resize_sve_vector_bytes() noexcept {
#if defined(ARUCO3CUDA_RESIZE_PROFILE_OPENCV_ARM_HAL) && ARUCO3CUDA_RESIZE_PROFILE_OPENCV_ARM_HAL
#if ARUCO3CUDA_RESIZE_ARM_SVE_BYTES >= 0
    return ARUCO3CUDA_RESIZE_ARM_SVE_BYTES;
#else
    static const int bytes = []() noexcept {
#if defined(__linux__) && defined(__aarch64__) && defined(HWCAP2_SVE2) && defined(PR_SVE_GET_VL)
        if ((getauxval(AT_HWCAP2) & HWCAP2_SVE2) != 0UL) {
            const int setting = prctl(PR_SVE_GET_VL);
            if (setting >= 0) {
                const int length = setting & PR_SVE_VL_LEN_MASK;
                if (length == 16 || length == 32 || length == 64) {
                    return length;
                }
            }
        }
#endif
        return 0;
    }();
    return bytes;
#endif
#else
    return 0;
#endif
}

const char* compiled_resize_backend() noexcept {
#if defined(ARUCO3CUDA_RESIZE_PROFILE_OPENCV_ARM_HAL) && ARUCO3CUDA_RESIZE_PROFILE_OPENCV_ARM_HAL
    switch (resize_sve_vector_bytes()) {
        case 16:
            return "SVE128";
        case 32:
            return "SVE256";
        case 64:
            return "SVE512";
        default:
            return "NEON";
    }
#else
    return "GENERIC";
#endif
}

bool resize_carotene_x_fma() noexcept {
#if defined(ARUCO3CUDA_RESIZE_PROFILE_OPENCV_ARM_HAL) &&                                  \
        ARUCO3CUDA_RESIZE_PROFILE_OPENCV_ARM_HAL && defined(ARUCO3CUDA_CAROTENE_X_FMA) && \
        ARUCO3CUDA_CAROTENE_X_FMA
    return true;
#else
    return false;
#endif
}

const char* compiled_carotene_x_fma() noexcept {
#if defined(ARUCO3CUDA_RESIZE_PROFILE_OPENCV_ARM_HAL) && ARUCO3CUDA_RESIZE_PROFILE_OPENCV_ARM_HAL
    return resize_carotene_x_fma() ? "FUSED" : "SEPARATE";
#else
    return "NOT_APPLICABLE";
#endif
}

}  // namespace aruco3cuda
