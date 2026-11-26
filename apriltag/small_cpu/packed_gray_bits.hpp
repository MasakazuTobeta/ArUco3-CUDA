// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cpu_gray_bits.hpp"

#if defined(__aarch64__) && defined(__AARCH64EL__)
namespace aruco3cuda::apriltag_small_cpu::detail {
using aruco3cuda::cpu::detail::GrayComparisonBits;
}
#endif
