// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#if defined(__aarch64__) && defined(__AARCH64EL__)
#include <arm_neon.h>

namespace aruco3cuda::cpu::detail {

/// Pack the results of Gray8 vector comparisons in pixel order.
/// Ownership applies to every method. The instance owns its lane weights and borrows no storage.
/// Synchronization applies to every method. The operations run synchronously and pack is read-only.
/// Example input is four consecutive comparison vectors with only the first pixel selected.
/// Example output is the word 1.
class GrayComparisonBits {
public:
    GrayComparisonBits() : weights_(vld1q_u8(kBitWeights)) {}

    /// Convert 64 comparison bytes to one bit per pixel.
    /// @param first Comparison bytes for pixels 0 through 15.
    /// @param second Comparison bytes for pixels 16 through 31.
    /// @param third Comparison bytes for pixels 32 through 47.
    /// @param fourth Comparison bytes for pixels 48 through 63.
    /// @return The selection word, with pixel zero in its least significant bit.
    /// Example input is four vectors of 0 or 255 with only the last pixel selected.
    /// Example output is the word with bit 63 set.
    std::uint64_t pack(uint8x16_t first, uint8x16_t second, uint8x16_t third,
                       uint8x16_t fourth) const {
        const auto low =
                vpaddq_u8(vandq_u8(first, this->weights_), vandq_u8(second, this->weights_));
        const auto high =
                vpaddq_u8(vandq_u8(third, this->weights_), vandq_u8(fourth, this->weights_));
        const auto pairs = vpaddq_u8(low, high);
        const auto bytes = vpadd_u8(vget_low_u8(pairs), vget_high_u8(pairs));
        return vget_lane_u64(vreinterpret_u64_u8(bytes), 0);
    }

private:
    static constexpr std::uint8_t kBitWeights[16] = {1, 2, 4, 8, 16, 32, 64, 128,
                                                     1, 2, 4, 8, 16, 32, 64, 128};
    uint8x16_t weights_;
};

}  // namespace aruco3cuda::cpu::detail
#endif
