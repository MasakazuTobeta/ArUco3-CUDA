// SPDX-License-Identifier: Apache-2.0
#include "aruco3cuda/dictionary.hpp"

#include <cstddef>
#include <cstdint>

#include "aruco3cuda/status.hpp"

namespace aruco3cuda {
namespace {

constexpr int kMaxMarkerSize = 7;
constexpr int kMaxBitCount = kMaxMarkerSize * kMaxMarkerSize;

/// 立っている bit 数を数える。CUDA 側では __popcll に置き換わる。
inline int popcount64(std::uint64_t value) {
    int count = 0;
    while (value != 0U) {
        value &= value - 1U;
        ++count;
    }
    return count;
}

bool is_valid_marker_size(int marker_size) {
    return marker_size >= 1 && marker_size <= kMaxMarkerSize;
}

}  // namespace

Status pack_marker_code(const std::uint8_t* bits, int marker_size, MarkerCode* out) {
    if (bits == nullptr || out == nullptr || !is_valid_marker_size(marker_size)) {
        return Status::kInvalidArgument;
    }
    const int bit_count = marker_size * marker_size;
    MarkerCode code = 0U;
    for (int i = 0; i < bit_count; ++i) {
        if (bits[i] != 0U) {
            code |= (static_cast<MarkerCode>(1) << i);
        }
    }
    *out = code;
    return Status::kOk;
}

Status unpack_marker_code(MarkerCode code, int marker_size, std::uint8_t* out_bits) {
    if (out_bits == nullptr || !is_valid_marker_size(marker_size)) {
        return Status::kInvalidArgument;
    }
    const int bit_count = marker_size * marker_size;
    for (int i = 0; i < bit_count; ++i) {
        out_bits[i] = static_cast<std::uint8_t>((code >> i) & 1U);
    }
    return Status::kOk;
}

Status rotate_marker_code(MarkerCode code, int marker_size, MarkerCode* out) {
    if (out == nullptr || !is_valid_marker_size(marker_size)) {
        return Status::kInvalidArgument;
    }
    std::uint8_t source[kMaxBitCount] = {};
    std::uint8_t rotated[kMaxBitCount] = {};
    const Status unpack_status = unpack_marker_code(code, marker_size, source);
    if (unpack_status != Status::kOk) {
        return unpack_status;
    }
    // 反時計回りに 90 度回転する。
    // 元の (row, col) は回転後の (marker_size - 1 - col, row) へ移る。
    for (int row = 0; row < marker_size; ++row) {
        for (int col = 0; col < marker_size; ++col) {
            const int destination_row = marker_size - 1 - col;
            const int destination_col = row;
            rotated[destination_row * marker_size + destination_col] =
                    source[row * marker_size + col];
        }
    }
    return pack_marker_code(rotated, marker_size, out);
}

Status match_dictionary(const DictionaryTable& table, MarkerCode candidate,
                        int max_correction_errors, DictionaryMatch* out) {
    if (out == nullptr || table.codes_ == nullptr || table.code_count_ <= 0 ||
        !is_valid_marker_size(table.marker_size_) || max_correction_errors < 0) {
        return Status::kInvalidArgument;
    }

    int best_id = -1;
    int best_rotation = 0;
    int best_distance = table.bit_count() + 1;

    for (int id = 0; id < table.code_count_; ++id) {
        for (int rotation = 0; rotation < 4; ++rotation) {
            const MarkerCode code = table.codes_[static_cast<std::size_t>(id) * 4U +
                                                 static_cast<std::size_t>(rotation)];
            const int distance = popcount64(candidate ^ code);
            if (distance < best_distance) {
                best_distance = distance;
                best_id = id;
                best_rotation = rotation;
            }
        }
    }

    if (best_distance > max_correction_errors) {
        out->id_ = -1;
        out->rotation_ = 0;
        out->distance_ = best_distance;
        return Status::kOk;
    }
    out->id_ = best_id;
    out->rotation_ = best_rotation;
    out->distance_ = best_distance;
    return Status::kOk;
}

Status minimum_hamming_distance(const DictionaryTable& table, int* out_distance) {
    if (out_distance == nullptr || table.codes_ == nullptr || table.code_count_ <= 1 ||
        !is_valid_marker_size(table.marker_size_)) {
        return Status::kInvalidArgument;
    }
    int minimum = table.bit_count() + 1;
    for (int a = 0; a < table.code_count_; ++a) {
        // 回転 0 を代表とし、相手側は 4 回転すべてと比較する。
        // 自分自身の回転同士は Dictionary の定義上比較対象にしない。
        const MarkerCode code_a = table.codes_[static_cast<std::size_t>(a) * 4U];
        for (int b = a + 1; b < table.code_count_; ++b) {
            for (int rotation = 0; rotation < 4; ++rotation) {
                const int distance =
                        popcount64(code_a ^ table.codes_[static_cast<std::size_t>(b) * 4U +
                                                         static_cast<std::size_t>(rotation)]);
                if (distance < minimum) {
                    minimum = distance;
                }
            }
        }
    }
    *out_distance = minimum;
    return Status::kOk;
}

}  // namespace aruco3cuda
