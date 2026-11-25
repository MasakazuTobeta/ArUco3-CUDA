// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "tag_decode.hpp"

#include <algorithm>
#include <optional>
#include <utility>
#include <vector>

namespace aruco3cuda::generated {
extern const DictionaryTable kDictApriltag36h11Table;
}

namespace aruco3cuda::apriltag_small_cpu::detail {
constexpr int kPayloadSide = 6;
constexpr int kBorderSide = kPayloadSide + 2;
constexpr int kRingSamples = 4 * kBorderSide;
constexpr int kRadius = 5;
constexpr int kNoMatchDistance = kPayloadSide * kPayloadSide + 1;

struct Decode {
    int id = -1;
    int rotation = 0;
    int distance = kNoMatchDistance;
    bool supported = false;
};

// Split the 36 bits into three disjoint words. Within two errors at least one
// word is exact. Within five errors at least one word differs by at most one bit.
// The full Hamming distance verifies the match and is unique through radius five.
class CodeIndex {
    static constexpr int kWordBits = 12;
    static constexpr int kBuckets = 1 << kWordBits;
    struct Group {
        std::array<std::uint32_t, kBuckets + 1> offsets{};
        std::array<std::uint32_t, kBuckets> exact_end{};
        std::vector<std::uint16_t> indices;
    };
    using Groups = std::array<Group, 3>;

    static int word(MarkerCode code, int group) {
        return (code >> (kWordBits * group)) & (kBuckets - 1);
    }

    static const Groups& groups() {
        static const Groups table = [] {
            Groups result;
            const auto& dictionary = generated::kDictApriltag36h11Table;
            for (int i = 0; i < 3; ++i) {
                auto& group = result[i];
                std::vector<std::vector<std::uint16_t>> exact(kBuckets);
                for (int j = 0; j < 4 * dictionary.code_count_; ++j)
                    exact[word(dictionary.codes_[j], i)].push_back(static_cast<std::uint16_t>(j));
                auto& indices = group.indices;
                indices.reserve((1 + kWordBits) * 4 * dictionary.code_count_);
                for (int key = 0; key < kBuckets; ++key) {
                    group.offsets[key] = indices.size();
                    indices.insert(indices.end(), exact[key].begin(), exact[key].end());
                    group.exact_end[key] = indices.size();
                    for (int bit = 0; bit < kWordBits; ++bit) {
                        const auto& neighbours = exact[key ^ (1 << bit)];
                        indices.insert(indices.end(), neighbours.begin(), neighbours.end());
                    }
                }
                group.offsets.back() = indices.size();
            }
            return result;
        }();
        return table;
    }

public:
    static std::pair<int, int> match(MarkerCode code, int radius = kRadius) {
        const auto& table = groups();
        const auto& dictionary = generated::kDictApriltag36h11Table;
        const int keys[] = {word(code, 0), word(code, 1), word(code, 2)};
        for (int pass = 0; pass < (radius > 2 ? 2 : 1); ++pass) {
            for (int i = 0; i < 3; ++i) {
                const auto& group = table[i];
                const int key = keys[i];
                const auto first = pass ? group.exact_end[key] : group.offsets[key];
                const auto end = pass ? group.offsets[key + 1] : group.exact_end[key];
                for (auto at = first; at < end; ++at) {
                    const int index = group.indices[at];
                    const int distance = bit_count(code ^ dictionary.codes_[index]);
                    if (distance <= radius)
                        return {index, distance};
                }
            }
        }
        return {-1, kNoMatchDistance};
    }
};

struct SampledPayload {
    std::optional<MarkerCode> bits;
    // A visible ring can justify refining a quad even when a payload cell falls outside the image.
    bool ring_supported = false;
};

template <bool kVisibleExterior = false>
inline SampledPayload sample_payload(const cv::Mat& gray, const Quad& corners,
                                     int minimum_ring_matches, float minimum_contrast) {
    Projection projection;
    if (!project(corners, projection)) {
        return {};
    }
    std::array<std::array<LumaSample, kRingSamples>, 2> ring{};
    constexpr float kInset = 0.5F / kBorderSide;
    int positive = 0;
    for (int j = 0; j < kBorderSide; ++j) {
        const float position = (j + 0.5F) / kBorderSide;
        for (int side = 0; side < 4; ++side) {
            const int sample = 4 * j + side;
            for (int color = 0; color < 2; ++color) {
                const float offset = color != 0 ? -kInset : kInset;
                const float x = side == 0 ? offset : side == 2 ? 1 - offset : position;
                const float y = side == 1 ? offset : side == 3 ? 1 - offset : position;
                ring[color][sample] = {x - 0.5F, y - 0.5F, projection.sample(gray, x, y)};
                if constexpr (kVisibleExterior) {
                    if (color == 1 && ring[color][sample].intensity < 0) {
                        // A complete print can have less than half a cell of visible white
                        // surround. Measure the same exterior ray inside the exposed image.
                        for (const float near : {.5F * kInset, .25F * kInset}) {
                            const float u = side == 0 ? -near : side == 2 ? 1 + near : position;
                            const float v = side == 1 ? -near : side == 3 ? 1 + near : position;
                            ring[color][sample] = {u - .5F, v - .5F, projection.sample(gray, u, v)};
                            if (ring[color][sample].intensity >= 0)
                                break;
                        }
                    }
                }
            }
            const float black = ring[0][sample].intensity;
            const float white = ring[1][sample].intensity;
            positive += black >= 0 && white >= black + minimum_contrast * 0.65F;
            if (positive + kRingSamples - sample - 1 < minimum_ring_matches) {
                return {};
            }
        }
    }
    Plane dark;
    Plane light;
    if (!fit_plane(ring[0], dark) || !fit_plane(ring[1], light) ||
        light.center - dark.center < minimum_contrast) {
        return {};
    }
    MarkerCode code = 0;
    for (int y = 0; y < kPayloadSide; ++y) {
        for (int x = 0; x < kPayloadSide; ++x) {
            const float u = (x + 1.5F) / kBorderSide;
            const float v = (y + 1.5F) / kBorderSide;
            const float threshold =
                    0.5F * (dark.at(u - 0.5F, v - 0.5F) + light.at(u - 0.5F, v - 0.5F));
            const float intensity = projection.sample(gray, u, v);
            if (intensity < 0)
                return {std::nullopt, true};
            code |= static_cast<MarkerCode>(intensity > threshold) << (y * kPayloadSide + x);
        }
    }
    return {code, true};
}

template <bool kVisibleExterior = false>
inline Decode decode(const cv::Mat& gray, const Quad& corners, int minimum_ring_matches,
                     float minimum_contrast, int maximum_corrected_bits) {
    const auto payload =
            sample_payload<kVisibleExterior>(gray, corners, minimum_ring_matches, minimum_contrast);
    if (!payload.bits)
        return {-1, 0, kNoMatchDistance, payload.ring_supported};
    const auto [matched, distance] = CodeIndex::match(*payload.bits);
    if (matched >= 0 && distance <= maximum_corrected_bits) {
        return {matched / 4, matched % 4, distance, true};
    }
    return {-1, 0, distance, true};
}

// At distance five or less, a 36h11 word identifies a unique ID and rotation. A refit
// can verify its known word directly and report the same correction count as discovery.
template <bool kVisibleExterior = false>
inline std::optional<int> verify_payload(const cv::Mat& gray, const Quad& corners, int id,
                                         int minimum_ring_matches, float minimum_contrast,
                                         int maximum_corrected_bits, int rotation = 0) {
    const auto& dictionary = generated::kDictApriltag36h11Table;
    if (id < 0 || id >= dictionary.code_count_ || rotation < 0 || rotation >= 4)
        return std::nullopt;
    const auto payload =
            sample_payload<kVisibleExterior>(gray, corners, minimum_ring_matches, minimum_contrast);
    if (!payload.bits)
        return std::nullopt;
    const int errors = bit_count(*payload.bits ^ dictionary.codes_[4 * id + rotation]);
    if (errors > std::min(kRadius, maximum_corrected_bits))
        return std::nullopt;
    return errors;
}

}  // namespace aruco3cuda::apriltag_small_cpu::detail
