// SPDX-License-Identifier: Apache-2.0
//
// Cross-checks the GPU dictionary matching against the CPU reference and
// OpenCV.
//
// The completion criterion is that every ID at all four rotations returns the
// same ID, rotation, and distance as the CPU. On top of that we include the
// case where a ratio sits near the threshold and is decided neither black nor
// white. Without agreement in that case there is nothing to distinguish this
// implementation from one that collapses everything into a bit pattern.
#include "dictionary_match.hpp"

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <opencv2/core.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/dictionary.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"
#include "aruco3cuda/workspace.hpp"
#include "candidate_filter.hpp"
#include "cell_decode.hpp"

namespace {

using aruco3cuda::DetectorConfig;
using aruco3cuda::MemorySpace;
using aruco3cuda::Status;
using aruco3cuda::Workspace;

bool has_cuda_device() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

const aruco3cuda::DictionaryTable& generated_table() {
    const aruco3cuda::DictionaryTable* table =
            aruco3cuda::find_builtin_dictionary("DICT_ARUCO_MIP_36h12");
    EXPECT_NE(table, nullptr);
    return *table;
}

cv::aruco::Dictionary opencv_dictionary() {
    return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_MIP_36h12);
}

/// Cell ratios for a single candidate; a cells-by-cells matrix including the
/// border.
struct RatioGrid {
    std::vector<float> values_;
    int cells_ = 0;

    /// Returns the inner cells in row-major order, the form the CPU reference
    /// expects.
    std::vector<float> inner(int border, int marker_size) const {
        std::vector<float> flat(static_cast<std::size_t>(marker_size) *
                                static_cast<std::size_t>(marker_size));
        for (int r = 0; r < marker_size; ++r) {
            for (int c = 0; c < marker_size; ++c) {
                flat[(static_cast<std::size_t>(r) * static_cast<std::size_t>(marker_size)) +
                     static_cast<std::size_t>(c)] =
                        this->values_[(static_cast<std::size_t>(r + border) *
                                       static_cast<std::size_t>(this->cells_)) +
                                      static_cast<std::size_t>(c + border)];
            }
        }
        return flat;
    }
};

/// Builds a ratio matrix from a codeword. The outer border is black (ratio 0).
RatioGrid grid_from_bits(const cv::Mat& bits, int border) {
    RatioGrid grid;
    grid.cells_ = bits.rows + (2 * border);
    grid.values_.assign(
            static_cast<std::size_t>(grid.cells_) * static_cast<std::size_t>(grid.cells_), 0.0F);
    for (int r = 0; r < bits.rows; ++r) {
        for (int c = 0; c < bits.cols; ++c) {
            grid.values_[(static_cast<std::size_t>(r + border) *
                          static_cast<std::size_t>(grid.cells_)) +
                         static_cast<std::size_t>(c + border)] =
                    bits.at<std::uint8_t>(r, c) != 0 ? 1.0F : 0.0F;
        }
    }
    return grid;
}

/// Hands the ratio matrix to the device, matches, and brings the result back.
class MatchRun {
public:
    MatchRun() = default;
    MatchRun(const MatchRun&) = delete;
    MatchRun& operator=(const MatchRun&) = delete;

    bool run(const std::vector<RatioGrid>& grids, const std::vector<std::uint8_t>& accepted,
             const aruco3cuda::DictionaryTable& table, const DetectorConfig& base) {
        DetectorConfig config = base;
        config.max_candidates_ = static_cast<int>(grids.size());
        config.max_markers_ = config.max_candidates_;
        const int marker_size = table.marker_size_;
        const int cells = grids[0].cells_;

        const std::size_t bytes =
                aruco3cuda::detail::candidate_workspace_bytes(config, 64, 64) +
                aruco3cuda::detail::cell_ratio_workspace_bytes(config, marker_size) +
                aruco3cuda::detail::device_dictionary_workspace_bytes(table) +
                aruco3cuda::detail::match_workspace_bytes(config);
        if (bytes == 0U ||
            this->workspace_.ensure_capacity(bytes, MemorySpace::kDevice, nullptr) != Status::kOk) {
            return false;
        }
        this->workspace_.reset();

        aruco3cuda::detail::CandidateFilterBuffers filter;
        aruco3cuda::detail::DeviceCandidates candidates;
        aruco3cuda::detail::CellRatioBuffers ratios;
        aruco3cuda::detail::DeviceDictionary dictionary;
        aruco3cuda::detail::MatchBuffers matches;
        if (aruco3cuda::detail::reserve_candidates(config, 64, 64, this->workspace_, &filter,
                                                   &candidates) != Status::kOk ||
            aruco3cuda::detail::reserve_cell_ratios(config, marker_size, this->workspace_,
                                                    &ratios) != Status::kOk ||
            aruco3cuda::detail::upload_dictionary(table, this->workspace_, &dictionary, nullptr) !=
                    Status::kOk ||
            aruco3cuda::detail::reserve_matches(config, this->workspace_, &matches) !=
                    Status::kOk) {
            return false;
        }
        if (ratios.cells_per_side_ != cells) {
            return false;
        }

        const auto plane = static_cast<std::size_t>(cells) * static_cast<std::size_t>(cells);
        std::vector<float> raw(grids.size() * plane);
        for (std::size_t i = 0; i < grids.size(); ++i) {
            if (grids[i].values_.size() != plane) {
                return false;
            }
            std::memcpy(raw.data() + (i * plane), grids[i].values_.data(), plane * sizeof(float));
        }
        const auto count = static_cast<int>(grids.size());
        if (cudaMemcpy(ratios.ratios_, raw.data(), raw.size() * sizeof(float),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(ratios.accepted_, accepted.data(), accepted.size(),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(candidates.count_, &count, sizeof(int), cudaMemcpyHostToDevice) !=
                    cudaSuccess) {
            return false;
        }
        if (aruco3cuda::detail::match_candidates_async(ratios, candidates, dictionary, config,
                                                       &matches, nullptr) != Status::kOk) {
            return false;
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            return false;
        }

        this->ids_.resize(grids.size());
        this->rotations_.resize(grids.size());
        this->distances_.resize(grids.size());
        const std::size_t plane_bytes = grids.size() * sizeof(std::int32_t);
        return cudaMemcpy(this->ids_.data(), matches.ids_, plane_bytes, cudaMemcpyDeviceToHost) ==
                       cudaSuccess &&
               cudaMemcpy(this->rotations_.data(), matches.rotations_, plane_bytes,
                          cudaMemcpyDeviceToHost) == cudaSuccess &&
               cudaMemcpy(this->distances_.data(), matches.distances_, plane_bytes,
                          cudaMemcpyDeviceToHost) == cudaSuccess;
    }

    const std::vector<std::int32_t>& ids() const { return this->ids_; }
    const std::vector<std::int32_t>& rotations() const { return this->rotations_; }
    const std::vector<std::int32_t>& distances() const { return this->distances_; }

private:
    Workspace workspace_;
    std::vector<std::int32_t> ids_;
    std::vector<std::int32_t> rotations_;
    std::vector<std::int32_t> distances_;
};

// Happy path: every ID at all four rotations agrees with the CPU reference and
// with OpenCV.
TEST(DictionaryMatchTest, matches_all_ids_and_rotations) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    const cv::aruco::Dictionary dictionary = opencv_dictionary();
    const aruco3cuda::DictionaryTable& table = generated_table();
    const DetectorConfig config;
    const int border = config.marker_border_bits_;

    std::vector<RatioGrid> grids;
    std::vector<int> expected_ids;
    std::vector<int> expected_rotations;
    grids.reserve(static_cast<std::size_t>(table.code_count_) * 4U);
    for (int id = 0; id < table.code_count_; ++id) {
        for (int rotation = 0; rotation < 4; ++rotation) {
            const cv::Mat bits = cv::aruco::Dictionary::getBitsFromByteList(
                    dictionary.bytesList.rowRange(id, id + 1), dictionary.markerSize, rotation);
            grids.push_back(grid_from_bits(bits, border));
            expected_ids.push_back(id);
            expected_rotations.push_back(rotation);
        }
    }
    const std::vector<std::uint8_t> accepted(grids.size(), 1U);

    MatchRun run;
    ASSERT_TRUE(run.run(grids, accepted, table, config));

    std::size_t id_mismatch = 0;
    std::size_t rotation_mismatch = 0;
    std::size_t opencv_mismatch = 0;
    for (std::size_t i = 0; i < grids.size(); ++i) {
        if (run.ids()[i] != expected_ids[i]) {
            ++id_mismatch;
        }
        if (run.rotations()[i] != expected_rotations[i]) {
            ++rotation_mismatch;
        }
        EXPECT_EQ(run.distances()[i], 0) << i;

        // Also cross-check against what OpenCV returns for the same ratios.
        cv::Mat ratio(dictionary.markerSize, dictionary.markerSize, CV_32FC1);
        const std::vector<float> flat = grids[i].inner(border, dictionary.markerSize);
        std::memcpy(ratio.ptr<float>(0), flat.data(), flat.size() * sizeof(float));
        int opencv_id = -1;
        int opencv_rotation = -1;
        const bool accepted_by_opencv = dictionary.identify(
                ratio, opencv_id, opencv_rotation, config.error_correction_rate_,
                static_cast<float>(config.valid_bit_threshold_));
        if (!accepted_by_opencv || opencv_id != run.ids()[i] ||
            opencv_rotation != run.rotations()[i]) {
            ++opencv_mismatch;
        }
    }
    std::printf(
            "[match] %zu cases: ID mismatches %zu, rotation mismatches %zu, "
            "mismatches vs OpenCV %zu\n",
            grids.size(), id_mismatch, rotation_mismatch, opencv_mismatch);
    EXPECT_EQ(id_mismatch, 0U);
    EXPECT_EQ(rotation_mismatch, 0U);
    EXPECT_EQ(opencv_mismatch, 0U);
}

// Boundary: ratios sitting near the threshold also agree with the CPU
// reference.
TEST(DictionaryMatchTest, matches_ambiguous_ratios) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    const cv::aruco::Dictionary dictionary = opencv_dictionary();
    const aruco3cuda::DictionaryTable& table = generated_table();
    const DetectorConfig config;
    const int border = config.marker_border_bits_;

    // Cover both sides of the 0.49 threshold and its 0.51 upper counterpart, as
    // well as the boundaries themselves.
    const std::vector<float> ratio_values = {0.0F,  0.25F, 0.48F, 0.49F, 0.50F,
                                             0.51F, 0.52F, 0.75F, 1.0F};
    std::mt19937_64 rng(20260828U);
    std::uniform_int_distribution<std::size_t> pick(0U, ratio_values.size() - 1U);
    std::uniform_int_distribution<int> cell_pick(
            0, (dictionary.markerSize * dictionary.markerSize) - 1);

    std::vector<RatioGrid> grids;
    const std::vector<int> ids_under_test = {0, 1, 42, 128, table.code_count_ - 1};
    grids.reserve(ids_under_test.size() * 200U);
    for (const int id : ids_under_test) {
        for (int trial = 0; trial < 200; ++trial) {
            const cv::Mat bits = cv::aruco::Dictionary::getBitsFromByteList(
                    dictionary.bytesList.rowRange(id, id + 1), dictionary.markerSize, 0);
            RatioGrid grid = grid_from_bits(bits, border);
            for (int k = 0; k < trial % 6; ++k) {
                const int cell = cell_pick(rng);
                const int row = (cell / dictionary.markerSize) + border;
                const int col = (cell % dictionary.markerSize) + border;
                grid.values_[(static_cast<std::size_t>(row) *
                              static_cast<std::size_t>(grid.cells_)) +
                             static_cast<std::size_t>(col)] = ratio_values[pick(rng)];
            }
            grids.push_back(grid);
        }
    }
    const std::vector<std::uint8_t> accepted(grids.size(), 1U);

    MatchRun run;
    ASSERT_TRUE(run.run(grids, accepted, table, config));

    std::size_t mismatch = 0;
    std::size_t ambiguous = 0;
    std::size_t rejected = 0;
    for (std::size_t i = 0; i < grids.size(); ++i) {
        const std::vector<float> flat = grids[i].inner(border, table.marker_size_);
        aruco3cuda::CellMasks masks;
        ASSERT_EQ(aruco3cuda::build_cell_masks(flat.data(), table.marker_size_,
                                               static_cast<float>(config.valid_bit_threshold_),
                                               &masks),
                  Status::kOk);
        if ((masks.not_black_ & masks.not_white_) != 0U) {
            ++ambiguous;
        }
        aruco3cuda::DictionaryMatch expected;
        ASSERT_EQ(
                aruco3cuda::identify_marker(table, masks, config.error_correction_rate_, &expected),
                Status::kOk);
        if (expected.id_ < 0) {
            ++rejected;
        }
        if (run.ids()[i] != expected.id_ || run.rotations()[i] != expected.rotation_ ||
            run.distances()[i] != expected.distance_) {
            ++mismatch;
        }
    }
    std::printf(
            "[match] %zu cases: mismatches %zu, containing an ambiguous cell %zu, "
            "rejected %zu\n",
            grids.size(), mismatch, ambiguous, rejected);
    EXPECT_EQ(mismatch, 0U);
    // Confirm that the ambiguous case and the rejection case were actually
    // exercised.
    EXPECT_GT(ambiguous, 0U);
    EXPECT_GT(rejected, 0U);
}

// Boundary: acceptance flips around the error-correction limit.
TEST(DictionaryMatchTest, correction_limit_boundary) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    const cv::aruco::Dictionary dictionary = opencv_dictionary();
    const aruco3cuda::DictionaryTable& table = generated_table();
    const DetectorConfig config;
    const int border = config.marker_border_bits_;
    // By default up to int(5 * 0.6) = 3 errors are tolerated.
    const int allowed = static_cast<int>(static_cast<double>(table.max_correction_bits_) *
                                         config.error_correction_rate_);
    ASSERT_EQ(allowed, 3);

    std::vector<RatioGrid> grids;
    for (int flips = 0; flips <= allowed + 2; ++flips) {
        const cv::Mat bits = cv::aruco::Dictionary::getBitsFromByteList(
                dictionary.bytesList.rowRange(42, 43), dictionary.markerSize, 0);
        RatioGrid grid = grid_from_bits(bits, border);
        for (int flip = 0; flip < flips; ++flip) {
            const int row = (flip / dictionary.markerSize) + border;
            const int col = (flip % dictionary.markerSize) + border;
            const std::size_t index =
                    (static_cast<std::size_t>(row) * static_cast<std::size_t>(grid.cells_)) +
                    static_cast<std::size_t>(col);
            grid.values_[index] = grid.values_[index] > 0.5F ? 0.0F : 1.0F;
        }
        grids.push_back(grid);
    }
    const std::vector<std::uint8_t> accepted(grids.size(), 1U);

    MatchRun run;
    ASSERT_TRUE(run.run(grids, accepted, table, config));
    for (std::size_t i = 0; i < grids.size(); ++i) {
        const auto flips = static_cast<int>(i);
        if (flips <= allowed) {
            EXPECT_EQ(run.ids()[i], 42) << "flips " << flips;
            EXPECT_EQ(run.distances()[i], flips) << "flips " << flips;
        } else {
            EXPECT_EQ(run.ids()[i], -1) << "flips " << flips;
        }
    }
}

// Happy path: a candidate that failed border verification is not matched.
TEST(DictionaryMatchTest, skips_candidates_rejected_by_border) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    const cv::aruco::Dictionary dictionary = opencv_dictionary();
    const aruco3cuda::DictionaryTable& table = generated_table();
    const DetectorConfig config;
    const int border = config.marker_border_bits_;

    std::vector<RatioGrid> grids;
    for (int id : {7, 7, 7}) {
        const cv::Mat bits = cv::aruco::Dictionary::getBitsFromByteList(
                dictionary.bytesList.rowRange(id, id + 1), dictionary.markerSize, 0);
        grids.push_back(grid_from_bits(bits, border));
    }
    // Pretend that only the middle candidate failed border verification.
    const std::vector<std::uint8_t> accepted = {1U, 0U, 1U};

    MatchRun run;
    ASSERT_TRUE(run.run(grids, accepted, table, config));
    EXPECT_EQ(run.ids()[0], 7);
    EXPECT_EQ(run.ids()[1], -1);
    EXPECT_EQ(run.ids()[2], 7);
}

// Failure path: invalid arguments perform no work.
TEST(DictionaryMatchTest, rejects_invalid_arguments) {
    Workspace workspace;
    const DetectorConfig config;
    const aruco3cuda::DictionaryTable& table = generated_table();

    aruco3cuda::detail::DeviceDictionary dictionary;
    EXPECT_EQ(aruco3cuda::detail::upload_dictionary(table, workspace, nullptr, nullptr),
              Status::kInvalidArgument);
    // A workspace with no reserved capacity fails to allocate.
    EXPECT_NE(aruco3cuda::detail::upload_dictionary(table, workspace, &dictionary, nullptr),
              Status::kOk);

    aruco3cuda::detail::MatchBuffers matches;
    EXPECT_EQ(aruco3cuda::detail::reserve_matches(config, workspace, nullptr),
              Status::kInvalidArgument);
    EXPECT_NE(aruco3cuda::detail::reserve_matches(config, workspace, &matches), Status::kOk);

    aruco3cuda::detail::CellRatioBuffers ratios;
    aruco3cuda::detail::DeviceCandidates candidates;
    EXPECT_EQ(aruco3cuda::detail::match_candidates_async(ratios, candidates, dictionary, config,
                                                         nullptr, nullptr),
              Status::kInvalidArgument);

    aruco3cuda::DictionaryTable empty;
    EXPECT_EQ(aruco3cuda::detail::device_dictionary_workspace_bytes(empty), 0U);
    EXPECT_GT(aruco3cuda::detail::device_dictionary_workspace_bytes(table), 0U);
    EXPECT_GT(aruco3cuda::detail::match_workspace_bytes(config), 0U);
}

}  // namespace
