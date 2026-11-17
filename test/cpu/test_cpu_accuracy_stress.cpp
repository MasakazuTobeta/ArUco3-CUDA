// SPDX-License-Identifier: Apache-2.0
#include "cpu_test_support.hpp"
#include "quality_scenes.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using aruco3cuda::CornerRefineMethod;
using aruco3cuda::DetectorConfig;
using aruco3cuda::Status;
using aruco3cuda::cpu::CpuDetector;
using aruco3cuda::cpu::CpuOptions;
using aruco3cuda::cpu::CpuResult;
using aruco3cuda::cpu::test_support::expect_same_detections;
using aruco3cuda::cpu::test_support::reference_detect;
using aruco3cuda::cpu_quality::FrameSpec;

std::array<CpuOptions, 7> accuracy_variants() {
    std::array<CpuOptions, 7> options;
    // Keep the current default as its own gate; explicitly preserve the other
    // ablations when an optimization becomes part of that default.
    for (std::size_t variant = 1U; variant < options.size(); ++variant) {
        options[variant].spatial_grouping_ = false;
        options[variant].filtered_contours_ = false;
        options[variant].sparse_pyramid_ = false;
        options[variant].cached_refinement_ = false;
        options[variant].exact_dictionary_lookup_ = false;
    }
    options[1].spatial_grouping_ = true;
    options[2].spatial_grouping_ = true;
    options[2].sparse_pyramid_ = true;
    options[2].sparse_tile_side_px_ = 3;
    options[2].sparse_max_work_ratio_ = 0.0;
    options[3].spatial_grouping_ = true;
    options[3].sparse_pyramid_ = true;
    options[3].filtered_contours_ = true;
    options[3].sparse_max_work_ratio_ = 0.0;
    options[4].spatial_grouping_ = true;
    options[4].filtered_contours_ = true;
    options[4].cached_refinement_ = true;
    options[5].exact_dictionary_lookup_ = true;
    options[6] = options[3];
    options[6].exact_dictionary_lookup_ = true;
    options[6].cached_refinement_ = true;
    return options;
}

class CpuAccuracyStressTest : public ::testing::Test {
protected:
    void SetUp() override {
        this->previous_threads_ = cv::getNumThreads();
        cv::setNumThreads(1);
    }
    void TearDown() override { cv::setNumThreads(this->previous_threads_); }

private:
    int previous_threads_ = 1;
};

TEST_F(CpuAccuracyStressTest, shuffled_changing_frames_match_opencv_without_temporal_reuse) {
    const cv::aruco::Dictionary dictionary =
            cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_MIP_36h12);
    const auto specs = aruco3cuda::cpu_quality::make_adversarial_specs(cv::Size(641, 479),
                                                                       dictionary.bytesList.rows);
    ASSERT_EQ(specs.size(), 96U);
    std::vector<cv::Mat> frames;
    frames.reserve(specs.size());
    for (const FrameSpec& spec : specs) {
        frames.push_back(aruco3cuda::cpu_quality::render_frame(spec, dictionary));
    }
    std::vector<std::size_t> order(specs.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::mt19937 random(2026111702U);
    std::shuffle(order.begin(), order.end(), random);
    // Revisit old content after unrelated images and a blank scene cut. The
    // input address is unchanged, so pointer-keyed or temporal caches fail.
    order.push_back(88U);
    order.push_back(0U);
    order.push_back(37U);
    order.push_back(88U);
    order.push_back(0U);
    const std::size_t pitch = 641U + 37U;
    std::vector<std::uint8_t> storage(pitch * 479U);
    cv::Mat slot(479, 641, CV_8UC1, storage.data(), pitch);
    ASSERT_FALSE(slot.isSubmatrix());
    ASSERT_FALSE(slot.isContinuous());
    const std::uint8_t* input_pointer = slot.data;
    cv::Mat roi_owner(491, 683, CV_8UC1);
    cv::Mat roi = roi_owner(cv::Rect(7, 3, 641, 479));
    const auto variants = accuracy_variants();
    for (bool aruco3 : {false, true}) {
        DetectorConfig config;
        config.use_aruco3_detection_ = aruco3;
        config.corner_refine_method_ =
                aruco3 ? CornerRefineMethod::kSubpix : CornerRefineMethod::kNone;
        for (int threads : {1, 4}) {
            cv::setNumThreads(threads);
            std::array<CpuDetector, 7> detectors;
            std::string message;
            for (std::size_t variant = 0; variant < detectors.size(); ++variant) {
                ASSERT_EQ(detectors[variant].initialize("DICT_ARUCO_MIP_36h12", config, &message,
                                                        variants[variant]),
                          Status::kOk)
                        << message;
            }
            std::size_t positive_frames = 0U;
            std::size_t clean_verified = 0U;
            for (std::size_t sequence = 0; sequence < order.size(); ++sequence) {
                const std::size_t index = order[sequence];
                SCOPED_TRACE("aruco3=" + std::to_string(aruco3) + " threads=" +
                             std::to_string(threads) + " frame=" + specs[index].name_);
                frames[index].copyTo(slot);
                ASSERT_EQ(slot.data, input_pointer);
                roi_owner.setTo(static_cast<int>((sequence * 53U) % 256U));
                slot.copyTo(roi);
                // Most frames exercise the full-image sparse route. Periodic
                // ROI frames verify its explicit fallback with the same content.
                const cv::Mat& input = sequence % 7U == 0U ? roi : slot;
                const CpuResult expected =
                        reference_detect(input, config, cv::aruco::DICT_ARUCO_MIP_36h12);
                positive_frames += expected.ids_.empty() ? 0U : 1U;
                if (specs[index].require_detection_) {
                    ASSERT_EQ(expected.ids_.size(), 1U);
                    ASSERT_EQ(expected.ids_[0], specs[index].markers_[0].id_);
                    ++clean_verified;
                }
                for (std::size_t variant = 0; variant < detectors.size(); ++variant) {
                    SCOPED_TRACE("variant=" + std::to_string(variant));
                    CpuResult actual;
                    ASSERT_EQ(detectors[variant].detect(input, &actual, &message), Status::kOk)
                            << message;
                    expect_same_detections(actual, expected);
                }
                EXPECT_EQ(cv::countNonZero(input != frames[index]), 0);
            }
            EXPECT_GE(positive_frames, 40U);
            EXPECT_GE(clean_verified, 12U);
            RecordProperty("positive_frames_aruco3_" + std::to_string(aruco3) + "_threads_" +
                                   std::to_string(threads),
                           static_cast<int>(positive_frames));
        }
    }
}

TEST_F(CpuAccuracyStressTest, decode_cell_sizes_and_margins_match_across_dictionary_families) {
    const std::vector<std::pair<std::string, cv::aruco::PredefinedDictionaryType>> dictionaries = {
            {"DICT_4X4_50", cv::aruco::DICT_4X4_50},
            {"DICT_5X5_1000", cv::aruco::DICT_5X5_1000},
            {"DICT_ARUCO_MIP_36h12", cv::aruco::DICT_ARUCO_MIP_36h12},
            {"DICT_7X7_1000", cv::aruco::DICT_7X7_1000}};
    for (const auto& entry : dictionaries) {
        const cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(entry.second);
        const auto specs = aruco3cuda::cpu_quality::make_adversarial_specs(
                cv::Size(641, 479), dictionary.bytesList.rows);
        std::vector<cv::Mat> images;
        for (std::size_t index : {std::size_t{0}, std::size_t{15}, std::size_t{35}}) {
            images.push_back(aruco3cuda::cpu_quality::render_frame(specs[index], dictionary));
        }
        for (int cell_size : {1, 2, 3, 4, 5, 8}) {
            for (double margin : {0.0, 0.13, 0.25, 0.49}) {
                SCOPED_TRACE(entry.first + " cell_size=" + std::to_string(cell_size) +
                             " margin=" + std::to_string(margin));
                DetectorConfig config;
                config.use_aruco3_detection_ = false;
                config.corner_refine_method_ = CornerRefineMethod::kNone;
                config.perspective_remove_pixel_per_cell_ = cell_size;
                config.perspective_remove_ignored_margin_per_cell_ = margin;
                CpuDetector detector;
                std::string message;
                ASSERT_EQ(detector.initialize(entry.first, config, &message), Status::kOk)
                        << message;
                for (const cv::Mat& image : images) {
                    CpuResult actual;
                    ASSERT_EQ(detector.detect(image, &actual, &message), Status::kOk) << message;
                    expect_same_detections(actual, reference_detect(image, config, entry.second));
                }
            }
        }
    }
}

TEST_F(CpuAccuracyStressTest, odd_large_frames_and_refinement_limits_match_opencv) {
    const cv::aruco::Dictionary dictionary =
            cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_MIP_36h12);
    const auto variants = accuracy_variants();
    for (cv::Size size :
         {cv::Size(1281, 721), cv::Size(1921, 1081), cv::Size(4097, 2049), cv::Size(7681, 1081)}) {
        const auto specs =
                aruco3cuda::cpu_quality::make_adversarial_specs(size, dictionary.bytesList.rows);
        for (std::size_t index : {std::size_t{0}, std::size_t{5}, std::size_t{18}}) {
            const cv::Mat image = aruco3cuda::cpu_quality::render_frame(specs[index], dictionary);
            for (int iterations : {1, 2, 30}) {
                for (double epsilon : {0.001, 0.1}) {
                    SCOPED_TRACE("size=" + std::to_string(size.width) + "x" +
                                 std::to_string(size.height) + " frame=" + specs[index].name_ +
                                 " iterations=" + std::to_string(iterations) +
                                 " epsilon=" + std::to_string(epsilon));
                    DetectorConfig config;
                    config.corner_refinement_max_iterations_ = iterations;
                    config.corner_refinement_min_accuracy_px_ = epsilon;
                    const CpuResult expected =
                            reference_detect(image, config, cv::aruco::DICT_ARUCO_MIP_36h12);
                    for (const CpuOptions& options : variants) {
                        CpuDetector detector;
                        std::string message;
                        ASSERT_EQ(detector.initialize("DICT_ARUCO_MIP_36h12", config, &message,
                                                      options),
                                  Status::kOk)
                                << message;
                        CpuResult actual;
                        ASSERT_EQ(detector.detect(image, &actual, &message), Status::kOk)
                                << message;
                        expect_same_detections(actual, expected);
                    }
                }
            }
        }
    }
}

TEST_F(CpuAccuracyStressTest, quality_renderer_is_deterministic_and_rejects_invalid_settings) {
    const cv::aruco::Dictionary dictionary =
            cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_MIP_36h12);
    const auto specs = aruco3cuda::cpu_quality::make_adversarial_specs();
    const cv::Mat first = aruco3cuda::cpu_quality::render_frame(specs[35], dictionary);
    const cv::Mat second = aruco3cuda::cpu_quality::render_frame(specs[35], dictionary);
    EXPECT_EQ(cv::countNonZero(first != second), 0);
    EXPECT_THROW(aruco3cuda::cpu_quality::make_adversarial_specs(cv::Size(0, 0)),
                 std::invalid_argument);
    EXPECT_THROW(aruco3cuda::cpu_quality::make_adversarial_specs(cv::Size(641, 479), 0),
                 std::invalid_argument);
    FrameSpec invalid = specs[0];
    invalid.noise_sigma_levels_ = std::numeric_limits<double>::infinity();
    EXPECT_THROW(aruco3cuda::cpu_quality::render_frame(invalid, dictionary), std::invalid_argument);
    invalid = specs[0];
    invalid.markers_[0].id_ = dictionary.bytesList.rows;
    EXPECT_THROW(aruco3cuda::cpu_quality::render_frame(invalid, dictionary), std::invalid_argument);
    invalid = specs[0];
    invalid.markers_[0].corners_[0].x = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(aruco3cuda::cpu_quality::render_frame(invalid, dictionary), std::invalid_argument);
}

TEST_F(CpuAccuracyStressTest, sparse_options_and_reinitialization_preserve_current_frame_results) {
    CpuDetector detector;
    DetectorConfig config;
    std::string message;
    for (int tile : {0, 65537}) {
        CpuOptions options;
        options.sparse_tile_side_px_ = tile;
        EXPECT_EQ(detector.initialize("DICT_4X4_50", config, &message, options),
                  Status::kInvalidConfig);
    }
    for (double ratio : {-1.0, 0.0, 1.0001, std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::infinity()}) {
        CpuOptions options;
        options.sparse_full_level_ratio_ = ratio;
        EXPECT_EQ(detector.initialize("DICT_4X4_50", config, &message, options),
                  Status::kInvalidConfig);
    }
    for (double ratio : {-1.0, std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::infinity()}) {
        CpuOptions options;
        options.sparse_max_work_ratio_ = ratio;
        EXPECT_EQ(detector.initialize("DICT_4X4_50", config, &message, options),
                  Status::kInvalidConfig);
    }
    const auto variants = accuracy_variants();
    for (int threads : {1, 2, 4}) {
        cv::setNumThreads(threads);
        for (bool aruco3 : {false, true}) {
            config.use_aruco3_detection_ = aruco3;
            config.corner_refine_method_ =
                    aruco3 ? CornerRefineMethod::kSubpix : CornerRefineMethod::kNone;
            for (std::size_t variant = 0; variant < variants.size(); ++variant) {
                for (bool mip : {false, true}) {
                    const auto family =
                            mip ? cv::aruco::DICT_ARUCO_MIP_36h12 : cv::aruco::DICT_4X4_50;
                    const std::string name = mip ? "DICT_ARUCO_MIP_36h12" : "DICT_4X4_50";
                    cv::Mat image = aruco3cuda::cpu::test_support::marker_scene(
                            family, mip ? 173 : 49, static_cast<int>(variant % 4U));
                    CpuOptions options = variants[variant];
                    options.sparse_tile_side_px_ = variant % 2U == 0U ? 1 : 65536;
                    options.sparse_full_level_ratio_ = variant % 2U == 0U ? 0.0001 : 1.0;
                    SCOPED_TRACE("threads=" + std::to_string(threads) +
                                 " aruco3=" + std::to_string(aruco3) +
                                 " variant=" + std::to_string(variant) + " dict=" + name);
                    ASSERT_EQ(detector.initialize(name, config, &message, options), Status::kOk)
                            << message;
                    CpuResult actual;
                    ASSERT_EQ(detector.detect(image, &actual, &message), Status::kOk) << message;
                    const CpuResult expected = reference_detect(image, config, family);
                    ASSERT_EQ(expected.ids_.size(), 1U);
                    expect_same_detections(actual, expected);
                    image.setTo(255);
                    ASSERT_EQ(detector.detect(image, &actual, &message), Status::kOk) << message;
                    expect_same_detections(actual, reference_detect(image, config, family));
                    EXPECT_TRUE(actual.ids_.empty());
                }
            }
        }
    }
}

TEST_F(CpuAccuracyStressTest, exact_dictionary_lookup_rebuilds_first_id_rules_for_current_config) {
    const std::vector<std::pair<std::string, cv::aruco::PredefinedDictionaryType>> dictionaries = {
            {"DICT_4X4_50", cv::aruco::DICT_4X4_50},
            {"DICT_5X5_1000", cv::aruco::DICT_5X5_1000},
            {"DICT_ARUCO_ORIGINAL", cv::aruco::DICT_ARUCO_ORIGINAL},
            {"DICT_ARUCO_MIP_36h12", cv::aruco::DICT_ARUCO_MIP_36h12},
            {"DICT_7X7_1000", cv::aruco::DICT_7X7_1000}};
    CpuDetector detector;
    std::string message;
    for (const auto& entry : dictionaries) {
        const cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(entry.second);
        std::vector<cv::Mat> images;
        for (int rotation = 0; rotation < 4; ++rotation) {
            cv::Mat marker;
            cv::aruco::generateImageMarker(dictionary, dictionary.bytesList.rows - 1,
                                           (dictionary.markerSize + 2) * 20, marker);
            for (int turn = 0; turn < rotation; ++turn) {
                cv::rotate(marker, marker, cv::ROTATE_90_CLOCKWISE);
            }
            cv::Mat image(360, 360, CV_8UC1, cv::Scalar(255));
            marker.copyTo(image(cv::Rect(80, 80, marker.cols, marker.rows)));
            images.push_back(image);
            cv::Mat flipped = image.clone();
            cv::Mat first_cell = flipped(cv::Rect(100, 100, 20, 20));
            cv::bitwise_not(first_cell, first_cell);
            images.push_back(std::move(flipped));
            for (bool all_cells : {false, true}) {
                cv::Mat half = image.clone();
                const int cells = all_cells ? dictionary.markerSize * dictionary.markerSize : 1;
                for (int cell = 0; cell < cells; ++cell) {
                    const int x = 100 + (cell % dictionary.markerSize) * 20;
                    const int y = 100 + (cell / dictionary.markerSize) * 20;
                    half(cv::Rect(x, y, 20, 20)).setTo(0);
                    half(cv::Rect(x, y, 10, 20)).setTo(255);
                }
                images.push_back(std::move(half));
            }
        }
        for (double correction_rate : {1.0, 0.0, 0.6}) {
            for (double bit_threshold : {1.0, 0.49, 0.0, 0.6, 0.5}) {
                DetectorConfig config;
                config.use_aruco3_detection_ = false;
                config.corner_refine_method_ = CornerRefineMethod::kNone;
                config.perspective_remove_pixel_per_cell_ = 20;
                config.perspective_remove_ignored_margin_per_cell_ = 0.0;
                config.error_correction_rate_ = correction_rate;
                config.valid_bit_threshold_ = bit_threshold;
                std::vector<CpuResult> expected;
                for (const cv::Mat& image : images) {
                    expected.push_back(reference_detect(image, config, entry.second));
                }
                ASSERT_FALSE(expected.front().ids_.empty());
                for (bool lookup : {false, true}) {
                    CpuOptions options;
                    options.packed_dictionary_ = false;
                    options.exact_dictionary_lookup_ = lookup;
                    SCOPED_TRACE(entry.first +
                                 " correction_rate=" + std::to_string(correction_rate) +
                                 " bit_threshold=" + std::to_string(bit_threshold) +
                                 " lookup=" + std::to_string(lookup));
                    ASSERT_EQ(detector.initialize(entry.first, config, &message, options),
                              Status::kOk)
                            << message;
                    for (std::size_t frame = 0; frame < images.size(); ++frame) {
                        SCOPED_TRACE(frame);
                        CpuResult actual;
                        ASSERT_EQ(detector.detect(images[frame], &actual, &message), Status::kOk)
                                << message;
                        expect_same_detections(actual, expected[frame]);
                    }
                }
            }
        }
    }
}

}  // namespace
