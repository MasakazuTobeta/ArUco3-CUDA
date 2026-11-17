// SPDX-License-Identifier: Apache-2.0
#include "cpu_component_candidates.hpp"
#include "cpu_detector.hpp"
#include "cpu_test_support.hpp"

#include <gtest/gtest.h>
#include <unistd.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "corpus_generator.hpp"

namespace {

using aruco3cuda::CornerRefineMethod;
using aruco3cuda::DetectorConfig;
using aruco3cuda::Status;
using aruco3cuda::cpu::CpuDetector;
using aruco3cuda::cpu::CpuOptions;
using aruco3cuda::cpu::CpuResult;

using aruco3cuda::cpu::test_support::expect_same_detections;
using aruco3cuda::cpu::test_support::marker_scene;
using aruco3cuda::cpu::test_support::reference_detect;

CpuOptions options_for_mask(unsigned int mask) {
    CpuOptions options;
    options.filtered_contours_ = false;
    options.spatial_grouping_ = false;
    options.sparse_pyramid_ = false;
    options.cached_refinement_ = false;
    options.exact_dictionary_lookup_ = false;
    options.reuse_pyramid_ = (mask & 1U) != 0U;
    options.packed_dictionary_ = (mask & 2U) != 0U;
    options.fused_cells_ = (mask & 4U) != 0U;
    options.multiwindow_threshold_ = (mask & 8U) != 0U;
    options.simple_contours_ = (mask & 16U) != 0U;
    options.component_prefilter_ = (mask & 32U) != 0U;
    options.dryrun_otsu_ = (mask & 64U) != 0U;
    options.parallel_windows_ = (mask & 128U) != 0U;
    options.parallel_decode_ = (mask & 256U) != 0U;
    return options;
}

class CpuDetectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        this->previous_thread_count_ = cv::getNumThreads();
        cv::setNumThreads(1);
    }
    void TearDown() override { cv::setNumThreads(this->previous_thread_count_); }

private:
    int previous_thread_count_ = 1;
};

TEST_F(CpuDetectorTest, rejects_uninitialized_calls_and_invalid_images) {
    CpuDetector detector;
    CpuResult result;
    result.ids_.push_back(123);
    std::string message;
    const cv::Mat image(320, 320, CV_8UC1, cv::Scalar(255));
    EXPECT_EQ(detector.detect(image, &result, &message), Status::kNotInitialized);
    EXPECT_FALSE(message.empty());
    EXPECT_EQ(result.ids_, std::vector<int>({123}));
    ASSERT_EQ(detector.initialize("DICT_ARUCO_MIP_36h12", DetectorConfig{}, &message), Status::kOk)
            << message;
    EXPECT_EQ(detector.detect(image, nullptr, &message), Status::kInvalidArgument);
    EXPECT_EQ(detector.detect(cv::Mat{}, &result, &message), Status::kInvalidImage);
    EXPECT_EQ(detector.detect(cv::Mat(8, 8, CV_8UC3), &result, &message), Status::kInvalidImage);
    EXPECT_EQ(detector.detect(cv::Mat(8, 8, CV_16UC1), &result, &message), Status::kInvalidImage);
    const int dimensions[] = {3, 4, 5};
    EXPECT_EQ(detector.detect(cv::Mat(3, dimensions, CV_8UC1), &result, &message),
              Status::kInvalidImage);
    cv::Mat invalid_stride = image.clone();
    invalid_stride.step[0] = 1U;
    EXPECT_EQ(detector.detect(invalid_stride, &result, &message), Status::kInvalidImage);
    EXPECT_EQ(detector.detect(cv::Mat(1, 65537, CV_8UC1), &result, &message),
              Status::kInvalidImage);
    cv::Mat oversized(1, 1, CV_8UC1);
    // Deliberately forge only the header. Validation must reject the product
    // before any OpenCV operation can read beyond the one allocated pixel.
    oversized.rows = 65536;
    oversized.cols = 65536;
    oversized.step[0] = 65536U;
    EXPECT_EQ(detector.detect(oversized, &result, &message), Status::kInvalidImage);
    EXPECT_EQ(result.ids_, std::vector<int>({123}));
}

TEST_F(CpuDetectorTest, rejects_invalid_configuration_and_dictionary) {
    CpuDetector detector;
    std::string message;
    EXPECT_EQ(detector.initialize("DICT_NOPE", DetectorConfig{}, &message),
              Status::kUnsupportedDictionary);
    EXPECT_FALSE(message.empty());
    std::vector<DetectorConfig> invalid(6);
    invalid[0].adaptive_thresh_win_size_min_px_ = 2;
    invalid[1].adaptive_thresh_win_size_step_px_ = 0;
    invalid[2].min_marker_length_ratio_original_img_ = std::numeric_limits<float>::quiet_NaN();
    invalid[3].perspective_remove_ignored_margin_per_cell_ = 0.5;
    invalid[4].corner_refinement_max_iterations_ = 0;
    invalid[5].min_side_length_canonical_img_px_ = 0;
    invalid[5].min_marker_length_ratio_original_img_ = 0.0F;
    for (std::size_t i = 0; i < invalid.size(); ++i) {
        SCOPED_TRACE(i);
        EXPECT_EQ(detector.initialize("DICT_ARUCO_MIP_36h12", invalid[i], &message),
                  Status::kInvalidConfig);
        EXPECT_FALSE(message.empty());
    }
}

TEST_F(CpuDetectorTest, all_optimization_combinations_preserve_ids_rotation_and_corners) {
    const cv::Mat image = marker_scene(cv::aruco::DICT_ARUCO_MIP_36h12, 140, 1);
    for (bool aruco3 : {false, true}) {
        DetectorConfig config;
        config.use_aruco3_detection_ = aruco3;
        config.corner_refine_method_ =
                aruco3 ? CornerRefineMethod::kSubpix : CornerRefineMethod::kNone;
        const CpuResult expected = reference_detect(image, config, cv::aruco::DICT_ARUCO_MIP_36h12);
        ASSERT_EQ(expected.ids_.size(), 1U);
        for (unsigned int mask = 0U; mask < 512U; ++mask) {
            SCOPED_TRACE("aruco3=" + std::to_string(aruco3) + " mask=" + std::to_string(mask));
            CpuDetector detector;
            std::string message;
            ASSERT_EQ(detector.initialize("DICT_ARUCO_MIP_36h12", config, &message,
                                          options_for_mask(mask)),
                      Status::kOk)
                    << message;
            CpuResult actual;
            ASSERT_EQ(detector.detect(image, &actual, &message), Status::kOk) << message;
            expect_same_detections(actual, expected);
        }
    }
}

TEST_F(CpuDetectorTest, full_corpus_matches_opencv_with_and_without_aruco3) {
    // One writer per file: this test owns a process-local directory and removes
    // it even after an assertion. No other CTest case consumes these files.
    const std::filesystem::path directory = std::filesystem::temp_directory_path() /
                                            ("aruco3cuda_cpu_full_" + std::to_string(::getpid()));
    struct Cleanup {
        std::filesystem::path directory_;
        ~Cleanup() {
            std::error_code ignored;
            std::filesystem::remove_all(this->directory_, ignored);
        }
    } cleanup{directory};
    aruco3cuda::corpusgen::CorpusConfig corpus;
    corpus.output_dir_ = directory.string();
    std::vector<aruco3cuda::corpusgen::SceneSpec> specs;
    ASSERT_TRUE(aruco3cuda::corpusgen::build_preset("full", &specs));
    ASSERT_EQ(specs.size(), 91U);
    std::array<CpuDetector, 8> detectors;
    std::array<DetectorConfig, 2> configs;
    configs[0].use_aruco3_detection_ = false;
    configs[0].corner_refine_method_ = CornerRefineMethod::kNone;
    std::string message;
    for (std::size_t mode = 0; mode < 2U; ++mode) {
        for (std::size_t variant = 0; variant < 4U; ++variant) {
            const std::array<unsigned int, 3> masks{0U, 399U, 511U};
            const CpuOptions options =
                    variant < masks.size() ? options_for_mask(masks[variant]) : CpuOptions{};
            ASSERT_EQ(detectors[mode * 4U + variant].initialize(corpus.dictionary_name_,
                                                                configs[mode], &message, options),
                      Status::kOk)
                    << message;
        }
    }
    for (std::size_t index = 0; index < specs.size(); ++index) {
        SCOPED_TRACE(specs[index].name_);
        aruco3cuda::corpusgen::GeneratedScene scene;
        ASSERT_TRUE(aruco3cuda::corpusgen::generate_scene(corpus, specs[index], index, &scene,
                                                          &message))
                << message;
        const cv::Mat image = cv::imread(scene.path_, cv::IMREAD_GRAYSCALE);
        ASSERT_FALSE(image.empty());
        for (std::size_t mode = 0; mode < 2U; ++mode) {
            SCOPED_TRACE("aruco3=" + std::to_string(mode));
            for (int threads : {1, 2, 4}) {
                SCOPED_TRACE("threads=" + std::to_string(threads));
                cv::setNumThreads(threads);
                const CpuResult expected =
                        reference_detect(image, configs[mode], cv::aruco::DICT_ARUCO_MIP_36h12);
                for (std::size_t variant = 0; variant < 4U; ++variant) {
                    SCOPED_TRACE("optimized=" + std::to_string(variant));
                    CpuResult actual;
                    ASSERT_EQ(detectors[mode * 4U + variant].detect(image, &actual, &message),
                              Status::kOk)
                            << message;
                    expect_same_detections(actual, expected);
                }
            }
        }
    }
}

TEST_F(CpuDetectorTest, predefined_dictionaries_and_all_four_rotations_match) {
    const std::vector<std::pair<std::string, cv::aruco::PredefinedDictionaryType>> dictionaries = {
            {"DICT_4X4_50", cv::aruco::DICT_4X4_50},
            {"DICT_4X4_100", cv::aruco::DICT_4X4_100},
            {"DICT_4X4_250", cv::aruco::DICT_4X4_250},
            {"DICT_4X4_1000", cv::aruco::DICT_4X4_1000},
            {"DICT_5X5_50", cv::aruco::DICT_5X5_50},
            {"DICT_5X5_100", cv::aruco::DICT_5X5_100},
            {"DICT_5X5_250", cv::aruco::DICT_5X5_250},
            {"DICT_5X5_1000", cv::aruco::DICT_5X5_1000},
            {"DICT_6X6_50", cv::aruco::DICT_6X6_50},
            {"DICT_6X6_100", cv::aruco::DICT_6X6_100},
            {"DICT_6X6_250", cv::aruco::DICT_6X6_250},
            {"DICT_6X6_1000", cv::aruco::DICT_6X6_1000},
            {"DICT_7X7_50", cv::aruco::DICT_7X7_50},
            {"DICT_7X7_100", cv::aruco::DICT_7X7_100},
            {"DICT_7X7_250", cv::aruco::DICT_7X7_250},
            {"DICT_7X7_1000", cv::aruco::DICT_7X7_1000},
            {"DICT_ARUCO_MIP_36h12", cv::aruco::DICT_ARUCO_MIP_36h12}};
    for (const auto& dictionary : dictionaries) {
        SCOPED_TRACE(dictionary.first);
        const cv::aruco::Dictionary table = cv::aruco::getPredefinedDictionary(dictionary.second);
        for (bool aruco3 : {false, true}) {
            DetectorConfig config;
            config.use_aruco3_detection_ = aruco3;
            config.corner_refine_method_ =
                    aruco3 ? CornerRefineMethod::kSubpix : CornerRefineMethod::kNone;
            CpuDetector detector;
            std::string message;
            ASSERT_EQ(detector.initialize(dictionary.first, config, &message), Status::kOk)
                    << message;
            for (int id : {0, table.bytesList.rows - 1}) {
                for (int rotation = 0; rotation < 4; ++rotation) {
                    SCOPED_TRACE("aruco3=" + std::to_string(aruco3) + " id=" + std::to_string(id) +
                                 " rotation=" + std::to_string(rotation));
                    const cv::Mat image = marker_scene(dictionary.second, id, rotation);
                    const CpuResult expected = reference_detect(image, config, dictionary.second);
                    ASSERT_EQ(expected.ids_.size(), 1U);
                    CpuResult actual;
                    ASSERT_EQ(detector.detect(image, &actual, &message), Status::kOk) << message;
                    expect_same_detections(actual, expected);
                }
            }
        }
    }
}

TEST_F(CpuDetectorTest, changing_size_stride_and_pixels_never_reuses_stale_data) {
    for (bool aruco3 : {false, true}) {
        DetectorConfig config;
        config.use_aruco3_detection_ = aruco3;
        config.corner_refine_method_ =
                aruco3 ? CornerRefineMethod::kSubpix : CornerRefineMethod::kNone;
        CpuDetector detector;
        std::string message;
        ASSERT_EQ(detector.initialize("DICT_ARUCO_MIP_36h12", config, &message), Status::kOk)
                << message;
        const cv::Mat marker = marker_scene(cv::aruco::DICT_ARUCO_MIP_36h12, 42, 3);
        for (int repetition = 0; repetition < 3; ++repetition) {
            for (const cv::Size& size : {cv::Size(320, 320), cv::Size(641, 479),
                                         cv::Size(1280, 720), cv::Size(320, 320)}) {
                cv::Mat owner(size.height + 9, size.width + 37, CV_8UC1, cv::Scalar(0));
                cv::Mat roi = owner(cv::Rect(7, 3, size.width, size.height));
                roi.setTo(255);
                marker.copyTo(roi(cv::Rect(0, 0, marker.cols, marker.rows)));
                ASSERT_FALSE(roi.isContinuous());
                CpuResult actual;
                ASSERT_EQ(detector.detect(roi, &actual, &message), Status::kOk) << message;
                expect_same_detections(
                        actual, reference_detect(roi, config, cv::aruco::DICT_ARUCO_MIP_36h12));
                // Keep the pointer and shape unchanged but replace the pixels.
                roi.setTo(255);
                ASSERT_EQ(detector.detect(roi, &actual, &message), Status::kOk) << message;
                expect_same_detections(
                        actual, reference_detect(roi, config, cv::aruco::DICT_ARUCO_MIP_36h12));
                EXPECT_TRUE(actual.ids_.empty());
            }
        }
    }
}

TEST_F(CpuDetectorTest, alternate_detection_settings_match_and_profile_is_optional) {
    const cv::Mat image = marker_scene(cv::aruco::DICT_ARUCO_MIP_36h12, 24, 2, 2);
    DetectorConfig config;
    config.marker_border_bits_ = 2;
    config.perspective_remove_pixel_per_cell_ = 6;
    config.perspective_remove_ignored_margin_per_cell_ = 0.2;
    config.adaptive_thresh_win_size_min_px_ = 4;
    config.adaptive_thresh_win_size_max_px_ = 24;
    config.adaptive_thresh_constant_ = 3.5;
    config.min_marker_length_ratio_original_img_ = 0.02F;
    config.valid_bit_threshold_ = 0.4;
    const CpuResult expected = reference_detect(image, config, cv::aruco::DICT_ARUCO_MIP_36h12);
    ASSERT_EQ(expected.ids_.size(), 1U);
    for (bool profile : {false, true}) {
        CpuOptions options;
        options.profile_stages_ = profile;
        CpuDetector detector;
        std::string message;
        ASSERT_EQ(detector.initialize("DICT_ARUCO_MIP_36h12", config, &message, options),
                  Status::kOk)
                << message;
        CpuResult actual;
        ASSERT_EQ(detector.detect(image, &actual, &message), Status::kOk) << message;
        expect_same_detections(actual, expected);
        const auto& times = actual.stage_times_;
        const double total = times.preprocess_ms_ + times.threshold_ms_ + times.candidates_ms_ +
                             times.decode_ms_ + times.refine_ms_;
        EXPECT_GE(total, 0.0);
        if (profile) {
            EXPECT_GT(total, 0.0);
        } else {
            EXPECT_DOUBLE_EQ(total, 0.0);
        }
    }
}

TEST_F(CpuDetectorTest, separate_markers_with_the_same_id_are_preserved) {
    const cv::Mat marker = marker_scene(cv::aruco::DICT_ARUCO_MIP_36h12, 42, 0);
    cv::Mat image(360, 680, CV_8UC1, cv::Scalar(255));
    marker.copyTo(image(cv::Rect(0, 0, marker.cols, marker.rows)));
    marker.copyTo(image(cv::Rect(350, 0, marker.cols, marker.rows)));
    DetectorConfig config;
    const CpuResult expected = reference_detect(image, config, cv::aruco::DICT_ARUCO_MIP_36h12);
    ASSERT_EQ(expected.ids_.size(), 2U);
    CpuDetector detector;
    std::string message;
    ASSERT_EQ(detector.initialize("DICT_ARUCO_MIP_36h12", config, &message), Status::kOk)
            << message;
    CpuResult actual;
    ASSERT_EQ(detector.detect(image, &actual, &message), Status::kOk) << message;
    expect_same_detections(actual, expected);
}

TEST_F(CpuDetectorTest, compact_contours_preserve_candidates_for_degenerate_and_noisy_shapes) {
    std::vector<cv::Mat> images;
    images.push_back(cv::Mat(319, 321, CV_8UC1, cv::Scalar(255)));
    cv::line(images.back(), cv::Point(17, 17), cv::Point(290, 290), cv::Scalar(0), 1);
    cv::line(images.back(), cv::Point(13, 200), cv::Point(300, 200), cv::Scalar(0), 1);
    cv::line(images.back(), cv::Point(155, 9), cv::Point(155, 302), cv::Scalar(0), 1);
    images.push_back(cv::Mat(319, 321, CV_8UC1, cv::Scalar(255)));
    for (int inset : {13, 47, 83, 113, 141}) {
        cv::rectangle(images.back(), cv::Point(inset, inset), cv::Point(320 - inset, 318 - inset),
                      cv::Scalar(0), 2);
    }
    images.push_back(cv::Mat(319, 321, CV_8UC1, cv::Scalar(255)));
    cv::rectangle(images.back(), cv::Rect(9, 9, 90, 90), cv::Scalar(0), cv::FILLED);
    cv::rectangle(images.back(), cv::Rect(99, 99, 90, 90), cv::Scalar(0), cv::FILLED);
    const std::vector<cv::Point> concave = {cv::Point(210, 15),  cv::Point(300, 15),
                                            cv::Point(300, 100), cv::Point(260, 100),
                                            cv::Point(260, 60),  cv::Point(210, 60)};
    cv::fillPoly(images.back(), std::vector<std::vector<cv::Point>>{concave}, cv::Scalar(0));
    cv::RNG random(20261117U);
    for (int i = 0; i < 12; ++i) {
        cv::Mat image(319, 321, CV_8UC1);
        random.fill(image, cv::RNG::UNIFORM, 0, 256);
        if ((i & 1) != 0) {
            cv::threshold(image, image, 127, 255, cv::THRESH_BINARY);
        }
        images.push_back(image);
    }
    for (bool aruco3 : {false, true}) {
        DetectorConfig config;
        config.use_aruco3_detection_ = aruco3;
        config.corner_refine_method_ =
                aruco3 ? CornerRefineMethod::kSubpix : CornerRefineMethod::kNone;
        // A small perimeter limit keeps isolated strokes in the candidate
        // extraction workload, instead of testing only early size rejection.
        config.min_side_length_canonical_img_px_ = 8;
        CpuDetector control;
        CpuDetector optimized;
        std::string message;
        ASSERT_EQ(
                control.initialize("DICT_ARUCO_MIP_36h12", config, &message, options_for_mask(0U)),
                Status::kOk)
                << message;
        ASSERT_EQ(optimized.initialize("DICT_ARUCO_MIP_36h12", config, &message,
                                       options_for_mask(127U)),
                  Status::kOk)
                << message;
        for (std::size_t index = 0; index < images.size(); ++index) {
            SCOPED_TRACE("aruco3=" + std::to_string(aruco3) + " image=" + std::to_string(index));
            CpuResult baseline;
            CpuResult actual;
            ASSERT_EQ(control.detect(images[index], &baseline, &message), Status::kOk) << message;
            ASSERT_EQ(optimized.detect(images[index], &actual, &message), Status::kOk) << message;
            EXPECT_EQ(actual.candidate_count_, baseline.candidate_count_);
            expect_same_detections(actual, baseline);
            expect_same_detections(actual, reference_detect(images[index], config,
                                                            cv::aruco::DICT_ARUCO_MIP_36h12));
        }
    }
}

TEST_F(CpuDetectorTest, corner_refinement_options_match_with_and_without_aruco3) {
    const cv::Mat image = marker_scene(cv::aruco::DICT_ARUCO_MIP_36h12, 121, 3);
    cv::Mat owner(333, 357, CV_8UC1);
    cv::RNG random(20261117U);
    random.fill(owner, cv::RNG::UNIFORM, 0, 256);
    cv::Mat roi = owner(cv::Rect(7, 5, image.cols, image.rows));
    roi.setTo(255);
    // A near-border marker makes the refinement patch cross the visible ROI
    // boundary. Parent pixels must not alter getRectSubPix's border behavior.
    image(cv::Rect(80, 80, 160, 160)).copyTo(roi(cv::Rect(3, 3, 160, 160)));
    for (bool aruco3 : {false, true}) {
        for (CornerRefineMethod refinement :
             {CornerRefineMethod::kNone, CornerRefineMethod::kSubpix}) {
            DetectorConfig config;
            config.use_aruco3_detection_ = aruco3;
            config.corner_refine_method_ = refinement;
            for (const CpuOptions& options : {options_for_mask(127U), CpuOptions{}}) {
                CpuDetector detector;
                std::string message;
                ASSERT_EQ(detector.initialize("DICT_ARUCO_MIP_36h12", config, &message, options),
                          Status::kOk)
                        << message;
                for (const cv::Mat& input : {image, roi}) {
                    SCOPED_TRACE("aruco3=" + std::to_string(aruco3) +
                                 " refinement=" + std::to_string(static_cast<int>(refinement)) +
                                 " cached=" + std::to_string(options.cached_refinement_) +
                                 " roi=" + std::to_string(input.isSubmatrix()));
                    CpuResult actual;
                    ASSERT_EQ(detector.detect(input, &actual, &message), Status::kOk) << message;
                    const CpuResult expected =
                            reference_detect(input, config, cv::aruco::DICT_ARUCO_MIP_36h12);
                    if (!aruco3) {
                        ASSERT_EQ(expected.ids_, std::vector<int>({121}));
                    }
                    expect_same_detections(actual, expected);
                }
            }
        }
    }
}

TEST_F(CpuDetectorTest, opencv_thread_counts_preserve_compatibility) {
    cv::Mat image(640, 960, CV_8UC1, cv::Scalar(255));
    for (int row = 0; row < 2; ++row) {
        for (int column = 0; column < 3; ++column) {
            const cv::Mat tile =
                    marker_scene(cv::aruco::DICT_ARUCO_MIP_36h12, row * 3 + column, column);
            tile.copyTo(image(cv::Rect(column * 320, row * 320, 320, 320)));
        }
    }
    for (int threads : {1, 2, 4}) {
        SCOPED_TRACE(threads);
        cv::setNumThreads(threads);
        for (bool aruco3 : {false, true}) {
            DetectorConfig config;
            config.use_aruco3_detection_ = aruco3;
            config.corner_refine_method_ =
                    aruco3 ? CornerRefineMethod::kSubpix : CornerRefineMethod::kNone;
            CpuDetector detector;
            std::string message;
            ASSERT_EQ(detector.initialize("DICT_ARUCO_MIP_36h12", config, &message,
                                          options_for_mask(127U)),
                      Status::kOk)
                    << message;
            CpuResult actual;
            ASSERT_EQ(detector.detect(image, &actual, &message), Status::kOk) << message;
            const CpuResult expected =
                    reference_detect(image, config, cv::aruco::DICT_ARUCO_MIP_36h12);
            ASSERT_EQ(expected.ids_.size(), 6U);
            expect_same_detections(actual, expected);
        }
    }
}

TEST_F(CpuDetectorTest, reinitialization_replaces_dictionary_and_invalidates_failed_setup) {
    CpuDetector detector;
    const std::vector<std::pair<std::string, cv::aruco::PredefinedDictionaryType>> dictionaries = {
            {"DICT_5X5_1000", cv::aruco::DICT_5X5_1000},
            {"DICT_4X4_50", cv::aruco::DICT_4X4_50},
            {"DICT_ARUCO_MIP_36h12", cv::aruco::DICT_ARUCO_MIP_36h12}};
    std::string message;
    CpuResult result;
    for (const auto& dictionary : dictionaries) {
        SCOPED_TRACE(dictionary.first);
        DetectorConfig config;
        config.perspective_remove_pixel_per_cell_ = dictionary.first == "DICT_4X4_50" ? 3 : 5;
        const cv::Mat image = marker_scene(dictionary.second, 24, 1);
        ASSERT_EQ(detector.initialize(dictionary.first, config, &message, options_for_mask(127U)),
                  Status::kOk)
                << message;
        ASSERT_EQ(detector.detect(image, &result, &message), Status::kOk) << message;
        expect_same_detections(result, reference_detect(image, config, dictionary.second));
    }
    const std::vector<int> previous_ids = result.ids_;
    EXPECT_EQ(detector.initialize("DICT_UNKNOWN", DetectorConfig{}, &message),
              Status::kUnsupportedDictionary);
    EXPECT_EQ(detector.detect(cv::Mat(320, 320, CV_8UC1), &result, &message),
              Status::kNotInitialized);
    EXPECT_EQ(result.ids_, previous_ids);
}

TEST_F(CpuDetectorTest, dryrun_otsu_matches_for_low_contrast_and_variance_boundaries) {
    const cv::Mat binary = marker_scene(cv::aruco::DICT_ARUCO_MIP_36h12, 140, 2);
    for (const std::pair<int, int>& levels :
         {std::pair<int, int>(0, 255), std::pair<int, int>(120, 128), std::pair<int, int>(127, 128),
          std::pair<int, int>(127, 129)}) {
        cv::Mat image(binary.size(), CV_8UC1, cv::Scalar(levels.first));
        image.setTo(levels.second, binary);
        for (double min_stddev : {0.0, 0.49, 0.5, 0.51, 1.0, 4.0, 5.0, 128.0}) {
            SCOPED_TRACE("black=" + std::to_string(levels.first) +
                         " white=" + std::to_string(levels.second) +
                         " min_stddev=" + std::to_string(min_stddev));
            DetectorConfig config;
            config.adaptive_thresh_constant_ = 0.0;
            config.min_otsu_std_dev_ = min_stddev;
            const CpuResult expected =
                    reference_detect(image, config, cv::aruco::DICT_ARUCO_MIP_36h12);
            for (unsigned int mask : {0U, 63U, 127U}) {
                CpuDetector detector;
                std::string message;
                ASSERT_EQ(detector.initialize("DICT_ARUCO_MIP_36h12", config, &message,
                                              options_for_mask(mask)),
                          Status::kOk)
                        << message;
                CpuResult actual;
                ASSERT_EQ(detector.detect(image, &actual, &message), Status::kOk) << message;
                EXPECT_GT(actual.candidate_count_, 0U);
                expect_same_detections(actual, expected);
            }
        }
    }
}

TEST_F(CpuDetectorTest, maximum_even_window_is_rounded_consistently_in_both_threshold_routes) {
    DetectorConfig config;
    config.adaptive_thresh_win_size_min_px_ = 4096;
    config.adaptive_thresh_win_size_max_px_ = 4096;
    const cv::Mat image = marker_scene(cv::aruco::DICT_ARUCO_MIP_36h12, 42, 0);
    const CpuResult expected = reference_detect(image, config, cv::aruco::DICT_ARUCO_MIP_36h12);
    for (bool multiwindow : {false, true}) {
        CpuOptions options;
        options.multiwindow_threshold_ = multiwindow;
        CpuDetector detector;
        std::string message;
        ASSERT_EQ(detector.initialize("DICT_ARUCO_MIP_36h12", config, &message, options),
                  Status::kOk)
                << message;
        CpuResult actual;
        ASSERT_EQ(detector.detect(image, &actual, &message), Status::kOk) << message;
        expect_same_detections(actual, expected);
    }
}

TEST_F(CpuDetectorTest, component_candidate_experiment_recovers_clean_marker_ground_truth) {
    // This option deliberately changes candidate generation. Its IDs and
    // oriented corners are tested against rendered geometry, rather than
    // requiring its candidate/rejection set to equal the contour detector's.
    for (bool aruco3 : {false, true}) {
        DetectorConfig config;
        config.use_aruco3_detection_ = aruco3;
        config.corner_refine_method_ =
                aruco3 ? CornerRefineMethod::kSubpix : CornerRefineMethod::kNone;
        CpuOptions options = options_for_mask(127U);
        options.component_candidates_ = true;
        CpuDetector detector;
        std::string message;
        ASSERT_EQ(detector.initialize("DICT_ARUCO_MIP_36h12", config, &message, options),
                  Status::kOk)
                << message;
        for (int rotation = 0; rotation < 4; ++rotation) {
            SCOPED_TRACE("aruco3=" + std::to_string(aruco3) +
                         " rotation=" + std::to_string(rotation));
            const cv::Mat image = marker_scene(cv::aruco::DICT_ARUCO_MIP_36h12, 42, rotation);
            CpuResult result;
            ASSERT_EQ(detector.detect(image, &result, &message), Status::kOk) << message;
            ASSERT_EQ(result.ids_, std::vector<int>({42}));
            ASSERT_EQ(result.corners_.size(), 1U);
            ASSERT_EQ(result.corners_[0].size(), 4U);
            std::array<cv::Point2f, 4> expected = {
                    cv::Point2f(80.0F, 80.0F), cv::Point2f(239.0F, 80.0F),
                    cv::Point2f(239.0F, 239.0F), cv::Point2f(80.0F, 239.0F)};
            for (cv::Point2f& point : expected) {
                for (int turn = 0; turn < rotation; ++turn) {
                    point = cv::Point2f(319.0F - point.y, point.x);
                }
            }
            for (std::size_t corner = 0; corner < 4U; ++corner) {
                EXPECT_LE(cv::norm(result.corners_[0][corner] - expected[corner]), 2.0);
            }
            // The experiment can omit inner contours. Its independent
            // rejection count is intentionally not asserted against OpenCV.
            const CpuResult baseline =
                    reference_detect(image, config, cv::aruco::DICT_ARUCO_MIP_36h12);
            EXPECT_EQ(result.ids_.size(), baseline.ids_.size());
        }
    }
}

TEST_F(CpuDetectorTest, validates_numerical_cpu_options) {
    CpuDetector detector;
    std::string message;
    CpuOptions options;
    options.threshold_parallel_min_pixels_ = -1;
    EXPECT_EQ(detector.initialize("DICT_ARUCO_MIP_36h12", DetectorConfig{}, &message, options),
              Status::kInvalidConfig);
    options.threshold_parallel_min_pixels_ = 0;
    options.min_parallel_candidates_ = 0;
    EXPECT_EQ(detector.initialize("DICT_ARUCO_MIP_36h12", DetectorConfig{}, &message, options),
              Status::kInvalidConfig);
    options.min_parallel_candidates_ = 1;
    for (double band : {-1.0, 65.0, std::numeric_limits<double>::quiet_NaN(),
                        std::numeric_limits<double>::infinity()}) {
        options.component_edge_band_px_ = band;
        EXPECT_EQ(detector.initialize("DICT_ARUCO_MIP_36h12", DetectorConfig{}, &message, options),
                  Status::kInvalidConfig);
    }
    for (double band : {0.0, 64.0}) {
        options.component_edge_band_px_ = band;
        EXPECT_EQ(detector.initialize("DICT_ARUCO_MIP_36h12", DetectorConfig{}, &message, options),
                  Status::kOk)
                << message;
    }
}

TEST_F(CpuDetectorTest, moved_detectors_keep_results_and_sources_can_be_reinitialized) {
    const cv::Mat image = marker_scene(cv::aruco::DICT_ARUCO_MIP_36h12, 42, 1);
    DetectorConfig config;
    const CpuResult expected = reference_detect(image, config, cv::aruco::DICT_ARUCO_MIP_36h12);
    CpuDetector source;
    std::string message;
    ASSERT_EQ(source.initialize("DICT_ARUCO_MIP_36h12", config, &message), Status::kOk) << message;
    CpuResult result;
    ASSERT_EQ(source.detect(image, &result, &message), Status::kOk) << message;
    CpuDetector constructed(std::move(source));
    ASSERT_EQ(constructed.detect(image, &result, &message), Status::kOk) << message;
    expect_same_detections(result, expected);
    EXPECT_EQ(source.detect(image, &result, &message), Status::kNotInitialized);
    EXPECT_EQ(source.detect(image, nullptr, &message), Status::kInvalidArgument);
    ASSERT_EQ(source.initialize("DICT_ARUCO_MIP_36h12", config, &message), Status::kOk) << message;
    ASSERT_EQ(source.detect(image, &result, &message), Status::kOk) << message;
    expect_same_detections(result, expected);
    CpuDetector assigned;
    assigned = std::move(source);
    ASSERT_EQ(assigned.detect(image, &result, &message), Status::kOk) << message;
    expect_same_detections(result, expected);
    EXPECT_EQ(source.detect(image, &result, &message), Status::kNotInitialized);
    ASSERT_EQ(source.initialize("DICT_ARUCO_MIP_36h12", config, &message), Status::kOk) << message;
    ASSERT_EQ(source.detect(image, &result, &message), Status::kOk) << message;
    expect_same_detections(result, expected);
}

TEST_F(CpuDetectorTest, component_extractor_rejects_degenerate_and_nonquadrilateral_shapes) {
    using aruco3cuda::cpu::detail::ComponentCandidates;
    ComponentCandidates extractor;
    DetectorConfig config;
    std::vector<cv::Mat> invalid;
    invalid.push_back(cv::Mat(241, 319, CV_8UC1, cv::Scalar(0)));
    invalid.push_back(cv::Mat(241, 319, CV_8UC1, cv::Scalar(255)));
    invalid.push_back(cv::Mat(241, 319, CV_8UC1, cv::Scalar(0)));
    invalid.back().at<unsigned char>(120, 159) = 255U;
    invalid.push_back(cv::Mat(241, 319, CV_8UC1, cv::Scalar(0)));
    cv::line(invalid.back(), cv::Point(19, 19), cv::Point(210, 210), cv::Scalar(255), 1);
    invalid.push_back(cv::Mat(241, 319, CV_8UC1, cv::Scalar(0)));
    cv::circle(invalid.back(), cv::Point(159, 120), 70, cv::Scalar(255), cv::FILLED);
    invalid.push_back(cv::Mat(241, 319, CV_8UC1, cv::Scalar(0)));
    cv::rectangle(invalid.back(), cv::Rect(49, 90, 220, 50), cv::Scalar(255), cv::FILLED);
    cv::rectangle(invalid.back(), cv::Rect(134, 30, 50, 180), cv::Scalar(255), cv::FILLED);
    // Known limitation, rather than an invalid quadrilateral: the stroked
    // rectangle's beveled outer corners do not pass the current extrema gates.
    // Keep the observed rejection explicit next to the sharp-frame positive.
    invalid.push_back(cv::Mat(241, 319, CV_8UC1, cv::Scalar(0)));
    cv::rectangle(invalid.back(), cv::Point(60, 40), cv::Point(240, 200), cv::Scalar(255), 4);
    for (std::size_t index = 0; index < invalid.size(); ++index) {
        SCOPED_TRACE(index);
        std::vector<std::vector<cv::Point2f>> candidates;
        std::vector<std::vector<cv::Point>> contours;
        extractor.append(invalid[index], config, 2.0, candidates, contours);
        EXPECT_TRUE(candidates.empty());
        EXPECT_TRUE(contours.empty());
    }
    cv::Mat square(241, 319, CV_8UC1, cv::Scalar(0));
    // Filled rectangles give sharp integer corners. A stroked cv::rectangle
    // bevels its outer corners; accepting that shape is not guaranteed by the
    // experimental extrema/inlier filter.
    cv::rectangle(square, cv::Rect(60, 40, 181, 161), cv::Scalar(255), cv::FILLED);
    cv::rectangle(square, cv::Rect(64, 44, 173, 153), cv::Scalar(0), cv::FILLED);
    std::vector<std::vector<cv::Point2f>> candidates;
    std::vector<std::vector<cv::Point>> contours;
    extractor.append(square, config, 2.0, candidates, contours);
    ASSERT_EQ(candidates.size(), 1U);
    ASSERT_EQ(contours.size(), 1U);
    ASSERT_EQ(candidates[0].size(), 4U);
    EXPECT_GE(contours[0].size(), 128U);
}

TEST_F(CpuDetectorTest, packed_dictionary_preserves_ambiguous_cells_and_first_qualifying_id) {
    const cv::aruco::Dictionary dictionary =
            cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_MIP_36h12);
    const int last_id = dictionary.bytesList.rows - 1;
    const cv::Mat last_bits = cv::aruco::Dictionary::getBitsFromByteList(
            dictionary.bytesList.row(last_id), dictionary.markerSize);
    const cv::Mat first_bits = cv::aruco::Dictionary::getBitsFromByteList(
            dictionary.bytesList.row(0), dictionary.markerSize);
    const cv::Mat clean = marker_scene(cv::aruco::DICT_ARUCO_MIP_36h12, last_id, 0);
    std::vector<cv::Mat> images{clean};
    const auto make_half_cell = [](cv::Mat& image, int row, int column) {
        // Eight cells across the marker, each twenty pixels. With a 160-pixel
        // canonical image this gives an exact 0.5 white-pixel ratio per cell.
        const cv::Rect cell(80 + (column + 1) * 20, 80 + (row + 1) * 20, 20, 20);
        image(cell).setTo(0);
        image(cv::Rect(cell.x, cell.y, 10, 20)).setTo(255);
    };
    for (int half_cells : {3, 12}) {
        cv::Mat image = clean.clone();
        for (int index = 0; index < half_cells; ++index) {
            make_half_cell(image, index / dictionary.markerSize, index % dictionary.markerSize);
        }
        images.push_back(std::move(image));
    }

    cv::Mat first_qualifying = clean.clone();
    std::array<bool, 36> known{};
    int differing_known = 0;
    for (int row = 0; row < dictionary.markerSize; ++row) {
        for (int column = 0; column < dictionary.markerSize; ++column) {
            const int index = row * dictionary.markerSize + column;
            if (last_bits.at<unsigned char>(row, column) ==
                first_bits.at<unsigned char>(row, column)) {
                known[static_cast<std::size_t>(index)] = true;
            } else if (differing_known < 3) {
                known[static_cast<std::size_t>(index)] = true;
                ++differing_known;
            } else {
                make_half_cell(first_qualifying, row, column);
            }
        }
    }
    ASSERT_EQ(differing_known, 3);
    int first_distance = 36;
    cv::Mat rotated = first_bits.clone();
    for (int rotation = 0; rotation < 4; ++rotation) {
        int distance = 0;
        for (int row = 0; row < dictionary.markerSize; ++row) {
            for (int column = 0; column < dictionary.markerSize; ++column) {
                const std::size_t index =
                        static_cast<std::size_t>(row * dictionary.markerSize + column);
                if (known[index] && rotated.at<unsigned char>(row, column) !=
                                            last_bits.at<unsigned char>(row, column)) {
                    ++distance;
                }
            }
        }
        first_distance = std::min(first_distance, distance);
        cv::rotate(rotated, rotated, cv::ROTATE_90_CLOCKWISE);
    }
    // At thresholds >=0.5 the half cells set neither mask. ID 0 has three
    // errors and qualifies first, while the later ID has zero errors. Choosing
    // the nearest dictionary entry instead would therefore return the last ID.
    ASSERT_EQ(first_distance, 3);
    ASSERT_EQ(static_cast<int>(dictionary.maxCorrectionBits * 0.6), 3);
    images.push_back(std::move(first_qualifying));

    for (double valid_bit_threshold : {0.1, 0.49, 0.5, 0.51, 0.9}) {
        DetectorConfig config;
        config.use_aruco3_detection_ = false;
        config.corner_refine_method_ = CornerRefineMethod::kNone;
        config.perspective_remove_pixel_per_cell_ = 20;
        config.perspective_remove_ignored_margin_per_cell_ = 0.0;
        config.valid_bit_threshold_ = valid_bit_threshold;
        for (std::size_t image_index = 0; image_index < images.size(); ++image_index) {
            SCOPED_TRACE("threshold=" + std::to_string(valid_bit_threshold) +
                         " image=" + std::to_string(image_index));
            const CpuResult expected =
                    reference_detect(images[image_index], config, cv::aruco::DICT_ARUCO_MIP_36h12);
            if (image_index == 0U || image_index == 1U) {
                ASSERT_EQ(expected.ids_, std::vector<int>({last_id}));
            }
            if (image_index >= 2U && valid_bit_threshold < 0.5) {
                // A half cell sets both masks below 0.5 and always counts as
                // an error; these images exceed the three-error allowance.
                ASSERT_TRUE(expected.ids_.empty());
            }
            if (image_index == images.size() - 1U && valid_bit_threshold >= 0.5) {
                ASSERT_EQ(expected.ids_, std::vector<int>({0}));
            }
            for (unsigned int mask : {0U, 2U}) {
                for (bool lookup : {false, true}) {
                    CpuOptions options = options_for_mask(mask);
                    options.exact_dictionary_lookup_ = lookup;
                    CpuDetector detector;
                    std::string message;
                    ASSERT_EQ(
                            detector.initialize("DICT_ARUCO_MIP_36h12", config, &message, options),
                            Status::kOk)
                            << message;
                    CpuResult actual;
                    ASSERT_EQ(detector.detect(images[image_index], &actual, &message), Status::kOk)
                            << message;
                    expect_same_detections(actual, expected);
                }
            }
        }
    }
}

TEST_F(CpuDetectorTest, threshold_prefix_budget_changes_preserve_public_detector_results) {
    CpuDetector detector;
    const cv::Mat marker = marker_scene(cv::aruco::DICT_ARUCO_MIP_36h12, 42, 3);
    cv::Mat owner(491, 677, CV_8UC1, cv::Scalar(0));
    cv::Mat roi = owner(cv::Rect(7, 3, 641, 479));
    roi.setTo(255);
    marker.copyTo(roi(cv::Rect(11, 17, marker.cols, marker.rows)));
    const std::vector<cv::Mat> images{marker, roi, marker};
    const std::vector<std::size_t> budgets = {0U,
                                              1U,
                                              1024U,
                                              64U * 1024U,
                                              2U * 1024U * 1024U,
                                              4U * 1024U * 1024U,
                                              std::numeric_limits<std::size_t>::max()};
    for (bool aruco3 : {false, true}) {
        DetectorConfig config;
        config.use_aruco3_detection_ = aruco3;
        config.corner_refine_method_ =
                aruco3 ? CornerRefineMethod::kSubpix : CornerRefineMethod::kNone;
        for (std::size_t budget : budgets) {
            SCOPED_TRACE("aruco3=" + std::to_string(aruco3) + " budget=" + std::to_string(budget));
            CpuOptions options;
            options.threshold_max_prefix_bytes_ = budget;
            std::string message;
            ASSERT_EQ(detector.initialize("DICT_ARUCO_MIP_36h12", config, &message, options),
                      Status::kOk)
                    << message;
            for (const cv::Mat& image : images) {
                CpuResult actual;
                ASSERT_EQ(detector.detect(image, &actual, &message), Status::kOk) << message;
                expect_same_detections(
                        actual, reference_detect(image, config, cv::aruco::DICT_ARUCO_MIP_36h12));
            }
        }
    }
}

}  // namespace
