// SPDX-License-Identifier: Apache-2.0
#include "cpu_box_sample.hpp"
#include "cpu_detector.hpp"
#include "cpu_grid_refine.hpp"

#include <gtest/gtest.h>
#include <limits>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

namespace {
using aruco3cuda::DetectorConfig;
using aruco3cuda::Status;
using aruco3cuda::cpu::CpuDetector;
using aruco3cuda::cpu::CpuOptions;
using aruco3cuda::cpu::CpuResult;
using aruco3cuda::cpu::detail::fit_decoded_grid;

TEST(DecodedGrid, fits_projected_cells_using_only_visible_pitched_pixels) {
    const auto dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h11);
    cv::Mat cells, tag;
    dict.generateImageMarker(17, 8, cells, 1);
    dict.generateImageMarker(17, 80, tag, 1);
    cv::Mat storage(120, 157, CV_8UC1, cv::Scalar(14));
    auto img = storage(cv::Rect(7, 9, 128, 96));
    img.setTo(244);
    tag.colRange(2, 80).copyTo(img(cv::Rect(0, 8, 78, 80)));
    const std::vector<cv::Point2f> truth{
            {-2.5F, 7.5F}, {77.5F, 7.5F}, {77.5F, 87.5F}, {-2.5F, 87.5F}};
    std::vector<cv::Point2f> fitted{{-.5F, 7.5F}, {77.5F, 8.0F}, {77.5F, 87.5F}, {-.5F, 87.5F}};
    auto packed = fitted;
    ASSERT_TRUE(fit_decoded_grid(img, cells, 6, fitted));
    ASSERT_TRUE(fit_decoded_grid(img.clone(), cells, 6, packed));
    EXPECT_EQ(fitted, packed);
    for (std::size_t i = 0; i < 4; ++i)
        EXPECT_LT(cv::norm(fitted[i] - truth[i]), 1.4);
    const auto saved = fitted;
    img.setTo(105);
    EXPECT_FALSE(fit_decoded_grid(img, cells, 6, fitted));
    EXPECT_EQ(fitted, saved);
    EXPECT_FALSE(fit_decoded_grid(img, cells, 1, fitted));
}

TEST(DecodedGrid, detector_preserves_id_and_order_in_all_quarter_turns) {
    const auto dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h11);
    CpuOptions options;
    options.decoded_grid_radius_px_ = 6;
    CpuDetector detector;
    DetectorConfig config;
    std::string message;
    EXPECT_EQ(detector.initialize("DICT_4X4_50", config, &message, options),
              Status::kInvalidConfig);
    ASSERT_EQ(detector.initialize("DICT_APRILTAG_36h11", config, &message, options), Status::kOk);
    cv::Mat tag;
    dict.generateImageMarker(17, 120, tag, 1);
    const std::array<cv::Point2f, 4> truth{
            {{39.5F, 39.5F}, {159.5F, 39.5F}, {159.5F, 159.5F}, {39.5F, 159.5F}}};
    for (int rotation = 0; rotation < 4; ++rotation) {
        cv::Mat image(200, 200, CV_8UC1, cv::Scalar(245));
        tag.copyTo(image(cv::Rect(40, 40, 120, 120)));
        CpuResult result;
        ASSERT_EQ(detector.detect(image, &result, &message), Status::kOk);
        ASSERT_EQ(result.ids_, std::vector<int>{17});
        for (int i = 0; i < 4; ++i)
            EXPECT_LT(cv::norm(result.corners_[0][i] - truth[(i + rotation) % 4]), .7);
        cv::rotate(tag, tag, cv::ROTATE_90_CLOCKWISE);
    }
    options.decoded_grid_radius_px_ = 33;
    EXPECT_EQ(detector.initialize("DICT_APRILTAG_36h11", config, &message, options),
              Status::kInvalidConfig);
}
}  // namespace

TEST(DecodedGridSampling, centered_samples_match_scalar_on_pitched_odd_images) {
    cv::Mat storage(40, 131, CV_8UC1);
    cv::randu(storage, 0, 256);
    for (int width : {3, 31, 63, 95, 97, 105}) {
        const cv::Mat image = storage(cv::Rect(3, 1, width, 35));
        for (int step = 2; step <= 8; ++step) {
            const auto size = cv::Size(std::max(1, cvRound(static_cast<double>(width) / step)),
                                       cvRound(35.0 / step));
            cv::Mat sampled;
            aruco3cuda::cpu::detail::sample_box_grid(image, sampled, size, step);
            for (int y = 0; y < size.height; ++y)
                for (int x = 0; x < size.width; ++x) {
                    const int cx = std::min(x * step + (step - 2) / 2, width - 1);
                    const int cy = std::min(y * step + (step - 2) / 2, 34);
                    const auto row_value = [&](int row) {
                        return (image.at<unsigned char>(row, cx) +
                                image.at<unsigned char>(row, std::min(cx + 1, width - 1)) + 1) /
                               2;
                    };
                    EXPECT_EQ(sampled.at<unsigned char>(y, x),
                              (row_value(cy) + row_value(std::min(cy + 1, 34)) + 1) / 2);
                }
        }
    }
}

TEST(DecodedGridSampling, public_detector_handles_odd_pitched_frames_and_empty_results) {
    using namespace aruco3cuda;
    cpu::CpuDetector detector;
    DetectorConfig config;
    config.min_side_length_canonical_img_px_ = 12;
    cpu::CpuOptions options;
    options.candidate_box_stride_px_ = 6;
    std::string message;
    EXPECT_EQ(detector.initialize("DICT_APRILTAG_36h11", config, &message, options),
              Status::kInvalidConfig);
    options.decode_at_source_resolution_ = true;
    options.decoded_grid_radius_px_ = 6;
    ASSERT_EQ(detector.initialize("DICT_APRILTAG_36h11", config, &message, options), Status::kOk);
    const auto dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h11);
    cv::Mat tag, parent(238, 301, CV_8UC1, cv::Scalar(7));
    auto image = parent(cv::Rect(3, 4, 285, 229));
    image.setTo(238);
    dict.generateImageMarker(253, 166, tag);
    tag.copyTo(image(cv::Rect(71, 29, 166, 166)));
    cpu::CpuResult result;
    ASSERT_EQ(detector.detect(image, &result, &message), Status::kOk) << message;
    ASSERT_EQ(result.ids_, std::vector<int>{253});
    const auto pitched = result.corners_;
    cv::Mat invalid_pitch = image;
    invalid_pitch.step[0] = std::numeric_limits<std::size_t>::max();
    EXPECT_EQ(detector.detect(invalid_pitch, &result, &message), Status::kInvalidImage);
    EXPECT_EQ(result.corners_, pitched);
    ASSERT_EQ(detector.detect(image.clone(), &result, &message), Status::kOk) << message;
    ASSERT_EQ(result.corners_, pitched);
    EXPECT_LT(cv::norm(result.corners_[0][0] - cv::Point2f(70.5F, 28.5F)), .7);
    ASSERT_EQ(detector.detect(cv::Mat(2, 1, CV_8UC1, cv::Scalar(100)), &result, &message),
              Status::kOk)
            << message;
    EXPECT_TRUE(result.ids_.empty());
    for (const int invalid : {-1, 1, 9}) {
        options.candidate_box_stride_px_ = invalid;
        EXPECT_EQ(detector.initialize("DICT_APRILTAG_36h11", config, &message, options),
                  Status::kInvalidConfig);
    }
}

TEST(DecodedGridSampling, supplemental_search_keeps_interior_and_recovers_two_same_side_tags) {
    using namespace aruco3cuda;
    const auto dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h11);
    cv::Mat frame(480, 640, CV_8UC1, cv::Scalar(235)), tag;
    dict.generateImageMarker(17, 96, tag);
    tag.colRange(2, 96).copyTo(frame(cv::Rect(0, 40, 94, 96)));
    tag.colRange(1, 96).copyTo(frame(cv::Rect(0, 230, 95, 96)));
    dict.generateImageMarker(33, 128, tag);
    tag.copyTo(frame(cv::Rect(355, 105, 128, 128)));
    cpu::CpuDetector detector;
    cpu::CpuOptions options;
    options.decode_at_source_resolution_ = true;
    options.decoded_grid_radius_px_ = 6;
    DetectorConfig config;
    config.min_marker_length_ratio_original_img_ = .04F;
    config.min_distance_to_border_px_ = 0;
    config.adaptive_thresh_win_size_step_px_ = 20;
    cpu::CpuResult plain, augmented;
    std::string message;
    ASSERT_EQ(detector.initialize("DICT_APRILTAG_36h11", config, &message, options), Status::kOk);
    ASSERT_EQ(detector.detect(frame, &plain, &message), Status::kOk);
    ASSERT_NE(std::find(plain.ids_.begin(), plain.ids_.end(), 33), plain.ids_.end());
    options.search_frame_edges_ = true;
    ASSERT_EQ(detector.initialize("DICT_APRILTAG_36h11", config, &message, options), Status::kOk);
    ASSERT_EQ(detector.detect(frame, &augmented, &message), Status::kOk) << message;
    ASSERT_GE(augmented.ids_.size(), plain.ids_.size());
    for (std::size_t i = 0; i < plain.ids_.size(); ++i) {
        EXPECT_EQ(augmented.ids_[i], plain.ids_[i]);
        EXPECT_EQ(augmented.corners_[i], plain.corners_[i]);
    }
    EXPECT_EQ(std::count(augmented.ids_.begin(), augmented.ids_.end(), 17), 2);
    EXPECT_EQ(std::count(augmented.ids_.begin(), augmented.ids_.end(), 33), 1);
    options.frame_edge_max_pixels_ = 0;
    ASSERT_EQ(detector.initialize("DICT_APRILTAG_36h11", config, &message, options), Status::kOk);
    ASSERT_EQ(detector.detect(frame, &augmented, &message), Status::kOk);
    EXPECT_EQ(augmented.ids_, plain.ids_);
    EXPECT_EQ(augmented.corners_, plain.corners_);
}
