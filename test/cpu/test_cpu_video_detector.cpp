// SPDX-License-Identifier: Apache-2.0
#include "cpu_video_detector.hpp"

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#include <limits>
#include <string>

#include "cpu_test_support.hpp"

namespace aruco3cuda::cpu {
namespace {

constexpr char kDictionaryName[] = "DICT_ARUCO_MIP_36h12";
constexpr auto kDictionary = cv::aruco::DICT_ARUCO_MIP_36h12;
constexpr int kMarkerSidePx = 128;

class CpuVideoDetectorTest : public testing::Test {
protected:
    void SetUp() override {
        this->previous_threads_ = cv::getNumThreads();
        cv::setNumThreads(1);
    }

    void TearDown() override { cv::setNumThreads(this->previous_threads_); }

private:
    int previous_threads_ = 1;
};

void draw_marker(cv::Mat& frame, int id, const cv::Point& location, bool rotate = false,
                 int black = 0, int white = 255) {
    cv::Mat marker;
    cv::aruco::generateImageMarker(cv::aruco::getPredefinedDictionary(kDictionary), id,
                                   kMarkerSidePx, marker);
    if (rotate)
        cv::rotate(marker, marker, cv::ROTATE_90_CLOCKWISE);
    if (black != 0 || white != 255)
        marker.convertTo(marker, CV_8U, (white - black) / 255.0, black);
    marker.copyTo(frame(cv::Rect(location, marker.size())));
}

void expect_marker_geometry(const CpuResult& expected, const CpuResult& actual,
                            double tolerance_px) {
    ASSERT_EQ(expected.ids_.size(), expected.corners_.size());
    ASSERT_EQ(actual.ids_.size(), actual.corners_.size());
    ASSERT_EQ(expected.ids_.size(), actual.ids_.size());
    for (std::size_t i = 0; i < expected.ids_.size(); ++i) {
        ASSERT_EQ(expected.corners_[i].size(), 4U);
        ASSERT_EQ(actual.corners_[i].size(), 4U);
    }
    const auto baseline_order = test_support::detection_order(expected);
    const auto video_order = test_support::detection_order(actual);
    for (std::size_t i = 0; i < expected.ids_.size(); ++i) {
        const std::size_t baseline = baseline_order[i];
        const std::size_t video = video_order[i];
        EXPECT_EQ(expected.ids_[baseline], actual.ids_[video]);
        for (std::size_t corner = 0; corner < 4U; ++corner) {
            const double delta_px =
                    cv::norm(expected.corners_[baseline][corner] - actual.corners_[video][corner]);
            EXPECT_LE(delta_px, tolerance_px) << "ordered corner " << corner;
        }
    }
}

TEST_F(CpuVideoDetectorTest, DetectsNewAndExistingMarkersFromCurrentPixels) {
    for (const bool use_aruco3 : {false, true}) {
        SCOPED_TRACE(use_aruco3);
        DetectorConfig config;
        config.use_aruco3_detection_ = use_aruco3;
        CpuVideoDetector video;
        CpuDetector baseline;
        ASSERT_EQ(video.initialize(kDictionaryName, config), Status::kOk);
        ASSERT_EQ(baseline.initialize(kDictionaryName, config), Status::kOk);

        // Overwrite the same borrowed buffer on each frame. Its parent has unrelated pixels.
        cv::Mat parent(749, 1313, CV_8UC1, cv::Scalar(51));
        cv::Mat frame = parent(cv::Rect(15, 9, 1280, 720));
        CpuVideoResult result;
        const auto compare_frame = [&](const char* step, bool full, std::size_t count) {
            SCOPED_TRACE(step);
            CpuResult expected;
            ASSERT_EQ(baseline.detect(frame, &expected), Status::kOk);
            ASSERT_EQ(expected.ids_.size(), count);
            ASSERT_EQ(video.detect(frame, &result), Status::kOk);
            EXPECT_EQ(result.full_frame_, full);
            EXPECT_EQ(result.roi_count_ == 0U, full);
            expect_marker_geometry(expected, result.detections_, full ? 0.0 : 2.5);
        };

        frame.setTo(255);
        draw_marker(frame, 63, {374, 157});
        compare_frame("first frame", true, 1U);
        compare_frame("stationary", false, 1U);
        frame.setTo(255);
        draw_marker(frame, 63, {385, 165}, true);
        draw_marker(frame, 171, {892, 543});
        compare_frame("rotation, movement and distant new ID", false, 2U);
        frame.setTo(255);
        draw_marker(frame, 63, {385, 165}, true);
        draw_marker(frame, 63, {892, 543});
        compare_frame("replacing a prior ID forces full search", true, 2U);
        compare_frame("two instances with the same ID use regional search", false, 2U);
        frame.setTo(255);
        draw_marker(frame, 63, {892, 543});
        compare_frame("losing one of two instances forces full search", true, 1U);
        compare_frame("unchanged remaining instance uses regional search again", false, 1U);
    }
}

TEST_F(CpuVideoDetectorTest, RefreshFindsMarkerWhoseChangeIsBelowThreshold) {
    CpuVideoOptions options;
    options.refresh_interval_ = 4;
    options.change_threshold_ = 160;
    CpuVideoDetector video;
    ASSERT_EQ(video.initialize(kDictionaryName, DetectorConfig{}, nullptr, options), Status::kOk);
    cv::Mat frame(720, 1280, CV_8UC1, cv::Scalar(220));
    CpuVideoResult result;
    ASSERT_EQ(video.detect(frame, &result), Status::kOk);
    EXPECT_TRUE(result.full_frame_);
    ASSERT_TRUE(result.detections_.ids_.empty());

    // Every changed sample is at most 120 levels from the previous frame.
    draw_marker(frame, 51, {873, 543}, false, 100, 220);
    for (int i = 1; i < options.refresh_interval_; ++i) {
        ASSERT_EQ(video.detect(frame, &result), Status::kOk);
        EXPECT_FALSE(result.full_frame_);
        EXPECT_TRUE(result.detections_.ids_.empty());
    }
    ASSERT_EQ(video.detect(frame, &result), Status::kOk);
    EXPECT_TRUE(result.full_frame_);
    EXPECT_EQ(result.detections_.ids_, std::vector<int>{51});
}

TEST_F(CpuVideoDetectorTest, RefreshesAfterCutResizeResetAndInvalidFrame) {
    CpuVideoDetector video;
    CpuVideoResult result;
    cv::Mat frame(720, 1280, CV_8UC1, cv::Scalar(255));
    draw_marker(frame, 63, {374, 157});
    EXPECT_EQ(video.detect(frame, &result), Status::kNotInitialized);
    ASSERT_EQ(video.initialize(kDictionaryName, DetectorConfig{}), Status::kOk);
    ASSERT_EQ(video.detect(frame, &result), Status::kOk);
    EXPECT_TRUE(result.full_frame_);

    CpuVideoOptions invalid;
    invalid.max_region_fraction_ = std::numeric_limits<double>::quiet_NaN();
    EXPECT_EQ(video.initialize(kDictionaryName, DetectorConfig{}, nullptr, invalid),
              Status::kInvalidConfig);
    ASSERT_EQ(video.detect(frame, &result), Status::kOk);
    EXPECT_FALSE(result.full_frame_);
    ASSERT_EQ(result.detections_.ids_, std::vector<int>{63});

    const CpuResult saved = result.detections_;
    const cv::Mat color(64, 64, CV_8UC3, cv::Scalar(1, 2, 3));
    std::string message;
    EXPECT_EQ(video.detect(color, &result, &message), Status::kInvalidImage);
    EXPECT_FALSE(message.empty());
    expect_marker_geometry(saved, result.detections_, 0.0);
    ASSERT_EQ(video.detect(frame, &result), Status::kOk);
    EXPECT_TRUE(result.full_frame_);
    ASSERT_EQ(video.detect(frame, &result), Status::kOk);
    EXPECT_FALSE(result.full_frame_);
    frame.setTo(24);
    ASSERT_EQ(video.detect(frame, &result), Status::kOk);
    EXPECT_TRUE(result.full_frame_);
    EXPECT_TRUE(result.detections_.ids_.empty());

    cv::Mat resized(480, 640, CV_8UC1, cv::Scalar(255));
    draw_marker(resized, 51, {219, 187});
    ASSERT_EQ(video.detect(resized, &result), Status::kOk);
    EXPECT_TRUE(result.full_frame_);
    EXPECT_EQ(result.detections_.ids_, std::vector<int>{51});
    video.reset();
    ASSERT_EQ(video.detect(resized, &result), Status::kOk);
    EXPECT_TRUE(result.full_frame_);
    EXPECT_EQ(video.detect(resized, nullptr), Status::kInvalidArgument);
}

}  // namespace
}  // namespace aruco3cuda::cpu
