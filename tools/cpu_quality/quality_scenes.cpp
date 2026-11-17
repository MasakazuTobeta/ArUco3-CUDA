// SPDX-License-Identifier: Apache-2.0
#include "quality_scenes.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace aruco3cuda::cpu_quality {
namespace {

constexpr double kPi = 3.14159265358979323846;

MarkerSpec make_marker(int id, cv::Point2f center, float side, double angle_deg, double perspective,
                       cv::RNG& random) {
    const double angle = angle_deg * kPi / 180.0;
    const float cosine = static_cast<float>(std::cos(angle));
    const float sine = static_cast<float>(std::sin(angle));
    const float half = side * 0.5F;
    const std::array<cv::Point2f, 4> local = {cv::Point2f(-half, -half), cv::Point2f(half, -half),
                                              cv::Point2f(half, half), cv::Point2f(-half, half)};
    MarkerSpec marker;
    marker.id_ = id;
    for (std::size_t corner = 0; corner < local.size(); ++corner) {
        const float distortion = static_cast<float>(perspective) * side * 0.20F;
        marker.corners_[corner] = center +
                                  cv::Point2f(local[corner].x * cosine - local[corner].y * sine,
                                              local[corner].x * sine + local[corner].y * cosine) +
                                  cv::Point2f(random.uniform(-distortion, distortion),
                                              random.uniform(-distortion, distortion));
    }
    return marker;
}

}  // namespace

std::vector<FrameSpec> make_adversarial_specs(cv::Size size, int dictionary_count) {
    if (size.width < 320 || size.height < 320 || dictionary_count <= 0) {
        throw std::invalid_argument("quality preset requires dimensions >=320 and a dictionary");
    }
    std::vector<FrameSpec> result;
    cv::RNG random(2026111701U);
    const float longest = static_cast<float>(std::max(size.width, size.height));
    const float side = std::min(std::max(140.0F, 64.0F + 0.12F * longest),
                                0.4F * static_cast<float>(std::min(size.width, size.height)));
    const auto add = [&](const std::string& family, int index) {
        FrameSpec spec;
        spec.name_ = "quality_" + family + "_" + std::to_string(index);
        spec.family_ = family;
        spec.size_ = size;
        spec.seed_ = 2026111701U + static_cast<std::uint64_t>(result.size()) * 65537U;
        return spec;
    };
    const auto id_for = [&](int index) { return (17 + index * 37) % dictionary_count; };
    for (int index = 0; index < 12; ++index) {
        FrameSpec spec = add("clean_pose", index);
        const cv::Point2f center(
                static_cast<float>(size.width) * (index % 2 == 0 ? 0.36F : 0.64F) + 0.37F,
                static_cast<float>(size.height) * 0.5F + 0.49F);
        spec.markers_.push_back(
                make_marker(id_for(index), center, side, index * 30.0, 0.0, random));
        spec.require_detection_ = side > 40.0F + 0.05F * longest;
        result.push_back(std::move(spec));
    }
    for (int index = 0; index < 12; ++index) {
        FrameSpec spec = add("projective", index);
        spec.markers_.push_back(
                make_marker(id_for(index + 12),
                            cv::Point2f(size.width * 0.5F + 0.49F, size.height * 0.5F + 0.37F),
                            side, 13.0 + 23.0 * index, 0.08 + 0.035 * index, random));
        spec.blur_sigma_px_ = 0.4;
        spec.noise_sigma_levels_ = 3.0;
        spec.background_kind_ = index % 3;
        result.push_back(std::move(spec));
    }
    for (int blur = 0; blur < 4; ++blur) {
        for (int noise = 0; noise < 4; ++noise) {
            const int index = blur * 4 + noise;
            FrameSpec spec = add("blur_noise", index);
            spec.markers_.push_back(make_marker(
                    id_for(index + 24),
                    cv::Point2f(size.width * 0.5F + 0.125F, size.height * 0.5F + 0.625F), side,
                    index * 19.0, 0.15, random));
            spec.blur_sigma_px_ =
                    std::array<double, 4>{0.0, 0.6, 1.1, 1.8}[static_cast<std::size_t>(blur)];
            spec.noise_sigma_levels_ =
                    std::array<double, 4>{0.0, 5.0, 12.0, 24.0}[static_cast<std::size_t>(noise)];
            spec.background_kind_ = 1;
            result.push_back(std::move(spec));
        }
    }
    for (int index = 0; index < 12; ++index) {
        FrameSpec spec = add("lighting", index);
        spec.markers_.push_back(
                make_marker(id_for(index + 40),
                            cv::Point2f(size.width * 0.5F + 0.375F, size.height * 0.5F + 0.875F),
                            side, index * 29.0, 0.05, random));
        spec.contrast_ = std::array<double, 6>{
                0.07, 0.12, 0.2, 0.35, 0.6, 1.0}[static_cast<std::size_t>(index % 6)];
        spec.illumination_gradient_ = index < 6 ? 0.4 : 1.0;
        result.push_back(std::move(spec));
    }
    const std::array<float, 12> offsets{-2.0F, -1.0F, -0.5F, -0.1F, 0.0F, 0.1F,
                                        0.5F,  1.0F,  2.0F,  4.0F,  8.0F, 16.0F};
    for (int index = 0; index < 12; ++index) {
        FrameSpec spec = add("size_bound", index);
        spec.markers_.push_back(
                make_marker(id_for(index + 52),
                            cv::Point2f(size.width * 0.5F + 0.49F, size.height * 0.5F + 0.51F),
                            32.0F + 0.05F * longest + offsets[static_cast<std::size_t>(index)],
                            index % 3 == 0 ? 0.0 : 17.0, 0.0, random));
        result.push_back(std::move(spec));
    }
    for (int index = 0; index < 12; ++index) {
        FrameSpec spec = add("border_occlusion", index);
        const float left = std::array<float, 6>{
                -3.0F, -0.1F, 0.0F, 2.0F, 3.0F, 6.0F}[static_cast<std::size_t>(index % 6)];
        spec.markers_.push_back(make_marker(
                id_for(index + 64), cv::Point2f(left + side * 0.5F, size.height * 0.5F + 0.125F),
                side, 0.0, 0.0, random));
        spec.occlusion_fraction_ = index < 6 ? 0.0 : 0.15;
        result.push_back(std::move(spec));
    }
    for (int index = 0; index < 12; ++index) {
        FrameSpec spec = add("multi_dense", index);
        const int count = std::array<int, 3>{2, 4, 9}[static_cast<std::size_t>(index % 3)];
        const int columns = count == 2 ? 2 : (count == 4 ? 2 : 3);
        const int rows = (count + columns - 1) / columns;
        const float cell_width = static_cast<float>(size.width) / columns;
        const float cell_height = static_cast<float>(size.height) / rows;
        const float marker_side = std::min(cell_width, cell_height) * 0.60F;
        for (int marker = 0; marker < count; ++marker) {
            const int id = index % 4 == 0 ? id_for(index + 76) : id_for(index + marker + 76);
            spec.markers_.push_back(make_marker(
                    id,
                    cv::Point2f(
                            (static_cast<float>(marker % columns) + 0.5F) * cell_width + 0.25F,
                            (static_cast<float>(marker / columns) + 0.5F) * cell_height + 0.75F),
                    marker_side, (index + marker) * 11.0, 0.06, random));
        }
        spec.background_kind_ = 2;
        spec.noise_sigma_levels_ = index % 2 == 0 ? 0.0 : 9.0;
        result.push_back(std::move(spec));
    }
    for (int index = 0; index < 8; ++index) {
        FrameSpec spec = add("negative", index);
        spec.background_kind_ = index % 4;
        spec.noise_sigma_levels_ = index < 4 ? 0.0 : 35.0;
        result.push_back(std::move(spec));
    }
    return result;
}

cv::Mat render_frame(const FrameSpec& spec, const cv::aruco::Dictionary& dictionary,
                     int border_bits) {
    if (spec.size_.width < 1 || spec.size_.height < 1 || spec.size_.width > 65536 ||
        spec.size_.height > 65536 || border_bits < 1 || border_bits > 16 ||
        !(spec.blur_sigma_px_ >= 0.0 && spec.blur_sigma_px_ <= 16.0) ||
        !(spec.noise_sigma_levels_ >= 0.0 && spec.noise_sigma_levels_ <= 128.0) ||
        !(spec.contrast_ > 0.0 && spec.contrast_ <= 1.0) ||
        !(spec.illumination_gradient_ >= 0.0 && spec.illumination_gradient_ <= 1.0) ||
        !(spec.occlusion_fraction_ >= 0.0 && spec.occlusion_fraction_ <= 1.0) ||
        spec.background_kind_ < 0 || spec.background_kind_ > 3 ||
        static_cast<std::uint64_t>(spec.size_.width) *
                        static_cast<std::uint64_t>(spec.size_.height) >
                128U * 1024U * 1024U) {
        throw std::invalid_argument("invalid quality frame settings");
    }
    cv::Mat image(spec.size_, CV_8UC1, cv::Scalar(235));
    cv::RNG random(spec.seed_);
    if (spec.background_kind_ == 1) {
        random.fill(image, cv::RNG::NORMAL, 220, 12);
    } else if (spec.background_kind_ == 2) {
        for (int index = 0; index < 24; ++index) {
            const cv::Point first(random.uniform(0, image.cols), random.uniform(0, image.rows));
            const cv::Point second(random.uniform(0, image.cols), random.uniform(0, image.rows));
            cv::line(image, first, second, cv::Scalar(random.uniform(40, 220)), 1 + index % 3);
        }
    } else if (spec.background_kind_ == 3) {
        for (int y = 0; y < image.rows; ++y) {
            std::uint8_t* row = image.ptr<std::uint8_t>(y);
            for (int x = 0; x < image.cols; ++x) {
                row[x] = ((x / 9 + y / 9) & 1) == 0 ? 55U : 215U;
            }
        }
    }
    constexpr int kCanonicalSide = 256;
    const std::array<cv::Point2f, 4> canonical_corners = {
            cv::Point2f(-0.5F, -0.5F), cv::Point2f(255.5F, -0.5F), cv::Point2f(255.5F, 255.5F),
            cv::Point2f(-0.5F, 255.5F)};
    for (const MarkerSpec& marker : spec.markers_) {
        if (marker.id_ < 0 || marker.id_ >= dictionary.bytesList.rows) {
            throw std::invalid_argument("quality marker ID is outside the dictionary");
        }
        for (const cv::Point2f& corner : marker.corners_) {
            if (!std::isfinite(corner.x) || !std::isfinite(corner.y)) {
                throw std::invalid_argument("quality marker corner is not finite");
            }
        }
        cv::Mat canonical;
        dictionary.generateImageMarker(marker.id_, kCanonicalSide, canonical, border_bits);
        const cv::Mat transform =
                cv::getPerspectiveTransform(canonical_corners.data(), marker.corners_.data());
        cv::Mat warped;
        cv::warpPerspective(canonical, warped, transform, image.size(), cv::INTER_LINEAR,
                            cv::BORDER_CONSTANT, cv::Scalar(255));
        cv::Mat mask;
        cv::warpPerspective(cv::Mat(kCanonicalSide, kCanonicalSide, CV_8UC1, cv::Scalar(255)), mask,
                            transform, image.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT,
                            cv::Scalar(0));
        warped.copyTo(image, mask);
        if (spec.occlusion_fraction_ > 0.0) {
            const cv::Rect bounds = cv::boundingRect(
                    std::vector<cv::Point2f>(marker.corners_.begin(), marker.corners_.end()));
            const int covered_width =
                    std::max(1, static_cast<int>(bounds.width * spec.occlusion_fraction_));
            const cv::Rect covered(bounds.x + bounds.width / 3, bounds.y, covered_width,
                                   bounds.height);
            const cv::Rect clipped = covered & cv::Rect(0, 0, image.cols, image.rows);
            if (!clipped.empty()) {
                image(clipped).setTo(235);
            }
        }
    }
    if (spec.blur_sigma_px_ > 0.0) {
        cv::GaussianBlur(image, image, cv::Size(), spec.blur_sigma_px_);
    }
    cv::Mat noise;
    if (spec.noise_sigma_levels_ > 0.0) {
        noise.create(image.size(), CV_16SC1);
        random.fill(noise, cv::RNG::NORMAL, 0.0, spec.noise_sigma_levels_);
    }
    for (int y = 0; y < image.rows; ++y) {
        std::uint8_t* row = image.ptr<std::uint8_t>(y);
        const std::int16_t* noise_row = noise.empty() ? nullptr : noise.ptr<std::int16_t>(y);
        for (int x = 0; x < image.cols; ++x) {
            const double gradient =
                    1.0 - spec.illumination_gradient_ * 0.45 * static_cast<double>(x) / image.cols;
            const double value = (128.0 + (row[x] - 128.0) * spec.contrast_) * gradient +
                                 (noise_row == nullptr ? 0.0 : static_cast<double>(noise_row[x]));
            row[x] = cv::saturate_cast<std::uint8_t>(value);
        }
    }
    return image;
}

}  // namespace aruco3cuda::cpu_quality
