// SPDX-License-Identifier: Apache-2.0
#include "isolated_ink.hpp"

#include "binary_median.hpp"
#include "diffuse_edge_profile.hpp"
#include "faint_guide.hpp"
#include "guide_contours.hpp"
#include "local_contour_proof.hpp"
#include "noisy_crop_filter.hpp"
#include "noisy_ink.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <iterator>
#include <limits>
#include <stdexcept>

using namespace aruco3cuda::apriltag_small_cpu;

namespace {
void test_binary_guide_median() {
    cv::RNG rng(2580187);
    cv::Mat expected, result;
    const int old_threads = cv::getNumThreads();
    cv::setNumThreads(1);
    for (int width : {1, 2, 15, 16, 17, 31, 32, 33, 64, 249, 320}) {
        for (int height : {1, 2, 3, 17, 180}) {
            cv::Mat parent(height + 4, width + 14, CV_8UC1, cv::Scalar(255));
            auto mask = parent(cv::Rect(3, 2, width, height));
            for (int cutoff : {29, 151, 223}) {
                rng.fill(mask, cv::RNG::UNIFORM, 0, 256);
                cv::threshold(mask, mask, cutoff, 255, cv::THRESH_BINARY);
                cv::medianBlur(mask, expected, 3);
                detail::median_binary3(mask, result);
                if (cv::norm(expected, result, cv::NORM_INF) != 0)
                    throw std::runtime_error("Binary guide median differs from pinned OpenCV");
            }
        }
    }
    cv::setNumThreads(old_threads);
}

void test_noisy_crop_filter() {
    detail::NoisyCropFilter filter;
    cv::Mat expected, actual;
    cv::RNG rng(0x6f27aa08);
    for (int height : {2, 4, 6, 18, 38}) {
        for (int width : {2, 4, 6, 16, 18, 30, 32, 34, 94}) {
            cv::Mat parent(height + 5, width + 11, CV_8UC1, cv::Scalar(255));
            auto source = parent(cv::Rect(3, 2, width, height));
            rng.fill(source, cv::RNG::UNIFORM, 0, 256);
            for (float sigma : {.8F, 1.25F}) {
                cv::resize(source, expected, {width / 2, height / 2}, 0, 0, cv::INTER_AREA);
                cv::GaussianBlur(expected, expected, {3, 3}, sigma, sigma);
                filter.filter(source, sigma, actual);
                if (cv::norm(expected, actual, cv::NORM_INF) != 0)
                    throw std::runtime_error("Noisy crop filter differed from pinned OpenCV");
            }
        }
    }
}

void test_local_contour_proof() {
#if CV_SIMD128
    cv::RNG rng(0x26b84c37);
    std::size_t accepted = 0, rejected = 0;
    for (int trial = 0; trial < 720; ++trial) {
        const int width = 16 * (1 + trial % 6), height = 15 + trial % 53;
        cv::Mat parent(height + 8, width + 13, CV_8U, cv::Scalar(trial % 2 ? 0 : 255));
        cv::Mat source = parent(cv::Rect(5, 3, width, height));
        const int reference = rng.uniform(0, 256);
        const int spread = 1 + trial % 52;
        const int difference = 1 + trial % 36;
        rng.fill(source, cv::RNG::UNIFORM, std::max(0, reference - spread),
                 std::min(256, reference + spread + 1));
        if (trial % 3 == 0) {
            source.setTo(255);
            // Test thin diagonals, edge contacts, long horizontal runs and halos.
            const int x = trial % width, y = trial % (height - 7);
            cv::line(source, {x, y}, {std::min(x + 6, width - 1), y + 6}, cv::Scalar(0));
            if (trial % 2)
                cv::line(source, {0, y + 3}, {width - 1, y + 3}, cv::Scalar(0));
        }
        cv::Mat mean, residual, mask;
        cv::boxFilter(source.clone(), mean, CV_8U, {15, 15}, {-1, -1}, true, cv::BORDER_REPLICATE);
        cv::subtract(mean, source, residual);
        cv::compare(residual, difference, mask, cv::CMP_GE);
        bool expected = false;
        // Independent scalar scan of every selected middle-row run, using
        // the complete native OpenCV classifier as the reference.
        for (int y = detail::kMinimumGuideContourSide - 3; y + 1 < height && !expected;
             y += detail::kMinimumGuideContourSide - 2) {
            for (int x = 0; x < width && !expected; ++x) {
                if (!mask.at<std::uint8_t>(y, x))
                    continue;
                int end = x;
                while (end + 1 < width && mask.at<std::uint8_t>(y, end + 1))
                    ++end;
                bool above = false, below = false;
                for (int col = std::max(0, x - 1); col <= std::min(width - 1, end + 1); ++col) {
                    above |= mask.at<std::uint8_t>(y - 1, col) != 0;
                    below |= mask.at<std::uint8_t>(y + 1, col) != 0;
                }
                expected = above && below;
                x = end;
            }
        }
        const bool possible = detail::may_have_local_contour15(source, difference);
        if (!possible && expected)
            throw std::runtime_error("Sparse local contour proof differs from full OpenCV mask");
        possible ? ++accepted : ++rejected;
    }
    if (!accepted || !rejected)
        throw std::runtime_error("Both local contour proof branches must execute");
    for (int width : {1, 15, 17, 31, 33}) {
        if (!detail::may_have_local_contour15(cv::Mat::zeros(32, width, CV_8U), 4))
            throw std::runtime_error("Unsupported guide width must request the full mask");
    }
#endif
}

void test_guide_contours() {
    aruco3cuda::cpu::detail::ContourScanner scratch;
    std::vector<std::vector<cv::Point>> expected;
    std::vector<std::vector<cv::Point>> actual;
    cv::RNG rng(99361);
    for (int trial = 0; trial < 144; ++trial) {
        const int width = 1 + rng.uniform(0, 180);
        const int height = 1 + rng.uniform(0, 96);
        cv::Mat parent(height + 6, width + 11, CV_8U, cv::Scalar(trial & 1 ? 0 : 255));
        cv::Mat input = parent(cv::Rect(7, 2, width, height));
        rng.fill(input, cv::RNG::UNIFORM, 0, 256);
        cv::threshold(input, input, 25 + trial, 255, cv::THRESH_BINARY);
        if (trial % 3 && width > 42 && height > 30) {
            cv::rectangle(input, cv::Rect(2, 2, 40, 27), cv::Scalar(255), cv::FILLED);
            cv::rectangle(input, cv::Rect(9, 7, 22, 17), cv::Scalar(0), cv::FILLED);
        }
        cv::findContours(input.clone(), expected, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        expected.erase(std::remove_if(expected.begin(), expected.end(),
                                      [](const auto& path) {
                                          const auto box = cv::boundingRect(path);
                                          return std::min(box.width, box.height) <
                                                 detail::kMinimumGuideContourSide;
                                      }),
                       expected.end());
        const auto snapshot = parent.clone();
        detail::find_guide_contours(input, scratch, actual);
        if (expected != actual || cv::countNonZero(parent != snapshot))
            throw std::runtime_error("Filtered guide contours disagree with pinned OpenCV");
    }
}

void test_fixed_sampled_contours() {
    aruco3cuda::cpu::detail::ContourScanner scanner;
    cv::RNG rng(166394087);
    for (int width : {63, 64, 65, 639, 640, 641, 703, 704, 705}) {
        for (int cutoff : {25, 128, 225}) {
            cv::Mat parent(85, width + 11, CV_8UC1, cv::Scalar(255));
            cv::Mat mask = parent(cv::Rect(3, 2, width, 80));
            rng.fill(mask, cv::RNG::UNIFORM, 0, 256);
            cv::threshold(mask, mask, cutoff, 255, cv::THRESH_BINARY_INV);
            cv::rectangle(mask, {4, 3, 55, 65}, cv::Scalar(255), cv::FILLED);
            cv::rectangle(mask, {9, 8, 45, 55}, cv::Scalar(0), cv::FILLED);
            const auto keep = [](const std::vector<cv::Point>& path) {
                if (path.size() < 4)
                    return false;
                const auto bounds = cv::boundingRect(path);
                return std::min(bounds.width, bounds.height) >= 5;
            };
            std::vector<std::vector<cv::Point>> reference, filtered, actual;
            cv::findContours(mask.clone(), reference, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
            std::copy_if(reference.begin(), reference.end(), std::back_inserter(filtered), keep);
            scanner.for_each<cv::CHAIN_APPROX_SIMPLE, true>(mask, true, [&](const auto& path) {
                if (keep(path))
                    actual.push_back(path);
            });
            std::reverse(actual.begin(), actual.end());
            if (actual != filtered)
                throw std::runtime_error("Fixed sampled paths differ from OpenCV RETR_LIST order");
            const int words = (width + 63) / 64;
            std::vector<std::uint64_t> packed(std::size_t(words) * mask.rows);
            for (int y = 0; y < mask.rows; ++y)
                for (int x = 0; x < width; ++x)
                    packed[std::size_t(y) * words + x / 64] |=
                            std::uint64_t(mask.ptr<std::uint8_t>(y)[x] != 0) << (x % 64);
            actual.clear();
            scanner.for_each_packed<cv::CHAIN_APPROX_SIMPLE, true>(mask.size(), packed,
                                                                   [&](const auto& path) {
                                                                       if (keep(path))
                                                                           actual.push_back(path);
                                                                   });
            std::reverse(actual.begin(), actual.end());
            if (actual != filtered)
                throw std::runtime_error("Packed sampled paths differ from OpenCV RETR_LIST order");
        }
    }
}

void test_ordered_guide_hull() {
    detail::OutlineHull scratch;
    std::vector<cv::Point> expected, actual;
    const auto check = [&](const std::vector<cv::Point>& path) {
        cv::convexHull(path, expected);
        scratch.build(path, cv::boundingRect(path), actual);
        if (expected != actual)
            throw std::runtime_error("Ordered guide hull differs from pinned OpenCV");
    };
    cv::RNG rng(700723);
    for (int trial = 0; trial < 90; ++trial) {
        const int width = rng.uniform(5, 129), height = rng.uniform(5, 115);
        cv::Mat parent(height + 8, width + 11, CV_8UC1, cv::Scalar(255));
        cv::Mat mask = parent(cv::Rect(3, 4, width, height));
        rng.fill(mask, cv::RNG::UNIFORM, 0, 256);
        cv::threshold(mask, mask, (191 * trial) % 256, 255, cv::THRESH_BINARY);
        std::vector<std::vector<cv::Point>> paths;
        cv::findContours(mask.clone(), paths, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        for (auto& path : paths) {
            check(path);
            std::reverse(path.begin(), path.end());
            check(path);
        }
    }
    std::vector<cv::Point> repeated(128, {1, 0});
    check(repeated);
    for (int x = 2; x <= 23; ++x)
        repeated.emplace_back(x, 0);
    check(repeated);
    // Repeated extremes on a cross-connected outline force the reference tie rule.
    const std::vector<cv::Point> crossing{{0, 0}, {21, 19}, {0, 19}, {21, 0}};
    for (const auto& point : crossing)
        repeated.insert(repeated.end(), 40, point);
    check(repeated);
}

void check_nearest_guide() {
    cv::RNG rng(26371);
    cv::Mat actual, expected;
    for (int width : {4, 15, 32, 60, 64, 68, 121, 124, 1920, 3840}) {
        for (int height : {4, 13, 16, 28}) {
            cv::Mat storage(height + 8, width + 17, CV_8UC1);
            rng.fill(storage, cv::RNG::UNIFORM, 0, 256);
            const cv::Mat input = storage(cv::Rect(3, 4, width, height));
            const cv::Mat snapshot = storage.clone();
            for (int step : {2, 4, 5}) {
                for (const auto mode : {cv::INTER_NEAREST, cv::INTER_NEAREST_EXACT}) {
                    cv::resize(input, expected,
                               {(width + step - 1) / step, (height + step - 1) / step}, 0, 0, mode);
                    detail::resize_nearest_guide(input, actual, step, mode);
                    if (actual.size() != expected.size() ||
                        cv::norm(actual, expected, cv::NORM_INF) != 0)
                        throw std::runtime_error("Nearest guide differs from pinned OpenCV");
                }
            }
            if (cv::countNonZero(storage != snapshot))
                throw std::runtime_error("Nearest guide modified its input view");
        }
    }
}

void test_nearest_guide() {
    const int previous = cv::getNumThreads();
    for (int workers : {1, 4}) {
        cv::setNumThreads(workers);
        check_nearest_guide();
    }
    cv::setNumThreads(previous);
}

void test_guide_mask() {
    cv::RNG rng(83191);
    cv::Mat mask, mean, expected;
    std::vector<std::uint8_t> scratch;
    for (int width : {1, 3, 15, 16, 17, 31, 32, 64, 68, 191, 320}) {
        cv::Mat backing(62, width + 14, CV_8U);
        const cv::Mat input = backing(cv::Rect(3, 4, width, 51));
        for (int iteration = 0; iteration < 12; ++iteration) {
            rng.fill(backing, cv::RNG::UNIFORM, 0, 256);
            for (int window : {3, 15, 63})
                for (int difference : {1, 4, 36, 37, 255, 256}) {
                    cv::adaptiveThreshold(input, expected, 255, cv::ADAPTIVE_THRESH_MEAN_C,
                                          cv::THRESH_BINARY_INV, window, difference);
                    detail::local_ink_mask(input, window, difference, mean, mask);
                    if (cv::norm(mask, expected, cv::NORM_INF) != 0)
                        throw std::runtime_error("local proposal mask differs from adaptive mean");
                }
            cv::threshold(input, mask, 155 + 7 * iteration, 255, cv::THRESH_BINARY);
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            if (detail::exceeds_external_limit(mask, contours.size(), scratch))
                throw std::runtime_error("isolated witnesses overcounted exterior contours");
        }
    }
    for (int width : {4, 32, 64, 68, 124, 256}) {
        cv::Mat backing(28, width + 33, CV_8U);
        rng.fill(backing, cv::RNG::UNIFORM, 0, 256);
        const cv::Mat image = backing(cv::Rect(7, 3, width, 20));
        detail::sample_area4(image, mask);
        for (int y = 0; y < mask.rows; ++y)
            for (int x = 0; x < mask.cols; ++x) {
                const int sum = int(cv::sum(image(cv::Rect(4 * x + 1, 4 * y + 1, 2, 2)))[0]);
                if (mask.at<std::uint8_t>(y, x) != (sum + 2) / 4)
                    throw std::runtime_error(
                            "averaged guide differs from the native center pixels");
            }
    }
    // Compare both signs of the SIMD cap decision with an independent column-ray count.
    for (int width : {3, 17, 18, 35, 79}) {
        cv::Mat backing(66, width + 6, CV_8U);
        cv::Mat binary = backing(cv::Rect(2, 3, width, 60));
        for (int iteration = 0; iteration < 16; ++iteration) {
            rng.fill(backing, cv::RNG::UNIFORM, 0, 32);
            cv::threshold(binary, binary, 16 + iteration, 255, cv::THRESH_BINARY);
            std::size_t count = 0;
            for (int x = 1; x < width - 1; ++x) {
                int y = 0;
                while (y < 40 && binary.at<std::uint8_t>(y, x) == 0)
                    ++y;
                if (y == 0 || y == 40)
                    continue;
                if (cv::countNonZero(binary(cv::Rect(x - 1, y - 1, 3, 3))) == 1)
                    ++count;
            }
            for (std::size_t limit = 0; limit < 18; ++limit)
                if (detail::exceeds_external_limit(binary, limit, scratch) != (count > limit))
                    throw std::runtime_error(
                            "SIMD exterior contour limit disagreed with ray count");
        }
    }
    mask = cv::Mat::zeros(40, 80, CV_8U);
    for (int x = 3; x < 75; x += 3)
        mask.at<std::uint8_t>(11, x) = 255;
    if (!detail::exceeds_external_limit(mask, 12, scratch))
        throw std::runtime_error("clear column witnesses missed");
    cv::rectangle(mask, {1, 1}, {77, 35}, cv::Scalar(255));
    if (detail::exceeds_external_limit(mask, 1, scratch))
        throw std::runtime_error("enclosed islands counted as external");
}
void test_diffuse_crossing() {
    // An independent Gaussian edge on a linear illumination ramp. Exercise both
    // polarities, subpixel phases and a stride with poisoned unused columns.
    for (int sign : {-1, 1}) {
        for (float phase : {-.47F, -.12F, .19F, .48F}) {
            for (float shift : {-.19F, .16F}) {
                std::array<std::uint8_t, 45> line{};
                line.fill(247);
                for (int j = -4; j <= 4; ++j) {
                    const float fraction = .5F + .5F * std::erf((j + phase - shift) * .86F /
                                                                (.78F * std::sqrt(2.F)));
                    line[5 * (j + 4)] = cv::saturate_cast<std::uint8_t>(110.F + 1.5F * j +
                                                                        sign * 82.F * fraction);
                }
                float measured = 0, weight = 0;
                if (!detail::diffuse_crossing(line.data() + 20, 5, phase, sign * .86F, 8, measured,
                                              weight) ||
                    !std::isfinite(weight) || weight <= 0 || std::abs(measured - shift) > .075F)
                    throw std::runtime_error("diffuse native edge phase or polarity incorrect");
                if (detail::diffuse_crossing(line.data() + 20, 5, phase, -sign * .86F, 8, measured,
                                             weight))
                    throw std::runtime_error("accepted a reversed diffuse transition");
            }
        }
    }
}
void print_tag(cv::Mat& image, cv::Point origin, int black, int white,
               std::uint64_t code = 0xd5d628584ULL) {
    constexpr int pitch = 12;
    image(cv::Rect(origin - cv::Point(pitch, pitch), cv::Size(10 * pitch, 10 * pitch)))
            .setTo(white);
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x) {
            const bool payload = x > 0 && x < 7 && y > 0 && y < 7;
            const bool light = payload && (code >> (6 * (y - 1) + x - 1) & 1);
            image(cv::Rect(origin + cv::Point(x * pitch, y * pitch), cv::Size(pitch, pitch)))
                    .setTo(light ? white : black);
        }
}
bool scalar_quiet(const cv::Mat& image, const std::vector<Detection>& tags, int limit) {
    if (tags.size() > 3 || std::min(image.cols, image.rows) < 20)
        return false;
    std::vector<std::array<cv::Point2f, 4>> expanded;
    for (const auto& tag : tags) {
        auto q = tag.corners_px_;
        const cv::Point2f center = .25F * (q[0] + q[1] + q[2] + q[3]);
        for (auto& point : q) {
            const auto delta = point - center;
            point += 9.F / std::max(1.F, std::sqrt(delta.dot(delta))) * delta;
        }
        expanded.push_back(q);
    }
    for (int y = 0; y < image.rows; ++y) {
        std::vector<bool> masked(image.cols, false);
        for (const auto& q : expanded) {
            float left = float(image.cols), right = 0;
            for (int i = 0; i < 4; ++i) {
                auto a = q[i], b = q[(i + 1) % 4];
                if (a.y > b.y)
                    std::swap(a, b);
                if (y >= a.y && y < b.y) {
                    const float crossing = a.x + (b.x - a.x) * ((y - a.y) / (b.y - a.y));
                    left = std::min(left, crossing);
                    right = std::max(right, crossing);
                }
            }
            if (right > left) {
                const int first = std::max(0, cvFloor(left) - 4);
                const int last = std::min(image.cols - 4, cvCeil(right) + 4);
                for (int x = first; x < last; ++x)
                    masked[x] = true;
            }
        }
        const auto* row = image.ptr<std::uint8_t>(y);
        for (int x = 0; x < image.cols; ++x) {
            if (masked[x])
                continue;
            const int direction = x < image.cols - 4 ? 1 : -1;
            for (int offset : {1, 4})
                if (std::abs(row[x] - row[x + direction * offset]) > limit ||
                    std::abs(row[x] - image.ptr<std::uint8_t>(
                                              std::min(image.rows - 1, y + offset))[x]) > limit)
                    return false;
        }
    }
    return true;
}

void compare_quiet_native_pixels() {
    detail::IsolatedInk search;
    cv::RNG rng(0x892c7a);
    const std::vector<Detection> tags{{1, {{{7, 6}, {37, 8}, {31, 27}, {6, 22}}}, 0},
                                      {3, {{{25, 15}, {51, 10}, {57, 37}, {24, 32}}}, 0}};
    for (const auto size :
         {cv::Size(19, 24), cv::Size(20, 20), cv::Size(63, 35), cv::Size(64, 48), cv::Size(65, 49),
          cv::Size(68, 65), cv::Size(127, 51), cv::Size(135, 81)}) {
        cv::Mat storage(size.height + 5, size.width + 19, CV_8U);
        rng.fill(storage, cv::RNG::UNIFORM, 0, 256);
        cv::Mat image = storage(cv::Rect({7, 3}, size));
        for (int test = 0; test < 60; ++test) {
            const int limit = test % 7 == 0 ? 0 : (test % 9 == 0 ? 255 : 3);
            image.setTo(105);
            if (test % 4 != 0) {
                for (int y = 0; y < image.rows; ++y) {
                    auto* row = image.ptr<std::uint8_t>(y);
                    for (int x = 0; x < image.cols; ++x)
                        row[x] += (x + 2 * y) / 21 + rng.uniform(0, 2);
                }
            }
            if (test % 3) {
                const int y = (test % 5 ? (test * 17) % image.rows : image.rows - 1);
                const int x = (test % 6 ? (test * 16 + (test % 4)) % image.cols : image.cols - 1);
                image.at<std::uint8_t>(y, x) = 117;
            }
            const std::vector<Detection> footprints(tags.begin(), tags.begin() + test % 3);
            const bool expected = scalar_quiet(image, footprints, limit);
            if (search.quiet_outside(image, footprints, limit) != expected ||
                search.quiet_outside(image.clone(), footprints, limit) != expected)
                throw std::runtime_error(
                        "native contrast differs across bands, stride or decoded footprints");
        }
    }
}

void check_large_quiet_boundaries() {
    detail::IsolatedInk search;
    cv::Mat parent(2054, 3085, CV_8U, cv::Scalar(219));
    auto image = parent(cv::Rect(5, 3, 3072, 2049));
    image.setTo(147);
    if (!search.quiet_outside(image, {}, 3))
        throw std::runtime_error("A constant native ROI was rejected");
    // Include a 64-column group boundary, overlapping 32-row bands, the
    // backward four-pixel check and the final short band of a strided image.
    for (auto point :
         {cv::Point(64, 32), cv::Point(255, 35), cv::Point(3069, 1761), cv::Point(3071, 2048)}) {
        auto& pixel = image.at<std::uint8_t>(point);
        pixel = 150;
        if (!search.quiet_outside(image, {}, 3))
            throw std::runtime_error("A native contrast at the threshold was rejected");
        pixel = 151;
        if (search.quiet_outside(image, {}, 3))
            throw std::runtime_error("Native band bounds hid a contrast beyond the threshold");
        pixel = 147;
    }
}

void shifted_diffuse_corners() {
    cv::Mat storage(149, 177, CV_8UC1, cv::Scalar(154));
    cv::Mat image = storage(cv::Rect(3, 4, 163, 137));
    print_tag(image, {33, 19}, 133, 154);
    cv::GaussianBlur(image, image, {}, 1.2, 1.2, cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);
    const std::array<cv::Point2f, 4> truth{
            {{32.5F, 18.5F}, {128.5F, 18.5F}, {128.5F, 114.5F}, {32.5F, 114.5F}}};
    auto seed = truth;
    seed[0] += {-1.5F, 1.25F};
    seed[1] += {1.75F, 1.5F};
    seed[2] += {1.F, -1.25F};
    seed[3] += {-1.25F, -1.F};
    std::vector<Detection> tags{{0, seed, 0}};
    detail::refine_diffuse_tags(image, 8, tags, VideoConfig{}.maximum_diffuse_reseed_cells_);
    for (int i = 0; i < 4; ++i)
        if (cv::norm(tags.front().corners_px_[i] - truth[i]) > .18)
            throw std::runtime_error("Native rails failed to relocate an off-centre blurred tag");
    auto rotated = seed;
    std::rotate(rotated.begin(), rotated.begin() + 1, rotated.end());
    tags = {{1, seed, 0}, {0, rotated, 0}};
    detail::refine_diffuse_tags(image, 8, tags, VideoConfig{}.maximum_diffuse_reseed_cells_);
    if (tags[0].corners_px_ != seed || tags[1].corners_px_ != rotated)
        throw std::runtime_error("Diffuse recovery accepted the wrong printed ID or corner order");
    image.setTo(154);
    tags = {{0, seed, 0}};
    detail::refine_diffuse_tags(image, 8, tags, VideoConfig{}.maximum_diffuse_reseed_cells_);
    if (tags[0].corners_px_ != seed)
        throw std::runtime_error("Missing current-frame edges shifted a diffuse candidate");
}

void test_noisy_ink() {
    constexpr int width = 641, height = 359;
    cv::Mat storage(height + 8, width + 22, CV_8U, cv::Scalar(255));
    cv::Mat image = storage(cv::Rect(7, 3, width, height));
    cv::Mat paper(height, width, CV_8U, cv::Scalar(164));
    print_tag(paper, {200, 100}, 127, 164);
    cv::GaussianBlur(paper, paper, {}, 1.05);
    cv::RNG sequence(77329145);
    const auto expose = [&](bool print) {
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                image.at<std::uint8_t>(y, x) = cv::saturate_cast<std::uint8_t>(
                        (print ? int(paper.at<std::uint8_t>(y, x)) : 164) - 30 + 48. * x / width +
                        sequence.gaussian(10));
    };
    VideoConfig settings;
    ReadNoiseConfig noise;
    detail::NoisyInk detector;
    expose(true);
    const int rows = (height - 8) / 13 + 1, columns = (width - 8) / 17 + 1, count = rows * columns;
    if (!detail::NoisyInk::plausible(image, settings, noise, 13, 17, count, count / 2, 56))
        throw std::runtime_error("Distributed camera noise on a shaded field was missed");
    const auto found = detector.find(image, settings, noise);
    const auto& tags = found.ordinary_;
    if (tags.size() != 1 || tags[0].id_ != 0 ||
        cv::norm(tags[0].corners_px_[0] - cv::Point2f(199.5F, 99.5F)) > 1.6)
        throw std::runtime_error("Noisy source crop lost the print or changed its orientation");
    // Perturb four different rail intersections. Only the known oriented code
    // on this strided source crop can confirm the internal transition fit.
    cv::Mat averaged;
    cv::blur(image, averaged, {5, 5}, {-1, -1}, cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);
    const std::array<cv::Point2f, 4> truth{
            {{199.5F, 99.5F}, {295.5F, 99.5F}, {295.5F, 195.5F}, {199.5F, 195.5F}}};
    Detection shifted = tags.front();
    shifted.corners_px_ = truth;
    shifted.corners_px_[0] += {-1.1F, 1.3F};
    shifted.corners_px_[1] += {1.7F, 1.1F};
    shifted.corners_px_[2] += {1.4F, -1.3F};
    shifted.corners_px_[3] += {-1.4F, -1.2F};
    auto wrong = shifted;
    wrong.id_ = 1;
    const auto seed = shifted.corners_px_;
    if (detail::fit_noisy_grid(averaged, noise, wrong) || wrong.corners_px_ != seed)
        throw std::runtime_error("Unverified noisy interior fit changed a wrong ID");
    auto rotated = shifted;
    std::rotate(rotated.corners_px_.begin(), rotated.corners_px_.begin() + 1,
                rotated.corners_px_.end());
    const auto reversed = rotated.corners_px_;
    if (detail::fit_noisy_grid(averaged, noise, rotated) || rotated.corners_px_ != reversed)
        throw std::runtime_error("Noisy interior fit changed an unsupported corner order");
    auto blank = shifted;
    if (detail::fit_noisy_grid(cv::Mat(image.size(), CV_8U, cv::Scalar(127)), noise, blank) ||
        blank.corners_px_ != seed)
        throw std::runtime_error("A blank frame supplied a spurious internal-grid fit");
    if (!detail::fit_noisy_grid(averaged, noise, shifted))
        throw std::runtime_error("A noisy, displaced grid was rejected");
    for (int j = 0; j < 4; ++j)
        if (cv::norm(truth[j] - shifted.corners_px_[j]) > 1.15)
            throw std::runtime_error("A noisy interior fit did not correct the physical corner");
    ReadNoiseConfig disabled_fit = noise;
    disabled_fit.maximum_grid_shift_cells_ = 0;
    auto disabled = tags.front();
    if (detail::fit_noisy_grid(averaged, disabled_fit, disabled) ||
        disabled.corners_px_ != tags.front().corners_px_)
        throw std::runtime_error("A disabled internal-grid fit moved a noisy corner");
    expose(false);
    if (!detector.find(image, settings, noise).empty())
        throw std::runtime_error("Read-noise supplemental search reused a departed print");
    image.setTo(145);
    if (detail::NoisyInk::plausible(image, settings, noise, 13, 17, count, count / 2, 56))
        throw std::runtime_error("A flat image has no distributed high-pass signal");
    noise.enabled_ = false;
    expose(true);
    if (detail::NoisyInk::plausible(image, settings, noise, 13, 17, count, count / 2, 56) ||
        !detector.find(image, settings, noise).empty())
        throw std::runtime_error("Disabled read-noise search produced a candidate");
    noise.minimum_hull_retention_ = std::numeric_limits<float>::quiet_NaN();
    try {
        detail::NoisyInk::validate(noise);
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("Nonfinite noisy outline retention accepted");
}

void sheared_diffuse_corners() {
    cv::Mat paper(170, 170, CV_8UC1, cv::Scalar(158));
    print_tag(paper, {38, 36}, 63, 158);
    const std::array<cv::Point2f, 4> printed{
            {{37.5F, 35.5F}, {133.5F, 35.5F}, {133.5F, 131.5F}, {37.5F, 131.5F}}};
    const std::array<cv::Point2f, 4> truth{
            {{54.6F, 40.1F}, {147.4F, 71.7F}, {113.5F, 153.2F}, {28.6F, 124.7F}}};
    const auto projection = cv::getPerspectiveTransform(printed.data(), truth.data());
    cv::Mat storage(188, 212, CV_8UC1, cv::Scalar(17));
    auto image = storage(cv::Rect(7, 3, 181, 180));
    cv::warpPerspective(paper, image, projection, image.size(), cv::INTER_LINEAR,
                        cv::BORDER_REPLICATE);
    cv::GaussianBlur(image, image, {}, 1.60, 1.60, cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);
    auto seed = truth;
    seed[3] += {-3.9F, 3.8F};
    const Detection detection{0, seed, 0};
    std::vector<Detection> tags{detection};
    detail::refine_diffuse_tags(image, 8, tags, 0.F);
    if (cv::norm(tags.front().corners_px_[3] - truth[3]) < 4.)
        throw std::runtime_error("Severe-shear test must exercise the separate reseed");
    tags = {detection};
    detail::refine_diffuse_tags(image, 8, tags, VideoConfig{}.maximum_diffuse_reseed_cells_);
    for (int i = 0; i < 4; ++i)
        if (cv::norm(tags.front().corners_px_[i] - truth[i]) > .30)
            throw std::runtime_error(
                    "Internal transitions did not verify the displaced native corner");

    tags = {detection};
    detail::refine_diffuse_tags(image, 8, tags, .10F);
    if (tags.front().corners_px_ != seed)
        throw std::runtime_error("Native correction exceeded the configured cell displacement");
    auto rotated = seed;
    std::rotate(rotated.begin(), rotated.begin() + 1, rotated.end());
    tags = {{1, seed, 0}, {0, rotated, 0}};
    detail::refine_diffuse_tags(image, 8, tags, VideoConfig{}.maximum_diffuse_reseed_cells_);
    if (tags[0].corners_px_ != seed || tags[1].corners_px_ != rotated)
        throw std::runtime_error("Internal transition check accepted another payload or rotation");
    image.setTo(158);
    tags = {detection};
    detail::refine_diffuse_tags(image, 8, tags, VideoConfig{}.maximum_diffuse_reseed_cells_);
    if (tags.front().corners_px_ != seed)
        throw std::runtime_error("A blank frame provided support for a moved native corner");
    for (float invalid : {-.1F, 1.1F, std::numeric_limits<float>::quiet_NaN()}) {
        VideoConfig config;
        config.maximum_diffuse_reseed_cells_ = invalid;
        try {
            VideoDetector rejected(config);
        } catch (const std::invalid_argument&) {
            continue;
        }
        throw std::runtime_error("Invalid diffuse corner displacement was accepted");
    }
}

void test_noisy_inset_outline() {
    constexpr int width = 384, height = 320;
    const std::array<cv::Point2f, 4> raster{
            {{11.5F, 11.5F}, {107.5F, 11.5F}, {107.5F, 107.5F}, {11.5F, 107.5F}}};
    const std::array<cv::Point2f, 4> truth{
            {{149.103F, 88.283F}, {163.904F, 156.312F}, {113.892F, 168.079F}, {91.648F, 107.261F}}};
    cv::Mat print(120, 120, CV_8U);
    print_tag(print, {12, 12}, 136, 160);
    cv::Mat paper;
    cv::warpPerspective(print, paper, cv::getPerspectiveTransform(raster.data(), truth.data()),
                        {width, height}, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    cv::GaussianBlur(paper, paper, {}, .9);
    cv::Mat storage(height + 8, width + 22, CV_8U, cv::Scalar(255));
    cv::Mat image = storage(cv::Rect(7, 3, width, height));
    const auto expose = [&] {
        cv::RNG noise(17);
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                image.at<std::uint8_t>(y, x) = cv::saturate_cast<std::uint8_t>(
                        int(paper.at<std::uint8_t>(y, x)) - 16 + .09 * x +
                        4 * std::cos(.15 * x + .08 * y) + noise.gaussian(8));
    };
    expose();
    VideoConfig settings;
    settings.faint_guide_side_px_ = height / 4;
    detail::NoisyInk detector;
    cv::Mat guide;
    const auto found = detector.find(image, settings, settings.read_noise_, &guide);
    const auto& tags = found.outline_retries_;
    if (!found.ordinary_.empty() || tags.size() != 1 || tags.front().id_ != 0)
        throw std::runtime_error("Inset guide outline lost a source-supported print");
    for (int j = 0; j < 4; ++j)
        if (cv::norm(tags.front().corners_px_[j] - truth[j]) > 1.5)
            throw std::runtime_error("Outline retry moved a corner away from the printed rail");
    cv::Mat area4;
    cv::resize(image, area4, {width / 4, height / 4}, 0, 0, cv::INTER_AREA);
    if (guide.size() != area4.size() || cv::norm(guide, area4, cv::NORM_INF) != 0)
        throw std::runtime_error("Noisy guide does not match full quarter-scale block means");

    paper.setTo(160);
    expose();
    auto odd_width = image.colRange(0, width - 1);
    if (!detector.find(odd_width, settings, settings.read_noise_, &guide).empty() || !guide.empty())
        throw std::runtime_error("Odd-width empty frame reused the previous print or guide");
}

void sparse_native_rectangles() {
    VideoConfig settings;
    detail::IsolatedInk search;
    cv::Mat paper(720, 1280, CV_8UC1, cv::Scalar(150));
    print_tag(paper, {323, 216}, 140, 150);
    const std::vector<cv::Point2f> physical{
            {322.5F, 215.5F}, {418.5F, 215.5F}, {418.5F, 311.5F}, {322.5F, 311.5F}};
    cv::Mat storage(725, 1319, CV_8UC1, cv::Scalar(253));
    cv::Mat strided = storage(cv::Rect(9, 2, paper.cols, paper.rows));
    for (double angle : {18., 63., -29.}) {
        const auto transform = cv::getRotationMatrix2D({371.F, 264.F}, angle, .86);
        cv::warpAffine(paper, strided, transform, paper.size(), cv::INTER_LINEAR,
                       cv::BORDER_REPLICATE);
        cv::GaussianBlur(strided, strided, {}, 1.32, 1.32,
                         cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);
        bool local = false;
        const auto tags =
                search.find_faint(strided, settings, detail::FaintGuide::sparse, nullptr, &local);
        if (!local || tags.size() != 1 || tags.front().id_ != 0)
            throw std::runtime_error(
                    "Sparse native search failed to verify a rotated diffuse print");
        std::vector<cv::Point2f> expected;
        cv::transform(physical, expected, transform);
        for (std::size_t i = 0; i < expected.size(); ++i)
            if (cv::norm(expected[i] - tags.front().corners_px_[i]) > .7)
                throw std::runtime_error(
                        "Sparse native search altered corner orientation or location");
    }
    settings.sparse_faint_.guide_side_px_ = 0;
    bool local = true;
    if (!search.find_faint(strided, settings, detail::FaintGuide::sparse, nullptr, &local)
                 .empty() ||
        local)
        throw std::runtime_error("Disabled sparse proposals reused a previous detection");
    settings = {};
    strided.setTo(150);
    // A closed border without a valid payload cannot authorize a detection.
    cv::rectangle(strided, {300, 200}, {392, 279}, cv::Scalar(140), 9);
    if (!search.find_faint(strided, settings, detail::FaintGuide::sparse).empty())
        throw std::runtime_error("Sparse rectangle accepted an empty payload");
    for (int selector = 0; selector < 6; ++selector) {
        settings = {};
        auto& config = settings.sparse_faint_;
        switch (selector) {
            case 0:
                config.guide_side_px_ = 31;
                break;
            case 1:
                config.guide_side_px_ = 1025;
                break;
            case 2:
                config.maximum_initial_step_px_ = 0;
                break;
            case 3:
                config.maximum_changed_per_mille_ = 1001;
                break;
            case 4:
                config.maximum_box_aspect_ratio_ = .9F;
                break;
            case 5:
                config.minimum_hull_fill_ = std::numeric_limits<float>::quiet_NaN();
                break;
        }
        try {
            VideoDetector invalid(settings);
        } catch (const std::invalid_argument&) {
            continue;
        }
        throw std::runtime_error("Invalid sparse proposal configuration accepted");
    }
}

void structured_prints() {
    cv::Mat padded(725, 1293, CV_8UC1, cv::Scalar(250));
    cv::Mat image = padded(cv::Rect(4, 3, 1280, 720));
    for (int y = 0; y < image.rows; ++y) {
        auto* row = image.ptr<std::uint8_t>(y);
        for (int x = 0; x < image.cols; ++x)
            row[x] = 104 + (x + 2 * y) / 40;
    }
    for (int y = 51; y < 660; y += 64)
        for (int x = 40; x < 575; x += 83)
            image(cv::Rect(x, y, 49, 3)) -= 8;
    const auto blank = image.clone();
    print_tag(image, {227, 446}, 28, 130);
    print_tag(image, {822, 228}, 132, 147);
    VideoConfig settings;
    VideoDetector video(settings);
    const auto tags = video.detect(image);
    const auto at = [&](cv::Point2f corner) {
        return std::count_if(tags.begin(), tags.end(), [&](const Detection& tag) {
                   return tag.id_ == 0 && cv::norm(tag.corners_px_[0] - corner) < 1.5;
               }) == 1;
    };
    if (tags.size() != 2 || !at({226.5F, 445.5F}) || !at({821.5F, 227.5F}))
        throw std::runtime_error(
                "Separated faint and strong prints must retain both native locations");
    detail::IsolatedInk search;
    const auto strong = std::find_if(tags.begin(), tags.end(), [](const Detection& tag) {
        return tag.corners_px_[0].x < 400;
    });
    std::vector<Detection> explained{*strong};
    const auto additional =
            search.find_faint(image, settings, detail::FaintGuide::adaptive_fallback, &explained);
    if (additional.size() != 1 || additional.front().id_ != 0 ||
        cv::norm(additional.front().corners_px_[0] - cv::Point2f{821.5F, 227.5F}) > 1.5 ||
        !search.find_faint(image, settings, detail::FaintGuide::adaptive_fallback, &tags).empty())
        throw std::runtime_error(
                "Adaptive supplemental search repeated or suppressed a separate print");
    if (!video.detect(blank).empty())
        throw std::runtime_error("Structured blank retained a removed print");
    settings.maximum_isolated_regions_ = 0;
    if (!search.find_faint(image, settings, detail::FaintGuide::adaptive_fallback, &explained)
                 .empty())
        throw std::runtime_error("Disabled faint search returned adaptive proposals");
    for (int invalid : {-1, 1001}) {
        settings.faint_minimum_moderate_per_mille_ = invalid;
        try {
            VideoDetector rejected(settings);
        } catch (const std::invalid_argument&) {
            continue;
        }
        throw std::runtime_error("Invalid moderate density accepted");
    }
}

void test_noisy_retry_preserves_regular_observation() {
    constexpr std::uint64_t word =
            0x0000000e83be4b73ULL;  // tag36h11 ID 586 in canonical orientation.
    const std::array<cv::Point2f, 4> source{
            {{11.5F, 11.5F}, {107.5F, 11.5F}, {107.5F, 107.5F}, {11.5F, 107.5F}}};
    const std::array<cv::Point2f, 4> truth{{{172.1551666F, 235.9816437F},
                                            {150.1733856F, 177.1141968F},
                                            {205.0955505F, 154.8029022F},
                                            {221.0148163F, 226.0267792F}}};
    cv::Mat print(120, 120, CV_8U);
    print_tag(print, {12, 12}, 136, 160, word);
    cv::Mat ink;
    cv::compare(print, 136, ink, cv::CMP_EQ);
    const auto projection = cv::getPerspectiveTransform(source.data(), truth.data());
    cv::Mat weak, native_ink;
    const cv::Size frame{387, 321};
    cv::warpPerspective(print, weak, projection, frame, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    cv::warpPerspective(ink, native_ink, projection, frame);
    cv::GaussianBlur(weak, weak, {}, .8);
    cv::RNG noise(862170);
    for (int y = 0; y < weak.rows; ++y)
        for (int x = 0; x < weak.cols; ++x) {
            auto& pixel = weak.at<std::uint8_t>(y, x);
            pixel = cv::saturate_cast<std::uint8_t>(
                    pixel - 16 + .09 * x + 4 * std::cos(.15 * x + .08 * y) + noise.gaussian(11));
        }
    cv::Mat strong;
    cv::subtract(weak, native_ink * (24. / 255.), strong);
    VideoConfig config;
    config.faint_guide_side_px_ = 80;
    VideoDetector detector(config);
    const auto cold = detector.detect(weak);
    auto ordinary_config = config;
    ordinary_config.read_noise_.enabled_ = false;
    VideoDetector ordinary(ordinary_config);
    detector.reset();
    (void)ordinary.detect(strong);
    (void)detector.detect(strong);
    const auto preexisting = ordinary.detect(weak);
    const auto warm = detector.detect(weak);
    if (cold.size() != 1 || warm.size() != 1 || preexisting.size() != 1 || cold[0].id_ != 586 ||
        warm[0].id_ != 586 || preexisting[0].id_ != 586)
        throw std::runtime_error("Expected an oriented weak print on first and second exposures");
    for (int j = 0; j < 4; ++j) {
        if (warm[0].corners_px_[j] != preexisting[0].corners_px_[j])
            throw std::runtime_error("Noisy outline overwrite of an existing ordinary observation");
        if (cv::norm(cold[0].corners_px_[j] - truth[j]) > 1.6)
            throw std::runtime_error("Cold weak outline slipped past the native rail");
    }
}

void test_noisy_small_tag_beside_strong_print() {
    constexpr int paper = 163;
    const std::array<cv::Point2f, 4> raster{
            {{11.5F, 11.5F}, {107.5F, 11.5F}, {107.5F, 107.5F}, {11.5F, 107.5F}}};
    const std::array<cv::Point2f, 4> strong{{{134.0903931F, 139.6194916F},
                                             {172.8763733F, 149.3993988F},
                                             {164.5097046F, 175.2185516F},
                                             {126.8872910F, 165.7320251F}}};
    const std::array<cv::Point2f, 4> small{{{188.1331329F, 147.1181335F},
                                            {201.1645355F, 139.6896667F},
                                            {207.7183838F, 152.5497589F},
                                            {195.0779114F, 159.7553864F}}};
    cv::Mat image(320, 480, CV_8UC1, cv::Scalar(paper));
    const auto render = [&](const auto& corners, int ink, std::uint64_t word, cv::Rect printed) {
        cv::Mat print(120, 120, CV_8U), mask(120, 120, CV_8U, cv::Scalar(0));
        print_tag(print, {12, 12}, ink, paper, word);
        mask(printed).setTo(255);
        const auto projection = cv::getPerspectiveTransform(raster.data(), corners.data());
        cv::Mat samples, coverage;
        cv::warpPerspective(print, samples, projection, image.size(), cv::INTER_LINEAR,
                            cv::BORDER_REPLICATE);
        cv::warpPerspective(mask, coverage, projection, image.size());
        samples.copyTo(image, coverage);
    };
    render(strong, 30, 0x5246149aeULL, {0, 0, 120, 120});
    render(small, 95, 0x4888964aaULL, {12, 12, 96, 96});
    cv::GaussianBlur(image, image, {}, .36F);
    cv::Mat noise(image.size(), CV_16S);
    cv::RNG random(12370711509787580681ULL);
    random.fill(noise, cv::RNG::NORMAL, 0, 6);
    cv::add(image, noise, image, {}, CV_8U);
    VideoDetector detector;
    const auto tags = detector.detect(image);
    const auto target = std::find_if(tags.begin(), tags.end(),
                                     [](const Detection& tag) { return tag.id_ == 376; });
    if (tags.size() != 2 || target == tags.end())
        throw std::runtime_error("A strong tag hid a separate 15-pixel print");
    for (int j = 0; j < 4; ++j)
        if (cv::norm(target->corners_px_[j] - small[j]) > 2)
            throw std::runtime_error("The small print's corners or rotation changed");
}

void test_weak_tag_after_blank_noisy_frame() {
    constexpr int kId = 95;
    constexpr int kPaper = 185;
    constexpr std::uint64_t kWord = 0x24d4d5937ULL;
    const std::array<cv::Point2f, 4> corners{{{104.959869F, .399278641F},
                                              {125.022102F, 8.08088303F},
                                              {120.775238F, 33.0145874F},
                                              {95.8135223F, 23.4570255F}}};
    const std::array<cv::Point2f, 4> cells{{{0, 0}, {8, 0}, {8, 8}, {0, 8}}};
    const cv::Matx33d projection = cv::getPerspectiveTransform(corners.data(), cells.data());
    cv::Mat blank(384, 640, CV_8UC1, cv::Scalar(kPaper));
    auto frame = blank.clone();
    const cv::Rect region = (cv::boundingRect(corners) + cv::Size(6, 6) - cv::Point(3, 3)) &
                            cv::Rect({}, frame.size());
    for (int y = region.y; y < region.br().y; ++y) {
        auto* row = frame.ptr<std::uint8_t>(y);
        for (int x = region.x; x < region.br().x; ++x) {
            int ink_samples = 0;
            for (float dy : {-.3333F, 0.F, .3333F}) {
                for (float dx : {-.3333F, 0.F, .3333F}) {
                    const auto sample = projection * cv::Vec3d(x + dx, y + dy, 1);
                    const int u = cvFloor(sample[0] / sample[2]);
                    const int v = cvFloor(sample[1] / sample[2]);
                    if (u >= 0 && u < 8 && v >= 0 && v < 8) {
                        ink_samples += u == 0 || v == 0 || u == 7 || v == 7 ||
                                       ((kWord >> (6 * (v - 1) + u - 1)) & 1) == 0;
                    }
                }
            }
            row[x] = cv::saturate_cast<std::uint8_t>(kPaper - ink_samples * 26.F / 9);
        }
    }
    cv::GaussianBlur(frame, frame, {}, .32);
    cv::RNG random(7703702696976432145ULL);
    cv::Mat noise(frame.size(), CV_16S);
    for (auto* image : {&frame, &blank}) {
        random.fill(noise, cv::RNG::NORMAL, 0, 2);
        cv::add(*image, noise, *image, {}, CV_8U);
    }
    VideoDetector detector;
    if (!detector.detect(blank).empty())
        throw std::runtime_error("A blank noisy frame produced a tag");
    const auto tags = detector.detect(frame);
    if (tags.size() != 1 || tags.front().id_ != kId)
        throw std::runtime_error("Missed the weak tag entering a stationary noisy view");
    for (int i = 0; i < 4; ++i) {
        if (cv::norm(corners[i] - tags.front().corners_px_[i]) > 1.5)
            throw std::runtime_error("The entering tag has incorrect corner order or position");
    }
}

}  // namespace

int main() {
    test_binary_guide_median();
    cv::setNumThreads(1);
    test_diffuse_crossing();
    test_noisy_ink();
    test_noisy_crop_filter();
    test_ordered_guide_hull();
    test_nearest_guide();
    test_guide_mask();
    test_guide_contours();
    test_fixed_sampled_contours();
    test_local_contour_proof();
    shifted_diffuse_corners();
    sheared_diffuse_corners();
    sparse_native_rectangles();
    structured_prints();
    compare_quiet_native_pixels();
    check_large_quiet_boundaries();
    test_noisy_inset_outline();
    test_noisy_retry_preserves_regular_observation();
    test_noisy_small_tag_beside_strong_print();
    test_weak_tag_after_blank_noisy_frame();
    cv::Mat storage(721, 1317, CV_8UC1, cv::Scalar(0));
    cv::Mat image = storage(cv::Rect(7, 1, 1280, 720));
    for (int x = 0; x < image.cols; ++x)
        image.col(x).setTo(120 + 40 * x / image.cols);
    print_tag(image, {200, 200}, 24, 230);
    VideoConfig settings;
    Detector native, patch;
    detail::IsolatedInk search;
    std::vector<Detection> tags;
    if (!search.find(image, 125, settings, native, patch, tags) || tags.size() != 1 ||
        tags[0].id_ != 0)
        throw std::runtime_error("isolated source tag not confirmed");
    // Both levels of the second tag are above the guide's dark cutoff. The residual
    // variation must request ordinary discovery, even though the first tag decoded.
    print_tag(image, {800, 440}, 123, 164);
    tags.clear();
    if (search.find(image, 125, settings, native, patch, tags) || !tags.empty())
        throw std::runtime_error("weak second target was silently skipped");
    image.setTo(130);
    if (!search.quiet_outside(image, {}, settings.faint_background_difference_))
        throw std::runtime_error("constant native background rejected");
    // A period-four pattern aliases on a four-pixel grid. Check every source phase.
    for (int x = 0; x < image.cols; x += 4)
        image.col(x).setTo(142);
    if (search.quiet_outside(image, {}, settings.faint_background_difference_))
        throw std::runtime_error("periodic horizontal ink skipped in native background");
    image.setTo(130);
    for (int y = 0; y < image.rows; y += 4)
        image.row(y).setTo(142);
    if (search.quiet_outside(image, {}, settings.faint_background_difference_))
        throw std::runtime_error("periodic vertical ink skipped in native background");
    image.setTo(138);
    print_tag(image, {544, 288}, 124, 138);
    tags = search.find_faint(image, settings);
    if (tags.size() != 1 || tags.front().id_ != 0)
        throw std::runtime_error("low-contrast native ink not confirmed");
    image.at<std::uint8_t>(image.rows - 1, image.cols - 1) = 119;
    if (search.quiet_outside(image, tags, settings.faint_background_difference_))
        throw std::runtime_error("unexplained signal on final row skipped");
    // Isolated sensor fluctuations add hundreds of raw proposals on this matte field.
    // The print location and payload still have to agree with the original strided image.
    image.setTo(138);
    print_tag(image, {540, 300}, 118, 138);
    cv::RNG noise(924357);
    for (int y = 0; y < image.rows; ++y) {
        auto* pixels = image.ptr<std::uint8_t>(y);
        for (int x = 0; x < image.cols; ++x)
            pixels[x] = cv::saturate_cast<std::uint8_t>(pixels[x] + noise.gaussian(.9));
    }
    tags = search.find_faint(image, settings);
    if (tags.size() != 1 || tags.front().id_ != 0 ||
        cv::norm(tags.front().corners_px_[0] - cv::Point2f(539.5F, 299.5F)) > .25)
        throw std::runtime_error("faint native tag lost in sensor noise");
    settings.maximum_isolated_regions_ = 0;
    if (search.find(image, 125, settings, native, patch, tags))
        throw std::runtime_error("disabled guide accepted a frame");
}
