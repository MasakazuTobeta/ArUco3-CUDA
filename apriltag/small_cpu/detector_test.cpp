// SPDX-License-Identifier: Apache-2.0
#include "aruco3cuda/apriltag/small_cpu/detector.hpp"
#include "aruco3cuda/apriltag/small_cpu/video_detector.hpp"

#include "aruco3cuda/dictionary.hpp"
#include "automatic_search.hpp"
#include "border_search.hpp"
#include "bounded_quad_approximation.hpp"
#include "coarse_mask.hpp"
#include "component_candidates.hpp"
#include "cubic_patch.hpp"
#include "curved_edges.hpp"
#include "dense_board_search.hpp"
#include "edge_curvature.hpp"
#include "geometry.hpp"
#include "grid_alignment.hpp"
#include "guide_average.hpp"
#include "kalibr_grid_search.hpp"
#include "large_board_search.hpp"
#include "local_payload_fit.hpp"
#include "luma_distribution.hpp"
#include "noisy_print_search.hpp"
#include "planar_lattice.hpp"
#include "planar_tag_shapes.hpp"
#include "small_tag_refinement.hpp"
#include "uniform_raster.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#if defined(__x86_64__) && defined(__GNUC__)
#include <xmmintrin.h>
#endif

namespace aruco3cuda::generated {
extern const DictionaryTable kDictApriltag36h11Table;
}

namespace {
using aruco3cuda::apriltag_small_cpu::CoarseConfig;
using aruco3cuda::apriltag_small_cpu::ComponentConfig;
using aruco3cuda::apriltag_small_cpu::Config;
using aruco3cuda::apriltag_small_cpu::Detector;

constexpr int kTagId = 17;
constexpr int kCells = 8;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void check_cubic_patch() {
    using aruco3cuda::apriltag_small_cpu::detail::enlarge_patch_cubic;
    const int previous_threads = cv::getNumThreads();
    const bool previous_optimized = cv::useOptimized();
    cv::RNG rng(58365932);
    cv::Mat expected, enlarged;
    std::vector<std::int32_t> rows;
    for (const bool optimized : {true, false}) {
        cv::setUseOptimized(optimized);
        cv::setNumThreads(1);
        for (int width : {1, 2, 3, 4, 5, 7, 8, 9, 28, 35, 67, 121, 257, 16383}) {
            for (int height : {1, 3, 7, 13, 64}) {
                cv::Mat storage(height + 5, width + 9, CV_8UC1, cv::Scalar(255));
                const cv::Mat crop = storage(cv::Rect(4, 2, width, height));
                rng.fill(crop, cv::RNG::UNIFORM, 0, 256);
                cv::resize(crop, expected, {}, 2., 2., cv::INTER_CUBIC);
                enlarge_patch_cubic(crop, enlarged, rows);
                require(expected.size() == enlarged.size() && expected.type() == enlarged.type() &&
                                cv::norm(expected, enlarged, cv::NORM_INF) == 0,
                        "Cubic patch differs from OpenCV");
            }
        }
    }
    cv::setUseOptimized(previous_optimized);
    cv::setNumThreads(previous_threads);
}

template <class Function>
void require_invalid(Function func) {
    try {
        func();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("Expected std::invalid_argument");
}

void put_tag(cv::Mat& frame, cv::Point origin, int cell_px, int rotation, int flips, int dark,
             int light, int id = kTagId, int border = 1) {
    const auto word = aruco3cuda::generated::kDictApriltag36h11Table.codes_[4 * id];
    const int size = (6 + 2 * border) * cell_px;
    cv::Mat marker(size, size, CV_8UC1, cv::Scalar(dark));
    constexpr std::array<int, 4> kFlipCells = {0, 8, 14, 29};
    for (int y = 0; y < 6; ++y) {
        for (int x = 0; x < 6; ++x) {
            const int bit = 6 * y + x;
            bool white = ((word >> bit) & 1U) != 0;
            for (int k = 0; k < flips; ++k) {
                white ^= bit == kFlipCells[k];
            }
            marker(cv::Rect((x + border) * cell_px, (y + border) * cell_px, cell_px, cell_px))
                    .setTo(white ? light : dark);
        }
    }
    for (int k = 0; k < rotation; ++k) {
        cv::rotate(marker, marker, cv::ROTATE_90_CLOCKWISE);
    }
    marker.copyTo(frame(cv::Rect(origin, marker.size())));
}

void check_small_tag_refinement() {
    using namespace aruco3cuda::apriltag_small_cpu;
    cv::Mat storage(143, 171, CV_8UC1, cv::Scalar(207));
    auto frame = storage(cv::Rect(9, 8, 141, 117));
    put_tag(frame, {31, 24}, 5, 0, 0, 32, 207);
    cv::GaussianBlur(frame, frame, {5, 5}, .6, .6, cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);

    const detail::Quad expected{{{30.5F, 23.5F}, {70.5F, 23.5F}, {70.5F, 63.5F}, {30.5F, 63.5F}}};
    auto biased = expected;
    for (auto& p : biased)
        p += cv::Point2f(.375F, -.375F);
    std::vector<Detection> tags{{kTagId, biased, 0}, {kTagId + 1, biased, 2}};
    SmallTagRefinementConfig config;
    detail::refine_small_tag_corners(frame, tags, config);
    require(tags[0].corners_px_ == biased, "Small-tag refinement ignored its frame area gate");
    config.minimum_frame_pixels_ = frame.total();
    detail::refine_small_tag_corners(frame, tags, config);
    require(tags[0].id_ == kTagId && tags[0].corrected_bits_ == 0 && tags[1].id_ == kTagId + 1 &&
                    tags[1].corners_px_ == biased && tags[1].corrected_bits_ == 2,
            "Small-tag refinement accepted a mismatched ID");
    for (std::size_t i = 0; i < expected.size(); ++i)
        require(detail::length(tags[0].corners_px_[i] - expected[i]) < .20F,
                "Small-tag aperture fit did not recover the biased corner");

    auto coarse = expected;
    for (auto& p : coarse)
        p.x += 2.75F;
    std::vector<Detection> corrected{{kTagId, coarse, 3}, {kTagId + 1, coarse, 3}};
    detail::refine_small_tag_corners(frame, corrected, config);
    require(corrected[0].corrected_bits_ == 0 && corrected[1].corrected_bits_ == 3 &&
                    corrected[1].corners_px_ == coarse,
            "Signed boundary retry did not verify only the original oriented ID");
    for (std::size_t i = 0; i < expected.size(); ++i)
        require(detail::length(corrected[0].corners_px_[i] - expected[i]) < .20F,
                "Signed boundary retry did not recover the biased corner");

    config.enabled_ = false;
    tags[0].corners_px_ = biased;
    detail::refine_small_tag_corners(frame, tags, config);
    require(tags[0].corners_px_ == biased, "Disabled small-tag refinement changed a corner");

    VideoConfig bad;
    bad.small_tag_refinement_.minimum_side_px_ = 100.F;
    require_invalid([&] { VideoDetector rejected(bad); });
}

void require_corners(const std::array<cv::Point2f, 4>& corners, cv::Point origin, int side_px,
                     int rotation) {
    const float left = origin.x - 0.5F;
    const float top = origin.y - 0.5F;
    const std::array<cv::Point2f, 4> expected{{{left, top},
                                               {left + side_px, top},
                                               {left + side_px, top + side_px},
                                               {left, top + side_px}}};
    for (int k = 0; k < 4; ++k) {
        const cv::Point2f d = corners[k] - expected[(k + rotation) % 4];
        require(d.dot(d) < 1.0F, "Incorrect ordered corner");
    }
}

void check_local_payload_fit() {
    using namespace aruco3cuda::apriltag_small_cpu;
    using namespace aruco3cuda::apriltag_small_cpu::detail;
    constexpr int id = 1, cell = 9;
    const cv::Point origin{22, 18};
    const Point first = Point(origin) - Point(.5F, .5F);
    const Quad corners{first, first + Point(8 * cell, 0), first + Point(8 * cell, 8 * cell),
                       first + Point(0, 8 * cell)};
    const auto* codes = aruco3cuda::generated::kDictApriltag36h11Table.codes_;
    cv::Mat storage(125, 160, CV_8UC1, cv::Scalar(5));
    cv::Mat image = storage(cv::Rect(4, 3, 140, 115));
    for (int rotation = 0; rotation < 4; ++rotation) {
        image.setTo(110);
        put_tag(image, origin, cell, rotation, 0, 45, 110, id);
        for (int y = 0; y < image.rows; ++y) {
            auto* row = image.ptr<std::uint8_t>(y);
            for (int x = 0; x < image.cols; ++x) {
                const float dx = (x - origin.x - 4.7F * cell) / (1.4F * cell);
                const float dy = (y - origin.y - 3.6F * cell) / (1.6F * cell);
                row[x] += static_cast<std::uint8_t>(40 * std::exp(-.5F * (dx * dx + dy * dy)));
            }
        }
        // Compare oriented corner orders using the canonical word, independent of table rotation.
        Quad canonical{};
        for (int k = 0; k < 4; ++k)
            canonical[k] = corners[(k + rotation) % 4];
        const cv::Mat before = storage.clone();
        require(matches_local_payload(image, canonical, codes[4 * id], 15.F) &&
                        matches_local_payload(image.clone(), canonical, codes[4 * id], 15.F),
                "Local word check missed a strided tag under nonuniform illumination");
        require(!matches_local_payload(image, canonical, codes[4 * id + 1], 15.F) &&
                        !matches_local_payload(image, canonical, codes[4 * (id + 1)], 15.F) &&
                        !matches_local_payload(image, canonical, codes[4 * id] ^ 1U, 15.F),
                "Local word check accepted a different payload or orientation");
        require(!matches_local_payload(image, canonical, codes[4 * id], 255.F),
                "Local word check ignored the requested contrast");
        require(cv::norm(storage, before, cv::NORM_INF) == 0,
                "Local word check changed source pixels");
        put_tag(image, origin, cell, rotation, 1, 45, 110, id);
        require(!matches_local_payload(image, canonical, codes[4 * id], 15.F),
                "Local word check accepted a damaged word as exact");
    }
    Quad clipped = corners;
    for (auto& point : clipped)
        point.x -= origin.x + cell;
    require(!matches_local_payload(image, clipped, codes[4 * id], 15.F) &&
                    !matches_local_payload(image, Quad{}, codes[4 * id], 15.F),
            "Local word check accepted missing source evidence");
    image.setTo(100);
    require(!matches_local_payload(image, corners, codes[4 * id], 15.F),
            "Local word check accepted a uniform image");
    Config config;
    config.components_.emplace();
    auto& fit = config.components_->local_payload_fit_.emplace();
    fit.minimum_side_px_ = 7;
    require_invalid([&] { Detector invalid(config); });
    fit = {};
    fit.minimum_displacement_px_ = std::numeric_limits<float>::quiet_NaN();
    require_invalid([&] { Detector invalid(config); });
    VideoConfig video;
    video.local_payload_fit_.maximum_displacement_cells_ = -1;
    require_invalid([&] { VideoDetector invalid(video); });
    video.local_payload_fit_.maximum_displacement_cells_ = 0;
    VideoDetector disabled(video);
}

// A smooth light pool shifts one native bit across the border-plane threshold.
void check_component_fit_under_uneven_light() {
    using namespace aruco3cuda::apriltag_small_cpu;
    using detail::Quad;
    constexpr int cell = 9, id = 1;
    cv::Mat sheet(108, 108, CV_8UC1, cv::Scalar(146));
    put_tag(sheet, {18, 18}, cell, 0, 0, 73, 146, id);
    for (int y = 0; y < sheet.rows; ++y) {
        for (int x = 0; x < sheet.cols; ++x) {
            const double u = (x + .5) / cell - 2, v = (y + .5) / cell - 2;
            const auto pool = [&](double a, double b, double cx, double cy) {
                return std::exp(-a * (u - cx) * (u - cx) - b * (v - cy) * (v - cy));
            };
            const double illumination = pool(.12, .23, 4.9, 4.2) - .37 * pool(.25, .14, 1.7, 5.8) +
                                        .10 * std::sin((u + .82 * v) / 1.19);
            auto& pixel = sheet.at<std::uint8_t>(y, x);
            pixel = cv::saturate_cast<std::uint8_t>(pixel + 72 * illumination);
        }
    }
    auto transform = cv::getRotationMatrix2D({53.5F, 53.5F}, 22, .65);
    transform.at<double>(0, 2) += 11.05;
    transform.at<double>(1, 2) += 12.27;
    cv::Mat frame;
    cv::warpAffine(sheet, frame, transform, {131, 129}, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 146);
    const Quad printed{{{17.5, 17.5}, {89.5, 17.5}, {89.5, 89.5}, {17.5, 89.5}}};
    Quad truth;
    cv::transform(printed, truth, transform);
    Config settings;
    auto& component = settings.components_.emplace();
    component.threshold_ = 104;
    const auto original = Detector(settings).detect(frame);
    component.local_payload_fit_.emplace();
    const auto fitted = Detector(settings).detect(frame);
    require(original.size() == 1 && fitted.size() == 1 && original[0].id_ == id &&
                    fitted[0].id_ == id && original[0].corrected_bits_ == 2 &&
                    fitted[0].corrected_bits_ == 0,
            "Exact local payload did not recover the supported component fit");
    for (int k = 0; k < 4; ++k)
        require(detail::length(fitted[0].corners_px_[k] - truth[k]) < .15F,
                "Local fit missed the physical position or orientation");
    settings.maximum_corrected_bits_ = 2;
    const auto restricted = Detector(settings).detect(frame);
    require(restricted.size() == 1 && restricted[0].corners_px_ == original[0].corners_px_ &&
                    restricted[0].corrected_bits_ == 2,
            "Local fit exceeded the configured native-read Hamming limit");
}

void check_edge_fit_bounds() {
    using namespace aruco3cuda::apriltag_small_cpu::detail;
    struct FitSettings {
        bool uniform;
        int stride;
    };
    constexpr std::array<FitSettings, 5> kFits{
            {{true, 0}, {false, 0}, {false, 2}, {false, 3}, {false, 4}}};
    for (const int side : {29, 93}) {
        cv::Mat storage(side + 42, side + 64, CV_8UC1, cv::Scalar(3));
        cv::Mat frame = storage(cv::Rect(9, 6, side + 16, side + 16));
        for (const cv::Point margin : {cv::Point{2, 2}, {3, 5}, {7, 8}, {11, 13}, {14, 14}}) {
            frame.setTo(220);
            frame(cv::Rect(margin.x, margin.y, side, side)).setTo(26);
            const Point left_top = Point(margin) - Point(.5F, .5F);
            const Quad reference = {left_top, left_top + Point(side, 0),
                                    left_top + Point(side, side), left_top + Point(0, side)};
            const auto dense = frame.clone();
            for (const auto offset : {Point{-.35F, .45F}, Point{.55F, -.6F}}) {
                auto initial = reference;
                for (auto& corner : initial) {
                    corner += offset;
                }
                for (const auto& fit : kFits) {
                    auto strided_corners = initial;
                    auto dense_corners = initial;
                    require(refine_quad(frame, strided_corners, fit.uniform, fit.stride) &&
                                    refine_quad(dense, dense_corners, fit.uniform, fit.stride),
                            "Edge fit failed beside a strided ROI boundary");
                    require(strided_corners == dense_corners, "Edge fit read outside the ROI");
                    for (int j = 0; j < 4; ++j) {
                        require(length(strided_corners[j] - reference[j]) < .75F,
                                "Edge fit displaced the measured edge");
                    }
                }
            }
        }
    }

    const cv::Mat gray(45, 45, CV_8UC1, cv::Scalar(220));
    for (const float invalid :
         {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        Quad corners{{{0.F, 0.F}, {invalid, 0.F}, {20.F, 20.F}, {0.F, 20.F}}};
        require(!refine_quad(gray, corners, false), "Edge fit accepted a nonfinite quadrilateral");
    }
}

void check_two_border_prints() {
    using namespace aruco3cuda::apriltag_small_cpu;
    detail::KalibrGridSearch search;
    const cv::Point standard_origin(37, 23), wide_origin(209, 73);
    cv::Mat storage(223, 421, CV_8UC1, cv::Scalar(3));
    cv::Mat frame = storage(cv::Rect(11, 9, 391, 199));
    for (int rotation = 0; rotation < 4; ++rotation) {
        frame.setTo(230);
        put_tag(frame, standard_origin, 5, rotation, 0, 12, 236, 101);
        require(!search.has_candidate(frame) && search.detect(frame).empty(),
                "Two-border supplement accepted an ordinary tag");
        const auto ordinary = Detector{}.detect(frame);
        require(ordinary.size() == 1 && ordinary[0].id_ == 101,
                "Missing ordinary tag on the mixed-print fixture");
        require(search.uncovered_region(frame, ordinary).empty(),
                "The current ordinary tag must account for its ink");

        const int cell = 3 + rotation;
        const int id = 200 + rotation;
        put_tag(frame, wide_origin, cell, rotation, 0, rotation % 2 ? 60 : 12, 235, id, 2);
        const auto roi = search.uncovered_region(frame, ordinary);
        require(roi.contains(wide_origin) && search.has_candidate(frame(roi)),
                "Missed current two-border tag outside the ordinary marker");
        const auto additions = search.detect(frame(roi));
        require(additions.size() == 1 && additions[0].id_ == id,
                "Incorrect sparse two-border addition");
        require_corners(additions[0].corners_px_, wide_origin - roi.tl(), 10 * cell, rotation);
        require(search.detect(frame).size() == 1,
                "Global discovery must distinguish one- and two-border geometries");
        const auto mixed = VideoDetector{}.detect(frame);
        const auto has_id = [&](int expected) {
            return std::any_of(mixed.begin(), mixed.end(),
                               [&](const Detection& tag) { return tag.id_ == expected; });
        };
        require(mixed.size() == 2 && has_id(101) && has_id(id),
                "The first video frame must report both print conventions");
        VideoConfig disabled;
        disabled.kalibr_.enabled_ = false;
        const auto normal_only = VideoDetector(disabled).detect(frame);
        require(std::none_of(normal_only.begin(), normal_only.end(),
                             [&](const Detection& tag) { return tag.id_ == id; }),
                "Disabled two-border support still reported the wide-border tag");
    }
    cv::RNG noise(4816);
    noise.fill(frame, cv::RNG::UNIFORM, 7, 62);
    frame(cv::Rect(21, 22, 67, 67)).setTo(148);
    put_tag(frame, {28, 29}, 5, 2, 0, 18, 148, 81, 2);
    frame(cv::Rect(246, 109, 47, 47)).setTo(171);
    put_tag(frame, {252, 115}, 4, 1, 0, 30, 171, 83);
    const auto& two_cell = search.detect(frame, true);
    const auto& ordinary = search.ordinary_tags();
    require(two_cell.size() == 1 && two_cell[0].id_ == 81 && ordinary.size() == 1 &&
                    ordinary[0].id_ == 83,
            "The same dark field must support exact words at both cell widths");
    const std::array<cv::Point2f, 4> dark_single_corners = {
            cv::Point2f{283.5F, 114.5F}, {283.5F, 146.5F}, {251.5F, 146.5F}, {251.5F, 114.5F}};
    for (int j = 0; j < 4; ++j)
        require(cv::norm(ordinary.front().corners_px_[j] - dark_single_corners[j]) < .005,
                "Exact ordinary print in the dark mixed field lost pixel-center alignment");
    const auto dark_video = VideoDetector{}.detect(frame);
    for (const int id : {81, 83}) {
        require(std::count_if(dark_video.begin(), dark_video.end(),
                              [&](const Detection& t) { return t.id_ == id; }) == 1,
                "Cold video search lost or duplicated a print on a dark field");
    }
    frame.setTo(230);
    const cv::Point near_border(frame.cols - 41, 63);
    put_tag(frame, near_border, 4, 3, 0, 10, 238, 171, 2);
    const auto edge = search.detect(frame);
    require(edge.size() == 1 && edge[0].id_ == 171,
            "A clipped white surround hid an otherwise visible two-border print");
    require_corners(edge[0].corners_px_, near_border, 40, 3);
    frame.setTo(230);
    require(search.uncovered_region(frame, {}).empty() && !search.has_candidate(frame) &&
                    search.detect(frame).empty(),
            "Two-border search retained a prior frame");
    require_invalid([&] { search.has_candidate(cv::Mat{}); });
    require_invalid([&] { search.detect(cv::Mat(7, 11, CV_16U)); });
    for (int level : {-1, 256}) {
        VideoConfig invalid_video;
        invalid_video.kalibr_.preview_dark_level_ = level;
        require_invalid([&] { VideoDetector detector(invalid_video); });
        invalid_video.kalibr_.enabled_ = false;
        require_invalid([&] { VideoDetector detector(invalid_video); });
    }
    KalibrGridConfig large_preview;
    large_preview.preview_short_side_ = std::numeric_limits<int>::max();
    require(!detail::KalibrGridSearch(large_preview).has_candidate(frame),
            "Large preview target changed an empty result");
    KalibrGridConfig invalid;
    invalid.maximum_component_side_px_ = 32768;
    require_invalid([&] { detail::KalibrGridSearch search_invalid(invalid); });
    invalid = {};
    invalid.grid_pitch_ = std::numeric_limits<float>::quiet_NaN();
    require_invalid([&] { detail::KalibrGridSearch search_invalid(invalid); });
}

void check_two_border_video_history() {
    using namespace aruco3cuda::apriltag_small_cpu;
    cv::Mat storage(243, 411, CV_8UC1), frame = storage(cv::Rect(3, 5, 395, 229));
    cv::Mat background(frame.size(), CV_8UC1);
    cv::RNG(19318).fill(background, cv::RNG::UNIFORM, 7, 62);
    background(cv::Rect(36, 27, 165, 128)).setTo(150);
    VideoDetector video;
    const auto board = [&](int first) {
        background.copyTo(frame);
        for (int i = 0; i < 12; ++i)
            put_tag(frame, {42 + 39 * (i % 4), 33 + 39 * (i / 4)}, 3, i % 4, 0, 18, 150, first + i,
                    2);
        const auto got = video.detect(frame);
        require(got.size() == 12, "Incorrect warm-board tag count");
        for (int i = 0; i < 12; ++i)
            require(std::count_if(got.begin(), got.end(),
                                  [&](const Detection& tag) { return tag.id_ == first + i; }) == 1,
                    "A warm-board ID was stale or missing");
    };
    board(300);
    board(300);
    board(340);
    background.copyTo(frame);
    require(video.detect(frame).empty(), "A textured blank retained a prior two-border board");
    frame.setTo(230);
    put_tag(frame, {88, 67}, 5, 1, 0, 14, 230, 430);
    const auto single = video.detect(frame);
    require(single.size() == 1 && single[0].id_ == 430,
            "A prior two-border board interfered with a subsequent one-cell frame");

    board(300);
    video.reset();
    background.copyTo(frame);
    require(video.detect(frame).empty(), "Explicit reset retained a two-border board");
    board(340);
    cv::Mat larger(frame.rows + 19, frame.cols + 21, CV_8UC1, cv::Scalar(230));
    require(video.detect(larger).empty(), "Dimension change retained a two-border board");
    put_tag(larger, {51, 72}, 4, 2, 0, 14, 230, 431);
    const auto resized = video.detect(larger);
    require(resized.size() == 1 && resized[0].id_ == 431,
            "Ordinary search was lost after a change in frame size");
}

void check_appearance(Detector& detector) {
    constexpr int kCellPx = 4;
    const cv::Point kOrigin(63, 17);
    cv::Mat parent(137, 299, CV_8UC1, cv::Scalar(2));
    cv::Mat roi = parent(cv::Rect(9, 13, 181, 99));
    for (int rotation = 0; rotation < 4; ++rotation) {
        for (int flips = 0; flips <= 4; ++flips) {
            roi.setTo(225);
            put_tag(roi, kOrigin, kCellPx, rotation, flips, 0, 255);
            const cv::Mat before = parent.clone();
            const auto result = detector.detect(roi);
            require(cv::countNonZero(before != parent) == 0, "Input pixels were modified");
            if (flips == 4) {
                require(result.empty(), "Four-bit corruption was accepted");
                continue;
            }
            require(result.size() == 1 && result[0].id_ == kTagId, "Expected tag ID was missed");
            require(result[0].corrected_bits_ == flips, "Unexpected bit correction count");
            require_corners(result[0].corners_px_, kOrigin, kCells * kCellPx, rotation);
        }
    }

    roi.setTo(180);
    const cv::Point kAnother(113, 47);
    put_tag(roi, kOrigin, kCellPx, 0, 0, 94, 179);
    put_tag(roi, kAnother, kCellPx, 2, 0, 94, 179);
    const auto low_contrast = detector.detect(roi);
    require(low_contrast.size() == 2, "Bright-frame route missed a low-contrast tag");
    for (const auto& tag : low_contrast) {
        require(tag.id_ == kTagId, "Unexpected tag ID on bright frame");
        const bool left = tag.corners_px_[0].x < 100.0F;
        require_corners(tag.corners_px_, left ? kOrigin : kAnother, kCells * kCellPx, left ? 0 : 2);
    }

    roi.setTo(103);
    put_tag(roi, kOrigin, kCellPx, 0, 0, 30, 103);
    const auto dark_scene = detector.detect(roi);
    require(dark_scene.size() == 1 && dark_scene[0].id_ == kTagId,
            "Dark-frame threshold missed the expected tag");
    roi.setTo(127);
    require(detector.detect(roi).empty(), "Detector retained a tag from the preceding frame");
    require(detector.detect(cv::Mat(1, 1, CV_8UC1, cv::Scalar(0))).empty(),
            "Single-pixel frame returned a tag");
}

void check_noisy_print_blocks() {
    using namespace aruco3cuda::apriltag_small_cpu;
    cv::RNG rng(9173);
    for (const int width : {1, 4, 5, 6, 7, 128, 129, 131}) {
        for (const int height : {2, 4, 5, 6, 7, 129}) {
            cv::Mat storage(height + 6, width + 11, CV_8UC1, cv::Scalar(247));
            cv::Mat image = storage(cv::Rect(5, 2, width, height));
            rng.fill(image, cv::RNG::UNIFORM, 0, 256);
            cv::Mat guide;
            detail::average_gray4x4(image, guide);
            require(guide.size() == cv::Size((width + 3) / 4, (height + 3) / 4),
                    "Wrong noisy guide size");
            for (int y = 0; y < guide.rows; ++y)
                for (int x = 0; x < guide.cols; ++x) {
                    const cv::Rect block(4 * x, 4 * y, std::min(4, width - 4 * x),
                                         std::min(4, height - 4 * y));
                    const auto expected =
                            cv::saturate_cast<std::uint8_t>(cv::mean(image(block))[0]);
                    require(guide.at<std::uint8_t>(y, x) == expected,
                            "Noisy guide sampled pixels outside its native block");
                }
        }
    }

    // Both guides must use only the current strided source and match their separate readers.
    for (const int width : {63, 64, 65, 128, 1920}) {
        for (const int height : {4, 11, 12}) {
            cv::Mat storage(height + 5, width + 19, CV_8UC1);
            rng.fill(storage, cv::RNG::UNIFORM, 0, 256);
            const cv::Mat image = storage(cv::Rect(7, 2, width, height));
            const cv::Size reduced(width / 4, height / 4);
            cv::Mat area, phase, expected;
            for (const auto mode :
                 {detail::QuarterGuideSample::nearest, detail::QuarterGuideSample::central_area}) {
                area = cv::Mat(1, 1, CV_8UC1, cv::Scalar(239));
                phase = cv::Mat(1, 1, CV_8UC1, cv::Scalar(41));
                if (!detail::try_sample_gray4x4(image, area, phase, mode)) {
                    require(area.size() == cv::Size(1, 1) && phase.size() == cv::Size(1, 1) &&
                                    area.at<std::uint8_t>(0) == 239 &&
                                    phase.at<std::uint8_t>(0) == 41,
                            "Unavailable fused sampling changed its outputs");
                    continue;
                }
                cv::resize(image, expected, reduced, 0, 0, cv::INTER_AREA);
                require(cv::norm(area, expected, cv::NORM_INF) == 0,
                        "Fused sampling changed full area means");
                if (mode == detail::QuarterGuideSample::nearest) {
                    cv::resize(image, expected, reduced, 0, 0, cv::INTER_NEAREST_EXACT);
                    require(cv::norm(phase, expected, cv::NORM_INF) == 0,
                            "Fused sampling changed the nearest phase");
                } else {
                    for (int y = 0; y < reduced.height; ++y) {
                        for (int x = 0; x < reduced.width; ++x) {
                            const cv::Rect center(4 * x + 1, 4 * y + 1, 2, 2);
                            const int sum = static_cast<int>(cv::sum(image(center))[0]);
                            require(phase.at<std::uint8_t>(y, x) == (sum + 2) / 4,
                                    "Fused sampling changed the central pixel mean");
                        }
                    }
                }
            }
        }
    }

    // Cross the streaming dispatch size and exercise cropped final blocks and heap scratch.
    for (const cv::Size size : {cv::Size(2047, 1024), cv::Size(2048, 1023), cv::Size(2048, 1024),
                                cv::Size(2051, 1027), cv::Size(4160, 1024)}) {
        cv::Mat storage(size.height + 6, size.width + 11, CV_8UC1, cv::Scalar(247));
        const cv::Mat image = storage(cv::Rect(5, 2, size.width, size.height));
        rng.fill(image, cv::RNG::UNIFORM, 0, 256);
        const cv::Size full(size.width / 4, size.height / 4);
        const cv::Size extent((size.width + 3) / 4, (size.height + 3) / 4);
        cv::Mat expected, guide;
        cv::resize(image(cv::Rect(0, 0, full.width * 4, full.height * 4)), expected, full, 0, 0,
                   cv::INTER_AREA);
        detail::average_gray4x4(image, guide);
        require(guide.size() == extent &&
                        cv::norm(guide(cv::Rect({}, full)), expected, cv::NORM_INF) == 0,
                "Streaming guide changed a complete block");
        for (int y = 0; y < extent.height; ++y) {
            for (int x = (y < full.height ? full.width : 0); x < extent.width; ++x) {
                const cv::Rect block(4 * x, 4 * y, std::min(4, size.width - 4 * x),
                                     std::min(4, size.height - 4 * y));
                require(guide.at<std::uint8_t>(y, x) ==
                                cv::saturate_cast<std::uint8_t>(cv::mean(image(block))[0]),
                        "Streaming guide changed a partial block");
            }
        }
    }

    // The full vector and scalar tail both have to reproduce INTER_AREA at half ties.
    constexpr int kSums = 16 * 255 + 1;
    constexpr int kRows = 3;
    cv::Mat source_storage(4 * kRows + 3, 4 * kSums + 17, CV_8UC1, cv::Scalar(63));
    auto source = source_storage(cv::Rect(5, 2, 4 * kSums, 4 * kRows));
    for (int row = 0; row < kRows; ++row) {
        for (int col = 0; col < kSums; ++col) {
            const int sum = (col + row * 1061) % kSums;
            for (int i = 0; i < 16; ++i) {
                source.at<std::uint8_t>(4 * row + i / 4, 4 * col + i % 4) =
                        static_cast<std::uint8_t>(sum / 16 + (i < sum % 16));
            }
        }
    }
    cv::Mat destination(kRows + 2, kSums + 21, CV_8UC1, cv::Scalar(199));
    auto guide = destination(cv::Rect(9, 1, kSums, kRows));
    cv::Mat expected;
    cv::resize(source, expected, guide.size(), 0, 0, cv::INTER_AREA);
    detail::average_gray4x4(source, guide);
    require(guide.data == destination.ptr<std::uint8_t>(1) + 9 &&
                    cv::countNonZero(guide != expected) == 0,
            "Full Gray8 block averaging changed a rounded value or replaced a strided output");
    guide.setTo(199);
    require(cv::countNonZero(destination != 199) == 0,
            "Full Gray8 block averaging modified pixels outside the output view");
#if defined(__x86_64__) && defined(__GNUC__)
    const unsigned original_mode = _MM_GET_ROUNDING_MODE();
    _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
    try {
        cv::resize(source, expected, guide.size(), 0, 0, cv::INTER_AREA);
        detail::average_gray4x4(source, guide);
    } catch (...) {
        _MM_SET_ROUNDING_MODE(original_mode);
        throw;
    }
    _MM_SET_ROUNDING_MODE(original_mode);
    require(cv::countNonZero(guide != expected) == 0,
            "Full Gray8 block averaging disagreed with INTER_AREA under MXCSR rounding");
#endif
}

void check_noisy_print_search() {
    using namespace aruco3cuda::apriltag_small_cpu;
    VideoConfig video;
    detail::NoisyPrintSearch search(video.noisy_print_);
    cv::Mat storage(347, 539, CV_8UC1, cv::Scalar(11));
    cv::Mat image = storage(cv::Rect(13, 7, 509, 329));
    const cv::Point origin(418, 238);
    constexpr int kCell = 11, kRotation = 1, kId = 113;
    cv::RNG random(982114);
    const auto expose = [&](bool printed) {
        image.setTo(178);
        if (printed)
            put_tag(image, origin, kCell, kRotation, 0, 21, 225, kId);
        cv::GaussianBlur(image, image, {5, 5}, .9, .9, cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);
        for (int y = 0; y < image.rows; ++y)
            for (int x = 0; x < image.cols; ++x)
                image.at<std::uint8_t>(y, x) = cv::saturate_cast<std::uint8_t>(
                        image.at<std::uint8_t>(y, x) + random.gaussian(6.5));
    };
    expose(true);
    const cv::Mat saved = storage.clone();
    std::vector<Detection> tags;
    search.append(image, video, tags);
    require(tags.size() == 1 && tags.front().id_ == kId, "Noisy source print was missed");
    require_corners(tags.front().corners_px_, origin, kCells * kCell, kRotation);
    std::vector<Detection> packed;
    search.append(image.clone(), video, packed);
    require(packed.size() == 1 && packed.front().corners_px_ == tags.front().corners_px_ &&
                    cv::countNonZero(saved != storage) == 0,
            "Noisy print depends on row pitch or changed its source");
    search.append(image, video, tags);
    require(tags.size() == 1 && tags.front().corners_px_ == packed.front().corners_px_,
            "Noisy print was appended twice");
    video.noisy_print_.enabled_ = false;
    detail::NoisyPrintSearch disabled(video.noisy_print_);
    tags.clear();
    disabled.append(image, video, tags);
    require(tags.empty(), "Disabled noisy print search added a tag");
    expose(false);
    search.append(image, video, tags);
    require(tags.empty(), "Noisy print search retained a vanished tag");
    for (int invalid : {-1, 1001}) {
        auto settings = VideoConfig{}.noisy_print_;
        settings.minimum_noisy_patches_per_mille_ = invalid;
        require_invalid([&] { detail::NoisyPrintSearch detector(settings); });
        settings.minimum_noisy_patches_per_mille_ = 750;
        settings.maximum_dark_patches_per_mille_ = invalid;
        require_invalid([&] { detail::NoisyPrintSearch detector(settings); });
    }
    for (float invalid : {-1.F, 32.F, std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN()}) {
        auto settings = VideoConfig{}.noisy_print_;
        settings.minimum_noise_sigma_ = invalid;
        require_invalid([&] { detail::NoisyPrintSearch detector(settings); });
    }
    for (int invalid : {-1, 256}) {
        auto settings = VideoConfig{}.noisy_print_;
        settings.maximum_ink_luma_ = invalid;
        require_invalid([&] { detail::NoisyPrintSearch detector(settings); });
    }
    for (int invalid : {0, 256}) {
        auto settings = VideoConfig{}.noisy_print_;
        settings.minimum_local_contrast_ = invalid;
        require_invalid([&] { detail::NoisyPrintSearch detector(settings); });
    }
}

void check_coarse() {
    Config config;
    config.coarse_.emplace();
    Detector detector(config);
    cv::Mat backing(261, 359, CV_8UC1, cv::Scalar(1));
    cv::Mat roi = backing(cv::Rect(9, 7, 329, 239));
    const cv::Point origin(111, 67);
    constexpr int kCellPx = 10;
    for (const int surround : {225, 103}) {
        for (int rotation = 0; rotation < 4; ++rotation) {
            roi.setTo(surround);
            put_tag(roi, origin, kCellPx, rotation, 1, 0, surround);
            const cv::Mat before = backing.clone();
            const auto tags = detector.detect(roi);
            require(cv::countNonZero(before != backing) == 0, "Coarse search changed input pixels");
            require(tags.size() == 1 && tags[0].id_ == kTagId && tags[0].corrected_bits_ == 1,
                    "Coarse search missed the expected marker");
            require_corners(tags[0].corners_px_, origin, kCells * kCellPx, rotation);
        }
    }
    put_tag(roi, origin, kCellPx, 0, 4, 0, 103);
    require(detector.detect(roi).empty(), "Coarse search accepted four-bit corruption");
    roi.setTo(225);
    require(detector.detect(roi).empty(), "Coarse search retained preceding candidates");

    config.coarse_->sample_stride_px_ = 1;
    require_invalid([&] { Detector invalid(config); });
    config.coarse_->sample_stride_px_ = 3;
    config.coarse_->grid_radius_px_ = 33;
    require_invalid([&] { Detector invalid(config); });
    config.coarse_->grid_radius_px_ = 6;
    config.coarse_->minimum_decode_contrast_ = std::numeric_limits<float>::quiet_NaN();
    require_invalid([&] { Detector invalid(config); });

    // The dark paper dots land on the coarse-sample lattice. The only useful
    // outline is a hole, and its native edge correction recovers the printed ring.
    cv::Mat halftone(201, 259, CV_8UC1, cv::Scalar(228));
    for (int y = 1; y < halftone.rows; y += 3)
        for (int x = 1; x < halftone.cols; x += 3)
            halftone.at<std::uint8_t>(y, x) = 5;
    const cv::Point print_origin(100, 67);
    put_tag(halftone, print_origin, 8, 3, 0, 134, 228, 18);
    const auto restored = detector.detect(halftone);
    require(restored.size() == 1 && restored[0].id_ == 18,
            "Native retry missed a printed tag inside a coarse mask hole");
    require_corners(restored[0].corners_px_, print_origin, 64, 3);
}

void check_component_candidates() {
    aruco3cuda::apriltag_small_cpu::detail::ComponentCandidates workspace;
    std::vector<std::vector<cv::Point>> candidates;
    ComponentConfig component;
    for (int width : {23, 24, 25, 31, 32, 33, 65}) {
        cv::Mat parent(42, width + 17, CV_8UC1, cv::Scalar(255));
        cv::Mat mask = parent(cv::Rect(5, 3, width, 37));
        mask.setTo(0);
        const cv::Rect split(0, 0, 9, 9);
        const cv::Rect separate(width - 9, 28, 9, 9);
        cv::rectangle(mask, split, cv::Scalar(255), 1);
        // The two vertical runs start with distinct labels. The last row
        // connects both, including an exclusive-end diagonal adjacency.
        mask.row(0).colRange(1, 8).setTo(0);
        mask.at<std::uint8_t>(7, 8) = 0;
        mask.at<std::uint8_t>(7, 7) = 255;
        cv::rectangle(mask, separate, cv::Scalar(255), 1);
        mask.at<std::uint8_t>(17, width - 1) = 255;
        mask.at<std::uint8_t>(21, 0) = 255;
        component.maximum_quad_side_samples_ = 12;
        workspace.extract(mask, component, candidates);
        require(candidates.size() == 2, "Component merging or separation failed");
        require(cv::boundingRect(candidates[0]) == split &&
                        cv::boundingRect(candidates[1]) == separate,
                "Component hull or scan order changed");
        const auto first = candidates;
        workspace.extract(mask.clone(), component, candidates);
        require(candidates == first, "Pitched mask differs from packed mask");
        component.maximum_quad_side_samples_ = 8;
        workspace.extract(mask, component, candidates);
        require(candidates.empty(), "Maximum component size was ignored");
        mask.setTo(255);
        component.maximum_quad_side_samples_ = 60;
        workspace.extract(mask, component, candidates);
        require(candidates.empty(), "Solid component accepted as a border");
    }
}

void check_native_component_runs() {
    using aruco3cuda::apriltag_small_cpu::detail::CoarseMask;
    using aruco3cuda::apriltag_small_cpu::detail::component_threshold;
    using aruco3cuda::apriltag_small_cpu::detail::ComponentCandidates;
    CoarseMask mask;
    ComponentCandidates candidates;
    std::vector<std::vector<cv::Point>> expected, actual;
    cv::RNG random(24645);
    std::size_t count = 0;
    for (int width : {1, 2, 63, 64, 65, 127, 128, 129, 255, 256, 257, 271, 288, 319, 383, 384, 385,
                      1024, 1025}) {
        for (int height : {1, 2, 37, 67}) {
            cv::Mat parent(height + 5, width + 19, CV_8UC1, cv::Scalar(17));
            auto crop = parent(cv::Rect(7, 2, width, height));
            for (int threshold : {0, 1, 69, 106, 192, 254, 255}) {
                for (bool exposure : {false, true}) {
                    ComponentConfig config;
                    config.threshold_ = threshold;
                    if (exposure)
                        config.exposure_.emplace();
                    random.fill(crop, cv::RNG::UNIFORM, 0, 256);
                    if (width > 85 && height > 20) {
                        crop.row(0).setTo(0);
                        crop.row(2).setTo(255);
                        // The outline crosses the first packed word on wide images.
                        const int x = width >= 170 ? 104 : width - 65;
                        cv::rectangle(crop, cv::Rect(x - 5, 5, 65, 25), cv::Scalar(255),
                                      cv::FILLED);
                        cv::rectangle(crop, cv::Rect(x, 8, 50, 19), cv::Scalar(0), 4);
                    }
                    const auto snapshot = parent.clone();
                    const auto& binary = mask.build_components(crop, config);
                    candidates.extract(binary, config, expected);
                    candidates.extract_native_stride2(crop, component_threshold(crop, config),
                                                      config, actual);
                    require(expected == actual,
                            "Native component runs changed candidate order or geometry");
                    require(cv::countNonZero(parent != snapshot) == 0,
                            "Native component runs changed source or padding");
                    count += actual.size();
                }
            }
        }
    }
    require(count > 20, "Native component run comparison missed its positive controls");
}

void check_dark_cross_components() {
    using aruco3cuda::apriltag_small_cpu::detail::ComponentCandidates;
    ComponentCandidates workspace;
    ComponentConfig config;
    const auto cross = cv::getStructuringElement(cv::MORPH_CROSS, {3, 3});
    cv::RNG random(981723);
    cv::Mat mask, core;
    std::vector<std::vector<cv::Point>> expected, packed;
    std::size_t positives = 0;
    for (int width : {1, 2, 16, 63, 64, 65, 127, 128, 129, 257}) {
        for (int height : {1, 2, 3, 46, 71}) {
            cv::Mat parent(height + 8, width + 19, CV_8U, cv::Scalar(0));
            auto crop = parent(cv::Rect(9, 3, width, height));
            for (int cutoff : {0, 1, 40, 76, 192, 254, 255}) {
                random.fill(crop, cv::RNG::UNIFORM, 0, 256);
                if (width >= 63 && height >= 46) {
                    const int x = width >= 80 ? 57 : width - 18;
                    const cv::Rect window(x - 2, 6, 17, 25);
                    crop(window).setTo(255);
                    cv::rectangle(crop, {x, 9, 13, 19}, cv::Scalar(0), 4);
                    cv::rectangle(crop, {0, height - 14, 12, 14}, cv::Scalar(0), 4);
                }
                const auto snapshot = parent.clone();
                cv::threshold(crop, mask, cutoff, 255, cv::THRESH_BINARY_INV);
                cv::erode(mask, core, cross);
                workspace.extract(core, config, expected);
                workspace.extract_dark_cross(crop, cutoff, config, packed);
                require(expected == packed, "Packed cross changed ordered component geometry");
                require(cv::countNonZero(parent != snapshot) == 0,
                        "Packed cross modified the source region or its parent");
                positives += packed.size();
            }
        }
    }
    require(positives > 20, "Packed cross comparison lost its positive controls");
}

void check_component_detector() {
    Config config;
    config.components_.emplace();
    for (int radius : {0, 6}) {
        config.components_->grid_radius_px_ = radius;
        Detector detector(config);
        cv::Mat backing(195, 273, CV_8UC1, cv::Scalar(0));
        cv::Mat image = backing(cv::Rect(7, 9, 251, 181));
        cv::RNG random(827);
        const cv::Point first(29, 43), second(155, 91);
        constexpr int kCellPx = 6;
        for (int rotation = 0; rotation < 4; ++rotation) {
            random.fill(image, cv::RNG::UNIFORM, 180, 246);
            cv::rectangle(image, {108, 23, 23, 27}, cv::Scalar(34), 3);
            put_tag(image, first, kCellPx, rotation, 0, 21, 232);
            put_tag(image, second, kCellPx, (rotation + 1) % 4, 2, 21, 232);
            const auto before = backing.clone();
            const auto tags = detector.detect(image);
            require(cv::countNonZero(before != backing) == 0, "Component search changed input");
            require(tags.size() == 2, "Component search missed a tag or decoded a distractor");
            for (const auto& tag : tags) {
                const bool left = tag.corners_px_[0].x < 100;
                require(tag.id_ == kTagId && tag.corrected_bits_ == (left ? 0 : 2),
                        "Incorrect component decode");
                require_corners(tag.corners_px_, left ? first : second, kCells * kCellPx,
                                left ? rotation : (rotation + 1) % 4);
            }
        }
        random.fill(image, cv::RNG::UNIFORM, 180, 246);
        cv::rectangle(image, {108, 23, 23, 27}, cv::Scalar(34), 3);
        require(detector.detect(image).empty(), "Negative frame returned a component tag");
        require(detector.detect(cv::Mat(1, 1, CV_8UC1, cv::Scalar(0))).empty(),
                "Single-pixel component frame returned a tag");
    }

    config.components_->grid_radius_px_ = 1;
    require_invalid([&] { Detector invalid(config); });
    config.components_->grid_radius_px_ = 0;
    config.components_->maximum_quad_side_samples_ = 32768;
    require_invalid([&] { Detector invalid(config); });
    config.components_->maximum_quad_side_samples_ = 600;
    config.components_->minimum_fill_ratio_ = std::numeric_limits<double>::quiet_NaN();
    require_invalid([&] { Detector invalid(config); });
    config.components_->minimum_fill_ratio_ = 0.22;
    config.components_->sample_stride_px_ = 1;
    require_invalid([&] { Detector invalid(config); });
    config.components_->sample_stride_px_ = 2;
    config.coarse_.emplace();
    require_invalid([&] { Detector invalid(config); });
}

void check_fused_mask() {
    cv::Mat backing(167, 171, CV_8UC1);
    cv::RNG random(55);
    random.fill(backing, cv::RNG::UNIFORM, 0, 256);
    aruco3cuda::apriltag_small_cpu::detail::CoarseMask workspace;
    CoarseConfig coarse;
    ComponentConfig component;
    coarse.dim_sample_floor_ = 0;
    for (int stride : {2, 3, 4}) {
        coarse.sample_stride_px_ = stride;
        component.sample_stride_px_ = stride;
        const int center = (stride - 1) / 2;
        for (int height : {1, 2, 3, 4, 32, 33, 127, 128, 129}) {
            for (int width : {1, 2, 3, 4, 31, 32, 33, 34, 47, 48, 49, 63, 64, 65, 127, 128, 129}) {
                const cv::Mat gray = backing(cv::Rect(3, 7, width, height));
                cv::Mat sampled((height + stride - 1) / stride, (width + stride - 1) / stride,
                                CV_8UC1);
                for (int y = 0; y < sampled.rows; ++y)
                    for (int x = 0; x < sampled.cols; ++x)
                        sampled.at<std::uint8_t>(y, x) =
                                gray.at<std::uint8_t>(std::min(y * stride + center, height - 1),
                                                      std::min(x * stride + center, width - 1));
                for (int threshold : {0, 11, 85, 128, 148, 220, 254, 255}) {
                    Config config;
                    config.dark_threshold_ = config.bright_threshold_ = threshold;
                    cv::Mat expected;
                    cv::threshold(sampled, expected, threshold, 255, cv::THRESH_BINARY_INV);
                    component.threshold_ = threshold;
                    require(cv::countNonZero(expected !=
                                             workspace.build_components(gray, component)) == 0,
                            "Component mask differs from thresholded visible samples");
                    component.sampled_contours_ = true;
                    require(cv::countNonZero(expected !=
                                             workspace.build_components(gray, component)) == 0,
                            "Fused contour mask differs from thresholded visible samples");
                    component.sampled_contours_ = false;
                    cv::erode(expected, expected, cv::Mat());
                    const auto& actual = workspace.build(gray, config, coarse);
                    require(cv::countNonZero(expected != actual) == 0,
                            "Fused coarse mask disagrees with threshold and erosion");
                }
            }
        }
    }
}

void check_large_tag_rails() {
    using aruco3cuda::apriltag_small_cpu::LargeTagRefinementConfig;
    using aruco3cuda::apriltag_small_cpu::VideoConfig;
    using aruco3cuda::apriltag_small_cpu::VideoDetector;
    using namespace aruco3cuda::apriltag_small_cpu::detail;

    cv::Mat sharp(183, 221, CV_8UC1, cv::Scalar(216));
    const cv::Point origin(56, 42);
    constexpr int cell_px = 12;
    put_tag(sharp, origin, cell_px, 0, 0, 45, 216);
    cv::Mat parent(sharp.rows + 11, sharp.cols + 23, CV_8UC1, cv::Scalar(7));
    auto frame = parent(cv::Rect(cv::Point(8, 2), sharp.size()));
    cv::GaussianBlur(sharp, frame, {15, 15}, 2.4);
    const auto snapshot = parent.clone();
    const Point a(origin.x - .5F, origin.y - .5F);
    const float side = cell_px * kCells;
    const Quad expected{a, a + Point(side, 0), a + Point(side, side), a + Point(0, side)};
    const Quad coarse{expected[0] + Point(2, 3), expected[1] + Point(-3, 2),
                      expected[2] + Point(-2, -2), expected[3] + Point(1, -3)};
    LargeTagRefinementConfig config;
    for (bool straight : {false, true}) {
        Quad refined;
        require(fit_long_tag_edges(frame, coarse, refined, config, straight),
                "A full blurred marker did not support its outer rails");
        for (std::size_t i = 0; i < expected.size(); ++i)
            require(length(refined[i] - expected[i]) < .35F, "Rail fit missed the physical corner");
        const auto check = decode(frame, refined, config.minimum_ring_matches_,
                                  config.minimum_decode_contrast_, 0);
        require(check.id == kTagId && check.rotation == 0, "Rail fit changed the oriented code");
        config.maximum_corner_motion_px_ = .1F;
        require(!fit_long_tag_edges(frame, coarse, refined, config, straight),
                "Rail fit ignored its displacement limit");
        config.maximum_corner_motion_px_ = 11.F;
    }
    require(cv::countNonZero(snapshot != parent) == 0, "Rail fit changed its input or padding");

    VideoConfig invalid;
    invalid.large_tag_refinement_.search_radius_px_ = 17;
    require_invalid([&] { VideoDetector detector(invalid); });
    invalid = {};
    invalid.large_tag_refinement_.minimum_side_px_ = std::numeric_limits<float>::quiet_NaN();
    require_invalid([&] { VideoDetector detector(invalid); });
}

void check_adaptive_tile_mask() {
    aruco3cuda::apriltag_small_cpu::detail::CoarseMask workspace;
    CoarseConfig coarse;
    coarse.dim_sample_floor_ = 255;
    coarse.dim_sample_percentage_ = 0;
    constexpr int stride = 3;
    constexpr int tile = 13;
    coarse.sample_stride_px_ = stride;
    coarse.tile_side_samples_ = tile;
    std::vector<std::uint64_t> packed;
    for (int columns : {63, 64, 65, 511, 512, 513, 639, 640, 641, 650, 1280}) {
        for (int tail : {0, 1, 2}) {
            const int width = stride * columns - tail;
            cv::Mat parent(124, width + 19, CV_8UC1, cv::Scalar(tail == 1 ? 0 : 255));
            auto gray = parent(cv::Rect(7, 2, width, 118 - tail));
            cv::Mat sampled((gray.rows + stride - 1) / stride, columns, CV_8UC1);
            for (int y = 0; y < gray.rows; ++y) {
                for (int x = 0; x < gray.cols; ++x) {
                    if (columns <= 65) {
                        // Contrast-one plateaux survive erosion. Exercise cutoff 255 on
                        // active near-white tiles and cutoff zero on inactive black tiles.
                        const bool alternate = (x / (5 * stride) + y / (5 * stride)) % 2 == 0;
                        gray.at<std::uint8_t>(y, x) =
                                tail == 2 ? (x >= width / 2 && alternate)
                                          : (x < width / 2 && alternate ? 254 : 255);
                    } else {
                        gray.at<std::uint8_t>(y, x) =
                                ((x / 113 + y / 35) % 4) * 41 + (x * 17 + y * 19) % 23;
                    }
                }
            }
            for (int y = 0; y < sampled.rows; ++y)
                for (int x = 0; x < sampled.cols; ++x)
                    sampled.at<std::uint8_t>(y, x) =
                            gray.at<std::uint8_t>(std::min(stride * y + 1, gray.rows - 1),
                                                  std::min(stride * x + 1, gray.cols - 1));
            for (int contrast : {0, 1, 16, 95, 180, 255}) {
                coarse.tile_minimum_contrast_ = contrast;
                cv::Mat expected(sampled.size(), CV_8UC1, cv::Scalar(0));
                const cv::Rect bounds({}, sampled.size());
                for (int y = 0; y < sampled.rows; y += tile) {
                    for (int x = 0; x < sampled.cols; x += tile) {
                        const auto neighbors =
                                cv::Rect(x - tile, y - tile, 3 * tile, 3 * tile) & bounds;
                        double low, high;
                        cv::minMaxLoc(sampled(neighbors), &low, &high);
                        if (high - low < contrast)
                            continue;
                        const auto region = cv::Rect(x, y, tile, tile) & bounds;
                        cv::Mat output = expected(region);
                        cv::threshold(sampled(region), output, std::floor((low + high) / 2), 255,
                                      cv::THRESH_BINARY_INV);
                    }
                }
                cv::erode(expected, expected, cv::Mat());
                cv::Point origin(-1, -1);
                const auto& actual = workspace.build(gray, Config{}, coarse, &origin);
                require(origin == cv::Point() && cv::countNonZero(expected != actual) == 0,
                        "Adaptive coarse mask changed neighboring tile support or ROI sampling");
                if (columns <= 65 && contrast == 1) {
                    const int foreground = cv::countNonZero(expected);
                    require(foreground > 0 &&
                                    static_cast<std::size_t>(foreground) < expected.total(),
                            "Missing packed cutoff boundary coverage");
                }
                const cv::Mat& packed_shape =
                        workspace.build(gray, Config{}, coarse, &origin, false, &packed);
                const int words = (columns + 63) / 64;
                require(origin == cv::Point() && packed_shape.size() == expected.size() &&
                                packed.size() == static_cast<std::size_t>(words) * expected.rows,
                        "Packed adaptive threshold changed extent or origin");
                for (int y = 0; y < expected.rows; ++y) {
                    const auto* row = expected.ptr<std::uint8_t>(y);
                    const auto* bits = packed.data() + static_cast<std::size_t>(y) * words;
                    for (int x = 0; x < expected.cols; ++x)
                        require(((bits[x / 64] >> (x % 64)) & 1U) == (row[x] != 0),
                                "Packed tile cutoff disagrees with OpenCV");
                }
            }
        }
    }
}

void check_bounded_adaptive_mask() {
    aruco3cuda::apriltag_small_cpu::detail::CoarseMask masks;
    Config config;
    CoarseConfig coarse;
    coarse.dim_sample_floor_ = 255;
    coarse.dim_sample_percentage_ = 0;
    cv::Mat backing(887, 1963, CV_8UC1, cv::Scalar(255));
    cv::RNG rng(9923941);
    std::vector<std::uint64_t> packed;
    bool saw_partial = false, saw_empty = false, saw_full = false;
    for (int trial = 0; trial < 60; ++trial) {
        coarse.sample_stride_px_ = 2 + trial % 3;
        coarse.tile_side_samples_ = 3 + trial % 30;
        auto source = backing(cv::Rect(4, 3, 1937 - trial % 5, 859 - trial % 7));
        source.setTo(107);
        if (trial % 4 == 0)
            rng.fill(source, cv::RNG::UNIFORM, 0, 255);
        if (trial % 4 == 1 || trial % 4 == 2) {
            const int x = trial % 4 == 1 ? 0 : source.cols - 163;
            rng.fill(source(cv::Rect(x, 118, 163, 79)), cv::RNG::UNIFORM, 0, 255);
        }
        cv::Point full_origin(-1, -1);
        const auto full = masks.build(source, config, coarse, &full_origin).clone();
        require(full_origin == cv::Point(), "Full adaptive mask changed origin");
        cv::Point origin;
        const auto& region = masks.build(source, config, coarse, &origin, true);
        cv::Mat restored = cv::Mat::zeros(full.size(), CV_8UC1);
        if (!region.empty()) {
            cv::Size whole;
            cv::Point offset;
            region.locateROI(whole, offset);
            require(!region.isSubmatrix() && region.isContinuous() && offset == cv::Point() &&
                            whole == region.size(),
                    "Adaptive crop changed its isolated image extent");
            region.copyTo(restored(cv::Rect(origin, region.size())));
        }
        require(cv::countNonZero(restored != full) == 0, "Restored adaptive mask changed pixels");
        saw_empty |= region.empty();
        saw_partial |= !region.empty() && region.size() != full.size();
        cv::Point packed_origin;
        const auto& packed_shape =
                masks.build(source, config, coarse, &packed_origin, true, &packed);
        require(packed_origin == origin && packed_shape.size() == region.size(),
                "Packed adaptive mask changed extent or origin");
        const int words = (region.cols + 63) / 64;
        require(packed.size() == static_cast<std::size_t>(words) * region.rows,
                "Packed adaptive mask has the wrong number of words");
        for (int y = 0; y < region.rows; ++y) {
            const auto* pixels = region.ptr<std::uint8_t>(y);
            const auto* bits = packed.data() + static_cast<std::size_t>(y) * words;
            for (int x = 0; x < region.cols; ++x)
                require(((bits[x / 64] >> (x % 64)) & 1U) == (pixels[x] != 0),
                        "Packed adaptive erosion changed a mask pixel");
            const int tail = region.cols % 64;
            if (tail)
                require((bits[words - 1] >> tail) == 0, "Packed adaptive mask left its tail set");
        }

        saw_full |= region.size() == full.size();
    }
    require(saw_empty && saw_partial && saw_full, "Missing adaptive mask extent coverage");
}

void check_cropped_eroded_mask() {
    aruco3cuda::apriltag_small_cpu::detail::CoarseMask workspace;
    CoarseConfig coarse;
    coarse.dim_sample_floor_ = 0;
    Config config;
    config.dark_threshold_ = config.bright_threshold_ = 85;
    cv::Mat storage(253, 391, CV_8UC1, cv::Scalar(0));
    // Poisoned padding surrounds each native ROI. Vary the final subsample width,
    // row-probe phase, SIMD width, and clipped support on all four sides.
    for (int stride : {2, 3, 4}) {
        coarse.sample_stride_px_ = stride;
        for (int phase = 0; phase < 3 * stride; ++phase) {
            cv::Mat gray = storage(cv::Rect(11, 7, 371 - phase, 233 - phase));
            const cv::Rect frame({}, gray.size());
            const cv::Rect interior(75 + phase, 43 + phase, 39, stride * 3 + 1);
            for (const auto& patch : {cv::Rect(), interior, cv::Rect(-8, -8, 30, 32),
                                      cv::Rect(gray.cols - 17, gray.rows - 19, 30, 32),
                                      cv::Rect(36, gray.rows - 5, 81, 18),
                                      cv::Rect(gray.cols - 3, 18, 10, 64), frame}) {
                gray.setTo(86);
                if (!(patch & frame).empty())
                    gray(patch & frame).setTo(85);
                const auto full = workspace.build(gray, config, coarse).clone();
                cv::Point origin(-1, -1);
                const auto& result = workspace.build(gray, config, coarse, &origin);
                cv::Mat restored(full.size(), CV_8UC1, cv::Scalar(0));
                if (!result.empty())
                    result.copyTo(restored(cv::Rect(origin, result.size())));
                require(cv::countNonZero(restored != full) == 0,
                        "Cropped mask changed the result of full sampled erosion");
                if (patch.empty())
                    require(result.empty(), "Empty field retained a crop");
                if (patch == interior)
                    require(!result.empty() && result.total() < full.total() && origin.x > 0 &&
                                    origin.y > 0,
                            "Interior support was not bounded by a cropped mask");
            }
        }
    }
}

void check_decoded_grid_alignment() {
    using aruco3cuda::apriltag_small_cpu::Detection;
    using aruco3cuda::apriltag_small_cpu::detail::align_decoded_grid;
    using aruco3cuda::apriltag_small_cpu::detail::try_align_sharp_grid;
    using aruco3cuda::apriltag_small_cpu::detail::try_fit_sharp_border;
    cv::Mat backing(165, 191, CV_8UC1, cv::Scalar(3));
    auto gray = backing(cv::Rect(13, 9, 151, 139));
    const cv::Point origin(35, 27);
    const std::array<cv::Point2f, 4> expected{
            {{34.5F, 26.5F}, {98.5F, 26.5F}, {98.5F, 90.5F}, {34.5F, 90.5F}}};
    const std::array<cv::Point2f, 4> offset{{{1, 1}, {-1, 2}, {1, -2}, {-2, -1}}};
    for (int rotation = 0; rotation < 4; ++rotation) {
        gray.setTo(220);
        put_tag(gray, origin, 8, rotation, 0, 80, 220);
        Detection tag{kTagId, {}, 0};
        for (int i = 0; i < 4; ++i)
            tag.corners_px_[i] = expected[(i + rotation) % 4] + offset[i];
        auto border = tag.corners_px_;
        require(try_fit_sharp_border(gray, border), "Supported sharp outer border was rejected");
        require_corners(border, origin, 64, rotation);
        const cv::Mat before = backing.clone();
        require(align_decoded_grid(gray, tag), "Valid decoded grid failed to align");
        require_corners(tag.corners_px_, origin, 64, rotation);
        Detection sharp{kTagId, {}, 0};
        for (int i = 0; i < 4; ++i)
            sharp.corners_px_[i] = expected[(i + rotation) % 4] + 0.25F * offset[i];
        require(try_align_sharp_grid(gray, sharp), "Clean strip measurements failed to align");
        for (int i = 0; i < 4; ++i)
            require(cv::norm(sharp.corners_px_[i] - expected[(i + rotation) % 4]) < 0.02,
                    "Sharp transition fit moved off the known pixel boundary");
        require(tag.id_ == kTagId && tag.corrected_bits_ == 0 &&
                        cv::countNonZero(before != backing) == 0,
                "Alignment changed the image or decode metadata");
    }

    Detection wrong{kTagId + 1, expected, 0};
    gray.setTo(220);
    put_tag(gray, origin, 8, 0, 0, 80, 220);
    require(!try_align_sharp_grid(gray, wrong) && wrong.corners_px_ == expected,
            "Unconfirmed sharp-grid fit changed the report");
    require(!align_decoded_grid(gray, wrong) && wrong.corners_px_ == expected,
            "Unconfirmed oriented payload changed the reported corners");
    Detection truncated{kTagId, expected, 0};
    truncated.corners_px_[1] += cv::Point2f(-6.0F, 11.0F);
    require(align_decoded_grid(gray, truncated), "Failed to recover a truncated corner");
    require_corners(truncated.corners_px_, origin, 64, 0);

    Detection flat{kTagId, expected, 0};
    gray.setTo(127);
    require(!try_fit_sharp_border(gray, flat.corners_px_) && flat.corners_px_ == expected,
            "A flat image changed the proposed outer border");
    require(!align_decoded_grid(gray, flat) && flat.corners_px_ == expected,
            "A flat frame produced a grid correction");
    Detection degenerate{kTagId, {}, 0};
    require(!align_decoded_grid(gray, degenerate), "Accepted a zero-area quad");
    flat.id_ = -1;
    require(!align_decoded_grid(gray, flat), "Accepted an invalid decoded ID");
}

void check_displaced_payload_preserves_measured_border() {
    using aruco3cuda::apriltag_small_cpu::Detection;
    using aruco3cuda::apriltag_small_cpu::detail::align_decoded_grid;

    cv::Mat backing(152, 161, CV_8UC1, cv::Scalar(8));
    auto gray = backing(cv::Rect(7, 11, 129, 127));
    const cv::Point origin(27, 27);
    constexpr int cell = 8;
    const std::array<cv::Point2f, 4> border{
            {{26.5F, 26.5F}, {90.5F, 26.5F}, {90.5F, 90.5F}, {26.5F, 90.5F}}};
    for (int rotation = 0; rotation < 4; ++rotation) {
        gray.setTo(210);
        put_tag(gray, origin, cell, rotation, 0, 33, 210);
        const cv::Rect payload(origin + cv::Point(cell, cell), cv::Size(6 * cell, 6 * cell));
        const cv::Mat ink = gray(payload).clone();
        gray(payload).setTo(33);
        ink.copyTo(gray(payload + cv::Point(2, 2)));
        cv::GaussianBlur(gray, gray, {}, 1.4, 1.4, cv::BORDER_REPLICATE | cv::BORDER_ISOLATED);
        Detection tag{kTagId, {}, 0};
        for (int k = 0; k < 4; ++k)
            tag.corners_px_[k] = border[(k + rotation) % 4];
        const auto measured = tag.corners_px_;
        require(align_decoded_grid(gray, tag) && tag.corners_px_ == measured,
                "A displaced payload pulled the corners away from the measured outer border");
    }
}

void check_local_midrange_mask() {
    Config config;
    auto& component = config.components_.emplace();
    component.sampled_contours_ = true;
    cv::Mat parent(69, 113, CV_8UC1, cv::Scalar(255));
    auto region = parent(cv::Rect(7, 3, 93, 57));
    cv::RNG random(57);
    random.fill(region, cv::RNG::UNIFORM, 17, 149);
    const auto before = parent.clone();
    aruco3cuda::apriltag_small_cpu::detail::CoarseMask workspace;
    for (const auto size : {cv::Size(1, 1), cv::Size(3, 9), region.size()}) {
        const auto gray = region(cv::Rect({0, 0}, size));
        for (int stride : {2, 3, 4}) {
            component.sample_stride_px_ = stride;
            const int center = (stride - 1) / 2;
            const auto sample = [&](int y, int x) {
                return gray.at<std::uint8_t>(std::min(y * stride + center, gray.rows - 1),
                                             std::min(x * stride + center, gray.cols - 1));
            };
            for (int side : {3, 31, 7}) {
                component.local_midrange_side_samples_ = side;
                for (int offset : {0, 5, 127}) {
                    component.local_midrange_offset_ = offset;
                    const auto& mask = workspace.build_components(gray, component);
                    require(mask.rows == (gray.rows + stride - 1) / stride &&
                                    mask.cols == (gray.cols + stride - 1) / stride,
                            "Unexpected local mask dimensions");
                    for (int y = 0; y < mask.rows; ++y) {
                        for (int x = 0; x < mask.cols; ++x) {
                            int low = 255, high = 0;
                            for (int j = std::max(0, y - side / 2);
                                 j <= std::min(mask.rows - 1, y + side / 2); ++j) {
                                for (int i = std::max(0, x - side / 2);
                                     i <= std::min(mask.cols - 1, x + side / 2); ++i) {
                                    const int value = sample(j, i);
                                    low = std::min(low, value);
                                    high = std::max(high, value);
                                }
                            }
                            const int threshold = cvRound((low + high) * 0.5 - offset);
                            require(mask.at<std::uint8_t>(y, x) ==
                                            (sample(y, x) < threshold ? 255 : 0),
                                    "Local mask differs from the visible neighborhood midpoint");
                        }
                    }
                }
            }
        }
    }
    require(cv::countNonZero(parent != before) == 0, "Local mask changed its input");

    component.sample_stride_px_ = 2;
    component.local_midrange_offset_ = 5;
    component.local_midrange_side_samples_ = 3;
    region.setTo(190);
    const cv::Point origin(15, 9);
    put_tag(region, origin, 4, 3, 0, 142, 190);
    const auto tags = Detector(config).detect(region);
    require(tags.size() == 1 && tags[0].id_ == kTagId && tags[0].corrected_bits_ == 0,
            "Local threshold missed an elevated black level");
    require_corners(tags[0].corners_px_, origin, 32, 3);

    for (int side : {-1, 1, 4, 33}) {
        component.local_midrange_side_samples_ = side;
        require_invalid([&] { Detector invalid(config); });
    }
    component.local_midrange_side_samples_ = 3;
    for (int offset : {-1, 128}) {
        component.local_midrange_offset_ = offset;
        require_invalid([&] { Detector invalid(config); });
    }
    component.local_midrange_offset_ = 5;
    component.exposure_.emplace();
    require_invalid([&] { Detector invalid(config); });
    component.exposure_.reset();
    component.sampled_contours_ = false;
    component.alternate_contour_threshold_ = 148;
    config.recovery_.emplace();
    require_invalid([&] { Detector invalid(config); });
}

void check_sampled_contours() {
    Config config;
    auto& sampled = config.components_.emplace();
    sampled.sampled_contours_ = true;
    cv::Mat backing(170, 297, CV_8UC1, cv::Scalar(1));
    auto frame = backing(cv::Rect(7, 9, 271, 145));
    constexpr int kCellPx = 8;
    const cv::Point origin(73, 47);

    frame.setTo(235);
    put_tag(frame, origin, kCellPx, 0, 0, 120, 235);
    require(Detector(config).detect(frame).empty(), "Lower threshold accepted lighter dark cells");
    sampled.threshold_ = 148;
    for (int stride : {2, 3, 4}) {
        sampled.sample_stride_px_ = stride;
        Detector detector(config);
        for (int rotation = 0; rotation < 4; ++rotation) {
            frame.setTo(235);
            put_tag(frame, origin, kCellPx, rotation, 0, 120, 235);
            const auto before = backing.clone();
            const auto tags = detector.detect(frame);
            require(tags.size() == 1 && tags.front().id_ == kTagId,
                    "Sampled contours missed the expected tag");
            require_corners(tags.front().corners_px_, origin, 8 * kCellPx, rotation);
            require(cv::countNonZero(before != backing) == 0,
                    "Sampled contours changed input pixels");
        }
        frame.setTo(235);
        put_tag(frame, origin, kCellPx, 0, 4, 120, 235);
        require(detector.detect(frame).empty(), "Sampled contours accepted four corrupted bits");
    }
    sampled.threshold_ = 256;
    require_invalid([&] { Detector invalid(config); });
}

void check_alternating_contours() {
    Config config;
    auto& components = config.components_.emplace();
    config.recovery_.emplace();
    components.alternate_contour_threshold_ = 148;
    cv::Mat backing(170, 297, CV_8UC1, cv::Scalar(1));
    auto frame = backing(cv::Rect(7, 9, 271, 145));
    cv::Mat other_backing(171, 327, CV_8UC1, cv::Scalar(1));
    auto same_size = other_backing(cv::Rect(9, 10, 271, 145));
    auto resized = other_backing(cv::Rect(9, 10, 273, 145));
    constexpr int kCellPx = 8;
    const cv::Point origin(73, 47);

    const auto draw_light_tag = [&](cv::Mat& gray) {
        gray.setTo(235);
        put_tag(gray, origin, kCellPx, 2, 0, 120, 235);
    };
    draw_light_tag(frame);
    draw_light_tag(same_size);
    Config sampled_config;
    sampled_config.components_.emplace();
    sampled_config.components_->sampled_contours_ = true;
    sampled_config.components_->threshold_ = 148;
    const auto sampled = Detector(sampled_config).detect(frame);
    require(sampled.size() == 1, "Missing isolated sampled reference for alternation");

    Detector detector(config);
    require(detector.detect(frame).empty(), "First frame skipped component discovery");
    require_invalid([&] { detector.detect(cv::Mat()); });
    const auto second = detector.detect(same_size);
    require(second.size() == 1 && second.front().id_ == sampled.front().id_ &&
                    second.front().corners_px_ == sampled.front().corners_px_,
            "Second frame differed from sampled discovery after invalid input and pitch change");
    require_invalid([&] { detector.detect(cv::Mat(5, 8, CV_8UC3)); });
    const auto recovered = detector.detect(frame);
    require(recovered.size() == 1 && recovered.front().id_ == kTagId &&
                    recovered.front().corrected_bits_ == 0,
            "One recovery history must span both discovery phases");
    require_corners(recovered.front().corners_px_, origin, 8 * kCellPx, 2);

    draw_light_tag(resized);
    require(detector.detect(resized).empty(), "Size change did not restart component discovery");
    require(detector.detect(resized).size() == 1,
            "Sampled discovery did not resume after resizing");
    detector.reset();
    require(detector.detect(resized).empty(), "Reset retained the recovery history");
    detector.reset();
    require(detector.detect(resized).empty(), "Reset did not reset the discovery phase");
    require(detector.detect(resized).size() == 1, "Sampled discovery did not resume after reset");
    resized.setTo(235);
    require(detector.detect(resized).empty(), "Recovery replayed an absent tag");

    Config invalid = config;
    invalid.recovery_.reset();
    require_invalid([&] { Detector checked(invalid); });
    invalid = config;
    invalid.components_->sampled_contours_ = true;
    require_invalid([&] { Detector checked(invalid); });
    invalid = config;
    invalid.components_->exposure_.emplace();
    require_invalid([&] { Detector checked(invalid); });
    invalid = config;
    invalid.components_->sample_stride_px_ = 3;
    require_invalid([&] { Detector checked(invalid); });
    for (int threshold : {-1, 256}) {
        invalid = config;
        invalid.components_->alternate_contour_threshold_ = threshold;
        require_invalid([&] { Detector checked(invalid); });
    }
    invalid = config;
    invalid.components_.reset();
    invalid.automatic_.emplace().components_.alternate_contour_threshold_ = 148;
    require_invalid([&] { Detector checked(invalid); });

    components.alternate_contour_threshold_ = 0;
    Detector minimum(config);
    require(minimum.detect(frame).empty() && minimum.detect(frame).empty(),
            "Zero alternate threshold was interpreted as unset");
}

void check_component_exposure() {
    ComponentConfig config;
    Config automatic;
    automatic.automatic_.emplace();
    require(!config.exposure_ && automatic.automatic_->components_.exposure_.has_value(),
            "Exposure defaults differ from the selected search mode");

    aruco3cuda::apriltag_small_cpu::detail::CoarseMask workspace;
    cv::Mat backing(89, 147, CV_8UC1, cv::Scalar(1));
    cv::Mat image = backing(cv::Rect(9, 7, 109, 69));
    image.setTo(160);
    const std::array<int, 12> levels{0, 20, 66, 150, 155, 160, 170, 180, 185, 190, 240, 255};
    for (int i = 0; i < static_cast<int>(levels.size()); ++i)
        image.at<std::uint8_t>(7 + 29 * (i / 4), 11 + 31 * (i % 4)) = levels[i];

    const auto require_level = [&](int level, cv::Mat frame) {
        frame.at<std::uint8_t>(0, 0) = static_cast<std::uint8_t>(level);
        frame.at<std::uint8_t>(0, 2) = static_cast<std::uint8_t>(std::min(255, level + 1));
        const auto before = frame.clone();
        const cv::Mat mask = workspace.build_components(frame, config).clone();
        require(mask.at<std::uint8_t>(0, 0) == 255 &&
                        mask.at<std::uint8_t>(0, 1) == (level == 255 ? 255 : 0),
                "Incorrect component exposure threshold");
        require(cv::countNonZero(mask != workspace.build_components(before, config)) == 0,
                "Pitched exposure mask differs from packed mask");
        config.sampled_contours_ = true;
        require(cv::countNonZero(mask != workspace.build_components(frame, config)) == 0,
                "Contour mask differs after exposure adjustment");
        config.sampled_contours_ = false;
        require(cv::countNonZero(before != frame) == 0, "Exposure mask changed source data");
    };

    require_level(106, image);
    config.exposure_.emplace();
    require_level(124, image);
    image.at<std::uint8_t>(7, 73) = 65;
    require_level(106, image);
    image.at<std::uint8_t>(7, 73) = 66;
    image.at<std::uint8_t>(65, 73) = 180;
    require_level(109, image);
    image.at<std::uint8_t>(65, 73) = 240;
    config.exposure_->maximum_adjustment_ = 4;
    require_level(110, image);
    config.exposure_->maximum_adjustment_ = 39;
    config.threshold_ = 250;
    require_level(255, image);
    config.threshold_ = 106;
    require_level(106, cv::Mat(7, 11, CV_8UC1, cv::Scalar(245)));

    Config invalid;
    invalid.components_ = config;
    invalid.components_->exposure_->adjustment_percent_ = 101;
    require_invalid([&] { Detector detector(invalid); });
}

void check_luma_distribution() {
    using aruco3cuda::apriltag_small_cpu::detail::LumaDistribution;
    LumaDistribution::Bins counts{};
    const LumaDistribution empty(counts);
    require(empty.total() == 0 && empty.percentile(90) == 0, "Empty luminance histogram changed");

    constexpr std::array<int, 10> samples{0, 8, 8, 8, 42, 42, 115, 218, 218, 255};
    for (const int sample : samples)
        ++counts[sample];
    const LumaDistribution luma(counts);
    for (int level = -1; level <= 257; ++level) {
        const auto below = std::count_if(samples.begin(), samples.end(),
                                         [level](int sample) { return sample < level; });
        require(luma.below(level) == below, "Luminance threshold lost its exclusive endpoint");
    }
    for (int percent = 1; percent <= 100; ++percent) {
        const int rank = (10 * percent + 99) / 100;
        require(luma.percentile(percent) == samples[rank - 1], "Luminance percentile tie changed");
    }
}

void check_automatic_detector() {
    using aruco3cuda::apriltag_small_cpu::detail::AutomaticRoute;
    using aruco3cuda::apriltag_small_cpu::detail::choose_automatic_route;
    Config config;
    config.automatic_.emplace();
    config.automatic_->components_.minimum_quad_side_samples_ = 12;
    Detector detector(config);
    cv::Mat backing(283, 551, CV_8UC1, cv::Scalar(0));
    cv::Mat image = backing(cv::Rect(9, 11, 509, 251));

    image.setTo(179);
    put_tag(image, {75, 41}, 3, 1, 1, 21, 230);
    require(choose_automatic_route(image, *config.automatic_) == AutomaticRoute::kNative,
            "Sparse native frame was routed incorrectly");
    auto tags = detector.detect(image);
    require(tags.size() == 1 && tags[0].id_ == kTagId && tags[0].corrected_bits_ == 1,
            "Automatic native search missed the expected tag");
    require_corners(tags[0].corners_px_, {75, 41}, 24, 1);

    image.setTo(235);
    put_tag(image, {38, 39}, 10, 3, 0, 0, 235);
    put_tag(image, {172, 93}, 10, 0, 0, 0, 235);
    require(choose_automatic_route(image, *config.automatic_) == AutomaticRoute::kCoarse,
            "Large high-contrast tags were routed incorrectly");
    require(detector.detect(image).size() == 2, "Automatic coarse search missed an expected tag");

    cv::RNG random(725);
    random.fill(image, cv::RNG::UNIFORM, 112, 203);
    for (const auto& origin :
         std::array<cv::Point, 4>{{{31, 36}, {123, 52}, {215, 78}, {290, 128}}})
        put_tag(image, origin, 6, 2, 0, 21, 232);
    put_tag(image, {434, 175}, 2, 0, 0, 21, 232);
    require(choose_automatic_route(image, *config.automatic_) == AutomaticRoute::kComponents,
            "Textured frame was routed incorrectly");
    const cv::Mat before = backing.clone();

    Config component_config;
    component_config.components_ = config.automatic_->components_;
    Detector components(component_config);
    require(components.detect(image).size() == 4, "Unexpected fixture component count");
    tags = detector.detect(image);
    require(tags.size() == 5,
            "Supplemental native search missed a tag or duplicated an observation");
    for (const auto& tag : tags)
        require(tag.id_ == kTagId && tag.corrected_bits_ == 0, "Incorrect automatic decode");
    require(cv::countNonZero(before != backing) == 0, "Automatic search modified input pixels");

    image.setTo(235);
    require(detector.detect(image).empty(), "Automatic search retained the previous results");
    require(detector.detect(cv::Mat(1, 1, CV_8UC1, cv::Scalar(0))).empty(),
            "Smallest valid frame returned a tag in automatic search");
    require_invalid([&] { detector.detect(cv::Mat{}); });
    require_invalid([&] { detector.detect(cv::Mat(32, 32, CV_8UC3)); });

    config.coarse_.emplace();
    require_invalid([&] { Detector invalid(config); });
    config.coarse_.reset();
    config.components_.emplace();
    require_invalid([&] { Detector invalid(config); });
    config.components_.reset();
    config.automatic_->component_max_flat_per_mille_ = 1001;
    require_invalid([&] { Detector invalid(config); });
    config.automatic_->component_max_flat_per_mille_ = 600;
    config.automatic_->components_.sample_stride_px_ = 0;
    require_invalid([&] { Detector invalid(config); });
}

void check_dense_region() {
    Config config;
    auto& automatic = config.automatic_.emplace();
    // Force the ordinary routes to reject the bright black ink in this fixture.
    config.dark_threshold_ = 80;
    config.bright_threshold_ = 105;
    automatic.coarse_.minimum_quad_side_samples_ = 150;
    automatic.components_.minimum_quad_side_samples_ = 150;
    auto& dense = automatic.dense_region_;
    Detector detector(config);

    Config disabled = config;
    disabled.automatic_->dense_region_.enabled_ = false;
    Detector baseline(disabled);
    Config capped = config;
    capped.automatic_->dense_region_.maximum_roi_pixels_ = 16;
    Detector capped_detector(capped);
    Config native_only = config;
    native_only.automatic_->dense_region_.maximum_upsample_source_pixels_ = 0;
    Detector unscaled(native_only);

    cv::Mat backing(377, 503, CV_8UC1, cv::Scalar(3));
    cv::Mat image = backing(cv::Rect(13, 9, 457, 353));
    constexpr int kColumns = 9;
    constexpr int kRows = 7;
    constexpr int kCellPx = 2;
    constexpr int kSide = kCells * kCellPx;
    constexpr int kPitch = 21;

    for (const cv::Point first : {cv::Point(103, 91), cv::Point(3, 5)}) {
        image.setTo(250);
        for (int y = 0; y < kRows; ++y)
            for (int x = 0; x < kColumns; ++x)
                put_tag(image, first + cv::Point(x * kPitch, y * kPitch), kCellPx, (x + y) % 4, 0,
                        137, 250);
        const cv::Mat before = backing.clone();

        const auto tags = detector.detect(image);
        const auto packed_tags = detector.detect(image.clone());
        const auto unscaled_tags = unscaled.detect(image);
        require(tags.size() == kColumns * kRows && packed_tags.size() == tags.size() &&
                        unscaled_tags.size() == tags.size(),
                "Dense region did not recover the expected array");
        bool resized_corners = false;
        std::array<bool, kColumns * kRows> seen{};
        for (std::size_t i = 0; i < tags.size(); ++i) {
            const auto& tag = tags[i];
            require(tag.id_ == kTagId && tag.corrected_bits_ == 0, "Unexpected dense region ID");
            require(tag.corners_px_ == packed_tags[i].corners_px_,
                    "Pitched dense region changed corners");
            const auto center = (tag.corners_px_[0] + tag.corners_px_[2]) * .5F;
            const int x = cvRound((center.x - first.x - (kSide - 1) * .5F) / kPitch);
            const int y = cvRound((center.y - first.y - (kSide - 1) * .5F) / kPitch);
            require(x >= 0 && x < kColumns && y >= 0 && y < kRows, "Incorrect dense region offset");
            const int slot = y * kColumns + x;
            require(!seen[slot], "Dense region duplicated one location");
            seen[slot] = true;
            require_corners(tag.corners_px_, first + cv::Point(x * kPitch, y * kPitch), kSide,
                            (x + y) % 4);
            const auto other = std::find_if(
                    unscaled_tags.begin(), unscaled_tags.end(), [&tag](const auto& candidate) {
                        const auto delta = tag.corners_px_[0] - candidate.corners_px_[0];
                        return delta.dot(delta) < 1.0F;
                    });
            require(other != unscaled_tags.end(), "Unscaled dense tag location differs");
            resized_corners |= tag.corners_px_ != other->corners_px_;
        }
        require(resized_corners, "Dense fixture did not exercise the upsample branch");
        require(baseline.detect(image).empty(), "Fixture did not isolate the dense path");
        require(capped_detector.detect(image).empty(),
                "Dense region exceeded its source pixel cap");
        require(cv::countNonZero(before != backing) == 0, "Dense region modified input pixels");
    }

    // Exercise the fallback and merge with an ordinary native tag in the same frame.
    put_tag(image, {207, 123}, 4, 1, 0, 0, 250);
    automatic.native_low_percentile_floor_ = 0;
    automatic.native_high_percentile_ceiling_ = 255;
    automatic.native_max_dark_per_mille_ = 1000;
    dense.minimum_verified_tags_ = 100;
    dense.replace_minimum_tags_ = 100;
    Detector fallback(config);
    const auto original = fallback.detect(image);
    require(original.size() == 1, "Uncertain dense region did not fall back to the ordinary route");
    require_corners(original[0].corners_px_, {207, 123}, 32, 1);

    dense.minimum_verified_tags_ = 6;
    Detector merging(config);
    const auto merged = merging.detect(image);
    require(merged.size() == 1 + kColumns * kRows, "Merging lost or duplicated a marker");
    require(merged[0].id_ == original[0].id_ && merged[0].corners_px_ == original[0].corners_px_,
            "Merging changed the ordinary result");

    image.setTo(250);
    require(detector.detect(image).empty(), "Dense search retained results from a preceding frame");
    dense.maximum_roi_pixels_ = 0;
    require_invalid([&] { Detector invalid(config); });
    dense.maximum_roi_pixels_ = 1;
    dense.upsample_sharpen_percent_ = -1;
    require_invalid([&] { Detector invalid(config); });
    dense.upsample_sharpen_percent_ = 8;
    dense.minimum_verified_tags_ = 101;
    require_invalid([&] { Detector invalid(config); });
}

void check_border_detector() {
    Config config;
    config.border_.emplace();
    config.automatic_.emplace();
    Detector detector(config);

    cv::Mat backing(244, 320, CV_8UC1, cv::Scalar(210));
    constexpr int kSide = 64;
    const cv::Rect view(20, 25, 260, 185);
    cv::Mat frame = backing(view);
    for (int side = 0; side < 4; ++side) {
        backing.setTo(210);
        cv::Point origin(96, 92);
        if (side == 0)
            origin.y = view.y - 12;
        if (side == 1)
            origin.x = view.x + view.width - kSide + 12;
        if (side == 2)
            origin.y = view.y + view.height - kSide + 12;
        if (side == 3)
            origin.x = view.x - 12;
        put_tag(backing, origin, 8, side, 0, 16, 240);
        const cv::Mat before = backing.clone();
        const auto tags = detector.detect(frame);
        require(tags.size() == 1 && tags[0].id_ == kTagId && tags[0].corrected_bits_ == 0,
                "Border search missed a partially visible marker");
        require_corners(tags[0].corners_px_, origin - view.tl(), kSide, side);
        require(cv::countNonZero(before != backing) == 0, "Border search wrote input pixels");
    }

    backing.setTo(210);
    frame(cv::Rect(25, 0, 43, 90)).setTo(16);
    require(detector.detect(frame).empty(), "A bare border stripe decoded");
    require(detector.detect(cv::Mat(1, 640, CV_8UC1, cv::Scalar(200))).empty(),
            "A single-row image returned a border tag");
    require(detector.detect(cv::Mat(480, 1, CV_8UC1, cv::Scalar(200))).empty(),
            "A single-column image returned a border tag");
    cv::Mat invalid_stride = frame;
    invalid_stride.step[0] = frame.cols - 1;
    require_invalid([&] { detector.detect(invalid_stride); });
    config.border_->minimum_known_cells_ = 12;
    require_invalid([&] { Detector invalid(config); });
}

void check_validation(Detector& detector) {
    Config config;
    config.maximum_corrected_bits_ = 4;
    require_invalid([&] { Detector invalid(config); });
    config.maximum_corrected_bits_ = 3;
    config.minimum_contrast_ = std::numeric_limits<float>::quiet_NaN();
    require_invalid([&] { Detector invalid(config); });
    require_invalid([&] { detector.detect(cv::Mat{}); });
    require_invalid([&] { detector.detect(cv::Mat(32, 32, CV_8UC3)); });
    require_invalid([&] { detector.detect(cv::Mat(1, 32768, CV_8UC1)); });

    cv::Mat gray(10, 20, CV_8UC1);
    cv::Mat bad_stride = gray;
    bad_stride.step[0] = 19;
    require_invalid([&] { detector.detect(bad_stride); });
    cv::Mat truncated = gray;
    truncated.dataend = truncated.data + 199;
    require_invalid([&] { detector.detect(truncated); });
}

void check_video_recovery() {
    Config config;
    config.components_.emplace();
    Detector stateless(config);
    config.recovery_.emplace();
    Detector video(config);
    constexpr int kCell = 8;
    constexpr int kSide = kCell * kCells;
    constexpr int kRotation = 2;
    constexpr int kLight = 221;
    constexpr int kDimDark = 112;
    const cv::Point first(37, 43), second(111, 45), shift(8, 5);
    cv::Mat backing(187, 331, CV_8UC1, cv::Scalar(1));
    cv::Mat frame = backing(cv::Rect(13, 11, 271, 151));
    const auto render_pair = [&](cv::Point offset, int first_dark, int second_dark) {
        frame.setTo(kLight);
        put_tag(frame, first + offset, kCell, kRotation, 0, first_dark, kLight);
        put_tag(frame, second + offset, kCell, kRotation, 0, second_dark, kLight);
    };
    render_pair({}, 100, 100);
    const auto expected = stateless.detect(frame);
    const auto seeded = video.detect(frame);
    require(expected.size() == 2 && seeded.size() == 2, "Video did not discover both positions");
    for (std::size_t i = 0; i < expected.size(); ++i)
        require(seeded[i].id_ == expected[i].id_ &&
                        seeded[i].corners_px_ == expected[i].corners_px_ &&
                        seeded[i].corrected_bits_ == expected[i].corrected_bits_,
                "Video altered first-frame discovery results");

    // Overwrite the same pitched storage before the next call. Recovery owns its samples.
    render_pair(shift, kDimDark, kDimDark);
    const cv::Mat before = backing.clone();
    require(stateless.detect(frame).empty(), "Fixture failed to hide tags from ordinary search");
    require_invalid([&] { video.detect(cv::Mat()); });
    const auto recovered = video.detect(frame);
    require(recovered.size() == 2, "Video lost a nearby same-ID tag");
    require(cv::countNonZero(before != backing) == 0, "Video changed the pitched input");
    for (const auto& detection : recovered) {
        require(detection.id_ == kTagId && detection.corrected_bits_ == 0,
                "Video misdecoded the translated current marker");
        const bool is_left = detection.corners_px_[0].x < second.x;
        require_corners(detection.corners_px_, (is_left ? first : second) + shift, kSide,
                        kRotation);
    }

    render_pair(shift, kDimDark, 100);
    const auto ordinary = stateless.detect(frame);
    const auto supplemented = video.detect(frame);
    require(ordinary.size() == 1 && supplemented.size() == 2 &&
                    supplemented.front().corners_px_ == ordinary.front().corners_px_,
            "Recovery replaced or duplicated ordinary discovery");
    frame.setTo(kLight);
    require(video.detect(frame).empty(), "A removed tag survived video verification");
    render_pair(shift, kDimDark, kDimDark);
    require(video.detect(frame).empty(), "An old tag survived after an empty frame");

    const auto seed = [&] {
        render_pair({}, 100, 100);
        require(video.detect(frame).size() == 2, "Video seed fixture missed a tag");
        render_pair({}, kDimDark, kDimDark);
    };
    seed();
    video.reset();
    require(video.detect(frame).empty(), "Explicit reset kept cross-sequence candidates");
    seed();
    require(video.detect(frame(cv::Rect(0, 0, frame.cols - 1, frame.rows))).empty(),
            "Resolution change reused cross-resolution candidates");
    require(video.detect(frame).empty(), "Resolution change retained the older tracks");

    config.recovery_->maximum_tracks_ = 1;
    Detector capped(config);
    render_pair({}, 100, 100);
    require(capped.detect(frame).size() == 2, "Track cap limited full discovery");
    render_pair({}, kDimDark, kDimDark);
    require(capped.detect(frame).size() == 1, "Track cap failed to bound recovery");

    // At a fixed pose, replacement must come from a current exact payload.
    config.recovery_->maximum_translation_px_ = 0;
    Detector replacement(config);
    for (const int replacement_id : {kTagId, 72}) {
        for (const int flips : {1, 0}) {
            replacement.reset();
            frame.setTo(kLight);
            put_tag(frame, first, kCell, 0, 0, 100, kLight);
            require(replacement.detect(frame).size() == 1, "Failed to seed replacement test");
            frame.setTo(kLight);
            put_tag(frame, first, kCell, 1, flips, kDimDark, kLight, replacement_id);
            const auto found = replacement.detect(frame);
            if (flips) {
                require(found.empty(), "A corrected replacement payload was accepted");
            } else {
                require(found.size() == 1 && found[0].id_ == replacement_id,
                        "Replacement used the previous ID");
                require_corners(found[0].corners_px_, first, kSide, 1);
            }
        }
    }

    config.components_.reset();
    config.automatic_.emplace();
    config.automatic_->native_low_percentile_floor_ = 0;
    config.automatic_->native_high_percentile_ceiling_ = 255;
    config.automatic_->native_max_dark_per_mille_ = 1000;
    config.dark_threshold_ = 106;
    config.bright_threshold_ = 106;
    Detector automatic(config);
    render_pair({}, 100, 100);
    require(automatic.detect(frame).size() == 2, "Automatic discovery changed with recovery");
    render_pair({}, kDimDark, kDimDark);
    require(automatic.detect(frame).size() == 1, "Automatic recovery did not obey the track cap");

    config.recovery_->maximum_tracks_ = 0;
    require_invalid([&] { Detector invalid(config); });
    config.recovery_->maximum_tracks_ = 1;
    config.recovery_->maximum_translation_px_ = 33;
    require_invalid([&] { Detector invalid(config); });
}

void check_uniform_raster() {
    using aruco3cuda::apriltag_small_cpu::Detection;
    aruco3cuda::apriltag_small_cpu::detail::UniformRaster raster;
    cv::Mat backing(109, 193, CV_8U, cv::Scalar(231));
    cv::Mat gray = backing(cv::Rect(4, 3, 179, 99));
    const cv::Point origin(79, 29);
    for (const bool halo : {false, true}) {
        for (int rotation = 0; rotation < 4; ++rotation) {
            gray.setTo(218);
            if (halo)
                gray(cv::Rect(origin.x - 5, origin.y - 5, 50, 50)).setTo(242);
            put_tag(gray, origin, 5, rotation, 1, 25, 242);
            std::vector<Detection> tags;
            require(raster.read(gray, tags) && tags.size() == 1 && tags.front().id_ == kTagId &&
                            tags.front().corrected_bits_ == 1,
                    "Exact raster decode or bit count");
            require_corners(tags.front().corners_px_, origin, 40, rotation);
            gray.at<std::uint8_t>(gray.rows - 2, 7) = 217;
            tags.clear();
            require(!raster.read(gray, tags) && tags.empty(),
                    "Unexplained image change skipped discovery");
        }
    }
    gray.setTo(218);
    std::vector<Detection> tags;
    require(raster.read(gray, tags) && tags.empty(), "Uniform pitched raster");
    put_tag(gray, origin, 5, 0, 4, 25, 242);
    require(raster.read(gray, tags) && tags.empty(), "Over-budget raster code accepted");
    gray.at<std::uint8_t>(origin.y + 1, origin.x + 1) = 26;
    require(!raster.read(gray, tags) && tags.empty(), "Nonuniform cell skipped ordinary search");
}

void check_video_detector() {
    using aruco3cuda::apriltag_small_cpu::VideoConfig;
    using aruco3cuda::apriltag_small_cpu::VideoDetector;
    VideoConfig config;
    config.full_search_interval_ = 3;
    VideoDetector detector(config);
    cv::Mat parent(251, 471, CV_8UC1, cv::Scalar(255));
    const cv::Rect roi(13, 17, 440, 220);
    cv::Mat frame = parent(roi);
    for (int cycle = 0; cycle < 2; ++cycle) {
        frame.setTo(196);
        require(detector.detect(frame).empty(), "Empty initial video frame");
        // Change the caller-owned pitched input in place between calls.
        for (int index = 0; index < 4; ++index) {
            frame.setTo(196);
            const cv::Point first(41 + 3 * index, 46), second(241 - 4 * index, 115);
            put_tag(frame, first, 6, 1, 0, 40, 196);
            put_tag(frame, second, 4, 2, 0, 40, 196);
            const auto results = detector.detect(frame);
            require(results.size() == 2, "Lost duplicate IDs in different locations");
            for (const auto& tag : results) {
                require(tag.id_ == kTagId, "Video reported an unexpected ID");
                if (tag.corners_px_[0].x < 180)
                    require_corners(tag.corners_px_, first, 48, 1);
                else
                    require_corners(tag.corners_px_, second, 32, 2);
            }
        }
        frame.setTo(196);
        require(detector.detect(frame).empty(), "Video repeated tags after disappearance");
        require_invalid([&] { detector.detect(cv::Mat{}); });
        require_invalid([&] { detector.detect(cv::Mat(16, 16, CV_8UC3)); });
        require(detector.detect(frame(cv::Rect(0, 0, 1, 1))).empty(), "Tiny video frame");
        detector.reset();
    }
    // Recover a clipped tag while still discovering a remote, smaller tag in
    // the same sparse, high-resolution frame. Changing the contents must also
    // clear all previous islands on the next call.
    cv::Mat large(1087, 1953, CV_8UC1, cv::Scalar(221));
    const cv::Rect crop(29, 3, 1921, 1081);
    put_tag(large, crop.tl() + cv::Point(-12, 144), 8, 0, 0, 20, 232, 53);
    put_tag(large, crop.tl() + cv::Point(1511, 817), 5, 1, 0, 83, 221, 79);
    const auto simultaneous = detector.detect(large(crop));
    require(simultaneous.size() == 2, "Clipped and interior tags must coexist");
    for (const auto& tag : simultaneous) {
        if (tag.id_ == 79)
            require_corners(tag.corners_px_, {1511, 817}, 40, 1);
        else
            require(tag.id_ == 53, "Unexpected tag in sparse border scene");
    }
    large.setTo(221);
    require(detector.detect(large(crop)).empty(), "Border search retained an old detection");
    cv::Mat textured(489, 671, CV_8U);
    const cv::Rect textured_crop(21, 3, 642, 481);
    for (int y = 0; y < textured.rows; ++y) {
        auto* row = textured.ptr<std::uint8_t>(y);
        for (int x = 0; x < textured.cols; ++x)
            row[x] = static_cast<std::uint8_t>(104 + (x + y / 2) % 51);
    }
    put_tag(textured, textured_crop.tl() + cv::Point(-8, 60), 8, 1, 0, 36, 195, 42);
    put_tag(textured, textured_crop.tl() + cv::Point(325, 260), 6, 2, 0, 29, 202, 63);
    const auto midtone = detector.detect(textured(textured_crop));
    require(midtone.size() == 2 &&
                    std::any_of(midtone.begin(), midtone.end(),
                                [](const auto& tag) { return tag.id_ == 42; }) &&
                    std::any_of(midtone.begin(), midtone.end(),
                                [](const auto& tag) { return tag.id_ == 63; }),
            "Midtone boundary discovery lost a simultaneous interior or clipped tag");
    config.full_search_interval_ = 0;
    require_invalid([&] { VideoDetector invalid(config); });
}

void check_midtone_edge_refinement() {
    using namespace aruco3cuda::apriltag_small_cpu;
    cv::Mat parent(178, 237, CV_8UC1, cv::Scalar(123));
    cv::Mat image = parent(cv::Rect(7, 5, 215, 161));
    put_tag(image, {11, 38}, 8, 0, 0, 38, 184);
    cv::GaussianBlur(image, image, {5, 5}, .8);
    const std::array<cv::Point2f, 4> truth{
            {{10.5F, 37.5F}, {74.5F, 37.5F}, {74.5F, 101.5F}, {10.5F, 101.5F}}};
    Detection seed{kTagId, truth, 0};
    for (auto& q : seed.corners_px_)
        q += cv::Point2f(.9F, -.65F);
    detail::BorderSearch search(BorderConfig{}, 3);
    std::vector<Detection> disabled{seed}, fitted{seed}, owned{seed};
    search.align_visible(image, disabled);
    search.align_visible(image, fitted, .2F);
    search.align_visible(image.clone(), owned, .2F);
    require(disabled[0].corners_px_ == seed.corners_px_,
            "Native edge margin zero changed the seed");
    require(fitted[0].corners_px_ == owned[0].corners_px_, "Native edge fit depends on row stride");
    for (int i = 0; i < 4; ++i)
        require(cv::norm(fitted[0].corners_px_[i] - truth[i]) < .18,
                "Fully visible print retained the boundary proposal offset");
    seed.id_ = kTagId + 1;
    std::vector<Detection> wrong{seed};
    search.align_visible(image, wrong, .2F);
    require(wrong[0].corners_px_ == seed.corners_px_, "Native edge fit accepted the wrong payload");

    cv::Mat cropped = image(cv::Rect(15, 0, image.cols - 15, image.rows));
    seed.id_ = kTagId;
    for (auto& q : seed.corners_px_)
        q.x -= 15;
    std::vector<Detection> clipped{seed}, reference{seed};
    search.align_visible(cropped, reference);
    search.align_visible(cropped, clipped, .2F);
    require(clipped[0].corners_px_ == reference[0].corners_px_,
            "Clipped print used the full edge fit");
    for (float ratio : {-.1F, 1.1F, std::numeric_limits<float>::quiet_NaN()}) {
        VideoConfig config;
        config.midtone_edge_refinement_margin_ratio_ = ratio;
        require_invalid([&] { VideoDetector invalid(config); });
    }
}

void check_dense_midrange_mask() {
    using aruco3cuda::apriltag_small_cpu::detail::dense_midrange_mask;
    cv::RNG rng(719358);
    // Deliberately give every matrix a different pitch and nonzero origin.
    for (int width : {1, 16, 63, 64, 65, 127, 128, 129, 193}) {
        for (int height : {1, 3, 4, 5, 17}) {
            const cv::Size size(width, height), coarse((width + 3) / 4, (height + 3) / 4);
            const cv::Size tiles((width + 15) / 16, (height + 15) / 16);
            auto view = [](cv::Size dimensions, int pad, int fill) {
                cv::Mat parent(dimensions.height + 3, dimensions.width + pad, CV_8UC1,
                               cv::Scalar(fill));
                return parent(cv::Rect(cv::Point(pad / 3, 1), dimensions));
            };
            auto gray = view(size, 13, 0), high = view(coarse, 9, 0), floor = view(coarse, 22, 0);
            auto enabled = view(tiles, 12, 0);
            rng.fill(gray, cv::RNG::UNIFORM, 0, 256);
            rng.fill(floor, cv::RNG::UNIFORM, 0, 256);
            rng.fill(high, cv::RNG::UNIFORM, 0, 256);
            cv::max(high, floor, high);
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x)
                    if (x % 4 != 3) {
                        const int midpoint = (floor.at<std::uint8_t>(y / 4, x / 4) +
                                              high.at<std::uint8_t>(y / 4, x / 4) + 1) /
                                             2;
                        gray.at<std::uint8_t>(y, x) =
                                cv::saturate_cast<std::uint8_t>(midpoint + (x % 4) - 1);
                    }
            for (int gate : {0, 1, 2}) {
                for (int y = 0; y < enabled.rows; ++y)
                    for (int x = 0; x < enabled.cols; ++x)
                        enabled.at<std::uint8_t>(y, x) =
                                gate == 2 ? ((x + y) % 2 ? 255 : 0) : gate * 255;
                for (int contrast : {0, 1, 24, 127, 254, 255}) {
                    auto low = view(coarse, 17, 0);
                    floor.copyTo(low);
                    auto out = view(size, 21, 73);
                    dense_midrange_mask(gray, low, high, enabled, contrast, out);
                    for (int y = 0; y < height; ++y)
                        for (int x = 0; x < width; ++x) {
                            const int a = floor.at<std::uint8_t>(y / 4, x / 4),
                                      b = high.at<std::uint8_t>(y / 4, x / 4);
                            const int expected =
                                    enabled.at<std::uint8_t>(y / 16, x / 16) && b - a > contrast &&
                                                    2 * gray.at<std::uint8_t>(y, x) < a + b
                                            ? 255
                                            : 0;
                            require(out.at<std::uint8_t>(y, x) == expected,
                                    "Dense midpoint mask differs from scalar oracle");
                        }
                    require(out.datastart[0] == 73 && out.data[-1] == 73,
                            "Dense mask modified storage outside the ROI");
                }
            }
        }
    }
}

void check_beveled_candidate() {
    using namespace aruco3cuda::apriltag_small_cpu;
    cv::Mat mask(64, 64, CV_8UC1, cv::Scalar(0));
    const std::vector<cv::Point> outline{{6, 12}, {15, 8}, {30, 14}, {33, 45}, {7, 37}};
    cv::fillConvexPoly(mask, outline, cv::Scalar(255));
    detail::ComponentCandidates finder;
    ComponentConfig config;
    config.minimum_quad_hull_ratio_ = .5;
    std::vector<std::vector<cv::Point>> polygons;
    finder.extract(mask, config, polygons);
    require(polygons.empty(), "Ordinary component search accepted a beveled contour");
    detail::SupplementalComponents supplemental;
    finder.extract(mask, config, polygons, &supplemental);
    require(polygons.empty() && supplemental.polygons_.size() == 1 &&
                    supplemental.polygons_.front().size() == 4,
            "Could not isolate supplemental beveled quadrilateral");
    const auto& quad = supplemental.polygons_.front();
    require(std::any_of(quad.begin(), quad.end(),
                        [](cv::Point p) { return cv::norm(p - cv::Point(6, 4)) < 2.5; }),
            "Extrapolated the wrong beveled corner");
}

void check_dense_board_search() {
    using namespace aruco3cuda::apriltag_small_cpu::detail;
    DenseBoardSearch detector;
    cv::Mat storage(141, 303, CV_8UC1, cv::Scalar(39));
    cv::Mat board = storage(cv::Rect(5, 7, 287, 125));
    board.setTo(174);
    for (int i = 0; i < 12; ++i)
        put_tag(board, {13 + (i % 6) * 42, 21 + (i / 6) * 51}, 3, i % 4, 0, 48, 174, i);
    const auto before = storage.clone();
    const auto tags = detector.detect(board);
    require(tags.size() == 12, "Dense board lost a marker");
    std::array<bool, 12> seen{};
    for (const auto& tag : tags) {
        const int id = tag.id_;
        require(id >= 0 && id < 12 && !seen[id], "Incorrect dense-board ID or duplicate");
        seen[id] = true;
        require_corners(tag.corners_px_, {13 + (id % 6) * 42, 21 + (id / 6) * 51}, 24, id % 4);
    }
    require(cv::countNonZero(storage != before) == 0, "Dense board modified strided input");
    board.setTo(174);
    require(detector.detect(board).empty(), "Dense board replayed disappeared tags");
    put_tag(board, {14, 17}, 3, 0, 0, 48, 174);
    require(detector.detect(board).empty(), "Sparse input passed the dense candidate gate");
    cv::Mat wide(9, 16401, CV_8UC1, cv::Scalar(172));
    require(detector.detect(wide).empty(), "Native-width dense mask accepted uniform input");
    require_invalid([&] { detector.detect(cv::Mat{}); });
    DenseBoardConfig invalid;
    invalid.maximum_corrected_bits_ = 3;
    require_invalid([&] { DenseBoardSearch bad(invalid); });
    invalid = {};
    invalid.minimum_planar_tags_ = 7;
    require_invalid([&] { DenseBoardSearch bad(invalid); });
    invalid = {};
    invalid.maximum_lattice_center_shift_px_ = std::numeric_limits<float>::quiet_NaN();
    require_invalid([&] { DenseBoardSearch bad(invalid); });
    invalid = {};
    invalid.maximum_planar_correction_px_ = -1;
    require_invalid([&] { DenseBoardSearch bad(invalid); });
    invalid = {};
    invalid.minimum_normalized_bow_ = -1;
    require_invalid([&] { DenseBoardSearch bad(invalid); });
    invalid = {};
    invalid.minimum_curvature_r_squared_ = std::numeric_limits<float>::quiet_NaN();
    require_invalid([&] { DenseBoardSearch bad(invalid); });
}

void check_planar_tag_shapes() {
    using aruco3cuda::apriltag_small_cpu::Detection;
    using namespace aruco3cuda::apriltag_small_cpu::detail;
    const Quad unit{{{-1, -1}, {1, -1}, {1, 1}, {-1, 1}}};
    std::vector<Detection> truth, measured;
    cv::RNG noise(1901);
    for (int i = 0; i < 64; ++i) {
        const Point center(40 + 40 * (i % 8) + 2.8F * (i / 8), 32 + 35 * (i / 8));
        Quad corners;
        for (int j = 0; j < 4; ++j) {
            const auto axis = unit[(j + i) % 4];
            const float radius = i == 1 ? 18.F : 11.F;
            const float shear = i == 2 ? .35F : .08F;
            corners[j] = center + radius * Point(axis.x + shear * axis.y, axis.y);
        }
        truth.push_back({100 + i, corners, i % 3});
        for (auto& corner : corners)
            corner += Point(noise.uniform(-.19F, .19F), noise.uniform(-.19F, .19F));
        measured.push_back({100 + i, corners, i % 3});
    }
    auto fitted = measured;
    refine_planar_tag_shapes(fitted, 16, .7F);
    double original_error = 0, final_error = 0;
    for (std::size_t i = 0; i < fitted.size(); ++i) {
        require(fitted[i].id_ == measured[i].id_ &&
                        fitted[i].corrected_bits_ == measured[i].corrected_bits_,
                "Planar shape fit changed the decoded payload");
        if (i == 1 || i == 2) {
            require(fitted[i].corners_px_ == measured[i].corners_px_,
                    "Planar shape fit moved a marker of a different size or pose");
            continue;
        }
        Point center_change{};
        for (int j = 0; j < 4; ++j) {
            const Point before = measured[i].corners_px_[j] - truth[i].corners_px_[j];
            const Point after = fitted[i].corners_px_[j] - truth[i].corners_px_[j];
            original_error += before.dot(before);
            final_error += after.dot(after);
            center_change += after - before;
        }
        require(length(center_change) < .001F, "Planar shape fit moved a tag center");
    }
    require(final_error < .65 * original_error, "Consistent rotated planar shapes did not improve");
    for (auto& corner : measured[3].corners_px_)
        corner += Point(2.5F, 1.F);
    fitted = measured;
    refine_regular_tag_centers(fitted, 16, .5);
    double before_centers = 0, after_centers = 0;
    for (std::size_t i = 0; i < fitted.size(); ++i) {
        if (i >= 1 && i <= 3) {
            require(fitted[i].corners_px_ == measured[i].corners_px_,
                    "Lattice moved a different size, pose or off-grid marker");
            continue;
        }
        const auto projected_center = [](const Quad& q) {
            const auto d = q[2] - q[0], e = q[3] - q[1];
            return q[0] + d * (cross(q[1] - q[0], e) / cross(d, e));
        };
        const auto expected = projected_center(truth[i].corners_px_);
        const auto before = projected_center(measured[i].corners_px_) - expected;
        const auto after = projected_center(fitted[i].corners_px_) - expected;
        before_centers += before.dot(before);
        after_centers += after.dot(after);
    }
    require(after_centers < .6 * before_centers, "Lattice did not refine regular tag centers");
    auto irregular = measured;
    for (auto& tag : irregular) {
        const Point shift(noise.uniform(-6.F, 6.F), noise.uniform(-6.F, 6.F));
        for (auto& corner : tag.corners_px_)
            corner += shift;
    }
    fitted = irregular;
    refine_regular_tag_centers(fitted, 16, .5);
    for (std::size_t i = 0; i < fitted.size(); ++i)
        require(fitted[i].corners_px_ == irregular[i].corners_px_,
                "Irregular centers passed lattice consensus");
    for (std::size_t i = 0; i < measured.size(); ++i) {
        const float factor = i % 2 ? 1.F : 1.6F;
        const auto& q = measured[i].corners_px_;
        const auto center = .25F * (q[0] + q[1] + q[2] + q[3]);
        for (auto& corner : measured[i].corners_px_)
            corner = center + factor * (corner - center);
    }
    fitted = measured;
    refine_planar_tag_shapes(fitted, 16, .7F);
    for (std::size_t i = 0; i < fitted.size(); ++i)
        require(fitted[i].corners_px_ == measured[i].corners_px_,
                "Inconsistent board scale passed consensus");
}

void check_edge_curvature_consensus() {
    using namespace aruco3cuda::apriltag_small_cpu::detail;
    EdgeCurvatureEvidence measured, flat, random, insufficient, parallel;
    for (int i = 0; i < 48; ++i) {
        const float theta = float(CV_PI * (i % 7) / 7);
        const Point normal(std::cos(theta), std::sin(theta));
        const double radius = .0625 * ((i % 13) - 6);
        const double sag = .37 * radius + .015 * normal.x - .022 * normal.y;
        measured.add(radius, normal, sag, 1);
        flat.add(radius, normal, .01 * sag, 1);
        random.add(radius, normal, sag, 1);
        random.add(radius, normal, -sag, 1);
        parallel.add(.25, {1, 0}, .3, 1);
        if (i < 12)
            insufficient.add(radius, normal, sag, 1);
    }
    require(measured.coherent(.045, .2),
            "Missed coherent native curvature with shifted optical center");
    require(!flat.coherent(.045, .2), "Near-zero source curvature changed a planar fit");
    require(!random.coherent(.045, .2), "Inconsistent native edges passed the radial gate");
    require(!insufficient.coherent(.045, .2), "Sparse native samples passed the radial gate");
    require(!parallel.coherent(.045, .2), "A single edge direction entered the radial fit");
}

void check_refined_board_codes() {
    using namespace aruco3cuda::apriltag_small_cpu::detail;
    DenseBoardConfig native;
    native.maximum_planar_correction_px_ = 0;
    native.maximum_lattice_center_shift_px_ = 0;
    DenseBoardSearch independent(native), refined;
    cv::Mat board(520, 520, CV_8UC1, cv::Scalar(165));
    for (int i = 0; i < 64; ++i)
        put_tag(board, {11 + 63 * (i % 8), 13 + 63 * (i / 8)}, 5, i % 4, 0, 66, 165, i);
    const auto transform = cv::getRotationMatrix2D({259.5F, 259.5F}, 6., .90);
    cv::warpAffine(board, board, transform, board.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                   cv::Scalar(165));
    cv::Mat frame;
    cv::resize(board, frame, {179, 179}, 0, 0, cv::INTER_AREA);
    cv::GaussianBlur(frame, frame, {3, 3}, .48);
    const auto original = independent.detect(frame);
    const auto adjusted = refined.detect(frame);
    require(original.size() == adjusted.size() && adjusted.size() >= 40,
            "Dense grid verification unexpectedly changed marker count");
    int checked = 0;
    for (std::size_t i = 0; i < adjusted.size(); ++i) {
        const auto& tag = adjusted[i];
        require(tag.id_ == original[i].id_, "Planar refinement changed a decoded ID");
        if (tag.corners_px_ == original[i].corners_px_)
            continue;
        const auto read = decode(frame, tag.corners_px_, native.minimum_ring_matches_,
                                 native.minimum_decode_contrast_, native.maximum_corrected_bits_);
        if (read.id >= 0) {
            require(read.id == tag.id_ && read.rotation == 0 &&
                            read.distance == tag.corrected_bits_,
                    "Planar corners disagree with the final canonical payload");
            ++checked;
        }
    }
    require(checked >= 16, "Too few independently re-readable refined grids in fixture");
}

void check_sparse_board_probe() {
    using namespace aruco3cuda::apriltag_small_cpu;
    detail::BoardSeedPolicy policy{3, 6, 80, 6};
    const auto square = [](int id, float x, float y, float side) {
        return Detection{id,
                         {cv::Point2f(x, y), cv::Point2f(x + side, y),
                          cv::Point2f(x + side, y + side), cv::Point2f(x, y + side)},
                         0};
    };
    std::vector<Detection> seeds{square(0, 0, 0, 16), square(1, 40, 0, 16), square(2, 136, 0, 16)};
    require(!policy.dispersed(seeds),
            "A neighbor exactly six sides away should retain full search");
    seeds.back() = square(2, 140, 0, 16);
    require(policy.dispersed(seeds), "Isolated current seeds did not qualify");
    seeds.back().id_ = seeds.front().id_;
    require(!policy.dispersed(seeds), "Repeated IDs entered the sparse probe");
    seeds.back() = square(2, 160, 0, 81);
    require(!policy.dispersed(seeds), "Too few small seeds entered the sparse probe");

    // The distant quiet zone contains a faint cluster outside every strong seed's neighborhood.
    cv::Mat backing(542, 967, CV_8U);
    cv::Mat frame = backing(cv::Rect(2, 1, 960, 540));
    for (int y = 0; y < frame.rows; ++y) {
        auto* row = frame.ptr<std::uint8_t>(y);
        for (int x = 0; x < frame.cols; ++x)
            row[x] = static_cast<std::uint8_t>(
                    20 + (19 * (x / 12) + 31 * (y / 9) + (x / 53) * y / 41) % 17 * 11);
    }
    std::array<cv::Point, 3> origins{{{70, 66}, {795, 77}, {100, 443}}};
    for (int i = 0; i < 3; ++i) {
        const auto at = origins[i];
        frame(cv::Rect(at.x - 6, at.y - 6, 36, 36)).setTo(224);
        put_tag(frame, at, 3, i, 0, 13, 224, 300 + i);
    }
    frame(cv::Rect(451, 260, 110, 85)).setTo(159);
    for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 4; ++x)
            put_tag(frame, {460 + 24 * x, 269 + 24 * y}, 2, (x + y) % 4, 0, 105, 159,
                    320 + 4 * y + x);
    VideoConfig complete;
    complete.dispersed_seed_spacing_sides_ = 0;
    VideoDetector baseline(complete), trial;
    const auto expected = baseline.detect(frame);
    const auto observed = trial.detect(frame);
    require(expected.size() == 15 && observed.size() == expected.size(),
            "Sparse discovery lost prints outside the strong seeds");
    for (const auto& before : expected)
        require(std::any_of(observed.begin(), observed.end(),
                            [&](const auto& tag) {
                                return tag.id_ == before.id_ &&
                                       tag.corners_px_ == before.corners_px_;
                            }),
                "Sparse verification changed the reference ID or ordered corners");
    complete.dispersed_seed_spacing_sides_ = -1;
    require_invalid([&] { VideoDetector invalid(complete); });
    complete.dispersed_seed_spacing_sides_ = std::numeric_limits<float>::infinity();
    require_invalid([&] { VideoDetector invalid(complete); });
}

void check_oblique_board_neighbors() {
    using namespace aruco3cuda::apriltag_small_cpu;
    using namespace aruco3cuda::apriltag_small_cpu::detail;
    cv::Mat backing(365, 236, CV_8UC1, cv::Scalar(231));
    const cv::Mat frame = backing(cv::Rect(5, 9, 220, 340));
    const cv::Size tag_size(16, 80);
    const auto origin = [](int col, int row) { return cv::Point(50 + 22 * col, 34 + 110 * row); };
    const auto draw = [&](cv::Point at, cv::Size size, int id, int rotation) {
        cv::Mat square(80, 80, CV_8U);
        put_tag(square, {}, 10, rotation, 0, 12, 231, id);
        cv::resize(square, frame(cv::Rect(at, size)), size, 0, 0, cv::INTER_AREA);
        const Point a = Point(at) - Point(.5F, .5F);
        Quad bounds{{a, a + Point(size.width, 0), a + Point(size.width, size.height),
                     a + Point(0, size.height)}};
        std::rotate(bounds.begin(), bounds.begin() + rotation, bounds.end());
        return Detection{id, bounds, 0};
    };
    std::vector<Detection> seeds;
    for (int row = 0; row < 2; ++row)
        for (int column = 0; column < 3; ++column) {
            const int ix = row * 3 + column;
            seeds.push_back(draw(origin(column, row), tag_size, 21 + ix * 17, (ix + 1) % 4));
        }
    const auto target = draw(origin(3, 0), tag_size, 402, 3);
    // A real print of another size near the adjacent site must keep its own geometry.
    draw(origin(3, 1) + cv::Point(4, 20), {8, 40}, 7, 1);
    auto tags = seeds;
    LargeBoardSearch search;
    search.recover_oblique(frame, tags);
    require(tags.size() == seeds.size() + 1 && tags.back().id_ == target.id_,
            "Oblique board did not recover exactly the independently printed neighbor");
    for (std::size_t j = 0; j < seeds.size(); ++j)
        require(tags[j].id_ == seeds[j].id_ && tags[j].corners_px_ == seeds[j].corners_px_,
                "Oblique search modified an existing source measurement");
    for (int j = 0; j < 4; ++j)
        require(length(tags.back().corners_px_[j] - target.corners_px_[j]) < .7F,
                "Oblique neighbor lost native geometry or decoded rotation");
    search.recover_oblique(frame, tags);
    require(tags.size() == seeds.size() + 1, "Oblique search duplicated an occupied site");

    LargeBoardConfig disabled;
    disabled.oblique_.maximum_area_ratio_ = 0;
    tags = seeds;
    LargeBoardSearch(disabled).recover_oblique(frame, tags);
    require(tags.size() == seeds.size(), "Disabled oblique recovery emitted a neighbor");
    tags = seeds;
    frame(cv::Rect(origin(3, 0), tag_size)).setTo(231);
    search.recover_oblique(frame, tags);
    require(tags.size() == seeds.size(), "Missing or differently sized print was extrapolated");
    disabled.oblique_.maximum_area_ratio_ = std::numeric_limits<float>::quiet_NaN();
    require_invalid([&] { LargeBoardSearch rejected(disabled); });
    disabled.oblique_.maximum_area_ratio_ = .26F;
    disabled.oblique_.minimum_seeds_ = 2;
    require_invalid([&] { LargeBoardSearch rejected(disabled); });
}

void check_stationary_patch_retry() {
    using aruco3cuda::apriltag_small_cpu::VideoConfig;
    using aruco3cuda::apriltag_small_cpu::VideoDetector;
    constexpr int kWidth = 3840, kHeight = 2160, kCellPx = 64;
    constexpr int kPrintSidePx = 8 * kCellPx, kId = 448;
    const cv::Point origin(1347, 1015);
    cv::Mat backing(kHeight + 4, kWidth + 12, CV_8U, cv::Scalar(17));
    auto valid = backing(cv::Rect(5, 2, kWidth, kHeight));
    std::array<double, kWidth> columns;
    for (int x = 0; x < kWidth; ++x)
        columns[x] = std::cos(x / 32.);
    for (int y = 0; y < kHeight; ++y) {
        auto* row = valid.ptr<std::uint8_t>(y);
        const double luma = 12 * std::sin(y / 37.);
        for (int x = 0; x < kWidth; ++x)
            row[x] = static_cast<std::uint8_t>(132 + luma * columns[x]);
    }
    const int paper_side = kPrintSidePx + 2 * kCellPx;
    valid(cv::Rect(origin - cv::Point(kCellPx, kCellPx), cv::Size(paper_side, paper_side)))
            .setTo(213);
    const auto background = valid.clone();
    put_tag(valid, origin, kCellPx, 0, 0, 65, 213, kId);
    auto invalid = valid.clone();
    const auto center = cv::Rect(origin + cv::Point(3 * kCellPx, 3 * kCellPx),
                                 cv::Size(2 * kCellPx, 2 * kCellPx));
    cv::subtract(cv::Scalar(65 + 213), invalid(center), invalid(center));

    VideoConfig config;
    VideoDetector retry(config);
    config.stationary_patch_retry_.enabled_ = false;
    VideoDetector ordinary(config);
    const auto expect_tag = [&](const auto& observed) {
        require(observed.size() == 1 && observed.front().id_ == kId,
                "Stationary retry missed or duplicated the source tag");
        const cv::Point2f first = cv::Point2f(origin) - cv::Point2f(.5F, .5F);
        require(cv::norm(observed.front().corners_px_[0] - first) < 1.,
                "Stationary retry returned coordinates relative to the crop");
    };
    for (int pass = 0; pass < 2; ++pass) {
        retry.reset();
        ordinary.reset();
        for (int before = 0; before < 2; ++before) {
            require(retry.detect(invalid).empty(), "Invalid payload emitted a tag");
            require(ordinary.detect(invalid).empty(), "Invalid payload emitted a tag");
        }
        expect_tag(retry.detect(valid));
        require(ordinary.detect(valid).empty(), "Fixture no longer reaches the retry route");
        expect_tag(retry.detect(valid));
        require(retry.detect(invalid).empty(), "Changed payload reused a prior detection");
        expect_tag(retry.detect(valid));
        require(retry.detect(background).empty(), "Stationary retry returned a removed print");
    }
    require(backing.at<std::uint8_t>(0, 0) == 17, "Stationary retry wrote outside the input view");
    config.stationary_patch_retry_.small_change_limit_px_ = 0;
    require_invalid([&] { VideoDetector bad(config); });
}

void check_mixed_large_board() {
    using aruco3cuda::apriltag_small_cpu::VideoDetector;
    using aruco3cuda::apriltag_small_cpu::detail::LargeBoardConfig;
    using aruco3cuda::apriltag_small_cpu::detail::LargeBoardSearch;
    cv::Mat backing(1007, 1401, CV_8UC1, cv::Scalar(7));
    // Start the board at an unaligned ROI and leave partial downsample cells on every right edge.
    auto frame = backing(cv::Rect(9, 11, 1301, 961));
    for (int y = 0; y < frame.rows; ++y)
        for (int x = 0; x < frame.cols; ++x)
            frame.ptr<std::uint8_t>(y)[x] = static_cast<std::uint8_t>(62 + (x + y) * 122 / 2260);
    const std::array<cv::Point, 4> origins{{{54, 60}, {456, 60}, {54, 522}, {456, 522}}};
    for (int i = 0; i < 4; ++i) {
        const auto origin = origins[i];
        frame(cv::Rect(origin.x - 40, origin.y - 40, 400, 400)).setTo(237);
        put_tag(frame, origin, 40, i, 0, 8, 237, 30 + i);
    }
    const cv::Point small(1081, 325);
    auto small_patch = frame(cv::Rect(small.x - 6, small.y - 6, 44, 44));
    small_patch.setTo(167);
    put_tag(frame, small, 4, 1, 0, 68, 167, 210);
    for (int y = 0; y < small_patch.rows; ++y)
        for (int x = 0; x < small_patch.cols; ++x)
            small_patch.ptr<std::uint8_t>(y)[x] += x - 22;
    const auto before = backing.clone();

    LargeBoardSearch search;
    const auto& large = search.detect(frame);
    require(large.size() == 4, "Large board did not isolate the four reference seeds");
    const auto check = [](const auto& detections) {
        std::array<int, 5> counts{};
        for (const auto& tag : detections) {
            if (tag.id_ == 210)
                ++counts[4];
            else if (tag.id_ >= 30 && tag.id_ < 34)
                ++counts[tag.id_ - 30];
            else
                throw std::runtime_error("Unrelated decoded ID on the mixed-scale board");
        }
        require(counts == std::array<int, 5>{1, 1, 1, 1, 1},
                "Mixed-scale board is missing an ID or duplicated a location");
    };
    check(search.detect_mixed(frame));
    VideoDetector detector;
    check(detector.detect(frame));
    require(cv::countNonZero(before != backing) == 0, "Board search changed a pitched input");
    for (const auto levels : {std::array<int, 2>{16, 91}, std::array<int, 2>{138, 214}}) {
        small_patch.setTo(levels[1]);
        put_tag(frame, small, 4, 1, 0, levels[0], levels[1], 210);
        check(search.detect_mixed(frame));
        check(detector.detect(frame));
    }
    frame.setTo(100);
    require(search.detect_mixed(frame).empty() && detector.detect(frame).empty(),
            "Stale detection on an empty frame after the board");
    require(search.detect_mixed(frame(cv::Rect(0, 0, 1, 1))).empty(),
            "A one-pixel input returned a board");

    LargeBoardConfig invalid;
    invalid.adaptive_window_ = 4;
    require_invalid([&] { LargeBoardSearch rejected(invalid); });
    invalid.adaptive_window_ = 31;
    invalid.small_threshold_ = -1;
    require_invalid([&] { LargeBoardSearch rejected(invalid); });
}

void check_bounded_closed_approximation() {
    using aruco3cuda::apriltag_small_cpu::detail::approximate_closed_quad;
    std::array<std::vector<cv::Point>, 3> paths{{
            {{-2048, -2048}, {0, -2047}, {2047, -2048}, {2047, 0}, {2047, 2047}, {-2048, 2047}},
            {{1, 0}, {19, 0}, {20, 1}, {20, 19}, {19, 20}, {1, 20}, {0, 19}, {0, 1}},
            {{0, 0},
             {2, 3},
             {4, 0},
             {6, 3},
             {8, 0},
             {10, 3},
             {12, 0},
             {14, 3},
             {16, 0},
             {18, 3},
             {20, 0},
             {20, 20},
             {0, 20}},
    }};
    for (auto& path : paths) {
        for (std::size_t start = 0; start < path.size(); ++start) {
            for (double epsilon : {0., .5, 1., 2., 8.}) {
                std::vector<cv::Point> expected, actual;
                cv::approxPolyDP(path, expected, epsilon, true);
                const bool accepted = approximate_closed_quad(path, epsilon, actual);
                require(accepted == (expected.size() == 4),
                        "Bounded approximation changed the quad decision");
                require(!accepted || actual == expected,
                        "Bounded approximation changed ordered quad vertices");
            }
            std::rotate(path.begin(), path.begin() + 1, path.end());
        }
    }
}

void check_codebook_lookup() {
    using namespace aruco3cuda::apriltag_small_cpu::detail;
    const auto& dictionary = aruco3cuda::generated::kDictApriltag36h11Table;
    const int count = 4 * dictionary.code_count_;
    for (int i = 0; i < count; ++i) {
        auto word = dictionary.codes_[i];
        for (int distance = 0; distance <= kRadius; ++distance) {
            require(CodeIndex::match(word) == std::make_pair(i, distance),
                    "Hamming lookup lost a rotated code");
            require((CodeIndex::match(word, 2).first == i) == (distance <= 2),
                    "Hamming radius changed");
            // At distance three every 12-bit group has been corrupted.
            const int bit = 12 * (distance % 3) + (i + distance / 3) % 12;
            word ^= aruco3cuda::MarkerCode{1} << bit;
        }
    }
    std::uint64_t state = 0xe727a6;
    for (int sample = 0; sample < 2048; ++sample) {
        state = state * 6364136223846793005ULL + 1;
        const auto word = state >> 28;
        const int radius = sample % (kRadius + 1);
        std::pair<int, int> expected{-1, 37};
        for (int i = 0; i < count; ++i) {
            const int distance = bit_count(dictionary.codes_[i] ^ word);
            if (distance <= radius) {
                expected = {i, distance};
                break;
            }
        }
        require(CodeIndex::match(word, radius) == expected,
                "Word index disagrees with exhaustive search");
    }
}

void check_known_payload() {
    using namespace aruco3cuda::apriltag_small_cpu::detail;
    const int high_id = aruco3cuda::generated::kDictApriltag36h11Table.code_count_ - 1;
    cv::Mat storage(84, 99, CV_8UC1, cv::Scalar(205));
    cv::Mat frame = storage(cv::Rect(3, 7, 80, 70));
    const Quad quad{{{11.5F, 9.5F}, {59.5F, 9.5F}, {59.5F, 57.5F}, {11.5F, 57.5F}}};
    for (int rotation = 0; rotation < 4; ++rotation) {
        for (int errors = 0; errors <= 4; ++errors) {
            put_tag(frame, {12, 10}, 6, rotation, errors, 20, 205, high_id);
            const auto found = decode(frame, quad, 22, 9.F, errors);
            require(found.id == high_id && found.distance == errors,
                    "Discovery missed printed word");
            for (int r = 0; r < 4; ++r) {
                const auto verified = verify_payload(frame, quad, high_id, 22, 9.F, errors, r);
                require(r == found.rotation ? verified == errors : !verified,
                        "Known-word verification changed the orientation or correction count");
            }
            require(!verify_payload(frame, quad, kTagId, 22, 9.F, 5, found.rotation),
                    "Known-word verification accepted another ID");
            if (errors > 0)
                require(!verify_payload(frame, quad, high_id, 22, 9.F, errors - 1, found.rotation),
                        "Known-word verification exceeded the correction budget");
        }
    }
    frame.setTo(205);
    put_tag(frame, {2, 10}, 6, 0, 0, 20, 205);
    auto boundary = quad;
    for (auto& point : boundary)
        point.x -= 10;
    require(!verify_payload(frame, boundary, kTagId, 32, 9.F, 0) &&
                    verify_payload<true>(frame, boundary, kTagId, 32, 9.F, 0) == 0,
            "Known-word verification lost the visible exterior sampling rule");
    require(!verify_payload(frame, quad, high_id + 1, 22, 9.F, 5) &&
                    !verify_payload(frame, quad, -1, 22, 9.F, 5) &&
                    !verify_payload(frame, quad, high_id, 22, 9.F, 5, 4) &&
                    !verify_payload(frame, Quad{}, kTagId, 22, 9.F, 0),
            "Known-word verification accepted an invalid ID, rotation or quad");
    frame.setTo(200);
    require(!verify_payload(frame, quad, kTagId, 22, 9.F, 5),
            "Known-word verification accepted a flat frame");
}

}  // namespace

int main() {
    check_cubic_patch();
    check_codebook_lookup();
    check_known_payload();
    Detector detector;
    check_appearance(detector);
    check_edge_fit_bounds();
    check_local_payload_fit();
    check_component_fit_under_uneven_light();
    check_two_border_prints();
    check_two_border_video_history();
    check_validation(detector);
    check_border_detector();
    check_luma_distribution();
    check_automatic_detector();
    check_dense_region();
    check_noisy_print_blocks();
    check_noisy_print_search();
    check_coarse();
    check_bounded_closed_approximation();
    check_fused_mask();
    check_adaptive_tile_mask();
    check_large_tag_rails();
    check_small_tag_refinement();
    check_cropped_eroded_mask();
    check_bounded_adaptive_mask();
    check_local_midrange_mask();
    check_decoded_grid_alignment();
    check_displaced_payload_preserves_measured_border();
    check_component_candidates();
    check_native_component_runs();
    check_dark_cross_components();
    check_component_detector();
    check_component_exposure();
    check_sampled_contours();
    check_alternating_contours();
    check_video_recovery();
    check_uniform_raster();
    check_video_detector();
    check_midtone_edge_refinement();
    check_dense_midrange_mask();
    check_beveled_candidate();
    check_dense_board_search();
    check_planar_tag_shapes();
    check_edge_curvature_consensus();
    check_refined_board_codes();
    check_stationary_patch_retry();
    check_mixed_large_board();
    check_oblique_board_neighbors();
    check_sparse_board_probe();
}
