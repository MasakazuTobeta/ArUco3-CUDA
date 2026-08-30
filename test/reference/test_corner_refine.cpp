// SPDX-License-Identifier: Apache-2.0
//
// Cross-checks the subpixel corner refinement against the CPU reference.
//
// Unlike the earlier stages, this one is an iterative solver, so the order of
// the floating-point operations feeds straight into the result. A difference of
// 1 ULP can change the iteration count or whether the termination test fires,
// which turns into a discrete difference. We therefore measure it in two
// layers:
//
// 1. Compare against the result of calling OpenCV's cv::cornerSubPix through
//    the same procedure, and report the number of exactly matching corners and
//    the maximum deviation.
// 2. The pyramid-climbing procedure itself (findCornerInPyrImage) is
//    deterministic, so the choice of scale and of the window radius is pinned
//    down separately.
//
// The oracle namespace below transcribes the arithmetic and the evaluation
// order of OpenCV's cv::cornerSubPix and cv::getRectSubPix. The sources it was
// transcribed from, modules/imgproc/src/cornersubpix.cpp and
// modules/imgproc/src/samplers.cpp, carry a 3-clause BSD header rather than the
// Apache-2.0 that covers OpenCV 4.x as a whole, and name Intel Corporation and
// the OpenCV Foundation as copyright holders. Because 3-clause BSD requires
// that source redistribution retain the copyright notice and the disclaimer,
// the corresponding notice is placed in the NOTICE file at the repository root.
// See PR-005 in docs/code-provenance.md for details.
#include "corner_refine.hpp"

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/device_probe.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"
#include "aruco3cuda/workspace.hpp"
#include "detection_emit.hpp"
#include "device_image.hpp"
#include "preprocess.hpp"

namespace {

using aruco3cuda::DetectorConfig;
using aruco3cuda::MemorySpace;
using aruco3cuda::Status;
using aruco3cuda::Workspace;

bool has_cuda_device() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

/// Builds a scene with a real marker placed on a white background.
///
/// With a uniform black square the window interior is flat, the determinant
/// approaches 0, and the solution diverges. A real marker has a pattern inside
/// it, so this scene exercises branches closer to production use.
cv::Mat make_scene(int width, int height, const cv::Rect& square) {
    const cv::aruco::Dictionary dictionary =
            cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_MIP_36h12);
    cv::Mat marker;
    dictionary.generateImageMarker(17, square.width, marker, 1);
    cv::Mat scene(height, width, CV_8UC1, cv::Scalar(235));
    // A perfectly flat background leaves no gradient outside the marker, so
    // add a gentle pattern.
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            scene.at<std::uint8_t>(y, x) = cv::saturate_cast<std::uint8_t>(220 + ((x + y) % 16));
        }
    }
    marker.copyTo(scene(cv::Rect(square.x, square.y, square.width, square.height)));
    return scene;
}

/// A verbatim host transcription of cv::cornerSubPix and cv::getRectSubPix.
///
/// The transcribed sources are 3-clause BSD; the copyright notice lives in the
/// NOTICE file at the repository root.
///
/// The purpose is to measure "did the implementation transcribe this
/// correctly?" in isolation from machine-to-machine differences. Comparing
/// against cv::cornerSubPix itself mixes in whether the OpenCV build fuses
/// multiply-add (which the compiler and the ISA decide). This oracle is
/// compiled under the same no-fusion assumption as the GPU side
/// (-ffp-contract=off is set in CMakeLists.txt). If the GPU matches this oracle
/// bit for bit, the transcription is faithful.
namespace oracle {

/// The interior path of getRectSubPix_8u32f.
void sample_inside(const cv::Mat& image, int ip_x, int ip_y, float a, float b, int patch_side,
                   std::vector<float>* patch) {
    const float a12 = a * (1.0F - b);
    const float a22 = a * b;
    const float b1 = 1.0F - b;
    const float b2 = b;
    const double s = (1.0 - static_cast<double>(a)) / static_cast<double>(a);
    for (int row = 0; row < patch_side; ++row) {
        const std::uint8_t* source = image.ptr<std::uint8_t>(ip_y + row) + ip_x;
        const std::uint8_t* next = source + image.step;
        float previous = (1.0F - a) * ((b1 * static_cast<float>(source[0])) +
                                       (b2 * static_cast<float>(next[0])));
        for (int j = 0; j < patch_side; ++j) {
            const float t = (a12 * static_cast<float>(source[j + 1])) +
                            (a22 * static_cast<float>(next[j + 1]));
            (*patch)[(static_cast<std::size_t>(row) * static_cast<std::size_t>(patch_side)) +
                     static_cast<std::size_t>(j)] = previous + t;
            previous = static_cast<float>(static_cast<double>(t) * s);
        }
    }
}

/// The border path of getRectSubPix_Cn_, including the result of adjustRect.
void sample_border(const cv::Mat& image, int ip_x, int ip_y, float a, float b, int patch_side,
                   std::vector<float>* patch) {
    const float a11 = (1.0F - a) * (1.0F - b);
    const float a12 = a * (1.0F - b);
    const float a21 = (1.0F - a) * b;
    const float a22 = a * b;
    const float b1 = 1.0F - b;
    const float b2 = b;
    const auto pitch = static_cast<long long>(image.step);
    long long offset = 0;
    int rect_x = 0;
    int rect_width = 0;
    int rect_y = 0;
    int rect_height = 0;
    if (ip_x >= 0) {
        offset += ip_x;
    } else {
        rect_x = std::min(-ip_x, patch_side);
    }
    if (ip_x < image.cols - patch_side) {
        rect_width = patch_side;
    } else {
        rect_width = image.cols - ip_x - 1;
        if (rect_width < 0) {
            offset += rect_width;
            rect_width = 0;
        }
    }
    if (ip_y >= 0) {
        offset += static_cast<long long>(ip_y) * pitch;
    } else {
        rect_y = -ip_y;
    }
    if (ip_y < image.rows - patch_side) {
        rect_height = patch_side;
    } else {
        rect_height = image.rows - ip_y - 1;
        if (rect_height < 0) {
            offset += static_cast<long long>(rect_height) * pitch;
            rect_height = 0;
        }
    }
    offset -= rect_x;

    const std::uint8_t* source = image.data + offset;
    for (int i = 0; i < patch_side; ++i) {
        const std::uint8_t* second = source + pitch;
        if (i < rect_y || i >= rect_height) {
            second -= pitch;
        }
        float* destination = patch->data() +
                             (static_cast<std::size_t>(i) * static_cast<std::size_t>(patch_side));
        float edge = (b1 * static_cast<float>(source[rect_x])) +
                     (b2 * static_cast<float>(second[rect_x]));
        for (int j = 0; j < rect_x; ++j) {
            destination[j] = edge;
        }
        edge = (b1 * static_cast<float>(source[rect_width])) +
               (b2 * static_cast<float>(second[rect_width]));
        for (int j = rect_width; j < patch_side; ++j) {
            destination[j] = edge;
        }
        for (int j = rect_x; j < rect_width; ++j) {
            destination[j] = (static_cast<float>(source[j]) * a11) +
                             (static_cast<float>(source[j + 1]) * a12) +
                             (static_cast<float>(second[j]) * a21) +
                             (static_cast<float>(second[j + 1]) * a22);
        }
        if (i < rect_height) {
            source = second;
        }
    }
}

void sample_patch(const cv::Mat& image, float center_x, float center_y, int patch_side,
                  std::vector<float>* patch) {
    const float shift = static_cast<float>(patch_side - 1) * 0.5F;
    const float cx = center_x - shift;
    const float cy = center_y - shift;
    const int ip_x = static_cast<int>(std::floor(cx));
    const int ip_y = static_cast<int>(std::floor(cy));
    if (ip_x >= 0 && (ip_x + patch_side) < image.cols && ip_y >= 0 &&
        (ip_y + patch_side) < image.rows) {
        float a = cx - static_cast<float>(ip_x);
        const float b = cy - static_cast<float>(ip_y);
        a = std::max(a, 0.0001F);
        sample_inside(image, ip_x, ip_y, a, b, patch_side, patch);
        return;
    }
    sample_border(image, ip_x, ip_y, cx - static_cast<float>(ip_x), cy - static_cast<float>(ip_y),
                  patch_side, patch);
}

/// Transcription of cv::cornerSubPix for a single point.
void corner_sub_pix(const cv::Mat& image, int radius, int max_iterations, double eps,
                    cv::Point2f* point, std::vector<std::int32_t>* counters) {
    const int win_side = (radius * 2) + 1;
    const int patch_side = win_side + 2;
    std::vector<float> mask(static_cast<std::size_t>(win_side) *
                            static_cast<std::size_t>(win_side));
    for (int i = 0; i < win_side; ++i) {
        const float y = static_cast<float>(i - radius) / static_cast<float>(radius);
        const float vy = std::exp(-y * y);
        for (int j = 0; j < win_side; ++j) {
            const float x = static_cast<float>(j - radius) / static_cast<float>(radius);
            mask[(static_cast<std::size_t>(i) * static_cast<std::size_t>(win_side)) +
                 static_cast<std::size_t>(j)] = vy * std::exp(-x * x);
        }
    }
    std::vector<float> patch(static_cast<std::size_t>(patch_side) *
                             static_cast<std::size_t>(patch_side));

    const cv::Point2f start = *point;
    cv::Point2f current = start;
    int iteration = 0;
    double error = 0.0;
    do {
        double a = 0.0;
        double b = 0.0;
        double c = 0.0;
        double bb1 = 0.0;
        double bb2 = 0.0;
        sample_patch(image, current.x, current.y, patch_side, &patch);
        int k = 0;
        for (int i = 0; i < win_side; ++i) {
            const float* row =
                    patch.data() +
                    (static_cast<std::size_t>(i + 1) * static_cast<std::size_t>(patch_side)) + 1U;
            const double py = i - radius;
            for (int j = 0; j < win_side; ++j, ++k) {
                const double m = static_cast<double>(mask[static_cast<std::size_t>(k)]);
                const double tgx =
                        static_cast<double>(row[j + 1]) - static_cast<double>(row[j - 1]);
                const double tgy = static_cast<double>(row[j + patch_side]) -
                                   static_cast<double>(row[j - patch_side]);
                const double gxx = tgx * tgx * m;
                const double gxy = tgx * tgy * m;
                const double gyy = tgy * tgy * m;
                const double px = j - radius;
                a += gxx;
                b += gxy;
                c += gyy;
                bb1 += (gxx * px) + (gxy * py);
                bb2 += (gxy * px) + (gyy * py);
            }
        }
        const double determinant = (a * c) - (b * b);
        if (std::fabs(determinant) <= (DBL_EPSILON * DBL_EPSILON)) {
            ++(*counters)[3];
            break;
        }
        const double scale = 1.0 / determinant;
        cv::Point2f next;
        next.x = static_cast<float>(static_cast<double>(current.x) + (c * scale * bb1) -
                                    (b * scale * bb2));
        next.y = static_cast<float>(static_cast<double>(current.y) - (b * scale * bb1) +
                                    (a * scale * bb2));
        const float dx = next.x - current.x;
        const float dy = next.y - current.y;
        error = static_cast<double>((dx * dx) + (dy * dy));
        if (!(next.x >= 0.0F && next.x < static_cast<float>(image.cols) && next.y >= 0.0F &&
              next.y < static_cast<float>(image.rows))) {
            ++(*counters)[1];
            break;
        }
        current = next;
    } while (++iteration < max_iterations && error > eps);

    (*counters)[4] += iteration;
    if (std::fabs(current.x - start.x) > static_cast<float>(radius) ||
        std::fabs(current.y - start.y) > static_cast<float>(radius)) {
        ++(*counters)[2];
        current = start;
    }
    *point = current;
}

/// Transcription of findCornerInPyrImage.
void refine(const std::vector<cv::Mat>& pyramid, int start_level, float scale_init,
            const DetectorConfig& config, std::vector<cv::Point2f>* corners,
            std::vector<std::int32_t>* counters = nullptr) {
    std::vector<std::int32_t> local(aruco3cuda::detail::kRefineCounterCount, 0);
    std::vector<std::int32_t>* sink = (counters != nullptr) ? counters : &local;
    if (scale_init != 1.0F) {
        for (cv::Point2f& point : *corners) {
            point.x *= scale_init;
            point.y *= scale_init;
        }
    }
    for (int level = start_level - 1; level >= 0; --level) {
        const cv::Mat& image = pyramid[static_cast<std::size_t>(level)];
        const int radius = std::max(image.cols, image.rows) > 1080 ? 5 : 3;
        for (cv::Point2f& point : *corners) {
            point.x *= 2.0F;
            point.y *= 2.0F;
            if (!(point.x >= 0.0F && point.x < static_cast<float>(image.cols) && point.y >= 0.0F &&
                  point.y < static_cast<float>(image.rows))) {
                ++(*sink)[1];
                continue;
            }
            corner_sub_pix(image, radius, config.corner_refinement_max_iterations_,
                           config.corner_refinement_min_accuracy_px_ *
                                   config.corner_refinement_min_accuracy_px_,
                           &point, sink);
        }
    }
}

}  // namespace oracle

/// Walks the same procedure as OpenCV's findCornerInPyrImage on the host.
void refine_reference(const std::vector<cv::Mat>& pyramid, int start_level, float scale_init,
                      const DetectorConfig& config, std::vector<cv::Point2f>* corners) {
    if (scale_init != 1.0F) {
        for (cv::Point2f& point : *corners) {
            point *= scale_init;
        }
    }
    for (int level = start_level - 1; level >= 0; --level) {
        for (cv::Point2f& point : *corners) {
            point *= 2.0F;
        }
        const int win = std::max(pyramid[static_cast<std::size_t>(level)].cols,
                                 pyramid[static_cast<std::size_t>(level)].rows) > 1080
                                ? 5
                                : 3;
        cv::cornerSubPix(pyramid[static_cast<std::size_t>(level)], *corners, cv::Size(win, win),
                         cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
                                          config.corner_refinement_max_iterations_,
                                          config.corner_refinement_min_accuracy_px_));
    }
}

/// Hands the image and the corners to the device and refines them.
class RefineRun {
public:
    RefineRun() = default;
    RefineRun(const RefineRun&) = delete;
    RefineRun& operator=(const RefineRun&) = delete;

    bool run(const cv::Mat& scene, const std::vector<cv::Point2f>& corners,
             const DetectorConfig& base) {
        DetectorConfig config = base;
        const auto detection_count = static_cast<int>(corners.size() / 4U);
        config.max_candidates_ = std::max(detection_count, 1);
        config.max_markers_ = config.max_candidates_;
        config.max_width_px_ = scene.cols;
        config.max_height_px_ = scene.rows;

        aruco3cuda::DeviceProbeResult probe;
        this->multi_processor_count_ = (aruco3cuda::probe_device(0, &probe) == Status::kOk)
                                               ? probe.multi_processor_count_
                                               : 0;
        if (this->image_.reserve(scene.cols, scene.rows) != Status::kOk ||
            this->image_.upload(scene.data, scene.cols, scene.rows, scene.step) != Status::kOk) {
            return false;
        }
        const aruco3cuda::ImageViewU8 view = this->image_.view();

        aruco3cuda::detail::ScalePlan plan;
        if (aruco3cuda::detail::plan_scales(config, scene.cols, scene.rows, &plan) != Status::kOk) {
            return false;
        }
        this->plan_ = plan;

        const std::size_t bytes =
                aruco3cuda::detail::preprocess_workspace_bytes(plan, scene.cols, scene.rows) +
                aruco3cuda::detail::detection_workspace_bytes(config) +
                aruco3cuda::detail::corner_refine_workspace_bytes(config);
        if (bytes == 0U ||
            this->workspace_.ensure_capacity(bytes, MemorySpace::kDevice, nullptr) != Status::kOk) {
            return false;
        }
        this->workspace_.reset();

        aruco3cuda::detail::PreprocessBuffers preprocess;
        aruco3cuda::detail::DetectionEmitBuffers emit;
        aruco3cuda::detail::DeviceDetections detections;
        aruco3cuda::detail::CornerRefineBuffers refine;
        if (aruco3cuda::detail::reserve_preprocess(plan, view, this->workspace_, &preprocess) !=
                    Status::kOk ||
            aruco3cuda::detail::reserve_detections(config, this->workspace_, &emit, &detections) !=
                    Status::kOk ||
            aruco3cuda::detail::reserve_corner_refine(config, this->workspace_, &refine) !=
                    Status::kOk) {
            return false;
        }
        if (aruco3cuda::detail::build_pyramid_async(&preprocess, config, nullptr) != Status::kOk) {
            return false;
        }

        // Inject the corners directly; the upstream stages are bypassed.
        std::vector<float> plane(static_cast<std::size_t>(detection_count));
        for (int corner = 0; corner < 4; ++corner) {
            for (int d = 0; d < detection_count; ++d) {
                plane[static_cast<std::size_t>(d)] = corners[(static_cast<std::size_t>(d) * 4U) +
                                                             static_cast<std::size_t>(corner)]
                                                             .x;
            }
            if (cudaMemcpy(detections.corner_x_ +
                                   (static_cast<std::ptrdiff_t>(corner) * detections.capacity_),
                           plane.data(), plane.size() * sizeof(float),
                           cudaMemcpyHostToDevice) != cudaSuccess) {
                return false;
            }
            for (int d = 0; d < detection_count; ++d) {
                plane[static_cast<std::size_t>(d)] = corners[(static_cast<std::size_t>(d) * 4U) +
                                                             static_cast<std::size_t>(corner)]
                                                             .y;
            }
            if (cudaMemcpy(detections.corner_y_ +
                                   (static_cast<std::ptrdiff_t>(corner) * detections.capacity_),
                           plane.data(), plane.size() * sizeof(float),
                           cudaMemcpyHostToDevice) != cudaSuccess) {
                return false;
            }
        }
        if (cudaMemcpy(detections.count_, &detection_count, sizeof(int), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
            return false;
        }

        aruco3cuda::detail::PyramidRef pyramid;
        if (aruco3cuda::detail::make_pyramid_ref(preprocess, &pyramid) != Status::kOk) {
            return false;
        }
        this->pyramid_widths_.clear();
        for (int level = 0; level < pyramid.level_count_; ++level) {
            this->pyramid_widths_.push_back(pyramid.width_[level]);
        }
        if (aruco3cuda::detail::refine_corners_async(
                    pyramid, plan, config,
                    aruco3cuda::detail::refine_block_count(this->multi_processor_count_), &refine,
                    &detections, nullptr) != Status::kOk ||
            cudaDeviceSynchronize() != cudaSuccess) {
            return false;
        }
        return this->download(detections, refine, detection_count);
    }

    const std::vector<cv::Point2f>& corners() const { return this->corners_; }
    const std::vector<std::int32_t>& counters() const { return this->counters_; }
    const aruco3cuda::detail::ScalePlan& plan() const { return this->plan_; }
    const std::vector<int>& pyramid_widths() const { return this->pyramid_widths_; }

private:
    bool download(const aruco3cuda::detail::DeviceDetections& detections,
                  const aruco3cuda::detail::CornerRefineBuffers& refine, int count) {
        const auto total = static_cast<std::size_t>(count);
        std::vector<float> x(total * 4U);
        std::vector<float> y(total * 4U);
        for (int corner = 0; corner < 4; ++corner) {
            const std::ptrdiff_t offset =
                    static_cast<std::ptrdiff_t>(corner) * detections.capacity_;
            if (cudaMemcpy(x.data() + (static_cast<std::size_t>(corner) * total),
                           detections.corner_x_ + offset, total * sizeof(float),
                           cudaMemcpyDeviceToHost) != cudaSuccess ||
                cudaMemcpy(y.data() + (static_cast<std::size_t>(corner) * total),
                           detections.corner_y_ + offset, total * sizeof(float),
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
                return false;
            }
        }
        this->corners_.clear();
        for (std::size_t d = 0; d < total; ++d) {
            for (int corner = 0; corner < 4; ++corner) {
                const std::size_t index = (static_cast<std::size_t>(corner) * total) + d;
                this->corners_.emplace_back(x[index], y[index]);
            }
        }
        this->counters_.assign(aruco3cuda::detail::kRefineCounterCount, 0);
        return cudaMemcpy(this->counters_.data(), refine.diagnostics_.counters_,
                          this->counters_.size() * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost) == cudaSuccess;
    }

    aruco3cuda::hybrid::DeviceImage image_;
    int multi_processor_count_ = 0;
    Workspace workspace_;
    std::vector<cv::Point2f> corners_;
    std::vector<std::int32_t> counters_;
    std::vector<int> pyramid_widths_;
    aruco3cuda::detail::ScalePlan plan_;
};

/// Result of a comparison.
struct Comparison {
    std::size_t exact_ = 0;
    std::size_t total_ = 0;
    double max_distance_ = 0.0;
    double sum_square_ = 0.0;

    double rmse() const {
        return (this->total_ == 0U)
                       ? 0.0
                       : std::sqrt(this->sum_square_ / static_cast<double>(this->total_));
    }
};

Comparison compare(const std::vector<cv::Point2f>& expected,
                   const std::vector<cv::Point2f>& actual) {
    Comparison result;
    result.total_ = std::min(expected.size(), actual.size());
    for (std::size_t i = 0; i < result.total_; ++i) {
        const double dx = static_cast<double>(actual[i].x) - static_cast<double>(expected[i].x);
        const double dy = static_cast<double>(actual[i].y) - static_cast<double>(expected[i].y);
        if (actual[i].x == expected[i].x && actual[i].y == expected[i].y) {
            ++result.exact_;
        }
        const double distance = std::sqrt((dx * dx) + (dy * dy));
        result.max_distance_ = std::max(result.max_distance_, distance);
        result.sum_square_ += (dx * dx) + (dy * dy);
    }
    return result;
}

/// Builds the scene and the initial corners. The corners are the square's
/// corners nudged slightly.
void make_case(int width, int height, const cv::Rect& square,
               const aruco3cuda::detail::ScalePlan& plan, std::vector<cv::Point2f>* corners) {
    const auto scale = static_cast<float>(plan.fxfy_);
    const float left = static_cast<float>(square.x) * scale;
    const float top = static_cast<float>(square.y) * scale;
    const float right = static_cast<float>(square.x + square.width) * scale;
    const float bottom = static_cast<float>(square.y + square.height) * scale;
    // Corners in segmentation coordinates. Round to integers so they have the
    // same form as the output of candidate extraction.
    corners->clear();
    corners->emplace_back(std::round(left), std::round(top));
    corners->emplace_back(std::round(right), std::round(top));
    corners->emplace_back(std::round(right), std::round(bottom));
    corners->emplace_back(std::round(left), std::round(bottom));
    (void)width;
    (void)height;
}

// Happy path: the pyramid-climbing refinement matches OpenCV's cornerSubPix.
TEST(CornerRefineTest, matches_opencv_corner_subpix) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    const DetectorConfig config;
    struct Case {
        int width_;
        int height_;
        cv::Rect square_;
    };
    const std::vector<Case> cases = {
            {1280, 720, cv::Rect(400, 260, 160, 160)},
            {1280, 720, cv::Rect(200, 150, 320, 320)},
            {640, 480, cv::Rect(180, 140, 200, 200)},
            {1920, 1080, cv::Rect(700, 400, 400, 400)},
    };

    Comparison total;
    for (const Case& item : cases) {
        const cv::Mat scene = make_scene(item.width_, item.height_, item.square_);
        aruco3cuda::detail::ScalePlan plan;
        ASSERT_EQ(aruco3cuda::detail::plan_scales(config, item.width_, item.height_, &plan),
                  Status::kOk);
        std::vector<cv::Point2f> corners;
        make_case(item.width_, item.height_, item.square_, plan, &corners);

        RefineRun run;
        ASSERT_TRUE(run.run(scene, corners, config)) << item.width_ << "x" << item.height_;

        // Build the same pyramid with OpenCV and walk the same procedure.
        std::vector<cv::Mat> pyramid;
        cv::buildPyramid(scene, pyramid, plan.level_count_ - 1);
        ASSERT_EQ(static_cast<int>(pyramid.size()), plan.level_count_);
        std::vector<cv::Point2f> expected = corners;
        const auto scale_init =
                static_cast<float>(
                        pyramid[static_cast<std::size_t>(plan.closest_level_index_)].cols) /
                static_cast<float>(plan.segmentation_width_px_);
        refine_reference(pyramid, plan.closest_level_index_, scale_init, config, &expected);

        const Comparison result = compare(expected, run.corners());
        std::printf("[refine] %dx%d: exact %zu/%zu, max deviation %.6f px, RMSE %.6f px\n",
                    item.width_, item.height_, result.exact_, result.total_, result.max_distance_,
                    result.rmse());
        total.exact_ += result.exact_;
        total.total_ += result.total_;
        total.max_distance_ = std::max(total.max_distance_, result.max_distance_);
        total.sum_square_ += result.sum_square_;
    }
    std::printf("[refine] total: exact %zu/%zu, max deviation %.6f px, RMSE %.6f px\n",
                total.exact_, total.total_, total.max_distance_, total.rmse());
    EXPECT_LT(total.rmse(), 0.01);
    EXPECT_LT(total.max_distance_, 0.5);
}

// Happy path: still matches OpenCV when the starting positions are perturbed.
//
// Changing the starting position changes the iteration count and whether the
// branch that falls back to the starting position on poor convergence is taken.
// To actually exercise the branches of the iterative solver, we feed in
// randomly scattered starting positions.
TEST(CornerRefineTest, matches_opencv_for_perturbed_starts) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    const DetectorConfig config;
    const cv::Rect square(400, 260, 160, 160);
    const cv::Mat scene = make_scene(1280, 720, square);
    aruco3cuda::detail::ScalePlan plan;
    ASSERT_EQ(aruco3cuda::detail::plan_scales(config, 1280, 720, &plan), Status::kOk);

    std::vector<cv::Mat> pyramid;
    cv::buildPyramid(scene, pyramid, plan.level_count_ - 1);
    const auto scale_init =
            static_cast<float>(pyramid[static_cast<std::size_t>(plan.closest_level_index_)].cols) /
            static_cast<float>(plan.segmentation_width_px_);

    std::vector<cv::Point2f> base;
    make_case(1280, 720, square, plan, &base);

    std::mt19937_64 rng(20260828U);
    std::uniform_int_distribution<int> shift(-1, 1);
    std::vector<cv::Point2f> corners;
    // Feed the corners of 24 markers through in a single call.
    for (int trial = 0; trial < 24; ++trial) {
        for (const cv::Point2f& point : base) {
            corners.emplace_back(point.x + static_cast<float>(shift(rng)),
                                 point.y + static_cast<float>(shift(rng)));
        }
    }

    RefineRun run;
    ASSERT_TRUE(run.run(scene, corners, config));

    std::vector<cv::Point2f> expected = corners;
    refine_reference(pyramid, plan.closest_level_index_, scale_init, config, &expected);

    const Comparison result = compare(expected, run.corners());
    std::printf(
            "[refine] %zu corners with perturbed starts: exact %zu, max deviation %.6f px, "
            "RMSE %.6f px\n",
            result.total_, result.exact_, result.max_distance_, result.rmse());
    std::printf(
            "[refine] counters: refined %d, out of image %d, poor convergence %d, "
            "singular %d, iterations %d\n",
            run.counters()[0], run.counters()[1], run.counters()[2], run.counters()[3],
            run.counters()[4]);
    EXPECT_LT(result.rmse(), 0.01);
    EXPECT_LT(result.max_distance_, 0.5);
    // Input close to production use must not enter the degenerate branches. If
    // it does, the scene or the starting positions are bad and this test is no
    // longer measuring what it intends to. The degenerate branches themselves
    // are pinned down by matches_transcribed_oracle_bit_exactly.
    EXPECT_EQ(run.counters()[2], 0);
    EXPECT_EQ(run.counters()[3], 0);
    EXPECT_EQ(run.counters()[1], 0);
}

// Happy path: matches the verbatim oracle bit for bit.
//
// Comparing against cv::cornerSubPix itself mixes in whether the OpenCV build
// fuses multiply-add (which the compiler and the ISA decide). Here we compare
// against the transcribed oracle with fusion disabled, and measure **only that
// the transcription is faithful**. Degenerate input, where the determinant
// approaches 0, is mixed in as well. There the deviation from cv::cornerSubPix
// opens up discretely, but agreement with the oracle must still hold.
TEST(CornerRefineTest, matches_transcribed_oracle_bit_exactly) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    const DetectorConfig config;
    const cv::Rect square(400, 260, 160, 160);
    const cv::Mat scene = make_scene(1280, 720, square);
    aruco3cuda::detail::ScalePlan plan;
    ASSERT_EQ(aruco3cuda::detail::plan_scales(config, 1280, 720, &plan), Status::kOk);

    std::vector<cv::Mat> pyramid;
    cv::buildPyramid(scene, pyramid, plan.level_count_ - 1);
    const auto scale_init =
            static_cast<float>(pyramid[static_cast<std::size_t>(plan.closest_level_index_)].cols) /
            static_cast<float>(plan.segmentation_width_px_);

    std::vector<cv::Point2f> base;
    make_case(1280, 720, square, plan, &base);

    // Perturb heavily to produce degenerate windows: once the window leaves
    // the marker pattern the determinant approaches 0.
    std::mt19937_64 rng(20260828U);
    std::uniform_int_distribution<int> shift(-6, 6);
    std::vector<cv::Point2f> corners;
    for (int trial = 0; trial < 32; ++trial) {
        for (const cv::Point2f& point : base) {
            corners.emplace_back(point.x + static_cast<float>(shift(rng)),
                                 point.y + static_cast<float>(shift(rng)));
        }
    }

    RefineRun run;
    ASSERT_TRUE(run.run(scene, corners, config));

    std::vector<cv::Point2f> expected = corners;
    std::vector<std::int32_t> oracle_counters(aruco3cuda::detail::kRefineCounterCount, 0);
    oracle::refine(pyramid, plan.closest_level_index_, scale_init, config, &expected,
                   &oracle_counters);
    const Comparison against_oracle = compare(expected, run.corners());

    std::vector<cv::Point2f> opencv_result = corners;
    refine_reference(pyramid, plan.closest_level_index_, scale_init, config, &opencv_result);
    const Comparison against_opencv = compare(opencv_result, run.corners());

    std::printf(
            "[refine] %zu corners including degenerate ones: exact vs oracle %zu "
            "(max deviation %.6f px), exact vs cv::cornerSubPix %zu "
            "(max deviation %.6f px, RMSE %.6f px)\n",
            against_oracle.total_, against_oracle.exact_, against_oracle.max_distance_,
            against_opencv.exact_, against_opencv.max_distance_, against_opencv.rmse());
    std::printf(
            "[refine] counters: refined %d, out of image %d, poor convergence %d, "
            "singular %d, iterations %d\n",
            run.counters()[0], run.counters()[1], run.counters()[2], run.counters()[3],
            run.counters()[4]);

    // Faithfulness of the transcription is required as a bit-exact match even
    // on degenerate input.
    EXPECT_EQ(against_oracle.exact_, against_oracle.total_);
    EXPECT_EQ(against_oracle.max_distance_, 0.0);
    // Confirm that the degenerate branches were actually taken.
    EXPECT_GT(run.counters()[2] + run.counters()[3], 0);

    // **Cross-check the counters against the oracle's counters.**
    //
    // Corners can agree even when the control flow differs. In particular, a
    // corner that fell back to its starting position on poor convergence agrees
    // with the oracle trivially, because the fallback target *is* the starting
    // position. In this test more than half of the 128 corners take that
    // fallback path, so corner agreement alone cannot detect that a branch
    // shifted under parallelization.
    //
    // The absolute counter values must not be pinned down. We have measured
    // that on degenerate input the control flow differs across machines (DGX
    // Spark reports 70/81/83/837 while RTX 5070 Ti reports 78/66/87/711). The
    // pyramid, the mask, the scene, the compiler, and glibc were all verified
    // to be identical on the three machines; the cause is still unidentified.
    // The machine-independent invariant is that the GPU and the oracle take the
    // same control flow.
    EXPECT_EQ(run.counters()[0], 128) << "number of refined corners";
    for (int i = 1; i < aruco3cuda::detail::kRefineCounterCount; ++i) {
        EXPECT_EQ(run.counters()[static_cast<std::size_t>(i)],
                  oracle_counters[static_cast<std::size_t>(i)])
                << "counter " << i;
    }
}

// Boundary: still matches the oracle bit for bit on the path where the window
// extends past the image.
//
// getRectSubPix has an interior path and a border path; the latter fills
// everything outside the rectangle chosen by adjustRect with the edge value.
// While the corners sit near the middle of the image the border path is
// **never taken once**. We place the marker at the image edges to take it
// explicitly.
TEST(CornerRefineTest, matches_oracle_on_image_border) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    const DetectorConfig config;
    // Place the marker against the four corners and the four edges. At full
    // resolution the window radius is 5 and the patch is 13, so a corner within
    // 6 px of an edge takes the border path.
    const int side = 160;
    struct Placement {
        int x_;
        int y_;
    };
    const std::vector<Placement> placements = {
            {0, 0},   {1280 - side, 0},  {0, 720 - side}, {1280 - side, 720 - side},
            {560, 0}, {560, 720 - side}, {0, 280},        {1280 - side, 280},
    };

    std::size_t total_oracle_mismatch = 0;
    std::size_t border_cases = 0;
    for (const Placement& place : placements) {
        const cv::Mat scene = make_scene(1280, 720, cv::Rect(place.x_, place.y_, side, side));
        aruco3cuda::detail::ScalePlan plan;
        ASSERT_EQ(aruco3cuda::detail::plan_scales(config, 1280, 720, &plan), Status::kOk);
        std::vector<cv::Point2f> corners;
        make_case(1280, 720, cv::Rect(place.x_, place.y_, side, side), plan, &corners);

        std::vector<cv::Mat> pyramid;
        cv::buildPyramid(scene, pyramid, plan.level_count_ - 1);
        const auto scale_init =
                static_cast<float>(
                        pyramid[static_cast<std::size_t>(plan.closest_level_index_)].cols) /
                static_cast<float>(plan.segmentation_width_px_);

        // Count how many corners sit near an edge, as evidence that the
        // border path was taken.
        for (const cv::Point2f& point : corners) {
            const float x = point.x * scale_init * 4.0F;
            const float y = point.y * scale_init * 4.0F;
            if (x < 8.0F || y < 8.0F || x > 1272.0F || y > 712.0F) {
                ++border_cases;
            }
        }

        RefineRun run;
        ASSERT_TRUE(run.run(scene, corners, config)) << place.x_ << "," << place.y_;
        std::vector<cv::Point2f> expected = corners;
        oracle::refine(pyramid, plan.closest_level_index_, scale_init, config, &expected);
        const Comparison result = compare(expected, run.corners());
        total_oracle_mismatch += (result.total_ - result.exact_);
    }
    std::printf("[refine] 8 edge placements: mismatches vs oracle %zu, corners near an edge %zu\n",
                total_oracle_mismatch, border_cases);
    EXPECT_EQ(total_oracle_mismatch, 0U);
    // Confirm that the border path was actually taken. If it was not, this test
    // only exercises the interior path and adds nothing over the existing
    // tests.
    EXPECT_GT(border_cases, 0U);
}

// Happy path: the refined corners come back in full-resolution coordinates.
TEST(CornerRefineTest, output_is_in_original_resolution) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    const DetectorConfig config;
    const cv::Rect square(400, 260, 160, 160);
    const cv::Mat scene = make_scene(1280, 720, square);
    aruco3cuda::detail::ScalePlan plan;
    ASSERT_EQ(aruco3cuda::detail::plan_scales(config, 1280, 720, &plan), Status::kOk);
    std::vector<cv::Point2f> corners;
    make_case(1280, 720, square, plan, &corners);

    RefineRun run;
    ASSERT_TRUE(run.run(scene, corners, config));
    ASSERT_EQ(run.corners().size(), 4U);
    // Input is in segmentation coordinates (427 wide); output is at full
    // resolution (1280 wide).
    EXPECT_LT(corners[0].x, 200.0F);
    EXPECT_NEAR(static_cast<double>(run.corners()[0].x), square.x, 2.0);
    EXPECT_NEAR(static_cast<double>(run.corners()[0].y), square.y, 2.0);
    EXPECT_NEAR(static_cast<double>(run.corners()[2].x), square.x + square.width, 2.0);
    EXPECT_NEAR(static_cast<double>(run.corners()[2].y), square.y + square.height, 2.0);
    std::printf("[refine] refined corners (%.4f, %.4f) (%.4f, %.4f)\n",
                static_cast<double>(run.corners()[0].x), static_cast<double>(run.corners()[0].y),
                static_cast<double>(run.corners()[2].x), static_cast<double>(run.corners()[2].y));
}

// Happy path: the progress counters are recorded.
TEST(CornerRefineTest, records_diagnostics) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    const DetectorConfig config;
    const cv::Rect square(400, 260, 160, 160);
    const cv::Mat scene = make_scene(1280, 720, square);
    aruco3cuda::detail::ScalePlan plan;
    ASSERT_EQ(aruco3cuda::detail::plan_scales(config, 1280, 720, &plan), Status::kOk);
    std::vector<cv::Point2f> corners;
    make_case(1280, 720, square, plan, &corners);

    RefineRun run;
    ASSERT_TRUE(run.run(scene, corners, config));
    ASSERT_EQ(run.counters().size(), 5U);
    EXPECT_EQ(run.counters()[0], 4);
    std::printf(
            "[refine] counters: refined %d, out of image %d, poor convergence %d, "
            "singular %d, iterations %d\n",
            run.counters()[0], run.counters()[1], run.counters()[2], run.counters()[3],
            run.counters()[4]);
    // There are two pyramid levels, so the iteration count is at least twice
    // the number of corners.
    EXPECT_GE(run.counters()[4], 8);
}

// Happy path: running the same input twice produces the same result.
TEST(CornerRefineTest, results_are_deterministic) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "skipping: no CUDA device available in this environment";
    }
    const DetectorConfig config;
    const cv::Rect square(300, 200, 240, 240);
    const cv::Mat scene = make_scene(1280, 720, square);
    aruco3cuda::detail::ScalePlan plan;
    ASSERT_EQ(aruco3cuda::detail::plan_scales(config, 1280, 720, &plan), Status::kOk);
    std::vector<cv::Point2f> corners;
    make_case(1280, 720, square, plan, &corners);

    RefineRun first;
    RefineRun second;
    ASSERT_TRUE(first.run(scene, corners, config));
    ASSERT_TRUE(second.run(scene, corners, config));
    ASSERT_EQ(first.corners().size(), second.corners().size());
    for (std::size_t i = 0; i < first.corners().size(); ++i) {
        EXPECT_FLOAT_EQ(first.corners()[i].x, second.corners()[i].x) << i;
        EXPECT_FLOAT_EQ(first.corners()[i].y, second.corners()[i].y) << i;
    }
}

// Failure path: invalid arguments perform no work.
TEST(CornerRefineTest, rejects_invalid_arguments) {
    Workspace workspace;
    const DetectorConfig config;
    aruco3cuda::detail::CornerRefineBuffers buffers;
    EXPECT_EQ(aruco3cuda::detail::reserve_corner_refine(config, workspace, nullptr),
              Status::kInvalidArgument);
    EXPECT_NE(aruco3cuda::detail::reserve_corner_refine(config, workspace, &buffers), Status::kOk);

    aruco3cuda::detail::PyramidRef pyramid;
    aruco3cuda::detail::ScalePlan plan;
    aruco3cuda::detail::DeviceDetections detections;
    EXPECT_EQ(aruco3cuda::detail::refine_corners_async(pyramid, plan, config, 64, nullptr,
                                                       &detections, nullptr),
              Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::detail::refine_corners_async(pyramid, plan, config, 64, &buffers, nullptr,
                                                       nullptr),
              Status::kInvalidArgument);

    EXPECT_GT(aruco3cuda::detail::corner_refine_workspace_bytes(config), 0U);

    aruco3cuda::detail::PyramidRef empty;
    EXPECT_EQ(aruco3cuda::detail::make_pyramid_ref(aruco3cuda::detail::PreprocessBuffers(), &empty),
              Status::kOk);
}

}  // namespace
