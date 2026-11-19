// SPDX-License-Identifier: Apache-2.0
#include "aruco3cuda/apriltag/cpu/detector.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace ac = aruco3cuda::apriltag_cpu;
using Clock = std::chrono::steady_clock;
struct Family {
    const char* name_;
    ac::Family cpu_;
    cv::aruco::PredefinedDictionaryType cv_;
};
constexpr std::array<Family, 4> kFamilies{
        {{"tag16h5", ac::Family::k16h5, cv::aruco::DICT_APRILTAG_16h5},
         {"tag25h9", ac::Family::k25h9, cv::aruco::DICT_APRILTAG_25h9},
         {"tag36h10", ac::Family::k36h10, cv::aruco::DICT_APRILTAG_36h10},
         {"tag36h11", ac::Family::k36h11, cv::aruco::DICT_APRILTAG_36h11}}};

struct Options {
    int threads_ = 1;
    int warmup_ = 5;
    int samples_ = 30;
    int decimate_ = 1;
    bool exact_ = false;
    bool check_ = false;
    bool cpu_first_ = false;
    std::string stream_;
    std::string image_;
    std::string family_ = "tag36h11";
};

int integer(const std::string& value, int low, int high) {
    std::size_t consumed = 0;
    const int number = std::stoi(value, &consumed);
    if (consumed != value.size() || number < low || number > high) {
        throw std::invalid_argument("Invalid numeric option");
    }
    return number;
}

Options options(int argc, char** argv) {
    Options result;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--exact") {
            result.exact_ = true;
            continue;
        }
        if (key == "--cpu-first") {
            result.cpu_first_ = true;
            continue;
        }
        if (key == "--check") {
            result.check_ = true;
            continue;
        }
        if (++i == argc)
            throw std::invalid_argument("Missing option value");
        const std::string value = argv[i];
        if (key == "--image")
            result.image_ = value;
        else if (key == "--stream")
            result.stream_ = value;
        else if (key == "--family")
            result.family_ = value;
        else if (key == "--threads")
            result.threads_ = integer(value, 1, 256);
        else if (key == "--decimate")
            result.decimate_ = integer(value, 1, 4);
        else if (key == "--warmup")
            result.warmup_ = integer(value, 0, 10000);
        else if (key == "--samples")
            result.samples_ = integer(value, 1, 10000);
        else
            throw std::invalid_argument("Unknown option: " + key);
    }
    return result;
}

std::vector<ac::Detection> detect(ac::Detector& detector, const cv::Mat& image) {
    const std::size_t bytes = (image.rows - 1) * image.step + image.cols;
    return detector.detect(image.data, bytes, image.cols, image.rows, static_cast<int>(image.step));
}

cv::aruco::ArucoDetector reference(const cv::aruco::Dictionary& dictionary, int decimate,
                                   bool exact = false) {
    cv::aruco::DetectorParameters params;
    params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_APRILTAG;
    params.useAruco3Detection = false;
    params.aprilTagQuadDecimate = decimate;
    if (exact)
        params.errorCorrectionRate = 0.0;
    return cv::aruco::ArucoDetector(dictionary, params);
}

void print_percentiles(std::vector<double>& samples) {
    std::sort(samples.begin(), samples.end());
    for (const double percentile : {0.5, 0.95, 0.99}) {
        const auto index = static_cast<std::size_t>(std::ceil(samples.size() * percentile)) - 1;
        std::cout << ',' << samples[index];
    }
}

void benchmark(const cv::Mat& image, ac::Detector& cpu, cv::aruco::ArucoDetector& opencv,
               const Options& opts, const std::string& name) {
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<ac::Detection> found;
    std::array<std::vector<double>, 2> times;
    for (int i = -opts.warmup_; i < opts.samples_; ++i) {
        for (int j = 0; j < 2; ++j) {
            const int route = (std::abs(i) + j) % 2;
            const auto start = Clock::now();
            if (route == 0)
                opencv.detectMarkers(image, corners, ids);
            else
                found = detect(cpu, image);
            const double ms =
                    std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            if (i >= 0)
                times[route].push_back(ms);
        }
    }
    for (int i = 0; i < 2; ++i) {
        std::cout << (i ? "cpu," : "opencv,") << name << ',' << image.cols << ',' << image.rows
                  << ',' << (i ? found.size() : ids.size());
        print_percentiles(times[i]);
        std::cout << '\n';
    }
    for (const auto& detection : found) {
        std::cout << "cpu_detection," << detection.id_;
        for (const auto& corner : detection.corners_)
            std::cout << ',' << corner.x_ << ',' << corner.y_;
        std::cout << '\n';
    }
    for (std::size_t i = 0; i < ids.size(); ++i) {
        std::cout << "opencv_detection," << ids[i];
        for (const auto& corner : corners[i])
            std::cout << ',' << corner.x << ',' << corner.y;
        std::cout << '\n';
    }
}

// Cycle through independent pre-rendered frames. ID, position, rotation and marker
// count change. Camera I/O, rendering and correctness assertions stay outside timing.
void stream_benchmark(const cv::aruco::Dictionary& dictionary, cv::aruco::ArucoDetector& opencv,
                      ac::Detector& cpu, const Options& opts) {
    if (opts.stream_ != "720p" && opts.stream_ != "1080p") {
        throw std::invalid_argument("--stream expects 720p or 1080p");
    }
    const cv::Size shape = opts.stream_ == "720p" ? cv::Size(1280, 720) : cv::Size(1920, 1080);
    std::vector<cv::Mat> frames;
    constexpr int counts[] = {0, 1, 4, 12};
    cv::Mat marker;
    for (int frame = 0; frame < 16; ++frame) {
        cv::Mat image(shape, CV_8UC1);
        for (int y = 0; y < shape.height; ++y) {
            auto* row = image.ptr<unsigned char>(y);
            for (int x = 0; x < shape.width; ++x) {
                row[x] = static_cast<unsigned char>(170 + ((x / 64 + y / 48 + frame) % 9) * 8);
            }
        }
        std::vector<int> expected;
        for (int i = 0; i < counts[frame % 4]; ++i) {
            const int id = (i + frame) % dictionary.bytesList.rows;
            expected.push_back(id);
            const int side = 50 + 6 * ((i + frame) % 7);
            cv::aruco::generateImageMarker(dictionary, id, side, marker);
            for (int rotation = 0; rotation < (frame + i) % 4; ++rotation) {
                cv::rotate(marker, marker, cv::ROTATE_90_CLOCKWISE);
            }
            const int x = (i % 4) * shape.width / 4 + 20 + frame;
            const int y = (i / 4) * shape.height / 3 + 20 + frame;
            marker.copyTo(image(cv::Rect(x, y, side, side)));
        }
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        opencv.detectMarkers(image, corners, ids);
        std::vector<int> candidate_ids;
        for (const auto& detection : detect(cpu, image))
            candidate_ids.push_back(detection.id_);
        std::sort(candidate_ids.begin(), candidate_ids.end());
        std::sort(expected.begin(), expected.end());
        std::sort(ids.begin(), ids.end());
        if (ids != expected || candidate_ids != expected) {
            throw std::runtime_error("Marker ID mismatch on stream frame " + std::to_string(frame));
        }
        frames.push_back(std::move(image));
    }
    for (int pass = 0; pass < 2; ++pass) {
        const bool candidate = (pass == 1) != opts.cpu_first_;
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        const auto submit = [&](int i) {
            const auto& frame = frames[static_cast<std::size_t>(i) % frames.size()];
            if (candidate)
                return detect(cpu, frame).size();
            opencv.detectMarkers(frame, corners, ids);
            return ids.size();
        };
        for (int i = 0; i < opts.warmup_; ++i)
            submit(i);
        std::vector<double> samples;
        samples.reserve(opts.samples_);
        std::size_t count = 0;
        const auto start = Clock::now();
        for (int i = 0; i < opts.samples_; ++i) {
            const auto call = Clock::now();
            count += submit(i);
            samples.push_back(
                    std::chrono::duration<double, std::milli>(Clock::now() - call).count());
        }
        const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
        std::cout << (candidate ? "cpu" : "opencv") << ",stream," << shape.width << ','
                  << shape.height << ',' << count;
        print_percentiles(samples);
        std::cout << ',' << opts.samples_ / seconds << '\n';
    }
}

void validate(const Family& family, int threads, int decimate) {
    const auto dictionary = cv::aruco::getPredefinedDictionary(family.cv_);
    auto opencv = reference(dictionary, decimate);
    ac::Detector cpu({family.cpu_, threads, decimate});
    // Exercise a strided ROI, including bytes outside the visible rows.
    cv::Mat backing(320, 522, CV_8UC1, cv::Scalar(71));
    cv::Mat image = backing(cv::Rect(3, 3, 512, 300));
    constexpr int kX = 57, kY = 90, kSize = 100;
    for (const int id : {0, dictionary.bytesList.rows - 1}) {
        cv::Mat marker;
        cv::aruco::generateImageMarker(dictionary, id, kSize, marker);
        for (int turns = 0; turns < 4; ++turns) {
            image.setTo(255);
            marker.copyTo(image(cv::Rect(kX, kY, kSize, kSize)));
            const auto found = detect(cpu, image);
            std::vector<int> ids;
            std::vector<std::vector<cv::Point2f>> corners;
            opencv.detectMarkers(image, corners, ids);
            if (found.size() != 1 || ids != std::vector<int>{id} || found.front().id_ != id) {
                throw std::runtime_error("Detection mismatch in " + std::string(family.name_));
            }
            for (int i = 0; i < 4; ++i) {
                const auto& p = found.front().corners_[i];
                if (std::hypot(p.x_ - corners[0][i].x, p.y_ - corners[0][i].y) > 1.5) {
                    std::cerr << "family=" << family.name_ << " id=" << id << " turns=" << turns
                              << " i=" << i << " got=" << p.x_ << ',' << p.y_
                              << " expected=" << corners[0][i] << '\n';
                    throw std::runtime_error("Oriented corner mismatch");
                }
            }
            cv::rotate(marker, marker, cv::ROTATE_90_CLOCKWISE);
        }
    }
    const int border_cells = dictionary.markerSize + 2;
    constexpr int kCellPx = 16;
    const int side = border_cells * kCellPx;
    const int correctable = static_cast<int>(dictionary.maxCorrectionBits * 0.6);
    cv::Mat damaged;
    cv::aruco::generateImageMarker(dictionary, 1, side, damaged);
    for (int errors = 1; errors <= correctable; ++errors) {
        const cv::Rect cell(errors * kCellPx, kCellPx, kCellPx, kCellPx);
        cv::bitwise_not(damaged(cell), damaged(cell));
        image.setTo(255);
        damaged.copyTo(image(cv::Rect(kX, kY, side, side)));
        const auto found = detect(cpu, image);
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        opencv.detectMarkers(image, corners, ids);
        if (ids != std::vector<int>{1} || found.size() != 1 || found.front().id_ != 1) {
            throw std::runtime_error("Correctable payload rejected by " +
                                     std::string(family.name_));
        }
    }
    // The payload alone is insufficient. Erasing the entire printed ring must
    // never produce the same result as a valid marker with a few flipped bits.
    cv::aruco::generateImageMarker(dictionary, 1, side, damaged);
    damaged.rowRange(0, kCellPx).setTo(255);
    damaged.rowRange(side - kCellPx, side).setTo(255);
    damaged.colRange(0, kCellPx).setTo(255);
    damaged.colRange(side - kCellPx, side).setTo(255);
    image.setTo(255);
    damaged.copyTo(image(cv::Rect(kX, kY, side, side)));
    if (!detect(cpu, image).empty())
        throw std::runtime_error("Missing black border accepted");
    image.setTo(171);
    if (!detect(cpu, image).empty())
        throw std::runtime_error("False positive on constant frame");
    for (int side_px = 1; side_px <= 9; ++side_px) {
        cv::Mat tiny = image(cv::Rect(0, 0, side_px, side_px));
        tiny.setTo(160);
        tiny.at<unsigned char>(0, 0) = 100;
        if (!detect(cpu, tiny).empty())
            throw std::runtime_error("False positive on tiny image");
    }
    try {
        cpu.detect(image.data, 1, image.cols, image.rows, static_cast<int>(image.step));
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("Short input accepted");
}

int run(const Options& opts) {
    cv::setNumThreads(opts.threads_);
    if (opts.check_) {
        try {
            ac::Detector invalid({ac::Family::k36h11, 1, 1, 4});
            throw std::runtime_error("Unsupported correction budget accepted");
        } catch (const std::invalid_argument&) {
        }
        for (const auto& family : kFamilies) {
            for (int decimate : {1, 2})
                validate(family, opts.threads_, decimate);
        }
        return 0;
    }
    auto family = std::find_if(kFamilies.begin(), kFamilies.end(),
                               [&](const Family& f) { return f.name_ == opts.family_; });
    if (family == kFamilies.end())
        throw std::invalid_argument("Unknown family");
    const auto dictionary = cv::aruco::getPredefinedDictionary(family->cv_);
    auto opencv = reference(dictionary, opts.decimate_, opts.exact_);
    ac::Detector cpu({family->cpu_, opts.threads_, opts.decimate_, opts.exact_ ? 0 : -1});
    std::cout << std::setprecision(9);
    if (!opts.stream_.empty()) {
        std::cout << "route,scene,width,height,count,p50_ms,p95_ms,p99_ms,frames_per_second\n";
        stream_benchmark(dictionary, opencv, cpu, opts);
        return 0;
    }
    std::cout << "route,scene,width,height,count,p50_ms,p95_ms,p99_ms\n";
    if (!opts.image_.empty()) {
        const auto image = cv::imread(opts.image_, cv::IMREAD_GRAYSCALE);
        if (image.empty())
            throw std::invalid_argument("Image failed to load");
        benchmark(image, cpu, opencv, opts, "input");
        return 0;
    }
    cv::Mat marker;
    cv::aruco::generateImageMarker(dictionary, 7, 112, marker);
    cv::Mat image(720, 1280, CV_8UC1, cv::Scalar(255));
    benchmark(image, cpu, opencv, opts, "blank");
    marker.copyTo(image(cv::Rect(110, 110, marker.cols, marker.rows)));
    benchmark(image, cpu, opencv, opts, "one");
    cv::RNG rng(4242);
    rng.fill(image, cv::RNG::UNIFORM, 130, 240);
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 4; ++col)
            marker.copyTo(image(cv::Rect(65 + col * 300, 45 + row * 215, 112, 112)));
    }
    benchmark(image, cpu, opencv, opts, "textured12");
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        return run(options(argc, argv));
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
