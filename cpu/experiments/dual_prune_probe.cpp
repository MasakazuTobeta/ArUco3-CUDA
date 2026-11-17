// SPDX-License-Identifier: Apache-2.0
// Archived rejection probe: dual component pruning preserves tested contours but is slower.
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "hybrid/cpu_candidates.hpp"

namespace {

using Contours = std::vector<std::vector<cv::Point>>;

// Independent experiment using OpenCV's public CCL API. Returned matrices borrow
// either the input or this instance's scratch storage; all work is synchronous.
class DualPruner {
public:
    cv::Mat apply(const cv::Mat& input, int minimum, bool foreground = true,
                  bool background = true) {
        const cv::Mat* image = &input;
        if (foreground) {
            const int count = cv::connectedComponentsWithStats(*image, this->foreground_labels_,
                                                               this->statistics_, this->centroids_,
                                                               8, CV_32S, cv::CCL_GRANA);
            this->values_.assign(static_cast<std::size_t>(count), 255U);
            this->values_[0] = 0U;
            bool changed = false;
            for (int label = 1; label < count; ++label) {
                const int area = this->statistics_.at<int>(label, cv::CC_STAT_AREA);
                if (static_cast<std::int64_t>(area) * 8 < minimum) {
                    this->values_[static_cast<std::size_t>(label)] = 0U;
                    changed = true;
                }
            }
            if (changed) {
                this->filtered_.create(input.size(), CV_8UC1);
                for (int y = 0; y < input.rows; ++y) {
                    const int* labels = this->foreground_labels_.ptr<int>(y);
                    std::uint8_t* destination = this->filtered_.ptr<std::uint8_t>(y);
                    for (int x = 0; x < input.cols; ++x) {
                        destination[x] = this->values_[static_cast<std::size_t>(labels[x])];
                    }
                }
                image = &this->filtered_;
            }
        }
        if (background) {
            cv::bitwise_not(*image, this->inverse_);
            const int count = cv::connectedComponentsWithStats(
                    this->inverse_, this->background_labels_, this->statistics_, this->centroids_,
                    4, CV_32S, cv::CCL_GRANA);
            this->values_.assign(static_cast<std::size_t>(count), 0U);
            bool changed = false;
            for (int label = 1; label < count; ++label) {
                const int* statistics = this->statistics_.ptr<int>(label);
                const bool touches_border =
                        statistics[cv::CC_STAT_LEFT] == 0 || statistics[cv::CC_STAT_TOP] == 0 ||
                        statistics[cv::CC_STAT_LEFT] + statistics[cv::CC_STAT_WIDTH] ==
                                input.cols ||
                        statistics[cv::CC_STAT_TOP] + statistics[cv::CC_STAT_HEIGHT] == input.rows;
                if (!touches_border &&
                    static_cast<std::int64_t>(statistics[cv::CC_STAT_AREA]) * 8 < minimum) {
                    this->values_[static_cast<std::size_t>(label)] = 255U;
                    changed = true;
                }
            }
            if (changed) {
                if (image != &this->filtered_) {
                    input.copyTo(this->filtered_);
                }
                for (int y = 0; y < input.rows; ++y) {
                    const int* labels = this->background_labels_.ptr<int>(y);
                    std::uint8_t* destination = this->filtered_.ptr<std::uint8_t>(y);
                    for (int x = 0; x < input.cols; ++x) {
                        if (this->values_[static_cast<std::size_t>(labels[x])] != 0U) {
                            destination[x] = 255U;
                        }
                    }
                }
                image = &this->filtered_;
            }
        }
        return *image;
    }

private:
    cv::Mat foreground_labels_;
    cv::Mat background_labels_;
    cv::Mat statistics_;
    cv::Mat centroids_;
    cv::Mat filtered_;
    cv::Mat inverse_;
    std::vector<std::uint8_t> values_;
};

Contours eligible_contours(const cv::Mat& binary, int minimum) {
    Contours all;
    Contours valid;
    cv::findContours(binary, all, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    for (std::vector<cv::Point>& contour : all) {
        if (contour.size() >= static_cast<std::size_t>(minimum)) {
            valid.push_back(std::move(contour));
        }
    }
    return valid;
}

bool check(const cv::Mat& input, int minimum, std::size_t iteration) {
    DualPruner pruner;
    const Contours expected = eligible_contours(input, minimum);
    const cv::Mat filtered = pruner.apply(input, minimum);
    const Contours actual = eligible_contours(filtered, minimum);
    if (expected != actual) {
        std::cerr << "contour mismatch iteration " << iteration << " dimensions " << input.size()
                  << " minimum " << minimum << " expected " << expected.size() << " actual "
                  << actual.size() << '\n';
        cv::imwrite("dual-prune-failure.png", input);
        cv::imwrite("dual-prune-failure-filtered.png", filtered);
        return false;
    }
    aruco3cuda::DetectorConfig config;
    config.use_aruco3_detection_ = false;
    config.min_marker_perimeter_rate_ =
            static_cast<double>(minimum) / std::max(input.cols, input.rows);
    config.max_marker_perimeter_rate_ = 64.0;
    std::vector<std::vector<cv::Point2f>> expected_quads;
    std::vector<std::vector<cv::Point2f>> actual_quads;
    Contours expected_paths;
    Contours actual_paths;
    aruco3cuda::hybrid::find_quad_candidates(input, config, expected_quads, expected_paths);
    aruco3cuda::hybrid::find_quad_candidates(filtered, config, actual_quads, actual_paths);
    if (expected_quads != actual_quads || expected_paths != actual_paths) {
        std::cerr << "quad mismatch iteration " << iteration << '\n';
        return false;
    }
    return true;
}

int check_random_and_shapes() {
    cv::RNG random(81753);
    std::size_t checked = 0U;
    for (int dimensions = 1; dimensions <= 12; ++dimensions) {
        cv::Mat input(dimensions, dimensions, CV_8UC1);
        for (const int density : {1, 5, 15, 35, 50, 65, 85, 95, 99}) {
            for (int sample = 0; sample < 60; ++sample) {
                random.fill(input, cv::RNG::UNIFORM, 0, 100);
                cv::compare(input, density, input, cv::CMP_LT);
                for (const int minimum : {1, 8, 9, 16, 24, 32, 64, 128}) {
                    if (!check(input, minimum, checked)) {
                        return 1;
                    }
                    ++checked;
                }
            }
        }
    }
    for (int sample = 0; sample < 3000; ++sample) {
        cv::Mat input(64, 64, CV_8UC1, cv::Scalar(sample % 2 != 0 ? 255 : 0));
        for (int shape = 0; shape < 12; ++shape) {
            const cv::Point first(random.uniform(0, 64), random.uniform(0, 64));
            const cv::Point second(random.uniform(0, 64), random.uniform(0, 64));
            const cv::Scalar color(shape % 2 != 0 ? 255 : 0);
            if (shape % 3 == 0) {
                cv::line(input, first, second, color, random.uniform(1, 4));
            } else if (shape % 3 == 1) {
                cv::rectangle(input,
                              cv::Rect(std::min(first.x, second.x), std::min(first.y, second.y),
                                       std::abs(first.x - second.x) + 1,
                                       std::abs(first.y - second.y) + 1),
                              color, random.uniform(1, 4));
            } else {
                cv::circle(input, first, random.uniform(1, 22), color, random.uniform(1, 4));
            }
        }
        for (const int minimum : {9, 16, 32, 64, 128, 256}) {
            if (!check(input, minimum, checked)) {
                return 1;
            }
            ++checked;
        }
    }
    for (int radius = 1; radius < 14; ++radius) {
        cv::Mat input(64, 64, CV_8UC1, cv::Scalar(0));
        for (int inner_radius = radius; inner_radius > 0; --inner_radius) {
            cv::rectangle(input,
                          cv::Rect(32 - inner_radius, 32 - inner_radius, 2 * inner_radius + 1,
                                   2 * inner_radius + 1),
                          cv::Scalar(inner_radius % 2 != 0 ? 255 : 0), 1);
        }
        for (const int minimum : {9, 16, 32, 64, 128, 256}) {
            if (!check(input, minimum, checked)) {
                return 1;
            }
            ++checked;
        }
    }
    std::cout << "exact contour and quad comparisons " << checked << '\n';
    return 0;
}

// QA contributed exhaustive input enumeration after the initial random probe.
int check_exhaustive() {
    std::size_t checked = 0U;
    for (const cv::Size size : {cv::Size(3, 3), cv::Size(4, 4), cv::Size(5, 3), cv::Size(3, 5)}) {
        cv::Mat input(size, CV_8UC1);
        const unsigned int pixels = static_cast<unsigned int>(size.area());
        const unsigned int masks = 1U << pixels;
        for (unsigned int bits = 0U; bits < masks; ++bits) {
            for (unsigned int pixel = 0U; pixel < pixels; ++pixel) {
                const int y = static_cast<int>(pixel) / size.width;
                const int x = static_cast<int>(pixel) % size.width;
                input.ptr<std::uint8_t>(y)[x] = (bits & (1U << pixel)) != 0U ? 255U : 0U;
            }
            for (const int minimum : {9, 17, 25, 33}) {
                if (!check(input, minimum, checked)) {
                    std::cerr << "exhaustive mask " << bits << " dimensions " << size << '\n';
                    return 1;
                }
                ++checked;
            }
        }
        std::cout << "exhaustive passed " << size << " masks " << masks << " total_checks "
                  << checked << '\n';
    }
    return 0;
}

int benchmark(const std::string& input_dir) {
    std::size_t checked = 0U;
    for (const std::string filename :
         {"cpu_1280x720_n4_clean.png", "cpu_1280x720_n4_noise.png", "cpu_3840x2160_n4_clean.png",
          "cpu_3840x2160_n4_noise.png"}) {
        const cv::Mat gray = cv::imread(input_dir + "/" + filename, cv::IMREAD_GRAYSCALE);
        if (gray.empty()) {
            std::cerr << "cannot load " << input_dir << '/' << filename << '\n';
            return 2;
        }
        std::vector<cv::Mat> binaries(3);
        std::size_t index = 0U;
        for (const int window : {3, 13, 23}) {
            cv::adaptiveThreshold(gray, binaries[index++], 255, cv::ADAPTIVE_THRESH_MEAN_C,
                                  cv::THRESH_BINARY_INV, window, 7.0);
        }
        aruco3cuda::DetectorConfig config;
        config.use_aruco3_detection_ = false;
        const int minimum = static_cast<int>(config.min_marker_perimeter_rate_ *
                                             std::max(gray.rows, gray.cols));
        for (const cv::Mat& binary : binaries) {
            if (!check(binary, minimum, checked++)) {
                return 1;
            }
        }
        for (int variant = 0; variant < 3; ++variant) {
            std::vector<double> samples_ms;
            DualPruner pruner;
            std::size_t points = 0U;
            std::size_t quads = 0U;
            for (int iteration = -3; iteration < 11; ++iteration) {
                const auto start = std::chrono::steady_clock::now();
                for (const cv::Mat& binary : binaries) {
                    const cv::Mat filtered =
                            variant == 0 ? binary
                                         : pruner.apply(binary, minimum, variant == 2, true);
                    std::vector<std::vector<cv::Point2f>> candidates;
                    Contours contours;
                    aruco3cuda::hybrid::find_quad_candidates(filtered, config, candidates,
                                                             contours);
                    if (iteration == 0) {
                        quads += candidates.size();
                        for (const std::vector<cv::Point>& contour : contours) {
                            points += contour.size();
                        }
                    }
                }
                const auto stop = std::chrono::steady_clock::now();
                if (iteration >= 0) {
                    samples_ms.push_back(
                            std::chrono::duration<double, std::milli>(stop - start).count());
                }
            }
            std::sort(samples_ms.begin(), samples_ms.end());
            std::cout << filename << " variant " << variant << " p50 " << samples_ms[5]
                      << " p95/p99 " << samples_ms.back() << " quads " << quads
                      << " eligible_quad_points " << points << '\n';
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    cv::setNumThreads(1);
    try {
        if (argc > 1 && std::string(argv[1]) == "benchmark") {
            return benchmark(argc > 2 ? argv[2] : "data/cpu-experiment");
        }
        if (argc > 1 && std::string(argv[1]) == "exhaustive") {
            return check_exhaustive();
        }
        return check_random_and_shapes();
    } catch (const std::exception& error) {
        std::cerr << "dual prune probe: " << error.what() << '\n';
        return 2;
    }
}
