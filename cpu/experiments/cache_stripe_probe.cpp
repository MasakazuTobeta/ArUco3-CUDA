// SPDX-License-Identifier: Apache-2.0
// Paired microbenchmark of the selected prefix budgets; detector evaluation is separate.
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "cpu_threshold.hpp"

namespace {

const std::vector<std::size_t> kBudgets = {0U, 2U * 1024U * 1024U, 4U * 1024U * 1024U};
const std::vector<int> kWindows = {3, 13, 23};

// Expected matrix payloads for these three windows, not RSS or allocator capacity.
std::size_t planned_scratch_bytes(const cv::Size& size, std::size_t budget) {
    constexpr std::int64_t kRadius = 11;
    const std::int64_t padded_width = size.width + 2 * kRadius;
    const std::uint64_t row_bytes = static_cast<std::uint64_t>(padded_width + 1) * 4U;
    std::int64_t max_rows =
            std::numeric_limits<std::int32_t>::max() / (255 * padded_width) - 2 * kRadius;
    if (budget > 0U) {
        max_rows = std::min(max_rows,
                            static_cast<std::int64_t>(budget / row_bytes) - (2 * kRadius + 1));
    }
    if (max_rows <= 0) {
        return 0U;
    }
    const std::int64_t count = (size.height + max_rows - 1) / max_rows;
    const std::int64_t rows = (size.height + count - 1) / count;
    const std::int64_t padding_rows = budget == 0U ? size.height : rows;
    return static_cast<std::size_t>((padding_rows + 2 * kRadius) * padded_width +
                                    (rows + 2 * kRadius + 1) * (padded_width + 1) * 4);
}

int check_pixels() {
    cv::RNG random(813535);
    std::size_t comparisons = 0U;
    for (const cv::Size size :
         {cv::Size(427, 240), cv::Size(1920, 1080), cv::Size(3840, 2160), cv::Size(3841, 2161)}) {
        cv::Mat parent(size.height + 4, size.width + 8, CV_8UC1);
        const cv::Mat image = parent(cv::Rect(3, 2, size.width, size.height));
        for (int kind = 0; kind < 2; ++kind) {
            if (kind == 0) {
                parent.setTo(255);
            } else {
                random.fill(parent, cv::RNG::UNIFORM, 0, 256);
            }
            for (const double constant : {-0.01, 0.0, 7.2}) {
                std::vector<cv::Mat> expected(3);
                for (std::size_t window = 0U; window < kWindows.size(); ++window) {
                    cv::adaptiveThreshold(image, expected[window], 255, cv::ADAPTIVE_THRESH_MEAN_C,
                                          cv::THRESH_BINARY_INV, kWindows[window], constant);
                }
                for (const std::size_t budget : kBudgets) {
                    aruco3cuda::cpu::detail::ThresholdWorkspace workspace;
                    std::vector<cv::Mat> outputs;
                    workspace.apply(image, kWindows, constant, &outputs, 32768, budget);
                    for (std::size_t window = 0U; window < kWindows.size(); ++window) {
                        if (cv::countNonZero(outputs[window] != expected[window]) != 0) {
                            std::cerr << "pixel mismatch " << size << " budget " << budget
                                      << " window " << kWindows[window] << " constant " << constant
                                      << '\n';
                            return 1;
                        }
                        ++comparisons;
                    }
                }
            }
        }
    }
    std::cout << "exact budget plane comparisons " << comparisons << '\n';
    return 0;
}

int benchmark() {
    cv::RNG random(813535);
    std::cout << std::setprecision(9);
    for (const cv::Size size : {cv::Size(427, 240), cv::Size(1920, 1080), cv::Size(3840, 2160)}) {
        cv::Mat parent(size.height + 4, size.width + 8, CV_8UC1);
        random.fill(parent, cv::RNG::UNIFORM, 0, 256);
        const cv::Mat image = parent(cv::Rect(3, 2, size.width, size.height));
        std::vector<aruco3cuda::cpu::detail::ThresholdWorkspace> workspaces(3);
        std::vector<std::vector<cv::Mat>> outputs(3);
        std::vector<std::vector<double>> samples_ms(3);
        for (int iteration = 0; iteration < 90; ++iteration) {
            // Rotate the first route to avoid assigning one budget a fixed
            // measurement position within each frame.
            for (int step = 0; step < 3; ++step) {
                const std::size_t variant = static_cast<std::size_t>((iteration + step) % 3);
                const auto start = std::chrono::steady_clock::now();
                workspaces[variant].apply(image, kWindows, 7.0, &outputs[variant], 32768,
                                          kBudgets[variant]);
                const auto stop = std::chrono::steady_clock::now();
                if (iteration >= 10) {
                    samples_ms[variant].push_back(
                            std::chrono::duration<double, std::milli>(stop - start).count());
                }
            }
        }
        for (std::size_t variant = 0U; variant < kBudgets.size(); ++variant) {
            for (std::size_t window = 0U; window < kWindows.size(); ++window) {
                if (cv::countNonZero(outputs[0][window] != outputs[variant][window]) != 0) {
                    return 1;
                }
            }
            std::vector<double>& values = samples_ms[variant];
            std::sort(values.begin(), values.end());
            std::cout << size.width << 'x' << size.height << ",threads=" << cv::getNumThreads()
                      << ",budget=" << kBudgets[variant] << ",p50=" << values[39]
                      << ",p95=" << values[75] << ",p99=" << values[79]
                      << ",planned_scratch_bytes=" << planned_scratch_bytes(size, kBudgets[variant])
                      << '\n';
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const int threads = argc > 2 ? std::stoi(argv[2]) : 1;
        if (threads < 1) {
            std::cerr << "thread count must be positive\n";
            return 2;
        }
        cv::setNumThreads(threads);
        return argc > 1 && std::string(argv[1]) == "benchmark" ? benchmark() : check_pixels();
    } catch (const std::exception& error) {
        std::cerr << "cache stripe probe: " << error.what() << '\n';
        return 2;
    }
}
