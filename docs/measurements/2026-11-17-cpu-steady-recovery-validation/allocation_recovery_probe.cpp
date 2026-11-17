// SPDX-License-Identifier: Apache-2.0
// Ordinary allocation-exception regression; run only against the corrected helper.
#include "cpu_sparse_pyramid.hpp"

#include <opencv2/imgproc.hpp>

#include <atomic>
#include <cstddef>
#include <iostream>
#include <new>
#include <stdexcept>
#include <vector>

namespace {
std::atomic<std::size_t> failure_bytes{0U};
std::atomic<int> injected_count{0};
}  // namespace

// GNU ld forwards unarmed requests to the ordinary operator new, including
// sanitizer interception. No replacement delete or production hook is needed.
extern "C" void* __real__Znwm(std::size_t bytes);
extern "C" void* __wrap__Znwm(std::size_t bytes) {
    std::size_t expected = failure_bytes.load(std::memory_order_relaxed);
    if (expected != 0U && expected == bytes &&
        failure_bytes.compare_exchange_strong(expected, 0U, std::memory_order_relaxed)) {
        injected_count.fetch_add(1, std::memory_order_relaxed);
        throw std::bad_alloc();
    }
    return __real__Znwm(bytes);
}

namespace {
void check(bool condition, const char* text) {
    if (!condition) {
        throw std::runtime_error(text);
    }
}

void fail_map_and_recover(aruco3cuda::cpu::detail::SparsePyramid& sparse, cv::Mat& input,
                          std::size_t map_bytes, const char* scenario) {
    const cv::Size reduced((input.cols + 1) / 2, (input.rows + 1) / 2);
    const std::vector<cv::Size> sizes{input.size(), reduced};
    const cv::Rect last(reduced.width - 1, reduced.height - 1, 1, 1);
    check(map_bytes == static_cast<std::size_t>(reduced.area()), "unexpected tile map size");
    const unsigned char* pointer = input.data;
    const int before = injected_count.load(std::memory_order_relaxed);
    sparse.reset(1, 1.0);
    failure_bytes.store(map_bytes, std::memory_order_relaxed);
    bool caught = false;
    try {
        (void)sparse.region(input, sizes, 1, last);
    } catch (const std::bad_alloc&) {
        caught = true;
    } catch (...) {
        failure_bytes.store(0U, std::memory_order_relaxed);
        throw;
    }
    failure_bytes.store(0U, std::memory_order_relaxed);
    check(caught, "controlled allocation exception was not caught");
    check(injected_count.load(std::memory_order_relaxed) == before + 1,
          "unexpected allocation failure count");

    // Mimic the next detector frame at the same pointer after the failed call.
    input.setTo(239);
    check(input.data == pointer, "input storage changed");
    sparse.reset(1, 1.0);
    std::vector<cv::Mat> expected;
    cv::buildPyramid(input, expected, 1);
    const cv::Mat recovered = sparse.region(input, sizes, 1, last);
    check(cv::countNonZero(recovered != expected[1](last)) == 0, "retry pixel mismatch");
    const cv::Mat complete = sparse.region(input, sizes, 1, cv::Rect(cv::Point(), reduced));
    check(cv::countNonZero(complete != expected[1]) == 0, "complete-level mismatch");
    std::cout << "{\"scenario\":\"" << scenario << "\",\"map_bytes\":" << map_bytes
              << ",\"controlled_exception\":true,\"retry_exact\":true}\n";
}
}  // namespace

int main() {
    try {
        cv::setNumThreads(1);
        aruco3cuda::cpu::detail::SparsePyramid sparse;
        cv::Mat image(33, 65, CV_8UC1, cv::Scalar(17));
        const std::vector<cv::Size> sizes{image.size(), cv::Size(33, 17)};
        sparse.reset(64, 1.0);
        (void)sparse.region(image, sizes, 1, cv::Rect(0, 0, 1, 1));
        // Same pixel shape: Mat::create is a no-op and the map grows from 1 to 561 bytes.
        fail_map_and_recover(sparse, image, 561U, "same_shape_tile_change");
        // Changed shape: pixel allocation succeeds, then the map must grow again.
        cv::Mat larger(67, 131, CV_8UC1, cv::Scalar(71));
        fail_map_and_recover(sparse, larger, 2244U, "changed_image_shape");
        check(injected_count.load(std::memory_order_relaxed) == 2,
              "expected two injected failures");
        return 0;
    } catch (const std::exception& error) {
        failure_bytes.store(0U, std::memory_order_relaxed);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
