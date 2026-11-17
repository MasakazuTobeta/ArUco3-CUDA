// SPDX-License-Identifier: Apache-2.0
// Compare all approximation vertices, including paths rejected as non-quads.
#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "aruco3cuda/workspace.hpp"
#include "contour_approx.cuh"

namespace {
using aruco3cuda::detail::ContourPoint;
using aruco3cuda::detail::ContourSlice;

__global__ void serial_approximations(const ContourPoint* source, const int* offsets, int paths,
                                      ContourPoint* output, int* counts, ContourSlice* stack,
                                      double epsilon_rate) {
    for (int path = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); path < paths;
         path += static_cast<int>(gridDim.x * blockDim.x)) {
        const int offset = offsets[path];
        const int length = offsets[path + 1] - offset;
        counts[path] = aruco3cuda::detail::contour_approximate_closed(
                source + offset, length, output + offset, stack + offset, length * epsilon_rate);
    }
}

__global__ void warp_approximations(const ContourPoint* source, const int* offsets, int paths,
                                    ContourPoint* output, int* counts, ContourSlice* stack,
                                    double epsilon_rate) {
    for (int path = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) / 32U); path < paths;
         path += static_cast<int>(gridDim.x * blockDim.x / 32U)) {
        const int offset = offsets[path];
        const int length = offsets[path + 1] - offset;
        const int count = aruco3cuda::detail::contour_approximate_closed_warp(
                source + offset, length, output + offset, stack + offset, length * epsilon_rate);
        if ((threadIdx.x & 31U) == 0U) {
            counts[path] = count;
        }
    }
}

__global__ void compare_approximations(const int* offsets, int paths, const int* expected_count,
                                       const ContourPoint* expected, const int* serial_count,
                                       const ContourPoint* serial, const int* warp_count,
                                       const ContourPoint* warp, int* mismatches) {
    for (int path = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); path < paths;
         path += static_cast<int>(gridDim.x * blockDim.x)) {
        const int count = expected_count[path];
        bool mismatch = serial_count[path] != count || warp_count[path] != count;
        if (!mismatch) {
            for (int index = 0; index < count; ++index) {
                const int point = offsets[path] + index;
                mismatch = mismatch || serial[point].x_ != expected[point].x_ ||
                           serial[point].y_ != expected[point].y_ ||
                           warp[point].x_ != expected[point].x_ ||
                           warp[point].y_ != expected[point].y_;
            }
        }
        if (mismatch) {
            atomicAdd(mismatches, 1);
        }
    }
}

class SequentialOracle {
public:
    SequentialOracle() : threads_(cv::getNumThreads()) { cv::setNumThreads(1); }
    ~SequentialOracle() { cv::setNumThreads(this->threads_); }

private:
    int threads_;
};

void append_contours(const cv::Mat& image, std::vector<std::vector<cv::Point>>& paths) {
    std::vector<std::vector<cv::Point>> current;
    cv::findContours(image.clone(), current, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    for (auto& path : current) {
        paths.push_back(std::move(path));
    }
}

TEST(ContourApproxTest, cooperative_and_serial_vertices_match_public_oracle) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
        GTEST_SKIP();
    [[maybe_unused]] const SequentialOracle oracle;
    std::vector<std::vector<cv::Point>> paths;
    for (unsigned int bits = 0; bits < 65536U; ++bits) {
        cv::Mat image(4, 4, CV_8UC1);
        for (int pixel = 0; pixel < 16; ++pixel) {
            image.data[pixel] = (bits & (1U << pixel)) != 0 ? 255U : 0U;
        }
        append_contours(image, paths);
    }
    std::mt19937 random(93U);
    for (int trial = 0; trial < 1000; ++trial) {
        cv::Mat image(1 + random() % 32U, 1 + random() % 32U, CV_8UC1);
        for (int y = 0; y < image.rows; ++y) {
            for (int x = 0; x < image.cols; ++x) {
                image.at<std::uint8_t>(y, x) = random() % 100U < 40U ? 255U : 0U;
            }
        }
        append_contours(image, paths);
    }
    cv::Mat rings = cv::Mat::zeros(180, 300, CV_8UC1);
    cv::rectangle(rings, cv::Point(20, 20), cv::Point(100, 120), cv::Scalar(255), 1);
    cv::rectangle(rings, cv::Point(160, 40), cv::Point(260, 140), cv::Scalar(255), 1);
    cv::line(rings, cv::Point(100, 70), cv::Point(160, 70), cv::Scalar(255));
    append_contours(rings, paths);
    std::vector<ContourPoint> points;
    std::vector<int> offsets{0};
    for (const auto& path : paths) {
        for (const cv::Point& point : path) {
            points.push_back({point.x, point.y});
        }
        ASSERT_LT(points.size(), static_cast<std::size_t>(std::numeric_limits<int>::max()));
        offsets.push_back(static_cast<int>(points.size()));
    }
    const int path_count = static_cast<int>(paths.size());
    const std::size_t point_bytes = points.size() * sizeof(ContourPoint);
    const std::size_t count_bytes = paths.size() * sizeof(int);
    const std::size_t offset_bytes = offsets.size() * sizeof(int);
    aruco3cuda::Workspace workspace;
    ASSERT_EQ(workspace.ensure_capacity(5U * aruco3cuda::align_up(point_bytes, 256U) +
                                                3U * aruco3cuda::align_up(count_bytes, 256U) +
                                                aruco3cuda::align_up(offset_bytes, 256U) + 256U,
                                        aruco3cuda::MemorySpace::kDevice),
              aruco3cuda::Status::kOk);
    auto allocate = [&](std::size_t bytes) {
        void* pointer = nullptr;
        EXPECT_EQ(workspace.allocate(bytes, 256U, &pointer), aruco3cuda::Status::kOk);
        return pointer;
    };
    auto* source = static_cast<ContourPoint*>(allocate(point_bytes));
    auto* serial = static_cast<ContourPoint*>(allocate(point_bytes));
    auto* cooperative = static_cast<ContourPoint*>(allocate(point_bytes));
    auto* expected = static_cast<ContourPoint*>(allocate(point_bytes));
    auto* stack = static_cast<ContourSlice*>(allocate(points.size() * sizeof(ContourSlice)));
    auto* serial_count = static_cast<int*>(allocate(count_bytes));
    auto* warp_count = static_cast<int*>(allocate(count_bytes));
    auto* expected_count = static_cast<int*>(allocate(count_bytes));
    auto* device_offsets = static_cast<int*>(allocate(offset_bytes));
    auto* mismatches = static_cast<int*>(allocate(sizeof(int)));
    ASSERT_EQ(cudaMemcpy(source, points.data(), point_bytes, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device_offsets, offsets.data(), offset_bytes, cudaMemcpyHostToDevice),
              cudaSuccess);
    for (const double rate : {0.0, 0.01, 0.03, 0.1}) {
        SCOPED_TRACE(rate);
        std::vector<ContourPoint> reference(points.size());
        std::vector<int> reference_counts(paths.size());
        for (int path = 0; path < path_count; ++path) {
            std::vector<cv::Point> result;
            cv::approxPolyDP(paths[path], result, paths[path].size() * rate, true);
            reference_counts[path] = static_cast<int>(result.size());
            for (std::size_t index = 0; index < result.size(); ++index) {
                reference[static_cast<std::size_t>(offsets[path]) + index] = {result[index].x,
                                                                              result[index].y};
            }
        }
        ASSERT_EQ(cudaMemcpy(expected, reference.data(), point_bytes, cudaMemcpyHostToDevice),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpy(expected_count, reference_counts.data(), count_bytes,
                             cudaMemcpyHostToDevice),
                  cudaSuccess);
        ASSERT_EQ(cudaMemset(mismatches, 0, sizeof(int)), cudaSuccess);
        serial_approximations<<<128U, 128U>>>(source, device_offsets, path_count, serial,
                                              serial_count, stack, rate);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        warp_approximations<<<128U, 128U>>>(source, device_offsets, path_count, cooperative,
                                            warp_count, stack, rate);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        compare_approximations<<<128U, 128U>>>(device_offsets, path_count, expected_count, expected,
                                               serial_count, serial, warp_count, cooperative,
                                               mismatches);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        int errors = -1;
        ASSERT_EQ(cudaMemcpy(&errors, mismatches, sizeof(int), cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_EQ(errors, 0);
    }
}
}  // namespace
