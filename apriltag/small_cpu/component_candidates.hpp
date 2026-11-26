// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "aruco3cuda/apriltag/small_cpu/detector.hpp"

#include <cstdint>
#include <limits>
#include <vector>

namespace aruco3cuda::apriltag_small_cpu::detail {

// Extra low-fill or beveled quadrilaterals for source-rim verification.
struct SupplementalComponents {
    double minimum_fill_ratio_ = .19;
    std::vector<std::vector<cv::Point>> polygons_;
};

class ComponentCandidates {
public:
    // Input has fewer than 32,768 rows and columns and contains only 0 or 255.
    void extract(const cv::Mat& mask, const ComponentConfig& config,
                 std::vector<std::vector<cv::Point>>& candidates,
                 SupplementalComponents* supplemental = nullptr);

    // Use the stride-two center samples of the visible Gray8 input directly.
    void extract_native_stride2(const cv::Mat& gray, int threshold, const ComponentConfig& config,
                                std::vector<std::vector<cv::Point>>& candidates);

    // Select the native threshold path only on hosts with a supported implementation.
    static bool native_stride2_available(int width);
    static bool pack_trace_samples(const cv::Mat& gray, int threshold,
                                   std::vector<std::uint64_t>& packed);

    // Threshold and erode by a five-pixel cross on the visible ROI, then emit runs.
    void extract_dark_cross(const cv::Mat& gray, int threshold, const ComponentConfig& config,
                            std::vector<std::vector<cv::Point>>& candidates);

    // A row of Gray8 mask positions encoded as consecutive low-to-high bits.
    // The unused high bits of every partial word are zero.
    void extract_packed(const cv::Mat& gray, const std::vector<std::uint64_t>& bits,
                        const ComponentConfig& config,
                        std::vector<std::vector<cv::Point>>& candidates,
                        SupplementalComponents* supplemental);

private:
    enum class Input { binary, stride_two, dark_cross, packed };
    template <Input kInput>
    void extract_impl(const cv::Mat& input, const ComponentConfig& config,
                      std::vector<std::vector<cv::Point>>& candidates,
                      SupplementalComponents* supplemental, int threshold,
                      const std::uint64_t* bits = nullptr);

    struct Run {
        int left_;
        int end_;
        int row_;
        int label_;
    };

    struct Stats {
        int left_ = std::numeric_limits<int>::max();
        int top_ = std::numeric_limits<int>::max();
        int right_ = 0;
        int bottom_ = 0;
        int area_ = 0;
        int envelope_ = -1;
        void merge(const Stats& other);
    };

    struct Envelope {
        int row_ = -1;
        int left_ = 0;
        int right_ = 0;
        std::vector<cv::Point> left_chain_;
        std::vector<cv::Point> right_chain_;

        void clear();
        void append(const Run& run);
        void flush();
    };

    int root(int label);
    std::vector<std::uint64_t> packed_rows_;
    std::vector<Run> runs_;
    std::vector<int> parents_;
    std::vector<Stats> stats_;
    std::vector<bool> supplemental_envelopes_;
    std::vector<Envelope> envelopes_;
    std::vector<cv::Point> hull_;
    std::vector<cv::Point> quad_;
};

}  // namespace aruco3cuda::apriltag_small_cpu::detail
