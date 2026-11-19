// SPDX-License-Identifier: Apache-2.0
#include "aruco3cuda/apriltag/gpu/detector.hpp"
#include "aruco3cuda/apriltag_dictionary.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using aruco3cuda::apriltag_gpu::Config;
using aruco3cuda::apriltag_gpu::Detector;

void expect(bool value, const char* detail) {
    if (!value)
        throw std::runtime_error(detail);
}
void check(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

struct Frame {
    int width_, height_;
    std::vector<std::uint8_t> pixels_;
    std::uint8_t* device_ = nullptr;
    std::size_t pitch_ = 0;
    Frame(int width = 323, int height = 181)
        : width_(width), height_(height), pixels_(width_ * height_, 255) {
        check(cudaMallocPitch(reinterpret_cast<void**>(&device_), &pitch_, width_, height_));
    }
    ~Frame() { cudaFree(device_); }
    void upload() const {
        check(cudaMemcpy2D(device_, pitch_, pixels_.data(), width_, width_, height_,
                           cudaMemcpyHostToDevice));
    }
    void draw(const aruco3cuda::DictionaryTable& table, int id, int rotation, int left, int top,
              int width = 0, int height = 0) {
        const int side = table.marker_size_, cells = side + 2;
        if (!width)
            width = cells * 12;
        if (!height)
            height = cells * 12;
        // Use the independent canonical word and rotate coordinates on the page.
        const auto bits = table.codes_[4 * id];
        for (int r = 0; r < cells; ++r)
            for (int c = 0; c < cells; ++c) {
                int col = c, row = r;
                for (int j = 0; j < rotation; ++j) {
                    const int old = row;
                    row = col;
                    col = cells - 1 - old;
                }
                const bool white = c > 0 && c <= side && r > 0 && r <= side &&
                                   ((bits >> ((r - 1) * side + c - 1)) & 1U);
                for (int y = (row * height + cells - 1) / cells;
                     y < ((row + 1) * height + cells - 1) / cells; ++y)
                    for (int x = (col * width + cells - 1) / cells;
                         x < ((col + 1) * width + cells - 1) / cells; ++x)
                        pixels_[(top + y) * width_ + left + x] = white ? 255 : 0;
            }
    }
};
void partial_border(const aruco3cuda::DictionaryTable& family) {
    Frame image;
    Detector detector(family, image.width_, image.height_);
    // A cable touches just the top-left border. Its endpoint is a misleading
    // directional extremum, while most of the vertical edge is unobstructed.
    std::fill(image.pixels_.begin(), image.pixels_.end(), 255);
    image.draw(family, 17, 2, 28, 32);
    for (int y = 32; y < 47; ++y)
        std::fill_n(image.pixels_.begin() + y * image.width_ + 12, 16, 0);
    for (int y = 0; y < image.height_; ++y)
        for (int x = 0; x < image.width_; ++x) {
            auto& value = image.pixels_[y * image.width_ + x];
            value = (value ? 228 : 12) + (37 * x + 19 * y) % 13 - 6;
        }
    image.upload();
    auto tags = detector.detect(image.device_, image.pitch_, nullptr);
    expect(tags.size() == 1 && tags[0].id_ == 17, "cable on marker border");
    const float corners[] = {123.5F, 127.5F, 27.5F, 127.5F, 27.5F, 31.5F, 123.5F, 31.5F};
    for (int j = 0; j < 8; ++j)
        expect(std::abs(tags[0].corners_[j] - corners[j]) < 1, "refit contaminated corner");
}

void small_tag_recovery(const aruco3cuda::DictionaryTable& family) {
    Frame image;
    Config cfg;
    cfg.decimate_ = 3;
    Detector coarse(family, image.width_, image.height_, cfg);
    cfg.decimate_ = 0;
    Detector adaptive(family, image.width_, image.height_, cfg);
    cfg.max_tags_ = 2;
    Detector bounded(family, image.width_, image.height_, cfg);

    for (int repeat = 0; repeat < 2; ++repeat) {
        image.draw(family, 17, 2, 28, 28, 32, 32);
        image.draw(family, 17, 2, 100, 33, 12, 18);
        image.upload();
        expect(coarse.detect(image.device_, image.pitch_, nullptr).size() == 1,
               "undersampled second marker");
        auto tags = adaptive.detect(image.device_, image.pitch_, nullptr);
        expect(tags.size() == 2 && tags[0].id_ == 17 && tags[1].id_ == 17 &&
                       std::abs(tags[0].corners_[0] - tags[1].corners_[0]) > 40,
               "recover distinct small markers");
        bool overflow = false;
        try {
            (void)bounded.detect(image.device_, image.pitch_, nullptr);
        } catch (const std::overflow_error&) {
            overflow = true;
        }
        expect(overflow, "fine-pass overflow must reject invocation");
        std::fill(image.pixels_.begin(), image.pixels_.end(), 255);
        image.upload();
        expect(adaptive.detect(image.device_, image.pitch_, nullptr).empty(),
               "small-marker and blank replay");
        expect(bounded.detect(image.device_, image.pitch_, nullptr).empty(),
               "recover from fine-pass overflow");
    }
    Frame roi(127, 123);
    Detector cropped(family, roi.width_, roi.height_);
    roi.draw(family, 17, 2, 100, 33, 12, 18);
    roi.upload();
    auto tags = cropped.detect(roi.device_, roi.pitch_, nullptr);
    expect(tags.size() == 1 && tags[0].id_ == 17, "small ROI needs no coarse trigger");
}
void crowded_texture(const aruco3cuda::DictionaryTable& family) {
    Frame image;
    Config cfg;
    cfg.decimate_ = 1;
    cfg.cluster_slots_ = 4096;
    Detector detector(family, image.width_, image.height_, cfg);
    for (int y = 0; y < image.height_; ++y)
        for (int x = 0; x < image.width_; ++x)
            image.pixels_[y * image.width_ + x] = ((x ^ y) & 1) ? 255 : 0;
    image.upload();
    bool rejected = false;
    try {
        (void)detector.detect(image.device_, image.pitch_, nullptr);
    } catch (const std::overflow_error&) {
        rejected = true;
    }
    expect(rejected, "dense checkerboard must reject a saturated cluster table");
    std::fill(image.pixels_.begin(), image.pixels_.end(), 255);
    image.draw(family, 17, 0, 28, 32);
    image.upload();
    auto tags = detector.detect(image.device_, image.pitch_, nullptr);
    expect(tags.size() == 1 && tags[0].id_ == 17, "reuse after textured overflow");
}

void exercise(const char* name) {
    auto* family = aruco3cuda::find_apriltag_dictionary(name);
    expect(family != nullptr, "missing family");
    Frame image;
    Detector detector(*family, image.width_, image.height_);
    image.upload();
    expect(detector.detect(image.device_, image.pitch_, nullptr).empty(), "blank image");
    for (int turn = 0; turn < 4; ++turn) {
        image.draw(*family, 7, turn, 28, 32);
        image.draw(*family, 7, turn, 183, 32);
        image.upload();
        auto tags = detector.detect(image.device_, image.pitch_, nullptr);
        expect(tags.size() == 2, "retain both separated instances of same ID");
        bool found[2] = {false, false};
        for (const auto& tag : tags) {
            expect(tag.id_ == 7 && tag.hamming_ == 0, "identity and Hamming distance");
            const int right = tag.corners_[0] > 150;
            expect(!found[right], "duplicate tag");
            found[right] = true;
            const int left = right ? 183 : 28, side = 12 * (family->marker_size_ + 2);
            const int x[4] = {left, left + side, left + side, left},
                      y[4] = {32, 32, 32 + side, 32 + side};
            for (int j = 0; j < 4; ++j) {
                const int k = (turn + j) % 4;
                expect(std::abs(tag.corners_[2 * j] - (x[k] - .5F)) < 1.1F &&
                               std::abs(tag.corners_[2 * j + 1] - (y[k] - .5F)) < 1.1F,
                       "canonical corners");
            }
        }
    }
    Config cfg;
    cfg.max_tags_ = 1;
    Detector bounded(*family, image.width_, image.height_, cfg);
    bool overflow = false;
    try {
        (void)bounded.detect(image.device_, image.pitch_, nullptr);
    } catch (const std::overflow_error&) {
        overflow = true;
    }
    expect(overflow, "marker overflow must reject invocation");
    cfg.max_tags_ = 1024;
    cfg.cluster_slots_ = 1;
    Detector crowded(*family, image.width_, image.height_, cfg);
    overflow = false;
    try {
        (void)crowded.detect(image.device_, image.pitch_, nullptr);
    } catch (const std::overflow_error&) {
        overflow = true;
    }
    expect(overflow, "cluster overflow must reject invocation");
    std::fill(image.pixels_.begin(), image.pixels_.end(), 255);
    image.upload();
    expect(bounded.detect(image.device_, image.pitch_, nullptr).empty(), "recover after overflow");
    bool invalid = false;
    try {
        (void)detector.detect(image.device_, image.width_ - 1, nullptr);
    } catch (const std::invalid_argument&) {
        invalid = true;
    }
    expect(invalid, "invalid pitch must be rejected");
    invalid = false;
    try {
        (void)detector.detect(image.device_, std::numeric_limits<std::size_t>::max(), nullptr);
    } catch (const std::invalid_argument&) {
        invalid = true;
    }
    expect(invalid, "pitch multiplication overflow must be rejected");
    invalid = false;
    try {
        cfg.cluster_slots_ = 7;
        Detector malformed(*family, image.width_, image.height_, cfg);
    } catch (const std::invalid_argument&) {
        invalid = true;
    }
    expect(invalid, "cluster table requires power-of-two capacity");
}
}  // namespace
int main() {
    try {
        int count = 0;
        auto status = cudaGetDeviceCount(&count);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
            (status == cudaSuccess && !count))
            return 77;
        check(status);
        for (auto name : {"DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9", "DICT_APRILTAG_36h10",
                          "DICT_APRILTAG_36h11"})
            exercise(name);
        const auto& family = *aruco3cuda::find_apriltag_dictionary("DICT_APRILTAG_36h11");
        crowded_texture(family);
        partial_border(family);
        small_tag_recovery(family);
        std::cout << "All four square families passed rotation, strided input, repeated ID, "
                     "overflow and replay checks.\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}
