// SPDX-License-Identifier: Apache-2.0
//
// セル比と border 検証を CPU 基準と突き合わせる。
//
// 比の 1 段の違い (既定では 1/16) は、Dictionary 照合の距離を変えて ID を
// 変えうる。標準偏差の閾値と border 誤り数の境界を含めて確かめる。
#include "cell_decode.hpp"

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"
#include "aruco3cuda/workspace.hpp"
#include "candidate_filter.hpp"
#include "cell_sample.hpp"

namespace {

using aruco3cuda::DetectorConfig;
using aruco3cuda::MemorySpace;
using aruco3cuda::Status;
using aruco3cuda::Workspace;

bool has_cuda_device() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

/// CPU 基準のセル比。hybrid の extract_cell_pixel_ratio と同じ手順。
cv::Mat reference_ratio(const cv::Mat& canonical, int marker_size, const DetectorConfig& config) {
    const int cells = marker_size + (2 * config.marker_border_bits_);
    const int cell_size = config.perspective_remove_pixel_per_cell_;
    const int cell_margin =
            static_cast<int>(config.perspective_remove_ignored_margin_per_cell_ * cell_size);
    cv::Mat ratio(cells, cells, CV_32FC1, cv::Scalar::all(0));
    cv::Mat mean;
    cv::Mat stddev;
    cv::Mat working = canonical.clone();
    const cv::Mat inner = working.colRange(cell_size / 2, working.cols - (cell_size / 2))
                                  .rowRange(cell_size / 2, working.rows - (cell_size / 2));
    cv::meanStdDev(inner, mean, stddev);
    if (stddev.ptr<double>(0)[0] < config.min_otsu_std_dev_) {
        ratio.setTo(mean.ptr<double>(0)[0] > 127.0 ? 1.0 : 0.0);
        return ratio;
    }
    cv::threshold(working, working, 125, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    for (int y = 0; y < cells; ++y) {
        for (int x = 0; x < cells; ++x) {
            const cv::Rect cell((x * cell_size) + cell_margin, (y * cell_size) + cell_margin,
                                cell_size - (2 * cell_margin), cell_size - (2 * cell_margin));
            const cv::Mat square = working(cell);
            ratio.at<float>(y, x) = static_cast<float>(cv::countNonZero(square)) /
                                    static_cast<float>(square.total());
        }
    }
    return ratio;
}

/// CPU 基準の外周誤り数。hybrid の count_border_errors と同じ走査。
int reference_border_errors(const cv::Mat& ratio, int marker_size, const DetectorConfig& config) {
    const int cells = marker_size + (2 * config.marker_border_bits_);
    const auto threshold = static_cast<float>(config.valid_bit_threshold_);
    int errors = 0;
    for (int y = 0; y < cells; ++y) {
        for (int k = 0; k < config.marker_border_bits_; ++k) {
            if (ratio.at<float>(y, k) > threshold) {
                ++errors;
            }
            if (ratio.at<float>(y, cells - 1 - k) > threshold) {
                ++errors;
            }
        }
    }
    for (int x = config.marker_border_bits_; x < cells - config.marker_border_bits_; ++x) {
        for (int k = 0; k < config.marker_border_bits_; ++k) {
            if (ratio.at<float>(k, x) > threshold) {
                ++errors;
            }
            if (ratio.at<float>(cells - 1 - k, x) > threshold) {
                ++errors;
            }
        }
    }
    return errors;
}

/// canonical 画像を直接与えてセル比を求める。
class DecodeRun {
public:
    DecodeRun() = default;
    DecodeRun(const DecodeRun&) = delete;
    DecodeRun& operator=(const DecodeRun&) = delete;

    bool run(const std::vector<cv::Mat>& canonicals, int marker_size, const DetectorConfig& base) {
        DetectorConfig config = base;
        config.max_candidates_ = static_cast<int>(canonicals.size());
        const int side = canonicals[0].cols;

        const std::size_t bytes =
                aruco3cuda::detail::candidate_workspace_bytes(config, 64, 64) +
                aruco3cuda::detail::canonical_workspace_bytes(config, marker_size) +
                aruco3cuda::detail::cell_ratio_workspace_bytes(config, marker_size);
        if (bytes == 0U ||
            this->workspace_.ensure_capacity(bytes, MemorySpace::kDevice, nullptr) != Status::kOk) {
            return false;
        }
        this->workspace_.reset();
        aruco3cuda::detail::CandidateFilterBuffers filter;
        aruco3cuda::detail::DeviceCandidates candidates;
        aruco3cuda::detail::CanonicalBuffers canonical;
        aruco3cuda::detail::CellRatioBuffers ratios;
        if (aruco3cuda::detail::reserve_candidates(config, 64, 64, this->workspace_, &filter,
                                                   &candidates) != Status::kOk ||
            aruco3cuda::detail::reserve_canonical(config, marker_size, this->workspace_,
                                                  &canonical) != Status::kOk ||
            aruco3cuda::detail::reserve_cell_ratios(config, marker_size, this->workspace_,
                                                    &ratios) != Status::kOk) {
            return false;
        }
        if (canonical.side_px_ != side) {
            return false;
        }
        // canonical 画像を直接転送する。射影変換は別の test で確認済みのため
        // ここでは分離して比の算出だけを見る。
        const auto plane = static_cast<std::size_t>(side) * static_cast<std::size_t>(side);
        std::vector<std::uint8_t> raw(canonicals.size() * plane);
        for (std::size_t i = 0; i < canonicals.size(); ++i) {
            if (canonicals[i].data == nullptr || !canonicals[i].isContinuous()) {
                return false;
            }
            std::memcpy(raw.data() + (i * plane), canonicals[i].data, plane);
        }
        const auto count = static_cast<int>(canonicals.size());
        if (cudaMemcpy(canonical.images_, raw.data(), raw.size(), cudaMemcpyHostToDevice) !=
                    cudaSuccess ||
            cudaMemcpy(candidates.count_, &count, sizeof(int), cudaMemcpyHostToDevice) !=
                    cudaSuccess) {
            return false;
        }
        if (aruco3cuda::detail::build_cell_ratios_async(canonical, candidates, config, marker_size,
                                                        &ratios, nullptr) != Status::kOk) {
            return false;
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            return false;
        }
        return download(ratios, canonicals.size());
    }

    const std::vector<cv::Mat>& ratios() const { return this->ratios_; }
    const std::vector<std::int32_t>& border_errors() const { return this->border_errors_; }
    const std::vector<std::uint8_t>& accepted() const { return this->accepted_; }
    const std::vector<std::int32_t>& thresholds() const { return this->thresholds_; }

private:
    bool download(const aruco3cuda::detail::CellRatioBuffers& buffers, std::size_t count) {
        const auto cells = static_cast<std::size_t>(buffers.cells_per_side_);
        std::vector<float> raw(count * cells * cells);
        std::vector<std::int32_t> errors(count);
        std::vector<std::uint8_t> flags(count);
        if (cudaMemcpy(raw.data(), buffers.ratios_, raw.size() * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(errors.data(), buffers.border_errors_, errors.size() * sizeof(std::int32_t),
                       cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(flags.data(), buffers.accepted_, flags.size(), cudaMemcpyDeviceToHost) !=
                    cudaSuccess) {
            return false;
        }
        this->thresholds_.resize(count);
        if (cudaMemcpy(this->thresholds_.data(), buffers.thresholds_, count * sizeof(std::int32_t),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            return false;
        }
        this->ratios_.clear();
        this->border_errors_ = std::move(errors);
        this->accepted_ = flags;
        for (std::size_t i = 0; i < count; ++i) {
            cv::Mat ratio(buffers.cells_per_side_, buffers.cells_per_side_, CV_32FC1);
            std::memcpy(ratio.ptr<float>(0), raw.data() + (i * cells * cells),
                        cells * cells * sizeof(float));
            this->ratios_.push_back(ratio);
        }
        return true;
    }

    Workspace workspace_;
    std::vector<cv::Mat> ratios_;
    std::vector<std::int32_t> border_errors_;
    std::vector<std::uint8_t> accepted_;
    std::vector<std::int32_t> thresholds_;
};

/// マーカーらしい canonical 画像を作る。外枠が黒、内側が乱数の bit。
cv::Mat make_marker_canonical(int marker_size, int cell_size, int border_bits, std::uint64_t seed,
                              int noise_levels) {
    const int cells = marker_size + (2 * border_bits);
    const int side = cells * cell_size;
    cv::Mat image(side, side, CV_8UC1, cv::Scalar(0));
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> bit(0, 1);
    std::uniform_int_distribution<int> noise(-noise_levels, noise_levels);
    for (int cy = border_bits; cy < cells - border_bits; ++cy) {
        for (int cx = border_bits; cx < cells - border_bits; ++cx) {
            const int base = bit(rng) ? 230 : 25;
            for (int y = 0; y < cell_size; ++y) {
                for (int x = 0; x < cell_size; ++x) {
                    image.at<std::uint8_t>((cy * cell_size) + y, (cx * cell_size) + x) =
                            cv::saturate_cast<std::uint8_t>(base + noise(rng));
                }
            }
        }
    }
    // 外枠にも軽い noise を入れる。全て 0 だと Otsu の経路が変わる。
    for (int y = 0; y < side; ++y) {
        for (int x = 0; x < side; ++x) {
            if (image.at<std::uint8_t>(y, x) == 0) {
                image.at<std::uint8_t>(y, x) = cv::saturate_cast<std::uint8_t>(15 + noise(rng));
            }
        }
    }
    return image;
}

/// 比と誤り数を突き合わせ、不一致セル数を返す。
std::size_t compare_ratios(const cv::Mat& expected, const cv::Mat& actual) {
    std::size_t mismatched = 0;
    for (int y = 0; y < expected.rows; ++y) {
        for (int x = 0; x < expected.cols; ++x) {
            if (expected.at<float>(y, x) != actual.at<float>(y, x)) {
                ++mismatched;
            }
        }
    }
    return mismatched;
}

// 正常系: マーカーらしい画像でセル比が CPU 基準と一致する。
TEST(CellDecodeTest, matches_reference_ratios) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const DetectorConfig config;
    std::vector<cv::Mat> canonicals;
    canonicals.reserve(64);
    for (int i = 0; i < 64; ++i) {
        canonicals.push_back(make_marker_canonical(6, 4, 1, 20260828U + i, 20));
    }
    DecodeRun run;
    ASSERT_TRUE(run.run(canonicals, 6, config));

    std::size_t mismatched_cells = 0;
    std::size_t mismatched_errors = 0;
    for (std::size_t i = 0; i < canonicals.size(); ++i) {
        const cv::Mat expected = reference_ratio(canonicals[i], 6, config);
        mismatched_cells += compare_ratios(expected, run.ratios()[i]);
        const int expected_errors = reference_border_errors(expected, 6, config);
        if (expected_errors != run.border_errors()[i]) {
            ++mismatched_errors;
        }
    }
    std::printf("[decode] %zu 枚: 比の不一致セル %zu / %zu、誤り数の不一致 %zu\n",
                canonicals.size(), mismatched_cells, canonicals.size() * 64U, mismatched_errors);
    EXPECT_EQ(mismatched_cells, 0U);
    EXPECT_EQ(mismatched_errors, 0U);
}

// 正常系: Otsu が選ぶ閾値そのものが OpenCV と一致する。
//
// 比だけを突き合わせると、閾値が 1 階調ずれても境界に画素が無ければ気付け
// ません。比は 1/16 刻みであり、境界に画素が無い場合は同じ比になります。
// OpenCV の cv::threshold は THRESH_OTSU のとき選んだ閾値を戻り値で返すので、
// 整数で直接比べます。
TEST(CellDecodeTest, otsu_threshold_matches_opencv_exactly) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const DetectorConfig config;
    std::vector<cv::Mat> canonicals;
    canonicals.reserve(256);
    // 階調の分布を広く振る。noise の幅を変えると histogram の形が変わり、
    // 漸化式の途中で最大が入れ替わる位置も変わる。
    for (int i = 0; i < 256; ++i) {
        canonicals.push_back(make_marker_canonical(6, 4, 1, 20260830U + i, 1 + (i % 60)));
    }
    DecodeRun run;
    ASSERT_TRUE(run.run(canonicals, 6, config));
    ASSERT_EQ(run.thresholds().size(), canonicals.size());

    std::size_t mismatched = 0;
    std::size_t uniform_cases = 0;
    int smallest = 255;
    int largest = 0;
    for (std::size_t i = 0; i < canonicals.size(); ++i) {
        // 低分散の経路へ入った候補は閾値を求めないので 0 が入る。
        cv::Mat mean;
        cv::Mat stddev;
        const int cell = config.perspective_remove_pixel_per_cell_;
        const cv::Mat inner = canonicals[i]
                                      .colRange(cell / 2, canonicals[i].cols - (cell / 2))
                                      .rowRange(cell / 2, canonicals[i].rows - (cell / 2));
        cv::meanStdDev(inner, mean, stddev);
        if (stddev.ptr<double>(0)[0] < config.min_otsu_std_dev_) {
            ++uniform_cases;
            EXPECT_EQ(run.thresholds()[i], 0) << i;
            continue;
        }
        cv::Mat working = canonicals[i].clone();
        const double expected =
                cv::threshold(working, working, 125, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        if (run.thresholds()[i] != static_cast<std::int32_t>(expected)) {
            ++mismatched;
        }
        smallest = std::min(smallest, run.thresholds()[i]);
        largest = std::max(largest, run.thresholds()[i]);
    }
    std::printf("[decode] 閾値 %zu 件: 不一致 %zu、低分散 %zu、範囲 %d から %d\n",
                canonicals.size(), mismatched, uniform_cases, smallest, largest);
    EXPECT_EQ(mismatched, 0U);
    // 閾値が 1 つの値に張り付いていたら、この test は分布を振れていない。
    EXPECT_LT(smallest, largest);
}

// 境界値: 分散が小さい画像では全セルが同じ比になる。
TEST(CellDecodeTest, low_variance_fills_uniform_ratio) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const DetectorConfig config;
    // 明るい定数と暗い定数。min_otsu_std_dev_ (既定 5.0) を下回る。
    std::vector<cv::Mat> canonicals = {
            cv::Mat(32, 32, CV_8UC1, cv::Scalar(200)), cv::Mat(32, 32, CV_8UC1, cv::Scalar(30)),
            cv::Mat(32, 32, CV_8UC1, cv::Scalar(127)), cv::Mat(32, 32, CV_8UC1, cv::Scalar(128))};
    DecodeRun run;
    ASSERT_TRUE(run.run(canonicals, 6, config));
    for (std::size_t i = 0; i < canonicals.size(); ++i) {
        const cv::Mat expected = reference_ratio(canonicals[i], 6, config);
        EXPECT_EQ(compare_ratios(expected, run.ratios()[i]), 0U) << i;
        EXPECT_EQ(reference_border_errors(expected, 6, config), run.border_errors()[i]) << i;
    }
    // 平均 200 は 1.0、平均 30 と 127 は 0.0、平均 128 は 1.0 になる。
    EXPECT_FLOAT_EQ(run.ratios()[0].at<float>(0, 0), 1.0F);
    EXPECT_FLOAT_EQ(run.ratios()[1].at<float>(0, 0), 0.0F);
    EXPECT_FLOAT_EQ(run.ratios()[2].at<float>(0, 0), 0.0F);
    EXPECT_FLOAT_EQ(run.ratios()[3].at<float>(0, 0), 1.0F);
}

// 境界値: 標準偏差の閾値をまたぐ画像で判定が一致する。
TEST(CellDecodeTest, otsu_threshold_boundary) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const DetectorConfig config;
    std::vector<cv::Mat> canonicals;
    canonicals.reserve(21);
    // 振幅を少しずつ変え、標準偏差が 5.0 の前後をまたぐようにする。
    for (int amplitude = 0; amplitude <= 20; ++amplitude) {
        cv::Mat image(32, 32, CV_8UC1, cv::Scalar(128));
        for (int y = 0; y < 32; ++y) {
            for (int x = 0; x < 32; ++x) {
                image.at<std::uint8_t>(y, x) = cv::saturate_cast<std::uint8_t>(
                        128 + (((x + y) % 2 == 0) ? amplitude : -amplitude));
            }
        }
        canonicals.push_back(image);
    }
    DecodeRun run;
    ASSERT_TRUE(run.run(canonicals, 6, config));
    std::size_t mismatched = 0;
    for (std::size_t i = 0; i < canonicals.size(); ++i) {
        const cv::Mat expected = reference_ratio(canonicals[i], 6, config);
        mismatched += compare_ratios(expected, run.ratios()[i]);
    }
    std::printf("[decode] 振幅 0 から 20 の不一致セル %zu\n", mismatched);
    EXPECT_EQ(mismatched, 0U);
}

// 境界値: border 誤り数の上限で合否が変わる。
TEST(CellDecodeTest, border_error_limit) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    const DetectorConfig config;
    // 外枠の白いセル数を 0 から 28 まで増やし、上限 12 で合否が変わることを見る。
    std::vector<cv::Mat> canonicals;
    canonicals.reserve(29);
    for (int white_cells = 0; white_cells <= 28; ++white_cells) {
        cv::Mat image = make_marker_canonical(6, 4, 1, 42U, 5);
        int painted = 0;
        for (int cy = 0; cy < 8 && painted < white_cells; ++cy) {
            for (int cx = 0; cx < 8 && painted < white_cells; ++cx) {
                const bool is_border = (cy == 0 || cy == 7 || cx == 0 || cx == 7);
                if (!is_border) {
                    continue;
                }
                for (int y = 0; y < 4; ++y) {
                    for (int x = 0; x < 4; ++x) {
                        image.at<std::uint8_t>((cy * 4) + y, (cx * 4) + x) = 240U;
                    }
                }
                ++painted;
            }
        }
        canonicals.push_back(image);
    }
    DecodeRun run;
    ASSERT_TRUE(run.run(canonicals, 6, config));
    for (std::size_t i = 0; i < canonicals.size(); ++i) {
        const cv::Mat expected = reference_ratio(canonicals[i], 6, config);
        const int expected_errors = reference_border_errors(expected, 6, config);
        EXPECT_EQ(run.border_errors()[i], expected_errors) << "白セル " << i;
        // 既定は marker_size^2 * 0.35 = 12。12 は通り 13 で落ちる。
        EXPECT_EQ(run.accepted()[i] != 0U, expected_errors <= 12) << "白セル " << i;
    }
}

// 境界値: cell 辺長を変えると余白が生じ、比の分母が変わる。
TEST(CellDecodeTest, cell_margin_changes_denominator) {
    if (!has_cuda_device()) {
        GTEST_SKIP() << "CUDA device が無い環境のため skip する";
    }
    DetectorConfig config;
    // cell 8 では余白が (int)(0.13 * 8) = 1 になり、分母が 36 になる。
    config.perspective_remove_pixel_per_cell_ = 8;
    std::vector<cv::Mat> canonicals = {make_marker_canonical(6, 8, 1, 777U, 15)};
    DecodeRun run;
    ASSERT_TRUE(run.run(canonicals, 6, config));
    const cv::Mat expected = reference_ratio(canonicals[0], 6, config);
    EXPECT_EQ(compare_ratios(expected, run.ratios()[0]), 0U);
    // 分母が 36 であることを、比が 1/36 の倍数であることで確かめる。
    const float value = run.ratios()[0].at<float>(3, 3);
    EXPECT_NEAR(value * 36.0F, std::round(value * 36.0F), 1e-4F);
}

// 異常系: 引数が不正なら実行しない。
TEST(CellDecodeTest, rejects_invalid_arguments) {
    Workspace workspace;
    const DetectorConfig config;
    aruco3cuda::detail::CellRatioBuffers ratios;
    EXPECT_EQ(aruco3cuda::detail::reserve_cell_ratios(config, 6, workspace, nullptr),
              Status::kInvalidArgument);
    EXPECT_EQ(aruco3cuda::detail::reserve_cell_ratios(config, 0, workspace, &ratios),
              Status::kInvalidArgument);
    EXPECT_NE(aruco3cuda::detail::reserve_cell_ratios(config, 6, workspace, &ratios), Status::kOk);

    aruco3cuda::detail::CanonicalBuffers canonical;
    aruco3cuda::detail::DeviceCandidates candidates;
    EXPECT_EQ(aruco3cuda::detail::build_cell_ratios_async(canonical, candidates, config, 6, nullptr,
                                                          nullptr),
              Status::kInvalidArgument);

    EXPECT_EQ(aruco3cuda::detail::cells_per_side(config, 0), 0);
    EXPECT_EQ(aruco3cuda::detail::cells_per_side(config, 6), 8);
    EXPECT_EQ(aruco3cuda::detail::cell_ratio_workspace_bytes(config, 0), 0U);
}

}  // namespace
