// SPDX-License-Identifier: Apache-2.0
// Original synthetic fixtures rendered through OpenCV's predefined Dictionary API.
#include <unistd.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "aruco3cuda/util/json_writer.hpp"
#include "aruco3cuda/util/sha256.hpp"

#ifndef ARUCO3CPU_HIGHID_SOURCE_SHA256
#define ARUCO3CPU_HIGHID_SOURCE_SHA256 ""
#endif

namespace {
namespace fs = std::filesystem;
using aruco3cuda::util::JsonWriter;

constexpr int kMarkerCount = 16;
constexpr int kBackgroundGray = 235;
constexpr std::uint64_t kNoiseSeed = 45168U;
constexpr double kNoiseSigmaLevels = 18.0;

struct SceneRecord {
    std::string filename_;
    cv::Size size_;
    int marker_side_px_ = 0;
    double noise_sigma_levels_ = 0.0;
    std::string png_sha256_;
    std::string pixels_sha256_;
    std::array<cv::Rect, kMarkerCount> bounds_{};
};

fs::path output_path(int argc, char** argv) {
    if (argc != 3 || std::string(argv[1]) != "--output-dir" || std::string(argv[2]).empty()) {
        throw std::invalid_argument("usage: aruco3cpu_generate_high_id_corpus --output-dir DIR");
    }
    fs::path output = fs::absolute(fs::path(argv[2])).lexically_normal();
    while (output.filename().empty() && output.parent_path() != output) {
        output = output.parent_path();
    }
    return output;
}

void require_available_output(const fs::path& output) {
    std::error_code error;
    const fs::file_status status = fs::symlink_status(output, error);
    if (error && error != std::errc::no_such_file_or_directory) {
        throw fs::filesystem_error("cannot inspect output directory", output, error);
    }
    if (!error && fs::exists(status) &&
        (fs::is_symlink(status) || !fs::is_directory(status) || !fs::is_empty(output))) {
        throw std::runtime_error("output must be a new or empty directory, not a symlink: " +
                                 output.string());
    }
}

/// Publish a complete generated corpus in one same-filesystem directory rename.
class StagingDirectory {
public:
    explicit StagingDirectory(const fs::path& output) {
        require_available_output(output);
        fs::create_directories(output.parent_path());
        const std::string prefix = "." + output.filename().string() + ".high-id-" +
                                   std::to_string(static_cast<long long>(getpid())) + "-";
        for (int attempt = 0; attempt < 32; ++attempt) {
            const fs::path candidate = output.parent_path() / (prefix + std::to_string(attempt));
            std::error_code error;
            if (fs::create_directory(candidate, error)) {
                this->path_ = candidate;
                return;
            }
            if (error && error != std::errc::file_exists) {
                throw fs::filesystem_error("cannot create staging directory", candidate, error);
            }
        }
        throw std::runtime_error("cannot obtain a unique staging directory");
    }
    ~StagingDirectory() {
        if (!this->path_.empty()) {
            std::error_code ignored;
            fs::remove_all(this->path_, ignored);
        }
    }
    StagingDirectory(const StagingDirectory&) = delete;
    StagingDirectory& operator=(const StagingDirectory&) = delete;

    const fs::path& path() const { return this->path_; }
    void publish(const fs::path& output) {
        require_available_output(output);
        // rename refuses a nonempty destination even if another producer wins the race.
        fs::rename(this->path_, output);
        this->path_.clear();
    }

private:
    fs::path path_;
};

std::string file_hash(const fs::path& path) {
    std::string digest;
    if (!aruco3cuda::util::sha256_file(path.string(), &digest)) {
        throw std::runtime_error("cannot hash file: " + path.string());
    }
    return digest;
}

std::string matrix_hash(const cv::Mat& image) {
    aruco3cuda::util::Sha256 hasher;
    const std::size_t row_bytes = static_cast<std::size_t>(image.cols) * image.elemSize();
    for (int row = 0; row < image.rows; ++row) {
        hasher.update(image.ptr(row), row_bytes);
    }
    return hasher.finalize();
}

SceneRecord save_scene(const fs::path& directory, const cv::Mat& image, cv::Size size,
                       int marker_side_px, double noise_sigma_levels,
                       const std::array<cv::Rect, kMarkerCount>& bounds) {
    SceneRecord scene;
    scene.filename_ = "highid_" + std::to_string(size.width) + "x" + std::to_string(size.height) +
                      (noise_sigma_levels == 0.0 ? "_clean.png" : "_noise.png");
    scene.size_ = size;
    scene.marker_side_px_ = marker_side_px;
    scene.noise_sigma_levels_ = noise_sigma_levels;
    scene.bounds_ = bounds;
    const fs::path path = directory / scene.filename_;
    if (fs::exists(path)) {
        throw std::runtime_error("refusing to replace an existing generated image: " +
                                 path.string());
    }
    if (!cv::imwrite(path.string(), image)) {
        throw std::runtime_error("PNG encoder could not write: " + path.string());
    }
    const cv::Mat decoded = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
    if (decoded.type() != CV_8UC1 || decoded.size() != image.size() ||
        cv::norm(decoded, image, cv::NORM_INF) != 0.0) {
        throw std::runtime_error("PNG readback differs from rendered pixels: " + path.string());
    }
    scene.png_sha256_ = file_hash(path);
    scene.pixels_sha256_ = matrix_hash(image);
    return scene;
}

std::vector<SceneRecord> generate(const fs::path& directory,
                                  const cv::aruco::Dictionary& dictionary) {
    const std::array<cv::Size, 3> sizes = {cv::Size(1280, 720), cv::Size(1920, 1080),
                                           cv::Size(3840, 2160)};
    const std::array<int, 3> marker_sides_px = {128, 180, 300};
    std::vector<SceneRecord> scenes;
    scenes.reserve(4U);
    for (std::size_t index = 0; index < sizes.size(); ++index) {
        const cv::Size size = sizes[index];
        const int side = marker_sides_px[index];
        cv::Mat image(size, CV_8UC1, cv::Scalar(kBackgroundGray));
        std::array<cv::Rect, kMarkerCount> bounds{};
        for (int marker_index = 0; marker_index < kMarkerCount; ++marker_index) {
            cv::Mat marker;
            cv::aruco::generateImageMarker(dictionary, 999 - marker_index, side, marker, 1);
            const int x = (marker_index % 4) * size.width / 4 + (size.width / 4 - side) / 2;
            const int y = (marker_index / 4) * size.height / 4 + (size.height / 4 - side) / 2;
            const cv::Rect placement(x, y, side, side);
            if ((placement & cv::Rect(cv::Point(), size)) != placement) {
                throw std::runtime_error("marker placement exceeds its image");
            }
            marker.copyTo(image(placement));
            bounds[static_cast<std::size_t>(marker_index)] = placement;
        }
        scenes.push_back(save_scene(directory, image, size, side, 0.0, bounds));
        if (size.height == 720) {
            cv::Mat image16;
            image.convertTo(image16, CV_16SC1);
            cv::Mat noise(size, CV_16SC1);
            cv::RNG random(kNoiseSeed);
            random.fill(noise, cv::RNG::NORMAL, 0.0, kNoiseSigmaLevels);
            image16 += noise;
            cv::Mat noisy;
            image16.convertTo(noisy, CV_8UC1);
            scenes.push_back(save_scene(directory, noisy, size, side, kNoiseSigmaLevels, bounds));
        }
    }
    return scenes;
}

void write_manifest(const fs::path& path, const cv::aruco::Dictionary& dictionary,
                    const std::vector<SceneRecord>& scenes) {
    std::ofstream stream;
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream.open(path, std::ios::binary);
    JsonWriter writer(stream, 2);
    writer.begin_object();
    writer.member_int("schema_version", 1);
    writer.member_string("generator", "aruco3cpu_generate_high_id_corpus");
    writer.member_string("generator_source", "cpu/experiments/generate_high_id_corpus.cpp");
    writer.member_string("generator_source_sha256", ARUCO3CPU_HIGHID_SOURCE_SHA256);
    writer.member_string("compiler", __VERSION__);
    writer.member_string("opencv_version", CV_VERSION);
    const std::string build = cv::getBuildInformation();
    writer.member_string("opencv_build_sha256",
                         aruco3cuda::util::sha256_bytes(build.data(), build.size()));
    writer.member_int("opencv_threads", cv::getNumThreads());
    writer.member_bool("opencv_optimized", cv::useOptimized());
    writer.member_string("dictionary", "DICT_6X6_1000");
    writer.member_string("dictionary_bytes_sha256", matrix_hash(dictionary.bytesList));
    writer.member_int("dictionary_marker_size", dictionary.markerSize);
    writer.member_int("dictionary_max_correction_bits", dictionary.maxCorrectionBits);
    writer.member_int("marker_border_bits", 1);
    writer.member_int("grid_rows", 4);
    writer.member_int("grid_columns", 4);
    writer.member_int("background_gray", kBackgroundGray);
    writer.member_int("noise_seed", static_cast<long long>(kNoiseSeed));
    writer.member_string("noise_operation",
                         "CV_16S normal noise added before saturating CV_8U conversion");
    writer.member_string("corner_convention",
                         "outer marker corner pixel centers, ordered TL,TR,BR,BL");
    writer.key("scenes");
    writer.begin_array();
    for (const SceneRecord& scene : scenes) {
        writer.begin_object();
        writer.member_string("file", scene.filename_);
        writer.member_int("width_px", scene.size_.width);
        writer.member_int("height_px", scene.size_.height);
        writer.member_int("marker_side_px", scene.marker_side_px_);
        writer.member_int("marker_count", kMarkerCount);
        writer.member_double("noise_sigma_levels", scene.noise_sigma_levels_, 1);
        writer.member_string("png_sha256", scene.png_sha256_);
        writer.member_string("pixels_sha256", scene.pixels_sha256_);
        writer.key("markers");
        writer.begin_array();
        for (int index = 0; index < kMarkerCount; ++index) {
            const cv::Rect placement = scene.bounds_[static_cast<std::size_t>(index)];
            const int right = placement.x + placement.width - 1;
            const int bottom = placement.y + placement.height - 1;
            writer.begin_object();
            writer.member_int("id", 999 - index);
            writer.key("corners_xy_px");
            writer.begin_array();
            for (const int coordinate : {placement.x, placement.y, right, placement.y, right,
                                         bottom, placement.x, bottom}) {
                writer.value_int(coordinate);
            }
            writer.end_array();
            writer.end_object();
        }
        writer.end_array();
        writer.end_object();
    }
    writer.end_array();
    writer.end_object();
    stream << '\n';
    stream.close();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "usage: aruco3cpu_generate_high_id_corpus --output-dir DIR\n"
                     "DIR must be new or empty; existing files and symlinks are refused.\n";
        return 0;
    }
    try {
        const fs::path output = output_path(argc, argv);
        StagingDirectory staging(output);
        cv::setNumThreads(1);
        const cv::aruco::Dictionary dictionary =
                cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_1000);
        if (dictionary.markerSize != 6 || dictionary.bytesList.rows != 1000) {
            throw std::runtime_error("unexpected predefined dictionary dimensions");
        }
        const std::vector<SceneRecord> scenes = generate(staging.path(), dictionary);
        const fs::path manifest = staging.path() / "manifest.json";
        write_manifest(manifest, dictionary, scenes);
        const std::string manifest_hash = file_hash(manifest);
        std::ofstream checksum;
        checksum.exceptions(std::ios::failbit | std::ios::badbit);
        checksum.open(staging.path() / "manifest.sha256", std::ios::binary);
        checksum << manifest_hash << "  manifest.json\n";
        checksum.close();
        staging.publish(output);
        std::cout << "output_dir=" << output.string() << "\nscenes=" << scenes.size()
                  << "\nmanifest_sha256=" << manifest_hash << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "high-ID corpus generation failed: " << error.what() << '\n';
        return 1;
    }
}
