// SPDX-License-Identifier: Apache-2.0
// Prepared, ordered Gray8 sequences for measuring synchronous CPU detector calls.
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "aruco3cuda/util/json_writer.hpp"
#include "aruco3cuda/util/sha256.hpp"
#include "aruco3cuda/util/statistics.hpp"
#include "cpu_video_detector.hpp"

namespace {
using Clock = std::chrono::steady_clock;
using aruco3cuda::cpu::CpuDetector;
using aruco3cuda::cpu::CpuResult;
using aruco3cuda::cpu::CpuVideoDetector;
using aruco3cuda::cpu::CpuVideoResult;
using Quad = std::array<cv::Point2f, 4>;
using Labels = std::map<int, Quad>;
constexpr int kFrames = 24;
constexpr int kRefreshInterval = 12;
constexpr char kDictionary[] = "DICT_ARUCO_MIP_36h12";

struct Frame {
    cv::Mat image_;
    Labels labels_;
};

struct Run {
    std::vector<double> samples_;
    std::vector<CpuResult> results_;
    double sequence_ms_ = 0.0;
    int full_calls_ = 0;
};

struct Quality {
    int missing_ = 0;
    int extra_ = 0;
    int mismatched_id_frames_ = 0;
    int max_entry_delay_frames_ = 0;
    int undetected_entries_ = 0;
    double max_truth_corner_px_ = 0.0;
    double max_baseline_corner_px_ = 0.0;
};

// Each 8x8 block at 720p has a seeded, independent gray value. The texture
// is generated once per camera view and is identical on all frames in that view.
cv::Mat texture(int scale, int seed) {
    cv::RNG random(seed);
    cv::Mat blocks(90, 160, CV_8UC1);
    random.fill(blocks, cv::RNG::UNIFORM, 70, 190);
    cv::Mat image;
    cv::resize(blocks, image, {1280 * scale, 720 * scale}, 0.0, 0.0, cv::INTER_NEAREST);
    return image;
}

void add_marker(Frame& frame, const cv::aruco::Dictionary& dictionary, int scale, int id, int x,
                int y, bool clockwise) {
    const int side = 160 * scale;
    const cv::Rect box(x * scale, y * scale, side, side);
    const int margin = 14 * scale;
    const cv::Rect backing(box.x - margin, box.y - margin, side + 2 * margin, side + 2 * margin);
    frame.image_(backing).setTo(255);
    cv::Mat marker;
    dictionary.generateImageMarker(id, side, marker);
    if (clockwise) {
        cv::rotate(marker, marker, cv::ROTATE_90_CLOCKWISE);
    }
    marker.copyTo(frame.image_(box));
    const float left = static_cast<float>(box.x);
    const float top = static_cast<float>(box.y);
    const float right = static_cast<float>(box.x + side - 1);
    const float bottom = static_cast<float>(box.y + side - 1);
    Quad corners{{{left, top}, {right, top}, {right, bottom}, {left, bottom}}};
    if (clockwise) {
        std::rotate(corners.begin(), corners.begin() + 1, corners.end());
    }
    frame.labels_.emplace(id, corners);
}

std::vector<Frame> render(const std::string& scene, int scale) {
    const auto dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_MIP_36h12);
    const cv::Mat background =
            scene == "steady_clean" ? cv::Mat(720 * scale, 1280 * scale, CV_8UC1, cv::Scalar(223))
                                    : texture(scale, 1451);
    const cv::Mat cut = scene == "cut_texture" ? texture(scale, 8473) : cv::Mat{};
    std::vector<Frame> frames;
    for (int t = 0; t < kFrames; ++t) {
        const bool after_cut = !cut.empty() && t >= 11;
        Frame frame{(after_cut ? cut : background).clone(), {}};
        if (after_cut) {
            add_marker(frame, dictionary, scale, 77, 842, 450 + t, true);
        } else {
            if (scene != "entry_exit_texture" || t < 16) {
                const int shift = scene == "steady_clean" ? 0 : 4 * t;
                add_marker(frame, dictionary, scale, 7, 205 + shift, 174, false);
            }
            if (scene == "entry_exit_texture" && t >= 7) {
                add_marker(frame, dictionary, scale, 33, 938, 471 - 2 * t, true);
            }
        }
        frames.push_back(std::move(frame));
    }
    return frames;
}

void check(aruco3cuda::Status status, const std::string& message) {
    if (status != aruco3cuda::Status::kOk) {
        throw std::runtime_error("detector failed: " + message);
    }
}

template <typename Detect>
Run measure(const std::vector<Frame>& frames, Detect detect,
            CpuVideoDetector* reset_each_frame = nullptr) {
    Run run;
    run.results_.resize(frames.size());
    run.samples_.reserve(frames.size());
    const auto sequence_start = Clock::now();
    for (std::size_t t = 0; t < frames.size(); ++t) {
        if (reset_each_frame != nullptr)
            reset_each_frame->reset();
        const auto start = Clock::now();
        const bool full = detect(frames[t].image_, run.results_[t]);
        const auto end = Clock::now();
        run.full_calls_ += full ? 1 : 0;
        run.samples_.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    run.sequence_ms_ =
            std::chrono::duration<double, std::milli>(Clock::now() - sequence_start).count();
    return run;
}

double ordered_error(const std::vector<cv::Point2f>& detected, const cv::Point2f* expected) {
    if (detected.size() != 4U) {
        throw std::runtime_error("invalid detector corner count");
    }
    double maximum = 0.0;
    for (std::size_t i = 0; i < detected.size(); ++i) {
        const double value = cv::norm(detected[i] - expected[i]);
        if (!std::isfinite(value)) {
            throw std::runtime_error("nonfinite detector corner coordinates");
        }
        maximum = std::max(value, maximum);
    }
    return maximum;
}

bool contains(const std::vector<int>& ids, int id) {
    return std::find(ids.begin(), ids.end(), id) != ids.end();
}

Quality compare(const std::vector<Frame>& frames, const Run& run, const Run& baseline) {
    Quality score;
    for (std::size_t t = 0; t < frames.size(); ++t) {
        const Labels& labels = frames[t].labels_;
        const CpuResult& result = run.results_[t];
        const CpuResult& control = baseline.results_[t];
        if (result.corners_.size() != result.ids_.size() ||
            control.corners_.size() != control.ids_.size()) {
            throw std::runtime_error("inconsistent detector IDs and corners");
        }
        if (!std::is_permutation(result.ids_.begin(), result.ids_.end(), control.ids_.begin(),
                                 control.ids_.end())) {
            ++score.mismatched_id_frames_;
        }
        for (const auto& label : labels) {
            if (!contains(result.ids_, label.first)) {
                ++score.missing_;
            }
            // Search only within this continuous appearance of the rendered marker.
            if (t == 0U || frames[t - 1U].labels_.count(label.first) == 0U) {
                std::size_t seen = t;
                while (seen < frames.size() && frames[seen].labels_.count(label.first) != 0U &&
                       !contains(run.results_[seen].ids_, label.first)) {
                    ++seen;
                }
                if (seen == frames.size() || frames[seen].labels_.count(label.first) == 0U) {
                    ++score.undetected_entries_;
                } else {
                    score.max_entry_delay_frames_ =
                            std::max(score.max_entry_delay_frames_, static_cast<int>(seen - t));
                }
            }
        }
        for (std::size_t i = 0; i < result.ids_.size(); ++i) {
            const int id = result.ids_[i];
            const auto label = labels.find(id);
            const auto first = std::find(result.ids_.begin(), result.ids_.end(), id);
            if (label == labels.end() ||
                static_cast<std::size_t>(first - result.ids_.begin()) != i) {
                ++score.extra_;
                continue;
            }
            score.max_truth_corner_px_ =
                    std::max(score.max_truth_corner_px_,
                             ordered_error(result.corners_[i], label->second.data()));
            const auto ref = std::find(control.ids_.begin(), control.ids_.end(), id);
            if (ref != control.ids_.end()) {
                const auto index = static_cast<std::size_t>(ref - control.ids_.begin());
                if (control.corners_[index].size() != 4U) {
                    throw std::runtime_error("invalid baseline corner count");
                }
                score.max_baseline_corner_px_ =
                        std::max(score.max_baseline_corner_px_,
                                 ordered_error(result.corners_[i], control.corners_[index].data()));
            }
        }
    }
    return score;
}

void report(const std::string& scene, int resolution, const std::string& digest, int repeat,
            const std::string& route, const Run& run, const Quality& score) {
    aruco3cuda::util::SampleStatistics stats;
    if (!aruco3cuda::util::compute_statistics(run.samples_, &stats)) {
        throw std::runtime_error("empty timing samples");
    }
    aruco3cuda::util::JsonWriter row(std::cout, 0);
    row.begin_object();
    row.member_string("scene", scene);
    row.member_int("height_px", resolution);
    row.member_string("pixels_sha256", digest);
    row.member_string("route", route);
    row.member_int("repeat", repeat);
    row.member_int("frames", kFrames);
    row.member_int("refresh_interval", kRefreshInterval);
    row.member_int("full_calls", run.full_calls_);
    row.member_double("fps", 1000.0 * kFrames / run.sequence_ms_, 6);
    row.member_double("first_call_ms", run.samples_.front(), 6);
    row.member_double("p50_ms", stats.p50_, 6);
    row.member_double("p95_ms", stats.p95_, 6);
    row.member_double("p99_ms", stats.p99_, 6);
    row.member_int("missed_markers", score.missing_);
    row.member_int("extra_markers", score.extra_);
    row.member_int("baseline_id_mismatch_frames", score.mismatched_id_frames_);
    row.member_double("max_truth_ordered_corner_px", score.max_truth_corner_px_, 6);
    row.member_double("max_baseline_ordered_corner_px", score.max_baseline_corner_px_, 6);
    row.member_int("max_entry_delay_frames", score.max_entry_delay_frames_);
    row.member_int("undetected_entries", score.undetected_entries_);
    row.end_object();
    std::cout << '\n';
}

void run_scene(const std::string& scene, int resolution, int repeats, bool table_baselines) {
    const auto frames = render(scene, resolution / 720);
    aruco3cuda::util::Sha256 pixels;
    for (const auto& frame : frames) {
        pixels.update(frame.image_.data, frame.image_.total());
    }
    const auto digest = pixels.finalize();
    CpuDetector detector;
    CpuVideoDetector video;
    std::string message;
    check(detector.initialize(kDictionary, {}, &message), message);
    aruco3cuda::cpu::CpuVideoOptions options;
    options.refresh_interval_ = kRefreshInterval;
    check(video.initialize(kDictionary, {}, &message, options), message);
    auto full_call = [&](const cv::Mat& image, CpuResult& result) {
        check(detector.detect(image, &result, &message), message);
        return true;
    };
    auto video_call = [&](const cv::Mat& image, CpuResult& result) {
        CpuVideoResult output;
        check(video.detect(image, &output, &message), message);
        result = std::move(output.detections_);
        return output.full_frame_;
    };
    if (table_baselines) {
        const aruco3cuda::DetectorConfig config;
        cv::aruco::DetectorParameters parameters;
        parameters.useAruco3Detection = config.use_aruco3_detection_;
        parameters.minSideLengthCanonicalImg = config.min_side_length_canonical_img_px_;
        parameters.minMarkerLengthRatioOriginalImg = config.min_marker_length_ratio_original_img_;
        parameters.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
        const cv::aruco::ArucoDetector opencv(
                cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_MIP_36h12), parameters);
        auto opencv_call = [&](const cv::Mat& image, CpuResult& result) {
            opencv.detectMarkers(image, result.corners_, result.ids_);
            return true;
        };
        const Run control = measure(frames, full_call);
        (void)measure(frames, opencv_call);
        (void)measure(frames, video_call, &video);
        for (int r = 0; r < repeats; ++r) {
            Run reference;
            Run no_history;
            if (r % 2 == 0) {
                reference = measure(frames, opencv_call);
                no_history = measure(frames, video_call, &video);
            } else {
                no_history = measure(frames, video_call, &video);
                reference = measure(frames, opencv_call);
            }
            report(scene, resolution, digest, r + 1, "opencv", reference,
                   compare(frames, reference, control));
            report(scene, resolution, digest, r + 1, "no_history", no_history,
                   compare(frames, no_history, control));
        }
        return;
    }
    (void)measure(frames, full_call);
    (void)measure(frames, video_call);
    for (int r = 0; r < repeats; ++r) {
        video.reset();
        Run control;
        Run candidate;
        if (r % 2 == 0) {
            control = measure(frames, full_call);
            candidate = measure(frames, video_call);
        } else {
            candidate = measure(frames, video_call);
            control = measure(frames, full_call);
        }
        report(scene, resolution, digest, r + 1, "full", control,
               compare(frames, control, control));
        report(scene, resolution, digest, r + 1, "video", candidate,
               compare(frames, candidate, control));
    }
}

int positive_integer(const std::string& text) {
    int value = 0;
    const char* end = text.data() + text.size();
    const auto parsed = std::from_chars(text.data(), end, value);
    if (parsed.ec != std::errc{} || parsed.ptr != end || value <= 0) {
        throw std::invalid_argument("expected a positive integer: " + text);
    }
    return value;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        int repeats = 3;
        int resolution = 720;
        bool table_baselines = false;
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--help") {
                std::cout << "aruco3cuda_cpu_video_tracking_trial [--resolution 720|2160] "
                             "[--repeats N] [--table-baselines]\n";
                return EXIT_SUCCESS;
            }
            if (option == "--table-baselines") {
                table_baselines = true;
                continue;
            }
            if ((option != "--resolution" && option != "--repeats") || ++i >= argc) {
                throw std::invalid_argument("unknown option or missing argument: " + option);
            }
            if (option == "--resolution") {
                resolution = positive_integer(argv[i]);
            } else {
                repeats = positive_integer(argv[i]);
            }
        }
        if (resolution != 720 && resolution != 2160) {
            throw std::invalid_argument("resolution must be 720 or 2160");
        }
        cv::setNumThreads(1);
        for (const char* scene :
             {"steady_clean", "motion_texture", "entry_exit_texture", "cut_texture"}) {
            run_scene(scene, resolution, repeats, table_baselines);
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
