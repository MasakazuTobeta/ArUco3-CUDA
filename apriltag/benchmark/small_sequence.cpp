// SPDX-License-Identifier: Apache-2.0
#include "aruco3cuda/apriltag/small_cpu/detector.hpp"
#include "aruco3cuda/apriltag/small_cpu/video_detector.hpp"
#include "host_sequence.hpp"

namespace {
enum class Search { kDefault, kComponents, kSampledContours, kAuto, kVideo };
class SmallDetector {
public:
    SmallDetector(int stride, Search search, int grid_radius, int threshold,
                  std::optional<int> alternate_threshold, bool border, bool recover) {
        if (search == Search::kVideo) {
            if (stride != 2 || border || recover)
                throw std::invalid_argument("video search requires stride 2 and owns its recovery");
            video_.emplace();
        } else {
            detector_.emplace(configuration(stride, search, grid_radius, threshold,
                                            alternate_threshold, border, recover));
        }
    }
    void reset() {
        if (video_)
            video_->reset();
        else
            detector_->reset();
    }
    void detect(const cv::Mat& image) {
        results_ = video_ ? video_->detect(image) : detector_->detect(image);
    }
    void write(std::size_t index) const {
        for (const auto& tag : this->results_)
            apriltag_benchmark::write_tag(index, tag.id_, tag.corners_px_);
    }

private:
    static aruco3cuda::apriltag_small_cpu::Config configuration(
            int stride, Search search, int grid_radius, int threshold,
            std::optional<int> alternate_threshold, bool border, bool recover) {
        aruco3cuda::apriltag_small_cpu::Config config;
        if (search == Search::kComponents || search == Search::kSampledContours) {
            config.components_.emplace();
            config.components_->sample_stride_px_ = stride;
            config.components_->grid_radius_px_ = grid_radius;
            config.components_->alternate_contour_threshold_ = alternate_threshold;
            if (search == Search::kSampledContours) {
                config.components_->sampled_contours_ = true;
                config.components_->threshold_ = threshold;
            }
        } else if (search == Search::kAuto) {
            if (stride != 3)
                throw std::invalid_argument("automatic search requires stride 3");
            config.automatic_.emplace();
        } else if (stride != 0) {
            config.coarse_.emplace();
            config.coarse_->sample_stride_px_ = stride;
        }
        if (border)
            config.border_.emplace();
        if (recover)
            config.recovery_.emplace();
        return config;
    }

    std::optional<aruco3cuda::apriltag_small_cpu::Detector> detector_;
    std::optional<aruco3cuda::apriltag_small_cpu::VideoDetector> video_;
    std::vector<aruco3cuda::apriltag_small_cpu::Detection> results_;
};
}  // namespace
int main(int argc, char** argv) {
    try {
        Search search = Search::kDefault;
        int radius = 0;
        int threshold = 148;
        std::optional<int> alternate_threshold;
        int options = 0;
        if (argc >= 3 && std::string_view(argv[1]) == "--search") {
            const std::string_view mode(argv[2]);
            if (mode == "components")
                search = Search::kComponents;
            else if (mode == "sampled-contours")
                search = Search::kSampledContours;
            else if (mode == "video")
                search = Search::kVideo;
            else if (mode == "auto")
                search = Search::kAuto;
            else
                throw std::invalid_argument(
                        "expected --search components, sampled-contours, auto, or video");
            options = 2;
            if (argc >= 5 && std::string_view(argv[3]) == "--grid-radius") {
                if (search != Search::kComponents)
                    throw std::invalid_argument("grid radius requires component search");
                radius = apriltag_benchmark::integer(argv[4], 0, 32);
                options += 2;
            }
            if (argc > options + 1 && std::string_view(argv[options + 1]) == "--threshold") {
                if (search != Search::kSampledContours || argc <= options + 2)
                    throw std::invalid_argument("threshold requires sampled-contours and a value");
                threshold = apriltag_benchmark::integer(argv[options + 2], 0, 255);
                options += 2;
            }
            if (argc > options + 1 &&
                std::string_view(argv[options + 1]) == "--alternate-contours") {
                if (search != Search::kComponents || argc <= options + 2)
                    throw std::invalid_argument(
                            "alternate contours require component search and a threshold");
                alternate_threshold = apriltag_benchmark::integer(argv[options + 2], 0, 255);
                options += 2;
            }
        }
        const bool border = argc > options + 1 && std::string_view(argv[options + 1]) == "--border";
        options += border ? 1 : 0;
        const bool recover =
                argc > options + 1 && std::string_view(argv[options + 1]) == "--recover";
        options += recover ? 1 : 0;
        if (alternate_threshold && !recover)
            throw std::invalid_argument("alternate contours require --recover");
        return apriltag_benchmark::run_with_prepare<SmallDetector>(
                argc - options, argv + options, [](SmallDetector& detector) { detector.reset(); },
                search, radius, threshold, alternate_threshold, border, recover);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
