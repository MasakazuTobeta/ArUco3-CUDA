// SPDX-License-Identifier: Apache-2.0
#include "cpu_detector.hpp"
#include "host_sequence.hpp"

namespace {
class ContourDetector {
public:
    explicit ContourDetector(int stride) {
        aruco3cuda::cpu::CpuOptions options;
        options.candidate_box_stride_px_ = stride;
        options.decode_at_source_resolution_ = true;
        options.search_frame_edges_ = true;
        options.decoded_grid_radius_px_ = 6;
        aruco3cuda::DetectorConfig config;
        config.min_marker_length_ratio_original_img_ = .04F;
        config.min_side_length_canonical_img_px_ = 32;
        config.adaptive_thresh_win_size_min_px_ = 3;
        config.adaptive_thresh_win_size_max_px_ = 23;
        config.adaptive_thresh_win_size_step_px_ = 20;
        config.min_distance_to_border_px_ = 0;
        if (this->detector_.initialize("DICT_APRILTAG_36h11", config, &this->message_, options) !=
            aruco3cuda::Status::kOk)
            throw std::runtime_error(this->message_);
    }
    void detect(const cv::Mat& image) {
        if (this->detector_.detect(image, &this->result_, &this->message_) !=
            aruco3cuda::Status::kOk)
            throw std::runtime_error(this->message_);
    }
    void write(std::size_t index) const {
        for (std::size_t i = 0; i < this->result_.ids_.size(); ++i)
            apriltag_benchmark::write_tag(index, this->result_.ids_[i], this->result_.corners_[i]);
    }

private:
    aruco3cuda::cpu::CpuDetector detector_;
    aruco3cuda::cpu::CpuResult result_;
    std::string message_;
};
}  // namespace
int main(int argc, char** argv) {
    return apriltag_benchmark::main_impl<ContourDetector>(argc, argv);
}
