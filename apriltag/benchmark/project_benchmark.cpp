// SPDX-License-Identifier: Apache-2.0
#include "sequence.hpp"

#include "aruco3cuda/apriltag/gpu/detector.hpp"
#include "aruco3cuda/apriltag_dictionary.hpp"

namespace {
const aruco3cuda::DictionaryTable& table() {
    const auto* family = aruco3cuda::find_apriltag_dictionary("DICT_APRILTAG_36h11");
    if (family == nullptr) {
        throw std::runtime_error("missing 36h11 family");
    }
    return *family;
}

aruco3cuda::apriltag_gpu::Config settings(int decimate, int capacity) {
    aruco3cuda::apriltag_gpu::Config config;
    config.decimate_ = decimate;
    config.max_tags_ = capacity;
    return config;
}

class ProjectDetector {
public:
    static constexpr int kMinParameter = 0;
    static constexpr int kMaxParameter = 4;
    static constexpr size_t kBytesPerPixel = 1;

    ProjectDetector(uint16_t width, uint16_t height, int decimate, int capacity)
        : detector_(table(), width, height, settings(decimate, capacity)) {}

    const void* prepare(const std::vector<uint8_t>& gray) const { return gray.data(); }

    void detect(const apriltag_benchmark::DeviceInput& input) {
        this->tags_ = this->detector_.detect(static_cast<const uint8_t*>(input.pixels_),
                                             input.pitch_, input.stream_);
    }

    size_t size() const { return this->tags_.size(); }
    int id(size_t i) const { return this->tags_[i].id_; }
    float2 corner(size_t i, size_t c) const {
        return make_float2(this->tags_[i].corners_[2 * c], this->tags_[i].corners_[2 * c + 1]);
    }

private:
    aruco3cuda::apriltag_gpu::Detector detector_;
    std::vector<aruco3cuda::apriltag_gpu::Tag> tags_;
};
}  // namespace

int main(int argc, char** argv) {
    return apriltag_benchmark::main_impl<ProjectDetector>(argc, argv);
}
