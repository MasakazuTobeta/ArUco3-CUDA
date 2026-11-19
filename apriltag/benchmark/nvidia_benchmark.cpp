// SPDX-License-Identifier: Apache-2.0
#include "sequence.hpp"

#include <cuAprilTags.h>

#include <algorithm>

namespace {
void check_detector(int status, const char* operation) {
    if (status != 0) {
        throw std::runtime_error(std::string(operation) + ": status " + std::to_string(status));
    }
}

class NvidiaDetector {
public:
    static constexpr int kMinParameter = 1;
    static constexpr int kMaxParameter = 1024;
    static constexpr size_t kBytesPerPixel = sizeof(uchar3);

    NvidiaDetector(uint16_t width, uint16_t height, int tile, int capacity)
        : tags_(capacity), rgb_(static_cast<size_t>(width) * height) {
        this->image_.width = width;
        this->image_.height = height;
        check_detector(nvCreateAprilTagsDetector(&this->detector_, width, height, tile,
                                                 NVAT_TAG36H11, nullptr, 1.0F),
                       "initialize 36h11 detector");
    }

    ~NvidiaDetector() { cuAprilTagsDestroy(this->detector_); }

    const void* prepare(const std::vector<uint8_t>& gray) {
        std::transform(gray.begin(), gray.end(), this->rgb_.begin(),
                       [](uint8_t v) { return make_uchar3(v, v, v); });
        return this->rgb_.data();
    }

    void detect(const apriltag_benchmark::DeviceInput& input) {
        this->image_.dev_ptr = static_cast<uchar3*>(input.pixels_);
        this->image_.pitch = input.pitch_;
        this->count_ = 0;
        check_detector(
                cuAprilTagsDetect(this->detector_, &this->image_, this->tags_.data(), &this->count_,
                                  static_cast<uint32_t>(this->tags_.size()), input.stream_),
                "detect");
    }

    size_t size() const { return this->count_; }
    uint16_t id(size_t i) const { return this->tags_[i].id; }
    float2 corner(size_t i, size_t c) const { return this->tags_[i].corners[c]; }

private:
    cuAprilTagsHandle detector_ = nullptr;
    cuAprilTagsImageInput_t image_{};
    std::vector<cuAprilTagsID_t> tags_;
    std::vector<uchar3> rgb_;
    uint32_t count_ = 0;
};
}  // namespace

int main(int argc, char** argv) {
    return apriltag_benchmark::main_impl<NvidiaDetector>(argc, argv);
}
