// SPDX-License-Identifier: Apache-2.0
#include "host_sequence.hpp"

#include <opencv2/objdetect/aruco_detector.hpp>

namespace {
class OpenCvDetector {
public:
    explicit OpenCvDetector(int decimate)
        : detector_(cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h11),
                    parameters(decimate)) {}
    void detect(const cv::Mat& image) { detector_.detectMarkers(image, corners_, ids_); }
    void write(std::size_t frame) const {
        for (std::size_t j = 0; j < ids_.size(); ++j)
            apriltag_benchmark::write_tag(frame, ids_[j], corners_[j]);
    }

private:
    static cv::aruco::DetectorParameters parameters(int decimate) {
        if (decimate != 1)
            throw std::invalid_argument("The OpenCV quality reference uses native resolution");
        cv::aruco::DetectorParameters params;
        params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_APRILTAG;
        params.aprilTagQuadDecimate = 1;
        params.useAruco3Detection = false;
        return params;
    }
    cv::aruco::ArucoDetector detector_;
    std::vector<int> ids_;
    std::vector<std::vector<cv::Point2f>> corners_;
};
}  // namespace

int main(int argc, char** argv) {
    return apriltag_benchmark::main_impl<OpenCvDetector>(argc, argv);
}
