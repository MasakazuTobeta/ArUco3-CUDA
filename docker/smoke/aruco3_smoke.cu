// SPDX-License-Identifier: Apache-2.0
//
// Purpose:
//   Confirm that inside the container nvcc can compile a translation unit that
//   includes OpenCV, that the resulting executable runs on the GPU, and that the
//   OpenCV ArUco3 detection strategy behaves as expected.
//
// Role:
//   This is a smoke test of the container environment, not a unit test of the
//   project. smoke-test.sh looks only at the exit code, so every condition that
//   is checked is reflected in the exit code. The printf output exists to
//   pinpoint the cause and plays no part in the pass/fail decision.
//
// Constraint:
//   A single .cu is compiled standalone with nvcc and none of the project
//   libraries are linked, so src/core/cuda_check.hpp is not used and an
//   equivalent check is self-contained in this file.
#include <cstdio>

#include <cuda_runtime_api.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

namespace {

constexpr int kElementCount = 256;
constexpr int kMarkerId = 42;
constexpr int kMarkerSidePx = 160;
constexpr int kSceneWidthPx = 1280;
constexpr int kSceneHeightPx = 720;
constexpr int kMarkerOriginXPx = 400;
constexpr int kMarkerOriginYPx = 260;

/// Check the return value of a CUDA API call and, on failure, report it together
/// with the API name, the stage, and the device.
///
/// The output format matches src/core/cuda_check.cpp so that failures look the
/// same. The cudaError_t that was already obtained is used as is, rather than
/// re-reading it with cudaGetLastError(); re-reading it could pick up a
/// different failure and misreport the original cause.
///
/// @param error The return value to check.
/// @param api_name Name of the CUDA API that was called.
/// @param stage The processing stage.
/// @return true on success.
bool cuda_ok(cudaError_t error, const char* api_name, const char* stage) {
    if (error == cudaSuccess) {
        return true;
    }
    std::fprintf(stderr, "api=%s stage=%s device=0 error=%s (%s)\n", api_name, stage,
                 cudaGetErrorName(error), cudaGetErrorString(error));
    return false;
}

/// Replace each element with 255 minus its value.
///
/// Thread to data mapping:
///   Thread i writes exactly one element, out[i]. No two threads write to the
///   same location, so there is no race and neither atomics nor synchronization
///   are needed.
/// Boundary condition:
///   count may not be a multiple of the block size, so nothing is written out of
///   range.
__global__ void invert_kernel(unsigned char* out, int count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count) {
        out[index] = static_cast<unsigned char>(255 - out[index]);
    }
}

/// Check that the CUDA side works end to end. Returns false on failure.
bool run_cuda_check() {
    unsigned char host_buffer[kElementCount];
    for (int i = 0; i < kElementCount; ++i) {
        host_buffer[i] = static_cast<unsigned char>(i);
    }

    unsigned char* device_buffer = nullptr;
    if (!cuda_ok(cudaMalloc(&device_buffer, sizeof(host_buffer)), "cudaMalloc", "smoke.alloc")) {
        return false;
    }

    bool ok = cuda_ok(cudaMemcpy(device_buffer, host_buffer, sizeof(host_buffer),
                                 cudaMemcpyHostToDevice),
                      "cudaMemcpy", "smoke.upload");
    if (ok) {
        invert_kernel<<<1, kElementCount>>>(device_buffer, kElementCount);
        // Check the launch itself first. Without synchronizing, a failure
        // during execution would not surface.
        ok = cuda_ok(cudaGetLastError(), "cudaGetLastError", "smoke.invert_kernel");
    }
    if (ok) {
        ok = cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize", "smoke.invert_kernel");
    }
    if (ok) {
        ok = cuda_ok(cudaMemcpy(host_buffer, device_buffer, sizeof(host_buffer),
                                cudaMemcpyDeviceToHost),
                     "cudaMemcpy", "smoke.download");
    }

    // Always free, including on failure paths. A failing free does not
    // overwrite the original failure.
    const bool free_ok = cuda_ok(cudaFree(device_buffer), "cudaFree", "smoke.free");
    ok = ok && free_ok;
    if (!ok) {
        return false;
    }

    // A successful transfer does not mean the computation was correct. Check
    // the values as well.
    for (int i = 0; i < kElementCount; ++i) {
        const auto expected = static_cast<unsigned char>(255 - i);
        if (host_buffer[i] != expected) {
            std::fprintf(stderr, "stage=smoke.verify index=%d expected=%d actual=%d\n", i,
                         static_cast<int>(expected), static_cast<int>(host_buffer[i]));
            return false;
        }
    }
    std::printf("cuda_ok=1\n");
    return true;
}

/// Check that the OpenCV ArUco3 detection strategy works end to end. Returns
/// false on failure.
bool run_aruco_check() {
    const cv::aruco::Dictionary dictionary =
            cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_MIP_36h12);
    std::printf("dict markerSize=%d bytesList=%d maxCorrectionBits=%d\n", dictionary.markerSize,
                dictionary.bytesList.rows, dictionary.maxCorrectionBits);

    cv::Mat marker;
    dictionary.generateImageMarker(kMarkerId, kMarkerSidePx, marker, 1);
    cv::Mat scene(kSceneHeightPx, kSceneWidthPx, CV_8UC1, cv::Scalar(255));
    marker.copyTo(scene(cv::Rect(kMarkerOriginXPx, kMarkerOriginYPx, kMarkerSidePx,
                                 kMarkerSidePx)));

    cv::aruco::DetectorParameters params;
    params.useAruco3Detection = true;
    params.minSideLengthCanonicalImg = 32;
    params.minMarkerLengthRatioOriginalImg = 0.05F;

    const auto side = static_cast<float>(params.minSideLengthCanonicalImg);
    const float fxfy =
            side / (side + static_cast<float>(std::max(scene.cols, scene.rows)) *
                                   params.minMarkerLengthRatioOriginalImg);
    std::printf("fxfy=%.4f segmentation=%dx%d\n", static_cast<double>(fxfy),
                cvRound(fxfy * scene.cols), cvRound(fxfy * scene.rows));

    const cv::aruco::ArucoDetector detector(dictionary, params);
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> ids;
    detector.detectMarkers(scene, corners, ids);

    std::printf("detected=%zu\n", ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) {
        std::printf("  id=%d corner0=(%.2f, %.2f) corner2=(%.2f, %.2f)\n", ids[i],
                    static_cast<double>(corners[i][0].x), static_cast<double>(corners[i][0].y),
                    static_cast<double>(corners[i][2].x), static_cast<double>(corners[i][2].y));
    }
    if (ids.size() != 1U || ids[0] != kMarkerId) {
        std::fprintf(stderr,
                     "stage=smoke.aruco did not detect exactly one marker with the expected "
                     "ID %d\n",
                     kMarkerId);
        return false;
    }
    return true;
}

}  // namespace

int main() {
    if (!run_cuda_check()) {
        return 1;
    }
    if (!run_aruco_check()) {
        return 1;
    }
    return 0;
}
