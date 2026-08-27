// SPDX-License-Identifier: Apache-2.0
// 目的: container 内で nvcc が OpenCV を含む translation unit を compile でき、
//       ArUco3 検出戦略が動作することを確認する。
#include <cstdio>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

__global__ void invert_kernel(unsigned char* p, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { p[i] = static_cast<unsigned char>(255 - p[i]); }
}

int main() {
  // 1. CUDA 側が動くことを確認する
  unsigned char h[256];
  for (int i = 0; i < 256; ++i) { h[i] = static_cast<unsigned char>(i); }
  unsigned char* d = nullptr;
  cudaMalloc(&d, sizeof(h));
  cudaMemcpy(d, h, sizeof(h), cudaMemcpyHostToDevice);
  invert_kernel<<<1, 256>>>(d, 256);
  cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost);
  cudaFree(d);
  std::printf("cuda_ok=%d\n", h[10] == 245);

  // 2. OpenCV の ArUco3 検出戦略を確認する
  const cv::aruco::Dictionary dict =
      cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_MIP_36h12);
  std::printf("dict markerSize=%d bytesList=%d maxCorrectionBits=%d\n",
              dict.markerSize, dict.bytesList.rows, dict.maxCorrectionBits);

  cv::Mat marker;
  dict.generateImageMarker(42, 160, marker, 1);
  cv::Mat scene(720, 1280, CV_8UC1, cv::Scalar(255));
  marker.copyTo(scene(cv::Rect(400, 260, 160, 160)));

  cv::aruco::DetectorParameters params;
  params.useAruco3Detection = true;
  params.minSideLengthCanonicalImg = 32;
  params.minMarkerLengthRatioOriginalImg = 0.05f;

  const float s = static_cast<float>(params.minSideLengthCanonicalImg);
  const float fxfy = s / (s + std::max(scene.cols, scene.rows) *
                                  params.minMarkerLengthRatioOriginalImg);
  std::printf("fxfy=%.4f segmentation=%dx%d\n", fxfy,
              cvRound(fxfy * scene.cols), cvRound(fxfy * scene.rows));

  cv::aruco::ArucoDetector detector(dict, params);
  std::vector<std::vector<cv::Point2f>> corners;
  std::vector<int> ids;
  detector.detectMarkers(scene, corners, ids);

  std::printf("detected=%zu\n", ids.size());
  for (size_t i = 0; i < ids.size(); ++i) {
    std::printf("  id=%d corner0=(%.2f, %.2f) corner2=(%.2f, %.2f)\n", ids[i],
                corners[i][0].x, corners[i][0].y, corners[i][2].x, corners[i][2].y);
  }
  return ids.size() == 1 && ids[0] == 42 ? 0 : 1;
}
