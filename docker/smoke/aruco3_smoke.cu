// SPDX-License-Identifier: Apache-2.0
//
// 目的:
//   container 内で nvcc が OpenCV を含む translation unit を compile でき、
//   生成した実行 file が GPU 上で動作し、OpenCV の ArUco3 検出戦略が
//   期待どおり動くことを確認する。
//
// 位置付け:
//   container 環境の smoke test であり project の unit test ではない。
//   smoke-test.sh は exit code だけを見るため、確認したすべての条件を
//   exit code へ反映する。printf は原因の特定用であり合否の判定には使わない。
//
// 制約:
//   単一の .cu を nvcc で単体 compile し project の library を link しないため、
//   src/core/cuda_check.hpp を使わず同等の検査をこの file 内で完結させる。
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

/// CUDA API の戻り値を検査し、失敗時に API 名・処理段階・device を付けて報告する。
///
/// src/core/cuda_check.cpp と同じ書式で出力し、失敗の見え方を揃える。
/// 取得済みの cudaError_t をそのまま使い cudaGetLastError() で取り直さない。
/// 取り直すと別の失敗で上書きされ、元の原因を誤って報告する。
///
/// @param error 検査対象の戻り値。
/// @param api_name 呼び出した CUDA API 名。
/// @param stage 処理段階。
/// @return 成功なら true。
bool cuda_ok(cudaError_t error, const char* api_name, const char* stage) {
    if (error == cudaSuccess) {
        return true;
    }
    std::fprintf(stderr, "api=%s stage=%s device=0 error=%s (%s)\n", api_name, stage,
                 cudaGetErrorName(error), cudaGetErrorString(error));
    return false;
}

/// 各要素を 255 から引いた値へ置き換える。
///
/// thread とデータの対応:
///   thread i が out[i] を 1 要素だけ書き込む。書き込み先が thread 間で重複
///   しないため競合は発生せず、atomic も同期も不要である。
/// 境界条件:
///   count が block size の倍数でない場合に備え、範囲外へ書き込まない。
__global__ void invert_kernel(unsigned char* out, int count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count) {
        out[index] = static_cast<unsigned char>(255 - out[index]);
    }
}

/// CUDA 側の疎通を確認する。失敗時は false を返す。
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
        // 起動自体の失敗を先に確認する。同期しなければ実行時の失敗は現れない。
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

    // 失敗経路でも必ず解放する。free の失敗は元の失敗を上書きしない。
    const bool free_ok = cuda_ok(cudaFree(device_buffer), "cudaFree", "smoke.free");
    ok = ok && free_ok;
    if (!ok) {
        return false;
    }

    // 転送が成功しただけでは計算が正しいと言えない。値まで確認する。
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

/// OpenCV の ArUco3 検出戦略の疎通を確認する。失敗時は false を返す。
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
        std::fprintf(stderr, "stage=smoke.aruco 期待した ID %d を 1 個検出できなかった\n",
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
