# AprilTag 3

Use the CPU detector with a grayscale host buffer, or the CUDA detector with a
grayscale device buffer. Both return tag IDs and clockwise corners in the original
image coordinates, starting at the canonical top left of the tag. They support
tag16h5, tag25h9, tag36h10 and tag36h11. Reuse a detector across frames.

For single-threaded `tag36h11` video, the experimental `VideoDetector` processes
ordered Gray8 frames with one fixed configuration. It handles moving tags,
structured backgrounds and unrectified fisheye images. Small distant tags in
competition footage and low-contrast print throughput remain limiting cases.
[Video detector](small_cpu/README.md#ordered-video-search) ·
[Accuracy and FPS by workload](../Benchmark.md#results)

## Build

For CPU only, use a C++17 and C toolchain and CMake 3.24+. The build obtains
a checksum-pinned AprilRobotics source archive. Offline builds can supply that
source with `-DFETCHCONTENT_SOURCE_DIR_APRILTAG3=/path/to/source`.

```bash
cmake -S apriltag/cpu -B build/tag-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build/tag-cpu --parallel 4
```

The full build can export both targets for use through `find_package(aruco3cuda)`.
Enable the detectors you need. The NVIDIA SDK is used only by the optional comparison tool.

```bash
cmake --preset native -B build/tags -DARUCO3CUDA_BUILD_REFERENCE=OFF \
  -DARUCO3CUDA_BUILD_TESTS=OFF -DARUCO3CUDA_BUILD_APRILTAG_GPU=ON \
  -DARUCO3CUDA_BUILD_APRILTAG_CPU=ON
cmake --build build/tags --parallel 4
cmake --install build/tags --prefix "$PWD/build/tag-prefix"
build/tags/apriltag/gpu/aruco3cuda_apriltag_gpu_detect DICT_APRILTAG_36h11 frame.pgm
```

## Detect a frame in C++

Link `aruco3cuda::apriltag_cpu` or `aruco3cuda::apriltag_gpu`. The CPU API accepts
a strided view and its accessible byte length. The GPU detector is constructed
for a fixed resolution and a dictionary. It waits for the supplied stream
and returns host-owned results. Input storage belongs to the caller.

```cpp
#include <aruco3cuda/apriltag/cpu/detector.hpp>
#include <aruco3cuda/apriltag/gpu/detector.hpp>
#include <aruco3cuda/apriltag_dictionary.hpp>

// Host input, default family tag36h11.
aruco3cuda::apriltag_cpu::Detector cpu;
auto host_tags = cpu.detect(gray, size_bytes, width, height, stride_bytes);

// Device input. Construct once for this resolution and family.
auto family = aruco3cuda::find_apriltag_dictionary("DICT_APRILTAG_36h11");
aruco3cuda::apriltag_gpu::Detector gpu(*family, width, height);
auto device_input_tags = gpu.detect(device_gray, pitch_bytes, stream);
```

## Evaluation

The CPU baseline is OpenCV 4.14.0 with `CORNER_REFINE_APRILTAG`. At 640×480
each CPU detector found the same 83 of 136 rendered markers in each family, with
zero false positives, wrong IDs or rotation errors. On three outdoor photographs
the CPU detector recovered 43 of 47 AprilRobotics reference detections, including
all 25 observed by OpenCV. The photograph references come from an upstream
detector. With changing 720p and 1080p frames, the measured throughput gain
was 1.71–1.96× on DGX Spark and 1.71–3.59× on Jetson AGX Orin.
[CPU performance and accuracy](cpu/README.md) · [Inputs and scoring](evaluation/README.md)

The GPU baseline is NVIDIA cuAprilTags on its supported tag36h11 family. GPU
timing covers a completed call returning host-readable corners. Device input
and host input (including preparation and upload) are timed independently.
At 640×480, cuAprilTags detected 65 of 136 rendered markers and the CUDA
detector found all 65 plus 30 more. Both produced zero false positives and
orientation errors. Corner RMSE on their common detections was 1.515 px for
cuAprilTags and 0.183 px for the CUDA detector. In a separate comparison across
all four families, the CUDA detector missed two OpenCV CPU detections per family.

On the November 19 refresh, host-input 4K median latency improved by
6.12–19.12× across six CUDA hosts. On the 32 annotated 4K frames the
project found 35 of 48 tags, including all 18 cuAprilTags detections.
On the 96 annotated 1080p frames it found 100 of 136 and missed one
cuAprilTags detection.
[Current comparison and tail latency](../docs/measurements/2026-11-19-performance-refresh/README.md#apriltag-cuda-latency) ·
[Earlier GPU measurements](benchmark/README.md)
