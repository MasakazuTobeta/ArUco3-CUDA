# ArUco3-CUDA

[![CI](https://github.com/MasakazuTobeta/ArUco3-CUDA/actions/workflows/ci.yml/badge.svg)](https://github.com/MasakazuTobeta/ArUco3-CUDA/actions/workflows/ci.yml)

Detect ArUco markers on the GPU using the ArUco3 detection strategy from OpenCV.
Give the detector an 8-bit grayscale image in device or managed memory. It returns marker IDs,
rotations and four corners in the original image coordinates. Results can stay on the GPU
for the next stage or be downloaded to the host.

The CUDA library has C++, C and Python interfaces and supports 17 predefined dictionaries.
An experimental host-only detector is available for applications with CPU input.
For AprilTag families, see the separate [AprilTag 3 CPU and GPU detectors](apriltag/README.md).

## Detect your first marker

On a machine with an NVIDIA GPU, a compatible CUDA toolkit and C++ compiler, CMake 3.25+
and Ninja, run these commands from the repository root. They build the samples without
OpenCV, generate a marker and detect it. A [Docker environment](docs/design/docker-environment.md)
is available for DGX Spark, Jetson AGX Orin, Jetson AGX Thor and RTX 5070 Ti.

```bash
cmake --preset native -DARUCO3CUDA_BUILD_REFERENCE=OFF -DARUCO3CUDA_BUILD_TESTS=OFF
cmake --build --preset native --parallel 4
build/native/examples/generate_marker --id 42 --size 200 --margin 40 --output build/marker.pgm
build/native/examples/detect_image --input build/marker.pgm
```

Expect one detection with ID `42`. For Python examples, installation with CMake
`find_package`, or help with a missed marker, see the [sample guide](examples/README.md).

## Choose a detector

| Build | When to use it |
| --- | --- |
| `dgx-spark-steady` or `jetson-orin-steady` | GPU input on Spark or Orin. Ordered contours recover markers whose borders connect to dark background lines. The expanded quality evaluation uses these presets. |
| `native` and ordinary GPU presets | Run the original connected-component detector. It uses less workspace but can miss markers in those connected contours. |
| `cpu/standalone` | CPU input and host results. The optimized detector uses OpenCV 4.14 and processes each frame independently. It builds without CUDA. |

To build the evaluated GPU configuration in the pinned development environment, select
the preset for your machine. The reported runs used limits of 16,384 candidates and
1,024 markers, set at runtime. [Configuration and measurement conditions](docs/gpu-steady-state-report.md#measurement-conditions)

```bash
cmake --preset dgx-spark-steady -DOpenCV_DIR=/opt/opencv/lib/cmake/opencv4
cmake --build --preset dgx-spark-steady
ctest --preset dgx-spark-steady --output-on-failure
```

To try the optimized CPU detector alongside official OpenCV, build the standalone benchmark.
The last command generates sample images and runs both detectors. [CPU detector and tuning options](cpu/cpu_detector.md)

```bash
cmake -S cpu/standalone -B build/cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DARUCO3CUDA_BUILD_TESTS=OFF
cmake --build build/cpu --target aruco3cuda_cpu_benchmark --parallel 4
build/cpu/cpu/aruco3cuda_cpu_benchmark --input-dir build/cpu-scenes --generate \
  --variants opencv,optimized --iterations 2 --warmup 1
```

## Performance and detection quality

### ArUco

Steady detector-call latency in **milliseconds** on 16 industrial photographs
with ArUco3 enabled. The table gives the median across images of their
three-process median p50. All routes returned zero markers.

| Machine | OpenCV CPU (1T) | Optimized CPU (1T) | OpenCV CPU (4T) | Optimized CPU (4T) | Steady GPU (device) | Steady GPU (host) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DGX Spark GB10 | 2.934 | 1.843 | 1.368 | **0.790** | 1.340 | 1.553 |
| Jetson AGX Orin | 6.853 | 4.469 | 2.936 | **2.076** | 3.034 | 3.956 |
| Jetson AGX Thor | 5.086 | 3.110 | 2.211 | **1.333** | 1.956 | 2.098 |
| Xeon w7-3565X / RTX PRO 6000 | 3.573 | 2.318 | 1.415 | **1.060** | 1.207 | 1.404 |
| Core i9-12900KS / RTX 4090 | 2.974 | 1.944 | 1.143 | **0.807** | 0.828 | 1.106 |
| Core Ultra 7 265 / RTX 5070 Ti | 2.724 | 1.861 | 1.110 | **0.761** | 1.046 | 1.209 |

Host GPU times include upload and download. On 32 changing 4K frames,
host GPU p50 ranged from 1.31 to 4.23 ms across these machines.
[ArUco benchmark details](docs/measurements/2026-11-19-performance-refresh/README.md#aruco-changing-frames)

#### CPU video

One-core median detector-call latency (p50, ms) for a moving marker over
fixed texture. All routes process the same 24 rendered frames with ArUco3
enabled. “Single image” uses the project's full-frame detector. “No history”
resets the video detector before every frame. Ordered mode retains history
within the sequence. OpenCV 4.14.0 uses matching ArUco3 settings.

| CPU host | 720p OpenCV | 720p single image | 720p no history | 720p ordered | 4K OpenCV | 4K single image | 4K no history | 4K ordered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DGX Spark | 2.317 | 1.440 | 1.452 | 0.307 | 4.144 | 2.332 | 2.470 | 0.765 |
| Jetson AGX Orin | 5.107 | 3.459 | 3.564 | 0.839 | 10.631 | 6.051 | 6.466 | 2.475 |
| Jetson AGX Thor | 3.894 | 2.421 | 2.453 | 0.542 | 8.123 | 4.095 | 4.252 | 1.424 |
| Xeon w7-3565X | 2.663 | 1.988 | 2.035 | 0.419 | 5.342 | 3.107 | 3.316 | 1.122 |
| Core i9-12900KS | 2.273 | 1.818 | 1.847 | 0.385 | 4.399 | 2.839 | 2.938 | 1.023 |
| Core Ultra 7 265 | 2.159 | 1.737 | 1.772 | 0.355 | 4.185 | 2.700 | 2.803 | 0.937 |

Across the three textured sequences, the video detector delivered
1.89–3.71× full-frame FPS. All four
sequences had zero missed tags, extra detections, or first-appearance delays.
Full searches can raise tail latency. Small tags can also change detection
status when region and full-frame search use different scales.
[FPS, tail latency, detection quality and conditions](cpu/cpu_video_detector.md#measurement)

### AprilTag

#### CPU

Four-thread results on a 16-frame 3840×2160 `tag36h11` sequence with
0, 1, 4, or 12 tags per frame. The reference is OpenCV 4.14.0 with
`CORNER_REFINE_APRILTAG`. Both detectors process host grayscale images and
return host detections.

| CPU host | OpenCV p50 (ms) | Project CPU p50 (ms) | Throughput, OpenCV → project (FPS) |
| --- | ---: | ---: | ---: |
| DGX Spark | 95.214 | 37.735 | 10.49 → 26.44 |
| Jetson AGX Orin | 154.667 | 39.367 | 6.47 → 25.31 |
| Jetson AGX Thor | 107.938 | 22.105 | 9.22 → 43.40 |
| Xeon w7-3565X | 106.559 | 44.614 | 9.37 → 22.35 |
| Core i9-12900KS | 86.658 | 34.550 | 11.53 → 28.98 |
| Core Ultra 7 265 | 189.049 | 42.561 | 5.31 → 23.48 |

[CPU benchmark details, including one-thread results](docs/measurements/2026-11-19-performance-refresh/README.md#apriltag-cpu-at-4k)

**Single images vs. ordered video, one CPU core per host.** All routes
process the same predecoded `tag36h11` frames. “Single image” uses the
project's experimental `Detector`. “No history” resets `VideoDetector` before
every frame. Ordered mode retains history within each traversal. OpenCV 4.14.0
uses native-resolution AprilTag detection. Values are median per-frame
detector-call latencies (p50) in milliseconds. Lower is better.

| CPU host | 720p OpenCV | 720p single image | 720p no history | 720p ordered | 4K OpenCV | 4K single image | 4K no history | 4K ordered |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DGX Spark | 42.185 | 0.842 | 0.165 | 0.123 | 35.511 | 1.197 | 0.769 | 0.781 |
| Jetson AGX Orin | 107.260 | 2.094 | 0.606 | 0.448 | 67.484 | 4.293 | 2.660 | 2.666 |
| Jetson AGX Thor | 71.947 | 1.384 | 0.331 | 0.220 | 51.782 | 1.887 | 1.334 | 1.358 |
| Xeon w7-3565X | 48.807 | 1.018 | 0.225 | 0.153 | 52.589 | 1.995 | 1.101 | 1.091 |
| Core i9-12900KS | 44.914 | 0.906 | 0.220 | 0.145 | 37.730 | 1.371 | 0.689 | 0.682 |
| Core Ultra 7 265 | 41.746 | 0.846 | 0.218 | 0.140 | 140.663 | 1.360 | 0.667 | 0.668 |

On all six hosts, the video detector lowered median latency versus the
single-image detector even without history. Using history lowered p50
further on the 72-frame 720p motion input, but not consistently on the 32
independently rendered 4K frames. The p50 does not capture the long
OpenCV calls on the 4K checkerboard.
[Throughput, tail latency, detection quality and conditions](Benchmark.md#single-images-vs-ordered-video)

#### CUDA

Median detector-call latency (p50) on 32 changing 3840×2160 `tag36h11`
frames, from host grayscale input to host detections. Both GPU timings
include input preparation, upload, and result retrieval. FPS is the median
across runs of frames processed divided by total timed detector-call duration.

| GPU | NVIDIA cuAprilTags p50 (ms) | Project CUDA p50 (ms) | Throughput, cuAprilTags → project (FPS) |
| --- | ---: | ---: | ---: |
| DGX Spark GB10 | 10.304 | 1.280 | 94.30 → 581.92 |
| Jetson AGX Orin | 35.428 | 3.108 | 27.30 → 180.79 |
| Jetson AGX Thor | 19.427 | 1.016 | 49.39 → 462.63 |
| RTX PRO 6000 | 7.436 | 1.215 | 133.68 → 662.60 |
| RTX 4090 | 8.903 | 1.130 | 111.66 → 711.50 |
| RTX 5070 Ti | 8.208 | 0.860 | 117.30 → 852.23 |

On these 4K frames, the project detected 35 of 48 annotated tags, including
all 18 cuAprilTags detections.
[CUDA latency and detection quality](docs/measurements/2026-11-19-performance-refresh/README.md#apriltag-cuda-latency)

## Documentation

- [Integrate the C++ API](docs/design/public-api.md) · [C ABI and Python](docs/design/c-abi-and-python.md) · [Stream and buffer lifetimes](src/core/detector.md)
- [Project guide and report index](docs/project-overview.md) · [Architecture](docs/architecture.md) · [Contributing](CONTRIBUTING.md)

## License and trademarks

[Apache License 2.0](LICENSE) · [Third-party notices](NOTICE) · [Licensing and provenance policy](docs/ip-and-licensing.md)

ArUco names the marker scheme published by researchers at the Universidad de Córdoba.
This independent project has no affiliation with or endorsement from the university,
the official ArUco library or OpenCV.
