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

Detector-call latency in **milliseconds** on 16 industrial photographs with ArUco3 enabled.
Each process ran 20 warm-ups and 200 timed detections on each still image. The table gives
the median across images of their three-process median p50. All routes returned zero markers.

| Machine | OpenCV CPU (1T) | Optimized CPU (1T) | OpenCV CPU (4T) | Optimized CPU (4T) | Steady GPU (device) | Steady GPU (host) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DGX Spark GB10 | 2.984 | 1.882 | 1.395 | **0.817** | 3.405 | 3.464 |
| Jetson AGX Orin | 6.892 | 4.518 | 2.928 | **2.080** | 3.037 | 3.723 |

On Orin the device-resident GPU beat both single-thread CPU detectors on all 16 images.
The four-thread optimized CPU beat it on 15. Spark had substantial concurrent GPU work.
CPU times include host results. Device-to-device GPU times include stream synchronization.
Host-to-host GPU times also include upload and download. These timings predate the border-margin
correction. GPU throughput on changing video frames has yet to be measured.
[GPU latency, memory and measurement protocol](docs/gpu-steady-state-report.md)

- **CPU across six hosts.** With ArUco3 enabled, median per-image speedups over official OpenCV ranged from 1.45× to 1.63× with one thread and 1.33× to 1.67× with four. [Six-host comparison and absolute timings](docs/cpu-multihost-report.md#current-state)
- **Accuracy on annotated scenes.** On Spark and Orin, the corrected steady GPU and CPU each detected 113 of 136 markers with ArUco3 enabled and 124 with it disabled, with zero false positives. Recall on real photographs with visible markers remains unmeasured. [Quality evaluation](docs/gpu-steady-state-report.md#current-state)

## Documentation

- [Integrate the C++ API](docs/design/public-api.md) · [C ABI and Python](docs/design/c-abi-and-python.md) · [Stream and buffer lifetimes](src/core/detector.md)
- [Project guide and report index](docs/project-overview.md) · [Architecture](docs/architecture.md) · [Contributing](CONTRIBUTING.md)

## License and trademarks

[Apache License 2.0](LICENSE) · [Third-party notices](NOTICE) · [Licensing and provenance policy](docs/ip-and-licensing.md)

ArUco names the marker scheme published by researchers at the Universidad de Córdoba.
This independent project has no affiliation with or endorsement from the university,
the official ArUco library or OpenCV.
