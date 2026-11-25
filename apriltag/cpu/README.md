# AprilTag 3 on CPU

## Purpose

Detect AprilTag markers in host grayscale images and return IDs and four clockwise
corners in OpenCV's canonical order. The library supports `tag16h5`, `tag25h9`,
`tag36h10` and `tag36h11`. Each detector processes one family.

## Scope

The detector uses AprilRobotics' quad extraction, edge refinement and Hamming decoder.
It keeps the family, index and thread pool across calls. Each call scans its own pixels.
Uniform leading and trailing rows can be skipped with a halo aligned to the threshold tiles.
Connected components use one 32-bit entry per pixel for the parent and component size.
This halves union-find storage while retaining the existing component traversal and
representative tie-breaking, which determine the boundary cluster ordering.
Workers label disjoint row ranges and the caller joins their boundaries before
gradient workers read the component forest. The gradient lookups use read-only
parent traversal, so they never update a shared node.
After refinement it checks convexity, projected cell size, the black border and agreement
with the decoded payload. Each cell needs a majority of its 16 samples to match.
This rejects chance matches in textured backgrounds.

The default correction budgets are 1, 2, 2 and 3 bits in family order, matching OpenCV
4.14 at its default correction rate of 0.6. The [C++ header](../../include/aruco3cuda/apriltag/cpu/detector.hpp)
defines the controls, ownership and stride requirements. The dependency is pinned to a
[BSD-licensed AprilTag 3 revision](https://github.com/AprilRobotics/apriltag/tree/efa633d9dc6f4ee33bb79b0592250f4aa71e90a8)
with a compact decoder index. [Third-party notices](../../NOTICE)

## Build and use

From the repository root, build with CMake 3.24+, a C compiler and a C++17 compiler.
The first configure downloads and verifies the pinned source archive. For an offline
build set `FETCHCONTENT_SOURCE_DIR_APRILTAG3` to the extracted revision.
The build generates copies of the pinned `apriltag_quad_thresh.c` and `apriltag.c`.
The quad extractor uses the [project's compact union-find implementation](quad_union_find.h)
and makes its gradient lookups read-only. The detector propagates a failure to
allocate the component forest as `std::system_error` with `ENOMEM`. Other upstream
allocation paths retain their pinned behavior.
The supported dimensions cap the forest at 32,767² pixels plus one unused
entry, which fits the signed parent and component-size encodings. The allocator
checks the entry count against the platform's `size_t` before multiplication.
The source checkout stays untouched and the generated files retain upstream notices.

```sh
cmake -S apriltag/cpu -B build/april-host -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/april-host
ctest --test-dir build/april-host --output-on-failure
cmake --install build/april-host --prefix "$PWD/build/april-prefix"
```

CMake consumers can set `CMAKE_PREFIX_PATH` to this prefix, call
`find_package(aruco3_apriltag_cpu REQUIRED)` and link `aruco3cuda::apriltag_cpu`.
The standalone library requires neither OpenCV nor CUDA.
The standalone CTest uses the C runtime to verify component encoding, root
tie-breaking, read-only lookup, and checked allocation sizes. It injects an allocator
failure at the largest accepted size without reserving that much memory. Set
`BUILD_TESTING=OFF` to skip this standalone test.

```cpp
#include <aruco3cuda/apriltag/cpu/detector.hpp>

aruco3cuda::apriltag_cpu::Detector detector;  // tag36h11, one thread
// Reuse detector. gray is the first visible pixel of this frame.
auto tags = detector.detect(gray, accessible_bytes, width, height, stride);
```

## Current state

**Correctness.** The comparison sets OpenCV 4.14.0 to `CORNER_REFINE_APRILTAG`
with `useAruco3Detection=false`, one border cell, the ordinary error-correction rate
and full-resolution quad extraction on both routes. DGX Spark, Jetson AGX Orin and an
x86-64 Core i9-12900KS ran 96 annotated 640×480 frames per family. Each route
found the same 83 of 136 true markers per family on every host, with zero false
positives, ID errors or orientation errors.
Across families, paired corner RMSE was 1.162 px for OpenCV and 1.155 px for this
detector. The largest individual increase in maximum corner error was 0.40 px.
On the 1280×720 tag36h11 holdout the counts were 88 and 89, including every
OpenCV match. Shared corner RMSE was 1.134 and 1.129 px.
On all three hosts, 48 of 60 independently rendered tag36h11 controls passed with the
expected ID and orientation. Both routes rejected all twelve four-bit corruptions.
Both returned zero detections on all 16 industrial photographs for all four families
on Spark.

Three 799×533 Swarmathon camera images contain 47 upstream regression labels.
OpenCV matched 25 and this detector matched 43, retaining all 25 shared tags.
Both returned zero spurious detections or wrong rotations. Paired RMSE to those
reference coordinates was 0.039 and 0.346 px. The upstream test compares against
these coordinates with edge refinement disabled. Physical corner accuracy on these
images needs independently established reference coordinates.
[Input generation and scoring](../evaluation/README.md)

**Steady-state CPU speed.** One sequential caller cycles through 16 changing
grayscale frames containing 0, 1, 4 or 12 tags. Capture and rendering occur before timing.
Throughput is measured over the complete sequence. Each result takes the median
across three independent processes, each with 16 warm-ups and 192 timed calls.
The thread count applies to either detector. Every timed run returned 816 detections.

| Host | Stream | Threads | OpenCV p50 / p95 / p99 (ms) | AprilTag CPU p50 / p95 / p99 (ms) | OpenCV → AprilTag CPU (FPS) | Gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Spark | 720p | 1 | 7.14 / 8.48 / 9.04 | 4.07 / 5.20 / 7.91 | 136.4 → 233.1 | 1.71× |
| Spark | 1080p | 1 | 15.67 / 17.42 / 18.15 | 8.60 / 9.57 / 9.69 | 63.1 → 114.4 | 1.81× |
| Spark | 720p | 4 | 7.42 / 8.69 / 9.02 | 4.10 / 5.92 / 7.02 | 132.5 → 233.4 | 1.76× |
| Spark | 1080p | 4 | 16.43 / 17.98 / 18.48 | 8.00 / 11.65 / 12.19 | 60.7 → 119.0 | 1.96× |
| Orin | 720p | 1 | 15.95 / 19.37 / 19.60 | 9.24 / 11.55 / 11.78 | 61.0 → 104.1 | 1.71× |
| Orin | 1080p | 1 | 35.89 / 39.83 / 39.96 | 20.30 / 22.92 / 23.60 | 27.6 → 48.5 | 1.76× |
| Orin | 720p | 4 | 15.73 / 18.83 / 19.09 | 4.95 / 6.04 / 6.80 | 60.9 → 194.3 | 3.19× |
| Orin | 1080p | 4 | 35.77 / 39.54 / 40.00 | 10.02 / 11.28 / 11.95 | 27.7 → 99.3 | 3.59× |

An x86-64 spot check on the Core i9-12900KS (logical CPU 6, powersave governor)
reached 133.6 and 197.9 FPS at 720p and one thread, a 1.48× gain in one
192-frame run.

The 16 fixed industrial photos use separate tag36h11 per-image latency measurements with alternating route order,
two warm-ups and ten timed calls per route. Tail values have limited resolution.

| Host | Threads | OpenCV → AprilTag CPU (median ms) | Per-image gain, median (range) |
| --- | ---: | ---: | ---: |
| Spark | 1 | 197.9 → 63.0 | 2.94× (2.35–3.36×) |
| Spark | 4 | 197.9 → 30.6 | 6.02× (3.02–7.73×) |
| Orin | 1 | 440.0 → 150.6 | 2.78× (2.29–3.24×) |
| Orin | 4 | 435.7 → 57.7 | 7.66× (5.12–8.87×) |

The Spark Release build used GCC 13.3, Ubuntu 24.04, CPU 5 or CPUs 5–8 and the
performance governor. The Orin build used GCC 9.4, Ubuntu 20.04 on L4T R35.4.1,
CPU 8 or CPUs 8–11, MAXN and the schedutil governor. Existing background services and
clock policies were retained. File decoding, detector construction and compilation
are excluded. Timing covers host input to host results. An unrelated simulator was
active on Spark. Its 1080p four-thread candidate runs ranged from 92 to 146 FPS.

## Reproduce

Configure with `-DARUCO3_APRILTAG_CPU_COMPARE=ON` in the pinned OpenCV 4.14
environment and rebuild. The comparison tests cover every family, rotations,
error correction, strided images, tiny images, a missing border and malformed inputs. At decimations
1–4 they also check oriented corners against the printed square on one and four
workers. Each observation row records the SHA-256 of the scored manifest.

```sh
ctest --test-dir build/april-host --output-on-failure
build/april-host/apriltag_cpu_compare --image scene.png --family tag36h11
python3 apriltag/cpu/observe.py build/april-host/apriltag_cpu_compare \
  build/corpus-36h11/manifest.json build/april-quality
python3 apriltag/cpu/measure.py build/april-host/apriltag_cpu_compare \
  /path/to/prepared/photos build/april-timing --cores 5,6,7,8
```

For the 3840×2160 sequence, run `apriltag_cpu_compare --stream 2160p --warmup 16
--samples 192`. It uses the same marker sizes in pixels as the 720p and 1080p
sequences. Repeat in fresh processes with the same core and thread settings,
using `--cpu-first` to reverse the detector order on the second run.

Use the [fixed photograph identities](../../tools/gpu_quality/steady-inputs.json)
and select four available physical cores on the target. The measurement runner writes
local CSV files, input hashes and resolved binary hashes.

## Goals

Preserve IDs, recall, false-positive counts and geometric accuracy while reducing
latency with the same family, correction budget and CPU thread count as OpenCV.

## Open questions

How do small, oblique or partially occluded tags behave on real camera sequences?
The four missed outdoor tags were all in the same camera image. Further annotated
footage would help distinguish perspective, lighting and pixel-size limits.
