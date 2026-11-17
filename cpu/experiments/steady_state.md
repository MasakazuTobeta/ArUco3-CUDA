# Steady-state CPU experiments

## Purpose

Keep rejected or diagnostic approaches reproducible without placing them in the
production detector. Startup is outside the optimization acceptance criteria.

## Scope

These tools use public OpenCV APIs and local generated images. They perform no
tracking. The standalone experiment project is separate from both the CPU
detector and installed CUDA API.

## Current state

### Count-first contour traversal

`count_first_contours.cpp` adapts the same pinned Apache-2.0 OpenCV TRU contour
source as the [production workspace](../cpu_contours.md). It optionally marks
and counts a complete path without storing its coordinates, then retraces only
perimeter-eligible paths. Its provenance and notices are the same as the
production helper; no official ArUco GPLv3 source is used.

Build and run from the repository root after generating `data/cpu-experiment`
using the ordinary benchmark's `--generate` option:

```bash
cmake -S cpu/experiments -B build/cpu-experiments -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=/opt/opencv/lib/cmake/opencv4
cmake --build build/cpu-experiments --target aruco3cpu_count_first_probe -j2
taskset -c 5 build/cpu-experiments/aruco3cpu_count_first_probe
```

The probe compares ordered quads and full paths between the one-pass and
count-first methods before timing. It processes three threshold windows for
720p/4K clean/noisy frames, with and without ArUco3 downscaling. Timing has three
warmups, 20 samples, one OpenCV thread, alternating methods, and reports
p50/p95/p99. It measures contour extraction only, excludes thresholding, and
does not replace the actual-OpenCV detector correctness tests. With 20 samples,
p95/p99 have low tail resolution.

The archived diagnostic rejected this approach: full-resolution noisy 720p
contours increased from 35.02 to 40.65 ms and 4K from 253.24 to 276.63 ms.
Downscaled 4K noise improved from 3.38 to 3.17 ms, insufficient to justify the
other regressions. Production retains one-pass tracing.

### External compatibility inputs

`fetch_opencv_fixtures.py` downloads nine pinned regression/photo inputs and
checks both SHA-256 and Git blob hashes. `opencv-fixtures.json` records the
source revision and local-only usage; the images are not redistributed or
treated as independent ground truth.

```bash
python3 cpu/experiments/fetch_opencv_fixtures.py
build/cpu-opt/cpu/aruco3cuda_cpu_benchmark \
  --input-dir data/cpu-opencv-fixtures --dictionary DICT_6X6_250 \
  --variants previous,optimized --iterations 1 --warmup 0 --threads 4 --aruco3 0
```

Repeat across the 16 NxN families plus ORIGINAL and MIP, with both strategy
settings. The final sweep conditions (local `../../docs/measurements/2026-11-17-cpu-steady-final-external/environment.json`)
record every command and source/binary hash. Its timings are not performance
evidence because these are one-sample accuracy checks.

## Goals

- Preserve exact comparison gates and unfavorable results.
- Keep stage-only exploratory measurements separate from final latency claims.
- Retain source and input provenance for each experiment.

## Open questions

- Rejected approaches could have a different crossover point on other CPUs;
  only the measured DGX Spark workload is characterized here.
