# AprilTag GPU comparison

## Purpose

Compare the project detector with NVIDIA cuAprilTags on the same 36h11 images,
with a persistent detector and host-accessible IDs and corners after each call.

## Scope

NVIDIA publishes the Isaac ROS node under Apache-2.0. Its CUDA path calls a
precompiled SDK for 36h11. The public header carries Apache-2.0. The vendor
package declares the NVIDIA Isaac ROS Software License for the SDK. The
benchmark accepts an external header and platform archive and requires no ROS
installation. [Node implementation](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_apriltag/blob/a32cf16997c5405edde1b4e00bfba043e3ed2e5b/isaac_ros_apriltag/src/apriltag_node.cpp#L391-L499) ·
[Vendor package](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros/blob/94ada8a17231ec3d333664fc1320faa5e4bd1b7b/cuapriltags_vendor/package.xml) ·
[SDK terms](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros/blob/94ada8a17231ec3d333664fc1320faa5e4bd1b7b/LICENSE)

## Current state

The shared C++ harness uses one stream, one device input allocation, a fixed
image size and one detector per process. It cycles through the manifest in
order for every warm-up and timed loop. Each detected frame is synchronized.
Both routes return host results. File I/O, trace writing and initialization
are outside timing. Pose estimation is disabled for the NVIDIA route.

| Scope | Start | End |
| --- | --- | --- |
| `device` | Native input already uploaded and synchronized (gray8 for project, replicated RGB8 for NVIDIA) | Host results and stream completion |
| `host` | Original decoded host gray8 frame, before RGB expansion or upload | Host results and stream completion |

These are serial changing-frame call latencies. Camera capture, decoding,
queueing and pipelined throughput need a separate application-level study.
The NVIDIA runner retains the Isaac ROS tile size of 4. With a response at
capacity the harness fails the run so a truncated output cannot score as a pass.

### Quality on the common tag36h11 family

Both hosts produced identical observations for each route on the 96 annotated
640×480 images, the payload-damage sweep and the NVIDIA proof-of-life fixture.
The normalized corner order is compared to the tag's printed orientation.
[Inputs and scoring procedure](../evaluation/README.md)

| Corpus | True markers | NVIDIA detected | Project detected | NVIDIA / project corner RMSE on common detections |
| --- | ---: | ---: | ---: | ---: |
| 640×480 pose, projective, blur, lighting, occlusion, borders, background clutter and groups | 136 | 65 | 95 | 1.515 / 0.183 px |
| 1280×720 second-resolution holdout with the same scene generator | 136 | 57 | 104 | 1.164 / 0.190 px |
| Four rotations, three sizes, zero to four flipped payload cells | 60 | 36 | 36 | 0.707 / 0.001 px |

The project recovered every NVIDIA-detected ground-truth marker at both
resolutions. Both detectors produced zero false positives, wrong IDs or
orientation errors across these runs. Each decoded all 12 examples at each damage count up to two bits and
rejected the three- and four-bit variants. At 1920×1080 both returned ID 0 once
on NVIDIA's integration fixture. NVIDIA matched its nominal corners exactly.
The project corners satisfied the upstream 2 px per-coordinate tolerance.

Over all project detections, corner RMSE was 0.218 px at 640×480 and
0.228 px at 1280×720. The largest single errors were 1.26 and 1.31 px.
An earlier edge fit displaced one corner by 26.78 px in a dense 720p scene.
The reported build uses robust edge fitting after payload decoding and checks
the decoded result again. Every common marker had an equal or smaller maximum
corner error than NVIDIA on the two synthetic corpora.

The outdoor field check uses three 799×533 NASA Swarmathon photographs with
12, 25 and 10 reference positions, all ID 0. Their reference quads have edge
lengths from 10 to 46 px. The accompanying AprilRobotics regression files
record detector outputs. NVIDIA matched 15 of 47 positions. The project
matched 45 including all 15, with common-position RMSE of 0.685 px versus
1.681 px for NVIDIA. Its two other detections overlapped visible oblique
printed tags on the third image. The references do not verify their IDs or
corner order. The fixed-spacing 3 build matched only 23 positions and missed
three NVIDIA positions. Fixed-spacing 1 alone lost one NVIDIA-detected
marker in the synthetic 640×480 dense scene.
[Field corpus preparation and credits](../evaluation/README.md)

A supplemental partial-view test centered a 128×128 crop on each of the 47
reference quads. For each axis, subtract 64 from the rounded mean of the four
corner coordinates and clamp the origin to the image boundary. Preserve native
pixels and score only the central reference. Neighboring detections may appear
inside the crop and have no label in this test. The earlier adaptive build
lost one of NVIDIA's 15 central targets, in the first image at crop origin
(278,296). In the measured detector, automatic mode scans images of up to
36,864 pixels at full resolution. Both machines matched 45 of 47 central targets,
including all 15 NVIDIA targets.

### Changing-frame latency

Each of three fresh processes ran two warm-up passes over the same ordered
96-frame corpus and ten timed passes (960 calls per process). Project ran first
in rounds one and three, NVIDIA first in round two. Both used a tag capacity of
1,024. The maximum observed counts were 5 for NVIDIA and 8 for the project.
The project used its default adaptive segmentation and Hamming distance 2.
These full frames begin at spacing 3, with a full-resolution second pass on
small near-matches. Small regions of interest begin at spacing 1. Times below are
medians of the three process-level nearest-rank percentiles over all timed calls.

| Machine | Input | NVIDIA p50 / p95 / p99 (ms) | Project p50 / p95 / p99 (ms) | NVIDIA / project p50 |
| --- | --- | ---: | ---: | ---: |
| Jetson AGX Orin | Device | 1.152 / 3.047 / 3.441 | 0.437 / 0.629 / 0.963 | 2.64× |
| Jetson AGX Orin | Host | 1.670 / 3.559 / 3.872 | 0.589 / 0.877 / 1.040 | 2.83× |
| GeForce RTX 4090 | Device | 2.109 / 2.352 / 2.429 | 0.272 / 2.374 / 2.399 | 7.76× |
| GeForce RTX 4090 | Host | 2.231 / 2.531 / 2.636 | 0.302 / 2.418 / 2.436 | 7.38× |

Across processes, device p50 ranged from 1.145–1.158 ms for NVIDIA and
0.432–0.437 ms for the project on Orin. On RTX the ranges were 2.041–2.110
and 0.268–0.276 ms. Across the 2,880 calls per route, NVIDIA and the project
served 577 and 2,148 calls/s on Orin, and 635 and 1,418 calls/s on RTX.
The rate is the call count divided by total synchronized detector time.
This measures serial detector service rate with native device input ready each time.
Preparing the next frame, camera capture and JSON reporting are outside this scope.

At 1280×720, each process ran the 96 new frames in manifest order for two
warm-up and ten measured passes. The three NASA field photographs used eight
warm-up and 60 measured passes, one process per route and input path in both tests.

**1280×720 changing frames**

| Machine | Input | NVIDIA p50 / p95 / p99 (ms) | Project p50 / p95 / p99 (ms) | NVIDIA / project p50 |
| --- | --- | ---: | ---: | ---: |
| Jetson AGX Orin | Device | 3.298 / 6.306 / 7.304 | 0.688 / 1.231 / 2.575 | 4.80× |
| Jetson AGX Orin | Host | 4.449 / 8.036 / 8.613 | 1.183 / 1.688 / 3.116 | 3.76× |
| GeForce RTX 4090 | Device | 1.901 / 2.281 / 2.547 | 0.368 / 2.399 / 2.440 | 5.16× |
| GeForce RTX 4090 | Host | 2.422 / 2.838 / 3.144 | 0.488 / 2.529 / 2.594 | 4.96× |

**799×533 outdoor photographs with tags**

| Machine | Input | NVIDIA p50 / p95 / p99 (ms) | Project p50 / p95 / p99 (ms) | NVIDIA / project p50 |
| --- | --- | ---: | ---: | ---: |
| Jetson AGX Orin | Device | 2.563 / 2.965 / 3.101 | 1.972 / 2.313 / 2.515 | 1.30× |
| Jetson AGX Orin | Host | 3.049 / 3.522 / 4.229 | 2.355 / 2.690 / 2.728 | 1.29× |
| GeForce RTX 4090 | Device | 2.074 / 2.479 / 2.521 | 0.664 / 4.832 / 4.857 | 3.12× |
| GeForce RTX 4090 | Host | 2.277 / 2.495 / 2.599 | 0.642 / 4.862 / 4.896 | 3.55× |

For the same three outdoor photographs, forcing full-resolution sampling
(`--decimate 1`) retained all 15 NVIDIA reference positions, matched 45 of 47
and returned the same two detections outside the reference. On the shared
RTX with device input, the automatic mode had a lower median than NVIDIA
and processed fewer calls per second. Fixed full resolution had the highest
call rate on these three photographs.

| Machine | Input | Fixed full-resolution p50 / p95 / p99 (ms) | NVIDIA / automatic / fixed calls per second |
| --- | --- | ---: | ---: |
| Jetson AGX Orin | Device | 1.991 / 2.591 / 2.815 | 385 / 483 / 504 |
| Jetson AGX Orin | Host | 1.930 / 2.259 / 2.348 | 313 / 425 / 522 |
| GeForce RTX 4090 | Device | 0.368 / 2.441 / 2.481 | 565 / 521 / 953 |
| GeForce RTX 4090 | Host | 0.382 / 2.494 / 2.503 | 502 / 620 / 1,051 |

The RTX GPU had a concurrent external workload (98% utilization in a separate
snapshot). On this machine, NVIDIA had the lower device-input p95 at both
640×480 and 1280×720.

The earlier serial edge-fitting build used the same 96-frame inputs with
fixed decimation 3. Its Orin device p50 was 1.283 ms compared with NVIDIA's 1.155 ms.
The reported detector uses cooperative warp edge fitting, row-run seeding,
non-atomic root compression and robust post-decode corner fitting. Its frozen
`apriltag/gpu/detector.cu` SHA-256 was
`2ee07de9dcd5f686c079a29c56609bc23667d7b90062147d19e83c6dffcfc54b`.
The earlier source was
`df504e1b686228861269c702ae1f41c84919b0aeabe4db7e591e52a4855ecca9`.

### Industrial photographs

All 16 warehouse, manufacturing and equipment-room photographs were used at
their prepared full dimensions (longest edge up to 1920). The RGB PNGs were
verified against the [input manifest](../../tools/gpu_quality/steady-inputs.json).
OpenCV 4.14.0 decoded each with `IMREAD_GRAYSCALE`. The same gray8 PGM bytes
went to both hosts. [Photo sources and preprocessing](../../cpu/experiments/real_background.md)

Each process processed one still image five times for warm-up and 30 times
for measurement. The second of three rounds reversed route order. For each
image, take the median of the three per-process percentiles, then the median
across images. The ratio column is the median of the 16 per-image p50 ratios.
With 30 timed calls in a process its nearest-rank p99 is the sample maximum.

| Machine | Input | NVIDIA p50 / p95 / p99 (ms) | Project p50 / p95 / p99 (ms) | Median p50 ratio |
| --- | --- | ---: | ---: | ---: |
| Jetson AGX Orin | Device | 12.280 / 13.131 / 15.707 | 1.339 / 1.520 / 1.693 | 8.99× |
| Jetson AGX Orin | Host | 14.505 / 15.516 / 22.848 | 1.899 / 1.975 / 2.045 | 7.53× |
| GeForce RTX 4090 | Device | 1.813 / 2.786 / 3.093 | 0.303 / 2.212 / 2.228 | 6.50× |
| GeForce RTX 4090 | Host | 3.295 / 4.432 / 4.489 | 0.551 / 2.468 / 2.482 | 6.03× |

Both routes returned zero markers in every timed call. The project had lower
p50 for every image on both machines and input paths. Device p50 ratios ranged
from 6.66–10.66× on Jetson AGX Orin and 4.74–8.92× on GeForce RTX 4090. The input is
static within each process. The 96-frame sequence above exercises changing
images without restarting the detector.

### Kernel timing

Nsight Systems 2025.3.2 traced a separate RTX 4090 process per route and input
with `--trace=cuda --sample=none --cpuctxsw=none`. Each ran five warm-ups and
one timed detection. In the SQLite export, sum the CUDA kernel durations after
the final input upload completes (`CUPTI_ACTIVITY_KIND_MEMCPY`, host to device,
with byte count equal to the native input size). Six equal input uploads were
verified in each trace. The sums omit CPU work, transfers and synchronization.

| Input | NVIDIA kernels / duration | Project kernels / duration |
| --- | ---: | ---: |
| Clean 640×480, one marker | 17 / 0.0920 ms | 9 / 0.1701 ms |
| Warehouse racks, 1920×1496, zero detections | 17 / 0.8013 ms | 9 / 0.2120 ms |

### Reproduction environment

Release builds used the same source export on both machines. Orin used JetPack
5.1.2 (L4T R35.4.1), CUDA 11.4.315, GCC 9.4, Ubuntu 20.04, SM 87, MAXN and
CPU affinity 8–11. RTX 4090 used driver 575.64.03, CUDA 12.6.68, GCC 13.3,
Ubuntu 24.04, SM 89 and CPU affinity 2–5. The pinned vendor snapshot ships
JetPack 5.1, JetPack 6.1 and x86-64 CUDA 12.6 archives. It supplies no DGX Spark
archive, so this SDK comparison uses Orin and RTX.
[Published platform archives](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros/tree/94ada8a17231ec3d333664fc1320faa5e4bd1b7b/cuapriltags_vendor)

Clocks and pre-existing system loads were retained. The frozen source manifest SHA-256 was
`91968641f3c47d59fdce82c5659096178001c835a349557d7721866b83ea6cd1`.
The generated binary hashes, timings, observations and profiler output remain
local under `build/april-nvidia/`. The common 96-frame manifest SHA-256 was
`a3b05a54dfbda10f7c2781e6a2971975589e60706f8f929cdc109807a8a4c9e2`.

### Build and run

In an environment with the SDK available, build the external baseline with the
matching CUDA toolkit. The tested vendor snapshot is
`94ada8a17231ec3d333664fc1320faa5e4bd1b7b`.

```bash
cmake -S apriltag/benchmark -B build/nvidia -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCUAPRILTAGS_INCLUDE_DIR="$SDK/cuapriltags" \
  -DCUAPRILTAGS_LIBRARY="$SDK/lib/libcuapriltags.a"
cmake --build build/nvidia --target apriltag_nvidia_benchmark
python3 apriltag/benchmark/run_sequence.py build/nvidia/apriltag_nvidia_benchmark \
  build/corpus-36h11/manifest.json build/comparison/nvidia-device \
  --backend nvidia --scope device --warmup 2 --samples 10
```

Set `SDK` to the vendor installation prefix or supply the two paths directly.
The SHA-256 hashes of the tested upstream Git LFS objects were
`6b5d37afc13c72913bd96d576ae876ba2d4fe16bb1d0807fe4087f2995e69956`
for JetPack 5.1 and
`a7f32f58782de61dd23e507fe6ce5a907379498deec505669fe339ecd4facba7`
for x86 CUDA 12.6.

Enable `ARUCO3CUDA_BUILD_APRILTAG_GPU=ON` and `ARUCO3CUDA_BUILD_APRILTAG_BENCHMARK=ON`
in the top-level CMake build. Run `apriltag_project_benchmark` with the same script using `--backend project`
and `--decimate 0` for automatic sampling. Repeat with `--scope host`.
Use fresh processes and alternate route order between runs. The script verifies
the SHA-256 of each input and writes raw wall times and canonical observations.
It checks exact replay after sorting detections. Output directories must be new.

## Goals

Check IDs and corner order against annotated scenes before making a speed claim.
The NVIDIA runner maps SDK corners 2, 3, 0, 1 to canonical top left, top right,
bottom right, bottom left **in tag coordinates**. The mapping is fixed across
rotations. The project runner preserves the detector's canonical order.

## Open questions

Which scenes and deployment resolutions move the speed crossover? What is the
difference between synchronized call latency and application throughput under
real camera load?
