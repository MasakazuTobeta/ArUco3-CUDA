# AprilTag CPU benchmark

## Conditions

Measured on 2026-11-26 with tag36h11 and one CPU core per process. The project used `--search video 2` with the default video configuration and OpenCV 4.14.0. Each detector processed the same predecoded Gray8 frames in presentation order. Three processes per timed route each ran two warm-up and five timed traversals. Launch order rotated across the project builds and NAPA. Timing covers the synchronous detection call and returned results.

| Host | Core and power settings | GCC 13.3 Release target |
| --- | --- | --- |
| DGX Spark | Core 6, performance governor | `-mcpu=native` |
| Jetson Orin NX | Core 6, MAXN, schedutil | `-march=armv8.2-a -mtune=cortex-a78ae` |
| Core i9-12900KS | Core 12 | `-march=x86-64-v3` |

NAPA 0.6.1 used a 22 px minimum side and capacities of 64, 256 or 512 as required by the workload. Accuracy used fresh NAPA results and hash-matched official OpenCV 4.14.0 and NVIDIA cuAprilTags observations. The checks cover retention of correctly identified and oriented tags, corner RMSE on common detections, and errors and first-appearance delays where labels support them. [Sequence runner](apriltag/benchmark/small_sequence.cpp) · [Quality scorer](apriltag/benchmark/compare_video_quality.py)

## Results

| Workload group | Workloads | Faster than NAPA | Passed all quality checks | Passed both |
| --- | ---: | ---: | ---: | ---: |
| DGX Spark, main matrix | 92 | 52 | 84 | 47 |
| DGX Spark, general video subset | 46 | 39 | 41 | 35 |
| DGX Spark, additional 4K optical sequences | 4 | 4 | 4 | 4 |
| Jetson Orin NX, representative sequences | 13 | 12 | 9 | 8 |

On DGX Spark, 46 of the 47 workloads that passed both checks had separated FPS ranges. All four additional 4K optical sequences retained 60/60 labelled positions each. The x86 replay covered four workloads and the project was faster than NAPA on all four.

| Host and input | Project (FPS) | NAPA (FPS) | Project p50 / p95 / p99 (ms) |
| --- | ---: | ---: | ---: |
| DGX Spark, 720p motion | 6,399.1 | 2,857.7 | 0.131 / 0.386 / 0.482 |
| DGX Spark, mixed 4K video | 1,104.8 | 492.9 | 0.787 / 1.693 / 2.089 |
| DGX Spark, FRC holdout | 2,623.7 | 1,669.1 | 0.367 / 0.541 / 0.595 |
| Core i9, mixed 720p | 4,288.6 | 4,163.2 | 0.165 / 0.509 / 0.611 |
| Core i9, ICRA forward | 600.6 | 269.4 | 1.384 / 2.794 / 2.955 |
| Core i9, 4K tag entry | 1,180.8 | 600.8 | 0.718 / 2.263 / 2.307 |
| Core i9, 4K marker-free industrial | 618.7 | 453.5 | 1.677 / 2.324 / 2.336 |

On severe 4K wide-angle footage, Orin NX found 120/120 labelled positions at 29.3 FPS and NAPA found 98/120 at 128.4 FPS. On the partially labelled FRC holdout, Spark found 34/51 positions and NAPA found 48/51. Their corner RMSE on common detections was 0.943 and 0.833 px, respectively.

| Marker-free 3840×2160 input on DGX Spark | Project (FPS) | NAPA (FPS) | Official OpenCV (FPS) |
| --- | ---: | ---: | ---: |
| Five landscapes | 545.44 | 1,158.40 | 2.63 |
| Five industrial frames | 481.50 | 316.74 | 2.15 |

All three detectors returned zero tags on these controls. OpenCV was timed separately on the same core using `CORNER_REFINE_APRILTAG` at native resolution, with quad decimation 1 and one worker. [OpenCV runner](apriltag/benchmark/opencv_sequence.cpp)

## Single images vs. ordered video

The 2026-11-26 remeasurement used one pinned CPU core per host and one
OpenCV 4.14.0 thread. Project detectors are from `2227cff`. The experimental
single-image `Detector` uses its default native search without a coarse
override. The other two routes use the same default `VideoDetector`,
resetting before each frame or just before each traversal. The three project
routes each use three separate processes with rotating run order. Every
process uses two warm-up traversals and five timed traversals. The synchronous
call is timed on predecoded Gray8 images. Decoding, detector construction,
explicit resets and output logging are outside timing. This is detector-call
throughput, not pipeline throughput or cold process startup.

The OpenCV reference uses `cv::aruco::ArucoDetector` with `DICT_APRILTAG_36h11`,
`CORNER_REFINE_APRILTAG`, native-resolution quad extraction (`aprilTagQuadDecimate = 1`),
and ArUco3 disabled. The OpenCV runs are a separate follow-up batch using the
same pinned cores, hashed inputs and OpenCV builds. OpenCV uses three
processes for 720p and the first complete process on each host for 4K.
Each 4K OpenCV result includes all 160 timed calls after two warm-up
traversals. No between-process spread is reported for 4K OpenCV.

The 4K corpus includes a marker-free checkerboard. On Spark, OpenCV took
40.8 s per call at the median for that frame, and the frame accounted for
61% of OpenCV's total timed duration. The frame is included in all routes.
Overall FPS includes the full tail of the latency distribution, so it can
be much lower than the reciprocal of the median frame time.

| CPU host | Pinned CPU and policy | GCC Release target |
| --- | --- | --- |
| DGX Spark | 6, performance | 13.3, `-mcpu=native` |
| Jetson AGX Orin | 6, reported MAXN, schedutil | 9.4, `-march=armv8.2-a` |
| Jetson AGX Thor | 6, MAXN, schedutil | 13.3, `-march=armv8.2-a` |
| Xeon w7-3565X | 2, intel_pstate powersave, balance_performance | 13.3, `-march=x86-64-v3` |
| Core i9-12900KS | 2, intel_pstate powersave, balance_performance | 13.3, `-march=x86-64-v3` |
| Core Ultra 7 265 | 2, intel_pstate powersave, balance_performance | 13.3, `-march=x86-64-v3` |

Clocks were not locked and other services were left running. On Thor,
`-mcpu=native` failed to compile the SVE helper with this GCC toolchain.
The table uses the explicit ARMv8.2-A build without changing the source.
On Orin the power-mode query reported MAXN but could not read the EMC cap.

| CPU host | Route | 720p FPS (process range) | 720p p50 / p95 / p99 (ms) | 4K FPS (process range) | 4K p50 / p95 / p99 (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| DGX Spark | OpenCV | 24.28 (24.05–24.34) | 42.185 / 47.223 / 48.378 | 0.48 | 35.511 / 2379.401 / 41574.237 |
| DGX Spark | Single image | 904 (900–905) | 0.842 / 2.075 / 2.106 | 518 (516–519) | 1.197 / 3.063 / 25.488 |
| DGX Spark | Video, no history | 4,917 (4,894–4,929) | 0.165 / 0.375 / 0.400 | 1,146 (1,143–1,150) | 0.769 / 1.508 / 1.827 |
| DGX Spark | Video, ordered | 6,680 (6,673–6,708) | 0.123 / 0.371 / 0.458 | 1,109 (1,099–1,110) | 0.781 / 1.948 / 2.044 |
| Jetson AGX Orin | OpenCV | 9.58 (9.53–9.59) | 107.260 / 118.926 / 121.969 | 0.14 | 67.484 / 5830.870 / 171355.428 |
| Jetson AGX Orin | Single image | 369 (369–369) | 2.094 / 4.940 / 4.968 | 169 (169–169) | 4.293 / 8.991 / 66.837 |
| Jetson AGX Orin | Video, no history | 1,460 (1,456–1,465) | 0.606 / 1.193 / 1.220 | 349 (348–349) | 2.660 / 4.873 / 5.384 |
| Jetson AGX Orin | Video, ordered | 1,850 (1,843–1,852) | 0.448 / 1.276 / 1.548 | 344 (343–345) | 2.666 / 5.183 / 5.387 |
| Jetson AGX Thor | OpenCV | 14.37 (14.37–14.43) | 71.947 / 79.580 / 80.443 | 0.23 | 51.782 / 3983.919 / 95297.327 |
| Jetson AGX Thor | Single image | 553 (551–554) | 1.384 / 3.355 / 3.374 | 315 (312–315) | 1.887 / 4.706 / 43.276 |
| Jetson AGX Thor | Video, no history | 2,560 (2,560–2,565) | 0.331 / 0.675 / 0.710 | 659 (659–661) | 1.334 / 2.529 / 2.860 |
| Jetson AGX Thor | Video, ordered | 3,674 (3,662–3,676) | 0.220 / 0.670 / 0.857 | 648 (642–652) | 1.358 / 2.846 / 3.189 |
| Xeon w7-3565X | OpenCV | 21.13 (21.07–21.14) | 48.807 / 54.829 / 56.718 | 0.32 | 52.589 / 2396.638 / 73207.695 |
| Xeon w7-3565X | Single image | 759 (748–760) | 1.018 / 2.430 / 2.643 | 389 (388–389) | 1.995 / 3.909 / 28.336 |
| Xeon w7-3565X | Video, no history | 3,757 (3,018–3,775) | 0.225 / 0.497 / 0.511 | 745 (732–747) | 1.101 / 2.233 / 2.711 |
| Xeon w7-3565X | Video, ordered | 4,820 (4,803–4,834) | 0.153 / 0.562 / 0.707 | 717 (715–718) | 1.091 / 2.866 / 3.057 |
| Core i9-12900KS | OpenCV | 22.99 (22.98–23.00) | 44.914 / 50.385 / 52.647 | 0.44 | 37.730 / 2144.868 / 49665.666 |
| Core i9-12900KS | Single image | 844 (839–845) | 0.906 / 2.213 / 2.227 | 507 (506–509) | 1.371 / 3.244 / 24.095 |
| Core i9-12900KS | Video, no history | 3,894 (3,892–3,896) | 0.220 / 0.465 / 0.474 | 1,080 (1,079–1,080) | 0.689 / 1.719 / 1.967 |
| Core i9-12900KS | Video, ordered | 5,237 (5,193–5,250) | 0.145 / 0.493 / 0.630 | 1,033 (1,027–1,035) | 0.682 / 2.217 / 2.319 |
| Core Ultra 7 265 | OpenCV | 24.71 (24.65–24.75) | 41.746 / 46.755 / 47.547 | 0.52 | 140.663 / 2112.888 / 37312.430 |
| Core Ultra 7 265 | Single image | 908 (896–909) | 0.846 / 2.052 / 2.102 | 543 (539–543) | 1.360 / 2.982 / 21.180 |
| Core Ultra 7 265 | Video, no history | 4,028 (4,016–4,036) | 0.218 / 0.440 / 0.447 | 1,109 (1,106–1,110) | 0.667 / 1.681 / 1.867 |
| Core Ultra 7 265 | Video, ordered | 5,531 (5,526–5,535) | 0.140 / 0.456 / 0.574 | 1,054 (1,052–1,056) | 0.668 / 2.224 / 2.363 |

FPS is the process median of frames divided by summed call time. Latency
entries are medians of process-level nearest-rank percentiles. Each route
returned the same number of detections on all six hosts. Geometric matching
against the labels found the following correct detections per traversal.

| Input | Labelled tags | OpenCV | Single image | Video, no history | Video, ordered |
| --- | ---: | ---: | ---: | ---: | ---: |
| 720p motion | 79 fully visible | 78 | 61 | 79 | 79, plus one labelled partial tag |
| 4K changing renders | 48 | 26 | 30 | 46 | 46 |

No false positives, wrong IDs or wrong rotations were observed on these
inputs. Within each host, frame-zero IDs and corners were identical between
the two video modes. The accuracy results apply to these two workloads.
[Label matching](apriltag/benchmark/compare_video_quality.py)

The 720p input concatenates four 18-frame clips with tags composited over industrial photos, without resets between clips in ordered mode. The 4K input consists of changing renders, not a continuous camera motion sequence. This comparison does not measure the before/after change to the separate four-family CPU detector.

Build the [sequence runner](apriltag/small_cpu/README.md#ordered-video-search) and use the same ordered image paths for each command. Set `core` to the host's recorded CPU number. Repeat each project command in three fresh processes, rotating the three routes. Measure OpenCV separately in three processes for 720p and one for 4K.

```sh
taskset -c "$core" aruco3cuda_apriltag_small_sequence 0 2 5 frame-001.pgm frame-002.pgm ...
taskset -c "$core" aruco3cuda_apriltag_small_sequence --search video --reset-each-frame 2 2 5 frame-001.pgm frame-002.pgm ...
taskset -c "$core" aruco3cuda_apriltag_small_sequence --search video 2 2 5 frame-001.pgm frame-002.pgm ...
taskset -c "$core" aruco3cuda_apriltag_opencv_sequence 1 2 5 frame-001.pgm frame-002.pgm ...
```

Use `motion-720` (72 frames, manifest SHA-256 `da93b522c30b59b89cc2d5dc86aca88a81135904339d5e71158a6b20c1dc2dfe`) and `mixed-4k` (32 frames, `49446e87c2c8b628e93f56cdc36a3f199e2f58d2273392577d4bd4925fd65744`) from the preceding video study. The inputs and raw remeasurement output are locally retained, not included in a fresh checkout. Raw logs are under `build/apriltag-frame-vs-video/`.
