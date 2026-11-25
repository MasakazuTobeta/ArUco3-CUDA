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
