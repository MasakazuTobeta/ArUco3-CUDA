# Ordered-frame CPU ArUco trial

## Purpose

Measure whether motion and prior marker positions can reduce the work of
contour extraction in a continuous ArUco sequence.

## Scope

The trial consumes one ordered sequence of Gray8 host frames. It compares
the project CPU detector's full-frame calls to calls made through a
temporal wrapper using the same dictionary and detector settings. Decoded
marker IDs and ordered corners refer to the current input frame.

## Current state

`CpuVideoDetector` owns a full-image detector, a crop detector and two sparse
Gray8 sample buffers. Each call samples the new frame with nearest-neighbor
resize. It compares those samples to the previous successful frame and builds
rectangles around changed tiles and the prior detections. It pads and merges
overlapping rectangles, then applies the current CPU detector to the original
pixels within each rectangle. Corner offsets restore full-image coordinates.
History supplies search positions. All published detections come from decoding
the current frame.

First frames, resets and size changes request full-frame search. A periodic
refresh also requests it. Large or fragmented proposals, including abrupt
changes across the frame, fall back to the full frame when they exceed the
area or count limits. A failed call clears the history. With the defaults the
refresh interval is 30 frames, tiles target 32 px, the sample change cutoff
is 16 gray levels, padding is 40 px, and the crop budget is 16 regions or
35% of the frame. The benchmark can choose a shorter refresh interval.

If the crop results contain fewer instances of any previously detected ID,
the wrapper reruns detection on the full frame and uses that result. A tag
leaving the frame can trigger this check once. The next frame uses the newly
observed IDs and corners, so the departed tag is removed from history.

Changed pixels between sample locations or below the cutoff can escape local
search until a periodic refresh. Crop dimensions affect ArUco3 search scaling
and corner refinement. Detection counts and corner drift are therefore
measured against full-image detection and the rendered marker labels.

### Measurement

The six-host measurement used a standalone Release build with OpenCV
4.14.0 and `-O3 -DNDEBUG`. Each run used one OpenCV thread and one pinned
logical CPU. These hosts retained their existing frequency policies and
services. Jetson power mode reported MAXN on both boards. The Orin query
could not read the EMC cap.

| Host | Pinned CPU | GCC | Governor |
| --- | ---: | ---: | --- |
| DGX Spark | 6 | 13.3 | performance |
| Jetson AGX Orin | 6 | 9.4 | schedutil |
| Jetson AGX Thor | 6 | 13.3 | schedutil |
| Xeon w7-3565X | 2 | 13.3 | powersave |
| Core i9-12900KS | 2 | 13.3 | powersave |
| Core Ultra 7 265 | 2 | 13.3 | powersave |

The benchmark uses default ArUco3 settings, `DICT_ARUCO_MIP_36h12` and a
12-frame video refresh interval. Four 24-frame sequences contain a stationary
tag, motion over fixed texture, entry and exit on that texture, and an abrupt
camera cut. The marker side is 160 px at 720p and 480 px at 4K. The 4K
fixture also scales distances, texture cells and motion by three. All six
hosts produced matching SHA-256 values for the ordered decoded pixels.

There are three independent processes per resolution. In each process, each
route warms up on one full sequence and runs three timed traversals. Route
order alternates and video state resets before each timed traversal. Timing
includes the initial full search, region selection, fallbacks and returned
host detections. Rendering, initialization and scoring occur outside timing.
FPS uses the complete call loop. The table reports the median of the three
process medians for both FPS and per-call percentiles. Full-call counts and
ordered-corner differences use the maximum across traversals. Times compare
full-frame → video in ms.

| Host | Size | Sequence | FPS | p50 / p95 / p99 (ms) | Full calls / 24 | Max corner delta vs. full (px) |
| --- | --- | --- | --- | --- | ---: | ---: |
| DGX Spark | 720p | steady_clean | 5260 → 5793 (1.10×) | 0.189 → 0.170 / 0.197 → 0.203 / 0.202 → 0.214 | 2 | 1.059 |
| DGX Spark | 720p | motion_texture | 691 → 2497 (3.61×) | 1.440 → 0.307 / 1.500 → 1.494 / 1.539 → 1.561 | 2 | 0.035 |
| DGX Spark | 720p | entry_exit_texture | 670 → 1774 (2.65×) | 1.492 → 0.312 / 1.521 → 1.596 / 1.547 → 2.209 | 3 | 1.414 |
| DGX Spark | 720p | cut_texture | 687 → 2295 (3.34×) | 1.455 → 0.288 / 1.513 → 1.532 / 1.536 → 1.552 | 3 | 1.030 |
| DGX Spark | 4K | steady_clean | 1569 → 1708 (1.09×) | 0.632 → 0.562 / 0.647 → 0.785 / 0.697 → 0.808 | 2 | 0.064 |
| DGX Spark | 4K | motion_texture | 401 → 1068 (2.66×) | 2.332 → 0.765 / 3.057 → 2.574 / 3.123 → 3.155 | 2 | 0.062 |
| DGX Spark | 4K | entry_exit_texture | 365 → 775 (2.12×) | 2.582 → 0.804 / 3.097 → 3.208 / 3.151 → 3.950 | 3 | 0.063 |
| DGX Spark | 4K | cut_texture | 415 → 1001 (2.41×) | 2.326 → 0.731 / 3.112 → 2.975 / 3.162 → 3.182 | 3 | 3.542 |
| Jetson AGX Orin | 720p | steady_clean | 1445 → 1724 (1.19×) | 0.684 → 0.552 / 0.716 → 0.824 / 0.816 → 0.866 | 2 | 1.059 |
| Jetson AGX Orin | 720p | motion_texture | 288 → 934 (3.25×) | 3.459 → 0.839 / 3.581 → 3.635 / 3.598 → 3.650 | 2 | 0.035 |
| Jetson AGX Orin | 720p | entry_exit_texture | 274 → 662 (2.42×) | 3.694 → 0.899 / 3.751 → 3.873 / 3.775 → 5.379 | 3 | 1.414 |
| Jetson AGX Orin | 720p | cut_texture | 286 → 879 (3.08×) | 3.461 → 0.816 / 3.640 → 3.581 / 3.649 → 3.618 | 3 | 1.030 |
| Jetson AGX Orin | 4K | steady_clean | 354 → 486 (1.37×) | 2.810 → 1.926 / 2.904 → 3.407 / 3.097 → 3.451 | 2 | 0.064 |
| Jetson AGX Orin | 4K | motion_texture | 147 → 341 (2.32×) | 6.051 → 2.475 / 8.883 → 6.619 / 9.008 → 9.148 | 2 | 0.062 |
| Jetson AGX Orin | 4K | entry_exit_texture | 130 → 247 (1.89×) | 7.633 → 2.601 / 8.752 → 9.158 / 8.770 → 11.190 | 3 | 0.063 |
| Jetson AGX Orin | 4K | cut_texture | 155 → 325 (2.10×) | 6.032 → 2.400 / 8.726 → 7.897 / 8.742 → 9.103 | 3 | 3.542 |
| Jetson AGX Thor | 720p | steady_clean | 2301 → 2996 (1.30×) | 0.432 → 0.321 / 0.444 → 0.453 / 0.452 → 0.496 | 2 | 1.059 |
| Jetson AGX Thor | 720p | motion_texture | 412 → 1437 (3.49×) | 2.421 → 0.542 / 2.495 → 2.470 / 2.525 → 2.533 | 2 | 0.035 |
| Jetson AGX Thor | 720p | entry_exit_texture | 392 → 1013 (2.58×) | 2.571 → 0.556 / 2.596 → 2.695 / 2.611 → 3.758 | 3 | 1.414 |
| Jetson AGX Thor | 720p | cut_texture | 411 → 1342 (3.26×) | 2.428 → 0.509 / 2.503 → 2.491 / 2.542 → 2.551 | 3 | 1.030 |
| Jetson AGX Thor | 4K | steady_clean | 585 → 875 (1.50×) | 1.707 → 1.063 / 1.734 → 1.944 / 1.837 → 2.011 | 2 | 0.064 |
| Jetson AGX Thor | 4K | motion_texture | 218 → 570 (2.61×) | 4.095 → 1.424 / 6.103 → 4.478 / 6.197 → 6.358 | 2 | 0.062 |
| Jetson AGX Thor | 4K | entry_exit_texture | 189 → 404 (2.14×) | 4.925 → 1.491 / 6.250 → 6.497 / 6.375 → 7.303 | 3 | 0.063 |
| Jetson AGX Thor | 4K | cut_texture | 228 → 538 (2.36×) | 4.094 → 1.378 / 6.236 → 5.202 / 6.302 → 6.374 | 3 | 3.542 |
| Xeon w7-3565X | 720p | steady_clean | 2264 → 3429 (1.51×) | 0.440 → 0.275 / 0.447 → 0.459 / 0.452 → 0.485 | 2 | 1.059 |
| Xeon w7-3565X | 720p | motion_texture | 501 → 1824 (3.64×) | 1.988 → 0.419 / 2.031 → 2.012 / 2.041 → 2.024 | 2 | 0.035 |
| Xeon w7-3565X | 720p | entry_exit_texture | 479 → 1286 (2.68×) | 2.109 → 0.432 / 2.175 → 2.148 / 2.176 → 2.996 | 3 | 1.414 |
| Xeon w7-3565X | 720p | cut_texture | 497 → 1704 (3.43×) | 1.987 → 0.397 / 2.154 → 1.990 / 2.202 → 2.027 | 3 | 1.030 |
| Xeon w7-3565X | 4K | steady_clean | 989 → 1162 (1.18×) | 1.015 → 0.838 / 1.027 → 1.061 / 1.054 → 1.109 | 2 | 0.064 |
| Xeon w7-3565X | 4K | motion_texture | 297 → 736 (2.48×) | 3.107 → 1.122 / 4.511 → 3.250 / 4.630 → 4.532 | 2 | 0.062 |
| Xeon w7-3565X | 4K | entry_exit_texture | 256 → 515 (2.01×) | 3.435 → 1.216 / 4.632 → 4.728 / 4.661 → 5.354 | 3 | 0.063 |
| Xeon w7-3565X | 4K | cut_texture | 311 → 702 (2.26×) | 3.070 → 1.093 / 4.373 → 3.928 / 4.505 → 4.503 | 3 | 3.542 |
| Core i9-12900KS | 720p | steady_clean | 2712 → 4185 (1.54×) | 0.369 → 0.228 / 0.372 → 0.370 / 0.373 → 0.378 | 2 | 1.059 |
| Core i9-12900KS | 720p | motion_texture | 549 → 1996 (3.64×) | 1.818 → 0.385 / 1.866 → 1.823 / 1.885 → 1.886 | 2 | 0.035 |
| Core i9-12900KS | 720p | entry_exit_texture | 519 → 1412 (2.72×) | 1.945 → 0.391 / 1.989 → 2.009 / 2.004 → 2.758 | 3 | 1.414 |
| Core i9-12900KS | 720p | cut_texture | 552 → 1872 (3.39×) | 1.811 → 0.361 / 1.839 → 1.818 / 1.849 → 1.827 | 3 | 1.030 |
| Core i9-12900KS | 4K | steady_clean | 1153 → 1286 (1.12×) | 0.863 → 0.755 / 0.897 → 0.992 / 0.924 → 0.999 | 2 | 0.064 |
| Core i9-12900KS | 4K | motion_texture | 328 → 808 (2.46×) | 2.839 → 1.023 / 4.062 → 2.974 / 4.082 → 4.185 | 2 | 0.062 |
| Core i9-12900KS | 4K | entry_exit_texture | 288 → 583 (2.03×) | 3.062 → 1.068 / 4.091 → 4.195 / 4.101 → 4.852 | 3 | 0.063 |
| Core i9-12900KS | 4K | cut_texture | 341 → 777 (2.28×) | 2.826 → 0.996 / 4.046 → 3.638 / 4.088 → 4.147 | 3 | 3.542 |
| Core Ultra 7 265 | 720p | steady_clean | 2973 → 4608 (1.55×) | 0.336 → 0.204 / 0.341 → 0.343 / 0.343 → 0.360 | 2 | 1.059 |
| Core Ultra 7 265 | 720p | motion_texture | 575 → 2134 (3.71×) | 1.737 → 0.355 / 1.751 → 1.765 / 1.758 → 1.767 | 2 | 0.035 |
| Core Ultra 7 265 | 720p | entry_exit_texture | 549 → 1509 (2.75×) | 1.846 → 0.369 / 1.874 → 1.869 / 1.879 → 2.586 | 3 | 1.414 |
| Core Ultra 7 265 | 720p | cut_texture | 578 → 1985 (3.43×) | 1.732 → 0.336 / 1.751 → 1.743 / 1.772 → 1.767 | 3 | 1.030 |
| Core Ultra 7 265 | 4K | steady_clean | 1214 → 1406 (1.16×) | 0.820 → 0.685 / 0.849 → 0.949 / 0.881 → 0.958 | 2 | 0.064 |
| Core Ultra 7 265 | 4K | motion_texture | 345 → 878 (2.55×) | 2.700 → 0.937 / 3.792 → 2.845 / 3.833 → 3.899 | 2 | 0.062 |
| Core Ultra 7 265 | 4K | entry_exit_texture | 305 → 634 (2.08×) | 2.915 → 0.986 / 3.846 → 3.879 / 4.137 → 4.515 | 3 | 0.063 |
| Core Ultra 7 265 | 4K | cut_texture | 358 → 830 (2.32×) | 2.694 → 0.916 / 3.760 → 3.427 / 3.796 → 3.887 | 3 | 3.542 |

Every labelled tag was detected with the expected ID and corner ordering.
All rendered entries were detected on their first visible frame, with zero
extra detections across hosts. The maximum ordered-corner difference
against full search was 3.542 px on the 4K cut. The full retry on missing
prior IDs recovered a tag that the initial prototype missed after the cut.
Full searches and retries remain visible in p95 and p99 on some sequences.

A separate regression fixture places an entering tag below the sample
change threshold. With a four-frame refresh interval, the tag is first
detected three frames after entry. The zero delay in the timing fixtures
applies to those inputs. The regression suite also exercises strided input,
invalid inputs, cut, resize, reset and marker disappearance.

An independent 720p probe found a 64 px marker at the crop's ArUco3 scale.
The same pixels produced no detection during the next full refresh or the
following unchanged regional frame. Keep minimum marker size and detection
stability in the evaluation of any deployment configuration.

### Reproduction

Build the standalone project with the pinned OpenCV package. The sequence
runner renders input frames itself and records their identities. Run each
command three times in fresh processes and use the recorded core for the host.

```sh
cmake -S cpu/standalone -B build/cpu-video -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpu-video --target aruco3cuda_cpu_video_tracking_trial
taskset -c "$core" build/cpu-video/cpu/aruco3cuda_cpu_video_tracking_trial --resolution 720 --repeats 3
taskset -c "$core" build/cpu-video/cpu/aruco3cuda_cpu_video_tracking_trial --resolution 2160 --repeats 3
```

Raw timing and independent QA logs are locally retained under
`build/aruco-video-formal/`, `build/aruco-video-trial/` and
`build/aruco-video-qa/`. The formal build uses base commit `427dd2c` and
the video detector, benchmark and CPU CMake changes. The measurement
directories are local outputs.

## Goals

Reduce detector-call latency on scenes where the background changes only
locally. Evaluate ID recall, false positives, corner coordinates and
first-appearance delay with throughput and latency percentiles.

## Open questions

What region and refresh settings work on camera footage with compression,
exposure changes and motion blur? How does the detector behave near moving
crop boundaries, and what latency budget is available for periodic
full-frame calls?
