# Performance refresh · 2026-11-19–20 JST

## Purpose

Repeat steady-state ArUco and AprilTag measurements after GPU utilization
dropped on the shared benchmark hosts. Extend the comparison to annotated
1080p and 4K changing frames and native industrial photographs up to 8256×6192.

## Scope

ArUco uses DICT_ARUCO_MIP_36h12. Compare unmodified OpenCV 4.14.0, the optimized
CPU detector, Hybrid and the ordered-contour CUDA configuration. The prior
component extractor is a historical control with different detection quality.
AprilTag timings use tag36h11. Compare AprilTag CPU with OpenCV
CORNER_REFINE_APRILTAG, and the CUDA detector with the pinned NVIDIA cuAprilTags
archives. Use adaptive CUDA sampling unless the field run specifies spacing 1.

The detector revision is 46abcad44c1e48dd54f1e3f80feb4d6fb2150eab, based on
main at 9ba4b34710fa6fc20db1e0deb238d958cfd77dfb. The benchmark changes
on top of that revision add 2160p to the AprilTag CPU sequence, a changing-frame
ArUco CUDA adapter and explicit ordering of pageable uploads in the ArUco
timing and audit tools. The OpenCV revision is
0654a42e19215ef25b1d367d822f3c630447e7c7.

## Current state

### Workloads

| Study | Prepared input | Warm-up and measurement |
| --- | --- | --- |
| ArUco fixed images | Pinned seven synthetic scenes and 16 industrial photos; ArUco3 off uses seven synthetic scenes and the first three photos | 20 + 200 calls/image/process, three processes |
| ArUco historical input sweep | 28 synthetic scenes across 640×480, 720p, 1080p and 4K; ArUco3 on | 30 + 200 calls/image/process, three processes |
| ArUco changing frames | 32 annotated images per resolution, 1080p and 4K; ArUco3 on | CUDA uses two warm-up and ten measured passes, three processes. CPU compares intra-frame and four-worker layouts on the same four-core budget. |
| ArUco memory scopes | 720p and 4K positive controls | 30 + 200 calls/image/process, three processes |
| AprilTag CUDA changing frames | 96 annotated frames at 640×480, 720p and 1080p; 32 at 4K | Two warm-up and ten measured passes, with independent process counts recorded per setup |
| AprilTag CUDA photographs | Three outdoor field images with reference tags and 16 prepared industrial photographs | Eight + 60 passes on the field set. Five + 30 calls per prepared industrial image. |
| AprilTag CPU changing frames | 16 frames per sequence at 720p, 1080p and 4K; 0/1/4/12 tags | 16 + 192 calls/process, three processes, one and four threads |
| Native industrial spot checks | 3456×4608 and 8256×6192 from the same photo collection | ArUco 10 + 50 calls and AprilTag CUDA 3 + 20 calls per process. Additional native captures on selected hosts. |

The raw files preserve p50, p95 and p99, detection counts, process variation,
input and executable hashes, OpenCV provenance, host topology and GPU
telemetry. Phase order reverses between consecutive paired trials where
supported. The [input guide](inputs.md) records the prepared bytes and the
32-frame selection procedure.

### Timing intervals

CPU starts with decoded host grayscale and ends with host marker IDs and
ordered corners. ArUco CUDA-Resident starts with synchronized device pixels
and ends after detector stream completion. The resulting detections stay on the
device within the timed interval. ArUco CUDA-E2E includes host upload and
result download. AprilTag CUDA device-input timing includes host results.
AprilTag CUDA host-input timing also includes upload from gray8 and, for
cuAprilTags, preparing the native RGB8 input.

Changing-frame CUDA runs use one persistent detector and one stable device
input address. Report latency and serial detector service rate separately from
the CPU worker-pool saturated throughput. Primary measurements exclude
input decoding, initialization, capture and output serialization. Kernel-only
diagnostics use separate Nsight Systems runs.

Pageable uploads can return while DMA is active. The ArUco sequence adapter
enqueues the copy and detection on the same stream. The fixed-image timing
harness and changing-frame audit explicitly synchronize the pageable copy on
the default stream. The previous audit produced missed detections on changing
frames in x86 runs without that synchronization. The repeated timing and
quality results below use the synchronized tools.
[CUDA Runtime memory-copy synchronization](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html)

### Hosts

| Host | GPU | CPU affinity for the main runs |
| --- | --- | --- |
| 192.168.20.23 | DGX Spark GB10 | 5 or 5–8 |
| 192.168.20.30 | Jetson AGX Orin | 8 or 8–11 |
| 192.168.20.20 | Jetson AGX Thor | 4 or 4–7 |
| 192.168.20.41 | RTX PRO 6000 Blackwell Max-Q | 2 or 2–5 |
| 192.168.20.44 | GeForce RTX 4090 | 2 or 2,4,6,8 for the synchronized phase |
| 192.168.20.46 | GeForce RTX 5070 Ti | 2 or 2–5 |

The six reachable CUDA hosts were sampled at zero GPU utilization before
measurements. Other services remained active. An unloaded GPU sample does
not reserve the GPU, remove idle CUDA contexts, or isolate the CPU. In
particular, Spark retained a substantial unrelated CPU workload. The
Intel N97 host at 192.168.10.26 was unreachable on TCP port 22 during this study.

The RTX 5070 Ti global NVML was 610.57 and the loaded kernel driver was
610.43.02. Its measurements use the same-version official NVIDIA userspace
libraries from the extracted 610.43.02 release in a private container mount
and explicit NVIDIA device nodes. The benchmark process proved CUDA device
initialization and detected the expected ID on the NVIDIA 1080p fixture.

### ArUco changing frames

The 32-image sequence uses ArUco3 with a 32 px canonical side and a 0.05
minimum marker-length ratio. The CPU worker-pool results use four
independent single-thread detectors on the same four-core budget as the
intra-frame parallel comparison. CUDA p50/p95/p99 cover one synchronized
call at a time.

| Host | 4K CPU pool, OpenCV → optimized (frames/s) | 4K CUDA-Resident p50 (ms) | 1080p CUDA-E2E p50 (ms) | 4K CUDA-E2E p50 / p95 / p99 (ms) |
| --- | ---: | ---: | ---: | ---: |
| DGX Spark | 1,203 → 2,634 | 1.008 | 1.143 | 1.389 / 2.122 / 2.347 |
| Jetson AGX Orin | 412 → 843 | 2.365 | 2.640 | 4.226 / 6.190 / 6.433 |
| Jetson AGX Thor | 618 → 1,378 | 1.633 | 1.755 | 2.155 / 3.375 / 4.036 |
| RTX PRO 6000 / Xeon | 919 → 1,624 | 0.849 | 0.999 | 1.707 / 2.271 / 2.306 |
| RTX 4090 / Core i9 | 1,124 → 2,518 | 0.588 | 0.815 | 1.449 / 1.850 / 1.871 |
| RTX 5070 Ti / Core Ultra 7 | 1,252 → 2,588 | 0.741 | 0.848 | 1.306 / 1.786 / 1.808 |

CPU throughput measures a saturated batch from first service start to
last completion. The reported gains range from 1.77× to 2.24× over the
same OpenCV worker layout. GPU serial service rates and CPU per-call
latency percentiles are retained separately.

For the two annotated ArUco sequences, the audit matches marker instances
by position, checks ID and rotation, and measures corner error against
rendered truth. In all 24 runs across six hosts, OpenCV, optimized CPU and
CUDA found 39 of 48 tags with ArUco3 on and 43 of 48 with it off. There
were zero lost OpenCV detections, false positives, wrong IDs or wrong
rotations. Optimized CPU coordinates and ordered IDs matched OpenCV
exactly. The CUDA vs OpenCV comparison never worsened the worst-corner
error or RMSE of a shared truth marker by more than 0.001 px.

On Thor at 1080p with ArUco3 enabled, ID 153 in
`quality_multi_dense_1` has one corner 13.391 px from OpenCV's coordinate.
That corner's error against the renderer falls from 14.010 to 0.620 px.
The other three corners agree, and the marker's maximum corner error falls
from 14.010 to 9.402 px.

### ArUco fixed images and input memory

The [project README](../../../README.md#performance-and-detection-quality)
gives the median per-image p50 across the 16 prepared industrial photographs.
Every measured route returned zero detections on those photographs. The
median of the per-image optimized CPU gains is 1.46–1.64× at one thread and
1.33–1.68× at four. Each gain is calculated per image before taking the host
median.

With ArUco3 off, the separate CPU comparison also covers all 16 prepared
photographs, with ten warm-ups and 50 measured calls per image and process.
The median per-image gains are about 1.38–1.66× at one thread and
1.15–1.95× at four threads. All 16 images again return zero detections.

On Spark, the blank 640×480 control takes 0.384 ms on one OpenCV thread,
0.070 ms on the optimized CPU thread and 0.260 ms with device-input CUDA.
The 4K four-marker control returns four detections and takes 1.813,
0.702 and 1.075 ms on the same routes. The four-thread optimized CPU
takes 0.400 ms on that positive control.

Spark's device-input CUDA p50 across the 16 photographs is now 1.340 ms.
The earlier shared-GPU measurement was 3.405 ms. The older ordered-contour
study records the executable revision and approximately 94% unrelated GPU
utilization observed before that sweep. [Earlier ArUco measurement](../../gpu-steady-state-report.md)

In the separate Spark 4K memory probe, Hybrid uses 0.810 ms with device
input and 1.073 ms with pageable host input. CUDA-Resident uses 1.039 ms.
CUDA-E2E takes 1.566 ms with pageable input, 1.555 ms with caller-prepared
pinned input and 1.753 ms with managed input. The matched 4K control has
an optimized CPU four-thread peak RSS of 26.87 MiB, compared with 23.70 MiB
for OpenCV. RSS includes the host process and decoded image.

### AprilTag CUDA latency

Host-input p50 in milliseconds, shown as cuAprilTags → project. Each value is
the median of the independent process p50s. The original 720p pair on each
of the RTX PRO 6000 and RTX 4090 uses one process per route.

| GPU | 640×480 | 1280×720 | 1920×1080 | 3840×2160 | 4K p50 gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| DGX Spark GB10 | 0.724 → 0.372 | 1.444 → 0.513 | 2.836 → 0.667 | 10.304 → 1.280 | 8.05× |
| Jetson AGX Orin | 1.672 → 0.620 | 4.430 → 1.205 | 9.055 → 1.581 | 35.428 → 3.108 | 11.40× |
| Jetson AGX Thor | 0.819 → 0.345 | 2.118 → 0.424 | 4.743 → 0.542 | 19.427 → 1.016 | 19.12× |
| RTX PRO 6000 | 0.340 → 0.237 | 0.868 → 0.369 | 1.871 → 0.560 | 7.436 → 1.215 | 6.12× |
| RTX 4090 | 0.362 → 0.200 | 0.966 → 0.335 | 2.124 → 0.499 | 8.903 → 1.130 | 7.88× |
| RTX 5070 Ti | 0.308 → 0.193 | 0.858 → 0.287 | 1.930 → 0.416 | 8.208 → 0.860 | 9.54× |

At 4K, host-input p99 also fell on each host. Device-input p99 on the
RTX PRO 6000 rose from 3.452 to 4.354 ms despite a lower p50
(2.225 to 0.426 ms). At 640×480 on the same GPU cuAprilTags has the
lower device-input p50 (0.139 to 0.212 ms). On the three outdoor inputs
that GPU serves 2,539 cuAprilTags calls/s, 2,056 adaptive project calls/s
and 3,286 project calls/s with fixed spacing 1. These rates are the
medians of process-level calls per sum of synchronized detection time.

For the RTX 4090 at 640×480, the current device-input p50 is
0.186 → 0.163 ms. The earlier shared-GPU result was 2.109 → 0.272 ms
at the same resolution. [Earlier AprilTag measurement](../../../apriltag/benchmark/README.md#changing-frame-latency)

| 4K GPU | Device p50 / p95 / p99, cuAprilTags → project (ms) | Host p95 / p99, cuAprilTags → project (ms) |
| --- | --- | --- |
| DGX Spark GB10 | 8.993 / 11.099 / 11.575 → 0.937 / 2.717 / 5.485 | 12.513 / 12.938 → 3.039 / 5.802 |
| Jetson AGX Orin | 28.126 / 35.900 / 38.034 → 1.366 / 7.538 / 23.771 | 42.471 / 45.309 → 9.174 / 25.492 |
| Jetson AGX Thor | 17.004 / 21.764 / 22.825 → 0.925 / 3.827 / 14.075 | 24.089 / 25.243 → 3.868 / 14.173 |
| RTX PRO 6000 | 2.225 / 3.121 / 3.452 → 0.426 / 1.222 / 4.354 | 8.273 / 8.580 → 2.055 / 5.052 |
| RTX 4090 | 4.069 / 4.827 / 5.111 → 0.330 / 1.068 / 3.346 | 9.601 / 9.930 → 1.862 / 4.155 |
| RTX 5070 Ti | 4.383 / 5.659 / 5.867 → 0.351 / 1.201 / 3.733 | 9.473 / 9.741 → 1.722 / 4.193 |

All six hosts reproduced the same scored detections. Both detectors had
zero false positives, ID errors and rotation errors in these four annotated
sequences.

| Input | Ground truth | cuAprilTags TP | Project TP | Baseline truth lost by project |
| --- | ---: | ---: | ---: | ---: |
| 640×480 · 96 frames | 136 | 65 | 95 | 0 |
| 1280×720 · 96 frames | 136 | 57 | 104 | 0 |
| 1920×1080 · 96 frames | 136 | 47 | 100 | 1 |
| 3840×2160 · 32 frames | 48 | 18 | 35 | 0 |

At 4K, corner RMSE on the 18 shared detections fell from 1.334 to
0.342 px. The 1080p miss reproduced on every host and is ID 42 in
`quality_multi_dense_4`. The 1080p run therefore fails the criterion that
the project retain every cuAprilTags true detection.

### AprilTag CPU at 4K

The 2160p CPU benchmark uses the same changing 0/1/4/12-tag sequence recipe
as the previous 720p and 1080p runs. Larger image dimensions retain the
original tag sizes in pixels. Both detectors returned the expected
816 detections per 192 timed calls after checking the IDs on each frame
before timing.

| CPU host | 1T p50, OpenCV → project (ms) | 4T p50, OpenCV → project (ms) | 4T throughput, OpenCV → project (frames/s) |
| --- | ---: | ---: | ---: |
| DGX Spark | 93.967 → 52.532 | 95.214 → 37.735 | 10.49 → 26.44 |
| Jetson AGX Orin | 151.588 → 85.116 | 154.667 → 39.367 | 6.47 → 25.31 |
| Jetson AGX Thor | 107.735 → 53.303 | 107.938 → 22.105 | 9.22 → 43.40 |
| Xeon w7-3565X | 105.886 → 62.905 | 106.559 → 44.614 | 9.37 → 22.35 |
| Core i9-12900KS | 86.715 → 57.332 | 86.658 → 34.550 | 11.53 → 28.98 |
| Core Ultra 7 265 | 183.543 → 68.913 | 189.049 → 42.561 | 5.31 → 23.48 |

The 4T throughput gain is 2.38–4.71× in this 16-frame workload.

### Native 8256×6192 industrial photograph

The brewery machine-room photograph was decoded at its original pixel
dimensions. The following host-input measurements use the same gray8
pixels. Both ArUco and AprilTag returned zero detections on this image.

| Host | ArUco optimized CPU 4T (ms) | ArUco CUDA-E2E (ms) | AprilTag CUDA host, cuAprilTags → project (ms) |
| --- | ---: | ---: | ---: |
| DGX Spark | 1.832 | 3.191 | 62.626 → 9.153 |
| Jetson AGX Orin | 5.008 | 17.938 | 214.410 → 27.889 |
| Jetson AGX Thor | 3.298 | 4.848 | 128.069 → 12.520 |
| RTX PRO 6000 / Xeon | 1.807 | 4.337 | 50.755 → 5.805 |
| RTX 4090 / Core i9 | 1.480 | 5.529 | 55.837 → 7.423 |
| RTX 5070 Ti / Core Ultra 7 | 1.426 | 4.168 | 55.738 → 6.724 |

The other six native photographs span longest edges from 3872 to 5312 px.
The per-host records identify the photographs measured on each host and
retain their individual timings and observations.
[Source images and preparation](inputs.md#native-industrial-photographs)

### Kernel diagnostic

Nsight Systems 2025.3.2 recorded individual CUDA graph nodes
after five warm-ups. The values sum kernel durations for the final timed
call after its verified input upload.

| Detector and input | Kernels | RTX 4090 sum (ms) | DGX Spark sum (ms) |
| --- | ---: | ---: | ---: |
| ArUco CUDA, 1280×720, four markers | 129 | 0.519 | 0.630 |
| ArUco CUDA, 3840×2160, four markers | 131 | 0.736 | 0.972 |
| AprilTag CUDA, 3840×2160, one tag | 9 | 0.385 | 0.587 |
| cuAprilTags, 3840×2160, same tag | 17 | 2.459 | 7.415 |

The cuAprilTags kernels overlap and occupy a 2.210 ms span. The main
latency tables measure synchronized wall time and include the declared
transfer and result boundaries. On Spark the ArUco traces recorded peak
outstanding device allocations of 39.9 MiB at 720p and 63.0 MiB at 4K,
including the pitched input.

### Reproduction and retained evidence

Prepare and verify the datasets using the [input guide](inputs.md). Build
Release in the pinned OpenCV environment and link the external cuAprilTags
archive for the target platform under the [SDK terms and provenance](../../../apriltag/benchmark/README.md#scope).
Verify the pinned AprilRobotics source archive before setting the
`FETCHCONTENT_SOURCE_DIR_APRILTAG3` override. The [ARM build helper](build-study.sh)
selects the toolkit architecture and ordered-contour configuration used in this run.

```bash
study=docs/measurements/2026-11-19-performance-refresh
bash "$study/build-study.sh" "$PWD" build/refresh-build /path/to/pinned-sdk \
  dgx-spark-steady /path/to/pinned-apriltag-source
python3 "$study/run_april_gpu.py" --source "$PWD" \
  --binaries build/refresh-build/apriltag/benchmark --original-inputs /path/to/previous-inputs \
  --new-inputs /path/to/prepared-inputs --output build/refresh/apriltag-gpu --cpu 5 \
  --sets 640 720 1080 2160 field photos
bash "$study/run_host_phase.sh" "$PWD" build/refresh-build \
  /path/to/prepared-inputs /path/to/previous-inputs build/refresh 5,6,7,8
python3 "$study/summarize.py" --april build/refresh/apriltag-gpu --aruco build/refresh/aruco
```

Use the host's selected cores, toolkit and compatible SDK. Preserve the
NVIDIA archive hash, exact compiler configuration and command journal next to
the outputs. The quality scorer verifies the decoded pixels and ground-truth
marker correspondence. [ArUco scoring procedure](inputs.md)

The generated rows, clock and utilization snapshots, trace exports and
per-host summaries are retained locally in
`build/performance-refresh-20261119/` under the AprilTag worktree.
Use the ARM `aruco-synchronized` directories, the x86 `aruco-final-gpu`
directories and the RTX 5070 Ti `results/corrected` directory for current
ArUco CUDA measurements. The earlier ARM `aruco` directories contain the
CPU changing-frame runs and provisional CUDA rows. Orin's AprilTag
`apriltag-gpu-retry` contains the successful run in its existing JetPack
container after the first non-root launch failed at NvRm initialization.
[Measurement storage policy](../README.md)

## Goals

- Measure the established steady-state latency and throughput routes on the
  currently reachable hardware and repeat them on prepared high-resolution data.
- Check IDs, detection coverage, rotations and corner error against the
  annotated inputs before interpreting timing ratios.
- Preserve local raw evidence, including overflows, incomplete commands and
  separate GPU kernel traces.

## Open questions

- How does the 4K result change on a longer annotated sequence from production
  cameras, with independent subpixel corner labels?
- What p95 and p99 does each detector sustain under the intended application
  scheduler and camera pipeline?
