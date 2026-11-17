# CPU performance on six benchmark hosts

## Purpose

Determine whether the optimized CPU detector's advantage over actual OpenCV on
industrial backgrounds persists across the six supplied machines.

## Scope

This evaluation uses the same 16 real factory, warehouse and equipment-room
photographs as the [preceding Spark evaluation](cpu-real-background-report.md).
All images are retained, with the same prepared bytes, aspect ratios and longest
side at most 1920 pixels. These are public still photographs, not deployment
camera captures or independently annotated marker ground truth. The
[input record](../cpu/experiments/real_background.md) retains attribution and
preprocessing; no photographs are committed.

The detector implementation is unchanged at source snapshot
`c5c943232da465a5b0e89aec046787d8144dbab9`. Actual OpenCV is 4.14.0, commit
`0654a42e19215ef25b1d367d822f3c630447e7c7`, unmodified and built without CUDA.
The existing Spark binary is reused; the other architectures use Release builds
with `ARUCO3CPU_NATIVE=OFF`. No tracking, skipped frames or previous-frame image
or result reuse is introduced. GPUs are not used in this experiment.

## Current state

Generated measurement files referenced by local paths below are not version-controlled.
The report retains the results and conditions; raw outputs remain in local storage.

All six hosts completed the planned measurements. **All 384 image/configuration
comparisons beat OpenCV at aggregated p50, p95 and p99.** Each image/configuration
has three independent processes; all 1,152 individual-process p50 comparisons
also beat OpenCV. The minimum individual-process speedup is 1.037x.

The table gives the median of 16 image speedup ratios for each condition.
Each ratio is OpenCV p50 divided by optimized p50, after taking the median of
three process p50 values for that image. Values above one favor this detector.
These are neither pooled timings nor a guarantee for every image.

| CPU / host suffix | ArUco3 on, 1 thread | ArUco3 on, 4 threads | ArUco3 off, 1 thread | ArUco3 off, 4 threads |
| --- | ---: | ---: | ---: | ---: |
| DGX Spark GB10 / 20.23 | 1.577x | 1.674x | 1.660x | 1.838x |
| Jetson AGX Orin / 20.30 | 1.539x | 1.389x | 1.513x | 1.358x |
| Intel N97 / 10.26 | 1.509x | 1.443x | 1.469x | 1.277x |
| Xeon w7-3565X / 20.41 | 1.481x | 1.328x | 1.364x | 1.159x |
| Jetson AGX Thor / 20.20 | 1.629x | 1.623x | 1.639x | 1.961x |
| Core i9-12900KS / 20.44 | 1.450x | 1.442x | 1.356x | 1.242x |

Thus the earlier approximate 1.6–1.8x gain is specific to the Spark measurement;
it does not transfer unchanged to every CPU. Across these configurations, median
speedups range from 1.159x to 1.961x, corresponding to approximately 14–49% lower
detector-call latency. The smallest per-image aggregate margin is **1.041x** on
the Xeon, ArUco3 disabled with four threads, for `01_warehouse_racks.png`.
That small observed margin is not a guarantee under different loads or inputs.

All 16 photographs returned zero detections in every route/mode/thread/process.
The four positive-control configurations on each host returned four markers.
The benchmark's exact ID, oriented-corner and rejected-count comparisons passed.
Checks cover the initial untimed and final retained results; intermediate timed
outputs are not retained. These checks establish compatibility on this corpus,
not general field recognition accuracy.

### Measurement conditions

The [plan](measurements/2026-11-17-cpu-multihost/plan.md) was fixed before timings.
All routes use DICT_ARUCO_MIP_36h12 and matching detection settings.

- ArUco3 enabled: 30 warm-ups and 200 timed calls per route/image/process.
- ArUco3 disabled: 10 warm-ups and 50 timed calls per route/image/process,
  at the full prepared-image resolution. Secondary tails have limited resolution.
- Three independent processes for each of four conditions on each host.
- One thread or four threads pinned to distinct physical cores. Performance
  cores are selected on heterogeneous CPUs; affinity is recorded below.
- Alternating route order in the unchanged paired benchmark; no profiling.
- Host grayscale input to host results. Decoding, construction, validation,
  warm-up and collection of OpenCV rejected outputs are outside timing.

Startup latency is excluded. This is single-frame detector-call latency, not
camera-to-result latency, sustained video throughput or CUDA kernel time.
No task builds overlap timings on the same host.

| Host | CPU affinity: 1 thread / 4 threads | Build and retained environment |
| --- | --- | --- |
| 20.23 | 5 / 5,6,7,8 | GCC 13.3; existing Spark container; existing host services |
| 20.30 | 8 / 8,9,10,11 | GCC 9.4; Orin container, MAXN; preexisting busy system service |
| 10.26 | 0 / 0,1,2,3 | Portable GCC 13.3 x86 bundle; native Ubuntu 26.04.1 runtime |
| 20.41 | 16 / 16,17,18,19 | Same x86 bundle; native Ubuntu 24.04 runtime; preexisting CPU workload |
| 20.20 | 4 / 4,5,6,7 | GCC 13.3; Thor container, MAXN; preexisting simulation workload |
| 20.44 | 0 / 0,2,4,6 | Same x86 detector/OpenCV bytes; Ubuntu 24.04 container; preexisting CPU workload |

All three x86 hosts use byte-identical benchmark binaries and seven actual
resolved OpenCV libraries, independently checked by hash. OpenCV x86 has
SSE/SSE2/SSE3 baseline, runtime dispatch through AVX512_SKX and Intel IPP 2026.0.
Arm builds use NEON/FP16 baseline; compiler and dispatch differences are retained
in each host's build information. All builds use OpenCV's pthreads backend.
N97 uses its host glibc/libm; the whole OS runtime is not claimed identical.

Power modes, governors and unrelated services were retained. Orin's preexisting
`systemd-udevd` consumed approximately one core. Its observed CPU was avoided,
but the unrestricted service could migrate. The Xeon and i9 also had existing
CPU-bound work; Thor had a simulation workload. Frequency snapshots are recorded
observations, not fixed-clock claims. These conditions and differing toolchains
limit CPU-only interpretations of absolute cross-host latency rankings.

The largest optimized p50 process spread was 18.158% for one Thor four-thread
full-resolution scene; those measurements remain included. The reports retain
all per-image spreads, not confidence intervals. Existing libpng iCCP warnings
were recorded during decoding, outside the detector timing.

### Evidence and reproduction

The [measurement archive](measurements/2026-11-17-cpu-multihost/README.md) indexes
the [runner](measurements/2026-11-17-cpu-multihost/run_host.py), source/input
snapshots, build records, raw rows and exact commands. The
aggregate (local `measurements/2026-11-17-cpu-multihost/aggregate.json`) retains every
image/configuration result, and the
CSV (local `measurements/2026-11-17-cpu-multihost/scene-latencies.csv`) provides absolute
p50/p95/p99 latencies in milliseconds for both implementations.

The independent audit (local `measurements/2026-11-17-cpu-multihost/qa-audit.json`) checks
2,304 timed rows, all inputs/dimensions, 173 frozen source hashes, configuration,
OpenCV provenance, output compatibility, all controls, recomputed aggregates and
x86 runtime hashes. No detector source was changed for this evaluation.

## Goals

- Measure the advantage across all six supplied hosts without discarding losses.
- Preserve startup-free latency, tails, variability and correctness evidence.
- Keep observed per-host results distinct from general performance guarantees.

## Open questions

- Other images, dictionaries, camera exposure/blur/compression and native 4K
  inputs can change the result. The dense synthetic checkerboard crossover in
  the earlier CPU report is not superseded by this photographic experiment.
- Different power policies, concurrent applications, sustained thermal conditions
  and thread budgets require separate measurements.
- This corpus does not establish detection recall or sustained video throughput.
