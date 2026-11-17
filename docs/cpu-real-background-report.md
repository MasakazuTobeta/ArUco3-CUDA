# CPU performance on real industrial backgrounds

## Purpose

Evaluate the optimized CPU detector against actual OpenCV on real factory,
warehouse and equipment-room photographs where the configured detectors return
no markers. This addresses the gap between synthetic empty/noisy inputs and
industrial photographic backgrounds.

## Scope

This is an evaluation of unchanged implementation commit `d677c4f`, not another
optimization. The portable benchmark is the final repaired binary from the
[steady-state experiment](cpu-steady-state-report.md), SHA-256
`99581bfab75461f5e11bb9687966b2ba8d96248156dd5abd69f8a9209fc6bdcb`.
The host is DGX Spark GB10; actual OpenCV is 4.14.0, source commit
`0654a42e19215ef25b1d367d822f3c630447e7c7`.

Sixteen public photographs were selected by visible industrial content and
provenance before any detector execution: five warehouses/storage facilities,
five production scenes, three machine shops and three equipment/plant rooms.
All were retained regardless of subsequent detections or timings. Two agents
visually reviewed the contact sheet. These are real photographs, with no inserted
markers, artificial noise or crops. They are not the user's camera recordings
or a statistically sampled collection of factory backgrounds.

Inputs preserve aspect ratio, apply EXIF orientation and convert to RGB8 PNG.
Fifteen are downsized to a longest side of 1920 pixels; one native 1901x1494
image is unchanged in size. There is no upscaling. The
[source/transform record](../cpu/experiments/real_background.md) and
[pinned manifest](../cpu/experiments/real_background_manifest.json) retain source
pages, author/license, original/prepared hashes and exact Pillow settings.
Photographs remain local and are not redistributed in this repository.

## Current state

Generated measurement files referenced by local paths below are not version-controlled.
The report retains the results and conditions; raw outputs remain in local storage.

This section retains the original single-host measurement. A subsequent
[six-host evaluation](cpu-multihost-report.md) repeats the same inputs on Spark,
Orin, Thor, Xeon, Core i9 and Intel N97 and records the CPU-dependent gains.

All 16 photos returned zero detections in both strategies and both thread
budgets, consistently across all three processes and both routes. The optimized
implementation beats OpenCV at the aggregated per-scene p50, p95 and p99 in
all 64 image/configuration conditions. No photo was removed after execution.

| Strategy | Threads | Photos with zero detections | p50 wins vs OpenCV | Median speedup | Min–max speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| ArUco3 enabled | 1 | 16/16 | 16/16 | 1.578x | 1.406–1.788x |
| ArUco3 enabled | 4 | 16/16 | 16/16 | 1.679x | 1.290–1.988x |
| ArUco3 disabled | 1 | 16/16 | 16/16 | 1.662x | 1.388–1.946x |
| ArUco3 disabled | 4 | 16/16 | 16/16 | 1.833x | 1.316–2.143x |

Speedup is OpenCV p50 divided by optimized p50, using the median of three
process p50 values for each image. The aggregate is the median of these 16
ratios; it is not pooled latency or the ratio of total dataset runtimes. Every
individual process/image p50 comparison also wins: 192/192, minimum 1.279x.
Positive controls detect the expected four markers under both strategies.
These checks establish compatibility and background latency in this set;
zero detections alone do not establish field recognition accuracy.

Primary candidate p50 process-spread median/max is 5.23%/7.15% at one thread
and 1.28%/3.30% at four. Secondary values are 0.49%/1.02% and 1.78%/2.46%.
The gains exceed the observed variation in these runs, but the ranges are
descriptive process spreads, not statistical confidence intervals.

### Conditions and checks

The [plan](measurements/2026-11-17-cpu-real-background/plan.md) was written before
measurements. All routes use DICT_ARUCO_MIP_36h12, identical images and detection
settings. Primary runs enable ArUco3; secondary runs disable it and operate at
full **processed-image** resolution, not original camera resolution. The
secondary mode uses the same no-refinement setting for both routes.

One-thread runs use performance CPU 5; four-thread runs use CPUs 5–8. Each of
the four groups runs in three independent processes, with alternating route
order inside each process. Primary runs use 30 warm-ups and 200 samples per
image/route; secondary runs use 10 warm-ups and 50 samples. Percentiles with
50 samples have limited tail resolution. Stage profiling is disabled. No other
agent build/test/detector execution overlaps timing; unrelated host services,
ASLR and the host power/clock policy are retained. Raw frequency snapshots are
not treated as proof of a fixed physical clock.

The timed interval is host grayscale input to host IDs/corners. Decoding,
setup, warm-up and compatibility checks are excluded. OpenCV rejected outputs
are collected separately outside timing for compatibility. This is single-call
latency, not camera latency or video throughput; CUDA kernel time is inapplicable.

The benchmark compares initial untimed and final retained outputs against the
actual OpenCV oracle: IDs and per-marker oriented corners after detection
matching, and rejected counts. It does not save/check every intermediate timed
result. The orchestrator additionally verifies every measurement row, exact
input hashes, complete scene/route counts and stable detection counts across
all processes. Known four-marker positive controls run in both modes, preventing
an all-empty result from being mistaken for evidence that detection works.

One-sample, zero-warm-up eligibility checks precede timing and are retained only
as correctness records. Their durations are not used in performance conclusions.
Libpng reports a known-incorrect sRGB-profile warning on some prepared inputs;
image decoding occurs outside timing, both routes receive the same decoded
matrix, and the frozen inputs are not modified to suppress the warning.

### Per-scene latency

**ArUco3 enabled.** All times below are p50 milliseconds.

| Photograph | OpenCV 1t | Optimized 1t | Speedup 1t | OpenCV 4t | Optimized 4t | Speedup 4t |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| warehouse racks | 4.499 | 3.133 | 1.436x | 2.209 | 1.476 | 1.497x |
| warehouse fema | 3.560 | 2.307 | 1.543x | 1.472 | 0.983 | 1.498x |
| distribution sainsburys | 4.589 | 3.160 | 1.452x | 1.831 | 1.415 | 1.294x |
| steel pipe storage | 2.436 | 1.369 | 1.780x | 1.192 | 0.600 | 1.988x |
| warehouse adamhall | 3.648 | 2.395 | 1.523x | 1.544 | 1.015 | 1.522x |
| robot bakery | 5.147 | 3.662 | 1.406x | 2.082 | 1.583 | 1.315x |
| production elevator | 3.710 | 2.432 | 1.526x | 1.586 | 0.969 | 1.636x |
| can manufacturing | 2.576 | 1.489 | 1.730x | 1.229 | 0.654 | 1.880x |
| tannery | 3.279 | 2.058 | 1.593x | 1.469 | 0.830 | 1.771x |
| yarn production | 2.726 | 1.674 | 1.629x | 1.204 | 0.699 | 1.721x |
| cnc facility | 2.057 | 1.423 | 1.446x | 0.871 | 0.675 | 1.290x |
| noirlab workshop | 3.047 | 1.949 | 1.564x | 1.345 | 0.830 | 1.622x |
| haas lathes | 2.308 | 1.384 | 1.668x | 1.114 | 0.637 | 1.748x |
| weir machine room | 2.619 | 1.465 | 1.788x | 1.279 | 0.670 | 1.908x |
| brewery machine room | 2.998 | 1.845 | 1.625x | 1.418 | 0.823 | 1.722x |
| tower plant room | 2.930 | 1.792 | 1.636x | 1.427 | 0.798 | 1.788x |

Per-scene p95/p99, input hashes and process spread: ArUco3 enabled, 1t (local `measurements/2026-11-17-cpu-real-background/aruco3-1t/summary.json`), 4t (local `measurements/2026-11-17-cpu-real-background/aruco3-4t/summary.json`).

**ArUco3 disabled, full processed image.** All times below are p50 milliseconds.

| Photograph | OpenCV 1t | Optimized 1t | Speedup 1t | OpenCV 4t | Optimized 4t | Speedup 4t |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| warehouse racks | 44.476 | 24.621 | 1.806x | 19.376 | 11.365 | 1.705x |
| warehouse fema | 42.653 | 27.268 | 1.564x | 20.113 | 12.427 | 1.618x |
| distribution sainsburys | 40.154 | 25.567 | 1.571x | 19.456 | 11.452 | 1.699x |
| steel pipe storage | 23.612 | 12.322 | 1.916x | 12.881 | 6.009 | 2.143x |
| warehouse adamhall | 33.317 | 20.722 | 1.608x | 16.765 | 9.579 | 1.750x |
| robot bakery | 66.355 | 47.822 | 1.388x | 27.201 | 20.676 | 1.316x |
| production elevator | 41.205 | 26.193 | 1.573x | 19.408 | 10.646 | 1.823x |
| can manufacturing | 21.052 | 10.819 | 1.946x | 12.190 | 6.049 | 2.015x |
| tannery | 32.197 | 19.328 | 1.666x | 16.082 | 8.723 | 1.844x |
| yarn production | 25.541 | 14.917 | 1.712x | 14.081 | 7.549 | 1.865x |
| cnc facility | 22.872 | 15.449 | 1.480x | 10.312 | 6.411 | 1.608x |
| noirlab workshop | 29.860 | 18.420 | 1.621x | 15.030 | 8.259 | 1.820x |
| haas lathes | 21.302 | 11.392 | 1.870x | 11.815 | 5.749 | 2.055x |
| weir machine room | 24.461 | 13.101 | 1.867x | 13.223 | 6.263 | 2.111x |
| brewery machine room | 32.361 | 19.511 | 1.659x | 16.888 | 8.767 | 1.926x |
| tower plant room | 28.808 | 16.781 | 1.717x | 15.546 | 8.119 | 1.915x |

Per-scene p95/p99, input hashes and process spread: ArUco3 disabled, full processed image, 1t (local `measurements/2026-11-17-cpu-real-background/fullres-1t/summary.json`), 4t (local `measurements/2026-11-17-cpu-real-background/fullres-4t/summary.json`).


### Reproduction and integrity

Prepare the inputs using the pinned [fetcher](../cpu/experiments/fetch_real_background.py)
and its documented Pillow environment. In the existing CPU container, run from
the repository root:

```bash
python3 docs/measurements/2026-11-17-cpu-real-background/run_measurements.py \
  --input-dir data/cpu-real-background
```

The orchestrator refuses existing result directories. Use a separate checkout
and fresh output locations to reproduce without overwriting evidence. It checks
the frozen binary hash, runs positive controls/eligibility before the four
sequential timing groups, and uses the maintained per-process runner and
summarizer. All command/configuration/source hashes are recorded with each group.

The independent audit (local `measurements/2026-11-17-cpu-real-background/audit.json`)
verified all 384 timed measurement rows, the 192 individual-process p50 wins,
all input dimensions/hashes, frozen production source and binary, zero outputs,
positive controls, and recomputed summaries/tails/spreads. The
execution record (local `measurements/2026-11-17-cpu-real-background/execution.json`) and
aggregate (local `measurements/2026-11-17-cpu-real-background/aggregate.json`) preserve
completion and the reported comparison. Full evidence is indexed in the
[measurement archive](measurements/2026-11-17-cpu-real-background/README.md).

## Goals

- Establish the observed OpenCV comparison on diverse real industrial inputs.
- Preserve no-detection eligibility, exact compatibility, tails, variation and
  unfavorable results without removing difficult images.
- Keep historical synthetic results and this photographic evaluation distinct.

## Open questions

- These 16 still photographs do not establish universal factory performance,
  actual-camera latency, sustained video throughput or independent field accuracy.
- Results depend on dictionary/settings, exposure, blur, compression, scene
  content and resolution. Native 4K camera inputs remain unmeasured; other CPUs
  are covered separately by the [six-host evaluation](cpu-multihost-report.md).
- Zero configured detections do not independently prove absence of every possible
  marker in the photographs, or measure recognition recall.
