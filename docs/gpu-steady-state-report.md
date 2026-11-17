# GPU steady-state optimization and CPU-quality validation

## Purpose

Improve GPU latency using the CPU optimization findings while retaining detection
quality at least equivalent to pinned OpenCV. Initialization time is excluded.
Different corner coordinates are accepted only when detection quality and
ground-truth geometric accuracy are equivalent or better.

## Scope

This study compares merged baseline `badb673`, the explicit steady GPU presets,
official OpenCV 4.14.0 and the optimized CPU implementation. It does not use
tracking, frame skipping or previous-frame pixels/results. Device processing
remains on the GPU. Immutable coefficients, dictionaries and reusable storage
are permitted. Source provenance is recorded in [the provenance record](code-provenance.md).

The unchanged old GPU is a historical control, not a CPU-quality-equivalent
competitor. Its one-quad-per-component rule loses marker holes connected through
background lines. The expanded 96-scene corpus contains 136 true markers:

| ArUco3 | OpenCV / optimized CPU TP | Old GPU TP | Contour GPU TP on Spark | False positives |
| --- | ---: | ---: | ---: | ---: |
| On | 113 | 56 | 113 | 0 |
| Off | 124 | 65 | 124 | 0 |

These are unique-scene counts; repeated frames and blank cuts are excluded.
These contour counts include the border-margin review correction on Spark.
Final-binary validation is recorded below. Photographic agreement without
annotated ground truth establishes compatibility, not field recall.

## Current state

Review follow-up: the ordered-contour route omitted the configured image-border
margin. The correction checks the selected group representative before packing
and decoding, in segmentation-image coordinates, matching the CPU ordering.
Earlier GPU counts of 118/127 included five/three truths in border-occlusion
scenes that violated the configured margin and were rejected by CPU. Correct
filtering removes those extras; CPU-detectable truths are retained. Remaining
scene observations and ground-truth metrics are unchanged in the 96-scene audit.
The focused correction passes 42 grouping/detector tests on Spark and Orin,
including 64 changing-frame calls over both ArUco3 modes, all four borders and
direct/captured submission. Three grouping tests cover inclusive boundaries,
zero/impossible margins, invalid dimensions and representative/alternative
ordering. All four sanitizer tools pass these four new tests on both hosts.
The new end-to-end test fails when linked against the immutable previous core,
and the border-selection mutation also fails the boundary/group-order tests.
The corrected build additionally passes the complete 2398-call audit and 182
historical-image evaluations on each ARM host, including nondefault settings,
all 17 Dictionaries, industrial photographs and annotated close-marker chains.
Local follow-up records are `build/gpu-study/review-border-spark-full-matrix-complete/`
and `build/gpu-study/review-border-v3-orin-broad/`, with focused sanitizer and
failure-reproduction records alongside them. This correction is not separately
runtime-validated on RTX 4090; its earlier results below describe the pre-review build.
The timing tables below describe the earlier frozen source export; they have
not been remeasured after this correctness correction. Fixed-image repetition
measures detector-call latency here; changing-frame video throughput remains
outside this timing study.

CPU-equivalent quality is achieved on the expanded validation matrix. The
optimized contour route does **not** universally beat the old GPU or optimized
CPU. On Orin industrial photographs with ArUco3 enabled, resident GPU latency
beats one-thread OpenCV and optimized CPU on all 16 images, with median per-image
speedups of 2.328x and 1.524x. Four-thread optimized CPU wins 15 of 16. On the
busy Spark, optimized CPU wins all 16 with either thread budget. The old GPU
remains faster on all photographs, but fails the broader positive-marker quality
gate. Improving that quality costs both latency and memory.

The implementation combines the following separately checked changes. The
ordinary presets retain the historical extractor; `dgx-spark-steady` and
`jetson-orin-steady` select the common new algorithm explicitly.

| Change | Work avoided or parallelized | Qualification |
| --- | --- | --- |
| Ordered device contours | Recovers independent marker holes; parallel edge counting, warp polygon approximation and contour materialization | Required for expanded CPU-quality coverage; more work than the old insufficient extractor |
| Fused multiwindow threshold | Shares input traversal and local sums | Exact binary planes; explicit large-window fallback |
| Demand pyramid | Builds only the prefix of levels requested by current-frame candidates | Not a sparse-pixel implementation; no previous-frame image reuse |
| Density-selected fixed-point Otsu | Skips provably empty histogram-bin work | Exact recurrence; dense histograms retain the control path |
| Bounded candidate grids | Avoids launching capacity-sized grids on sparse frames | 512 blocks selected; dense stress cases can favor the capacity control |
| Resize coefficient preparation | Moves immutable Kleidi geometry arithmetic to setup | Exact pinned OpenCV rounding; ARM backend and FMA policy are explicit |
| Candidate grouping and retry | Uses quadrilateral perimeter and tries eligible close alternatives | Preserves CPU-detectable cases the earlier GPU omitted |

The [strategy options](../cmake/ArUco3CudaStrategies.md),
[resize profiles](../cmake/ArUco3CudaResizeProfile.md),
[contour design](../src/core/contour_candidates.md) and
[lifetime contract](../src/core/detector.md) describe the controls and limits.
Correct ordering across streams and failure-safe resource replacement are part
of the measured detector, including completion-event API overhead.

Intermediate ablations explain the selected controls rather than claiming
multiplicative gains. Eligible-edge materialization reduced a paired Orin
warehouse-frame pipeline from 4.196–4.237 ms to 3.081–3.100 ms across processes,
with identical GPU observations under that control. Blank frames instead paid
19–31 microseconds more. Fused three-window thresholding improved isolated
stage times by roughly 1.65–2.65x, while a single 63-pixel window lost on Spark.
The 512-block candidate policy improved sparse work, but dense 16384-active
decode stress cases could remain 10–20% slower than capacity-sized grids.
Kleidi coefficient preparation was retained; broader coefficient caching and
32-bit candidate-loop rewrites were not promoted. These are separate experiments
with their own inputs/scopes, documented in the module records, not a product
of speedup ratios or a claim that the old GPU is slower overall.

### Measurement conditions

- Seven fixed synthetic images and sixteen industrial photographs; all 23 are
  measured with ArUco3 enabled. Disabled measurements retain all seven synthetic
  images and the first three photographs by filename. The
  [fixed input manifest](../tools/gpu_quality/steady-inputs.json) records identities.
- DICT_ARUCO_MIP_36h12; identical detection settings. Candidate/marker limits are
  16384/1024 for equal-budget comparisons. A separate old-default resident control
  retains 4096/1024. A full-resolution checkerboard needs 5460 raw candidates;
  its default-budget overflow remains an explicit failure, never silent truncation.
- Three independent processes per route/mode, twenty warm-ups and two hundred
  measured calls per image/process. Process order reverses in the second run.
  CPU variants alternate internally. Single-frame latency only; no throughput claim.
- CPU uses one or four physical cores, native tuning enabled. Spark affinity is
  5 / 5,6,7,8; Orin is 8 / 8,9,10,11. GPU submission uses the first listed core.
- CUDA-Resident measures device input through stream completion, excluding host
  output download. CUDA-E2E includes pageable-host upload and result retrieval.
  CPU covers host grayscale input through host results. Initialization, file
  decoding and serialization are excluded. Neither wall interval is kernel time.
- Existing workloads, power policies and clocks are retained. No task-owned
  compilation, profiling or quality jobs overlap primary timing on the same GPU.
  Shared-host load prevents interpreting these measurements as a physical limit.

The [maintained runner](../tools/gpu_quality/steady_benchmark.md) verifies every
input and executable hash, build identity, budget, measurement scope, sample
count and nearest-rank percentile. Per-image p50/p95/p99 are medians of the
three process percentiles, with process ranges retained. Image-level speedup
ratios are calculated before taking medians across photographs.

| Host | Toolchain / runtime | Resize descriptor | Retained operating condition |
| --- | --- | --- | --- |
| DGX Spark GB10 | CUDA 13.0.88, GCC 13.3, driver 580.95.05; Ubuntu 24.04.4 container | OPENCV_ARM_HAL / SVE128 / SEPARATE | Performance CPU governor; substantial unrelated GPU work (94% before the sweep) |
| Jetson AGX Orin | CUDA 11.4.315, GCC 9.4, L4T R35.4.1; Ubuntu 20.04.6 container | OPENCV_ARM_HAL / NEON / FUSED | Existing MAXN mode, schedutil CPU governor; background system service retained |

Both link unmodified OpenCV 4.14.0 at commit
`0654a42e19215ef25b1d367d822f3c630447e7c7`. GPU clock snapshots are 2437 MHz
on Spark and 1300 MHz on Orin, not fixed-clock guarantees. Spark CPU
`scaling_cur_freq` again exceeded `scaling_max_freq`; the cause is unverified
and those readings are not described as physical controlled frequencies.
Orin power/utilization helpers were unavailable inside the measurement
container; a subsequent host query confirms MAXN and idle GPU after the sweep,
while reporting an EMC-cap permission error. The attempted Python CUDA-memory
query could not resolve `libcudart.so`; its failure and host-memory snapshots are
retained. Detector allocations and all measurement calls nevertheless completed.
These limitations are not grounds to relabel this as an isolated-host experiment.

### Validation and measurements

The final source snapshot passed 2398 audit calls plus 182 historical image
evaluations on each of Spark, Orin and RTX 4090. The matrix covers 17 Dictionaries,
configuration variants, close-marker chains, changed frames and industrial
photographs. Separate Spark and Orin preflights of all 23 timing inputs in both
modes also pass: no CPU/GPU differences, and all 16 photographs remain empty.
Both hosts completed all 42 planned timing commands: 891 measurement rows and
178200 timed calls per host. All samples, metadata and independent percentile
recalculations pass. Validation was hardened afterward to reject nonfinite
reported statistics and missing/duplicated rows; every original row was
revalidated, with unchanged timing binaries and samples. Synthetic corruption
tests now run without CUDA in CI.
Earlier stage experiments establish candidate selection, not final end-to-end
speed. Local evidence is retained
under ignored `build/gpu-study/`; generated records are absent from a fresh
checkout under the [measurement storage policy](measurements/README.md).

Before the border correction, the 96-scene ARM counts were 113/118 (on) and
124/127 (off). On x86, platform-dependent
rendered pixels gave CPU/GPU counts 114/119 with ArUco3 on and 124/127 off;
all have zero false positives and correct detected rotations. Across the
external matrix, each host recovers all 278 CPU detections. The largest positive
common-truth corner-error delta across variants/chains is 0.000003095 pixels
on ARM and 0.00006100 pixels on x86, below the predeclared 0.001-pixel numerical
classification band. Raw positive deltas are retained rather than rounded to
zero. No finite corpus establishes equivalence on every possible input.

On both ARM hosts, the complete randomized sequential CTest run initially
passes 627 of 628 cases without skips. The one failure is a test fixture that
hardcodes CPU 0, outside the container's allowed affinity. The corrected fixture
selects an allowed CPU and restores the original mask; its separate rerun
passes against unchanged frozen libraries. C ABI/Python coverage includes 24
binding and 12 device tests. Six deliberate-error lifetime tests pass on all
three hosts. Both Spark and Orin pass 103 selected tests under each of memcheck,
racecheck, initcheck and synccheck; RTX 4090 passes 61 focused functional tests
and 22 under each sanitizer. Deliberate API failures are excluded from sanitizer
runs. The x86 run uses CUDA 12.6 with its compatible installed driver; earlier
CUDA 13 device skips are rejected evidence.

The no-OpenCV historical-default build also compiles and embeds SM87/110/120/121.
Installed C/C++ consumers, standalone samples, 54 host tests and 24 Python
binding tests pass; C++/Python marker rendering is byte-identical. Host coverage
is 68.2% lines and 38.1% branch edges over the broad project scope, with all
shortfalls retained and explained in the [coverage record](implementation-plan.md#coverage).

The [maintained matrix runner](../tools/gpu_quality/run_matrix.md) records exact
commands, capacities, completion counts, input/binary hashes and source checks.
The tested production export has manifest SHA-256
`c6e689b9c24256d74ef34fb6e18e761fc409a97c45b3f75e124966226e3d2197`.
The subsequent border-margin correction changes candidate acceptance and is
validated separately above. The earlier affinity test correction, reproduction
tools and documentation did not change production behavior.

### Whole-detector latency

The following photo summaries are medians across image-level process medians,
in milliseconds. Enabled uses all 16 photographs; disabled uses the three fixed
warehouse photographs. Every route returns zero detections. These are not
pooled percentiles or a statistical sample of all factory scenes.

| Host / ArUco3 | OpenCV 1t | Optimized CPU 1t | OpenCV 4t | Optimized CPU 4t | Old GPU default resident | New GPU resident | New GPU E2E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Spark / on | 2.984 | 1.882 | 1.395 | 0.817 | 2.794 | 3.405 | 3.464 |
| Orin / on | 6.892 | 4.518 | 2.928 | 2.080 | 1.542 | 3.037 | 3.723 |
| Spark / off | 39.999 | 25.527 | 19.696 | 11.688 | 11.759 | 23.641 | 27.844 |
| Orin / off | 88.097 | 56.239 | 33.340 | 24.840 | 18.298 | 34.446 | 35.337 |

The old equal-budget resident medians are 2.832/1.643 ms (Spark/Orin, on) and
12.448/18.189 ms (off). Thus neither matching the budget nor retaining the
old 4096 default reverses the photographic old/new ordering. Enabled resident
speedups over old-default are 0.814x on Spark and 0.518x on Orin (medians of
image ratios; below one means slower). These negative-image results are retained
even though the old extractor's broader quality is insufficient.

For full-resolution photographs, new resident GPU beats one-thread optimized
CPU on all three images on both hosts, by median ratios 1.080x on Spark and
1.633x on Orin. Four-thread optimized CPU wins all three on both hosts.
E2E includes transfers and download and must not be inferred from resident results.

| Host / ArUco3 | New resident p50 / p95 / p99 | New E2E p50 / p95 / p99 |
| --- | ---: | ---: |
| Spark / on | 3.405 / 3.719 / 4.047 | 3.464 / 3.840 / 4.111 |
| Orin / on | 3.037 / 3.241 / 3.445 | 3.723 / 4.289 / 4.718 |
| Spark / off | 23.641 / 29.762 / 30.081 | 27.844 / 29.870 / 30.333 |
| Orin / off | 34.446 / 34.823 / 47.541 | 35.337 / 36.579 / 46.330 |

Each tail column is the median across images of the median of three process
percentiles. Full per-image/process ranges remain in local CSVs. The largest
relative new-resident p50 process spread is 2.9025–3.2640 ms for Spark steel storage (on),
and 1.6518–1.6990 ms for Orin 640x480 noise (on). No outliers were removed.

Synthetic positive, degraded and empty controls also show a crossover. The
following enabled-mode table retains all seven inputs; times are p50 ms and
the detection count agrees across the compared routes.

| Image | Detections | Spark OpenCV / optimized CPU 1t / new GPU resident | Orin OpenCV / optimized CPU 1t / new GPU resident |
| --- | ---: | ---: | ---: |
| 640x480 clean, four markers | 4 | 0.573 / 0.235 / 3.017 | 1.310 / 0.769 / 1.393 |
| 1280x720 clean, four markers | 4 | 0.729 / 0.262 / 2.853 | 1.729 / 0.937 / 1.308 |
| 1920x1080 uniform empty | 0 | 0.824 / 0.163 / 2.398 | 2.067 / 0.633 / 0.715 |
| 1920x1080 clean, sixteen markers | 16 | 1.513 / 0.815 / 3.143 | 4.152 / 2.947 / 2.060 |
| 3840x2160 clean, four large markers | 4 | 1.869 / 0.727 / 3.157 | 6.035 / 3.570 / 2.393 |
| 640x480 noise | 4 | 1.369 / 0.744 / 3.032 | 3.168 / 1.815 / 1.656 |
| 1280x720 blur | 0 | 0.607 / 0.144 / 2.531 | 1.354 / 0.500 / 0.819 |

With ArUco3 disabled, new GPU resident beats one-thread optimized CPU only on
the noise control among the seven synthetic inputs on each host: Spark
7.807 versus 9.879 ms, Orin 10.945 versus 20.129 ms. The empty full-HD case is
6.981 versus 2.417 ms on Spark and 8.002 versus 7.262 ms on Orin; the positive
4K case is 33.695 versus 11.198 ms and 49.454 versus 32.256 ms respectively.
Resolution alone is not a sufficient selection rule.

### Separate kernel diagnostic

A separate Spark Nsight Systems graph-node trace records the final binaries
with equal 16384/1024 limits and a separate old-default 4096/1024 control, all
with ArUco3 enabled. The table is the median of
the summed kernel durations in each of the last 20 graph launches, after the
first invocation and five warm-ups. Profiling perturbs execution; these values
must not be substituted for the primary wall-latency measurements.

| Image | Old GPU default 4096 | Old GPU equal 16384 | New GPU 16384 |
| --- | ---: | ---: | ---: |
| 1280x720, four clean markers | 0.501 ms | 0.529 ms | 0.633 ms |
| 1920x1080, uniform empty frame | 0.203 ms | 0.260 ms | 0.172 ms |
| Warehouse racks photograph | 0.665 ms | 0.791 ms | 1.269 ms |

The new empty-frame kernels are 1.51x faster at equal budget, and 1.18x faster
than the old default, in this diagnostic. The marker and
textured-image cases remain slower than the old, quality-deficient extractor.
The new warehouse kernel sum is dominated by polygon approximation (about
305 microseconds), contour counting (242) and cell decode (117). The blank-frame
trace instead retains edge initialization, root marking/compaction and fixed
launch costs. These observations identify remaining work; they do not prove a
hardware limit or attribute the complete wall/kernel gap to one cause.

Reproduction uses `nsys profile --trace=cuda --cuda-graph-trace=node --sample=none
--cpuctxsw=none`, wrapping the ordinary resident benchmark with `--warmup 5
--latency-iterations 20 --throughput-frames 0`. Export SQLite and group kernel
durations by graph-launch correlation ID; retain the last 20 complete launches.
Local `build/gpu-study/final-v2-profile-summary.json` and
`build/gpu-study/final-v2-profile-default-summary.json` record exact commands,
binary hashes, per-frame counts, spans and stage durations.

### Memory cost

The historical 91-image evaluator on Spark measures the following detector
workspace bytes with **4096 candidates and 1024 markers**, both before and after
the change. No workspace allocation occurs during the 91 detection calls in
either mode. These are detector arena statistics, not total CUDA-context or
process memory, and not the larger-budget primary benchmark configuration.

| ArUco3 | Old peak used / capacity (bytes) | New peak used / capacity (bytes) |
| --- | ---: | ---: |
| On | 17,512,724 / 22,685,440 | 27,781,140 / 38,957,056 |
| Off | 414,514,452 / 414,514,944 | 902,643,220 / 902,739,456 |

Recovering independent ordered contours increases memory relative to the
quality-deficient component extractor. Contour scratch was reduced from about
160 to 92 bytes per segmentation pixel during optimization, but full-resolution
detection still pays the larger contour cost. Configure image dimensions and
candidate limits for the deployment and treat explicit overflow as failure.

## Goals

- Preserve CPU-detected truths, correct IDs/rotation and per-truth corner quality.
- Include empty textured industrial backgrounds and unfavorable timing cases.
- Compare the whole detector as well as isolated stages, with resource limits visible.
- Keep a reviewable implementation and reproduction procedures without raw logs.

## Open questions

- Long transitive proximity chains need not produce the same groups as CPU's
  ordered grouping. Targeted chain tests cover demonstrated cases, not all inputs.
- Sparse pyramid pixels remain unimplemented; refinement footprints cannot be
  bounded by a fixed halo without accounting for iterative movement.
- Dense candidate populations can reverse the bounded-grid benefit; the capacity
  strategy remains an explicit option.
- Photographs do not establish deployment-camera recall, and shared-host timings
  do not establish isolated sustained throughput or a universal performance limit.
