# CPU optimization results

## Purpose

Evaluate an independently assembled CPU detector against the pinned OpenCV
implementation, using the GPU implementation's scheduling and representation
findings. Report the measured improvements and the remaining bottlenecks without
claiming a universal CPU speed limit.

## Scope

This report concerns branch `experiment/cpu-optimization` and the host-only
implementation in [`cpu/`](../cpu/cpu_detector.md). The primary implementation
retains contour-based candidate generation and OpenCV detection semantics.
The connected-component/extrema route is a separate, opt-in algorithm experiment.
There is no CUDA execution; kernel-only time is not applicable.

The available machine is DGX Spark GB10. Jetson Orin, Jetson Thor, and x86_64 were
not available for this experiment. Evaluation uses synthetic inputs; no real-image
speed or accuracy claim follows from these results.

## Current state

Generated measurement files referenced by local paths below are not version-controlled.
The report retains the results and conditions; raw outputs remain in local storage.

This historical evaluation's final detector source is snapshot `cc216b8`.
The [steady-state follow-up](cpu-steady-state-report.md) records subsequent
optimization work; its results must not be mixed with this snapshot's figures.
The default shared threshold
operation uses a 2 MiB prefix budget. The earlier `090c01c` measurements are
retained to document the initial optimization and the memory improvement.

### Measurement conditions

- OpenCV 4.14.0, commit `0654a42e19215ef25b1d367d822f3c630447e7c7`, Release build.
- GCC 13.3.0, C++17, `-O3 -DNDEBUG`; architecture tuning is off unless explicitly
  stated. Fast math is not enabled.
- The one-thread comparison pins both implementations to performance CPU 5.
  The four-thread comparison uses the same performance-core budget, CPUs 5-8.
  Both use the requested OpenCV thread count.
- ArUco3 is enabled for the primary comparison: canonical minimum side 32,
  original-image ratio 0.05, threshold windows 3/13/23, error correction 0.6,
  `DICT_ARUCO_MIP_36h12`, and subpixel refinement. Full-resolution comparisons
  disable ArUco3 and subpixel refinement together.
- The corpus contains the existing 91 `full` scenes plus 24 scenes with larger,
  detectable markers: four resolutions, 1/4/16 markers, clean/noisy variants.
  Of the 115 scenes, 48 have detections and 67 have none under the primary
  settings. Empty-scene gains must not be presented as decoding gains.
- Images are loaded and hashed outside the measured interval. The same image
  storage is passed to both routes. Timed OpenCV calls request IDs and corners,
  without rejected-corner output; rejected counts are validated separately.
- Primary latency: 30 warmups and 200 samples per scene, three independent
  processes, alternating route order. Report the median of independent process
  p50 values. p95/p99 summaries are likewise medians of the process percentiles,
  not pooled percentiles. Sequential throughput uses a separate 100-frame loop.
- Profiling is disabled for primary timings. Diagnostic stage runs are separate
  and include their instrumentation overhead.
- CPU frequency policy and ASLR are retained. Frequency snapshots, affinity,
  command lines, build flags, source hashes, binary hash and OpenCV provenance
  are in each measurement directory's `environment.json`. Other host services
  remain active; these are paired measurements, not an isolated-machine claim.
- Some raw `scaling_cur_freq` snapshots exceed the reported policy maximum
  (for example 7,150,000 versus 3,900,000 kHz). They are retained as kernel
  reports, not evidence of a fixed physical clock. The reporting discrepancy's
  cause was not established.

### One-thread results

The primary three-process comparison improves all 115 scenes. Among the 48 scenes
with detections, the median per-scene speedup is **2.04x** (range **1.13-3.38x**).
Across all scenes it is 3.90x, but that larger figure includes 67 empty scenes.
The tables use extended scenes named `cpu_<resolution>_n4_<clean|noise>.png`.

| Scene, four markers detected | OpenCV p50, ms | CPU experiment p50, ms | Speedup |
| --- | --- | --- | --- |
| 1280x720, clean | 0.727931 | 0.313038 | 2.33x |
| 1280x720, noise | 3.255176 | 2.842906 | 1.15x |
| 3840x2160, clean | 1.869378 | 1.142519 | 1.64x |
| 3840x2160, noise | 5.837862 | 5.112025 | 1.14x |

At 1280x720 clean, OpenCV/experiment p95 is 0.742250/0.320637 ms and p99 is
0.752026/0.328718 ms. Independent-process spread is recorded for every scene. Per-scene
tails, throughput and input hashes are retained in the raw measurements.

Raw data: one-thread measurements (local `measurements/2026-11-17-cpu-final-1t/environment.json`)
and aggregate (local `measurements/2026-11-17-cpu-final-1t/summary.json`).

### Four-thread results

With four OpenCV threads on the same four performance cores, the median speedup
among the 48 scenes with detections is **2.10x** (range **1.01-3.03x**). All 115
median point estimates favor the CPU experiment; the smallest improvement is
close to a tie: the 1.37% minimum gain is smaller than the corresponding
3-4% independent-run spread. The initial `090c01c` comparison also had one
p99 0.22% worse than OpenCV at `cpu_1920x1080_n16_noise.png`; that result is
retained in the earlier raw data. Tail outcomes at this scale are sensitive to
measurement variation; all-condition superiority is not claimed.

| Scene, four markers detected | OpenCV p50, ms | CPU experiment p50, ms | Speedup |
| --- | --- | --- | --- |
| 1280x720, clean | 0.416653 | 0.180734 | 2.31x |
| 1280x720, noise | 1.334423 | 1.146569 | 1.16x |
| 3840x2160, clean | 1.121224 | 0.721195 | 1.55x |
| 3840x2160, noise | 2.643949 | 2.361871 | 1.12x |

Raw data: four-thread measurements (local `measurements/2026-11-17-cpu-final-4t/environment.json`)
and aggregate (local `measurements/2026-11-17-cpu-final-4t/summary.json`).

### Full-resolution strategy

This separate sweep disables ArUco3 and subpixel refinement in both routes. It
uses all 24 extended clean/noisy scenes, 10 warmups, 50 latency samples and three
independent processes. Sequential throughput is not measured in this sweep.
With 50 samples, nearest-rank p99 is the sample maximum; these tails have less
resolution than the primary 200-sample comparison.

The median speedup across 24 scenes is **1.62x** at one thread (range
**1.07-3.86x**) and **1.57x** at four threads (range **1.01-4.00x**).
All p50 point estimates favor the experiment, while the smallest four-thread
result is again a near tie.

| Threads | Extended scene, four markers | OpenCV p50, ms | CPU p50, ms | Speedup |
| --- | --- | --- | --- | --- |
| 1 | 1280x720, clean | 4.877478 | 1.530742 | 3.19x |
| 1 | 1280x720, noise | 44.749883 | 40.827699 | 1.10x |
| 1 | 3840x2160, clean | 39.200128 | 10.724615 | 3.66x |
| 1 | 3840x2160, noise | 339.966133 | 309.714270 | 1.10x |
| 4 | 1280x720, clean | 3.199587 | 0.933444 | 3.43x |
| 4 | 1280x720, noise | 17.960777 | 16.353463 | 1.10x |
| 4 | 3840x2160, clean | 25.443972 | 7.666066 | 3.32x |
| 4 | 3840x2160, noise | 147.365122 | 132.922019 | 1.11x |

Unfavorable tails remain: at four threads `cpu_640x480_n16_noise.png` has p95
2.0% and p99 1.0% worse than OpenCV; `cpu_640x480_n1_noise.png` has p99 4.9%
worse despite a better p50. Noise-heavy contour extraction limits the gain.
Raw data and aggregates: one thread (local `measurements/2026-11-17-cpu-fullres-1t/environment.json`),
one-thread aggregate (local `measurements/2026-11-17-cpu-fullres-1t/summary.json`),
four threads (local `measurements/2026-11-17-cpu-fullres-4t/environment.json`),
four-thread aggregate (local `measurements/2026-11-17-cpu-fullres-4t/summary.json`).

### Setup, throughput and process memory

The isolated resource probe creates only the selected detector; it does not run
a compatibility oracle in the same process. Its setup-to-first-result interval
starts **after PNG decoding and OpenCV thread-policy selection**. It excludes
executable/library startup and image decoding, so it is not camera-to-result or
whole-process cold-start latency.

For the ArUco3-enabled 720p clean scene at one thread, the median of three
independent processes gives setup-to-first-result 1.220 ms for OpenCV and
0.956 ms for the experiment. The first-result advantage is smaller than the
steady-state latency advantage. At four threads on the 720p noisy scene the
first result is unfavorable: 2.352 ms for OpenCV versus 2.609 ms for the
experiment. First-frame improvement is not universal.

Primary throughput is measured separately in 100-frame serial loops; its raw
per-process values remain alongside the latency records. The isolated probe
also measures serial throughput, but those separate-process values must not be
substituted for the paired primary comparison.

Peak RSS below is the **lifetime process high-water mark**, including PNG
decoding, input storage, OpenCV, detector setup, 30 warmups and subsequent
frames. It is not retained workspace size or a detector-only allocation delta.
ArUco3 resource rows use three independent processes; full-resolution 4K stress
rows use one process per route and scene.

| Mode and scene, one thread | OpenCV peak RSS, MiB | Private control, MiB | Final CPU, MiB |
| --- | --- | --- | --- |
| ArUco3, 720p clean | 12.28 | 12.41 | 12.68 |
| Full resolution, 4K clean | 34.71 | 58.22 | 60.43 |
| Full resolution, 4K noise | 98.19 | 114.04 | 117.14 |

The initial optimized snapshot used 82.31 MiB on full-resolution 4K clean and
139.02 MiB on noise. Cache-budgeted stripes reduce those to 60.43 and
117.14 MiB. The final detector still uses more memory than OpenCV in these
one-thread full-resolution cases: it retains the threshold planes and reusable
buffers. The 2 MiB option bounds only the integral-owner payload, not total
process memory. Disabling shared thresholding also does not promise OpenCV's
allocation lifetime.

Final resource commands, scope and binary hash (local `measurements/2026-11-17-cpu-resources/environment.json`),
raw resource rows (local `measurements/2026-11-17-cpu-resources/resources.jsonl`),
resource aggregate (local `measurements/2026-11-17-cpu-resources/summary.json`), and
initial resource record (local `measurements/2026-11-16-cpu-resources/environment.json`)
retain both favorable and unfavorable cases.

### Retained implementation

- A shared, vectorized integral-image threshold operation reuses prefix sums
  across windows and avoids normalized mean images. Safe stripes retain the fast
  path on 4K images without signed 32-bit prefix overflow. The default 2 MiB
  prefix budget also bounds the cache working set. Padding is performed per
  stripe, and short last stripes use views into reusable owned storage.
- Pyramid dimensions are planned up front; pixel levels are generated only when
  needed, with reusable storage and the same OpenCV pixels.
- Packed dictionary comparisons preserve ambiguous cells, first qualifying ID,
  and rotation ties. Cell sampling reuses storage and avoids per-cell headers.
- Independent threshold windows are extracted in parallel and merged in the
  original order. Identification runs within a containment depth using reusable
  TLS workspaces, joins before ancestor suppression, and refines into fixed
  output slots. Single-thread and singleton work avoid parallel dispatch.

### Rejected and optional experiments

The one-thread ablation (local `measurements/2026-11-16-cpu-ablation-1t/environment.json`)
measures the switches independently on all 24 additional scenes. It is a
single-process diagnostic, not the source of the primary speedup figures.
Compressed-contour reconstruction, component prefiltering, Otsu dry run and
direct component candidates remain explicit options rather than assumed wins.
The [threshold sidecar](../cpu/cpu_threshold.md) records further rejected loop
nesting, native-helper and whole-image 64-bit-integral alternatives.

The initial one-thread ablation used profiling, 20 warmups and 80 measurements
in one process on 24 extended scenes. Its median speedups relative to the
unoptimized private control were 1.009x for lazy pyramids, 1.016x for packed
matching, 1.012x for fused cells and 1.271x for shared thresholds. The smaller
decode changes were still useful in some scenes (up to 1.118x and 1.067x).
Their composition and empty-scene behavior are measured by the primary sweep,
rather than inferred by multiplying these ratios.

Compressed-contour reconstruction had a median ratio of 0.899x relative to
the control; foreground-component pruning was 0.816x, and combining them was
0.770x. Ratios below one mean slower. Otsu dry run did not establish a useful
increment over fused cells. These options remain disabled by default.

The initial native-tuning trial (local `measurements/2026-11-16-cpu-native-1t/environment.json`)
used three processes on the 24 extended scenes. Normalizing each build's speedup
by its paired OpenCV result gave a native/generic median of 0.997x, ranging from
0.967x to 1.106x. It did not establish a consistent benefit; native tuning stays
an explicit, disabled-by-default option. This is an initial-snapshot trial,
not a final native-build performance claim.

The component/extrema trial (local `measurements/2026-11-16-cpu-components-1t/environment.json`)
used all 115 scenes, one process and stage profiling. It differs from OpenCV
on 14 scenes under the strict comparison, including an ID-set difference and
an 8.60 px corner difference. It is slower than OpenCV on 37 scenes. For example,
720p/four-marker clean was 0.733/0.907 ms for OpenCV/components; noise was
3.315/1.885 ms. Faster noise processing does not establish acceptable accuracy;
this route is not selected automatically.

Dual foreground-eight/background-four component pruning was also rejected.
The candidate sets and ordered contour paths passed 69,918 exploratory and
526,336 exhaustive small-image comparisons, but noisy 4K extraction regressed
from 306.8 to 473.3 ms including pruning. Clean-scene regressions were larger.
The [standalone archived probes](../cpu/experiments/README.md) preserve the code,
conditions, raw timings and limitations; these are not shipped detector paths.

The final cache refinement was accepted for its memory and timing behavior.
Three-process paired threshold probes reduced 4K scratch from 25,470,440 to
2,526,272 bytes, with p50 8.109/7.517 ms at one thread and 5.769/4.845 ms at
four threads (legacy/budgeted). Small segmentation images remain one stripe.
The [raw cache sweep and paired confirmation](measurements/2026-11-17-cpu-cache-stripe/README.md)
include rejected stripe sizes and working-set choices. Whole-detector RSS is
measured separately from this scratch payload.

### Validation

Release validation passes all 29 CPU tests; all 29 also pass in the Debug
coverage build. The checks include:

- 91 scenes x ArUco3 on/off x three implementations x 1/2/4 threads: exact IDs,
  ordered float corners and rejected counts, with 1,638 candidate calls.
- 512 combinations of compatible optimization switches, 17 dictionary families and quarter rotations,
  duplicate IDs, non-contiguous ROIs, changing sizes and same-pointer new pixels.
- Pixel-level threshold comparisons including 4K accumulator stress, odd sizes,
  one-pixel dimensions, maximum windows, fractional constants and half boundaries.
  Prefix budgets from tiny fallback values through `SIZE_MAX`, repeated budget
  changes and input/output aliasing exercise the final cache implementation.
- Moved-from/reinitialized states, invalid inputs, ambiguous cells and a first-ID
  case where ID 0 at distance 3 must beat a later ID at distance 0.

The initial implementation passed a full 25-test ASan/UBSan run (including the
complete corpus matrix) and the subsequently added packed-cell regression, with
leak detection enabled. After the cache refinement, all nine threshold tests and
the public detector budget-reinitialization test pass under ASan/UBSan and the
native build. This is focused verification of the changed implementation, not a
claim of a single full 29-test sanitizer run. The
initial validation (local `measurements/2026-11-16-cpu-validation/validation.json`) and
final cache validation (local `measurements/2026-11-17-cpu-validation/validation.json`)
retain commands, logs and source hashes.

The CPU library has no CUDA dependency in its dynamic linkage. Both its
standalone build and its optional integration with the regular CMake project
compile. Formatting, public-header documentation and relative-link checks pass.

CPU-module coverage (excluding the benchmark and resource CLI) is currently
94.4% lines, 100% functions and 66.3% branches. Remaining gaps include failure
handlers, unexercised validation branches, unreachable post-validation guards, initializer mapping and
C++ exceptional edges; this is not a claim of complete branch coverage.
Gcov counters are updated atomically to support instrumented OpenCV workers.

CPU coverage was collected through the standalone CXX-only project with
`-DARUCO3CUDA_ENABLE_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug`, followed by its full
CTest suite and a `gcovr` filter for `cpu/` (excluding the benchmark, resource
probe and archived experiments). The CUDA coverage preset remains a separate
GPU-suite route; it is not evidence for the new CPU module.

### Reproduction

The standalone build requires OpenCV 4.14 and GoogleTest. It does not require a
CUDA toolkit or device:

```bash
cmake -S cpu/standalone -B build/cpu-opt -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpu-opt -j
ctest --test-dir build/cpu-opt --output-on-failure --schedule-random -j 4
build/cpu-opt/cpu/aruco3cuda_cpu_benchmark --input-dir data/cpu-experiment \
  --generate --iterations 1 --warmup 0 --variants opencv,optimized
python3 cpu/run_experiments.py --binary build/cpu-opt/cpu/aruco3cuda_cpu_benchmark \
  --input-dir data/cpu-experiment --output-dir results/cpu-local-1t \
  --cpu-list 5 --threads 1 --runs 3 --warmup 30 --iterations 200 \
  --throughput-frames 100
python3 cpu/summarize_experiments.py results/cpu-local-1t
```

Use the environment's matching performance-core numbers. For four threads use
`--cpu-list 5,6,7,8 --threads 4` on the measured DGX Spark. For the full-resolution
sweep use `--aruco3 0 --filter cpu_`, replace the warmup/iteration counts with
10/50, and omit throughput.
Do not overwrite an existing result directory; the runner refuses to replace
recorded process outputs.

## Goals

- Select defaults from compatible, end-to-end measurements on common inputs and
  CPU budgets; retain unfavorable conditions and rejected experiments.
- Keep raw results and exact source/build fingerprints sufficient to reproduce
  each comparison.
- Stop at the observed optimization frontier for this machine and corpus,
  documenting the stage that remains dominant and optional accuracy tradeoffs.

## Open questions

- Results on real images and on the other target machines remain unmeasured.
- Synthetic output equality does not prove equality for all possible images.
- Remaining coverage gaps are recorded above; 100% C0/C1 has not been achieved.
- CPU power consumption and application-level camera-to-result latency are not
  measured. The reported interval begins with an already decoded grayscale image.
