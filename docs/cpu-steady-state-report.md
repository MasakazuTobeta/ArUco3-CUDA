# CPU steady-state optimization results

## Purpose

Push CPU detection throughput and latency beyond the previous CPU experiment,
while retaining the pinned OpenCV detector's outputs. Startup latency is not an
acceptance gate. This report records a measured optimization frontier, not a
proof that no faster implementation can exist.

## Scope

The host-only experiment targets DGX Spark GB10 and OpenCV 4.14.0, commit
`0654a42e19215ef25b1d367d822f3c630447e7c7`. Tracking, previous-frame detections,
pixel reuse across frames, temporal prediction, and skipped frames are excluded.
Retained allocation and immutable dictionary precomputation are allowed. Each
video frame is independently detected and validated.

The [previous report](cpu-benchmark-report.md) describes source snapshot
`cc216b8`, preserved in branch commit `e2bf432`. Its measurements remain
historical evidence. This follow-up uses a paired `previous` route with all new
switches disabled, as well as the actual OpenCV API and the new implementation.
Neither the installed CUDA API nor the main branch is changed.

## Current state

Generated measurement files referenced by local paths below are not version-controlled.
The report retains the results and conditions; raw outputs remain in local storage.

The selected defaults add filtered contours, sparse pyramids, cached refinement
and exact dictionary lookup. Spatial grouping remains disabled. Sparse tiles
have side eight, whole-level promotion ratio 0.5 and geometry work ratio 0.3.
The final implementation has passed the validation gates below. Repeated-process
latency, independent-frame throughput, resource use and final-source confirmation
are recorded separately. This is the selected measured frontier on GB10, not a
universal optimum across scenes or processors.

| Experiment | Mechanism | Important constraint |
| --- | --- | --- |
| Filtered contours | Reuse contour tracing storage and discard ineligible paths before materialization | Preserve path order and every polygon/quad decision |
| Fused contour preparation | Normalize directly into the retained padded center | Overwrite every input pixel each call; the zero guard is immutable and initialized on shape replacement |
| Sparse pyramid | Generate current-frame tiles demanded by decoding/refinement | Exact global image coordinates and border arithmetic; invalidate tile validity on every call |
| Geometry cost decision | Select sparse or full levels from current candidate geometry | No temporal state or changed detection thresholds |
| Cached refinement | Retain Gaussian weights and patch storage | Preserve iteration and interpolation arithmetic |
| Exact dictionary lookup | Precompute full scan answers for known definite words | Preserve first acceptable ID and rotation ties; ambiguous/unknown words retain the full scan |
| Spatial grouping | Conservative bounds before existing pair metrics | Retain ordering and tie behavior; measured separately |
| Independent-frame workers | One persistent detector per worker | All frames processed; service latency and saturated throughput reported separately |

The implementation details and numerical contracts are in the
[detector](../cpu/cpu_detector.md), [contour](../cpu/cpu_contours.md),
[sparse pyramid](../cpu/cpu_sparse_pyramid.md),
[grouping](../cpu/cpu_grouping.md), and
[video benchmark](../cpu/video_benchmark.md) sidecars.

### Source stages

The full primary, external, native, video, resource and stage measurements used
one frozen performance implementation. Final review then found a sparse-cache
recovery bug: an allocation exception could leave the new shape paired with an
old validity map. The repair marks the cache tag invalid before allocation and
publishes it only on success. It changes neither the existing-cache hot path
nor successful pixel arithmetic. The measured sparse source is
`ed06fec9707e08fc3d09cee0b3dc54bc3c3efe4849f0718727834ce1bf981b21`;
the repaired source is
`96ba75dce178a9cf856c77ac8f17219e6d794d9dde4a04121c87e28d95e43e88`.

The original measurements are preserved with `source-stage.json` labels. They
are not relabeled as executions of the repaired binary. Final verification and
the 44-scene quiet confirmation use the repaired source. The
[recovery validation](measurements/2026-11-17-cpu-steady-recovery-validation/README.md)
records the patch and checks; the
[orchestration record](measurements/2026-11-17-cpu-steady-reproduction/README.md)
explains the scheduling boundary and reproduction.

### Accuracy and validation

Exact agreement matches detections by ID and position, then requires identical
IDs, each marker's four ordered corners including implied rotation, and rejected
counts on the tested input/configuration. Marker-list ordering is not a separate
compatibility gate. This establishes no loss
relative to that OpenCV baseline on those cases. It does not establish universal
real-world accuracy or accuracy superior to OpenCV.

The follow-up adds a deterministic 96-scene degraded-marker corpus and
changing-frame/unchanged-pointer sequences, direct contour path checks, regional
pyramid/warp/refinement checks, configuration boundaries, dictionary lookup
ambiguity/tie cases, and reinitialization. The
[quality corpus sidecar](../tools/cpu_quality/quality_scenes.md) describes the
rendered ground truth and required positive controls.

Nine pinned `opencv_extra` regression/photo fixtures extend the synthetic
coverage. Their [download manifest](../cpu/experiments/opencv-fixtures.json)
records byte hashes and source commit; images remain local and are not
redistributed. These fixtures have no independently labeled detection ground
truth here. All 18 supported Dictionary families can be compared against the
same OpenCV configuration; zero-detection comparisons must not be described as
successful marker recognition.

The final-default external sweep passed 648 previous/optimized records:
nine images, 18 dictionaries, both ArUco3 settings, four OpenCV threads. The
optimized route had detections in 41 image/configuration pairs, totaling 278
detections across those pairs, with zero ID/corner/rejected-count differences. See the
raw conditions (local `measurements/2026-11-17-cpu-steady-final-external/environment.json`) and
summary (local `measurements/2026-11-17-cpu-steady-final-external/summary.json`). These one-sample
runs are accuracy checks only.

The final contour-preparation change also passed 68,536 exact path comparisons
against its predecessor. The [controlled diagnostic](measurements/2026-11-17-cpu-steady-padding/evidence.md)
records the initial code-placement confound, the shared-scanner comparison,
applied source hash and focused Release checks. Earlier complete measurement
directories carry `phase.json` labels and precede this last change. Primary static-image latency
figures use directories prefixed `cpu-steady-final-`; earlier timing must not be
presented as that source's timing.

The maintained suite has 46 cases. Before the allocation-recovery repair, all
passed in Release, native Release and Debug coverage. ASan/UBSan passed all 46
cases on the preceding contour-preparation source; its preparation delta passed
seven focused cases.
An additional cached-refinement test with ArUco3 disabled, including a strided
ROI whose sampling patch crosses its boundary, passed in all four builds.
CPU-module coverage is 96.2% lines, 67.8% branches and 100% functions. The
[validation record](measurements/2026-11-17-cpu-steady-validation/README.md)
explains source stages, two instrumented timeout reruns and uncovered branches.
The repaired final source passed all 46 Release cases, seven focused cases in
each of ASan/UBSan, native Release and coverage, and two controlled allocation
exceptions followed by exact current-frame recovery under sanitizers. Updated
coverage is 1,445/1,502 lines (96.2%), 1,373/2,024 branches (67.8%) and 50/50
functions. Only the changed sparse translation unit's old counters were cleared;
unchanged-module counters remain valid. This scoped final verification is
recorded separately in the recovery archive linked above.
OpenCV shared libraries are uninstrumented; this is not full-system coverage.
The [video CLI validation](measurements/2026-11-17-cpu-steady-video-validation/README.md)
checks worker lifecycle, injected failures and tuning options independently of
the video performance runs.

### Measurement conditions

The portable Release build uses GCC 13.3, `-O3 -DNDEBUG`, without fast-math or
machine-specific compiler tuning. The actual installed OpenCV detector is timed
through its public API. `previous` selects the earlier algorithm switches in the
same new executable; it is an ablation, not a rerun of an immutable historical
binary. Each route receives identical inputs and detector settings. Dictionary
lookup precomputation, detector construction, file decoding, validation and
warm-up are outside the steady-state timed calls. Output ID/corner creation is
inside; rejected outputs are checked separately outside timing. Stage profiling
is disabled in the primary latency runs.

One-thread runs use performance CPU 5; four-thread runs use CPUs 5–8. OpenCV's
thread setting and affinity are the same for all routes in a comparison. ASLR
remains enabled. The host power policy and unrelated host services are retained.
Raw governor/frequency snapshots are archived, but anomalous frequency readings
are not treated as proof of a fixed physical clock. Some primary measurements
overlap correctness/coverage work on efficiency CPUs 0–3. A representative
confirmation is also run after all such work ends; its scope is reported below.
No GPU work is performed, so CUDA kernel-only timing is inapplicable.

Each scene is measured in three independent processes. Tables use the median of
the three process p50 values, and speedups divide those scene medians. Aggregate
speedups are medians across scenes, not ratios of total elapsed time. Archived
summaries also contain per-scene p95/p99 and process spread; percentiles from
separate processes are not pooled. Empty-output scenes are separated from
positive detections because the former cannot establish decoding throughput.

| Corpus | Images | ArUco3 | Timed calls / warm-up per route and process | Dictionary |
| --- | ---: | --- | ---: | --- |
| Primary, historical plus extended marker sizes | 115 | On | 200 / 30 | MIP 36h12 |
| Degraded synthetic quality | 96 | On | 200 / 30 | MIP 36h12 |
| Extended full-resolution | 24 | Off | 50 / 10 | MIP 36h12 |
| Quality full-resolution | 96 | Off | 100 / 20 | MIP 36h12 |
| High-ID dictionary | 4 | On | 200 / 30 | 6x6 1000 |

Both thread budgets run every row. ArUco3-off timing uses no corner refinement;
accuracy tests additionally cover supported subpixel refinement. The high-ID
corpus contains IDs 984–999 and is deliberately a late-match dictionary
workload, not a representative ID distribution. Its
[generator](../cpu/experiments/generate_high_id_corpus.md) records inputs and
reproduction commands.

### Static-image latency

| Corpus | Threads | Positive / all scenes | Median speedup vs OpenCV, positive / all | Median speedup vs previous, positive | Candidate wins vs OpenCV, all | Candidate wins vs previous, all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| primary (local `measurements/2026-11-17-cpu-steady-final-primary-1t/summary.json`) | 1 | 48 / 115 | 2.453 / 4.055 | 1.168 | 115 / 115 | 95 / 115 |
| primary (local `measurements/2026-11-17-cpu-steady-final-primary-4t/summary.json`) | 4 | 48 / 115 | 2.433 / 3.648 | 1.198 | 115 / 115 | 96 / 115 |
| quality (local `measurements/2026-11-17-cpu-steady-final-quality-1t/summary.json`) | 1 | 66 / 96 | 2.250 / 3.223 | 1.124 | 96 / 96 | 93 / 96 |
| quality (local `measurements/2026-11-17-cpu-steady-final-quality-4t/summary.json`) | 4 | 66 / 96 | 2.452 / 2.920 | 1.077 | 96 / 96 | 76 / 96 |
| fullres (local `measurements/2026-11-17-cpu-steady-final-fullres-1t/summary.json`) | 1 | 24 / 24 | 1.770 / 1.770 | 1.099 | 24 / 24 | 17 / 24 |
| fullres (local `measurements/2026-11-17-cpu-steady-final-fullres-4t/summary.json`) | 4 | 24 / 24 | 1.715 / 1.715 | 1.135 | 24 / 24 | 20 / 24 |
| quality-fullres (local `measurements/2026-11-17-cpu-steady-final-quality-fullres-1t/summary.json`) | 1 | 76 / 96 | 2.869 / 3.612 | 1.041 | 95 / 96 | 96 / 96 |
| quality-fullres (local `measurements/2026-11-17-cpu-steady-final-quality-fullres-4t/summary.json`) | 4 | 76 / 96 | 3.014 / 3.502 | 1.060 | 95 / 96 | 94 / 96 |
| highid (local `measurements/2026-11-17-cpu-steady-final-highid-1t/summary.json`) | 1 | 4 / 4 | 2.324 / 2.324 | 1.125 | 4 / 4 | 4 / 4 |
| highid (local `measurements/2026-11-17-cpu-steady-final-highid-4t/summary.json`) | 4 | 4 / 4 | 1.942 / 1.942 | 1.087 | 4 / 4 | 4 / 4 |

All primary and quality ArUco3-on scenes are faster than OpenCV at p50, p95
and p99. The positive primary subset has median speedups of 1.168 and 1.198 over
the previous route at one and four threads respectively. These ratios do not
imply equal percentages of latency reduction. Several empty-output
and dense clean cases remain faster with the previous route.

With ArUco3 disabled, the 24 extended positive scenes all beat OpenCV, but seven
one-thread and four four-thread cases are slower than the previous route. The
largest previous-route regression is the one-thread 1080p/one-marker clean
scene: 1/0.920877 = 1.0859 times the previous latency (8.6% higher). The default
is therefore not a per-scene oracle that chooses the fastest historical route.


The full-resolution quality exception is `quality_negative_3.png`, a dense
checker pattern with no decoded marker. One-thread p50 is 132.074 ms for
OpenCV versus 139.714 ms optimized; four-thread p50 is 128.569 versus
138.589 ms. This case also loses at p95/p99 in both budgets. Its measured p50 is about 0.4% below `previous` at one thread and effectively
tied with it at four threads, but does not beat OpenCV. All other measured
quality full-resolution scenes beat OpenCV at those percentiles.

Across primary scenes, candidate process-spread median/max is 1.15%/8.85%
with one thread and 3.40%/8.30% with four. Small incremental differences should
be interpreted alongside these per-scene spreads, not as guaranteed gains.

### Representative detector calls

All values below are milliseconds. Each row has positive detections; the large
markers in the extended corpus remain above the ArUco3 lower bound. These are
detector-call latencies, not camera-to-result latencies.

| Scene | Threads | OpenCV p50 | Previous p50 | Optimized p50 | Optimized p95 | Optimized p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 720p, 4 clean | 1 | 0.741 | 0.319 | 0.304 | 0.321 | 0.334 |
| 720p, 4 noisy | 1 | 3.322 | 2.895 | 2.495 | 2.571 | 2.647 |
| 4K, 1 clean | 1 | 1.790 | 1.099 | 0.441 | 0.484 | 0.516 |
| 4K, 4 clean | 1 | 1.917 | 1.199 | 0.764 | 0.812 | 0.839 |
| 4K, 4 noisy | 1 | 5.990 | 5.266 | 4.084 | 4.165 | 4.219 |
| 4K, 16 clean | 1 | 2.456 | 1.626 | 1.634 | 1.714 | 1.739 |
| 720p, 4 clean | 4 | 0.436 | 0.186 | 0.170 | 0.199 | 0.210 |
| 720p, 4 noisy | 4 | 1.402 | 1.216 | 0.995 | 1.054 | 1.092 |
| 4K, 1 clean | 4 | 1.142 | 0.742 | 0.328 | 0.370 | 0.390 |
| 4K, 4 clean | 4 | 1.204 | 0.789 | 0.441 | 0.534 | 0.549 |
| 4K, 4 noisy | 4 | 2.770 | 2.488 | 1.690 | 1.762 | 1.799 |
| 4K, 16 clean | 4 | 1.402 | 0.988 | 0.980 | 1.043 | 1.076 |

### Independent-frame video throughput

Each resolution cycles through six different clean/noisy images with 1/4/16
markers, for 600 timed frames per route/layout and 36 warm-up frames. Three
independent processes alternate forward/reverse/forward route order. All use
the same four performance cores. Every timed result is checked, with independent
untimed rejected-count checks. This is a changing synthetic sequence, not a
camera capture or a tracked sequence.

| Resolution | Route | Layout | Frames/s | Service p50 ms | Service p95 ms | Service p99 ms |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1280x720 | opencv | 1 worker x 4 threads | 1010.1 | 0.891 | 1.526 | 1.579 |
| 1280x720 | opencv | 4 workers x 1 thread | 1779.2 | 1.442 | 3.883 | 3.919 |
| 1280x720 | cpu | 1 worker x 4 threads | 1581.7 | 0.495 | 1.102 | 1.162 |
| 1280x720 | cpu | 4 workers x 1 thread | 2820.8 | 1.439 | 2.578 | 2.630 |
| 3840x2160 | opencv | 1 worker x 4 threads | 482.3 | 2.536 | 2.939 | 3.049 |
| 3840x2160 | opencv | 4 workers x 1 thread | 890.4 | 2.921 | 6.889 | 7.048 |
| 3840x2160 | cpu | 1 worker x 4 threads | 841.4 | 1.356 | 2.062 | 2.122 |
| 3840x2160 | cpu | 4 workers x 1 thread | 1561.2 | 1.914 | 4.349 | 4.388 |

With independent workers, optimized throughput is 1.585x OpenCV at 720p and
1.753x at 4K under the same layout/core budget. Relative to its own intra-frame
layout, optimized throughput rises 1.783x/1.855x, while per-frame service latency
increases. The worker layout is therefore a throughput choice, not a claim of
lower individual-frame latency. FPS process spread is below 2.4% in all eight
cells.

The throughput interval runs from the first service start to the last completion
and includes dispatch gaps. Service percentiles exclude queue wait. Neither
metric includes camera acquisition, image decoding, detector setup or warm-up,
and reciprocal throughput must not be presented as frame latency. Outputs
retain frame indices even when worker completion order differs. Raw indexed
frames and execution metadata are in the
720p archive (local `measurements/2026-11-17-cpu-steady-video-1280x720/environment.json`) and
4K archive (local `measurements/2026-11-17-cpu-steady-video-3840x2160/environment.json`).

### Memory and initialization

Resource runs use a fresh process per route, six static scenes, both core
budgets, three repetitions, 30 warm-ups and 100 timed calls. RSS includes the
decoded input, OpenCV, shared-library pages and the process; no compatibility
oracle runs in that process. The ranges below are across the six per-scene
median peak RSS values, in MiB. They are not bounds on all possible inputs.

| Threads | OpenCV RSS range | Previous RSS range | Optimized RSS range |
| ---: | ---: | ---: | ---: |
| 1 | 21.32–23.95 | 21.32–24.69 | 21.32–23.13 |
| 4 | 21.32–28.75 | 21.32–28.70 | 21.32–26.59 |

For the 4K/four-marker clean scene at four threads, peak process RSS is
23.36 MiB OpenCV, 23.43 MiB previous and 26.59 MiB optimized. Sparse buffers
reserve full-size addressable planes; untouched reserved storage need not be
resident. Changing dimensions and four independent worker instances can have
different peaks. The video pool's combined peak RSS was not measured by this
single-detector resource probe.

Initialization is deliberately not an optimization gate. For the same 4K scene
at one thread, optimized setup is about 1.004 ms versus 0.042 ms previous; this
includes the immutable dictionary precomputation. Initial detection and total
time to first result are retained in the
resource records (local `measurements/2026-11-17-cpu-steady-resources/resources.jsonl`),
along with commands and provenance (local `measurements/2026-11-17-cpu-steady-resources/environment.json`).
Their isolated-process steady throughput is a separate experiment and must not
be substituted for the paired multi-scene latency table.

### Compiler tuning and remaining stage costs

The optional `-mcpu=native` build was compared over the 24 extended scenes with
80 samples and 15 warm-ups, in three pairs of independent processes with
alternating build order. Each process also timed actual OpenCV. The median
per-scene native speedup is 1.002 without normalization and 0.998 after
normalizing each build's gain against its own OpenCV baseline (range
0.992–1.003). No material gain was demonstrated, so portable compilation remains
the default. The native diagnostic (local `measurements/2026-11-17-cpu-steady-final-native/environment.json`)
and [native validation](measurements/2026-11-17-cpu-steady-final-native-validation/README.md)
retain flags, binary hashes and exact-output checks.

Stage profiles are separate one-process diagnostics (40 samples, 10 warm-ups).
They are not substituted into the primary timing claims. `trace` and `group`
are subintervals of candidate extraction; `decode` includes lazily generated
pyramid pixels. Stage medians need not sum to the total-call median.

| Four-marker scene | Threads | Total ms | Preprocess ms | Threshold ms | Trace ms | Group ms | Decode ms | Refine ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 720p clean | 1 | 0.312 | 0.025 | 0.082 | 0.091 | 0.003 | 0.087 | 0.020 |
| 720p noisy | 1 | 2.524 | 0.025 | 0.081 | 2.292 | 0.003 | 0.100 | 0.021 |
| 4K clean | 1 | 0.753 | 0.105 | 0.145 | 0.127 | 0.006 | 0.335 | 0.035 |
| 4K noisy | 1 | 4.059 | 0.122 | 0.143 | 3.395 | 0.007 | 0.351 | 0.036 |
| 720p clean | 4 | 0.171 | 0.013 | 0.052 | 0.059 | 0.004 | 0.026 | 0.011 |
| 720p noisy | 4 | 0.991 | 0.019 | 0.055 | 0.839 | 0.003 | 0.054 | 0.018 |
| 4K clean | 4 | 0.441 | 0.104 | 0.100 | 0.076 | 0.007 | 0.108 | 0.044 |
| 4K noisy | 4 | 1.687 | 0.118 | 0.107 | 1.212 | 0.007 | 0.206 | 0.039 |

In the noisy 4K one-thread diagnostic, tracing accounts for about 84% of the
call, and about 72% with four threads. Eliminating every other one-thread cost
would therefore yield only about 1.20x more speed under that profile; tracing
itself is not a proven lower bound. On clean 4K input the remaining cost is
distributed across preprocessing, thresholds and decoding/pyramid sampling.
Further large gains would need another measured tracing/sampling implementation
that preserves the same outputs. A different CPU, OpenCV HAL, workload or memory
budget can change this frontier. See the
one-thread profile (local `measurements/2026-11-17-cpu-steady-stages-1t/run-1.jsonl`) and
four-thread profile (local `measurements/2026-11-17-cpu-steady-stages-4t/run-1.jsonl`).

### Confirmation after validation

The final repaired binary repeats 44 scenes after all other agent build/test
processes finish: all 24 extended marker scenes, four blank resolutions, and
the first/last manifest entry from each of eight quality families. Selection
uses scene definitions, not observed speed. Each thread budget again uses three
processes, 200 samples and 30 warm-ups. Inputs are hash-identical to their
corresponding primary/quality scenes.

| Threads | Positive / all scenes | Median speedup vs OpenCV, positive / all | Candidate wins vs OpenCV | Median candidate latency ratio to matched primary/quality scene |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 34 / 44 | 1.857 / 2.307 | 44 / 44 | 1.00430 |
| 4 | 34 / 44 | 1.782 / 2.267 | 44 / 44 | 0.99739 |

The last column compares the same images rather than comparing medians of
different scene sets. Candidate per-scene latency ratios range from 0.995–1.020
with one thread and 0.953–1.049 with four; median changes are +0.43% and −0.26%.
Median changes in paired OpenCV speedup are +0.25%/−0.26%. The substantial
OpenCV advantage therefore persists after the repair and validation workload
ends. Positive-subset aggregate speedups differ from the 48-scene headline
because this subset has a different composition, including all dense extended
marker scenes.

This is representative confirmation, not a causal isolation experiment: the
subset/order changes allocation history, a recovery-only source change was
applied, and host services remain. OpenCV's per-scene p50 changes as much as
+24.6% on the 720p blank scene in one thread (+23.0% in four). Those baseline
outliers are retained, not attributed solely to background load or the repair.
The candidate still does not beat `previous` in every scene. See
one-thread results (local `measurements/2026-11-17-cpu-steady-quiet-1t/summary.json`),
four-thread results (local `measurements/2026-11-17-cpu-steady-quiet-4t/summary.json`),
and the selection manifest (local `measurements/2026-11-17-cpu-steady-quiet-1t/selection.json`).

### Rejected approaches and remaining costs

- A two-pass contour tracer first counted points, then retraced eligible paths.
  It passed exact path checks but traded small noisy/downscaled improvements for
  full-resolution and clean-scene regressions. The
  [standalone probe](../cpu/experiments/count_first_probe.cpp) retains this
  experiment outside the production build.
- A cursor into preallocated contour storage did not materially improve on
  retained vector capacity.
- A first regional-pyramid design invoked tiny OpenCV operations too often and
  regressed on dense scenes. Batched tile rectangles and a current-frame work
  estimate address that cost; both strategies still use identical pixels.
- Fixed small regional-refinement halos are unsafe because refinement can move
  outside the initial neighborhood. The retained implementation requests each
  actual sampling footprint.
- A manually reconstructed nearest-neighbor warp differed by a pixel on some
  fractional transforms. The retained helper delegates the actual global warp
  to OpenCV after ensuring all potentially sampled pixels are current.
- Scalar and SIMD running-sum threshold prototypes were slower than the existing
  cache-budgeted integral path in the sampled sizes; neither is selected.

The diagnostic manifest (local `measurements/2026-11-17-cpu-steady-diagnostics/manifest.json`)
labels intermediate measurements, their limitations, and file hashes. These
instrumented exploratory figures are not the final end-to-end speed claims.

## Goals

- Pass exact-output comparisons before selecting performance defaults.
- Measure actual OpenCV, the previous CPU implementation, and the final CPU
  implementation on identical images, settings, and core budgets.
- Preserve per-scene p50/p95/p99, independent process variation, unfavorable
  results, memory use, source/binary hashes, and environment details.
- Separate single-frame latency, stage diagnostics, and independent-frame
  throughput. CPU kernel-only timing is not applicable.

The subsequent [real industrial background evaluation](cpu-real-background-report.md)
measures this unchanged repaired implementation on public warehouse/factory
photographs. It is separate from the synthetic and external-compatibility
measurements above.

## Open questions

- The four-core video pool's combined peak memory and behavior under actual camera
  arrival/queue timing remain unmeasured; the reported throughput is saturated.
- Performance and exact behavior on Jetson Orin/Thor and x86_64 are unverified.
- Real deployment camera sequences and independent annotated real-image accuracy
  are not established by the synthetic and regression fixtures.
- Remaining gains depend on contour density, marker size/count, dictionary,
  available cores, and OpenCV/compiler/HAL configuration.
