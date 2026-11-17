# GPU steady-state optimization

## Purpose

Push GPU detection latency toward a measured practical limit while preserving
detection quality at least equivalent to the CPU baseline. The user explicitly
allows different corner coordinates when detection quality is equivalent or
better. Initialization latency is not an acceptance gate.

## Scope

Start from merged `main` at `badb673` and compare the unchanged GPU detector,
pinned official OpenCV 4.14.0 and the optimized CPU implementation. Maintain
device-resident detection; report host-input/result latency separately. No
tracking, skipped frames, previous-frame image/result reuse, reduced precision,
relaxed detection thresholds or reduced verification corpus are permitted as
speed shortcuts. Reusable storage and immutable precomputation are allowed.

Generated measurements, builds and exploratory probes remain in ignored local
storage, following the [measurement policy](measurements/README.md).

## Current state

Implementation and evaluation are complete in the
[steady-state report](gpu-steady-state-report.md), which records quality gains,
conditional speed gains and regressions. This is not a universal speedup claim. A frozen
baseline source export and Release build are retained locally under
`build/gpu-study/` and `build/gpu-baseline/`. The baseline export precedes every
optimization edit.

At the start of the experiment, the GPU already reused workspaces, packed
dictionary words, parallelized corner refinement and captured the launch
sequence in a CUDA Graph. CPU gains from those mechanisms cannot be counted
again. The initial hypotheses were:

| Experiment | Work to remove | Required guard |
| --- | --- | --- |
| Shared multiwindow threshold | Repeated input traversal, row-sum storage and launches | Exact binary planes, including borders, rounding and large-window fallback |
| Deferred and bounded pyramid | Levels built before any candidate requests them | Device-side requests, unchanged level selection and per-frame freshness |
| Sparse pyramid | Pixels outside actual canonical/refinement footprints | Exact global coordinates and dependencies; full route for dense geometry |
| Earlier component rejection | Quad passes for provably ineligible components | Conservative proof for the GPU component algorithm, not assumed CPU contour rules |
| Exact dictionary lookup | Full scans for known definite words | First acceptable ID and rotation ties; ambiguity/miss fallback |
| Fused short stages | Graph nodes and intermediate traffic | Preserve ordering, synchronization, error and overflow behavior |

The existing GPU extractor differs from the CPU contour extractor. Its prior
91-image evaluation does not prove quality on the newer degraded-marker corpus
or real photographs. Establish that baseline before accepting speed changes.

The frozen baseline reproduced the historical 91-image result (88 true positives
and no false positives for each route). The expanded 96-scene quality corpus
exposed a substantial existing gap: with ArUco3 enabled, CPU detects 113 of 136
true markers and GPU detects 56, both without false positives. Disabled results
are 124 versus 65. Graph and direct GPU outputs agree over all 118 calls of the
changing-frame sequence, so this is not stale graph state. The optimized CPU
matches official OpenCV exactly throughout.

Component/contour diagnosis finds multiple marker borders connected through
background lines. CPU recovers them through separate hole contours; the current
GPU's one quadrilateral per connected component loses those alternatives. This
requires investigating the contour approach deferred by
[ADR-0003](adr/0003-candidate-extraction-approach.md) before claiming CPU-equivalent
quality. Faster execution of the unchanged extractor does not satisfy the goal.

All 16 industrial photographs return zero detections on all three routes in
both modes. These are negative compatibility checks, not positive recall
evidence. Local baseline audit records are `build/gpu-study/quality-baseline-*.jsonl`.

The device-contour prototype and corrected Euclidean-perimeter grouping now
recover all CPU-detected truths in the 96-scene corpus. Unique-scene true-positive
counts are CPU 113 / GPU 118 with ArUco3 on, and CPU 124 / GPU 127 with it off,
with zero false positives and no per-truth corner-error degradation on shared
detections. The default 118-call sequence also passes on Orin. The larger
external matrix exposed missing close-contour retries; a device-only retry
prototype passes the identified critical cases and the complete external matrix.
See [ADR-0004](adr/0004-device-contours-for-cpu-quality.md).

The final combined steady preset passes the Spark, Orin and RTX 4090 quality
gates, each with 2,398 audit
calls (38 matrix cases, 11 configuration variants and targeted close-marker
chains) plus 182 historical-image evaluations across both ArUco3 modes.
Spark GPU observations and ground-truth metrics match the preceding validated
control exactly. External compatibility recovers all 278 CPU detections across
17 Dictionaries and both modes; all 16 industrial photographs remain empty.
The worst positive common-truth corner delta in configuration probes is about
3.1e-6 pixels, below the explicitly recorded 0.001-pixel numerical classification
threshold on ARM (the x86 maximum is 6.1e-5 pixels). These results use 16,384
candidates and 1,024 markers. See the [final evaluation record](gpu-steady-state-report.md).

The selected candidate combines exact fused thresholding, requested pyramid
levels, density-selected Otsu, bounded 512-block candidate stages, parallel
contour counting, warp approximation/materialization with eligible-edge jump
preparation, and immutable Kleidi-only resize coefficients. Explicit ARM HAL
profiles preserve pinned OpenCV rounding: SVE128/SEPARATE on Spark and NEON/FUSED
on Orin. GENERIC was separately checked on RTX 4090. Named CMake strategies and
compiled metadata retain comparison controls and do not silently identify
hardware-specific tuning as a common algorithm.

Adversarial review then found two inherited lifecycle defects: stream changes
did not order pending work, and failed reinitialization after dictionary arena
growth could retain stale state. An owned completion event and explicit
resource-transition invalidation address these defects. Six isolated fault
fixtures pass on all three GPUs. The final combined validation and timings use
the subsequent source freeze, including all completion-event API overhead.


Resource limits are part of the measured configuration. A full-resolution
checkerboard generates 5,460 raw quad candidates, exceeding the 4,096 default.
The broader audit uses an explicit 16,384 budget and retains the original
overflow as a resource-limit result. Silent truncation is never quality success.

### Work and validation order

1. Audit baseline detection quality on the existing full synthetic corpus, the
   CPU degraded-marker corpus, dictionary rotations, real industrial backgrounds
   and external compatibility inputs. Distinguish rendered ground truth from
   agreement with CPU on unannotated photographs. Record every mismatch.
2. Record stage/event and whole-call latency, candidate counts, memory use and
   environment. Keep diagnostic profiling separate from primary timings.
3. Implement independent ablations. Validate them before comparing speed and
   retain explicit control routes. Resolve baseline quality deficiencies rather
   than redefining the acceptance corpus around existing GPU successes.
4. Compare identical prepared pixels/settings in independent processes, with
   warm-up, alternating measurement order and p50/p95/p99. Cover zero-candidate,
   zero-detection textured, clean-marker, degraded-marker and dense-candidate
   cases, small and large images, and both supported strategy configurations.
5. Validate at least DGX Spark and Jetson Orin; extend to available Thor and x86
   GPUs. Use the same algorithm and explicit tuning options. Record existing
   background workloads; do not stop unrelated services or treat busy-host
   observations as isolated-hardware limits.
6. Run appropriate randomized tests, changing-frame/layout/stream tests and
   Compute Sanitizer. Preserve baseline GPU outputs for arithmetic-preserving
   changes and independently compare CPU/ground-truth quality.
7. Publish measured results, conditions, rejected alternatives and remaining
   bottlenecks. Update design/evaluation/README records and prepare a separate
   reviewable PR. Do not commit raw outputs.

### Quality evidence

Check missed CPU-detected ground-truth markers, false positives, ID and rotation,
and corner errors against rendered truth. Report per-scene failures as well as
aggregates; gains on easy scenes must not conceal losses on difficult ones.
For photographs lacking annotations, CPU agreement is compatibility evidence,
not a field-recall measurement. Every proposed optimization also needs direct
before/after GPU checks, including changing pixels at unchanged input addresses.
Corner-error tolerances and any existing discrepancy require explicit review
before a changed-output algorithm is accepted.

## Goals

- Improve steady-state GPU speed without reducing CPU-equivalent quality.
- Include empty industrial backgrounds in the performance acceptance cases.
- Investigate dominant remaining costs and report diminishing returns or rejected
  alternatives; do not describe the first improvement as a universal limit.
- Keep kernel execution, device-resident latency, transfers/result retrieval and
  independent-frame throughput distinct.

## Open questions

- The existing component/extrema extractor fails the expanded quality gate;
  it remains an explicit historical performance control.
- Which CPU-inspired optimizations survive GPU scheduling and memory costs?
- Several available GPUs currently run substantial unrelated workloads. Final
  claims must state the measured load conditions and repeatability limits.
- How much sparse bookkeeping is worthwhile on each supported GPU, and how can
  refinement footprints be bounded without assuming iterations stay near their
  initial position?
