# ADR-0004: Use device contours to recover CPU-equivalent candidate coverage

- Status: Accepted for explicit steady presets; historical fresh defaults retained
- Date: 2026-11-17
- Revisits: [ADR-0003](0003-candidate-extraction-approach.md)

## Purpose

Preserve CPU-equivalent detection quality while optimizing steady-state GPU
latency. A faster implementation that misses CPU-detectable true markers does
not meet the current requirement. Different corner coordinates are acceptable
when detection quality and geometric accuracy are equivalent or better.

## Scope

Candidate extraction, grouping geometry, bounded device storage, and asynchronous
failure reporting. Preserve device-resident processing and current-frame
independence. No CPU contour fallback, host candidate-count synchronization,
tracking, or relaxed detection thresholds are introduced.

## Current state

The merged component-extrema route at `badb673` reproduces the historical
91-image evaluation, but the expanded 96-scene corpus exposes a gap. With ArUco3
on, official and optimized CPU detect 113 of 136 true markers; GPU detects 56.
With ArUco3 off, the counts are 124 and 65. All have zero false positives on these
synthetic scenes. Graph and direct execution agree, excluding stale graph state
as the explanation. These figures are not field-recall measurements.

Dark background lines can connect several marker borders into one foreground
component while leaving independent marker-shaped hole contours. One quadrilateral
per foreground component cannot represent those alternatives. CPU contour tracing
recovers them. The earlier assumption that joined components necessarily prevent
both routes from recovering the markers is therefore insufficient.

The selectable `ORDERED_CONTOURS` route partitions oriented foreground/background
edges into cycles, reconstructs exact oriented pixel paths, approximates closed
polygons, and emits the existing candidate representation. It preserves
8-connected foreground and 4-connected background at diagonal contacts. No
host tracing or per-frame allocation is used. Its approximation helper retains
the pinned OpenCV file's BSD notice and provenance; see the
[module design](../../src/core/contour_candidates.md).

The first integrated 118-call changing-frame audit had no lost CPU-detected
truths, false positives, or ID/rotation regressions. That intermediate version
failed the complete quality gate because four unique scenes had worse corner error. A diagnostic
replay identifies candidate grouping as one cause: CPU ranks by quadrilateral
Euclidean perimeter, while GPU used contour point count. In `projective_9`,
inner and outer paths both contain 202 points, but their quad perimeters differ.
Starting refinement from the inner quad reproduces every GPU output float;
starting from the outer quad reproduces every CPU output float. The grouping
correction preserves contour length for sampling and uses a separate quad
perimeter for ranking and proximity tests. After correction, the default
118-call audit passes on Spark and Orin with no common-truth corner degradation.
On the 96 unique scenes, Spark detects 118 versus CPU's 113 true markers with
ArUco3 on, and 127 versus 124 with it off, with zero false positives.

The broader external compatibility matrix then isolated missing CPU close-contour
retries. `failmask9` requires an alternative contour to decode ID 0;
`regression_2492` needs five alternative-contour successes in full-resolution
5x5 detection. A device-only retry stage now recovers these critical cases.
It retains the original representative geometry for the containment tree,
inherits its canonical pyramid level for alternatives, and replaces its corners
only after the first successful eligible retry. The subsequent 936-call matrix
recovers all 278 external CPU observations across 17 Dictionaries and both modes.
All 16 factory photographs remain empty in both modes. Unannotated image agreement
is compatibility evidence, not ground-truth recall.

The full-resolution checkerboard negative needs 5,460 raw candidates across
three windows, exceeding the existing 4,096 default budget. The audited expanded
matrix explicitly uses 16,384 and retains the default-budget failure record.
Capacity changes must remain visible in correctness and timing reports.

The exact pixel path is not sufficient by itself: the expanded nondefault corpus
also exposed corner regressions from the former approximate segmentation resize.
The explicit ARM numerical profiles reproduce public `INTER_LINEAR` on the pinned
CPU builds. Spark selects `OPENCV_ARM_HAL` / `SVE128` / `SEPARATE`; Orin selects
`OPENCV_ARM_HAL` / `NEON` / `FUSED`. These are CPU HAL arithmetic descriptors,
not hardware-specific detection algorithms. Both profiles pass zero-tolerance
resize tests and the four Compute Sanitizer tools. An arbitrary OpenCV build is
not covered by those observations.

The combined Spark strategy checkpoint completes 2,398 audit calls: the 936-call
matrix, 1,298 nondefault configuration calls and 164 rendered close-chain calls.
It reproduces the validated direct-arithmetic checkpoint's complete GPU
observations exactly. CPU/GPU true positives on the 96 unique scenes remain
113/118 with ArUco3 on and 124/127 with it off; there are no false positives.
No common truth's maximum corner error or RMSE worsens by more than 0.001 px;
the largest raw variant worsening is 0.000003096 px. Both historical 91-scene
modes also pass. These results retain explicit capacity, source and numerical
profile identities in the [maintained audit](../../tools/gpu_quality/audit.md).
Final cross-hardware combination, lifecycle-error gates and paired performance
acceptance remain separate from this quality checkpoint.

## Decision and acceptance boundary

Provide the ordered contour route through explicit `dgx-spark-steady` and
`jetson-orin-steady` presets. Retain the old component route and the previous
fresh-build defaults as controls. Do not silently select a route from device
model, scene density or CPU availability. Default promotion requires the expanded
quality, resource and complete-detector performance record; the first recall
improvement alone does not authorize a speed claim.

Keep contour point count and quadrilateral perimeter distinct. Grouping uses the
quad metric; sampling and retry inheritance use contour length. A device retry
pass selects the first successful eligible alternative while containment retains
the original representative geometry. The GPU's transitive group construction
still differs from CPU ordered assignment; the tested close-chain corpus did not
expose a quality failure, which does not prove equivalence for every arrangement.

Use bounded reusable storage and report any overflow or internal inconsistency
through a device status word and the existing host completion boundary. Bits 1,
2, 4, 8 and 16 distinguish invalid cycles, point overflow, candidate overflow,
approximation overflow and invalid counters. Any nonzero value invalidates the
whole invocation and publishes zero detections, including across later threshold
windows. Both candidate routes expose this word. Marker-limit truncation remains
separate and explicitly reported; performance benchmarks reject partial output.
Host readback preserves the caller's output on internal/candidate failure. The
C ABI adds a separate borrowed status view without changing existing result
structure layouts; Python validates the status before accepting results.

Owned completion events order submissions and readback across streams. Steady
unchanged-layout submission remains asynchronous on the host; rebuilding,
reinitialization and destruction wait before reusing storage. Allocation failure
cannot leave old initialized pointers after partial resource replacement.
Error-only drains protect borrowed dictionary and readback storage. See the
[lifecycle contract](../../src/core/detector.md) and [binding contract](../design/c-abi-and-python.md).

### Exact performance transformations

The original four-edge-slot topology remains. The optimized route aggregates
cycle point counts and signed areas at final roots, uses warp-cooperative closed
polygon approximation with stable earliest-index ties, and materializes long
paths using jump tables over compact eligible edges. Scratch regions are reused
only after their earlier owners finish. Serial/full-slot variants remain explicit
controls. The proven point bound is `2WH+W+H`; checked caller budgets may impose a
smaller limit and must fail rather than silently shorten paths.

Maintained contour and approximation tests compare complete oriented paths and
all polygon vertices, including exhaustive 4×4 binary masks, diagonal contacts,
1D images, holes, strided ROIs, same-pointer graph replay and overflow. The compact
module passes its 13 tests and all four sanitizer tools on Spark and Orin.
Paired full-pipeline contour ablations reproduce IDs, rotations and every corner
float on 242 calls per host. These module and ablation gates do not substitute
for final combined-detector quality and timing.

Both steady presets also select exact fused thresholding with a supported-range
fallback, demand-driven complete pyramid prefixes, density-selected Otsu with
an occupied-bin cutoff of 96, a 512-block persistent candidate grid, and
Kleidi-only precomputed resize coefficients. All are explicit, recorded CMake
choices. The demand planner uses current device candidates and includes
refinement needs; it does not cache image content or implement sparse tiles.
Candidate density can still make some optimized stages slower, and blank images
retain scheduling and workspace costs. Measure device stages and complete
GPU-resident/end-to-end latency separately on both architectures before claiming
a performance frontier. See [strategy controls](../../cmake/ArUco3CudaStrategies.md).

## Goals

- Recover all CPU-detected synthetic truths without new false positives or wrong
  IDs/rotations, and preserve ground-truth corner accuracy per scene.
- Verify supported Dictionaries, configuration boundaries, changing frames,
  strided input, and negative industrial photographs.
- Reduce memory and latency without changing path topology, polygon arithmetic,
  candidate selection or failure semantics.
- Retain unfavorable workloads and validate the combined strategies on Spark and
  Orin with sanitizers and reproducible paired timing.

## Open questions

- The final combined route passes the expanded quality gate on Spark, Orin and
  RTX 4090. Measured latency and memory costs, including unfavorable cases, are
  recorded in the [steady-state report](../gpu-steady-state-report.md). Which
  application workloads justify the new quality/memory/latency tradeoff?
- What application-specific candidate/marker budgets are appropriate for textured
  full-resolution images? The audited 16,384 candidate limit is not a new default.
- Are additional CPU HAL descriptors or grouping compatibility changes needed
  for OpenCV builds and contour arrangements outside the maintained corpus?
- Fresh defaults remain historical controls: the final measurements do not
  justify a universal speed claim or automatic promotion of every strategy.

## Related records

- [Optimization and evaluation plan](../gpu-steady-optimization-plan.md)
- [Maintained audit](../../tools/gpu_quality/audit.md)
- [Grouping design](../../src/core/candidate_group.md)
- [Retry design](../../src/core/candidate_retry.md)
- [Contour topology, memory and ablations](../../src/core/contour_candidates.md)
- [Resize profiles](../../cmake/ArUco3CudaResizeProfile.md)
