# Device contour candidates

## Purpose

Recover eligible marker contours when their foreground borders are connected to
other objects. A foreground component can contain several marker-shaped holes;
one quadrilateral per component cannot represent those candidates.

## Scope

`contour_candidates.cu` is an experimental device-resident alternative to the
component-extrema candidate route. It consumes independently owned threshold
planes and produces the existing candidate structure. It has no CPU fallback,
input cache, host count readback, or per-frame allocation. Detector integration,
public resource limits, and final failure reporting belong to the caller.

## Current state

The prototype uses four directed crack-edge slots per pixel. An edge exists
between foreground and a cardinal background neighbor, including isolated zero
background outside the image. Edges keep foreground on the right. Choosing the
left continuation at a diagonal contact realizes 8-connected foreground and
4-connected background; simply relabeling background with an 8-connected
component kernel would implement a different topology.

Each active edge writes a unique predecessor. Atomic decreasing-root union-find
identifies independent cycles without shared visited-pixel flags. After union, edge threads read the immutable parent forest and accumulate exact
per-cycle pixel transitions, signed area, and the minimum eligible east-facing
edge. Warp groups combine equal-root contributions before integer atomics.
Cyclic pixel transitions equal the collapsed unit-path length; a cycle with no
transition represents one isolated pixel. The minimum east-facing edge gives the
same first enclosed background pixel because original edge order is preserved.
A final root pass applies the unchanged perimeter limits and chooses the
OpenCV-compatible start. `ARUCO3CUDA_EXPERIMENT_PARALLEL_CONTOUR_COUNT=0` retains
the serial counting route. Reverse traversal collapses adjacent edges from the
same pixel into an oriented unit path. This also represents isolated pixels,
narrow lines, self-touching paths, nested holes, and image-edge contours.

The cooperative materialization experiment also aggregates each cycle's crack
edge count. Five ordered pointer-doubling passes prepare the 32nd predecessor.
A warp then traverses 32 consecutive edges at a time; a ballot and stable lane
prefix retain foreground-pixel transitions in their original order. The final
partial tile uses the exact crack-edge count, and the closing repeated pixel is
handled identically to the serial traversal. Cycles with at most 32 edges keep
the serial helper. `ARUCO3CUDA_EXPERIMENT_COOPERATIVE_MATERIALIZE=0` retains serial
materialization for quality and timing ablations. The preparation passes are extra work, so this experiment needs density and
small-image measurements in addition to long-contour measurements.

The compact-preparation experiment records each active edge's descriptor during
parallel counting, without rewriting canonical root-map entries that other
threads read. A later pass selects only edges of perimeter-eligible cycles
longer than 32 edges. An unordered compact list drives the five jump passes;
original edge IDs still index the jump maps, so output order is unchanged.
`ARUCO3CUDA_EXPERIMENT_COMPACT_MATERIALIZE_EDGES=0` retains full-slot preparation.
The serial counting reference uses full-slot preparation as well. The compact
list uses stack scratch after the per-root edge counts, with the proven
`2WH+W+H` edge bound and an explicitly checked small-budget fallback. Its count
is reset on every invocation, including blank graph replays.

Cycle roots are compacted in increasing original edge order using block-local
ballots and an exclusive scan over block counts. Each orthogonal boundary cycle
contains at least four crack edges, so one root descriptor per input pixel is a
safe bound even before the tighter interface-edge bound below. The predecessor
and union arrays still use four slots per pixel; active-edge compaction remains
unimplemented.

Persistent kernel grids and compact scans consume the device root count. Empty
and inactive descriptor capacity is not scanned for path or candidate offsets.
Only perimeter-eligible paths receive storage through an exclusive prefix scan.
The closed Ramer-Douglas-Peucker approximation uses a disjoint bounded stack and
output slice per path. One full warp processes a path: lanes evaluate independent
point distances with the serial arithmetic order, and a stable maximum reduction
chooses the earliest traversal index on a tie. Stack traversal, split order, and
the final almost-collinear cleanup preserve the serial algorithm. No candidate
is rejected merely because an intermediate approximation has more than four
vertices. `ARUCO3CUDA_EXPERIMENT_COOPERATIVE_RDP=0` retains the serial CUDA route
for exact quality and timing ablations. Four-vertex convex approximations must satisfy the
configured corner-distance rule. Accepted quads are reordered clockwise and
compacted in descending root order. Individual path starting points and
orientation have a stronger compatibility gate than candidate-list ordering;
full detector grouping and ordering require independent validation for each
implementation revision.

### Resource and failure contract

The caller reserves storage once through `reserve_contour_candidates`. The
explicit `max_points` budget bounds the sum of eligible contour lengths for one
threshold plane. Internally, the allocation is capped at `2WH + W + H`. There
are `(W-1)H + W(H-1)` internal undirected pixel adjacencies and `2W + 2H` exterior
sides; each contributes at most one directed crack edge. Cycles partition those
edges, and collapsing repeated pixel centers cannot add path points. Therefore
the cap preserves every possible path and never weakens a smaller caller budget.
Checked 64-bit arithmetic derives the cap only after validating dimensions. Dimensions
must allow four times the pixel count plus the shared scan's 255-element
rounding margin to fit in a signed 32-bit integer.

Two edge-index arrays consume approximately 32 bytes per pixel. Compact root,
length, and offset arrays consume another 12 bytes per pixel. With the proven
point cap, paths, approximation output, and RDP stacks consume approximately
48 bytes per pixel, plus 24 bytes times the sum of width and height. Total module
storage is approximately 92 bytes per pixel before scan scratch and alignment,
versus 160 bytes per pixel in the frozen first prototype. At 3840 by 2160 this is
approximately 0.76 GB. Threshold windows reuse the storage sequentially.

The experimental detector integration additionally retains the legacy
`CandidateFilterBuffers` scratch alongside candidate outputs; these bytes are
not included in the module figures above. The initial integration selected the
contour route through a compile definition. End-of-dispatch diagnostics and
public resource-limit configuration are integration responsibilities; module
memory reductions alone do not establish total detector peak memory.

Parallel counting needs a root-to-descriptor map plus per-root signed area and
hole seed. Before materialization, reservation aliases the point pool for the
map and the approximation pool for area/seed only if their aligned allocated
byte counts cover those regions. Smaller configured point budgets receive
explicit additional regions; no capacity is assumed and no count data survives
into sampling. This preserves the ordinary large-image memory bound while
making small-budget overflow tests valid. Parent/predecessor links remain
unchanged until aggregation completes. Public dimension validation bounds a
32-lane edge-area reduction to a signed 32-bit sum; the cycle accumulator is
64-bit.

After counting, pointer-doubling reuses the map and area/seed regions. The final
jump map occupies approximation scratch, separate from the points being
written. Per-cycle edge counts occupy stack scratch until materialization
finishes. Reservation checks each alias's actual aligned byte capacity and
allocates a separate fallback when a small point budget cannot cover it. RDP
reuses these regions only in a later kernel; diagnostic callers must not read
the temporary counts or jump map after the invocation completes.

`failure_` is a device scalar with independent bits for invalid cycles, point
pool overflow, candidate overflow, and approximation-stack overflow. The scalar
is reset by `append=false` and remains sticky across appended windows. Any
nonzero value invalidates the result; a caller must report it at its normal
completion boundary. Submission success alone does not establish success of an
asynchronous invocation. Capacity limits must never silently turn into missing
markers. A failed invocation publishes zero candidates so later device stages
cannot read incomplete or stale slots. `accepted_total_` retains the accepted
candidate count for diagnostics even when the published count is zero.

Every invocation reconstructs edges and paths from the current image. Kernel
submission uses the supplied stream and is compatible with CUDA Graph capture;
replay with unchanged pointers must still observe changed image contents.

### Approximation provenance

`contour_approx.cuh` adapts the closed-contour branch of `approxPolyDP_` from
OpenCV `modules/imgproc/src/approx.cpp`, commit
`0654a42e19215ef25b1d367d822f3c630447e7c7` (OpenCV 4.14.0). The referenced file has
an Intel 2000 three-clause BSD header, retained in the helper. Its SHA-256 is
`d2b9fa2419c9eb6f880676e862d5b5fb38cd250b2db26da3638efc64181d4bcb`.
Changes replace OpenCV containers and automatic buffers with explicit caller
storage, retain only integer closed paths, and return an explicit overflow
result. Compile the CUDA source with `-fmad=false` to preserve the oracle's
floating-point comparison boundaries. The crack-edge topology implementation
was developed independently from the connectivity specification and public
OpenCV contour observations. No official ArUco GPL source was used.

### Validation

The preliminary host oracle checks exact oriented unit paths, including their
starting point, against `findContours(RETR_LIST, CHAIN_APPROX_NONE)` on all
65,536 binary 4-by-4 images and 10,000 deterministic random images with dimensions
1 through 16. The actual approximation helper is also checked against public
`approxPolyDP` at epsilon rates 0, 0.01, 0.03, and 0.1. These checks passed.
The oracle clones input and uses one OpenCV thread because the pinned stripe
tracer can otherwise produce nondeterministic results on small inputs.
A standalone CUDA probe also passed exact oriented-path comparisons on seven
masks: two connected rings and the three threshold windows of each of the
`quality_multi_dense_0` and `quality_projective_2` scenes. All device diagnostic
bits remained zero, and the connected rings produced both hole quadrilaterals.
Temporary probes and logs remain under ignored `build/gpu-study/`; they are not
maintained tests or available from a fresh checkout.

Maintained CUDA tests cover connected rings, isolated and narrow images, nested
and diagonal topology, strided ROI, random masks, perimeter and approximation
settings, appended windows, graph replay with changing contents, and explicit
overflow. GPU execution, Compute Sanitizer, full detector quality, and timings
must be recorded separately for each revision; host topology agreement does not
establish these. The initial integrated contour route recovered CPU detections
in the 118-call quality sequence, while separate grouping/corner-quality work
remained necessary. That initial result alone did not validate the later
compact-root, cooperative-RDP and parallel-count integrations. Their subsequent
module and full-pipeline gates are recorded below; the final combined
[steady-state report](../../docs/gpu-steady-state-report.md) covers quality,
sanitizers, memory and both favorable and unfavorable timing results.

The compact-root/cooperative-RDP module passes ten maintained contour tests plus
`ContourApproxTest.cooperative_and_serial_vertices_match_public_oracle`. The
latter compares every approximation vertex, including rejected non-quadrilateral
paths, against public OpenCV for all 65,536 binary 4-by-4 masks, 1,000 deterministic
random masks, connected rings, and four epsilon rates. All eleven tests pass
memcheck, initcheck, racecheck, and synccheck on DGX Spark. The module and new CUDA
test also compile for SM 87; later Jetson Orin runtime gates are recorded below.

An ignored diagnostic probe uses preuploaded binary masks, 20 warm-up graph
launches, and 100 measured launches. It records CUDA-event module time separately
from host launch/synchronization time and excludes allocation and input upload.
The frozen serial module's full-HD blank-mask median was approximately 5.18 ms;
compact roots reduced it to approximately 0.73 ms. On connected rings,
cooperative RDP reduced the compact module median from approximately 4.67 ms to
0.52 ms. These are module diagnostics under observed unrelated GPU load, not
whole-detector or CPU-relative speed claims. Raw samples and Nsight records
remain in ignored `build/gpu-study/` for ablation.

The parallel-count checkpoint passes twelve maintained tests and all four Compute
Sanitizer tools on DGX Spark. Its GPU outputs match the frozen retry integration
exactly on 236 quality-sequence calls and six calls using the three benchmark
inputs. An isolated whole-pipeline A/B keeps the frozen retry source, compiler
flags, Otsu implementation, and pixels unchanged and overlays only this module
and its tests. With 20 warm-ups and 200 resident-input iterations in three
alternating process pairs, warehouse-photograph wall medians changed from
7.54--7.62 ms to 4.13--4.19 ms. Separate 20-frame Nsight traces give median summed
kernel durations of 5.46 ms and 1.98 ms; materialization remains about 0.93 ms.
The same experiment showed a small clean-marker gain and no consistent blank
image wall-time gain. Unrelated GPU scheduling overhead is visible, so these
measurements do not establish a hardware latency limit. The unfavorable full-HD
module and density cases remain in the local A/B logs.


The cooperative materialization and compact-preparation checkpoints each pass
thirteen maintained tests and all four Compute Sanitizer tools on DGX Spark
(CUDA 13.0.88, driver 580.95.05, GCC 13.3, pinned OpenCV 4.14.0). The compact
variant also passes the thirteen tests with serial counting selected. Both
checkpoints preserve every GPU detection and corner float on the same 242
full-pipeline comparison calls.

With the same resident-input primary method, compact preparation versus five
full-slot passes changes warehouse wall medians from 3.64--3.67 ms to
3.38--3.42 ms and clean-marker medians from 2.96--2.98 ms to 2.80--2.86 ms.
Blank-image wall medians show no consistent gain. Separate 20-frame Nsight
profiles show the warehouse kernel sum moving from 1.48 ms to 1.22 ms:
preparation falls from 375 us to 64 us of compaction plus 45 us of jump passes,
while materialization remains approximately 63 us. These diagnostic kernel sums
exclude launch/synchronization overhead and do not constitute a hardware limit.

Module CUDA-event measurements retain unfavorable cases. Full-slot cooperative
materialization regressed full-HD photographs and dense masks versus serial
materialization; compact preparation recovers much of that work, but the
full-HD random-50-percent mask remains slower than the earlier parallel-count
checkpoint in separate paired runs (approximately 7.5 ms versus 6.1 ms).
The small blank-mask module also grows from approximately 29 us with serial
materialization to 33 us with compact cooperative materialization. Full-HD
photograph medians vary substantially across paired runs under unrelated GPU
load. No mask or high-percentile sample is discarded. Active-edge representation,
serial counting, full-slot preparation, and serial materialization snapshots
remain available locally for these ablations. Compact preparation adds one
256-byte aligned scalar at ordinary point budgets; smaller budgets use the
explicit fallback allocations described above.


The compact checkpoint is also validated on Jetson Orin with CUDA 11.4.315,
GCC 9.4, pinned OpenCV 4.14.0, and explicit ARM HAL / NEON / fused-x resize
compatibility. Its thirteen maintained tests pass all four Compute Sanitizer
tools. The count reference passes twelve tests, and all 242 paired full-pipeline
quality calls preserve IDs and corner floats exactly. The same source implements
both platforms; the OpenCV resize profile is an explicit build choice.

Orin resident-input comparisons pin the benchmark thread to CPU 8 and use the
same 20 warm-ups, 200 measurements, and three alternating process pairs. The
count reference versus compact materialization gives these wall-time median
ranges:

| Input | Parallel count + serial materialization | Compact cooperative materialization |
| --- | ---: | ---: |
| 1280x720, four clean markers | 1.607--1.615 ms | 1.499--1.507 ms |
| 1920x1275, warehouse photograph, no markers | 4.196--4.237 ms | 3.081--3.100 ms |
| 1920x1080, blank | 0.966--0.990 ms | 0.997--1.009 ms |

The blank case regresses by approximately 19--31 us in paired medians. Module
CUDA events likewise retain a small-blank regression (97 to 105 us), a
full-HD random-10-percent regression (6.66 to 6.93 ms), and a random-50-percent
regression (12.24 to 13.18 ms). Full-HD photographic module medians have no
consistent improvement, although the default reduced-resolution warehouse
pipeline improves. All p95/p99 values remain in the local logs, including a
compact clean-marker p99 outlier of 3.009 ms and warehouse p99 of 5.856 ms.
These are scoped performance tradeoffs, not evidence of a universal speedup.
The environment query reports MAXN and the benchmark observes 1300 MHz; the
query cannot read its EMC cap, so memory clocks are not established as controlled.
Raw Orin results are mirrored locally under ignored
`build/gpu-study/orin-contours/` and are not available in a fresh checkout.

## Goals

- Recover CPU-detected marker IDs and retain ground-truth precision and recall.
- Preserve exact unit-path topology and configuration-dependent candidate rules.
- Expose bounded-resource failures without partial-output ambiguity.
- Reduce memory and work after the quality gate, beginning with compact active
  boundary edges and compact eligible contour descriptors.
- Measure kernel time and end-to-end time independently on target hardware.

## Open questions

- The active-edge prototype under ignored `build/gpu-study/contour-active-source`
  reduces full-HD blank and photographic mask work, but regresses several small
  masks. It is preserved as a separate ablation and is not the current default.
  A strategy decision must consider density and whole-detector measurements.
- Parallel counting improves tested small masks but has unfavorable full-HD
  photographic and sparse-random cases. The unchanged serial materializer can
  lose cache warmth from serial counting; paired measurements must retain this
  effect rather than attributing all differences to unrelated GPU load.

- Does candidate order affect grouping or suppression on any maintained scene?
- Which configured point budget gives a practical resource contract on Jetson
  Orin while reporting all overflows explicitly?
- How much time comes from full-image edge slots and scans versus long sequential
  cycle traces? A measured profile must guide the next implementation.
- Can compact boundary masks and prefix allocation preserve the exact topology
  while reducing the full-resolution workspace sufficiently?
