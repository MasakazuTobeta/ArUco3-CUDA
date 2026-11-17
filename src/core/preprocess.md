# GPU preprocessing and resize compatibility

## Purpose

Preserve the selected pinned CPU resize arithmetic and avoid generating
full-frame pyramid levels that the current frame cannot read. Steady-state
processing introduces no host synchronization point.

## Scope

The experimental detector dispatch is selected at build time with
`ARUCO3CUDA_EXPERIMENT_DEMAND_PYRAMID=1`. The default dispatch remains the full
pyramid baseline while correctness and latency are evaluated. This flag is an
experimental ablation control, not a public configuration contract.

## Current state

Segmentation reads level zero directly, so its preparation and candidate
extraction do not depend on the Gaussian pyramid. The experiment moves pyramid
generation after candidate grouping. One device block computes the maximum
canonical level selected by the current candidates, and includes the levels
that corner refinement could read. Refinement starts below
`closest_level_index_`; it does not sample that starting level itself.

The canonical warp and demand planner share `choose_canonical_level`. Explicit
round-to-nearest float multiplication and subtraction preserve the original
non-fused selection even though the planner's translation unit permits FMA.
The existing convolution, global border reflection and integer rounding are
unchanged. Prefix levels are generated in order on the same stream.

For multilevel inputs, every call overwrites one workspace-owned demand scalar.
An empty candidate set requests level zero and leaves all generated planes
untouched. No previous-frame pixel or candidate metadata is treated as valid.
The full planes remain allocated; this experiment reduces computation rather
than storage. Single-level inputs need no demand kernel.

The captured CUDA Graph still contains the same fixed kernel sequence. Pyramid
kernels return before image reads and writes when their level is not requested.
Those empty launches and their block scheduling still have a cost, and the
demand reduction adds one launch. The experiment must demonstrate a net gain;
it is not intrinsically faster for every scene or device.

### Verification

The direct canonical tests compare full and demanded GPU output byte for byte
at multiple levels and canonical sizes. Generated prefix pixels are checked
against public OpenCV `buildPyramid`; undemanded levels must retain poison bytes.
Repeated image changes and candidate counts at unchanged device addresses run
both through a captured graph and on the default stream, including empty frames
and refinement-only level demand. These tests do not replace CPU/ground-truth
quality evaluation of the complete detector.

## Goals

- Preserve required pyramid and canonical bytes exactly.
- Measure device-event and end-to-end latency separately against the full route.
- Retain successful candidates without tracking, skipped frames or altered thresholds.

## Open questions

- Whether empty-level work avoidance offsets reduction and scheduling overhead
  on Spark and Orin, including textured images without detected markers.
- Whether bounded grid-stride kernels improve empty-level scheduling costs.
- Sparse tiles could reduce work within required levels, but need exact parent
  footprint propagation and current-frame invalidation. Sparse refinement also
  needs per-iteration footprints; a fixed initial corner halo is not sufficient.

## Code provenance

The demand reduction and shared selector are original rearrangements of this
repository's existing GPU pipeline. Pixel convolution and selection semantics
retain their existing pinned OpenCV compatibility basis. No external code was
copied and no official ArUco GPL source was consulted.

## Segmentation resize compatibility profiles

`ARUCO3CUDA_RESIZE_PROFILE=GENERIC` retains the 11-bit coefficient and truncated
vertical arithmetic used by the generic public OpenCV path. `OPENCV_ARM_HAL`
reproduces the pinned ARM HAL dispatch for grayscale downscaling: KleidiCV uses
8-bit weights, Carotene general resizing uses 7-bit weights, and its fast
one-axis path uses 11-bit weights. A gray-level difference can change the
selected contour and worsen refined corners; agreement within one gray level
is therefore insufficient for detector quality acceptance.

The same CUDA kernel runs on Spark and Orin. The host records a fixed HAL
compatibility descriptor when preparing the layout. The descriptor controls
KleidiCV's coordinate rounding schedule: SVE vector blocks periodically
recalibrate their coordinates, while NEON uses overlapping 8-pixel tail blocks.
`ARUCO3CUDA_RESIZE_ARM_BACKEND=AUTO` reads CPU SVE2 support and vector length once;
`NEON`, `SVE128`, `SVE256`, and `SVE512` explicitly override that choice. This is
an observable CPU compatibility choice, not GPU architecture tuning. The compiled
profile and resolved backend are reported in quality and benchmark metadata.
`ARUCO3CUDA_CAROTENE_X_FMA=FUSED` or `SEPARATE` independently describes the
horizontal Carotene intrinsic's contraction. Independent 1,254-case random CPU
oracle grids require this distinction. The pinned Orin GCC 9.4 build uses
`FUSED`; the pinned Spark GCC 13.3 build uses `SEPARATE`. CPU SVE support does not
select this arithmetic. Scalar vertical and fast-path coordinates use FMA in
both builds. Benchmark builds explicitly set and record the value. `AUTO` is currently accepted only with `GENERIC`; ARM HAL requires an explicit
choice. The OpenCV binary can have been built by a different compiler from this
library, so exact public-API tests remain mandatory. The ordinary default remains
`GENERIC`; evaluated ARM steady presets select their validated descriptors
explicitly rather than guessing an arbitrary linked OpenCV build.

The matching-profile test requires byte equality with public `cv::resize` using
`INTER_LINEAR`, including random strided ROIs, vector tails, block recalibration,
one-axis reductions, dimensions below HAL support and the nondefault detector
ratios that exposed corner-quality regressions. `INTER_LINEAR_EXACT` is a
different API behavior and is not this compatibility oracle. Full detector
quality checks must still compare ground-truth corners after pixel equality.

### Resize provenance

The scalar CUDA expressions were derived from the permissively licensed public
implementations below; SIMD instruction sequences were not copied. Their
coordinate quantization, rounding order, dispatch conditions and overlapping-tail
semantics influence the new ARM compatibility path. Required notices are retained
in [NOTICE](../../NOTICE).

- OpenCV commit `0654a42e19215ef25b1d367d822f3c630447e7c7`,
  [hal/carotene/src/resize.cpp](https://github.com/opencv/opencv/blob/0654a42e19215ef25b1d367d822f3c630447e7c7/hal/carotene/src/resize.cpp),
  BSD-3-Clause, copyright NVIDIA Corporation 2015.
- KleidiCV 26.03, `kleidicv/src/resize/resize_linear_generic_u8_neon.h`,
  `resize_linear_generic_sc.h`, `resize_linear_api.cpp`, and
  `adapters/opencv/kleidicv_hal.cpp`, Apache-2.0, copyright Arm Limited and/or
  affiliates 2023–2026. The release is the dependency pinned by OpenCV's
  [KleidiCV download configuration](https://github.com/opencv/opencv/blob/0654a42e19215ef25b1d367d822f3c630447e7c7/3rdparty/kleidicv/kleidicv.cmake).
- Existing generic arithmetic follows the same pinned OpenCV
  `modules/imgproc/src/resize.cpp`, which carries a BSD-3-Clause header.

No official ArUco GPL source was consulted. CPU oracle records must include the
OpenCV HAL build and resolved descriptor; a different CPU build can require an
explicit compatibility override.

## Immutable resize coefficients

The production `KLEIDI_PRECOMPUTED` policy reserves two device arrays containing source indices and
integer interpolation weights for the KleidiCV path, one entry per output column
and row. Carotene retains direct arithmetic: paired Spark measurements found
its 7-bit path neutral or slower with tables, while KleidiCV improved by about
1.5 times in the isolated resize stage. An internal reservation flag retains
all-mode coefficient preparation for exact tests and ablation. The arrays contain
no image samples. `prepare_segmentation_resize` initializes these arrays once
after a layout is reserved and before graph capture. It synchronizes only that
startup stream and publishes readiness after successful kernel completion.
Preparing uninitialized coefficients during capture is rejected without adding
initialization nodes to the graph.

Steady-frame interpolation reads the immutable arrays and four pixels from the
current input. A changed source/destination extent, HAL vector descriptor or
Carotene contraction descriptor makes the tables ineligible. Unprepared or
ineligible tables select the original direct arithmetic, which remains a byte
oracle and safe fallback. Input stride and pixel contents are never encoded in
the tables. Detector layout changes re-reserve and prepare storage before
recapture; no previous-frame image information is reused.

Workspace sizing includes aligned `16*(output_width+output_height)` bytes for
ARM profiles, including headroom when the maximum configured dimensions take a
table-free path but a nearby actual size needs tables. GENERIC arithmetic and
its workspace sizing remain unchanged. The CPU-profile getters live only in the
core library: the shared CPU configuration translation unit must not define a
second set with different compiler flags. Both CPU-first and core-first link
orders have maintained metadata regressions.

The exact random resize grid compares both direct and prepared GPU planes with
public OpenCV. Further tests change image contents, host ROI strides and device
pitch under repeated captures, verify coefficient bytes remain unchanged, and
exercise descriptor changes, failed preparation and capture rejection. This
implementation is experimental until its correctness and latency gates are
recorded; the initialization cost belongs to startup, not steady-frame timing.


### Coefficient validation and timing checkpoint

The all-mode coefficient implementation passed 22 focused preprocessing/detector
tests and four Compute Sanitizer tools on both Spark and Orin. Spark's 1,652
critical/variant sequence calls and Orin's 354 critical calls reproduced the
linked direct-arithmetic GPU observations and ground-truth metrics exactly.
A temporary Spark allocation failure affected tiny input buffers before resize;
those failure logs are retained separately from the unchanged successful rerun.
The precise shared-memory/driver cause was not established.

A paired segmentation-only diagnostic uses six fixed dimension pairs, 20 warmup
pairs and 200 measured pairs, alternates execution order, verifies output SHA-256
identity, and reports nearest-rank p50/p95/p99 for device events separately from
host launch-plus-synchronization time. Three fresh processes on Spark measured
the KleidiCV 641×479 → 383×287 case at 6.624–6.752 µs direct versus
4.416–4.448 µs prepared (medians). On Orin the same pairs improved by about
2.9–3.3 times, with substantial between-process latency variation. Spark's
Carotene 7-bit paths were neutral or slower; Orin's improved. The common
production policy therefore prepares only KleidiCV coefficients and leaves
Carotene direct. This is stage evidence, not a detector end-to-end speedup claim.

The local ignored records include `build/gpu-study/resize-coeff-stage-*.jsonl`
(all raw timing samples and image hashes), `resize-coeff-stage-probe.cpp`
(reproduction probe), `resize-coeff-critical/`, `resize-coeff-variants/`, and
`resize-coeff-orin/`. These paths are not included in a fresh checkout. The final
Kleidi-only production policy has its own tests for table allocation/readiness;
all-mode tables remain an internal correctness ablation rather than an
architecture-dependent default.
