# Sparse image pyramid and monitored refinement

## Purpose

Avoid generating pyramid pixels unrelated to the current frame's candidate sampling and
subpixel refinement. This is a steady-state CPU experiment; startup allocation cost is
outside its target. It never tracks markers or reuses previous-frame detections or pixels.

## Scope

`SparsePyramid` serves the optional `CpuOptions::sparse_pyramid_` route in `CpuDetector`.
It applies only to the ArUco3 strategy on whole host images. Input submatrices use the
existing full-level implementation, preserving OpenCV's input-ROI parent-border behavior.
The ordinary full-resolution strategy is unchanged. Each decode worker owns one workspace.

## Current state

The workspace reserves full-address-space pyramid buffers with per-level valid-tile maps.
Only validated regions may be read. A detector frame epoch invalidates every worker's
validity map before its first use of the new input, even when the input pointer, dimensions
and stride are identical. Allocation capacity survives; pixel validity does not.

Level initialization marks its tile-size tag invalid before either pixel or
validity-map allocation, then publishes the requested positive tag only after
both allocations and metadata updates succeed. A failed allocation therefore
forces initialization to retry on the next request, even if the pixel shape
already changed. The ordinary valid-cache path is unchanged.

`reset(tile_side_px, full_level_ratio)` is the corresponding direct helper boundary. Callers
must reset before changing input content. The helper stores no input image or detection
references. `sparse_tile_side_px_` controls request alignment, and
`sparse_full_level_ratio_` controls promotion to a whole-level computation. Neither option
changes detector thresholds or the expected output. The detector also exposes
`sparse_max_work_ratio_`: it estimates current-frame candidate rectangle work through parent
levels and chooses the ordinary complete-level route when that estimate is too large.
This is a performance choice using the current input only; zero disables the model.

### Regional pixels

An output sample at `(x,y)` of an 8-bit Gaussian pyramid reads the exact five-tap footprint
`(2x-2..2x+2, 2y-2..2y+2)` from the preceding level. A rectangular output request therefore
has a finite rectangular parent footprint. Parent origins are even, preserving the global
sampling phase. The helper recursively generates the footprint, calls OpenCV `pyrDown`,
and copies its valid interior to the appropriate global coordinates. Only actual global
image boundaries apply reflection; artificial crop boundaries lie outside the consumed
five-tap support.

Tile maps retain current-frame work across candidate sampling and refinement. A missing
rectangular tile group is generated in one operation, avoiding a `pyrDown` dispatch for
each tile. Existing overlap may be recomputed within the same worker. Large accumulated
demand promotes that level to one complete `pyrDown` operation.

### Canonical sampling

Sampling executes the actual OpenCV `warpPerspective` with the original global homography.
An earlier experiment regenerated its nearest coordinate map through the exported HAL
blockline function. It differed at a single half-integer border sample on some fractional
quads, so that approach was removed.

Before the actual warp, the helper bounds every possible source sample of the inverse
homography. For a denominator of one nonzero sign over the canonical rectangle, each
projective coordinate reaches its extrema at a rectangle vertex. Bounds are evaluated in
`long double`, expanded for floating-point evaluation and nearest rounding, then clamped
for OpenCV's signed-short nearest-map saturation and clipped to the source image. The
complete resulting pixel rectangle is generated before sampling. Denominator poles,
singular or nonfinite transforms, and a conservative floating-point error estimate of
at least one quarter pixel select a complete level instead. The actual warp consequently
keeps the installed OpenCV implementation's platform-specific map rounding.

A sentinel probe poisons unrequested cached storage with two different values before
otherwise identical calls. Its outputs must equal both one another and the actual warp
of a fully generated pyramid. This tests that stale or uninitialized tiles cannot affect
canonical samples.

### Refinement

The monitored refiner preserves global coordinates, OpenCV's Gaussian mask, float gradient
subtraction before widening, double accumulation order, iteration limits, error test,
out-of-image stopping rule and poor-convergence reset. Every iteration requests the
actual bilinear patch footprint before calling public `getRectSubPix` on the global
pyramid plane. A point that jumps beyond its initial window therefore requests its new
footprint; no fixed initial halo is assumed to contain the iteration path.

Masks and patch storage are reused. The numerical result is checked against actual
`cornerSubPix`, not only the CUDA oracle. In particular, the existing GPU oracle performs
some gradient subtraction after widening and is therefore not the CPU compatibility
reference for this helper.

## Code provenance

- The regional request/cache design and footprint planning are original work based on the
  public five-tap Gaussian pyramid operation. Pixel computation calls public OpenCV APIs;
  no OpenCV pyramid kernel was copied.
- The monitored refinement arithmetic is adapted from
  `modules/imgproc/src/cornersubpix.cpp`, OpenCV 4.14.0 commit
  `0654a42e19215ef25b1d367d822f3c630447e7c7`. Its header carries the 3-clause BSD
  license, with Intel Corporation (2000) and OpenCV Foundation (2013) copyrights.
  Those notices and the complete license remain in [NOTICE](../NOTICE), with the source
  relationship identified in `cpu_sparse_pyramid.cpp`.
- OpenCV `imgwarp.cpp` was inspected to understand nearest-map saturation and coordinate
  evaluation; canonical pixel mapping executes the installed OpenCV implementation.
- No official ArUco GPLv3 source was consulted or adapted.

## Goals

- Preserve IDs, rotation order, rejected counts and corner coordinates against actual
  OpenCV over synthetic, external and adversarial inputs.
- Exercise fractional projections, nondefault canonical sizes, odd pyramid dimensions,
  image borders, singular/pole homographies and same-address changed-frame sequences.
- Measure whether saved pixel computation exceeds region lookup, tile bookkeeping and
  repeated dispatch cost, using separate before/after options.

## Open questions

- The best tile granularity and whole-level promotion threshold depend on marker density.
- TLS workspaces can duplicate pyramid work across workers; a shared cache would need
  synchronization that prevents writes overlapping another worker's reads.
- Exact arithmetic parity and the nearest-map footprint assumptions need validation on
  Jetson Orin, Thor and x86, as well as the measured DGX Spark.
- Sparse pyramid storage reserves full-size addressable planes. Its peak memory differs
  from a compact ROI representation even when relatively few pixels are generated.
