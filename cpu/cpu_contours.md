# CPU contour workspace

## Purpose

Reduce allocation and copying for noise contours that cannot become marker
candidates, while retaining the complete ordered contour path of every accepted
quadrilateral.

## Scope

`ContourWorkspace` consumes a current-frame binary image and validated detector
settings. It performs no tracking and uses no previous-frame pixel values.
One workspace is required per concurrently processed threshold window.

## Current state

The implementation uses a sequential contour scan with reusable padded pixels,
trace storage and polygon-approximation storage. It rejects perimeter-ineligible
paths before approximation and materializes only accepted quadrilaterals. It
then reverses the appended result range to reproduce `RETR_LIST` discovery order.
Point coordinates, cyclic starting point, approximation epsilon, convexity and
minimum corner-distance tests retain the previous implementation's arithmetic.

Independent threshold windows can still run in parallel. This helper deliberately
does not use the upstream contour finder's internal stripe scheduler: tracing
from a stripe can write visited flags below that stripe, overlapping another
worker's storage. In the initial adaptation, direct four-thread random-input
results varied even when the minimum-size filter was disabled. A single scan per
owned threshold plane avoids these overlapping writes. This is not a change to
OpenCV or the compatibility baseline library.

The padded owner is zero-initialized when its shape changes. Each call thresholds
all current input pixels directly into the padded center, replacing the previous
copy-then-normalize passes. The scanner never writes the zero guard border; only
that constant guard content is reused between calls. No previous-frame center pixel is
read before it is overwritten. Constructor assignment creates and clears the
replacement before changing the owner, so an allocation failure cannot leave a
new shape with an uninitialized border.

Padding is explicitly isolated from any parent ROI. The pinned OpenCV fast
finder uses `copyMakeBorder(..., BORDER_CONSTANT)` without `BORDER_ISOLATED`;
therefore a direct ROI can expose its parent's pixels instead of the required
zero guard. Direct helper tests clone the oracle ROI to define an isolated input.
The detector's threshold outputs already own their storage, so its regular
OpenCV comparisons need no such adjustment.

### Provenance

The contour tracing, visited-byte rules and vectorized foreground/background
scan are adapted from OpenCV
[`modules/imgproc/src/contours_truco.cpp`](https://github.com/opencv/opencv/blob/0654a42e19215ef25b1d367d822f3c630447e7c7/modules/imgproc/src/contours_truco.cpp),
commit `0654a42e19215ef25b1d367d822f3c630447e7c7`.
The source SHA-256 is
`57625f4ee96306f467ddba303ef25d965401a9fba8e7ad7ea0ed52a332008cba`.
This particular source has no file-level license header; the inspected pinned
repository [LICENSE](https://github.com/opencv/opencv/blob/0654a42e19215ef25b1d367d822f3c630447e7c7/LICENSE)
is Apache-2.0. Attribution is retained in [NOTICE](../NOTICE).

Changes remove unsupported contour-approximation modes, the paged temporary
allocator, and the internal stripe scheduler. They add standard vector ownership,
reusable workspace, isolated padding, normalization directly into the padded
center, and immediate quadrilateral filtering.
The quadrilateral filtering arithmetic is adapted from this repository's existing
Apache-2.0 `hybrid/cpu_candidates.cpp`. No official ArUco GPLv3 source was used.

## Goals

- Verify exact ordered quadrilaterals and their full paths before benchmarking.
- Compare complete detections under one, two and four OpenCV threads, including
  changing frames and combinations of compatible optimization options.
- Retain the original finder as an explicit comparison and fallback route.
- Measure both noisy and clean images without assuming that allocation savings
  imply a universal speedup.

## Open questions

- Correctness and performance evidence is recorded in the
  [steady-state report](../docs/cpu-steady-state-report.md).
- The optimal balance of window-level and frame-level parallelism depends on
  available cores and contour density.
