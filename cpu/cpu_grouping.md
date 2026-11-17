# CPU candidate grouping

## Purpose

Avoid expensive corner correspondence and point-in-polygon tests when simple
spatial bounds establish that the result cannot change.

## Scope

This CPU experiment consumes candidates from the current frame only. It does not
retain positions, IDs, or predictions across frames. It uses the same grouping,
stable ordering, representative selection, alternate contours, and containment
rules as the existing hybrid helper.

## Current state

`filter_spatial_candidates` computes each quadrilateral's axis-aligned bounds
after the stable perimeter sort. Before a corner-distance comparison, it computes
the minimum separation of these bounds in either coordinate. Every corner
correspondence has at least that separation. The lower bound uses the same
nonnegative float summation, division by four, and square root as the original
metric. Monotonic rounding preserves the bound without replacing the original
acceptance threshold by a squared comparison. Overlapping bounds always proceed
to the original test unless the additional integer-centroid bound applies.
Nonfinite corners are outside the helper's preconditions.

For integer corners within the public image coordinate bound `[0, 65536]`, the
four-coordinate sums and quarter-pixel centroids are exact. The RMS of paired
corner distances is at least either coordinate separation of the centroids,
independently of cyclic correspondence. Float products, nonnegative summation
and square root introduce less than 0.1 pixel of absolute error in this bounded
domain (the largest exact RMS is `sqrt(2) * 65536`). The helper subtracts a full
pixel from the centroid bound before its conservative comparison. Fractional or
out-of-domain coordinates simply disable this additional bound. It never changes
the exact metric for a surviving pair.

Containment requires containment of the bounds. Candidates failing this necessary
condition skip the original point-in-polygon test. Every surviving pair still
uses the original helper, including its boundary rules.

The implementation adapts this repository's Apache-2.0
`hybrid/cpu_candidates.cpp`; no external source is copied. The same pinned OpenCV
4.14.0 compatibility baseline applies. See the [CPU module](cpu_detector.md) for
provenance and public input limits.

Exact-node validation passed, but the measured end-to-end benefit was negligible,
so the experiment remains disabled by default. It is available as an explicit
option and does not change the shared hybrid implementation. See the
[steady-state report](../docs/cpu-steady-state-report.md).

## Goals

- Preserve complete nodes, including order, retries, parent, and depth.
- Measure dense clean scenes as well as noise and sparse scenes.
- Retain only a demonstrated benefit and report any regression.

## Open questions

- Does computing bounds pay for itself at small candidate counts?
- Are additional conservative bounds worthwhile after this first pruning step?
