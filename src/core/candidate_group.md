# Candidate grouping perimeter

## Purpose

Keep the representative with the largest Euclidean quad perimeter so nested
threshold contours do not pull accepted marker corners inward.

## Scope

The CUDA grouping stage, its workspace, and focused reference tests. Contour
extraction, canonical level selection, and corner refinement are separate stages.

## Current state

`CandidateGroupBuffers::quad_perimeter_` stores the sum of the four Euclidean
edge lengths, using the CPU helper's float operation order. Group ranking and
proximity thresholds use this value. `DeviceCandidates::perimeter_` retains the
contour point count, including after grouping, because canonical level selection
uses that quantity. The existing reset kernel computes the quad perimeters, so
this correction adds scratch storage but no kernel launch.

The `quality_projective_9` scene exposed the distinction: its inner and outer
paths both contain 202 points, while their quad perimeters are approximately
261.351 and 268.446 pixels. Contour-length ranking selected the earlier inner
path. Replaying public OpenCV `cornerSubPix` from that seed reproduced all GPU
output coordinates exactly; replaying the outer seed reproduced all CPU output
coordinates exactly. The former seed produced a maximum ground-truth corner
error of 4.414 pixels, versus 0.550 pixels for the latter. This diagnosis concerns
candidate selection, not a change to the refinement algorithm.

Tests cover the actual projective quads, a winner whose contour length is smaller,
and a proximity decision where contour length would merge separate markers.
The full quality audit remains necessary because grouping can change detection
and suppression outcomes beyond the focused fixtures.

The ordered-contour detector supplies the actual segmentation width and height
for border rejection. The existing selection kernel checks all four corners of
the chosen group representative against the inclusive configured margin before
compaction. Rejected groups have no replacement representative, matching the CPU
ordering; interior alternatives cannot revive a border-rejected group. The
legacy component route has already filtered its inputs and passes zero dimensions
to preserve its existing behavior. The check adds no kernel launch or allocation.


## Goals

Retain CPU-equivalent or better detection quality across the annotated corpus
and independent real-image compatibility cases. Preserve the distinct meaning
of contour length in downstream sampling.

## Open questions

The existing CUDA union-find grouping merges proximity chains that the CPU
ordered grouping may leave separate. The experimental contour route now retains
eligible alternatives through [close-contour retries](candidate_retry.md).
Broader quality evidence is still required; fixing the perimeter quantity and
retry behavior does not establish universal equivalence.
