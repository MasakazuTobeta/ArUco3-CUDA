# Accuracy evaluation (accuracy)

## Purpose

Matches detection results against the ground truth of the synthetic corpus and computes precision, recall,
the agreement rate of ID and rotation, and the RMSE of the corners.

The [diff report](../report/report_diff.hpp) takes the CPU baseline implementation as its reference.
The baseline implementation is an oracle for compatibility, not ground truth. A marker that the baseline
itself misses does not show up as a difference, so even with zero differences there can still be markers
that are not being detected. A separate path that matches against ground truth is kept.

## Scope

Covers matching one image with `compare_to_truth`, aggregation with `accumulate`,
and computation of the metrics by `precision` / `recall` / `corner_rmse_px`.
Generating the corpus, running the routes, and classifying by condition are the responsibility of the CLI (`main.cpp`).

## Current state

The following are implemented.

- Matching ground truth to detections by proximity of the centroid
- Counting of true positives, false positives, and false negatives
- Counting of agreement in the corner ordering (rotation)
- RMSE and maximum of the corners

## Design decisions

### Matching is done by position, not by ID

Matching by ID produces one missed detection and one extra detection
when an ID is misread. Whether the ID of the same marker was misread, or something was falsely detected
at a different location, cannot be distinguished. Matching is done by proximity of the centroid, and a case where a match is
established but the ID differs is treated as "the ID was misread".

A single misread ID is counted toward both the false positives and the false negatives. Counting it on only
one side would make either precision or recall overlook that error.

### The matching radius is determined by side ratio

An absolute value would make the appropriate distance differ between a 16 pixel marker at 640x480 and a 256 pixel
marker at 3840x2160. It is specified as a ratio relative to one side of the ground truth.

### Precision and recall distinguish the case where they cannot be defined

When there is not a single detection, precision cannot be defined. Treating division by zero as 1.0 would mean
"the less you detect, the higher the precision", which misleads judgment. The same applies to recall when there is not
a single ground truth item. The corpus contains scenes with 0 markers,
so this actually occurs. The returned `bool` distinguishes it.

### Both RMSE and the maximum are kept

This is to distinguish the case where only one corner is far off from the case where all four corners are uniformly displaced.
With the maximum alone, the former and the latter look the same.

### Rotation agreement is judged by the corner ordering

The ground truth corners are ordered as (0,0), (S,0), (S,S), (0,S) in the marker coordinate system.
If the ID has been read correctly, the detection is in the same order, so any nonzero number of cyclic steps
means the rotation was judged incorrectly.

## Goals

- Allow the same metrics to be produced with annotations of a real-image dataset as ground truth
- Visualize and save the images where differences occurred

## See also

- [Diff report](../report/report_diff.hpp)
- [Synthetic corpus generator](../corpusgen/corpus_generator.md)
- [Evaluation plan](../../docs/evaluation-plan.md)
