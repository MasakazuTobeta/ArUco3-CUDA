# Accuracy evaluation CLI (aruco3cuda_evaluate)

## Purpose

Generates the synthetic corpus with a fixed seed, applies the three routes — CPU baseline, hybrid, and CUDA — to the same images,
and produces the accuracy metrics of the evaluation plan broken down by condition.

## Scope

Covers generating the corpus, running the three routes, matching against ground truth, classifying differences from the CPU baseline,
and recording device memory usage. Computing the metrics themselves is the responsibility of
[accuracy](accuracy.md). Speed is not measured. Speed is measured by the
[benchmark harness](../../bench/benchmark_harness.md).

## Current state

The following are output.

- Matching against ground truth (three divisions: overall, at or above the ArUco3 lower bound, and below it)
- Breakdowns by condition, by marker side length, and by resolution
- Counts by type of difference from the CPU baseline, and a list of the scenes with differences
- The maximum workspace usage and the number of allocations added during detection

## Design decisions

### The corpus is generated on the spot rather than read from the manifest

The manifest is JSON, and reading it requires a parser. The corpus generator can be used as a library,
so building the same scenes on the spot from the same seed yields the ground truth without needing a parser.
The generator's random seed is derived from the scene's sequence number, so as long as the preset is not
changed, the images match those produced by `aruco3cuda_corpusgen`.

### Recall is divided by "the size ArUco3 can detect in principle"

ArUco3 does not detect markers whose side after downscaling falls below `min_side_length_canonical_img`.
The lower bound is `S + L * tau_i`, which by default (S=32, tau_i=0.05) is
224 pixels at 3840x2160. The corpus deliberately includes sizes below this lower bound,
so the overall recall is dominated by the lower bound of the strategy. So that the implementation's misses
can be read out, a separate division collecting only the ground truth items at or above the lower bound is also produced.

False positives that belong to no ground truth item are not added into the divided aggregation.
Precision is therefore not defined for this division and is not displayed. This is to avoid presenting values that are
merely an artifact of how things are counted, such as "0 detections with 100% precision".

### The detector is rebuilt for each route

This is to match the workspace allocation to the dimensions of the image. Left at the default upper bound,
even a 640x480 scene would be measured while holding a region equivalent to 4K. That no allocation is added
during detection is confirmed by the difference in allocation counts before and after detection.

### Every scene with a difference is listed

Counts alone do not show under which condition it occurred, so the cause cannot be traced. Even when the count
grows, the list simply gets longer, leaving no room to select favorable results.

## Goals

- Save visualization images of the scenes where differences occurred
- Produce the same metrics with annotations of a real-image dataset as ground truth

## See also

- [Accuracy evaluation metrics](accuracy.md)
- [Diff report](../report/report_diff.hpp)
- [Synthetic corpus generator](../corpusgen/corpus_generator.md)
- [Evaluation plan](../../docs/evaluation-plan.md)
