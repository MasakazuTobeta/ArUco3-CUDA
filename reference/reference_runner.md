# reference_runner

## Purpose

Runs OpenCV's ArUco detection with fixed settings and saves the results in a machine-readable format. This produces the CPU baseline results used for differential comparison against the CUDA implementation and for measuring the crossover point.

## Scope

Covers detection on a single image using a predefined Dictionary, the mapping of detection settings, computation of the effective downscale factor, and JSON output of the results. Pose estimation, board detection, and switching the corner refinement method are out of scope.

## Current state

- Supports 18 predefined Dictionaries.
- Processes one or more still images loaded as 8-bit grayscale.
- Output is JSON with `schema_version` 1.

## Design decisions

### Detection results are sorted

The order returned by OpenCV's `detectMarkers` depends on the order in which candidates were extracted. This is awkward for differential comparison, so results are output with a stable sort by ID, then the x and y of the first corner.

### Rotation is not output

ArUco marker rotation does not appear as a separate item in the return value of `detectMarkers`. Rotation is expressed as the ordering of the corners. Guessing a value and outputting it would be a source of error in the baseline results, so the corner ordering is recorded as is and comparison is done by that ordering.

### The effective downscale factor is recorded

The `fxfy` of the ArUco3 detection strategy is computed with the formula recorded in the [detection pipeline design](../docs/design/detector-pipeline.md) and output together with the dimensions of the segmentation image. The default value of `minMarkerLengthRatioOriginalImg` is 0.0, in which case no downscaling occurs even with `useAruco3Detection` enabled. Without this value on record as a measurement condition, it is impossible to tell afterwards whether the effect of ArUco3 was measured at all.

### Execution time can be omitted

Specifying `--omit-timing` suppresses the `detect_ms` output. Execution time varies from run to run, so use this option for byte-level comparison against a golden file. Measuring time itself is the responsibility of the benchmark harness.

### The thread count is fixed to 1 by default

Reproducibility takes priority, so `cv::setNumThreads(1)` is the default. Specifying `--threads 0` follows OpenCV's default. In either case the actual thread count is recorded in the output.

### The checksum of the input is recorded

The SHA-256 of the input is recorded so that the correspondence between the result JSON and the input image can be verified afterwards. The implementation is `aruco3cuda::util::sha256_file`.

## Goals

- Allow baseline results, together with environment information, to be generated in bulk for the synthetic corpus and for real images.
- Share the same schema as the output of the CUDA implementation, so that the diff tool can read both through the same path.
- Allow the corner refinement method to be selected, to widen the range of comparison conditions.

## See also

- [Detection pipeline design](../docs/design/detector-pipeline.md)
- [Evaluation plan](../docs/evaluation-plan.md)
- [Implementation plan](../docs/implementation-plan.md)
- [Code provenance record](../docs/code-provenance.md)
