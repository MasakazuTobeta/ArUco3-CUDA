# GPU cell decoding and Otsu recurrence

## Purpose

Reduce redundant work in the canonical-image Otsu recurrence without changing
its floating-point states, selected threshold, cell ratios or border decisions.

## Scope

The experiment applies inside one candidate's current-frame histogram. It uses
no previous-frame data, approximate arithmetic, histogram-bin merging or
parallel reassociation of the sequential recurrence.

## Current state

The baseline preserves the pinned OpenCV recurrence's order. In particular,
guarded iterations retain an unnormalized first moment, so skipping arbitrary
empty bins or replacing the recurrence with prefix sums changes its behavior.

The default-off `ARUCO3CUDA_EXPERIMENT_OTSU_FIXED_POINT=1` prototype checks whether
an actual empty-bin iteration leaves the first moment unchanged. If so, another
empty bin has the same probability mass and the same floating-point operands:
its computed state is identical. Subsequent empty bins therefore copy that state
until a populated bin resets the condition. Every bin remains represented in
the argmax, preserving the earliest-threshold tie rule. Guarded states are reused
only after their own multiplication has reached a fixed point; the guard is not
assumed to normalize the first moment.

All eight cell-decode tests passed on Spark and Orin, including the existing random
threshold oracle and 384 added sparse/skewed/uniform histograms across canonical
cell sizes 3, 4 and 7. Thresholds themselves, ratios and border errors are
compared with public OpenCV 4.14.0, not merely final detected IDs.

A preliminary Nsight trace of the four-marker 720p scene identified cell decoding
as the largest kernel cost. The baseline decode kernel median was 187.584 us;
the combined threshold/pyramid/fixed-point prototype recorded 60.448 us. These
are the last 20 graph executions after one startup and five warm-up calls in
separate instrumented runs on a busy Spark. They motivate further evaluation;
they do not establish isolated end-to-end speedup or a final strategy choice.
Local records remain under `build/gpu-study/*nsys-clean*` and
`build/gpu-study/otsu-fixed-point-tests.log`.

An isolated Orin CUDA-event probe compares the unchanged core, unconditional
fixed-point shortcut, and a density-selected shortcut. Three independent runs
alternate variant order, with 50 warm-up and 150 measured graph launches per
case. Inputs include 0, 1, 8 and 128 candidates and uniform, two-level,
eight-level and dense 256-level canonical images. Threshold and cell-ratio
hashes agree for every variant. The unconditional shortcut improves two-level
cases by 5.95--6.74 times and eight-level cases by 3.34--4.08 times, but makes
dense cases approximately 5--8 percent slower. These are isolated stage graph
intervals, not detector end-to-end speedups.

Experimental mode `ARUCO3CUDA_EXPERIMENT_OTSU_FIXED_POINT=2` selects the original
recurrence when occupied histogram bins exceed the configurable build value
`ARUCO3CUDA_OTSU_FIXED_POINT_MAX_OCCUPIED_BINS` (currently 96). An additional
22-case density sweep found gains at 96 occupied bins and losses at 128 with
the unconditional shortcut; the selector removes the dense-loop branch cost.
The cutoff is a performance policy only: both recurrences preserve the same
arithmetic states. A parallel warp-ballot occupancy count reuses the existing
centroid barrier; it removes the sequential selector scan without changing the
centroid arithmetic. The final selector passed all eight tests and all four
Compute Sanitizer tools on both Spark and Orin. In preliminary three-process
Spark stage measurements, dense-case ratios are approximately 0.996--1.002 and
two-level cases improve by approximately 6.9--8.3 times. Complete detector
measurements remain separate. The [maintained probe](../../tools/gpu_quality/cell_decode_probe.md)
records input/output hashes and every event sample for reproducible comparisons. It remains subject to paired Spark and complete-detector
evaluation. The tested earlier cutoff of 128 passed all four Compute Sanitizer
tools on Orin with eight tests and no errors or race hazards; final configuration
validation is recorded separately. Local records are `build/gpu-study/otsu-orin/`.

## Goals

- Preserve every Otsu state and output threshold while removing proven redundant
  arithmetic.
- Validate sparse and dense histograms, guarded states, ties and input changes.
- Compare the isolated option on Spark and Orin and retain the original route
  for ablation before selecting defaults.

## Open questions

- Does the occupied-bin selector avoid material regressions on both targets
  across different histogram layouts and candidate counts?
- Complete detector validation and paired timing are recorded in the
  [steady-state report](../../docs/gpu-steady-state-report.md); dense-workload
  stage gains do not imply universal whole-detector gains.

## Provenance

The fixed-point shortcut is independently implemented on top of the existing
repository recurrence. Its arithmetic contract and original OpenCV sources are
recorded in the [pipeline design](../../docs/design/detector-pipeline.md) and
[code provenance](../../docs/code-provenance.md).
