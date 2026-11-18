# Bounded candidate grids

## Purpose

Avoid launching one block for every reserved candidate when the current frame
contains few or no candidates, while retaining device-only candidate counts.

## Scope

The prototype changes scheduling in canonical sampling, cell decoding and
Dictionary matching. Arithmetic, candidate order, output storage and candidate
capacity are unchanged. It never reads previous-frame image content.

## Current state

The default-off `ARUCO3CUDA_EXPERIMENT_PERSISTENT_CANDIDATES=1` route launches at
most `ARUCO3CUDA_CANDIDATE_GRID_BLOCKS` blocks (initial experiment: 128). Blocks
iterate candidate indices with a grid stride. A block barrier after each helper
call prevents a later candidate from overwriting shared memory still used by
another lane. The 64-bit loop index cannot overflow a positive signed-int
candidate capacity. Helper returns, including lane-zero Dictionary rejection, rejoin
that barrier. The control route retains one block per capacity slot.

Twenty-three sampling, decoding and Dictionary tests passed on both DGX Spark
and Jetson Orin with all four Compute Sanitizer tools reporting no errors or
race hazards. Added 515-candidate cases cross several block waves and alternate
accepted, rejected, uniform, nonuniform and unmatched candidates. Existing
Dictionary tests also cover 1000 candidates. Against the frozen generic-resize
contour/retry implementation, 263 Spark and 236 Orin audit calls preserve GPU
IDs, rotations and corner coordinates exactly. These scheduling comparisons do
not establish the final OpenCV resize-profile quality gate.

Three-process, 20-warm-up/200-iteration GPU-resident detector measurements with
16384 reserved candidates show lower latency for the tested blank 1080p,
four-marker 720p and warehouse scenes with the initial 128-block limit.
Three-process isolated probes on both targets show that 512 blocks avoid part
of the high-active-count penalty of 128 blocks, but full-capacity dense
histograms still favor the capacity control. The final candidate uses 512
explicitly and retains the capacity strategy; this is not a universal speedup.
A 32-bit induction/last-barrier-elision experiment passed the same tests and
sanitizers but gave no material gain and is not retained.

A preliminary stage sweep overlapped another task's correctness
run and its timing is invalid; only its matching input/output hashes are used.
Local records remain under `build/gpu-study/candidate-grid-*` and are not tracked.

The [cell decoding probe](../../tools/gpu_quality/cell_decode_probe.md) accepts
a fixed capacity and tests zero through full occupancy, preserving every timing
sample and output hash. This is an isolated stage interval, not complete-detector
latency. Use paired detector measurements for the latter.

## Goals

- Reduce unused-block dispatch without a host count download or synchronization.
- Preserve outputs across repeated block waves and changed current-frame data.
- Measure low and high active counts before choosing a common grid policy.

## Open questions

- Which grid size preserves useful occupancy on every maintained target?
- Does a common larger grid retain the empty-frame gain without dense regressions?
- The final combined detector and resize-profile quality gates pass on Spark,
  Orin and RTX 4090; full-detector timing and dense-workload qualifications are
  recorded in the [steady-state report](../../docs/gpu-steady-state-report.md).
