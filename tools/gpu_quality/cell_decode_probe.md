# Cell decoding density probe

## Purpose

Reproduce isolated Otsu density-policy comparisons without attributing a stage
speedup to the complete detector.

## Scope

The probe uploads generated canonical images before measurement and captures
only `build_cell_ratios_async`. It measures CUDA events around each graph launch.
These intervals can contain scheduling gaps; they are neither summed kernel
execution durations nor host-to-result detector latency. Use a profiler for the
former and the maintained detector benchmark for the latter.

## Current state

Build `aruco3cuda_cell_decode_probe` with reference support and
`ARUCO3CUDA_BUILD_CPU_EXPERIMENT=ON`. Use separate build directories with
`CMAKE_CUDA_FLAGS=-DARUCO3CUDA_EXPERIMENT_OTSU_FIXED_POINT=N`, where `N=0` is the
original recurrence, `N=1` always enables the empty-bin fixed-point shortcut,
and `N=2` selects it by histogram density. See the
[algorithm record](../../src/core/cell_decode.md) for the current cutoff and
validation state. Specify the target CUDA architecture and OpenCV package as
in the normal repository build instructions.

```bash
cmake --build build/gpu-cell-control --target aruco3cuda_cell_decode_probe
cmake --build build/gpu-cell-adaptive --target aruco3cuda_cell_decode_probe
mkdir -p build/gpu-study/cell-decode
build/gpu-cell-control/tools/gpu_quality/aruco3cuda_cell_decode_probe control \
  > build/gpu-study/cell-decode/control-1.jsonl
build/gpu-cell-adaptive/tools/gpu_quality/aruco3cuda_cell_decode_probe adaptive \
  > build/gpu-study/cell-decode/adaptive-1.jsonl
```

Run at least three independent processes per variant and alternate their order.
Retain source hashes, compiler flags, CPU affinity, GPU load, clocks and power
mode with the local records. The probe records device/driver/runtime identity;
it does not establish that the GPU was idle or its clocks controlled.

Each process produces 48 records: active counts 0, 1, 8 and 128 crossed with
1, 2, 8, 16, 32, 64, 96, 128, 160, 192, 224 and 256 intensity levels. Images are
32x32 bytes (six marker cells, one border cell, four pixels per cell). The
one-level image is uniformly 127. Other intensities are deterministic evenly
spaced levels selected by `(i * 37 + i / 13) % levels`, where `i` traverses the
candidate buffer. They are synthetic histogram probes, not camera images.

An optional second argument sets a fixed candidate capacity from 1 through
16384. For example, `cell_decode_probe persistent 16384` crosses active counts
0, 1, 8, 128, 512, 4096 and 16384 with the same twelve intensity levels
(84 records). Smaller capacities omit larger counts and include the capacity
itself exactly once. This separates empty-block dispatch cost from useful work
and includes several waves of a bounded candidate grid. Use the same capacity
and active counts for both variants; the default 48-case density sweep remains
unchanged. The capacity limit bounds probe allocation, not the detector API. An optional
third argument selects one intensity-level count from 1 through 256 instead
of the twelve-level sweep: `cell_decode_probe persistent 16384 2` produces
seven records for two-level images. This supports targeted paired comparisons
of uniform (1), sparse (2) and dense (256) distributions without changing
the warm-up or sample count. Record this selection with the results.

Each case has 50 warm-up and 150 measured launches. All event samples are
retained in microseconds, with the repository's nearest-rank p50/p95/p99.
Allocations, upload, graph instantiation, output download and hashing are outside
the measured interval. Output SHA-256 covers active thresholds, cell ratios,
border errors and acceptance bytes in that order. Compare both input and output
hashes before accepting a performance comparison. A zero-count case hashes an
empty output and measures dispatch overhead; it is not a correctness positive
control. GPU absence or any CUDA error returns failure rather than a skipped
success.

Earlier exploratory records used the upper middle sample for p50 with an even
sample count. This maintained tool uses nearest rank and preserves every sample,
so fresh reports can recompute statistics without that ambiguity.

## Goals

- Preserve exact output identity while comparing sparse and dense distributions.
- Expose unfavorable cases and distinguish stage intervals from detector speed.
- Keep generated records outside version control.

## Open questions

Evenly spaced intensities do not cover every histogram layout. The reference
suite separately exercises sparse/skewed distributions, threshold ties and
nondefault canonical sizes. Complete-detector tests remain mandatory.
