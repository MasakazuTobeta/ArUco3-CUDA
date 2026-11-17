# Paired steady-state detector benchmark

## Purpose

Compare official OpenCV, the optimized CPU implementation, the unchanged old GPU
algorithm and the optimized GPU candidate with reproducible input/resource
conditions and independently checked latency percentiles.

## Scope

`steady_benchmark.py` orchestrates existing benchmark executables. It does not
profile kernels or establish detection quality. Complete quality gates before
accepting timing results. Kernel profiling is a separate instrumented run.

## Current state

Supply three frozen executables, a directory of prepared PNGs with an
`input-manifest.json`, an unused output directory, four allowed logical CPU IDs,
and an expected-new-metadata JSON file. The manifest contains an `inputs` array
of `name`, `sha256`, and `family` objects; families are `synthetic` or
`industrial photograph`. The expected metadata file contains exactly
`gpu_resize_profile`, `gpu_resize_backend`, `gpu_carotene_x_fma`, and
`gpu_strategy_identity`, as selected by the build configuration. Metadata is
checked against every new GPU timing row; it is not inferred from timing output.

```bash
python3 tools/gpu_quality/steady_benchmark.py \
  --old-bench build/gpu-baseline-budget/bench/aruco3cuda_bench \
  --new-bench build/gpu-final/bench/aruco3cuda_bench \
  --cpu-bench build/gpu-final/cpu/aruco3cuda_cpu_benchmark \
  --inputs build/gpu-study/final-inputs \
  --expected-new-metadata build/gpu-study/expected-spark.json \
  --output build/gpu-study/final-benchmark-spark --cpu-list 5,6,7,8
```

The old executable must use the [benchmark-only budget adapter](baseline_budget.md).
The CPU executable must support `--save-samples`. Use the same native-tuning
policy as the reported CPU comparison and retain its compiler flags. `--dry-run`
writes the exact commands and manifest without executing a benchmark.

Defaults are three independent processes per variant, 20 warm-up frames and
200 latency samples. The variants are CPU with one/four threads (official and
optimized routes alternate internally), old/new GPU-resident, old/new GPU E2E,
and a separate old-default-budget GPU-resident control. Variant process order
reverses in the second repetition. Every route visits images in the same sorted
order. GPU work and the one-thread CPU route use the first supplied logical CPU;
the four-thread CPU route may use all four.

ArUco3-on visits every manifest image. ArUco3-off visits every synthetic image
and the first three industrial photographs in sorted filename order. This
selection is recorded before measurement. The maintained evaluation input set
contains seven synthetic and sixteen photographic PNGs, giving 23 on-mode and
10 off-mode images and 42 commands. A different manifest defines a different
experiment and must not be silently pooled with these results.

| Route | Measured interval |
| --- | --- |
| CPU | Host grayscale input to host detections |
| CUDA-Resident | Device-resident input; detection submission and stream completion, without host result download |
| CUDA-E2E | Pageable-host input upload, detection, synchronization and host result retrieval |

Initialization, loading, hashing and output serialization are excluded. Both GPU
intervals are wall latency, not kernel-only duration. No multiframe throughput
claim is made. Equal-budget GPU comparisons use 16384 candidates and 1024
markers; the separately labeled old-default control uses 4096/1024. The old
algorithm's detection-quality deficiencies remain, regardless of its latency.

The runner rejects failed processes, incomplete/duplicate rows, changed input
hashes, CPU incompatibilities, incorrect GPU budgets/profiles/scopes, and
percentiles inconsistent with retained samples. CPU samples use nine decimal
places; GPU samples use four, with the corresponding rounding tolerance. It
verifies that binaries and original inputs have not changed at completion.
Partial records remain available on failure; only `complete.json` denotes a
completed sweep. All generated records and copied inputs belong under ignored
local storage.

## Goals

- Preserve actual defaults and equal-budget comparisons as distinct evidence.
- Recompute p50/p95/p99 from individual samples rather than trusting summaries.
- Keep process variation and unfavorable cases visible.

## Open questions

Process-order reversal does not remove unrelated load or frequency drift. Record
host load, power policy, clocks, software identities and available memory before
and after the sweep. Compare per-image process summaries and retain ranges;
do not call this per-frame GPU pairing or pool per-process tail percentiles into
a single percentile. Every input still requires separate detection-quality and
resource-adequacy validation.
