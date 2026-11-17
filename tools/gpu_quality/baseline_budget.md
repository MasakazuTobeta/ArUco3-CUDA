# Baseline benchmark resource adapter

## Purpose

Compare the unchanged `badb673` GPU algorithm with the current implementation
using the same explicitly recorded candidate and marker capacities.

## Scope

`baseline_budget.patch` changes only `bench/main.cpp` and
`bench/benchmark_harness.cpp` from `badb673`. Detector, kernels, public headers,
Dictionary data, image processing and timing intervals are unchanged.

## Current state

The original benchmark has no command-line capacity overrides. The adapter adds
validated `--max-candidates` and `--max-markers` options, restores their values
after the CPU-to-CUDA settings mapping, and records both capacities and an adapter
identity in each timing row. Defaults remain 4096 candidates and 1024 markers.
The completion check rejects marker truncation rather than accepting a partial
result as a valid timing row. It adds no new core API calls or profile getters.

Reproduce in a separate empty source directory; preserve the original checkout:

```bash
mkdir -p build/gpu-study/baseline-budget-source
git archive badb673 | tar -xf - -C build/gpu-study/baseline-budget-source
patch -d build/gpu-study/baseline-budget-source -p1 \
  < tools/gpu_quality/baseline_budget.patch
```

Configure and build that source using the target's normal toolkit and architecture,
with reference support enabled. Retain the source revision, patch SHA-256, full
compiler configuration, binary SHA-256 and input manifest beside local output.
Compare the adapter's default-budget output with the original binary before
using larger capacities. Verify that every file outside the two benchmark files
matches the original export byte for byte.

Increasing the baseline capacity also increases its capacity-sized kernel grids;
this is intentional when comparing equal resource budgets. Report the budget,
not a mixture of a 4096-candidate baseline and a 16384-candidate new route.
The baseline's known detection-quality deficiencies remain: a faster timing does
not make that algorithm CPU-equivalent. Its old candidate-overflow behavior is
also unchanged. Keep quality checks separate from latency measurements.

## Goals

- Preserve the baseline algorithm and measurement scope.
- Make resource budgets explicit and equal across GPU variants.
- Keep generated builds, source exports and measurement output untracked.

## Open questions

The adapter cannot retrofit the new asynchronous failure contract without
changing the baseline. Quality and capacity adequacy must therefore be checked
separately for every accepted timing input.
