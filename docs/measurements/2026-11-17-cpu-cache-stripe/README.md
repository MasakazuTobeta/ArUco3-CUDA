# Cache-stripe threshold measurements

## Purpose

Record the bounded stripe-size sweep and paired prefix-budget confirmation that
selected the experimental CPU detector's 2 MiB threshold prefix budget.

## Scope

These are threshold microbenchmarks on DGX Spark GB10, not whole-detector
measurements. The interval includes padding, prefix generation, row dispatch and
all three output planes. Input generation, loading and correctness comparison
are excluded. CUDA kernel time is not applicable. Conditions, compiler flags,
affinity, source hashes and limitations are in conditions.json (local `conditions.json`).

## Current state

Generated measurement results, logs and environment captures referenced below
are local-only and ignored by Git. This document and reproduction sources remain
version-controlled; a fresh checkout does not contain the recorded outputs.

The initial `whole-*.csv` and `tiled-*.csv` files sweep row caps
32/64/128/256/512 and 0 (overflow-only stripes), comparing whole-image and
per-stripe padding. Each variant used ten warmups and forty samples in one
process; variants were measured sequentially. The `scratch_bytes` column reports
actual padded/prefix owner payloads from that scratch implementation, excluding
outputs, matrix headers and allocator capacity.

The six `confirm-{1,4}t-{1,2,3}.csv` files use three independent processes per
thread count, ten warmups and eighty measured frames, rotating the first route
within each frame. They compare per-stripe padding at budgets 2 MiB and 4 MiB
against legacy budget 0, whose padding remains whole-image. Affinity is CPU 5
for one thread and CPUs 5-8 for four threads. OpenCV 4.14.0 and GCC 13.3.0
`-O3 -DNDEBUG -std=c++17` were used without architecture-specific flags.
Nearest-rank p50/p95/p99 are retained for every process.

The confirmed 2 MiB setting was faster than 4 MiB and legacy geometry for both
full-resolution sizes with one and four threads. Its 427x240 segmentation-image
time stayed within measurement noise. It reduces padded/prefix payload at
3840x2160 from 25,470,440 to 2,526,272 bytes; this is not a total-RSS guarantee.
See [the threshold design](../../../cpu/cpu_threshold.md) for the aggregated
table, geometry, compatibility checks and retained fallback behavior.

The scratch probe ran before the maintained helper gained input-alias and
tiny-budget owner-release checks. Those guards are included by the archived
reproduction runner. They do not alter stripe geometry or normal nonaliasing
pixel decisions; end-to-end detector measurements are still required to assess
the resulting implementation's latency and RSS. Other host services remained
active, and clock/power settings were not changed by the probe.

### Reproduction

The maintained standalone runner is
[`cpu/experiments/cache_stripe_probe.cpp`](../../../cpu/experiments/cache_stripe_probe.cpp).
Build instructions and one/four-thread commands are in
[the experiment archive](../../../cpu/experiments/README.md).
Its default correctness mode checks every output plane against OpenCV on white
and random strided ROIs, including an odd 3841x2161 shape. Run benchmark mode in
three independent processes to reproduce the aggregation method. The runner
prints planned payloads from the maintained geometry rather than reading private
workspace storage.

## Goals

- Keep per-process percentiles and conditions reviewable.
- Preserve a small reproduction runner using the maintained threshold helper.
- Distinguish payload reductions and threshold timing from whole-process RSS
  and detector latency.

## Open questions

- Whether the same cache budget is beneficial on Jetson Orin and x86_64.
- Whether another budget is preferable for different window sets or input shapes.
