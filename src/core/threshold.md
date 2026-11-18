# GPU multiwindow threshold experiment

## Purpose

Reduce repeated image traversal and kernel launches in adaptive thresholding
without changing the binary pixels used by candidate extraction.

## Scope

The operation retains the repository's isolated `BORDER_REPLICATE` image
semantics, odd-window rounding, duplicate window outputs, nearest-even normalized
mean and `THRESH_BINARY_INV` comparison. It changes neither detection thresholds
nor frame selection, and never uses data from a previous image.

## Current state

The existing separable route computes a horizontal sum and then a vertical sum
for each threshold window. The default three windows therefore issue six kernels
and reuse one global-memory row-sum plane.

An experimental fused route loads a tile and the largest window's replicated
halo into shared memory. Disjoint row workers build horizontal prefix sums;
disjoint column workers then complete the integral tile. Every output thread
reads four integer values per window, applies the existing double normalization
and rounding, and writes its own pixel in each output plane. All threads,
including those outside a partial image tile, reach all three barriers before
any early return. Duplicate rounded window sizes remain separate outputs.

The steady presets select `ARUCO3CUDA_THRESHOLD_STRATEGY=FUSED_AUTO`; ordinary
builds retain the separable control. The earlier
`ARUCO3CUDA_ENABLE_FUSED_THRESHOLD=1` spelling is a compatibility option.
`ARUCO3CUDA_FUSED_THRESHOLD_MAX_WINDOW_PX` defaults to 63
and controls the experimental crossover limit. The block side retains
`DetectorConfig::cuda_block_dim_`. The fused launch additionally requires at most
48 KiB of dynamic shared memory; the bounded tile sum then fits in 32 bits.
Larger windows use the separable route. These are explicit build controls;
the [strategy record](../../cmake/ArUco3CudaStrategies.md) and
[whole-detector evaluation](../../docs/gpu-steady-state-report.md) retain their scope.

Both routes retain the existing workspace allocation layout so the experiment
does not combine kernel changes with allocation changes. The fused path does
not read or initialize the unused global row-sum plane.

### Large-window arithmetic

Configuration accepts a window maximum of 4096, which becomes 4097 after odd
rounding. A replicated all-white window sums to `255 * 4097 * 4097`, exceeding
signed 32-bit range regardless of input image size. The separable launcher
therefore chooses a 64-bit column accumulator only when the maximum possible
window sum exceeds `INT_MAX`; ordinary windows retain their 32-bit accumulator.
Horizontal sums still fit in 32 bits. This correctness condition is separate
from any fused-path performance claim.

### Verification plan

The threshold suite compares exact output bytes with the pinned OpenCV public
API, including random strided input, tiny and one-dimensional images, partial
tiles, halos larger than the image, fractional constants, duplicate/even
windows, block sides 4 through 32 and windows on both sides of the fused limit.
An all-white maximum-window case guards the overflow boundary.

Run the suite on both build routes and use Compute Sanitizer memcheck, racecheck,
initcheck and synccheck on the fused path. Then measure kernel-only and detector
end-to-end latency independently, including sparse and textured images, on
DGX Spark and Jetson Orin. Primitive timing alone cannot establish a detector
end-to-end speedup.

The isolated fused prototype passed all 11 threshold/window tests on DGX Spark
with CUDA 13.0 and pinned OpenCV 4.14.0. The new halo/ROI case covers 300
configurations and 1,020 exact binary-plane comparisons; the maximum-window
overflow case also agrees with OpenCV. The local record is
`build/gpu-study/threshold-tests.log`. This is a correctness result, not a
latency measurement or a full-detector validation.

All four Compute Sanitizer tools passed the two new boundary tests with zero
errors; racecheck also reported zero warnings/hazards. A separately linked copy
of the frozen baseline reproduced the maximum-window defect: a 3-by-2 white
image at constant zero produced zero white output pixels, whereas the repaired
route and OpenCV produce six. Those local records are under
`build/gpu-study/threshold-*.log` and
`build/gpu-study/threshold-probe/overflow-reproduction.log`.

### Initial stage diagnostic

On the busy DGX Spark development host, three independent paired processes used
the same deterministic random input, explicit streams, captured threshold
graphs, 30 warmups and 200 alternating samples per route. The table takes the
median of the three process p50 values. Both routes produced identical bytes
for every measured configuration. These are CUDA-event intervals around the
threshold graph, including gaps between its kernels, **not** sums of kernel-only
time or detector end-to-end latency. Unrelated GPU work remained active, so these
observations are provisional and do not establish isolated host performance.

| Input / windows / block side | Separable GPU interval | Fused GPU interval | Ratio, separable/fused |
| --- | ---: | ---: | ---: |
| 427x240 / 3,13,23 / 16 | 20.83 us | 12.61 us | 1.652x |
| 1920x1080 / 3,13,23 / 16 | 269.76 us | 137.86 us | 1.957x |
| 3840x2160 / 3,13,23 / 16 | 1526.11 us | 579.68 us | 2.633x |
| 427x240 / 3,33,63 / 16 | 35.10 us | 28.96 us | 1.212x |
| 427x240 / 63 / 16 | 20.70 us | 26.72 us | 0.775x |

The single-window regression is retained and argues against enabling the fused
route indiscriminately. The window-65 fallback remained at a 0.998x ratio.
The default three-window sweep also improved at tested block sides 8 and 32,
but its best measured fused configuration here used block side 16. Raw p50,
p95 and p99 records remain local in
`build/gpu-study/threshold-probe/run-{1,2,3}.jsonl`.

### Jetson Orin portability and stage diagnostic

The same isolated source and paired probe were built on Jetson AGX Orin with
CUDA 11.4.315, GCC 9.4 and pinned OpenCV 4.14.0. All 11 threshold/window tests
passed, and all four Compute Sanitizer tools passed the two new boundary tests
with zero errors/hazards. The frozen baseline's maximum-window overflow also
reproduced there. SHA-256 checks confirm that the candidate, baseline, maintained
test and probe sources match the Spark copies.

Three independent processes used the same 30 warmups, 200 alternating samples
and captured graphs. The process was pinned to CPU 8, within the container's
8–11 budget. MAXN and the host's dynamic clock policy were retained. GPU activity
was zero in the pre-run telemetry; unrelated host services remained running,
including a busy CPU process that can migrate into the measurement affinity.
These conditions are observations, not exclusive host access or locked clocks.

| Input / windows / block side | Separable GPU interval | Fused GPU interval | Ratio, separable/fused |
| --- | ---: | ---: | ---: |
| 427x240 / 3,13,23 / 16 | 94.11 us | 40.99 us | 2.296x |
| 1920x1080 / 3,13,23 / 16 | 1758.30 us | 659.39 us | 2.667x |
| 3840x2160 / 3,13,23 / 16 | 6901.54 us | 2600.58 us | 2.654x |
| 427x240 / 3,33,63 / 16 | 191.62 us | 84.61 us | 2.265x |
| 427x240 / 63 / 16 | 114.66 us | 79.84 us | 1.436x |

The single-window-63 case improves here despite regressing on Spark, so a final
selection rule cannot assume one universal crossover. The window-65 fallback
remains at a 0.998x ratio. Default windows at block sides 8 and 32 improve by
1.288x and 1.879x respectively. Raw percentiles, build/validation logs, source and
binary hashes, and host telemetry remain local under
`build/gpu-study/threshold-orin/`.

## Goals

- Preserve exact threshold pixels and asynchronous stream behavior.
- Reduce redundant memory traffic and launches where measured beneficial.
- Keep the previous route available for accuracy and performance ablation.
- Retain explicit strategy and crossover controls in evaluated presets.

## Open questions

- Shared-memory prefix construction has serial row/column work and may cost more
  than it saves, especially for large halos or small output blocks.
- Shared-memory occupancy and the best crossover can differ between devices.
- The final combined detector is evaluated separately in the steady-state
  report; isolated primitive gains must not be presented as whole-pipeline gains.

## Provenance

This is an independent implementation of integer summed-area tiles using the
repository's existing threshold semantics. It uses no external implementation
source. The CPU shared-window experiment supplied the work-sharing hypothesis;
OpenCV 4.14.0 is the public-API comparison oracle. See the
[CPU threshold sidecar](../../cpu/cpu_threshold.md) and
[pipeline design](../../docs/design/detector-pipeline.md).
