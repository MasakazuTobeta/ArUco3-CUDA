# benchmark_harness

## Purpose

Records the measurement conditions and the execution environment together with the results, so that performance data is kept in a form that can be reproduced and compared later. CPU, CUDA, and hybrid are measured within the same framework, so that the crossover point can be shown, including the conditions under which CPU is favorable.

## Scope

Covers latency and throughput measurement, statistics computation, environment information collection, and JSONL output. Visualization is the responsibility of `aggregate.py`.

## Current state

- The four routes `CPU`, `Hybrid`, `CUDA-E2E`, and `CUDA-Resident` can be measured.
- Four memory types can be measured: `M-Pageable`, `M-Pinned`, `M-Managed`, and `M-Device`. Some combinations of route and type are not allowed; `CUDA-E2E` is a route with host input and therefore does not accept `M-Device`.
- Kernel time (CUDA events) is not measured on any route. Stage times (`stages`) are wall-clock and include host synchronization. To keep the two distinct, `kernel` is never filled in with stage times.

- The measured results are in the [benchmark report](../docs/benchmark-report.md).

## Design decisions

### Image loading is not included in the measured interval

The `CPU` route repeats detection alone on an already loaded image. If `cv::imread` and `sha256_file` were called once per iteration, PNG decoding would account for 58% to 85% of the measured interval for the 1280x720 PNGs of the synthetic corpus. Real-time processing does not decode PNGs, so this would not be a valid comparison of detection time.

With this change, `p50` values are smaller than before. `schema_version` has been raised to 3 so that results from version 2 and earlier cannot be aggregated together with these.

### The recorded CUDA Toolkit version includes the patch version

`nvcc --version` prints `Cuda compilation tools, release 13.0, V13.0.88`, and the
capture used to stop at the comma. A measurement file could then not establish on
its own which Toolkit produced it. It is not a hypothetical gap: of the four
machines measured, the Jetson AGX Thor runs `13.0.48` and the GeForce RTX 5070 Ti
`13.0.88`, and both were recorded as `13.0`.

`cuda_toolkit` now carries the `V` token, falling back to the release field when
there is none. No key was added or renamed and no measured interval moved, so
`schema_version` 4 and 5 are aggregated together; the reasoning is with
`SUPPORTED_SCHEMA_VERSIONS` in `aggregate.py`.

### Startup cost is recorded separately

Warm-up is separated from the measured interval, but it is not discarded. The time until the first result is available (`startup.time_to_first_result_ms`) and the time for detection on the first image alone (`startup.first_frame_ms`) are recorded. CUDA context creation happens only once per process, so it goes into the environment information (`cuda_context_ms`) rather than into the per-route measurements.

Measured on DGX Spark, the CPU route takes 3.2 ms to the first image and the hybrid route takes 202 ms. The steady-state difference is 0.13 ms, so about 1600 frames are needed to break even. Looking only at the percentiles after warm-up, this difference does not appear in the results.

When several images are measured in one process, only the first image bears the cost of context creation and kernel loading. To measure startup cost, run one image per process.

### The measured interval differs by memory type for hybrid

`M-Device` assumes the image is already on the device and places the transfer outside the measured interval. This corresponds to the upper bound for a configuration where the camera feeds the GPU directly. `M-Pageable` transfers the host image every frame and includes the transfer in the measured interval. This is the value for starting from a host image, as the `CPU` route does.

### Unimplemented routes are not substituted with CPU

When an unimplemented route such as `--route CUDA-E2E` is specified, the run fails rather than silently falling back to the CPU route. A silent fallback would make the reported `route` disagree with the processing actually measured, and this could not be noticed afterwards.

### Route and memory type are independent axes

DGX Spark and Jetson Orin are both integrated GPUs, and the cost of an explicit copy differs greatly from a discrete GPU. `M-Pageable`, `M-Pinned`, `M-Managed`, and `M-Device` are recorded as an axis separate from the route. The identifiers follow the notation in the [evaluation plan](../docs/evaluation-plan.md).

### Latency and throughput are measured separately

Latency is measured one frame at a time, independently; throughput is derived from the total time of continuous processing. They are different metrics, and neither is converted from the other.

### Kernel time is not filled in with 0

There is no kernel time on the CPU route. It is `null` in JSON, not 0. Writing 0 would be misread during aggregation as "a very fast kernel".

### Measurement conditions are fixed down to the core type

The DGX Spark GB10 is a mix of Cortex-X925 (performance) and Cortex-A725 (efficiency). Under identical conditions, the assigned core type produces a 1.64x difference, so `--cpu-list` is provided to pin it. The core configuration and the actual affinity are recorded in the results. The Jetson AGX Orin has a uniform configuration of 12 Cortex-A78AE cores and is not subject to this effect.

### Run-to-run variance is not confused with percentiles within a single run

On the CPU route across all resolutions, p50 varies by 9% from differences in per-process memory layout (ASLR) alone. Disabling it with `setarch -R` was confirmed to make runs match exactly. The p50, p95, and p99 within a single run do not capture this variation.

For this reason the ASLR state is recorded in the results, and `aggregate.py` collects multiple runs under identical conditions and displays the run-to-run variance. Measurements are performed several times as independent processes.

### No claim is made about which is larger when the difference between routes is below the noise

The difference between `CUDA-Resident` and `CUDA-EndToEnd` is only the transfer to and from the host and the retrieval of the results.
At 640x480 the transfer is 307 KB, which is under 10 us at the bandwidth of an integrated GPU. Against the 1 ms of detection,
that does not reach 1%. In actual measurements the variation from clocks ramping up is larger, and
the ordering flipped depending on the order in which they were measured.

```
min Resident 1.071 ms / EndToEnd(pageable) 1.493 ms / EndToEnd(pinned) 1.208 ms
min Resident 1.228 ms / EndToEnd(pageable) 1.110 ms / EndToEnd(pinned) 2.262 ms
min Resident 1.226 ms / EndToEnd(pageable) 1.242 ms / EndToEnd(pinned) 1.160 ms
```

The tests therefore make no claim about which is larger; they check only that **the three routes produce the same detection results**,
and the times are displayed. Comparison at a magnitude where the difference is visible is the job of measurements on real machines (docs/measurements).

The "bracketing by time" of the previous section does not help either. Bracketing makes the minimum on the baseline side come from the most warmed-up
moment, which introduces a bias in the opposite direction. When the difference itself is smaller than the noise, no refinement of the measurement method
settles the direction.

### The measured interval of the full GPU route includes stream synchronization

`Detector::detect_async` only issues kernels and returns. Without including synchronization, only the cost of issuing
would be measured. `CUDA-Resident` calls `cudaStreamSynchronize` at the end of the interval, and
`CUDA-EndToEnd` synchronizes in `download`. Both therefore measure "from a GPU-resident image to the results".

### Tests that claim an ordering of times bracket the measurement by time

Claims written into tests such as "including loading is slower" or "the first image is slower than steady state"
flip direction when the machine is busy. Running `ctest -j 8` 15 times on a Jetson AGX Orin,
`cpu_route_excludes_image_loading` failed 5 times. The values on failure
were "detection only 5.308 ms, with loading 2.898 ms" — not noise, but a reversal of nearly 2x.

The cause is the difference in the time at which the measurements were taken. The Jetson's operating frequency moves a great deal under load, so
the speed of the machine itself differs between an earlier sample and a later one. Neither the median nor the minimum
resolves this as long as samples taken at different times are compared.

The baseline measurement is now taken twice, before and after the harness measurement, and compared against the minimum of both.
Whichever way the frequency moves, a sample taken at a nearby time always remains on one side. The sample count for this test
alone was also raised from 5 to 15. With 5, the minimum is left to chance.

A decision was also made to test only relationships that do not depend on load. `records_startup_cost`
claimed that "the first image is slower than steady state because the cache is cold", but the startup cost of the CPU
route is only a few ms and reverses easily. Only the relationship that holds by definition
(time to the first image >= detection time of the first image) was kept, and the values are displayed so they can be inspected
by eye.

Note that the same investigation also resolved the issue where reference tests fail when
`compute-sanitizer` is run 4 at a time in parallel. That one was not about time but a collision on a fixed path in `/tmp`,
resolved by putting the process number into the path.

### Outliers are not removed

Statistics are computed from all samples. Percentiles are obtained by the nearest-rank method, without interpolation. The returned value is always one of the measured values. If the aggregation method were implementation-dependent, comparison across environments would not hold. See `aruco3cuda::util::compute_statistics` for details.

Specifying `--save-samples` includes all samples in the results. Use it when the distribution itself needs to be saved.

### The downscale factor is always recorded in the measurement conditions

The effective downscale factor `fxfy` of ArUco3 is recorded as a condition. The default value of `minMarkerLengthRatioOriginalImg` is 0.0, in which case no downscaling occurs even with `useAruco3Detection` enabled. Without this value on record, it is impossible to tell afterwards whether the effect of ArUco3 was measured at all.

### Environment information is obtained from libraries wherever possible

The GPU name, Compute Capability, and whether the GPU is integrated are obtained from CUDA. This allows them to be recorded even in a container without `nvidia-smi`. The driver version and the Jetson power mode cannot be obtained from a library, so external commands are used; when they cannot be obtained, the fields are left as empty strings. They are not filled in by guesswork.

## Goals

- Record kernel time from CUDA events separately from wall-clock.
- Record peak device memory and the number of allocations per frame.
- Allow a sweep over resolution, marker count, and side length in a single run.
- Establish a measurement procedure with the clocks and power mode fixed.

## See also

- [Evaluation plan](../docs/evaluation-plan.md)
- [Implementation plan](../docs/implementation-plan.md)
- [reference_runner](../reference/reference_runner.md)
