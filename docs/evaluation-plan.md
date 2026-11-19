# Evaluation plan

## Purpose

Compare detector quality, latency and memory under reproducible conditions. Identify the inputs and hardware on which each route is faster.

## Scope

Unmodified OpenCV 4.14.0 provides the CPU compatibility baseline. Rendered markers provide ground truth for accuracy. The measured interval ends at marker IDs and ordered corners. Pose estimation is outside the evaluation.

## Current state

Use the report for the implementation and workload being reproduced. Each identifies its source, hardware and inputs.

| Study | Inputs and scope | Results and reproduction |
| --- | --- | --- |
| Historical component GPU, Hybrid and official CPU | Synthetic corpus on DGX Spark, Jetson AGX Orin, Jetson AGX Thor and RTX 5070 Ti. Accuracy uses 91 scenes. The latency sweep uses 28. | [Accuracy](accuracy-report.md) and [benchmarks](benchmark-report.md) |
| Optimized CPU and official CPU | Synthetic and degraded markers, independent-frame worker throughput, and still photographs on six hosts. The photographic study uses 16 industrial backgrounds and positive controls. | [CPU steady state](cpu-steady-state-report.md) and [six-host comparison](cpu-multihost-report.md) |
| Ordered-contour GPU, component GPU and both CPU implementations | Changing-frame quality audit, historical images, 17 Dictionaries and industrial photographs. Latency uses repeated still images on Spark and Orin. Quality after the border correction was verified on those two hosts. | [GPU steady-state report](gpu-steady-state-report.md), [quality matrix](../tools/gpu_quality/run_matrix.md) and [timing runner](../tools/gpu_quality/steady_benchmark.md) |

### Routes and timing boundaries

| Route in the benchmark | Input | End of measured interval |
| --- | --- | --- |
| `CPU` | Prepared host grayscale `cv::Mat` | OpenCV has returned host results. The optimized CPU study times the same boundary. |
| `Hybrid` | Host or device image, according to the memory kind | GPU preprocessing and synchronization plus CPU candidate extraction and decoding have completed. |
| `CUDA-Resident` | Device image | Detection on the submitted stream has completed with results on the device. Host download is excluded. |
| `CUDA-E2E` | Host image | The detector has processed the input and returned results to the host, including transfer and synchronization. |

Record input memory separately. `M-Pageable` uses ordinary host memory, `M-Pinned` uses a page-locked buffer prepared before timing, `M-Managed` allows migration on device access, and `M-Device` starts with resident pixels. The [measurement harness](../bench/benchmark_harness.md) specifies supported combinations and JSONL identifiers.

File decoding, checksums, initialization and allocation occur before steady-state timing. Record time to first result and CUDA context creation separately for a cold-start study. Report single-frame p50/p95/p99 and measure multi-frame throughput directly in frames/s. Worker service latency excludes queue wait. Repeated still-image calls measure latency with a persistent detector. A changing-frame correctness test supplies no video throughput measurement.

Stage wall times include host synchronization. Label kernel-only diagnostics by instrument and scope. The historical harness has no per-stage CUDA-event times. The steady-state study separately reports sums of GPU kernel durations from a Spark Nsight Systems trace. [Kernel diagnostic](gpu-steady-state-report.md#separate-kernel-diagnostic)

### Inputs and accuracy

The historical `full` corpus uses 8-bit grayscale at 640×480, 1280×720, 1920×1080 and 3840×2160, with 0, 1, 4 or 16 markers and `DICT_ARUCO_MIP_36h12`. Clean marker sides are 16, 32, 64, 128 or 256 px. Degraded scenes use 96 px markers and vary rotation, perspective, blur, noise, illumination, occlusion and image borders, alone or in combination. The generator skips layouts that cannot fit. [Corpus generator](../tools/corpusgen/corpus_generator.md)

Match detections by centroid proximity. Record misses, extra detections, incorrect IDs, rotation from cyclic corner order, and maximum corner error and RMSE. For synthetic images also report precision, false positives and recall overall, at or above the ArUco3 size bound, and below it. Compare per scene and by marker size and degradation. Unannotated photographs establish agreement with CPU. [Accuracy metrics](../tools/evaluate/accuracy.md)

For the steady GPU quality gate, check every CPU-detected ground-truth marker, false positives, IDs, rotations and per-truth corner errors. Record raw deltas and the declared 0.001 px numerical classification band. Exercise graph and direct submission, including different frames at the same device address. Keep overflow and partial outputs as failures. The checkerboard needs more than the default 4,096 candidates, so this audit uses 16,384 candidates and 1,024 markers. [Acceptance conditions](gpu-steady-optimization-plan.md) and [audit procedure](../tools/gpu_quality/audit.md)

### Measurement procedure

1. Record the source and binary identities, input hashes, hardware, OS, driver, CUDA, compiler and OpenCV build. Pin the toolchain container. Record power mode, clocks, shared workloads, CPU topology, affinity and ASLR state. Pin by core **type**. For example, CPU 0 is an efficiency core on Spark.
2. Fix the prepared pixels, Dictionary, detector configuration, effective scale `fxfy`, resource budgets, numerical profile and input memory. OpenCV's default `minMarkerLengthRatioOriginalImg=0` makes `fxfy=1` even with ArUco3 enabled. The historical CPU baseline uses one thread. The steady studies also compare four threads.
3. Check CPU compatibility and synthetic ground truth before timing. On a shared-memory GPU, capture `cudaMemGetInfo` and host memory/load alongside the run. Preserve allocation failures or device skips and confirm the intended device executed on retry. The steady studies retained unrelated workloads and the existing page cache. The historical cold-start and memory conditions are recorded in the [benchmark report](benchmark-report.md#reproducing-the-measurements).
4. Record warmups, sample counts and independent process counts. Alternate implementations in the same session to resolve small effects. Report nearest-rank percentiles with process spread and retain all samples, including outliers. If ASLR is disabled with `setarch -R`, record it. Compute startup break-even only where the steady-state difference exceeds run-to-run variation.
5. Report fast and slow cases, peak workspace and per-frame allocations. Record utilization and power where available. Keep profiler runs separate from primary timing, and identify the source revision for each result.

A matching generator seed does not guarantee matching pixels across hosts. Historical checksums cover 28 benchmark images. Eighteen differed between aarch64 and x86_64 and six differed between the two compared aarch64 machines. Compare routes using the same prepared bytes on each host. [Reproducibility details](accuracy-report.md#corpus-reproducibility)

Commit conditions, source/input manifests, reproduction commands and concise findings. Keep generated rows, environment captures, differing images and logs in local storage. The [measurement policy](measurements/README.md) distinguishes inputs available in a fresh checkout from locally retained output.

## Goals

- Reproduce per-scene quality and run memory, initialization, race and synchronization checks on the target devices.
- Measure accuracy and crossover points with annotated deployment images, including resolutions below 640×480 and above 4K.
- Add per-stage CUDA-event timing and measure detections per unit of energy on hardware with suitable power telemetry.

## Open questions

- Which real-camera sequences and annotations are available under suitable distribution terms, and which application tolerances should govern corner error and speed?
- How do changing-frame GPU throughput, toolkit versions and clock policies affect crossover points under deployment load?
- What causes generated pixels to differ across build environments with the same seed?
