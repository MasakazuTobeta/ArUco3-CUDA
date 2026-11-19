# Project guide

## Purpose

ArUco3-CUDA implements the ArUco3 detection strategy on the GPU. The CPU reference
is OpenCV's `cv::aruco::ArucoDetector`. ArUco3 speeds up detection by searching for
marker candidates in a downscaled image and refining corners at full resolution.
Downscaling trades off speed against the smallest marker the detector can find.

## Scope

Input is a single 8-bit grayscale image. Output is a marker ID, rotation and four
corners for each detection. Applications can use the corners for pose estimation.
Board refinement, pose estimation and ROS integration are outside the detector.
Supported marker families are listed in the [dictionary policy](dictionaries.md).

## Current state

| To… | Start here |
| --- | --- |
| Build the library and detect a marker | [Quick start](../README.md#detect-your-first-marker), [C++ and Python samples](../examples/README.md), [Docker profiles](design/docker-environment.md) |
| Integrate it into a GPU pipeline | [C++ API](design/public-api.md), [C ABI and Python](design/c-abi-and-python.md), [memory transfers](design/memory-transfer.md) |
| Understand or configure the detector | [Architecture](architecture.md), [detection stages](design/detector-pipeline.md), [build strategies](../cmake/ArUco3CudaStrategies.md) |
| Compare optimized CPU and steady GPU | [GPU latency, memory and corrected accuracy on Spark and Orin](gpu-steady-state-report.md), [CPU latency on six hosts](cpu-multihost-report.md) |
| Evaluate a host-only workload | [CPU algorithm](design/cpu-optimization.md), [independent-frame throughput](cpu-steady-state-report.md), [industrial backgrounds](cpu-real-background-report.md) |
| Interpret the original GPU and Hybrid results | [Four-host synthetic benchmark](benchmark-report.md), [accuracy against ground truth](accuracy-report.md) |
| Reproduce an evaluation | [Evaluation plan](evaluation-plan.md), [benchmark runner](../tools/gpu_quality/steady_benchmark.md), [measurement storage](measurements/README.md) |
| Contribute or review a change | [Contributing](../CONTRIBUTING.md), [implementation and coverage](implementation-plan.md), [terminology](terminology.md), [code provenance](code-provenance.md) |

## Goals

Keep detection results available to downstream GPU stages, verify accuracy against
ground truth and OpenCV, and publish reproducible measurements that show when CPU and
CUDA each perform well. The [architecture decisions](adr/0001-independent-implementation.md)
record why the work began independently of OpenCV.

## Open questions

How well does detection generalize to annotated camera footage, and which route gives
the best throughput on changing frames? See the [roadmap](roadmap.md).
