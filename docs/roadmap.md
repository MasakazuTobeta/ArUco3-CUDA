# Roadmap

## Purpose

Identify the evidence still needed to choose a detector for deployment and to
assess a possible OpenCV contribution. These goals have no scheduled dates.

## Scope

Detection quality, latency, throughput and hardware coverage for independent frames.
The [implementation plan](implementation-plan.md) tracks implementation and test coverage.

## Current state

The steady GPU and optimized CPU detectors have been compared on synthetic scenes
with known markers and industrial photographs with zero detections. The CPU study
also measured independent-frame workers. GPU timings cover repeated calls on fixed
images. [GPU study](gpu-steady-state-report.md) · [CPU study](cpu-steady-state-report.md)

## Goals

- **Field accuracy and video throughput.** Evaluate annotated footage with visible
  markers, including occlusion, motion and changes in marker size. Measure sustained
  throughput and frame latency with the same inputs and detection settings on CPU and GPU.
- **GPU timing.** Extend kernel profiling across hardware and evaluate per-stage CUDA
  events alongside wall-clock measurements. Assess startup separately for short runs.
  [Evaluation plan](evaluation-plan.md)
- **Dictionaries.** Assess the speed and storage cost of extending the 17 bundled
  dictionaries. [Dictionary policy](dictionaries.md)
- **Upstream contribution.** Assess maintenance cost and the value to OpenCV users
  before proposing an integration. [Original decision](adr/0001-independent-implementation.md) ·
  [OpenCV discussion](https://github.com/opencv/opencv/issues/27118)

## Open questions

- How does marker density in camera footage change the CPU/GPU crossover point?
- Which corner-error and latency limits matter for the target application?
- Which additional dictionaries and hardware warrant validation, and under which clock and power settings?
