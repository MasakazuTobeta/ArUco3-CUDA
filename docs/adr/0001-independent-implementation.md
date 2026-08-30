# ADR-0001: Develop the CUDA implementation first in an independent repository

- Status: Accepted
- Date: 2026-08-27

## Purpose

This ADR records the decision to validate technical viability, API, correctness, and performance in an independent repository before submitting a large CUDA implementation directly to OpenCV itself.

## Scope

This covers the CUDA implementation of the ArUco3 detection strategy, evaluation on DGX Spark and Jetson Orin, and the decision on a future contribution to OpenCV.

## Background

- OpenCV's `cv::aruco::ArucoDetector::detectMarkers` has no CUDA implementation.
- OpenCV Issue #27118 asks for a CUDA implementation, but the API, module, and implementation scope have not been agreed on.
- ArUco3 includes processing that is a poor fit for the GPU, such as contours, candidate consolidation, and variable-length output.
- Even in a unified memory environment there are costs for synchronization, cache, and kernel launches, so CUDA is not always faster.

## Decision

We will develop `ArUco3-CUDA` as an independent validation and reference implementation.

1. Iterate on the algorithm and API independently of OpenCV's review constraints.
2. Evaluate CPU, CUDA, and hybrid paths on the same inputs.
3. Target both DGX Spark GB10 and Jetson Orin.
4. Test OpenCV compatibility while minimizing the core's dependence on OpenCV types.
5. Present the results to Issue #27118 once viability is confirmed, and agree on the upstream API.
6. Do not copy or adapt the GPLv3 code of official ArUco; implement independently from papers, public specifications, and permissively licensed reference implementations.
7. License this repository and its contributions under the Apache License 2.0.

## Rationale

- Performance viability can be confirmed before a large PR.
- It reduces interface rework during OpenCV review.
- It lets us design appropriate backend selection, including the conditions under which the CPU wins.
- It demonstrates portability across Ampere and Blackwell rather than a Jetson-specific implementation.

## Consequences

### Benefits

- Experiments, measurements, and API changes can be made quickly.
- Reproducible benchmarks and a test corpus can accompany the upstream proposal.
- The work remains usable as a standalone library even if OpenCV does not adopt it.

### Drawbacks

- Later work is needed to port to OpenCV's module structure and API.
- The independent implementation and the upstream implementation may diverge.
- In the early stages it cannot be used directly from an OpenCV package.

## Conditions for moving to OpenCV

- Correctness differences are explainable.
- There is a performance advantage under practical conditions, including transfers or synchronization.
- Builds, tests, and benchmarks are reproducible on DGX Spark and Jetson Orin.
- The permissive license and the code's provenance are clear.
- Any patent clearance required for commercial use is complete.
- The module, branch, and API have been agreed on with the maintainers.

## Open questions

- Whether the upstream destination is `opencv` itself or `opencv_contrib`.
- The concrete API for `cv::cuda::ArucoDetector`.
- Whether to target OpenCV 4.x or 5.x first.

## See also

- [Project overview](../project-overview.md)
- [Architecture](../architecture.md)
- [Evaluation plan](../evaluation-plan.md)
- [Intellectual property and licensing policy](../ip-and-licensing.md)
- [OpenCV Issue #27118](https://github.com/opencv/opencv/issues/27118)
