# Real industrial background evaluation plan

## Purpose

Measure whether the current optimized CPU detector beats the pinned actual
OpenCV detector on real industrial photographs with no detected markers.

## Scope

The detector implementation and existing binary are frozen at commit
`d677c4f8ff700064f27ef393b72b61a0121d53cc`; no optimization changes are part of
this evaluation. The benchmark SHA-256 is
`99581bfab75461f5e11bb9687966b2ba8d96248156dd5abd69f8a9209fc6bdcb`.
The machine is DGX Spark GB10, OpenCV 4.14.0 pinned as in the preceding CPU report.
Public photographs approximate industrial backgrounds; they are not recordings
from the user's deployment camera or independently annotated marker ground truth.

## Current state

Generated measurement results, logs and environment captures referenced below
are local-only and ignored by Git. This document and reproduction sources remain
version-controlled; a fresh checkout does not contain the recorded outputs.

Before detector execution, select 16 diverse real photographs by visible scene
content: warehouse racks/aisles, machinery and manufacturing floors, piping and
plant equipment, and simpler industrial interiors. Retain source URLs, author,
license, original hashes and exact preprocessing. Images remain local under
ignored data storage; only manifests, scripts and measurements enter Git.
Normalize to a 1920-pixel longest side without upscaling, preserving aspect
ratio and the entire field of view. Record dimensions and processed hashes.

Freeze the selection before any timing. Keep every selected photo regardless of
latency or detections. An initial one-sample/zero-warm-up comparison checks output
compatibility and zero-detection eligibility; its timing is not performance
evidence. Report any nonzero-output cases separately instead of silently dropping
them. A known positive synthetic marker scene confirms that the unchanged binary
and selected configuration still detect markers.

Primary measurements use DICT_ARUCO_MIP_36h12 and default ArUco3 settings, with
actual OpenCV and `optimized` routes, stage profiling disabled. Compare one
thread pinned to CPU 5 and four threads pinned to CPUs 5–8. Each group uses three
independent processes, 30 warm-ups and 200 timed calls per route/image. Routes
alternate order within a process. Run groups sequentially; do not overlap them
with builds or other agent detector executions. Retain unrelated host services
and power policy, recording the environment rather than claiming fixed clocks.

Secondary full-resolution measurements disable ArUco3 at the full processed-image
resolution (the normalized photo, not the original camera dimensions), using the same two core
budgets and three processes, with 10 warm-ups and 50 samples. These lower-sample
tails have limited resolution. Default no-refinement applies equally to both
routes in this mode. Report secondary results separately from the primary mode.

Measure host grayscale image to host IDs/corners; image decoding, setup,
validation, warm-up and rejected-output collection by the OpenCV oracle are
outside timing. This is detector-call latency, not camera-to-result latency or
video throughput. There is no CUDA kernel-only timing.

## Goals

- Compare exact IDs and per-marker oriented corners after matching detections,
  and exact rejected counts, with no weakened tolerance. The existing benchmark
  validates before warm-up and after the timed batch; it does not retain every
  intermediate timed result.
- Verify complete record counts and input/source/binary hashes in every run;
  confirm zero detections consistently, not only on the first process.
- Retain per-scene p50/p95/p99 and process spread, including all losses.
- Aggregate medians across process percentiles per scene; never pool samples.
- State results for the tested scene set/configuration without a universal
  factory-background speed or detection-accuracy claim.

## Open questions

- Actual deployment resolution, exposure, motion blur, camera compression and
  background distribution are not represented by a controlled camera capture.
- Other dictionaries, CPUs and video scheduling policies remain outside scope.
