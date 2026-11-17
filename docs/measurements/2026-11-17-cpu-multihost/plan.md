# Multi-host industrial background evaluation plan

## Purpose

Repeat the real industrial background CPU comparison on all six user-supplied
benchmark hosts, excluding startup latency.

## Scope

Use the unchanged implementation at `c5c943232da465a5b0e89aec046787d8144dbab9`,
the same 16 prepared photographs and positive marker control from the
[preceding evaluation](../2026-11-17-cpu-real-background/plan.md), and actual
OpenCV 4.14.0 at `0654a42e19215ef25b1d367d822f3c630447e7c7`.
This is a CPU experiment; GPU hardware is inventory only. No CUDA kernel time
or video throughput is measured. No detector implementation changes are planned.

## Current state

Generated measurement results, logs and environment captures referenced below
are local-only and ignored by Git. This document and reproduction sources remain
version-controlled; a fresh checkout does not contain the recorded outputs.

The six hosts are DGX Spark (.23), Jetson AGX Orin (.30), Intel N97 (10.26),
Xeon w7-3565X (.41), Jetson AGX Thor (.20), and Core i9-12900KS (.44).
The frozen source/input archive contains a SHA-256 manifest. Each host checks
that manifest before execution. Release builds use native tuning disabled.
Record per-host compiler, OpenCV build information, dispatch and library hashes;
the same source does not imply identical binaries across architectures.

Run the following groups serially on each host, independently across hosts:

- Four positive controls: ArUco3 enabled/disabled, one/four threads. Require
  four detections and exact baseline agreement.
- Two eligibility groups: all 16 photos, both detection modes, four threads.
- Primary: ArUco3 enabled, one/four threads, three independent processes,
  30 warm-ups and 200 measured calls per route/image/process.
- Secondary: ArUco3 disabled, one/four threads, three independent processes,
  10 warm-ups and 50 measured calls per route/image/process.

Use DICT_ARUCO_MIP_36h12 and the existing detector defaults for each mode.
Pin four threads to four distinct physical cores; use performance cores on
heterogeneous CPUs. Record actual affinity and topology. Alternate routes using
the unchanged benchmark. Decode, construction, validation and warm-up are outside
the host-grayscale-to-host-results timed interval. Validate initial and final
retained outputs; intermediate timed outputs are not separately retained.
Aggregate the median of process percentiles per image, then describe the image
speedup distribution. Retain p50/p95/p99, individual-process results and all losses.

Existing host services and power policies remain unchanged. Do not overlap
task builds with timings on the same host. Record background load and before/after
environment. Orin has a preexisting busy system service; core selection avoids
its observed core, but cannot prevent migration. Xeon and i9 also have existing
containers. Results describe these measured environments, not isolated hardware
ceilings. N97 has no build toolchain; use the same portable x86 binary/library
bundle as the other x86 hosts, retaining host libc compatibility.

## Goals

- Complete and independently audit 384 timed rows per host, 2,304 total.
- Preserve all 16 inputs without selection based on performance.
- Check zero detections, rejected counts, IDs/corners compatibility and positive
  controls on every host. Retain failures rather than silently weakening checks.
- Clearly separate per-host relative speedups from absolute latency and avoid
  claiming that one architecture's gains apply to all CPUs.
- Retain reproducible commands, source/input/build provenance and raw results.

## Open questions

- Cross-host OpenCV SIMD dispatch and compiler versions differ; report these
  differences rather than interpreting absolute latency as a CPU-only ranking.
- Shared workloads, thermal state and host clock policy may affect variability.
- These public photographs are not deployment-camera data or independently
  annotated marker ground truth; output compatibility is not general accuracy.
