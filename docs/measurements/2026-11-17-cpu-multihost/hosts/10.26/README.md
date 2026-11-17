# Intel N97 CPU measurement

## Purpose

Measure the unchanged detector and pinned OpenCV on the four-core Intel N97.

## Scope

Host 192.168.10.26, Ubuntu 26.04.1. Affinity is CPU 0 for one thread and CPUs
0–3 for four threads. Inputs and measurement conditions follow the common plan.

## Current state

Generated measurement results, logs and environment captures referenced below
are local-only and ignored by Git. This document and reproduction sources remain
version-controlled; a fresh checkout does not contain the recorded outputs.

All ten groups completed, including 384 timed measurement rows, four positive
controls and two eligibility groups. All 16 photos returned zero detections;
the known positive control returned four markers in both modes/thread budgets.

The host has no compiler, CMake or Docker on its default PATH. The same portable
x86 Release binary and dependency bundle built on the i9 host were extracted
under `/home/tobeta/aruco3-cpu-multihost-20261117`; no system packages were installed.
GCC 13.3 compiled the detector with native tuning disabled. The bundled OpenCV
4.14.0 uses runtime SIMD dispatch and Intel IPP, as recorded in build metadata.
The host supplies glibc/libm; their hashes are recorded separately from the
bundled libraries. Thus the three x86 hosts share detector/OpenCV bytes, but
are not claimed to share the entire operating-system runtime.

`portable-checksums-verified.log` records verification of the deployed bundle.
`runtime-library-hashes.json` records the actual resolved libraries.
`post-run-verification.json` confirms all 298 source/input snapshot files still
match. Environment, raw commands, full results and per-image summaries are kept
alongside this file. No task builds overlapped timings; existing services and
power settings remained unchanged.

| Configuration | Median p50 speedup across 16 images |
| --- | ---: |
| ArUco3, 1 thread | 1.509x |
| ArUco3, 4 threads | 1.443x |
| Full-resolution, 1 thread | 1.469x |
| Full-resolution, 4 threads | 1.277x |

## Goals

Compare both implementations under identical inputs and CPU budgets without
requiring a persistent build environment on the benchmark machine.

## Open questions

Power, thermal state and background services were observed rather than controlled.
These still-image latencies do not establish video throughput or field accuracy.
