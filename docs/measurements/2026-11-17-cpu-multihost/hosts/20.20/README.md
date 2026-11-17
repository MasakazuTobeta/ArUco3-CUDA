# Industrial background CPU evaluation

## Purpose

Compare the unchanged experimental CPU detector with the pinned OpenCV detector
on the frozen 16-photo industrial background corpus.

## Scope

The common snapshot and runner define all source and input hashes. The same
source is compiled locally in the recorded Docker image, using Release
`-O3 -DNDEBUG` and `ARUCO3CPU_NATIVE=OFF`. No CUDA execution is involved.

## Current state

Generated measurement results, logs and environment captures referenced below
are local-only and ignored by Git. This document and reproduction sources remain
version-controlled; a fresh checkout does not contain the recorded outputs.

The NVIDIA Jetson AGX Thor host (`192.168.20.20`) retained MAXN power mode and its existing services, including a simulation container observed at approximately 7.5% CPU before evaluation. Affinity is CPU 4 for one thread and CPUs 4,5,6,7 for four threads; the group was selected before timings.
`host-before.json` and `host-after.json` contain raw hardware, power, process,
and frequency snapshots. These are observations, not controlled clocks.
`opencv-build-info.txt` records the actual baseline build and dispatch options.
`execution.json` records image, affinity, and working directory. The image ID is
recorded in `image.json`; `build.sh` records the build invocation.

Each of the four configurations retains three independent processes. ArUco3
uses 30 warm-up calls and 200 measured calls per route/image; full-resolution
mode uses 10 warm-up calls and 50 measured calls. Routes alternate ordering.
The benchmark measures detector-call latency with grayscale input already
available. Image decoding, construction, warm-up, validation, and collection
of rejected OpenCV candidates are outside timing. Setup latency is excluded.

All planned groups completed. Every background scene returned zero detections
with exact output checks, and all four positive controls returned four markers.
All 64 scene/configuration aggregates beat OpenCV at p50, p95, and p99.

| Configuration | Median p50 speedup across 16 scenes |
| --- | ---: |
| ArUco3, 1 thread | 1.629x |
| ArUco3, 4 threads | 1.623x |
| Full-resolution, 1 thread | 1.639x |
| Full-resolution, 4 threads | 1.961x |

The largest optimized process p50 spread was 18.158% for a four-thread
full-resolution scene; those samples remain included.

## Goals

Retain all images and measurements, validate zero outputs exactly against the
actual baseline, verify known four-marker controls in every thread/mode
combination, and report p50/p95/p99 and independent-process spread. See each
configuration's `summary.json` for the median of process percentiles per image;
percentiles are not pooled.

## Open questions

These photographs have no detections in the tested configuration; they are not
independently annotated deployment video. Other dictionaries, camera conditions,
and thread budgets remain outside this evaluation. Timed intermediate outputs
are not individually retained: checks cover the initial untimed and final
retained outputs.
