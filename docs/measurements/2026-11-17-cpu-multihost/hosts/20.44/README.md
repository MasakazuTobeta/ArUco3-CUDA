## Purpose

Measure the unchanged CPU detector against the pinned OpenCV implementation on all 16 industrial photographs.

## Scope

ArUco3 enabled/disabled, one/four threads, three independent timing processes per condition. The shared runner records correctness checks, p50/p95/p99 and all inputs. Decode, initialization and warm-up are excluded; no CUDA execution occurs.

## Current state

Generated measurement results, logs and environment captures referenced below
are local-only and ignored by Git. This document and reproduction sources remain
version-controlled; a fresh checkout does not contain the recorded outputs.

The portable executable SHA-256 is `3255d78af1ac0a4a95ca66c7c130e80b78c939a470ad03e15a30ffad24805334`. It was compiled on host 20.44 inside Ubuntu 24.04 with GCC 13.3, Release `-O3 -DNDEBUG`, and `ARUCO3CPU_NATIVE=OFF`. OpenCV commit `0654a42e19215ef25b1d367d822f3c630447e7c7` is unmodified. Its pthreads backend, IPP 2026 and runtime SIMD dispatch are recorded in `build-metadata/opencv-build-info.txt`. The same binary and OpenCV libraries are used on hosts 20.41, 20.44 and 10.26.

Execution uses the dedicated `aruco3-cpu-multihost:20261117` Ubuntu 24.04 container over an Ubuntu 22.04 host. One thread is pinned to CPU 0; four threads use CPUs 0,2,4,6, which are distinct physical Core i9 performance cores. The image and detector builds completed before measurement. The host's buildx client API was too old for its Docker daemon; `DOCKER_BUILDKIT=0` selected the functioning legacy builder. The Dockerfile, image inspection and build logs are retained here.

The unrelated `ball_caster_dem` workload was active at about one logical CPU on each x86 host, with unrestricted CPU affinity. It and other services were retained. The benchmark cores avoided the workload's observed initial core, but migration and shared-cache/frequency effects remain uncontrolled. `host-before.json` preserves topology, load and process snapshots; per-group environment records preserve clock/governor snapshots. No power settings or unrelated workloads were changed.

All 10 planned groups completed. Each mode/thread condition contains three timing processes and all 16 photographs. All 64 aggregated scene/configuration comparisons beat OpenCV at p50, p95 and p99. All positive controls found exactly four markers, and all background comparisons returned zero detections with exact retained output agreement.

| Mode | Threads | Median p50 speedup | Minimum p50 speedup |
| --- | ---: | ---: | ---: |
| aruco3 | 1 | 1.4503x | 1.3762x |
| aruco3 | 4 | 1.4421x | 1.2617x |
| fullres | 1 | 1.3564x | 1.2771x |
| fullres | 4 | 1.2419x | 1.1974x |

Individual-process p50 comparisons: 192/192 wins; minimum speedup 1.1968x. The process-level results are retained separately from medians across processes.

## Goals

Retain all images and outcomes, including regressions, and compare per-host paired timings without interpreting absolute cross-host latency as a controlled hardware ranking.

## Open questions

How much do the existing CPU/GPU workload and uncontrolled frequency policy affect latency variance? A fully idle-host follow-up would be required to isolate this effect.
