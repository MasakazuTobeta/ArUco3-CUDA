# DGX Spark CPU measurement

## Purpose

Repeat the same-host reference measurement as part of the six-host comparison.

## Scope

DGX Spark GB10 at 192.168.20.23; CPU 5 for one thread and CPUs 5–8 for four
threads. Use the common source, images, detector configuration and runner.

## Current state

Generated measurement results, logs and environment captures referenced below
are local-only and ignored by Git. This document and reproduction sources remain
version-controlled; a fresh checkout does not contain the recorded outputs.

All ten groups completed, including 384 timed measurement rows, four positive
controls and two eligibility groups. The unchanged, previously validated Release
binary was reused with native tuning disabled. GCC 13.3 and the pinned OpenCV
4.14.0 installation run inside the existing `aruco3-cpu-opt` container. That
container was stopped again after collection, restoring its initial state.

Build information, library hashes, source/input snapshot, before/after topology
and load, exact commands, raw results and all per-image summaries are retained
alongside this file. Post-run verification checked all 298 snapshot files.
No task build overlapped measurement. Existing unrelated services remained active.
Recorded frequency snapshots are not claims of a controlled physical clock;
see the repository's DGX Spark frequency caveat.

## Goals

Keep this fresh measurement distinct from the preceding single-host evaluation
while using the same detector binary and prepared images.

## Open questions

Existing background load and clock policy were retained. Results describe this
run rather than an isolated hardware ceiling or deployment-camera throughput.
