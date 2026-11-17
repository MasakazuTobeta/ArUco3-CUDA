# Multi-host CPU measurement evidence

## Purpose

Retain the measurements and reproduction tools for the six-host industrial
background CPU comparison in the [report](../../cpu-multihost-report.md).

## Scope

The [plan](plan.md) fixes the unchanged detector, actual pinned OpenCV 4.14.0,
16 photographs, one/four threads, ArUco3 enabled/disabled and three processes.
Startup, image decoding and setup are excluded. GPU execution is outside scope.

## Current state

Generated measurement results, logs and environment captures referenced below
are local-only and ignored by Git. This document and reproduction sources remain
version-controlled; a fresh checkout does not contain the recorded outputs.

- [Runner](run_host.py) verifies the frozen source/input manifest before running
  four positive controls, two eligibility groups and four repeated timing groups.
- Deployment hashes (local `deployment.json`) identify the source/input archive and
  portable x86 runtime bundle. Large images, binaries and bundles are not committed.
- `hosts/<address suffix>/` retains the executed commands and affinities,
  CPU topology/load, power/clock snapshots, OpenCV build information, build
  provenance, raw JSONL summaries, per-group validation and completion records.
- `aruco3-1t` and `aruco3-4t` use 30 warm-ups and 200 timed calls per route/image;
  `fullres-1t` and `fullres-4t` use 10 warm-ups and 50 timed calls. Each contains
  three independent processes. Fullres means the complete prepared image.
- `positive-control-*` and `eligibility-*` contain accuracy-only checks; their
  one-sample durations are excluded from performance results.
- [Aggregation script](aggregate.py) retains all per-image latencies, tails,
  losses and process spread without pooling distributions.
- [Independent audit](qa_audit.py) verifies raw rows against
  [frozen source hashes](qa-source-hashes.json), input dimensions/hashes,
  configuration, output equality, controls, provenance and saved summaries.

All six hosts completed the plan: 2,304 timed measurement rows and 384
image/configuration comparisons, all favoring the candidate at aggregated
p50/p95/p99. The aggregate (local `aggregate.json`) and latency CSV (local `scene-latencies.csv`)
retain every comparison. The audit result (local `qa-audit.json`) and
[QA review](qa-review.md) record verification and limitations.
Historical checksums (local `checksums.json`) identify the captured files at measurement time, before later documentation edits. They do not describe the current checkout.

To reproduce, prepare the pinned images as described in the
[input record](../../../cpu/experiments/real_background.md), use the frozen
source recorded in the report, and build the CPU standalone target against the
pinned OpenCV. The runner additionally requires a separately prepared local
`snapshot.json`; it is not supplied by a fresh checkout. Restore the original
manifest from local measurement storage to audit that exact run, or create a new
manifest for a new run. Its `git_head` identifies the source commit; `files` and
`sources` map relative file paths to SHA-256 digests, while `inputs` and `controls`
map image filenames to SHA-256 digests. The runner checks `files` before execution.
With that local manifest available, run from the source root:

```bash
python3 docs/measurements/2026-11-17-cpu-multihost/run_host.py \
  --binary build/cpu-opt/cpu/aruco3cuda_cpu_benchmark \
  --manifest snapshot.json --output-dir fresh-results \
  --cpu-one 5 --cpu-four 5,6,7,8
```

Select valid distinct physical cores for the machine being tested. The example
affinity is for DGX Spark. The runner refuses existing group directories. For
a relocated OpenCV installation, use `--provenance`; a portable runtime also
requires its library directory in `LD_LIBRARY_PATH`. Capture build information
and actual resolved library hashes separately as in the local host evidence.
The aggregation and audit scripts also require their completed local result
directories; they cannot reconstruct historical measurements from Git alone.

## Goals

- Make every reported comparison traceable to its raw paired measurements.
- Retain unfavorable cases and disclose background load and build differences.
- Distinguish exact checks on this corpus from general field accuracy.

## Open questions

- Background services and clocks are observed, not controlled. Cross-host
  compiler, dispatch and libc differences limit hardware-only interpretations.
- The secondary 50-sample tails have limited percentile resolution.
- These still photographs do not establish deployment-camera or video throughput.
