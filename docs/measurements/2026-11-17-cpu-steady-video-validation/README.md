# Steady-state video runner validation

## Purpose

Preserve the video runner's implementation checks, lifecycle failure tests,
focused sanitizer check and short timed smoke. This archive is not a formal
video-throughput comparison.

## Scope

This directory copies existing evidence. No detector execution or timing run was
performed while creating it. Commands and observed exit statuses are in
commands.json (local `commands.json`), and source/evidence fingerprints plus the exact
scope of each stage are in validation.json (local `validation.json`).

The [frozen source snapshot](source/video_benchmark.cpp) is byte-identical to
`cpu/video_benchmark.cpp` at archival, SHA256
`dcc87c11040f46a5258949396f7b33305cbfc8cfd86937da647c598d716c06e0`.
The eight tuning checks use this final source. Earlier lifecycle, sanitizer and
smoke runs occurred before the final `previous`/single-switch tuning mapping;
their actual executable hashes are retained in the logs. Historical source
hashes were not captured for those stages and are explicitly null. This is not
a claim that every older run used the final source snapshot.

## Current state

Generated measurement results, logs and environment captures referenced below
are local-only and ignored by Git. This document and reproduction sources remain
version-controlled; a fresh checkout does not contain the recorded outputs.

### Twelve CLI and lifecycle cases

The `cli/` directory preserves stdout and stderr for each case. The original
harness used a 20-second subprocess timeout and asserted the expected exit
status. Every successful validation-only run emitted null timing fields. All
twelve cases passed the original harness:

| Case | Expected/observed exit | Coverage |
| --- | --- | --- |
| `both_default` | 0 | Both actual routes, both layouts, six frames; indexed records |
| `reverse_fullres` | 0 | Reverse route/layout order, ArUco3 disabled, two-CPU budget |
| `round2` | 0 | Explicit experimental tuning, both layouts |
| `mixed_sizes` | 0 | Changing 640x480/1920x1080 frames, both routes/layouts |
| `setup_failure` | 1 | Exception before the ready barrier |
| `frame_failure` | 1 | Exception during an untimed frame phase |
| `spawn_failure` | 1 | Main-thread exception after partial thread creation |
| `invalid_workers` | 1 | Invalid worker-count validation |
| `no_images` | 1 | Empty input selection |
| `duplicate_pixels` | 1 | Sequence without different decoded frames |
| `wrong_budget` | 1 | CPU affinity does not match the requested thread budget |
| `duplicate_affinity` | 1 | Duplicate CPU indices |

These ran under outer affinity CPUs 0-3; some cases narrowed their process mask
to CPUs 0-1. The successful cases compare IDs, ordered corners and rejected
counts against actual OpenCV outside timing. Lifecycle failures are deliberate;
their nonzero status is the passing outcome, not a valid measurement.

### Eight final tuning checks

The `tuning/` records use executable SHA256
`ed2cb02da873dacf1b9b90c357d14f0f2762e1540e8533528a030ed4e448d73c`.
Each ran six changing frames on CPUs 0-3 with four independent workers, one
OpenCV thread and validation-only mode. All passed both metadata mapping checks
and exact output comparisons:

- `previous`: original six changes enabled; the five later changes disabled.
- `control`: both sets disabled.
- `spatial`, `sparse`, `contours`, `cached`, `lookup`: previous plus only the named
  later change.
- `round2`: all five later changes enabled.

The raw records include the selected switch values, binary hash, input hashes and
sequence hash. No performance samples were collected by these checks.

### Focused sanitizer process

asan-final.jsonl (local `asan-final.jsonl`) records a successful six-frame validation-only
process with both actual routes/layouts and reverse ordering, on CPUs 0-3.
Executable SHA256 is
`923acff2f7e90ca4fb1f5a53643442821375a4a250609f0974e0993ce473fc2c`.
Its exit status was 0 and no sanitizer diagnostics appeared in the original tool
output. Stderr was not separately saved, so this archive does not invent a
historical stderr log.

Only `cpu/video_benchmark.cpp` was compiled with
`-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer`.
The detector, utility library and OpenCV came from existing Release libraries.
The process used `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1` and
`UBSAN_OPTIONS=halt_on_error=1` inside the container shell. This validates the new
CLI/pool integration; it is not a full instrumented detector run or a sanitizer
rerun of the final tuning-mapping revision.

### Timed-path smoke

timed-smoke.jsonl (local `timed-smoke.jsonl`) retains the only timed run in this archive:
twelve changing frames, three workers versus one three-thread worker, both
actual routes, reverse order and CPUs 6-8. The reserved timing lease was released
when that run completed. It used executable SHA256
`a400f7be8853d4a03b6676572bb0688e8162f7e4d8f29afd4e257146434a053f`.

All four route/layout combinations checked twelve timed outputs each, giving 48
indexed timed samples. Frame indices, worker bounds, positive finite throughput
and ordered service percentiles were checked. Exact IDs/corners were compared
after the timed phase; rejected counts were checked in untimed passes. Timed
`CpuDetector` rejected counts were also checked; timed OpenCV calls omitted
rejected-corner coordinates, matching the documented output scope.

This finite smoke validates the measurement path, not a scheduling speedup.
Its service and saturated-throughput fields must not be substituted for a
three-process steady-state experiment. Startup, queue/camera latency and thread
teardown are outside the reported interval, as described in the
[video benchmark design](../../../cpu/video_benchmark.md).

### Rechecking the archive

This command requires the historical result and checksum files in local storage.
A fresh checkout contains the verifier and summary, but not those generated files.

```bash
python3 docs/measurements/2026-11-17-cpu-steady-video-validation/verify_evidence.py
```

This verifies archived hashes, record counts, tuning mappings and smoke sample
invariants. It executes no detector and collects no timings. Source/build and
complete commands needed for future runs remain in the
[video runner documentation](../../../cpu/video_benchmark.md).

## Goals

- Keep commands, successful checks and expected failures reviewable.
- Distinguish final-source checks from earlier integration checks.
- Preserve sanitizer and timing scope without claiming broader coverage.

## Open questions

- Final steady-state performance will be measured after the library and video
  source freeze; it is not established by this archive.
- Historical video source snapshots for the earlier three stages were not
  retained, although their executable hashes and runtime settings are available.
