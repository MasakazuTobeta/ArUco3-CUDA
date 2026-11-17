# CPU allocation-exception recovery validation

## Purpose

Repair sparse-cache recovery after an allocation exception while preserving
normal detection outputs. This record contains correctness checks, not timings
of detector performance.

## Scope

The only production edit is [recovery.patch](recovery.patch) in
`cpu/cpu_sparse_pyramid.cpp`. A zero tile tag invalidates a cache before pixel
or validity-map allocation. The requested positive tag is published only after
both allocations and metadata updates succeed. A failed call therefore forces
initialization on retry, without retaining duplicate full-size pixel storage.
The existing successful-cache path and detection arithmetic are unchanged.

Corrected source SHA-256:
`96ba75dce178a9cf856c77ac8f17219e6d794d9dde4a04121c87e28d95e43e88`.
Environment and source hashes (local `environment.json`),
binary hashes and flags (local `binary-hashes.json`), and individual `*.command.json`
files identify the source, commands, exit status and execution conditions.
All builds and checks used CPUs 0–3 with OpenCV 4.14.0.

## Current state

Generated measurement results, logs and environment captures referenced below
are local-only and ignored by Git. This document and reproduction sources remain
version-controlled; a fresh checkout does not contain the recorded outputs.

| Check on corrected source | Result | Evidence |
| --- | --- | --- |
| Randomized Release suite | 46 passed | log (local `release-tests.log`), JUnit (local `release.junit.xml`) |
| Focused ASan/UBSan | 7 passed | log (local `asan-tests.log`), JUnit (local `asan.junit.xml`) |
| Focused native Release | 7 passed | log (local `native-tests.log`), JUnit (local `native.junit.xml`) |
| Focused Debug coverage | 7 passed | log (local `coverage-tests.log`), JUnit (local `coverage.junit.xml`) |
| Controlled allocation exceptions and recovery | 2 passed under ASan/UBSan | [probe](allocation_recovery_probe.cpp), results (local `allocation-probe.log`) |
| Portable benchmark/video/resources and native benchmark | Rebuilt successfully | portable build (local `portable-binaries-build.log`), native build (local `native-build.log`) |

The seven focused cases cover all four direct sparse-pyramid tests, changing
image sizes/strides/pixels, current-default normal/ROI refinement, and sparse
configuration/reinitialization. The expensive shuffled sequence remains in the
full Release suite; it was not repeated under sanitizers for this allocation
ordering change. This is not a new full 46-case sanitizer or native run.

The ordinary allocation-exception probe uses GNU ld wrapping of operator new.
Every unarmed allocation is forwarded normally. It raises one `std::bad_alloc`
when a validity map grows to 561 bytes after a tile-size change, then another
at 2244 bytes after an image-shape change. Each exception is caught, injection
is disabled, and the same input storage is overwritten. Both retry pixels and
the complete recovered level must equal an actual OpenCV pyramid. Only the
corrected source was executed; the old invalid retry behavior was not run.
Leak checking and halt-on-error were enabled. No sanitizer diagnostic or output
mismatch occurred.

### Coverage refresh

Final CPU-module totals are **96.2% lines (1445/1502), 100% functions (50/50),
and 67.8% branches (1373/2024)**; decisions are 93.2% (386/414).
Summary (local `coverage-summary.json`), line report (local `coverage.txt`), and
aggregation command (local `coverage-report.command.json`) retain the results.

Only the changed implementation's `cpu_sparse_pyramid.cpp.gcda` was removed
before rebuilding. All counters for unchanged translation units were retained
from the preceding complete suite and refinement extension. The seven focused
cases regenerated the sparse counters and added to the unchanged counters;
the reset record (local `coverage-counter-reset.json`) lists the retained files.
Counters use atomic updates. The filter includes `cpu/` and excludes benchmark,
resource/video CLIs and experiments. OpenCV shared libraries and CUDA are not
instrumented. A clean reproduction must first establish the complete-suite
coverage counters before applying this selective refresh procedure.

The 100% C0/C1 goal remains unmet. The preceding validation record documents
failure/rare-input paths, unused experiment combinations, epoch rollover and
compiler source mapping that remain outside the exercised branches.

The [preceding checksummed archive](../2026-11-17-cpu-steady-validation/README.md)
was kept immutable; its 111 recorded file checksums still match. No completion
marker was created by this QA task. Performance confirmation and provenance
are controlled separately by the coordinator.

## Goals

- Keep retry state coherent after ordinary allocation exceptions.
- Preserve exact OpenCV behavior and existing assertions.
- Record the repair and focused verification separately from earlier evidence.

## Open questions

- Other CPU architectures and annotated real camera accuracy remain unverified.
- This probe exercises validity-vector allocation failure; it does not claim
  exhaustive injection of every possible allocation or external-library error.
