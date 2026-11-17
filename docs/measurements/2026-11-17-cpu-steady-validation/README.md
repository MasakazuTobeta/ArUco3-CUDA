# Steady-state CPU correctness validation

## Purpose

Validate the current CPU defaults against the installed OpenCV 4.14.0 detector,
including changing frames and reusable storage. These are correctness runs;
their execution durations are not detector latency measurements.

## Scope

The standalone CPU suite contains 46 tests. It checks exact IDs, oriented float
corners and rejected counts, plus intermediate threshold pixels, contour paths,
group nodes and sparse-pyramid pixels, sampling and refinement.

The full corpus case retains the three historical option masks and adds the
actual `CpuOptions{}` defaults: 91 scenes x 2 detection strategies x 4 routes x
1/2/4 OpenCV threads, or 2,184 tested detector calls. A separate sequence
shuffles 96 synthetic scenes, adds blank cuts and revisits, overwrites the same
strided storage, and compares seven option combinations. It requires positive
detections, including the clean ground-truth IDs. See the
[quality generator](../../../tools/cpu_quality/quality_scenes.md).

The QA commands recorded here pin builds and tests to DGX Spark CPUs 0–3.
Installed OpenCV shared libraries are the compatibility oracle and are not rebuilt with sanitizers or
gcov instrumentation. CUDA code, performance, and camera-image ground truth are
outside this validation. External-photo comparisons and native validation have
separate records.

## Current state

Generated measurement results, logs and environment captures referenced below
are local-only and ignored by Git. This document and reproduction sources remain
version-controlled; a fresh checkout does not contain the recorded outputs.

| Check | Result | Evidence |
| --- | --- | --- |
| Final preparation source, randomized Release suite | 46 passed | log (local `final-release-tests.log`), JUnit (local `final-release.junit.xml`) |
| ASan/UBSan, preceding preparation source | All 46 completed: 44 passed initially; the two long cases passed on targeted reruns | initial run (local `asan-tests.log`), targeted reruns (local `asan-long-cases.log`), JUnit (local `asan-long-cases.junit.xml`) |
| Final preparation change, focused ASan/UBSan | 7 passed | log (local `final-asan-focus.log`), JUnit (local `final-asan-focus.junit.xml`) |
| Final preparation source, randomized Debug coverage suite | 46 passed | log (local `coverage-tests.log`), JUnit (local `coverage.junit.xml`) |
| Additional current-default cached refinement gate | Passed in Release, ASan/UBSan, native Release and Debug coverage | Release (local `refinement-release-tests.log`), ASan (local `refinement-asan-tests.log`), native (local `refinement-native-tests.log`), coverage (local `refinement-coverage-tests.log`) |
| Native Release suite | 46 passed, recorded separately | [native validation](../2026-11-17-cpu-steady-final-native-validation/README.md) |
| Optional top-level CPU integration | Rebuilt after final preparation change | build log (local `top-level-final-build.log`) |

No sanitizer diagnostic or output mismatch was observed. Leak checking and
halt-on-error are enabled; exact options and commands are in the JSON records.
The final preparation change only replaces the copy-and-threshold preparation
with thresholding into an owned center surrounded by an immutable zero border.
The seven focused sanitizer cases cover contour paths, changing sizes and ROI
storage, blank scene cuts, dictionaries, thread policies, and large frames.

Coverage review identified an untested supported combination: cached refinement
with the ArUco3 detection strategy disabled and subpixel refinement enabled. The
existing refinement test now preserves its previous control and also checks the
actual defaults on a normal image and a strided ROI. A marker three pixels from
the visible ROI edge makes its refinement patch cross that edge; randomized
parent pixels must not affect the result. The expected marker must decode.
Only this strengthened case was rerun in the four builds above; the test count
remains 46 and production code did not change.

### Coverage

Final CPU-module coverage is **96.2% lines (1,444/1,501), 100% functions (50/50),
and 67.8% branches (1,373/2,024)**. Gcovr also reports 93.2% decisions (386/414).
Summary (local `coverage-summary.json`), line report (local `coverage.txt`),
uncovered lines (local `uncovered-lines.json`), and
aggregation command (local `coverage-refreshed-report.command.json`) retain the scope.

The filter includes `cpu/` implementation files and excludes the benchmark,
resource and video CLIs and archived experiments. It does not measure CUDA,
OpenCV, or the quality renderer. Atomic gcov counters support internal workers.
Counters were cleared before the complete suite; the later refinement test adds
to the same unchanged CPU-library counters. Only the changed test translation
unit's counters were removed before its rebuild. The
preceding summary (local `before-refinement-coverage-summary.json`) is retained.

The 100% C0/C1 goal is not met. Remaining gaps include:

- Allocation/library failure handlers and post-validation or allowlist guards.
- The 64-bit frame-epoch rollover, which cannot be reached by a practical frame
  sequence in this suite.
- Reachable combinations of disabled experiments, including parallel compact
  contours and the parallel fallback when shared thresholding is disabled.
- Sparse singular/all-outside warp cases and empty-window cache-release paths
  not exercised by the maintained tests.
- Gcov source-location mapping for dictionary initializers and a bounds
  initializer; dictionary lookup and bounds behavior are exercised independently.
- C++ exceptional edges included in the branch totals.

These gaps are reported rather than excluded to increase the percentages.

### Source and execution history

Environment (local `environment.json`) contains the final source hashes and defaults;
binary hashes (local `binary-hashes.json`) distinguish Release, sanitized, coverage and
native executables. The Git parent is contextual: sources were an uncommitted
working-tree snapshot, so the per-file hashes define the tested state.

The chronology is explicit:

1. Initial snapshot (local `environment-initial.json`) captured the enabled defaults.
2. Test-matrix update (local `source-update.json`) added the actual defaults to the
   historical three-route corpus comparison. An early sanitized run was
   deliberately interrupted before this test update; its log is retained and
   is not counted as a passing run.
3. Final preparation snapshot (local `source-final.json`) records the helper-only
   preparation change. ASan/UBSan checks of all 46 cases on the preceding source
   are supplemented by the seven focused checks for this change.
4. Refinement test update (local `source-refinement-test-update.json`) adds the
   meaningful supported configuration described above, without production edits.

The original 600 s CTest limit expired in the full corpus and changing-frame
sanitized cases. Neither emitted a sanitizer diagnostic. The same binaries and
assertions then passed in 761.56 s and 758.14 s with a generated 1800 s limit.
The historical run (local `historical-25-case-asan.log`) had already spent 556.41 s in
its smaller corpus case. The
[override script](set-instrumented-timeouts.py) changes only generated CTest
files; repository test properties and assertions remain unchanged. Overrides,
initial timeouts and successful reruns are all retained. Debug coverage received
the same instrumented limit proactively.

Exact command arrays, exit codes and timestamps are recorded in each
`*.command.json`. Configuration commands (local `configure-commands.json`) and
flags (local `build-configurations.json`) distinguish `-O3` Release, `-O2` ASan/UBSan,
and Debug coverage with `-O0 -fno-inline -fprofile-update=atomic`. Independent
native and external correctness tasks shared CPUs 0–3 during portions of these
runs. Performance work on CPUs 5/5–8 also overlapped. This does not turn the test
execution durations into controlled performance measurements.

## Goals

- Preserve exact OpenCV behavior and independent positive cases while optimizing.
- Keep assertions unchanged when increasing instrumented execution budgets.
- Retain source stages, failed timing budgets and successful targeted verification.
- Report remaining coverage and hardware limitations explicitly.

## Open questions

- The current CPU defaults have not been revalidated on Jetson Orin or another
  CPU architecture through this suite.
- External photographs without independent annotations establish compatibility,
  not absolute camera-image recall or precision.
- The remaining rare-input, failure and disabled-experiment branches need
  targeted meaningful cases if those routes become deployment requirements.
