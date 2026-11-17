# CPU PR adversarial review

## Purpose

Address [PR #24's provenance finding](https://github.com/MasakazuTobeta/ArUco3-CUDA/pull/24#discussion_r4595731543)
and independently challenge the detector, helper algorithms and measurement
tools beyond that single report.

## Scope

The reviewed publication baseline is `6ca23cf07e4fdc7c775c301e8f387ee40cfd9f41`.
Review covers output ownership and provenance, aggregation, validation mode,
detector lifecycle and dictionary lookup, ROI/aliasing, threshold arithmetic,
contour ordering, sparse-pyramid validity, refinement, and exception recovery.
No production detector C++ source or detection defaults are changed by this
follow-up. Existing GPU measurement files remain outside this change.

## Current state

### Confirmed and repaired findings

| Finding | Reproduction and resulting behavior |
| --- | --- |
| Rerunning the experiment could overwrite old metadata before refusing an existing timing file | An existing `run-1.jsonl` remained unchanged while its `environment.json` was replaced. Both the single-host and six-host runners now claim a new result directory atomically after preflight and create output files exclusively. Existing, empty, partial and symlink result directories are rejected without modification. |
| The six-host runner wrote top-level provenance before checking affinity or existing groups | Invalid affinity and reruns could replace `snapshot.json` and `input-hashes.json`. Preflight now completes before any result output. A fresh result subdirectory is required; auxiliary host captures belong beside it or are added after completion. |
| A bare binary filename could identify different files for hashing and execution | A same-named executable on `PATH` ran while the runner hashed the file in its current directory. The runner now resolves one absolute binary path and uses it for both operations. |
| Validation could silently disappear in optimized Python mode | The old six-host runner under `python -O` accepted a fake positive-control result of 99 markers instead of four, wrote `exact_comparisons: true`, and completed successfully. The runner and audit CLI now refuse `-O`, `-OO` and `PYTHONOPTIMIZE` before reading or writing results. |
| The summarizer could publish empty or incomparable batches | Empty/header-only runs and mixed thread configurations produced a summary; invalid percentiles and changing outputs could also pass. It now requires complete numbered runs, one identical configuration per process, nonempty paired records, finite positive ordered percentiles, matching image dimensions/hashes, and stable per-variant output metadata. |

Normal, failure and concurrent-invocation cases are covered by
[runner regression tests](../test/cpu/test_run_experiments.py) and
[summarizer regression tests](../test/cpu/test_summarize_experiments.py).
Before the repair, the targeted runner/optimized-mode reproductions failed their
preservation assertions or incorrectly reported success. The initial summarizer
regression suite also reproduced the invalid-input acceptance. Raw reproduction
logs remain local under `build/review-runner/` and `build/adversarial/`.

The stricter summarizer reproduces all 24 existing six-host group summaries
exactly. Published latency numbers and original raw measurement files were not
changed. See the [six-host report](cpu-multihost-report.md) for their scope.

### Detector and helper checks

The fresh Release build passed all 46 existing C++ test cases. Additional local
probes compare against the pinned OpenCV 4.14.0 public APIs:

| Probe | Checked cases |
| --- | ---: |
| Tiny and one-dimensional sparse-pyramid regions | 12,490 comparisons |
| Projective/singular warps, image-edge coordinates and poisoned unused cache storage | 1,800 comparisons |
| Threshold windows 3–4097, fractional/extreme offsets and workspace budgets | 1,280 outputs |
| Random contour quads, full point paths and ordering | 250 comparisons |
| Refinement near image borders and half-pixel positions, radii 1–256 | 84 configurations / 252 corners |
| Detector combinations: nested frames, edge markers, ROI, dictionary and numeric boundaries | 576 exact comparisons |
| One-shot allocation failure and subsequent detection recovery | 256 injection positions; 38 actual allocation failures |

All probes passed in Release and again with the fresh CPU library instrumented
by ASan/UBSan. Leak detection and halt-on-error were enabled. No sanitizer
diagnostic or output mismatch occurred. Allocation failures preserved the caller's
previous output and the next detection matched OpenCV exactly. No additional
production-detector defect was reproduced.

Probe sources and logs remain local under `build/adversarial/core*` and
`build/adversarial/helpers*`; these exploratory checks are separate from the
committed regression suites. The allocation wrapper covers `operator new` sites
linked into the CPU target, not every allocation inside shared OpenCV libraries.

### Reproduction of maintained tests

The tooling tests require Python and Git, with Linux affinity support for the
runners. They do not need OpenCV, CUDA or a GPU:

```bash
python3 -m unittest discover -s test/cpu -p 'test_*experiments.py' -v
```

Use the pinned OpenCV installation for the standalone C++ suite:

```bash
cmake -S cpu/standalone -B build/adversarial-release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=/opt/opencv/lib/cmake/opencv4
cmake --build build/adversarial-release --target aruco3cuda_cpu_tests -j4
ctest --test-dir build/adversarial-release --output-on-failure --schedule-random -j2
```

The sanitizer build uses `-fsanitize=address,undefined -fno-omit-frame-pointer`
for compilation and linking, `-O1 -g -DNDEBUG`, and the runtime options
`ASAN_OPTIONS=detect_leaks=1:halt_on_error=1` and `UBSAN_OPTIONS=halt_on_error=1`.
Validation runs on DGX Spark CPUs 5–8 with GCC 13.3 and the unchanged pinned
OpenCV shared libraries. It is correctness testing, not new latency evidence.

## Goals

- Reject measurement attempts that could corrupt existing provenance.
- Prevent malformed batches and disabled validation from yielding success reports.
- Exercise detector boundaries independently while preserving strict comparisons.
- Distinguish reproduced defects, passing checks and unverified behavior.

## Open questions

- Exploratory probes and the existing test corpus cannot prove equivalence for
  every possible image, homography, numeric setting or allocation failure.
- OpenCV shared libraries are not sanitizer-instrumented by this CPU-only build.
- Separate processes can still modify binaries or inputs after preflight; callers
  must keep measured inputs and executables immutable during an experiment.
- The current review's new sanitizer/probe execution is on Spark. It does not
  constitute a fresh six-host performance or full accuracy campaign.
