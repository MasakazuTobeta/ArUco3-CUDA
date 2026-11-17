# AGENTS Guide

Agents working in this repository should read `CONTRIBUTING.md` first. `CONTRIBUTING.md` is the authoritative source for the shared rules.

## Precedence

When rules conflict, apply them in this order.

1. Explicit instructions from the user
2. `CONTRIBUTING.md`
3. The `AGENTS.md` in the repo root or in the target directory
4. The `README.md` of the area concerned and `docs/**`

## First Read Checklist

- Read `CONTRIBUTING.md`.
- When changing documents, read `docs/AGENTS.md` and `docs/terminology.md`.
- Read the design document, ADR, and evaluation plan that correspond to what you are changing.
- For a non-trivial change, work out the purpose, dependencies, risks, verification method, and the documents that need updating.

## Working Principles

- Do not mix the current implementation with the goals.
- Do not assume that CUDA is always faster; compare CPU, CUDA, and hybrid approaches using measurement results.
- Record kernel-only time separately from time that includes input preparation, synchronization, and result retrieval.
- Use the same algorithm on DGX Spark and Jetson Orin, and keep machine-specific optimizations separate as explicit options.
- Do not leave thresholds and limits as fixed values in the source alone; structure them so they can be overridden from configuration.
- Do not copy code from external implementations without authorization. When a paper, a public specification, or a compatibly licensed implementation is used as a reference, record the source and its influence.
- Record important design decisions in an ADR or a related document.

## Documentation Rules

- Specification, design, and evaluation documents must have at least `Purpose`, `Scope`, `Current state`, `Goals`, and `Open questions`.
- When behavior, configuration, the public API, or evaluation conditions change, update the related documents in the same change.
- Place a sidecar `*.md` next to a module whose design decisions are not conveyed by Doxygen alone. See Documentation in `CONTRIBUTING.md` for where each area's records belong.
- Prefer Mermaid for overview diagrams, since diffs are easier to review.

## Completion Checklist

- Was compatibility with the relevant CPU baseline implementation confirmed?
- Were tests added for normal cases, error cases, and boundary values?
- Can CUDA errors and failures in asynchronous execution be detected?
- Are kernel-only and end-to-end measurements kept distinct?
- Was verification done on at least the target side of DGX Spark and Jetson Orin, or was the reason it was not verified recorded?
- Are the README, design, evaluation conditions, and ADRs free of contradictions?
- Are secrets, build outputs, and large datasets kept out of the commit?

## Operational Know-How

When a failure that could recur or an environment-specific caveat comes to light during work, add the cause and the correct procedure to this section.

### GCC 9 quality-renderer bounding box

- GCC 9.4 at `-O3` diagnosed potential null dereferences when `boundingRect` received a temporary vector copied from four stored corners. Passing the existing `std::array` directly removes the allocation and diagnostics while retaining identical pixels across 288 rendered frames. Keep warnings as errors; do not globally suppress null-pointer diagnostics to build the quality audit.

### Initializing an empty repository through a GitHub App

- In a completely empty repository, blob creation through the Git Data API fails with `409 Git Repository is empty`.
- First create a single file such as `README.md` through the Contents API to initialize the default branch.
- After initialization, multiple files can be combined into one commit in the order blob, tree, commit, ref update.

### Coverage of parallel CPU work

- Gcov's non-atomic counters can produce negative branch counts when OpenCV workers execute instrumented code concurrently. Build coverage targets with `-fprofile-update=atomic`; do not ignore negative-count parser errors as if they were valid coverage.
- After changing an instrumented source or its flags, remove the old `*.gcda` files in that coverage build directory before running the tests. Mixing checksums from different builds makes the report unreliable.

### Interpreting DGX Spark frequency snapshots

- During the CPU experiment, `scaling_cur_freq` sometimes exceeded `scaling_max_freq`. The cause is unverified. Preserve the raw values and governor in the results, but do not describe the snapshot as a controlled or confirmed physical clock. Use the recorded topology to identify performance cores.

### Pinned OpenCV contour oracle and ROIs

- OpenCV 4.14.0's `RETR_LIST` fast finder pads without `BORDER_ISOLATED`. A direct ROI with nonzero parent pixels can expose an invalid contour border. Clone an ROI before using this finder as an isolated-image oracle; keep the candidate helper's original strided ROI input. The detector's threshold planes already own their storage.
- The pinned TRU stripe tracer can write visited flags beyond its own stripe. Repeated four-thread scratch calls showed varying contours even with size filtering disabled. The CPU optimized helper uses one sequential scan per independently owned threshold window. Do not reintroduce overlapping stripe writes as a performance shortcut.

### Exact sparse-pyramid sampling

- Calling `warpPerspectiveBlocklineNN` directly did not reproduce every public `warpPerspective` result at fractional image-edge quads, although the ordinary detector corpus passed. The precise compiler/HAL cause was not established. The maintained sparse route prepares the required pixels and executes the actual global-coordinate public warp.
- Compare canonical bytes at half-integer borders and nondefault canonical sizes, and poison unrequested cached storage differently in repeated tests. Detection-ID agreement alone cannot establish that a sparse image sampler reads only initialized current-frame pixels.

### Container scripts supplied on standard input

- Use `docker exec -i` when passing a Python or shell script through a heredoc. Without `-i`, the interpreter can receive empty input and exit successfully without executing the checks. Verify expected output files or record counts as well as the exit status.

### Instrumented CPU test timeouts

- The full-corpus and changing-frame cases can exceed the inherited 600 s CTest limit under ASan/UBSan on GB10 efficiency cores. Preserve the timeout logs, check for sanitizer diagnostics, and rerun only the affected cases with a recorded generated-build timeout override. Do not reduce inputs or assertions to obtain a pass. The [steady-state validation record](docs/measurements/2026-11-17-cpu-steady-validation/README.md) retains the 1800 s override and successful unchanged-case reruns.
