# Implementation and verification

## Purpose

Use this document to choose the tests for a detector change and to interpret their coverage.

## Scope

The detector returns marker IDs and ordered corners. The [pipeline design](design/detector-pipeline.md) describes the stages and their numerical guarantees. The [evaluation plan](evaluation-plan.md) defines the inputs, metrics and timing boundaries for comparing implementations.

## Current state

| Implementation | What to verify | Design and evidence |
| --- | --- | --- |
| GPU, ordinary presets | Connected components and extreme-point quads. One quad per component can miss marker holes joined by background lines. | [Extraction decision](adr/0003-candidate-extraction-approach.md) and [four-machine accuracy report](accuracy-report.md) |
| GPU, explicit steady presets | Ordered contours preserve holes. Grouping, the border check on the representative, eligible retries and resize arithmetic affect agreement with CPU. | [Contour decision](adr/0004-device-contours-for-cpu-quality.md) and [steady-state validation](gpu-steady-state-report.md) |
| Hybrid | GPU preprocessing followed by host contour extraction and decoding. Compare candidates and final results on the same pixels. | [Pipeline design](design/detector-pipeline.md) and [historical benchmarks](benchmark-report.md) |
| Optimized CPU | Compare IDs and ordered corners against unmodified OpenCV across thread counts, both detection modes and changing frames. | [CPU design](design/cpu-optimization.md) and [CPU validation](cpu-steady-state-report.md) |

The historical component suite ran on DGX Spark, Jetson AGX Orin, Jetson AGX Thor and RTX 5070 Ti. The steady route's border-corrected quality matrix ran on Spark and Orin. RTX 4090 validation and the latency tables cover the earlier build. The [steady-state report](gpu-steady-state-report.md) identifies each source snapshot.

### Verification strategy

| Change | Checks |
| --- | --- |
| Preprocessing or decoding | Check pyramid and threshold bytes, dictionary identity and all four rotations against the oracles. For resize, sampling and refinement, use the tolerances for the selected arithmetic profile. The host refinement oracle disables FMA contraction to match the kernel. A one-ULP change can alter convergence, so also compare corners to the linked OpenCV build. [Stage guarantees](design/detector-pipeline.md) |
| Candidate extraction or grouping | Check oriented paths, diagonal contacts, border margins, nested candidates, equal IDs at separate positions, and failed representatives with eligible alternatives. The original containment tests caught mutations to parent choice, depth order, boundary handling and reach counting. [Grouping](../src/core/candidate_group.md) and [retries](../src/core/candidate_retry.md) |
| Detector and workspace | Exercise empty, strided, ROI, tiny and changing inputs, invalid arguments, limit overflow, same-pointer replay, cross-stream submission, reinitialization and injected allocation/dispatch/readback failures. The dictionary persists across frames. Frame scratch uses a separate arena sized from per-axis square bounds because the segmentation scale depends on the longer side. Check allocation counts and caller-visible status. [Detector lifetime](../src/core/detector.md) |
| Public interfaces | Test the installed C++ package with standalone examples, C ABI and Python results, CLI parsing and the render-then-detect round trip. Check synchronization and borrowed-view lifetimes across submissions. [Public API](design/public-api.md) and [bindings](design/c-abi-and-python.md) |
| Optimized CPU | The 46-case CPU suite covered the full corpus with 1, 2 and 4 threads, all 512 combinations of the original nine switches, new option combinations, dictionaries, ROI/lifecycle boundaries and subpixel refinement with ArUco3 disabled. [CPU validation record](measurements/2026-11-17-cpu-steady-validation/README.md) |

Run unit, conformance, differential, robustness, CLI and documentation checks for each change. Verify CUDA paths on physical devices. Run memcheck, racecheck, initcheck and synccheck with `ctest -L sanitizer`. Suites with `DeliberateError` in their names exercise intentional CUDA failures separately. Racecheck uses `--force-synchronization-limit 1` on every target to bound its analysis state and a timeout of 1,800 seconds. [Sanitizer configuration](../cmake/Aruco3CudaSanitizer.cmake)

Refinement grid sizing affects small-scene latency. The S10 kernel strides over corner slots and launches twice the device SM count, clamped to 32–64 blocks. Launching one block per slot had penalized Orin by spreading shared-memory work across extra waves. [Refinement launch policy](../src/core/corner_refine.hpp)

### Coverage

The target is 100% line (C0) and branch-edge (C1) coverage. The general host build runs `cmake --preset coverage` and `cmake --build build/coverage --target coverage-report`. The recorded builds below have different scopes.

| Instrumented scope | Lines | Branch edges | Functions |
| --- | ---: | ---: | ---: |
| Project C++ units, RTX 4090 steady build before the border correction | 5,850/8,579 (68.2%) | 5,336/14,007 (38.1%) | 81.7% |
| Detector host implementation (subset) | 417/505 (82.6%) | 331/636 (52.0%) | — |
| C ABI implementation (subset) | 226/277 (81.6%) | 83/216 (38.4%) | — |
| Standalone optimized CPU module | 1,445/1,502 (96.2%) | 1,373/2,024 (67.8%) | 50/50 (100%) |

**Project C++ build.** The RTX 4090 used CUDA 12.6, GCC 13.3, Debug, the GENERIC resize profile, native CPU tuning, gcovr 7.0 and atomic counters. Build-time discovery counters were cleared first. Of 628 configured cases, 624 passed and four required ARM/Kleidi. Decision coverage was 1,612/2,378 (67.8%). The ignored local directory `build/gpu-study/rtx4090-coverage-v2/` holds commands, hashes, logs and per-file results.

Seven standalone tool/benchmark entry points account for 2,065 of the 2,729 uncovered lines. Gaps also include general OS/OpenCV failures, graph and event failure combinations, C ABI exceptional paths, defensive guards, equal-ID sorting and ARM topology on x86. The denominator retains those lines and exception edges (4,488 throw edges). Bounded fault fixtures cover selected allocation, event, launch and copy failures.

**Standalone CPU build.** This scope excludes CLIs, archived experiments, OpenCV shared libraries and CUDA. The 46-test Debug run and refinement tests provided the initial counters. Seven focused tests refreshed only the changed sparse translation unit after the allocation-recovery fix. Remaining gaps include failure handlers, impossible post-validation inputs, 64-bit epoch rollover, disabled experiments, sparse-warp/cache-release paths and exceptional edges. The earlier cache-refinement build reached 822/871 lines, 773/1,166 branch edges and 32/32 functions. [Initial coverage](measurements/2026-11-17-cpu-steady-validation/README.md) and [recovery verification](measurements/2026-11-17-cpu-steady-recovery-validation/README.md)

**CUDA translation units.** gcov excludes both kernels and host helpers compiled by nvcc. Tests exercise invalid arguments, exhausted workspace, device boundaries and close-contour retry cases such as `failmask9` and `regression_2492`. Input partitions and sanitizer results are recorded in the [steady-state report](gpu-steady-state-report.md).

## Goals

- Extend coverage of host failures and develop a way to measure CUDA translation units.
- Reduce fixed work on small images with detections while preserving per-scene quality. Separate kernel diagnostics from complete-call latency.
- Verify application crossover points and corner accuracy on annotated real images.

## Open questions

- Which arrangements of close markers expose differences between CPU ordered grouping and GPU transitive grouping beyond the tested chains?
- Which additional [Dictionaries](dictionaries.md) or annotated image sets justify extending conformance tests?
- Why did pinned input take longer than pageable input on Spark in the historical [memory-type comparison](benchmark-report.md)?
