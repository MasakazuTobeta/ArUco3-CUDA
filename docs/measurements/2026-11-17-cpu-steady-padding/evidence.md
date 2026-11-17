# Fused contour padding diagnostic

## Purpose

Test whether writing normalized pixels directly into the padded center removes a redundant full image pass. This is a bounded late diagnostic. Following the controlled comparison, the planning agent authorized applying the small fused-preparation change; final acceptance gates are being repeated.

## Scope

Contour preparation plus unchanged tracing/group-independent candidate extraction. Single-thread CPU 9 only, pinned OpenCV /opt/opencv in aruco3-cpu-opt. Reuses existing contour probe inputs and helpers. This is not an end-to-end speed claim or an isolated-machine benchmark.

## Current state

Generated measurement results, logs and environment captures referenced below
are local-only and ignored by Git. This document and reproduction sources remain
version-controlled; a fresh checkout does not contain the recorded outputs.

The shared-scanner controlled run passed 68,536 exact comparisons (all 4x4 binary masks plus 3,000 changing random isolated ROIs with arbitrary nonzero values). Candidate coordinates and ordered full contour paths match frozen production. Persistent padding remained zero after every comparison. An additional 36 actual benchmark planes were compared exactly. Both modes use the same scanner function address and padded allocation; only preparation changes. Five warmup iterations; 40 measured iterations (20 for full-resolution 4K), alternating variant order. Raw logs include p50/p95/p99.

The initial separate-function experiment misleadingly regressed noisy scans by approximately 5%; the controlled single-function run removes that discrepancy. Its noise medians differ by only 0.1–0.24%, below a useful diagnostic gain. Empty and clean savings are consistent across both runs. Do not use the initial separate-function noisy difference as evidence about the preparation operation.

| Scene | ArUco3 | Old p50 ms | Fused p50 ms | Saving us |
| --- | --- | --- | --- | --- |
| empty_1280x720 | True | 0.019535 | 0.017503 | 2.032 |
| empty_1280x720 | False | 0.165178 | 0.133772 | 31.406 |
| cpu_1280x720_n4_clean | True | 0.090669 | 0.087677 | 2.992 |
| cpu_1280x720_n4_clean | False | 0.555740 | 0.515165 | 40.575 |
| cpu_1280x720_n4_noise | True | 2.113427 | 2.108371 | 5.056 |
| cpu_1280x720_n4_noise | False | 34.913576 | 34.870327 | 43.249 |
| empty_3840x2160 | True | 0.030447 | 0.024095 | 6.352 |
| empty_3840x2160 | False | 2.241791 | 1.746000 | 495.791 |
| cpu_3840x2160_n4_clean | True | 0.101629 | 0.094829 | 6.800 |
| cpu_3840x2160_n4_clean | False | 3.092016 | 2.652271 | 439.745 |
| cpu_3840x2160_n4_noise | True | 3.160973 | 3.156270 | 4.703 |
| cpu_3840x2160_n4_noise | False | 249.113675 | 248.742867 | 370.808 |

## Goals

The proposed patch removes redundant center copy/normalization while preserving exact arbitrary-nonzero input and isolated ROI semantics. The outer zero border is invariant scanner storage, not reused frame content. The complete center is overwritten on every append. Proposal uses an exception-safe constructor assignment when shape changes; the measured prototype used create plus zero-fill. These have identical steady-state work but shape-change allocation behavior was not timed.

## Open questions

Only the pinned GB10 environment was checked. Noise is neutral, not a meaningful gain. The planning agent accepted the consistent empty/clean benefit with neutral noise and authorized applying proposal.patch. Accuracy must remain exact; positive speed gains in every scene class are not an acceptance requirement. Formal measurements will be repeated on the changed source. Do not mix these diagnostic numbers into prior-source formal results.

## Concurrency and evidence

- Initial run: 2026-11-16T23:49:13Z–23:49:32Z, CPU 9; root benchmark PID7430 CPU5 plus QA PID5361 CPU2/PID6879 CPU1 concurrently.
- Controlled run: 2026-11-16T23:52:24Z–23:52:43Z, CPU 9; root benchmark PID7922 CPU5 plus QA PID5361 CPU2/PID7566 CPU3/PID7567 CPU2 concurrently.
- Raw result files: results.jsonl and shared-results.jsonl; captured process lists and UTC files accompany them.
- source-hashes.json pins the pre-fusion source. Production cpp and hpp hashes were unchanged throughout diagnostic timing.
- proposal.patch was applied to production with planning-agent authorization after diagnostic timing; production was formatted. All diagnostic files remain under ignored build/.

## Applied change validation

The exception-safe production implementation, formatted with the repository's clang-format rules, was compared directly against the old copy-plus-normalize implementation: 68,536 exact candidate/path comparisons passed, including all 4x4 masks, changing shapes, arbitrary nonzero foreground values, and isolated strided ROIs. The three existing CpuContoursTest Release cases passed in build/cpu-sparse-dev on CPUs 0–3. Logs: applied-exact-results.jsonl and applied-release-contours.log.

Changed production cpu/cpu_contours.cpp SHA-256:
3508f502208042ba8902f9e828ec2b7d713685c84e303a18dd7c8282a450ff87.
The helper source is frozen again. Final full Release, focused sanitizer, coverage, and final-source end-to-end measurements are assigned to QA/planning.
