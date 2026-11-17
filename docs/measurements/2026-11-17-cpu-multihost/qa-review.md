# Independent multi-host measurement review

## Purpose

Verify that the six-host CPU performance conclusions follow from complete,
comparable measurement records without changing or rerunning the detector.

## Scope

Review the unchanged implementation at `c5c943232da465a5b0e89aec046787d8144dbab9`,
the frozen 16-photo corpus, four detection/thread configurations per host,
three independent timing processes, and the positive and eligibility controls.
This review checks measurement integrity and the recorded compatibility checks;
it is not a new full detector accuracy or sanitizer campaign.

## Current state

Generated measurement results, logs and environment captures referenced below
are local-only and ignored by Git. This document and reproduction sources remain
version-controlled; a fresh checkout does not contain the recorded outputs.

The independent audit (local `qa-audit.json`) passes. The reproducible
[audit program](qa_audit.py) independently recomputes scene percentiles, speedup
ratios, process spreads and win/loss counts from the raw JSONL records. It checks:

- All 2,304 timed rows, covering 384 image/configuration comparisons across six
  hosts. No selected image, route or timing process is missing or duplicated.
- Exact prepared-image hashes and dimensions, zero detections, exact rejected
  counts, empty discrepancy fields and zero corner error in every retained row.
  The underlying benchmark checks initial untimed and final retained outputs;
  intermediate timed outputs are not retained.
- All 24 positive controls, covering both strategies and both thread budgets
  on each host, return the expected four detections with exact oracle agreement.
  All 12 background eligibility groups also pass.
- All 173 frozen non-document C++ source/header and CMakeLists hashes in each
  timing group's metadata, the complete snapshot digest, OpenCV version and
  unmodified pinned commit, detector settings, sample counts and warm-ups.
- The three x86 hosts share the benchmark binary and all 20 bundled libraries.
  The seven resolved OpenCV libraries on each x86 runtime match the bundle's
  hashes, including the i9 container's `/opt/opencv` installation. Host/container
  libc is intentionally separate and differs across operating systems.
- All 384 rows in the combined JSON and CSV, including every host/group
  statistic, agree with the independently recomputed results.

All 384 aggregated image/configuration comparisons beat OpenCV at p50, p95 and
p99. All 1,152 individual-process image p50 comparisons also win; the minimum
individual ratio is 1.037318x. The smallest aggregate p50 gain is 1.041326x for
warehouse racks on Xeon with ArUco3 disabled and four threads: 15.200693 ms for
OpenCV versus 14.597445 ms optimized. That scene's candidate process spread is
0.962%, and its reference spread is 1.755%.

The only distinct detector stderr message is the retained libpng warning about
a known-incorrect sRGB profile, appearing 420 times. Decoding is outside timing
and both routes receive the same decoded matrix. No input was changed to
suppress this warning.

Manual review confirms the x86 affinities use distinct physical cores, with
performance cores selected on i9. The execution records retain the Arm CPU
affinities, power policy and topology, build information, and before/after host
snapshots. The common runner executes groups serially on each host; task builds
finish before that host's measurements.

## Goals

Preserve all results and distinguish measured per-host improvements from a
universal speedup claim. No measurement-integrity or compatibility blocker was
found in the retained evidence. The previously reported Spark speedup does not
apply unchanged to other CPUs: the Xeon full-resolution four-thread median is
1.159x, while Thor's corresponding median is 1.961x.

## Open questions

- Existing workloads remain uncontrolled: Orin has a busy `systemd-udevd`
  service, and the workstation/Thor hosts have other workloads. Affinity avoids
  observed busy cores but cannot prevent migration or shared resource effects.
  Clock snapshots do not demonstrate fixed physical clocks.
- Thor's largest candidate process spread is 18.158% in one full-resolution
  four-thread scene. The results retain this variation. Process spreads are
  descriptive ranges, not confidence intervals.
- Compiler versions, OpenCV SIMD dispatch/IPP availability, operating systems
  and runtime libc differ. Absolute cross-host latency is not a controlled
  CPU-only hardware ranking.
- Fifty samples in the secondary mode provide limited p95/p99 resolution.
  Sixteen public photographs and positive controls do not establish deployment
  recognition recall, all-camera performance or sustained video throughput.
