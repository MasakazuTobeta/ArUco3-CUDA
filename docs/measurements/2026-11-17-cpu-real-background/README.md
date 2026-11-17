# Real-background CPU measurement evidence

## Purpose

Retain reproducible evidence for the [industrial-background report](../../cpu-real-background-report.md).

## Scope

Sixteen public industrial photographs, unchanged CPU implementation `d677c4f`,
actual pinned OpenCV 4.14.0, MIP 36h12, one/four threads, ArUco3 enabled/disabled.
Inputs are local-only, aspect-preserving photographs with longest side at most
1920 pixels. No deployment-camera or independent marker ground truth is implied.

## Current state

Generated measurement results, logs and environment captures referenced below
are local-only and ignored by Git. This document and reproduction sources remain
version-controlled; a fresh checkout does not contain the recorded outputs.

- [Plan](plan.md) fixes selection/eligibility and repeated measurement conditions.
- [Source manifest](../../../cpu/experiments/real_background_manifest.json) records
  attribution and transformations; input hashes (local `input-hashes.json`) pin timed PNGs.
- [Orchestrator](run_measurements.py), execution record (local `execution.json`), and
  log (local `orchestration.log`) preserve commands and successful completion.
- `positive-control-*` confirms four known markers in each strategy;
  `eligibility-*` checks the selected photographs. These one-sample durations
  are accuracy-only and are excluded from reported performance.
- `aruco3-1t`, `aruco3-4t`, `fullres-1t`, `fullres-4t` each contain three complete
  paired processes, environment/configuration/source hashes, row validation,
  and a maintained-summarizer output. Fullres means full processed-image size.
- Aggregate (local `aggregate.json`) retains per-group wins, tails and process spread.
- Independent audit (local `audit.json`) recomputes all summaries and validates hashes,
  configuration and positive controls; historical checksums (local `checksums.json`)
  identify captured files before later documentation edits.
- Post-run input verification (local `input-verification.log`) confirms originals and
  prepared images remained unchanged.

Every timed group retained all sixteen photos. All output counts are zero for
both routes; pre/post exact output checks pass. The 384 timed measurement rows
are summaries of detector-call samples, not saved results for every timed call.

## Goals

- Preserve complete paired records and any unfavorable conditions.
- Separate photographic input preparation, compatibility and steady-state timing.
- Make all reported medians and tails traceable to retained raw rows.

## Open questions

- The reported p95/p99 are medians of process percentiles, not pooled samples;
  secondary 50-sample tails have limited resolution.
- Actual camera exposure, blur, compression, native 4K and other CPUs remain
  outside this small photographic evaluation.
