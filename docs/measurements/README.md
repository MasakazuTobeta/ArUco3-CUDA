# Measurement storage

## Purpose

Keep measured conclusions and reproducible procedures in version control without
adding generated measurement output to source changes.

## Scope

This policy covers CPU, GPU and Hybrid experiments under `docs/measurements/`,
including historical measurements. It does not change reported numerical results
or measurement conditions.

## Current state

Generated timing rows, environment captures, logs, XML test results, CSV/JSON
aggregates, validation output and generated checksum records are local-only and
ignored by Git. Historical output already present in the measurement workspace
is preserved; removing it from the Git index does not delete the local files or
rewrite earlier commits. A fresh checkout contains no such historical output.

Authored reports, evaluation conditions, reproduction/audit scripts, source
patches and fixed source/input manifests remain version-controlled. In particular,
the [frozen CPU source manifest](2026-11-17-cpu-multihost/qa-source-hashes.json)
is a required audit input, not an output to discard. The
[photographic input manifest](../../cpu/experiments/real_background_manifest.json)
also remains version-controlled.

### Using or recreating the records

- Historical aggregation/audit commands require the original local result files
  and any generated input manifests described by the experiment's README. They
  cannot audit historical timings from a fresh checkout alone.
- For a new measurement, prepare the specified input images and manifests, build
  the pinned implementation and dependencies, and follow the recorded settings
  and commands. Use a fresh output directory when required by the runner. New
  timings describe the new run; they do not recover the original timing samples.
- GPU/Hybrid reproduction commands and environment conditions are in the
  [benchmark report](../benchmark-report.md) and
  [accuracy report](../accuracy-report.md).
- CPU prerequisites are recorded with the
  [six-host evaluation](2026-11-17-cpu-multihost/README.md),
  [real-background evaluation](2026-11-17-cpu-real-background/README.md) and
  [steady-state reproduction](2026-11-17-cpu-steady-reproduction/README.md).

References to local-only files use code-formatted paths rather than repository
links. Add an explicit ignore exception when a new fixed manifest is needed by a
maintained script; do not force-add generated output or broadly exempt JSON files.

## Goals

- Preserve local evidence while keeping generated output out of new commits.
- Keep reports, conditions and reproduction sources usable in a fresh checkout.
- Distinguish historical audits from new measurements and their prerequisites.

## Open questions

- No shared archival service or retention period is defined by this cleanup.
  Historical audits depend on access to the original local measurement storage.
