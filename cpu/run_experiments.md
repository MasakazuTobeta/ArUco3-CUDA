# CPU experiment runner

## Purpose

Run independent paired detector measurements while keeping each invocation's
metadata and output together.

## Scope

`run_experiments.py` runs an existing benchmark binary on Linux with an explicit
CPU affinity. It does not compile the detector or infer which source produced a
binary. Binary and source hashes are recorded for the caller's verification.

## Current state

Run from the repository root, selecting available CPU IDs and a **new** output
directory:

```bash
python3 cpu/run_experiments.py \
  --binary build/cpu-opt/cpu/aruco3cuda_cpu_benchmark \
  --input-dir data/cpu-experiment --output-dir results/new-experiment \
  --cpu-list 0 --threads 1 --runs 3 --iterations 200 --warmup 30
```

The binary path is resolved once; hashing and execution use that same absolute
path instead of selecting a potentially different executable from `PATH`.
After argument, affinity, binary and Git metadata preflight, the runner atomically
creates the final output directory. Existing directories are rejected, including
empty directories, incomplete attempts and directory symlinks. Do not precreate
the final directory; parent directories may already exist.

Metadata and per-process outputs are created exclusively. Two concurrent runs
targeting the same directory cannot overwrite each other's evidence. When a
benchmark fails, its metadata and any partial output remain available and the
runner returns a failure. A retry must use a different output directory; the
runner does not resume, erase or overwrite the failed attempt.

The [regression tests](../test/cpu/test_run_experiments.py) exercise output
ownership, reruns, incomplete directories, symlinks, missing and failing binaries,
concurrent invocations, binary-path provenance, and the multi-host orchestrator
using temporary fake benchmark executables. They need no OpenCV build or GPU.

`summarize_experiments.py` requires every numbered run, exactly one matching
benchmark configuration per process, and nonempty paired measurements. It rejects
invalid latency percentiles, inconsistent image hashes/dimensions, and changing
detection outputs for the same input/variant. It still reports discrepancies for
explicit algorithm-changing variants; aggregation is not a correctness oracle.
The [summary tests](../test/cpu/test_summarize_experiments.py) cover malformed and
mixed batches as well as valid percentile and speedup calculations. Both tooling
suites run in CI without OpenCV and through the standalone CPU CTest target.

## Goals

- Preserve existing results byte for byte when an invocation is rejected.
- Ensure recorded binary identity matches the executable selected for launch.
- Keep partial failure evidence inspectable without treating it as a complete run.

## Open questions

The runner does not protect against unrelated processes replacing a binary or
input after preflight, or intentionally modifying an active result directory.
Use immutable binaries and inputs while measuring. Successful process exit is
not an independent proof of detector accuracy or complete result validation.
