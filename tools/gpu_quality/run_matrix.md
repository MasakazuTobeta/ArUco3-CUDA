# Reproducible GPU quality matrix

## Purpose

Run the same broad quality checks on a frozen GPU build and retain enough
observations to distinguish detection quality from coordinate differences.

## Scope

`run_matrix.py` invokes the maintained audit and historical evaluator. It performs
no timing measurements. `render_close_chains.cpp` creates the additional 64
annotated proximity cases using only public OpenCV marker-rendering APIs. It
copies no external detector implementation. The oracle is the pinned Apache-2.0
OpenCV 4.14.0 build described in [the audit](audit.md).

## Current state

Build the reference tools and native CPU experiment using the applicable steady
preset. Generate the chain inputs into an unused ignored directory:

```bash
cmake --build build/gpu-final --target aruco3cuda_render_close_chains
build/gpu-final/tools/gpu_quality/aruco3cuda_render_close_chains \
  build/gpu-study/close-chain-images
```

The generator creates 64 grayscale PNGs and `truth.jsonl`: two or three 80-pixel
markers, gaps of -8, -2, 0, 1, 3, 7, 16 and 32 pixels, four border/connector
patterns, and all rotations. Ground-truth corner order follows marker rotation.
Negative gaps deliberately overlap markers; neither CPU recall nor absolute
corner accuracy is assumed perfect. This generator reproduces the original
64 images and truth file byte for byte with the same pinned OpenCV build.

Supply the 16 prepared industrial photographs and nine external regression
images described in the [audit](audit.md) and
[industrial-image report](../../docs/cpu-real-background-report.md). Image data
and generated logs are not committed. Expected directory cardinalities are
checked; optional `--input-manifest` verifies externally fixed SHA-256 identities.
Its object has `photos`, `external`, and `chains` objects mapping basenames to
SHA-256, including `chains/truth.jsonl` in the latter object.

```bash
python3 tools/gpu_quality/run_matrix.py \
  --build build/gpu-final \
  --photos data/cpu-real-background --external data/cpu-opencv-fixtures \
  --chains build/gpu-study/close-chain-images \
  --output build/gpu-study/quality-matrix \
  --profile OPENCV_ARM_HAL --backend SVE128 --fma SEPARATE \
  --strategy "$EXPECTED_GPU_STRATEGY" --execute
```

`EXPECTED_GPU_STRATEGY` is the exact build identity selected before running,
including every strategy and numeric limit. Copy the configured expected value,
not a value inferred from the result being validated. Orin uses `NEON/FUSED`;
the validated x86 build uses `GENERIC/GENERIC/NOT_APPLICABLE`. The output directory
must not exist. Without `--execute`, the script prints the full command plan and
does not launch any detector. Python optimization (`-O`) is rejected so validation
assertions cannot be disabled.

For immutable source exports, optionally pass both `--source` and `--manifest`.
The manifest's `source_files` object maps relative source paths to SHA-256. Every
listed source and both binary hashes are checked before and after execution.
These task-specific manifests are local records, not required repository files.
The runner also records CMake cache identity, requires `ARUCO3CPU_NATIVE=ON`, and
checks the expected profile, strategy and detection settings in every audit row.

| Group | Configurations | Detection calls |
| --- | ---: | ---: |
| 96 annotated degraded scenes, ArUco3 on/off | 2 | 236 |
| 16 industrial photographs, on/off | 2 | 54 |
| Nine external images, all 17 GPU dictionaries, on/off | 34 | 646 |
| Nondefault rendering/detection settings | 11 | 1,298 |
| 64 annotated proximity scenes, on/off | 2 | 164 |
| Total audit matrix | 51 | 2,398 |

The eleven variants are 640x480, 1280x721, cell size 2 and 7, minimum marker
ratio 0 and 0.1, minimum canonical side 16 and 48, refinement 100 iterations with
0.01 accuracy, ArUco3 off with cell size 7, and direct execution with otherwise
default settings. Each audit sequence includes repeated images and blank cuts.
All use explicit capacities of 16,384 candidates and 1,024 markers.

Two additional historical evaluations each process 91 images, retaining their
unchanged default capacities of 4,096 candidates and 1,024 markers. Their reported
workspace peak/capacity and allocation counts are recorded separately from the
larger audit and benchmark budget. Historical evaluator quality is available as
aggregate/bucket reports; detailed per-truth checks apply to the broad audit.

Acceptance requires exact optimized-CPU agreement, successful positive controls,
no discrete GPU regression, stable repeated-frame observations, and no greater
than 0.001-pixel worsening in either common-marker ground-truth RMSE or maximum
corner error. Every raw delta is retained, including positive deltas within that
numerical band. Unannotated inputs additionally check the audit's one-pixel
compatibility comparison; this does not create ground truth for those images.
Resource failures stop acceptance and preserve partial logs. Completion is not
inferred from a zero exit status alone: row counts and sequence numbers must match.

The maintained runner was checked by replaying all 53 existing case records and
then used for actual final-v2 GPU evaluation on Spark and RTX4090. Its promotion
made source-manifest verification optional and added historical memory extraction;
image selections, settings and quality thresholds were unchanged.

## Goals

- Make the exact tested matrix reproducible without committing measurements.
- Retain input, configuration, binary and optional source identity with results.
- Detect skipped work, incomplete output, resource failures and stale-frame reuse.

## Open questions

These finite cases do not prove equivalence for every image or configuration.
Unannotated compatibility cases cannot establish independent false-positive or
recall rates. Rendered pixels and CPU outcomes can vary with the pinned host
OpenCV build; evaluate against each host's actual CPU oracle and retain hashes.
