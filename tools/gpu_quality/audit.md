# GPU quality audit

## Purpose

Check GPU optimizations against CPU detection quality on inputs extending the
historical synthetic corpus. Preserve every output needed to distinguish a
corner difference from missed markers, false positives or wrong rotation.

## Scope

`audit.cpp` compares official OpenCV, the optimized CPU detector and the GPU
detector. It uses the existing 96-frame CPU quality renderer, or a supplied
directory of unannotated grayscale inputs. It makes no latency claim.

## Current state

The audit uses the pinned OpenCV 4.14.0 public API and the existing synthetic
[quality renderer](../cpu_quality/quality_scenes.md). It does not use official
ArUco GPLv3 sources or images. The dictionary, ArUco3 setting and corner-refinement
mode are identical across the three routes. Other settings retain the common
defaults. OpenCV uses one thread.

The default sequence shuffles the 96 frames with a fixed seed, revisits eight,
and inserts 14 blank cuts. The detector instances persist across all 118 calls.
Input pixels overwrite a strided ROI in one host allocation, with differently
poisoned surrounding storage, and upload to reusable pitched device storage.
The default explicit CUDA stream uses graph dispatch; `--direct` instead uses
the default stream to exercise direct dispatch. A changed-size image directory
also exercises workspace/graph replacement.

Every JSONL frame records the grayscale pixel hash, full oriented detections,
the original rendered truth when available, CPU/GPU differences at the existing
1-pixel compatibility tolerance, and per-truth quality metrics. GPU corner
quality must be judged against the rendered geometry on common detections;
corner compatibility with the CPU alone is not a quality measurement.

Exit codes:

- `0`: execution completed without the discrete failures below. This alone is
  **not** a corner-quality pass or proof of field accuracy.
- `1`: invalid input, CUDA failure, buffer overflow, or another execution error.
- `2`: optimized CPU differs exactly from OpenCV, GPU loses a CPU-detected truth
  or correct rotation, GPU adds false positives on annotated images, a clean
  positive control returns no detections, or unannotated images have discrete
  CPU/GPU differences. The complete log is still emitted to diagnose failures.

For unannotated photographs, differences are compatibility findings; they are
not independently established false positives or recall losses. For annotated
images, additional correctly detected truth is allowed. Repeated scenes and
blank cuts must be excluded from the primary unique-scene quality aggregate
and assessed separately for changing-frame stability.

Example invocations after building the audit:

```bash
gpu_quality --aruco3 1 > build/gpu-study/quality-aruco3.jsonl
gpu_quality --aruco3 0 > build/gpu-study/quality-fullres.jsonl
gpu_quality --aruco3 1 --direct > build/gpu-study/quality-direct.jsonl
gpu_quality --images data/prepared-industrial > build/gpu-study/quality-photos.jsonl
gpu_quality --dictionary DICT_4X4_50 > build/gpu-study/quality-4x4.jsonl
```

Run and retain the audit against the frozen pre-optimization library as well as
the candidate. Require a completion record; an exit or partial file is not
evidence that all inputs ran. Generated output remains local and ignored.

### Initial baseline diagnosis

The frozen baseline `badb673` was audited on DGX Spark with OpenCV 4.14.0 and
GCC 13.3. The 96 unique synthetic scenes contain 136 true markers. Replays and
blank cuts are excluded from these counts:

| ArUco3 | Route | True positives | False positives | False negatives |
| --- | --- | ---: | ---: | ---: |
| On | OpenCV / optimized CPU | 113 | 0 | 23 |
| On | GPU baseline | 56 | 0 | 80 |
| Off | OpenCV / optimized CPU | 124 | 0 | 12 |
| Off | GPU baseline | 65 | 0 | 71 |

Detected rotations agree with truth. CPU optimized outputs match OpenCV exactly.
Graph and direct GPU outputs match each other exactly on all 118 ArUco3-enabled
sequence calls, so the additional misses are not explained by graph staleness.

All 12 `multi_dense` frames and projective frames 2, 5, 8 and 11 have background
lines connecting marker borders into large foreground components. CPU detection
uses hole contours inside those components, whereas the original GPU method
produces one quadrilateral per foreground component. For two representative
641x479 inputs, public OpenCV `RETR_LIST`/`CHAIN_APPROX_NONE` gives the accepted
contours below on the 320x239 segmentation image at window size 3. Windows 13
and 23 produce no eligible quadrilateral for these examples.

| Frame | ID | Unit-chain points | Signed contour area | Approximate corners in segmentation coordinates |
| --- | ---: | ---: | ---: | --- |
| `quality_multi_dense_0` | 79 | 373 | +8865.5 | (32,73), (126,74), (128,165), (34,167) |
| `quality_multi_dense_0` | 79 | 365 | +8887.5 | (203,65), (295,84), (277,175), (185,155) |
| `quality_projective_2` | 35 | 227 | +4429.5 | (172,74), (206,131), (147,164), (115,110) |

Positive signed area identifies hole contours in this OpenCV convention. The
approximation epsilon is 0.03 times the unit-chain length, and the ArUco3 minimum
length is 128. A local instrumented copy of the CPU detector, using the public
OpenCV contour route, confirmed these exact paths were successfully decoded.

The 16 industrial photographs give zero detections on all three routes in both
ArUco3 modes. Nine external fixtures with the default MIP dictionary also give
zero, which is not positive detection coverage. With `DICT_4X4_50`, `failmask9`
gives one agreeing detection and `regression_3192` gives two CPU detections but
only one GPU detection (missing ID 4). The complete external matrix covers all
17 GPU-supported dictionaries and both ArUco3 modes: 306 unique image/configuration
pairs and 646 sequence calls, including the blank cuts and replays. All grayscale
pixel hashes and dimensions agree across configurations, and optimized CPU matches
OpenCV throughout. There are 41 CPU-positive image/configuration pairs, totaling
278 CPU detections versus 244 GPU detections. Across the unique pairs, comparison
finds 36 missing CPU detections and two extra GPU detections in 14 pairs, plus
23 corner displacements above the existing 1-pixel compatibility tolerance.

The main discrete differences are:

| Fixture | Configuration | CPU / GPU detections | Difference |
| --- | --- | ---: | --- |
| `regression_3192.png` | All four 4x4 dictionaries, either ArUco3 mode | 2 / 1 | Missing ID 4 |
| `regression_2492.png` | All four 5x5 dictionaries, ArUco3 off | 8 / 1 | Missing three instances of ID 13 and IDs 4, 14, 15, 1 |
| `recover.png` | 4x4-250, ArUco3 off | 0 / 1 | Extra ID 158 |
| `recover.png` | 4x4-1000, ArUco3 off | 1 / 2 | Extra ID 158 |

These unannotated fixtures establish compatibility gaps, not independent recall
or false-positive estimates. The overlapping dictionaries are separate tested
configurations, not independent image samples. Complete records, input identities
and process completion checks are local under
`build/gpu-study/external-baseline-matrix/`; two previously completed matching
cases were reused and 32 additional processes were run.

Raw audit records and exploratory component/contour diagnostic sources remain
local under `build/gpu-study/`. These baseline failures must be repaired before
claiming CPU-equivalent quality; faster execution of the same failures is not
an acceptance result.

### Targeted regression requirements

- Generate two one-pixel foreground rectangular rings in a 300x180 zero mask,
  with inclusive corners (20,20)–(100,120) and (160,40)–(260,140). Connect them
  with a foreground line (100,70)–(160,70). OpenCV returns two eligible hole
  contours of lengths 356 and 396; the outer contour has length 876 and a
  ten-vertex approximation. A single connected component must therefore yield
  both hole candidates. Generate this fixture in the test, not as a committed
  image or measurement log.
- Extend this case with diagonal contacts, nested rings and border clipping;
  compare ordered paths/corners against public OpenCV on isolated owned masks.
- Retain the full rendered `multi_dense` and projective examples as detection
  regressions, including repeated IDs. Candidate-count agreement alone does
  not establish successful decoding or correct suppression.
- Overwrite the same device image with a blank cut and then the earlier mask;
  compare graph and direct output to catch stale lists or contour scratch.
- Preserve counts, ID/rotation, false positives and common-detection corner
  quality separately. Overall corner RMSE can improve merely because hard
  markers were missed and excluded from that metric.

## Goals

- Establish existing quality gaps before making speed claims.
- Retain CPU-detected true markers and their correct IDs/rotations.
- Avoid additional false positives or worse corner quality against geometry.
- Verify output changes across frames without temporal reuse.

## Open questions

- Corner noninferiority requires review of common-marker errors and predefined
  acceptance criteria; the process exit code deliberately does not hide this.
- Unannotated real images provide compatibility checks, not ground truth.
- This initial audit does not replace the full 91-scene evaluator, all-dictionary
  rotation suite, boundary unit tests or Compute Sanitizer.
