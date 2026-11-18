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

The maintained [matrix runner and chain generator](run_matrix.md) reproduce the
51-case broad matrix plus both historical 91-image evaluations. Generated inputs,
source snapshots, per-frame observations and timing logs remain local ignored
outputs; the source tools and evaluation conditions are versioned.

The audit uses the pinned OpenCV 4.14.0 public API and the existing synthetic
[quality renderer](../cpu_quality/quality_scenes.md). It does not use official
ArUco GPLv3 sources or images. The dictionary, ArUco3 setting and corner-refinement
mode are identical across the three routes. Other settings retain the common
defaults. OpenCV uses one thread.

Configuration probes can set `--width` and `--height` for rendered scenes,
`--cell-size`, `--min-canonical-side`, `--min-marker-ratio`,
`--refine-iterations`, and `--refine-accuracy`. Defaults reproduce the original
audit. The detector settings are mapped explicitly to both CPU implementations
and the GPU, and recorded in every frame's `configuration` object. Width and
height do not resize supplied photographs. ArUco3 on uses subpixel refinement;
ArUco3 off uses no refinement, matching the GPU's supported combinations.
`--max-candidates` and `--max-markers` expose the device resource budgets and
are recorded alongside the algorithm settings. A budget overflow is an explicit
execution failure, not a detection-quality result. Keep that failure record
when rerunning with a larger budget; do not omit the triggering image.

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

### Contour and quad-perimeter audit checkpoint

An intermediate DGX Spark build using exact contour paths and Euclidean
quad-perimeter grouping completed the same 96-scene sequences. With an explicit
16,384-candidate budget, both modes retained every CPU-detected truth and correct
rotation, with no false positives:

| ArUco3 | OpenCV true positives | GPU true positives | Common-truth CPU RMSE | Common-truth GPU RMSE |
| --- | ---: | ---: | ---: | ---: |
| On | 113 | 118 | 2.424692 px | 2.357981 px |
| Off | 124 | 127 | 1.698154 px | 1.698154 px |

No common truth had a worse GPU corner maximum or squared-error sum. The 16
factory photographs remained empty in both modes. The default 4,096-candidate
off-mode run failed explicitly on `quality_negative_3`; the public CPU candidate
oracle counts 1,820 quads per threshold window, or 5,460 before grouping. That
resource failure is retained separately from the larger-budget quality result.

The external 17-dictionary/two-mode matrix still failed: 278 CPU observations
versus 250 GPU observations across 41 CPU-positive image/configuration pairs.
Eight pairs lost `failmask9.png` ID 0, a regression against the frozen GPU
baseline; four 5x5/off pairs each lost five observations in `regression_2492.png`.
CPU instrumentation showed that all these missing observations decode only from
eligible close contours after the largest representative fails. These findings
motivate [device close-contour retries](../../src/core/candidate_retry.md); the
checkpoint itself is not an accepted final quality result.

Raw checkpoint results remain local in ignored
`build/gpu-study/contour-group-quality-16384/`, including binary hash, full
observations, per-truth corner comparisons, and completion records for all 936
sequence calls. The original 4,096-candidate failure remains in
`build/gpu-study/contour-group-quality/`. These paths are not checkout contents.

### Close-contour and exact-resize checkpoint

**Superseded descriptor check:** subsequent link-order review found that the
audit/evaluator could resolve profile getters from the CPU library, producing a
GENERIC/NEON runtime descriptor despite compiled ARM kernels. The observations
below remain preserved diagnostic results, but are not acceptance evidence for
the intended SVE128 profile. Primitive exact tests used the correct core getters.
A core-only getter translation unit and CPU-first/core-first link tests now
require a fresh audit with checked per-frame metadata.

The next Spark checkpoint adds device close-contour retries and the
`OPENCV_ARM_HAL` / `SVE128` / `SEPARATE` resize compatibility descriptor. The
38-case matrix completed 936 sequence calls at the same explicit
16,384-candidate budget. All 278 external CPU observations were recovered across
17 Dictionaries and both modes, with no discrete differences or corner shifts
above the existing 1 px compatibility threshold. All 16 factory photographs
remained empty in both modes. Every one of the 304 revisited frames reproduced
all three routes' complete observations exactly; annotated rotations agreed.

On the 96 annotated scenes, CPU/GPU true positives were 113/118 with ArUco3 on
and 124/127 with it off; every route had zero false positives. Common-truth
corner RMSE was 2.424692 px for both CPU and GPU with ArUco3 on, and 1.698154 px
for both with it off. No common truth had a worse GPU maximum or RMSE. The
previous generic resize happened to improve some default-scene corners, but it
also caused material regressions under nondefault settings. Exact resize is
accepted against CPU quality, without claiming uniformly better corners than
every earlier GPU implementation.

Eleven additional 118-call variants cover 640×480 and 1280×721, canonical cell
sizes 2 and 7, marker ratios 0 and 0.1, canonical minimum sides 16 and 48,
100 refinement iterations with 0.01 accuracy, off-mode cell size 7, and direct
execution. They have no discrete regressions or failed positive controls. The
three previously material corner regressions at ratio 0.1 / sides 16 and 48
are resolved. The largest remaining per-truth maximum/RMSE worsening is
0.000003096 px. All raw deltas are retained; values at or below 0.001 px are
reported separately as numerical variation, not erased or used to excuse a
missed marker, wrong rotation or false positive.

This checkpoint used the explicitly nonfused Spark Carotene horizontal
arithmetic before its equivalent named CMake descriptor was integrated. Orin's
pinned CPU build requires a separately verified fused descriptor. Cross-hardware
validation and any later performance changes must retain their own source hashes
and exact pixel gates. The ignored local records are
`build/gpu-study/resize-arm-quality-16384/` (matrix, frozen core hashes and input
identity checks) and `build/gpu-study/resize-arm-variants/` (all variant corner
deltas). These raw outputs are not versioned or present in a fresh checkout.


The corrected core-only getter checkpoint subsequently reran the eight focused
resize/retry tests, all 38 matrix cases, all eleven variants, and both historical
91-scene modes. All 2,588 audit frames (including the three critical variant
reruns) explicitly reported `OPENCV_ARM_HAL`, `SVE128`, and `SEPARATE`. The counts
and corner conclusions above were reproduced with the intended descriptor;
304 revisits and the default/direct sequence remained exact. The 91-scene CPU/GPU
true positives were 88 with ArUco3 on and 436 with it off, with zero false
positives and identical aggregate corner errors. Valid direct-arithmetic control
records now reside in ignored `build/gpu-study/resize-linked-quality-16384/`,
`resize-linked-variants/`, and `resize-linked-history/`.

A further 64 rendered two/three-marker chains vary gaps from −8 to 32 pixels,
rotations, black/gray border connectors and overlapping outlines. Both modes
completed 82 sequence calls each. These cases exercise the documented difference
between transitive GPU grouping and CPU group formation; they have not
established a new quality failure or justified a wider grouping change. Their
rendered ID/corner truth and all raw observations are retained locally.

### Combined Spark quality gate before border-margin review

The checkpoint below predates the border-margin correction. Its additional GPU
border detections were later found to violate the configured margin; see the
[review follow-up](../../docs/gpu-steady-state-report.md#current-state).
Corrected 96-scene CPU/GPU true positives are 113/113 (on) and 124/124 (off).
Earlier prototype counts in this document are historical observations, not
evidence of better recall under identical border settings.

The final combined checkpoint uses ordered contours with edge-aggregate counting,
warp approximation/materialization and eligible-edge compaction; fused threshold,
demand pyramid, density-selected fixed-point Otsu (96 occupied-bin limit), a
512-block persistent candidate grid, and `KLEIDI_PRECOMPUTED`. The runtime profile
is explicitly `OPENCV_ARM_HAL` / `SVE128` / `SEPARATE`; the optimized CPU library
is built with `ARUCO3CPU_NATIVE=ON`.

All 2,398 audit calls (936 matrix, 1,298 configuration variants, 164 close-chain
sequence calls) assert that exact strategy/profile and the 16,384-candidate /
1,024-marker budgets. Inputs, configuration, GPU observations and per-truth
metrics are exactly equal to the validated direct-arithmetic checkpoint. The
main 96-scene true-positive counts remain CPU/GPU 113/118 with ArUco3 on and
124/127 with it off, with zero false positives. External observations remain
278/278 across all 17 supported Dictionaries and both modes; all factory images
remain empty. The 304 matrix revisits reproduce complete observations exactly.
The largest variant corner worsening against CPU is 0.000003096 px; no per-truth
maximum or RMSE worsens by more than 0.001 px.

The historical 91-scene evaluator also passes both modes (182 images total),
with CPU/GPU true positives of 88 and 436 respectively, zero false positives,
correct rotations and identical aggregate corner errors. The rendered chain
cases retain all 100 CPU observations in on-mode and all 93 in off-mode, with
no unknown/duplicate IDs or material common-truth corner degradation.

This is tested-scope quality acceptance, not proof for every possible image.
External photographs remain unannotated compatibility inputs, the explicit
candidate budget remains a resource condition, and the documented transitive
GPU grouping difference remains. Cross-hardware final-combination validation and
performance measurements are separate gates. The final local records are
`build/gpu-study/final-steady-quality-16384/`, `final-steady-variants/`, and
`final-steady-history/`; `final-acceptance.json` includes source-manifest and
CMake-cache hashes. Raw outputs remain ignored rather than versioned.
