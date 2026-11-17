# Detection Pipeline Design

## Purpose

This document defines the stage breakdown used when decomposing the ArUco3 detection strategy onto CUDA, the parallelization approach for each stage, the handling of variable-length output, the synchronization points, and the compatibility conditions against the CPU baseline results.

## Scope

The scope covers the stage breakdown of the detector core, the GPU suitability assessment of each stage, the design options for quadrilateral candidate extraction, the workspace design, and the handling of machine differences. Pose estimation, ChArUco boards, and the AprilTag-specific route are out of scope.

## Current state

S1 through S11 are implemented in CUDA. Fresh builds retain the component-extrema route and full-pyramid baseline. The explicit `dgx-spark-steady` and `jetson-orin-steady` presets select ordered contours, close-contour retry, and the steady-state strategies described below. They use the same detection algorithm on both devices; resize arithmetic is selected to match each pinned CPU OpenCV build. See [the strategy configuration](../../cmake/ArUco3CudaStrategies.md) and [ADR-0004](../adr/0004-device-contours-for-cpu-quality.md).

The historical component route matched 90 of 91 images, with a 3.804 px corner difference on one occluded image; the hybrid route matched all 91. Those results remain in [the historical accuracy evaluation](../accuracy-report.md), but do not establish compatibility on the expanded corpus. Background lines can join multiple marker borders into one foreground component while leaving separate detectable hole contours.

The combined Spark checkpoint passes 2,398 changing-frame audit calls plus both historical 91-scene modes. On the 96 annotated scenes, CPU/GPU true positives are 113/118 with ArUco3 on and 124/127 with it off, with no false positives. All 278 external CPU observations across 17 Dictionaries and both modes are recovered; factory photographs remain empty. The explicit candidate budget is 16,384. These are tested-scope quality results, not a universal recall or performance claim. See [the maintained audit](../../tools/gpu_quality/audit.md) for configurations, corner deltas, source identities, and the separate cross-hardware acceptance gate.

Unchanged-layout submission remains asynchronous on the host. Reinitialization, layout changes, result readback and exceptional recovery have explicit completion boundaries; see [submission and storage lifetime](../../src/core/detector.md).

For the ArUco3 detection strategy of OpenCV 4.x, which is the compatibility target, the following observed specification has been confirmed from the Apache-2.0 public headers and sources. For the retrieval sources and hashes, see [the Code Provenance record](../code-provenance.md).

### Observed specification of the OpenCV 4.x ArUco3 route

| Symbol | Corresponding parameter | Default |
| --- | --- | --- |
| `S` | `minSideLengthCanonicalImg` | 32 |
| `tau_i` | `minMarkerLengthRatioOriginalImg` | 0.0 |
| `W`, `H` | Width and height of the input grayscale | - |

1. Compute the downscale factor as `fxfy = S / (S + max(W, H) * tau_i)`. When `useAruco3Detection` is false, set `fxfy = 1`.
2. Build a grayscale pyramid of `num_levels = (int)(log2(W * H / S^2) / 2)` levels, with a scale of 2 per level.
3. Compute the level at which corner upsampling starts as `closest_pyr_image_idx = round(log2(1 / fxfy^2) / 2)`.
4. When `fxfy != 1`, downscale the input to `round(fxfy * W) x round(fxfy * H)`. Candidate extraction is performed only on this segmentation image.
5. The number of adaptive thresholding windows is `nScales = (winMax - winMin) / winStep + 1`. The default is the three window sizes 3, 13, and 23.
6. Threshold per window, obtain quadrilateral candidates by contour extraction and polygonal approximation, and merge the candidates from all windows.
7. Group nearby candidates using `minMarkerDistanceRate` and `minGroupDistance`, and keep a representative candidate.
8. For each candidate, perform the perspective transform and cell sampling at the pyramid level corresponding to the side length, then run border verification and Dictionary matching.
9. When ArUco3 is enabled, corner refinement is forced to the subpixel method. The corners are subpixel-corrected at each level while doubling one pyramid level at a time, and the window is 5 when the longest side exceeds 1080 and 3 otherwise.
10. The combination of `useAruco3Detection` being true with `S == 0` and `tau_i == 0.0` is rejected by OpenCV.

Of the above, the downscale factor and the segmentation image size have been confirmed by measurement in the automated tests of the CPU baseline runner. For 1280x720 with `S = 32` and `tau_i = 0.05`, `fxfy = 0.3333` and the segmentation image is 427x240.

The minimum side length that ArUco3 can detect is determined by the following expression. It follows from the requirement that the side length after downscaling be at least `S`.

```
side_px >= S + max(W, H) * tau_i
```

Note that `tau_i` itself is not the lower bound. For 1280x720 with `S = 32` and `tau_i = 0.05`, the lower bound is 96 pixels. In measurements, a 96 pixel marker was not detected, while a 128 pixel one was. Exactly at the boundary value, the result is affected by resampling.

For the same image, the detection time of the CPU baseline implementation was 1.112 ms at `tau_i = 0.05` and 5.917 ms at `tau_i = 0` (no downscaling). The detected IDs agree between the two. This shows that the ArUco3 detection strategy is effective on the CPU side as well, and at the same time that no benefit is obtained while `tau_i` is left at its default of 0.0.

The above is a compatibility baseline, not ground truth. The default value of `tau_i` is 0.0, in which case `fxfy = 1` and no downscaling occurs. Evaluating the speed benefit of ArUco3 requires setting `tau_i` explicitly.

## Goals

### Agreement of preprocessing with OpenCV

Compatibility is measured against the public operations of the pinned OpenCV build, including its selected host HAL.

| Stage | Corresponding OpenCV processing | Current compatibility condition |
| --- | --- | --- |
| S1 pyramid | `buildPyramid` (`pyrDown`) | Exact generated pixels; demand and full routes agree on every requested level |
| S2 segmentation | Public `resize(..., INTER_LINEAR)` | Exact bytes on maintained profile-specific pixel tests; generic and ARM HAL profiles remain explicit choices |
| S3 adaptive thresholding | `ADAPTIVE_THRESH_MEAN_C` and `THRESH_BINARY_INV` | Exact bytes for the separable and supported fused routes |

The pyramid uses integer arithmetic with `[1,4,6,4,1]`, `BORDER_REFLECT_101` and `(sum + 128) >> 8` rounding. The `DEMAND` strategy moves generation after grouping and computes the maximum required level on the device, including refinement needs. It generates complete prefix planes, retaining their allocation and fixed graph launch sequence. Empty frames request no generated levels; unused kernels return before reading image storage. Every invocation overwrites the demand scalar, so unchanged input addresses do not permit reuse of old pixels. Sparse tiles are not implemented.

Public `INTER_LINEAR` is not interchangeable with `INTER_LINEAR_EXACT`. The pinned implementation can select generic, Kleidi or Carotene arithmetic with different coordinate rounding and vector-tail behavior. Earlier generic GPU resize differed by one gray level on some ARM inputs; expanded nondefault tests showed that this could move candidate corners and materially worsen refined geometry. The maintained ARM profile reproduces the observed public resize arithmetic rather than accepting that difference.

The measured Spark profile is `OPENCV_ARM_HAL` / `SVE128` / `SEPARATE`; Orin uses `OPENCV_ARM_HAL` / `NEON` / `FUSED`. These describe the pinned CPU build, not GPU instructions. Both pass zero-tolerance public-resize tests on strided inputs, vector tails, small dimensions and nondefault ratios. A different OpenCV build requires its own validated descriptor. `KLEIDI_PRECOMPUTED` caches layout coefficients only for that path; coefficients contain no input pixels. Other paths retain direct arithmetic. See [resize configuration](../../cmake/ArUco3CudaResizeProfile.md) and [preprocessing design](../../src/core/preprocess.md).

`FUSED_AUTO` computes supported threshold windows in a tiled multiwindow kernel and falls back to the separable implementation outside its supported range. It preserves the same output bytes and shared-buffer ordering.

S3 matches OpenCV exactly. The mean is `boxFilter` applied with normalization, the border is `BORDER_REPLICATE`, rounding is round-half-to-even, and the test is "pixel - mean <= -floor(constant)" for 255. An even window is rounded up to an odd one. In the separable route, row/column sums retain the exact integer total without intermediate rounding. The fused route preserves the same normalization and comparison.

### Agreement of the option C hybrid route with OpenCV

For option C, the difference from the CPU baseline has been measured. Across 36 comparisons made up of 12 synthetic scenes (varying marker count, side length, in-plane rotation, projective distortion, blur, noise, illumination gradient, and resolution from 640x480 to 1920x1080) and 3 settings (ArUco3 enabled, OpenCV default, OpenCV default + subpixel correction), there were 0 missed detections, 0 extra detections, and a maximum corner difference of 0.0000 pixels. Scenes in which a marker is nested inside a surrounding black frame also match. The same results were obtained on DGX Spark and Jetson AGX Orin.

In that historical corpus, thresholding flips from the former S2 approximation did not change the contour vertices. This observation does not generalize to the expanded nondefault corpus; the explicit ARM resize profiles above address the later failures. The historical hybrid tests retain their 0.5 px bound.

Agreement required making the proximity merging of S6 follow the same rule as OpenCV. Because S3 tries three windows, slightly different candidates are obtained from the same marker. OpenCV ranks candidates by quadrilateral Euclidean perimeter and uses it in grouping and representative selection. The ordered GPU route computes this quantity separately from contour point count, which remains the sampling-level input. See [grouping details](../../src/core/candidate_group.md). If no representative is selected and the first candidate found is kept instead, the small candidate originating from the smallest window remains and the corners move inward. For a marker with a 160 pixel side at 1280x720, this difference was 7.9 pixels in original-size terms. On the segmentation image at a downscale factor of 1/3 it is 2.6 pixels, a magnitude that the 1 gray level difference of S2 cannot explain.

Among the non-representative candidates, eligible close contours are retained as alternatives when identification of the representative fails. The ordered GPU route implements these retries on the device, preserving the representative geometry for the containment tree and its canonical level for alternatives; the first successful eligible retry supplies the output corners. See [retry ordering and bounds](../../src/core/candidate_retry.md). In addition, the containment relations among the candidates are held as a tree; identification proceeds from the inner candidates, and once a marker is confirmed, its outer (parent) candidate is excluded from identification. Since both the outer and inner boundaries of a marker's black frame become candidates, without this suppression the same marker would be counted twice.

When `useAruco3Detection` is enabled, OpenCV overrides the corner refinement setting to `CORNER_REFINE_SUBPIX`, because subpixel correction is the only means of mapping corners obtained on the downscaled image back to the original size. This project does not override the setting. `Detector::initialize()` rejects the two combinations instead and returns `Status::kInvalidConfig`: `use_aruco3_detection_` enabled together with `corner_refine_method_ == kNone`, because the corners would stay in downscaled coordinates, and `use_aruco3_detection_` disabled together with `kSubpix`, because the pyramid then has a single level and the refinement never runs. Silently rewriting the setting would leave the caller's coordinate systems inconsistent without saying so.

### Agreement of the S7 perspective transform with OpenCV

For S7, we compared against calling OpenCV's `getPerspectiveTransform` and `warpPerspective` with `INTER_NEAREST`.

| Comparison target | Mismatching pixels |
| --- | --- |
| OpenCV on x86_64 | **0 / 40960** |
| OpenCV on aarch64 | 1 / 40960 (0.0024%) |

#### OpenCV itself does not match bit-for-bit across machines

`INTER_NEAREST` of `warpPerspective` has three routes.

| Route | Selection condition | Multiply-add |
| --- | --- | --- |
| `WarpPerspectiveLine_SSE4::processNN` | SSE4.1 is available on x86_64 | Not fused |
| `WarpPerspectiveLine_ProcessNN_CV_SIMD` | `CV_SIMD128_64F` is enabled; the NEON of aarch64 falls here | `v_muladd` lowers to `vfmaq_f64` and **is fused** |
| scalar | When neither of the above is available | Not fused |

Taking the SHA256 of the canonical image for the same input image and the same corners on two machines, the values differed.

```
DGX Spark GB10 (aarch64)   0042a7b7c3a3b1263796c7d8
GeForce RTX 5070 Ti (x86)  b60c6c57a26e0910ba6d9cba
```

Therefore "match OpenCV bit-for-bit" cannot be defined without specifying a machine.

#### The choice made in this project

We align with the non-fusing side (the semantics of scalar and SSE4.1). `cell_sample.cu` is compiled with `-fmad=false`.

- It matches OpenCV on x86_64 exactly.
- It differs from OpenCV on aarch64 in a very small part of the rounding boundary. In measurement, this is 1 pixel out of 40960.

Aligning with the fusing side is also possible, but it would then not match x86_64. Either choice fails to match one of the two. We chose the non-fusing side because the scalar route is the reference for OpenCV's semantics and does not depend on the presence of SIMD.

#### Details that matter for agreement

These are the points that the implementation must observe. Dropping any of them shifts the referenced pixel by one at a rounding boundary.

1. `a[i][6]` and `a[i][7]` of the coefficient matrix widen a **single-precision product** to double precision. Multiplying in double precision shifts the matrix by up to 1.9e-3 relative.
2. The 8-variable linear system is solved by Gaussian elimination with partial pivoting. On ties, the row with the smaller index is kept. The elimination forms `d = -1/A[i][i]` once and multiplies by it.
3. The inverse matrix is built with the 3x3 cofactor formula. The cofactors are formed first, and `1/det` is multiplied last.
4. The projective division takes the reciprocal first and then multiplies. Writing it as a division makes about 26% of the pixels differ by 1 ULP.
5. Rounding is round-half-to-even. `floor(v + 0.5)` is wrong for both ties and negative values.
6. When the denominator is exactly 0, the input at (0, 0) is referenced rather than a boundary value.

If one side of the canonical is at most 32, the horizontal block split of OpenCV (`bw0`) coincides with the side, so the origin is always 0 and there is no need to reproduce the ordering of the partial sums. With the default settings and DICT_ARUCO_MIP_36h12, the side is 32. Settings with a side greater than 64 require reproducing the split.

### First half of S8: Otsu and border verification

Cell ratios are computed from the canonical image, and candidates are filtered by the number of errors in the outer ring of cells. The three machines measured for this match the CPU baseline exactly (ratios 0 / 4096 cells, error counts 0 / 64 candidates).

Unlike S7, no machine difference appeared. The computations in this stage are the following three, and in none of them does SIMD multiply-add fusion reach the result.

| Computation | Type | Effect of fusion |
| --- | --- | --- |
| 256-level histogram | Integer counting | None |
| Sum and sum of squares of the inner region | `int` addition (exact, since the pixels are 8-bit) | None |
| Otsu recurrence | `double` | OpenCV computes it in scalar as well |

`cv::meanStdDev` uses the population variance (dividing by N) and computes the mean as `S * (1/N)`, not `S / N`. Since changing the order changes the last digit, we write it in the same order. The Otsu threshold update uses a strict greater-than, so on ties the smaller gray level is kept. The thresholding is "pixel > threshold", so equal pixels fall on the 0 side.

`cell_decode.cu` is also compiled with `-fmad=false`. Allowing fusion shifts the variance at the `minOtsuStdDev` boundary and changes whether the low-variance branch (the route that fills all cells with 1.0 or 0.0) is taken.

The threshold of 0.49 for treating a cell ratio as a bit, which the CPU baseline used implicitly, has been made explicit as `DetectorConfig::valid_bit_threshold_`. Both border verification and Dictionary matching use the same value.

The upper bound on the number of outer-ring errors is not the number of outer-ring cells but `markerSize` squared multiplied by `maxErroneousBitsInBorderRate`. With the defaults (36h12, border 1, rate 0.35) it is 12: 12 passes and 13 fails.

### Second half of S8: Dictionary matching

Matching is performed on the GPU. For all 250 IDs across 4 rotations (1000 cases), it matches both the CPU baseline and OpenCV.

OpenCV's `Dictionary::identify` has two rules that cannot be reproduced if a bit string is assumed.

#### Cell ratios cannot be collapsed into a single bit string

OpenCV represents a candidate with two masks.

| Mask | Condition to be set | Meaning |
| --- | --- | --- |
| `not0` | ratio > `validBitIdThreshold` | Not black |
| `not1` | ratio < `1 - validBitIdThreshold` | Not white |

With the default threshold of 0.49, the upper bound is 0.51. A ratio of 0.49 or below is determined black and 0.51 or above white, but **a ratio between the two sets both masks and is counted as an error whether the expected bit is 0 or 1**. Collapsing this into a single bit string erases that third state and changes the distance for candidates on the boundary.

The number of errors is obtained by the following expression. Since it is branch-free, it can be passed directly to `__popcll`.

```
error = not0 ^ ((not0 ^ not1) & codeword)
```

When `codeword` is 0, the right-hand side becomes `not0`; when it is 1, it becomes `not1`. OpenCV computes the same expression with the combination of `hal::and8u` and `hal::normHamming`.

The upper bound is computed in float. OpenCV's `1 - validBitIdThreshold` is a float operation, and computing it in double precision moves the threshold by 1 ULP.

#### The ID taken is the first one to satisfy the condition, not the one with the minimum distance

OpenCV scans IDs in ascending order and stops as soon as the allowed distance `int(maxCorrectionBits * errorCorrectionRate)` is satisfied. It does not search for the minimum distance over all IDs.

For DICT_ARUCO_MIP_36h12, the minimum distance between entries is 12 and the allowed distance is 3, so at most one ID satisfies the condition and the two agree. As rules, however, they are different, and for a dictionary with a smaller distance the result changes.

The existing `match_dictionary` is an API that returns the minimum distance. To avoid collapsing this distinction, a separate `identify_marker` was provided for the detection route. The GPU side takes the `atomicMin` of the IDs that satisfy the condition, obtaining the same result as the ascending-order suppression.

Rotations are examined in order over the four, updating on a strict less-than. On ties the smaller rotation is kept, and the scan stops at distance 0. This too is the same as OpenCV.

### S9 identification suppression and compaction

This stage runs on the GPU. **OpenCV has no stage called "duplicate removal."** Reading all of `detectMarkers`, we have confirmed that the only things running after `identifyCandidates` are corner refinement, the Multi dictionary-specific rejected cleanup, the inverse scaling by fxfy, and the duplication of the output. There is no ID-based duplicate removal anywhere, and if two markers with the same ID are present at separate positions, both are emitted.

Duplicates disappear as a result of identification suppression via the containment tree. Since both the outer and inner boundaries of a black frame become candidates, once a marker is found on the inner one, the candidate surrounding it can be left unidentified.

#### Containment tree

`selectedCandidates` is sorted in descending order of perimeter. OpenCV scans in descending order of index and, for each candidate, examines indices smaller than its own in descending order, making the first "candidate that surrounds it" that is found its parent.

```
parent[i] = max { j : j < i and i is inside j }   (-1 if none)
depth[j]  = max(depth[j], depth[i] + 1)
```

**`parent[i]` is the largest j, not the smallest j.** Since a smaller index means a larger perimeter, this means "the innermost among those that surround it." The propagation of depths must be done in descending order of index. Doing it in parallel would read depths that are not yet finalized.

The containment test is exactly `cv::pointPolygonTest` called with `measureDist = false`. If all four corners are inside or on the boundary, it is regarded as inside.

- **The boundary (return value 0) counts as inside.** There are three early returns — when a vertex coincides, when the point lies on a horizontal edge, and when it lies on a slanted edge — and none of them can be dropped.
- **A sign agreement test that assumes convexity cannot substitute for it.** The quadrilateral built by extreme point search can be concave, since the interior angle at c0 may be reflex. In measurement, 283 of 400 sets were concave.
- **The cross product is computed in 64-bit integers.** OpenCV uses double precision. If the coordinates are integers the signs match exactly, but in single precision the product of coordinate differences is rounded and the sign changes.

#### Suppression

OpenCV identifies from depth 0 onward and stops once the number of candidates reached covers the whole set.

```
depth = 0; counter = 0;
while (counter < ncandidates) {
    identify all v in depths[depth] and set was[v] = true
    for (v in depths[depth]) {
        if (identified) { walk up the ancestors, and for unvisited ones set was and counter++ }
        counter++;                     // unconditional, regardless of identification success
    }
    depth++;
}
```

There are two points to preserve.

1. **`was[]` is not consulted as a condition for skipping identification.** Even a candidate marked as an ancestor is identified normally once its depth is reached. The granularity of suppression is the depth, not the candidate.
2. **The reached count is double-counted.** A candidate counted as an ancestor is counted again when its own depth comes around. Since this works in the direction of stopping earlier, rewriting it into "strict per-candidate suppression" changes the result.

Because no depth is skipped, this scan **can be reduced to a single suppression depth**.

> Identified candidates = { v : depth[v] < stop_depth }

On the GPU, `stop_depth` is obtained by a sequential scan in a single block, and the rest becomes an ordinary compaction. Unlike the CPU route, "not performing identification" is not possible, because S7 and S8 on the GPU run unconditionally over all candidates. However, the identification result is a function of only the candidate's corners and the image and does not depend on the scan order, so if the results for all candidates are held in advance the scan can be reproduced afterward.

#### Output

The conditions for acceptance are that "the scan has reached it" and that "an ID is attached." The corners are emitted after undoing the rotation obtained from Dictionary matching.

```
new[i] = old[(i + 4 - rotation) % 4]
```

The undoing comes **before** subpixel correction (S10). OpenCV also undoes it immediately after identification and then calls the correction. Swapping the order would apply the correction to a different corner.

The output ordering preserves the candidate ordering as is. Since the merged candidates are in descending order of perimeter, the detections are also in descending order of perimeter. The final sorting is the responsibility of S11 and is not done in S9. Because the sorting on the CPU route runs after subpixel correction, sorting in S9 could change the order after the correction has moved the corners.

### S10 subpixel correction of the corners and restoration to original size

This stage runs on the GPU. It reproduces OpenCV's `findCornerInPyrImage` and `cv::cornerSubPix`.

#### Procedure for climbing the levels

```
corners *= scale_init                     // scale_init = width of the starting level / width of the segmentation
for (level = starting level - 1; level >= 0; --level) {
    corners *= 2
    cornerSubPix(pyramid[level], corners, window)
}
```

**The pyramid is built from the original size, before downscaling.** Since `buildPyramid` is called on the image before it is downscaled to the segmentation image, level 0 is the original size. By the time the climb finishes the corners are already in original-size coordinates, and the reciprocal of `fxfy` must not be applied separately. The inverse-scaling branch is not reached whether or not ArUco3 is in use.

**The window radius is determined by the size of the level, not by configuration.** It is `max(level width, level height) > 1080 ? 5 : 3`. `cornerRefinementWinSize` and `relativeCornerRefinmentWinSize` are used only on the route where ArUco3 is disabled.

#### One iteration

The weights are `exp(-((i-r)/r)^2) * exp(-((j-r)/r)^2)` evaluated in single precision. Since `zeroZone` is called with `Size(-1, -1)`, the branch that zeroes the center is not entered.

The patch is cut out with `getRectSubPix`. **This is not a straightforward bilinear interpolation.** It traverses a row from left to right, carrying the previous term multiplied by `(1-a)/a`.

```
prev = (1-a) * (b1*src[0] + b2*src[step])
for j:
    t = a12*src[j+1] + a22*src[j+1+step]
    dst[j] = prev + t
    prev = (float)(t * s)          // s = (1-a)/a in double precision
```

Mathematically it is the same as bilinear, but the rounding accumulates, so replacing it with the bilinear expression changes the values. When the window extends outside the image, a different route (`adjustRect`) is taken, and the area outside the rectangle is filled by interpolating the edge values in the vertical direction only.

The normal equations are in double precision and are accumulated one element at a time in row-major order. **Folding them in parallel changes the rounding and changes even the iteration count and whether the suppression is taken.** For that reason each of the five sums is accumulated on a single thread in ascending index order. The parallelism is placed inside the corner instead: **one block of 192 threads handles one corner**. The block gathers the patch and computes the per-element products into shared memory in parallel, five threads then fold the five chains, and thread 0 solves the 2x2 system and decides whether to iterate again. There are 4 corners per detection and at most 1024 detections, so the launch does not start one block per corner slot: a fixed number of blocks strides over the slots, and that number is derived from the SM count of the device.

The error is computed in single precision and then widened to double. `err = (dx*dx) + (dy*dy)` is a float operation, and its result goes into a `double err`.

#### How the iterative solver was verified

Unlike the preceding stages, a 1 ULP difference becomes a discrete difference. The iteration count changes by one, and if the branch that restores the initial position on convergence failure flips, the corners move by the window radius. For that reason the verification was split into two parts.

**First, that there is no transcription error.** An oracle transcribing `cv::cornerSubPix` and `cv::getRectSubPix` verbatim to the host is placed in the tests, and bit agreement is required. The oracle is compiled with `-ffp-contract=off`, giving it the same semantics as `-fmad=false` on the GPU side. Including degenerate inputs, the three machines measured for this matched 128/128.

**Second, the difference from OpenCV.** The difference from the actual `cv::cornerSubPix` varies by machine. On inputs close to real use, the three machines measured for this are within an RMSE of 0.000012 px, but on degenerate inputs where the determinant approaches 0, a difference of up to 6.72 px appears on aarch64. This is because the convergence-failure decision flips and is then doubled while climbing the levels. It is not an implementation error but the result of a rounding difference amplified into a discrete branch.

### Stage breakdown

This diagram shows the ordered-contour demand route. The `FULL` control builds the pyramid before segmentation; the component control omits close-contour retry.

```mermaid
flowchart TD
    S0["S0 input validation and level-zero view"] --> S2["S2 segmentation image generation"]
    S2 --> S3["S3 adaptive thresholding, nScales images"]
    S3 --> S4["S4 component labeling or contour cycles"]
    S4 --> S5["S5 quadrilateral candidate extraction"]
    S5 --> S6["S6 candidate filtering and proximity merging"]
    S6 --> S1["S1 requested pyramid prefix (DEMAND)"]
    S1 --> S7["S7 perspective transform and cell sampling"]
    S7 --> S8["S8 border verification and Dictionary matching"]
    S8 --> R["eligible close-contour retries"]
    R --> S9["S9 containment suppression and compaction"]
    S9 --> S10["S10 subpixel correction of corners and upsampling"]
    S10 --> S11["S11 result output"]
```

All stages consume current-frame data on one ordered stream. Stage numbers describe responsibilities rather than an immutable launch order.

### Parallelization approach per stage

| Stage | Unit of parallelism | Main technique | GPU suitability |
| --- | --- | --- | --- |
| S0 input validation | - | Boundary validation on the host | Out of scope (done on the host) |
| S1 pyramid | Output pixel | Launch a separable downsample kernel per level | High |
| S2 segmentation | Output pixel | Profile-selected public `INTER_LINEAR` arithmetic | High |
| S3 thresholding | Output pixel | Separable box mean or exact tiled multiwindow kernel | High |
| S4 topology | Pixel/edge and root | Component labeling or directed contour-cycle union | Medium |
| S5 candidate extraction | Component or contour | Extreme points, or exact unit paths and closed polygon approximation | Medium |
| S6 filtering/grouping | Candidate | Predicates, quad perimeter ranking, proximity groups and alternatives | Medium |
| S7 warp and sampling | Candidate | Block per candidate or bounded persistent grid, per-cell sampling | High |
| S8 border verification | Candidate | Block per candidate or persistent grid, shared histogram and integer sums | High |
| S8 Dictionary matching | Candidate and codeword | Popcount over packed codewords and a minimum reduction | High |
| S9 suppression and compaction | Candidate | Containment test per candidate, scan sequential in a single block, repacking by scan | Medium |
| S10 corner correction | Corner | One block per corner, striding over the corner slots; the accumulation of the normal equations is kept sequential | Medium |
| S11 output | - | Device buffer or host transfer | - |

S3 evaluates three window sizes by default. Separable processing emits each threshold plane in turn; the fused route emits supported planes together. Candidate extraction then visits windows serially on the same stream and appends to one list. Contour scratch, or the legacy labels/statistics/quads, is reused across windows. Cross-window grouping follows extraction.

### Design options for quadrilateral candidate extraction

Contour topology and closed polygon approximation now provide a device-only alternative to component extrema. Per-contour dependencies remain, but edge aggregation and warp cooperation reduce the sequential work.

| Option | Content | Positioning |
| --- | --- | --- |
| A | Connected components and extreme points | Fresh-build control; historical decision in [ADR-0003](../adr/0003-candidate-extraction-approach.md) |
| B | Device boundary cycles, exact pixel paths and polygon approximation | Implemented `ORDERED_CONTOURS` route in explicit steady presets; [ADR-0004](../adr/0004-device-contours-for-cpu-quality.md) |
| C | CPU contour extraction with GPU stages | Independent compatibility oracle; never an automatic fallback inside the device detector |

#### Procedure of option B

Directed foreground/background interfaces form cycles with 8-connected foreground and 4-connected background semantics. Both outer boundaries and holes produce oriented unit pixel paths. Canonical seeds and closing-point handling preserve the CPU contour convention, including diagonal contacts, one-pixel components and repeated path points. Closed polygon approximation uses the pinned OpenCV arithmetic and cleanup rules.

The explicit steady presets aggregate point counts and signed areas by final cycle root, approximate contours cooperatively within a warp, and materialize long paths using compact eligible-edge jump tables. Stable earliest-index reductions preserve polygon tie-breaking. Serial variants remain available for ablation. The point bound is `2WH+W+H` for segmentation dimensions `W,H`; caller limits can be smaller and must report overflow. Four original edge slots per pixel remain the topology representation. See [workspace lifetimes, proof and tests](../../src/core/contour_candidates.md).

#### Procedure of option A

```mermaid
flowchart TD
    T["thresholded image"] --> L["per-block union-find"]
    L --> M["inter-block merge and label compaction"]
    M --> A1["aggregate bbox, pixel count, and centroid per label with atomics"]
    A1 --> F1["coarse filter by area and bbox ratio"]
    F1 --> P1["pass1: point c0 farthest from the centroid"]
    P1 --> P2["pass2: point c2 farthest from c0"]
    P2 --> P3["pass3: points c1 and c3 farthest on each side of line c0c2"]
    P3 --> V["validity verification as a quadrilateral"]
```

1. The black frame of a marker becomes a square connected component with a hole inside on the thresholded image. Extreme point search targets all pixels of that component, so this route does not enumerate the hole separately. Joined components can therefore hide multiple valid hole candidates.
2. Each pass is implemented with an `atomicMax` over the distance and the pixel index packed into 64 bits, fully parallelized per label.
3. The corner ordering is normalized to a fixed winding, and candidates are filtered by conditions corresponding to `minCornerDistanceRate` and `minDistanceToBorder`.
4. Validity verification checks the fraction of component pixels that fall inside the estimated quadrilateral, and the bbox occupancy of the component relative to the quadrilateral area. The decision corresponding to `polygonalApproxAccuracyRate` in the CPU implementation is mapped onto this.

#### Benefits and limitations of option A

- Benefit: Contour ordering and variable-length contour buffers become unnecessary, and full parallelization per label is possible. It can be recorded as an independent design that does not port the structure of the official ArUco GPLv3 implementation.
- Limitation: Since it is not polygonal approximation itself, the candidate set may not match the CPU baseline results. In particular, the behavior differs for components whose corners are missing due to occlusion, and for cases where multiple markers touch and become a single component.
- Historical conclusion: [ADR-0003](../adr/0003-candidate-extraction-approach.md) chose option A from the small corpus below. The expanded quality failures trigger reconsideration in ADR-0004; that historical candidate agreement is not a general guarantee.

#### Measurements of option A

In the historical extraction-only evaluation, option A was compared against option C over 9 synthetic scenes x 2 settings. Since preprocessing and thresholding are common inputs, they are excluded from the measurement.

| Setting | Detections against ground truth | Candidates emitted only by option C | Candidates emitted only by option A | Maximum corner difference |
| --- | --- | --- | --- | --- |
| ArUco3 enabled (extraction on the downscaled image) | 38/38 | 0 | 0 | 1.414 px |
| ArUco3 disabled (extraction at original size) | 38/38 | 7 to 58 | 0 | 1.414 px |

The reason option C has more candidates with ArUco3 disabled is that option C creates a candidate per contour. The outer and inner boundaries of the black frame and the contours of the inner cells each become candidates. Since option A creates only one set of corners per connected component, no inner boundary appears. In those nine scenes with ArUco3 enabled, downscaling put inner boundaries below the perimeter bound. The expanded corpus contains eligible holes even after downscaling, so that explanation is specific to the historical inputs.

Which is faster depends on the content of the scene. On DGX Spark with ArUco3 enabled, on a plain scene with 4 markers, option A takes 0.543 ms against 0.168 ms for option C; adding noise to the same scene gives 0.530 ms for option A against 1.728 ms for option C. Option A is determined almost entirely by the pixel count, while option C is proportional to the number and length of the contours.

#### Quadrilateral-likeness decision of option A

The decision corresponding to `polygonalApproxAccuracyRate` is mapped onto two ratios. The measured values on synthetic shapes are as follows.

| Shape | Inside ratio | Edge support | Decision |
| --- | --- | --- | --- |
| Square (11 rotations) | 0.987 to 1.000 | 2.52 to 3.03 | Pass |
| Frame (4 rotations) | 0.973 to 1.000 | 2.60 to 3.02 | Pass |
| Small frame (side 32) | 0.875 | 2.61 | Pass |
| Circle | 0.642 | 4.97 | Reject |
| Ellipse | 0.639 | 3.99 | Reject |
| Hexagon | 0.665 | 3.01 | Reject |
| L shape | 0.941 to 0.956 | 1.32 to 1.39 | Reject |
| Cross | 0.850 | 1.71 | Reject |

The inside ratio alone cannot reject the L shape and the cross, and the edge support alone cannot reject the circle and the ellipse. Imposing both separates them. For a triangle, the line c0c2 coincides with one side and no point remains on one side of it, so the corners are not determined and it becomes invalid.

### Variable-length output and overflow

Candidate and marker capacities are explicit `DetectorConfig` limits. Contour workspace sizing also checks the image-derived point bound and arithmetic overflow. The expanded audit uses 16,384 candidates: a full-resolution checkerboard needs 5,460 across three windows and correctly fails the unchanged 4,096 default.

A detector-owned device failure word is rewritten on every invocation in both candidate routes. Bits report invalid contour cycles (1), contour point overflow (2), candidate overflow (4), polygon workspace overflow (8), and invalid candidate counters (16). Any nonzero value invalidates the whole result and publishes zero detections; failure remains sticky across threshold windows. Host download reports the failure without modifying the caller's previous output. Device consumers must inspect the word on the producing stream or after successful completion. Counter comparison alone cannot establish validity.

Marker capacity is a separate contract: with a zero failure word, `accepted_total_ > count_` signals marker truncation. The host returns `kMarkerOverflow` with the bounded partial marker list and its explicit flag; benchmarks reject such partial results. The C ABI exposes a separate borrowed failure-word view, preserving the existing result structure layout, and Python checks it before accepting a result. See [the public binding contract](c-abi-and-python.md).

### Workspace and memory policy

Initialization reserves reusable frame and persistent arenas from configuration and layout bounds. Layout changes can grow storage and recapture the graph; unchanged-layout submissions allocate no frame storage. Buffers share regions only when their lifetimes do not overlap. Ordered contours retain complete image-derived workspace even when few contours survive; a blank image does not imply negligible storage.

The same device algorithm runs on Spark and Orin. GPU-resident input, host upload, and result readback are different timing scopes even though both machines share physical memory between CPU and GPU. Stage device-event timing and separate kernel profiling complement end-to-end wall-clock measurements; no kernel-sum measurement is presented as caller latency.

### Synchronization and streams

The API borrows a caller stream and owns a completion event. Each successful or partially issued submission records completion; a later submission or download waits for it on the destination stream. This also handles different nonblocking streams and recycled caller stream handles. Steady unchanged-layout submissions remain asynchronous on the host, including graph launches.

Reinitialization, layout rebuild and destruction wait for owned completion before replacing storage. Resource replacement invalidates old views and initialized/result state before an allocation can fail. If event recording fails, the detector invalidates itself and drains the submitting stream; a failed drain forces a device-wide recovery attempt before later initialization or destruction. These are exceptional paths, not steady-state synchronization.

Download synchronizes before publishing host output. Upload/readback error paths drain outstanding copies before borrowed host input or local output storage can expire, preserving the original diagnostic. Input uploads and downstream readers submitted on other streams still require caller-established dependencies. See [the lifecycle contract and fault tests](../../src/core/detector.md).

### Handling of machine differences

Both steady presets select ordered contours, fused thresholding, demand pyramid, density-selected fixed-point Otsu (96 occupied bins), persistent candidate grids (512 blocks), edge aggregation, warp approximation/materialization and eligible-edge compaction. Resize descriptors differ only to reproduce the pinned CPU HAL arithmetic. Core-only getters record actual compiled strategies and numerical profiles in every audit/benchmark identity.

All strategy choices and tuning limits have explicit CMake overrides; fresh builds retain their previous settings. A persistent grid bounds launch size and loops over device counts without host readback. The 512-block choice has unfavorable dense-stage cases and is not a universal speed claim. `cuda_block_dim_` retains its narrower historical role below. See [configuration and controls](../../cmake/ArUco3CudaStrategies.md).

## Design decisions

- Fresh-build option A remains available; explicit steady presets select option B. Option C is a separate compatibility oracle, with no host contour fallback in the device route.
- Candidate extraction visits threshold windows serially because its scratch is shared. Fused S3 may produce threshold planes together; S6 groups the appended candidates afterward.
- Dictionary matching is done by popcount over packed codewords with the 4 rotations expanded in advance, so that the rotation search is not a branch.
- Subpixel correction of the corners is implemented on the GPU without permitting delegation to the CPU, because it is directly tied to the accuracy of ArUco3.
- The input is limited to 8-bit grayscale, and color conversion is the responsibility of the adapter.

### Historical 2D block-size evaluation

In the original implementation, `DetectorConfig::cuda_block_dim_` reached five of twelve 2D kernels (three preprocessing and two threshold kernels). Labeling, candidate filtering and extreme-point extraction fixed their blocks at 16. This historical count does not describe the added contour or fused-threshold kernels.

The original decision retained that scope. In measurements sweeping 8 / 16 / 32 on the three machines measured at the time, **16 was optimal on every machine**, and 32 was 3 to 8% slower. Those measurements supported retaining the original scope; they do not establish optimal launch policies for the newly added kernels.

To avoid redoing the same sweep, the measured facts are left here.

In that sweep, the machine-dependent value that materially affected S10 was its launched block count. That is not a setting but is **derived from the SM count of the device**. The details are in [Implementation structure and verification](../implementation-plan.md).

## Open questions

- How to tune the validity verification thresholds of option A on real images. The current inside ratio of 0.80 and edge support of 2.0 are based on measurements on synthetic shapes.
- Whether the combined explicit steady presets meet the final cross-hardware latency and memory acceptance criteria, including dense masks and empty natural photographs.
- Whether to include white marker detection equivalent to `detectInvertedMarker` in the initial scope.
- Whether additional CPU HAL builds need validated resize descriptors, and whether transitive GPU grouping differs on inputs beyond the maintained close-chain corpus.

## See also

- [Architecture](../architecture.md)
- [Public API draft](public-api.md)
- [Implementation plan](../implementation-plan.md)
- [Evaluation plan](../evaluation-plan.md)
- [Code Provenance record](../code-provenance.md)
