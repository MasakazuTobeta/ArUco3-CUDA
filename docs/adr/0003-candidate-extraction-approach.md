# ADR-0003: Adopt plan A (connected components and extreme point search) as the primary approach for quadrilateral candidate extraction

- Status: Accepted
- Date: 2026-08-27

## Purpose

Of the three plans raised in the [detection pipeline design](../design/detector-pipeline.md), decide which one to proceed with for S4 and S5 (quadrilateral candidate extraction). Because the subsequent decode implementation builds on this choice, we settle it now that the candidate extraction paths are in place.

## Scope

This covers everything from the binarized image up to obtaining a set of corner candidates. Preprocessing, thresholding, Dictionary matching, and subpixel refinement of the corners are out of scope.

## Background

### The three plans

| Plan | Content | State at the time of the decision |
| --- | --- | --- |
| A | Find the corners directly with connected component labeling and extreme point search | Implemented |
| B | Perform border following on the GPU and apply polygonal approximation | Not implemented |
| C | Return the binarized image to the host and find the corners with CPU contour tracing | Implemented |

### Measurements

`CandidateComparisonTest.plan_a_versus_plan_c` applies both routes to the same binarized image and compares them. We measured 9 synthetic scenes (marker count, in-plane rotation, projective distortion, blur, noise, illumination gradient, resolutions 1280x720 and 1920x1080) under two configurations. Times are the median of 21 runs after 5 warm-up runs, with the core pinned via `taskset -c 0`.

#### Accuracy

| Configuration | Detections against ground truth | Maximum corner difference (vs. ground truth) | Candidates produced only by plan C | Candidates produced only by plan A |
| --- | --- | --- | --- | --- |
| ArUco3 enabled (extraction on the downscaled image) | 38/38 | 1.556 px | 0 | 0 |
| ArUco3 disabled (extraction at full size) | 38/38 | 1.594 px | 7 to 58 | 0 |

With ArUco3 enabled, the candidate sets match exactly. The corner differences between shared candidates are 0, except for 1.414 px in the scene with blur added.

With ArUco3 disabled, plan C produces more candidates than plan A. Because plan C creates a candidate per contour, the outer and inner boundaries of the black border and the contours of the interior cells each become candidates. Plan A creates only one set of corners per connected component, so the inner boundaries do not appear. With ArUco3 enabled, downscaling puts the perimeter of the inner boundary below the lower bound (`4 * minSideLengthCanonicalImg` = 128), so it is dropped on the plan C side as well and no difference appears.

This difference is not a marker miss. Detections against ground truth are 38/38 under both configurations. The extra candidates from plan C are ones that the border verification in Dictionary matching drops. However, OpenCV has a route (`closeContours`) that tries another candidate from the same group when identification of the representative candidate fails, and plan A has no such alternative. We will measure the impact once identification is included in the run.

#### Speed

These are times for candidate extraction only (from labeling through consolidation). Preprocessing and thresholding are common to both routes and are therefore excluded.

| Configuration | Scene | DGX Spark plan A | DGX Spark plan C | Jetson Orin plan A | Jetson Orin plan C |
| --- | --- | --- | --- | --- | --- |
| ArUco3 enabled | plain scene (4 markers) | 0.543 ms | 0.168 ms | 0.746 ms | 0.250 ms |
| ArUco3 enabled | 9 markers | 0.665 ms | 0.323 ms | 0.815 ms | 0.467 ms |
| ArUco3 enabled | with noise | 0.530 ms | 1.728 ms | 0.803 ms | 2.299 ms |
| ArUco3 enabled | 1920x1080 | 0.518 ms | 0.172 ms | 0.756 ms | 0.263 ms |
| ArUco3 disabled | plain scene (4 markers) | 1.278 ms | 1.161 ms | 2.620 ms | 1.710 ms |
| ArUco3 disabled | 9 markers | 1.667 ms | 2.301 ms | 3.194 ms | 3.340 ms |
| ArUco3 disabled | with noise | 1.420 ms | 21.392 ms | 4.447 ms | 29.745 ms |
| ArUco3 disabled | 1920x1080 | 2.074 ms | 1.886 ms | 5.274 ms | 2.826 ms |

Plan A's time barely depends on the content of the scene and is determined almost entirely by the pixel count. Plan C is proportional to the number and length of contours, and stretches more than tenfold when noise is added. There is a crossover: plan C is faster in plain scenes, and plan A is faster in scenes with more contours. In the scene with ArUco3 enabled and noise, plan A is 3.3 times faster on DGX Spark and 2.9 times faster on Jetson Orin. With ArUco3 disabled and noise, the figures are 15 times and 6.7 times.

The reason plan A stays between 0.52 ms and 0.67 ms regardless of resolution when ArUco3 is enabled is that the segmentation image is small at 427x240, so the fixed cost of kernel launches dominates. Per frame there are 34 launches: 3 windows x 9 kernels, plus 7 kernels for consolidation.

### Why plan B was not evaluated

Plan A meets the accuracy requirements, and plan B has a higher implementation cost than plan A because it puts border following — a highly sequential process — on the GPU. We will reconsider it if plan A stops meeting the requirements.

## Decision

**Adopt plan A as the primary approach, and keep plan C as a fallback and as the baseline for differential verification.**

## Rationale

- The candidate set matches plan C exactly when ArUco3 is enabled, and the corner difference is within 1.414 px. Evaluating the ArUco3 detection strategy is the main purpose of this project, and having no difference under that configuration is sufficient grounds for the decision.
- Plan A's time is nearly constant regardless of the scene. In real-time processing the worst case determines the requirements, so it is easier to work with than plan C, which varies by more than tenfold with the content.
- Plan C needs to return the binarized image to the host. Given the goal of staying device-resident, making plan C the primary approach would leave a transfer in the route permanently.
- Plan A needs neither contour ordering nor variable-length buffers, and can be fully parallelized per label. Decode is also per label, so the structures line up.
- Plan A is an independent design that does not port the structure of the official ArUco GPLv3 implementation, which is consistent with the requirements of the [intellectual property and licensing policy](../ip-and-licensing.md).

## Consequences

### Benefits

- Everything through candidate extraction completes device-resident. No host transfer of the binarized image is needed.
- The worst-case execution time becomes easier to predict.
- Keeping plan C lets us continuously verify that changes to the GPU route have not changed the candidate set.

### Drawbacks

- For small segmentation images, plan C is faster. The conditions this project mainly assumes — ArUco3 enabled and a plain scene — fall exactly there. Reducing the fixed cost of kernel launches becomes the next task.
- Because only one set of corners is created per connected component, there can be no alternative candidate equivalent to OpenCV's `closeContours`.
- Two markers that overlap into a single component are either accepted as one quadrilateral or both dropped. On the CPU route the contour becomes an octagon and is dropped by polygonal approximation. Both miss the two markers, but the behavior differs.
- The quadrilateral test is not `polygonalApproxAccuracyRate` itself, but a mapping onto two ratios: the inner ratio and edge support. They separate cleanly on synthetic shapes, but the margin on real imagery is unmeasured.

### Conditions for reversal

If any of the following is confirmed at a later stage, we will reconsider implementing plan B or switching to plan C.

- On a real-imagery corpus, plan A's detection rate is 1% or more below plan C's.
- The absence of `closeContours` makes the identification success rate significantly lower than plan C's.
- Even after optimization, it is slower than plan C under the assumed conditions.

## Open questions

- How far the fixed cost of kernel launches can be reduced. Options include merging the 3 windows into a single kernel, or using CUDA Graph. We will decide by measurement.
- How to adjust the lower bounds of 0.80 for the inner ratio and 2.0 for edge support on real imagery. The current values are based on measurements of synthetic shapes (minimum 0.875 and 2.52 for shapes that should pass, maximum 0.665 and 1.71 for shapes that should be dropped).
- Whether to give plan A an equivalent of `closeContours`. That would require a way to create multiple corner candidates from one connected component.

## See also

- [ADR-0001: Develop the CUDA implementation first in an independent repository](0001-independent-implementation.md)
- [ADR-0002: Fix the build toolchain and target environment baseline](0002-toolchain-and-target-baseline.md)
- [Detection pipeline design](../design/detector-pipeline.md)
- [Implementation plan](../implementation-plan.md)
- [Evaluation plan](../evaluation-plan.md)
- [Benchmark report](../benchmark-report.md)
