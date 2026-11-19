# Independent CPU quality scenes

## Purpose

Exercise frame-independent detection optimizations on changing content and
adversarial geometry, while keeping an independent OpenCV CPU compatibility
oracle and explicit synthetic marker geometry. Reusing allocations must never
mean reusing prior-frame pixels or detections.

## Scope

`quality_scenes.{hpp,cpp}` provides an in-memory renderer and a 96-frame preset.
`aruco3cuda_cpu_quality_corpus` writes those frames as PNGs with a JSON manifest,
so the same inputs can be used by the benchmark. The default CLI output is PNG using `DICT_ARUCO_MIP_36h12`. For an AprilTag corpus,
pass the width, height and family, for example
`aruco3cuda_cpu_quality_corpus build/corpus-36h11 640 480 DICT_APRILTAG_36h11`.
This writes grayscale PGM frames and the same manifest schema, so the NVIDIA, GPU
and CPU adapters can consume identical pixels. Each directory must be empty before generation.

All generated inputs are synthetic. They do not establish performance or
accuracy on camera photographs. Existing real-image fixtures belong to a
separate compatibility evaluation when independent annotations are absent.

## Current state

The preset covers these families:

| Family | Frames | Conditions |
| --- | --- | --- |
| Clean pose | 12 | Different IDs, quarter and oblique rotations, fractional placement |
| Projective | 12 | Independently perturbed corners, mild blur/noise, changing backgrounds |
| Blur and noise | 16 | Four blur levels crossed with four noise levels |
| Lighting | 12 | Low contrast and horizontal illumination gradients |
| Size bound | 12 | Sides immediately above and below the nominal ArUco3 size bound |
| Border and occlusion | 12 | Negative/near-border positions and partial occluders |
| Multiple markers | 12 | Two, four or nine markers, including repeated IDs |
| Negative | 8 | Blank, noisy, linear and checkerboard backgrounds without markers |

At the default 641x479 size these specs contain 136 rendered markers. The
manifest records every ID, oriented corner, degradation parameter, per-frame
seed, image hash and OpenCV version. Geometry uses the outer pixel-boundary
convention: canonical corners are `(-0.5,-0.5)` through `(255.5,255.5)` before
the projective transform. Quarter/oblique rotation changes this transform,
rather than rotating the decoded output afterward.

The renderer uses public OpenCV Dictionary rendering and image transforms,
followed by deterministic noise and lighting. `occlusion_fraction_` controls a
vertical occluder's width as a fraction of the projected bounding box; it is
not an independently measured occluded-marker area. All parameters describe
generation conditions, not a promise that every placed marker is detectable.

The occluder bounding box passes the existing corner array directly to public
`boundingRect`. Constructing a temporary vector triggered eight
`-Wnull-dereference` diagnostics under GCC 9.4 at `-O3` on Jetson Orin; the direct
array compiles with the same warnings treated as errors. Comparing both Release
implementations on that host produced identical pixels for all 288 frames at
641x479, 1280x720 and 320x321. Only the diagnostic baseline comparison downgraded
that specific warning; production builds retain `-Werror`. The local comparison
record is `build/gpu-study/quality-renderer-orin.log`.

The maintained CPU tests compare IDs, oriented float corners, and rejected
counts exactly against the pinned OpenCV detector. Clean positive cases must
actually decode; the sequence also requires a substantial positive-frame
count so equality on empty outputs cannot satisfy the gate. Input pixels are
checked after detection to expose accidental modification.

The sequence shuffles the frames, inserts blank scene cuts, and revisits
earlier content. Most frames overwrite one non-contiguous full-image header at
the same address; periodic ROI views exercise explicit fallbacks. Each detector
persists across the sequence, while the reference output is computed afresh
from the current image. There is no tracking or use of a previous result.

Additional gates cover decode cell sizes/margins across dictionary families,
odd and large images, refinement iteration/epsilon settings, complete candidate
group trees, contour paths, and sparse-pyramid pixel/warp/refinement behavior.

### Reproduction

The standalone CPU build includes the generator:

```bash
cmake -S cpu/standalone -B build/cpu-quality -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpu-quality --target aruco3cuda_cpu_quality_corpus aruco3cuda_cpu_tests
build/cpu-quality/cpu_quality/aruco3cuda_cpu_quality_corpus data/cpu-quality
```

The output directory must be empty. Optional positional width and height
arguments choose another size. Generated images stay outside Git; retain
their manifest and hashes with any evaluation results.

## Provenance

This renderer is independently written against the public OpenCV 4.14.0 API,
commit `0654a42e19215ef25b1d367d822f3c630447e7c7`. Marker data comes only from
its predefined Dictionary API. The pixel-boundary convention follows this
repository's [existing generator](../corpusgen/corpus_generator.md); its source
is not copied into this module. No official ArUco GPLv3 implementation or assets
are used.

## Goals

- Keep accuracy gates independent of optimized contour, grouping, sampling, and
  pyramid implementation details.
- Establish exact OpenCV compatibility before accepting performance measurements.
- Preserve generation conditions, known geometry, positive cases and negative
  controls in reproducible input manifests.

## Open questions

- The generated inputs are not a replacement for annotated camera images.
- Exact agreement establishes equality with this OpenCV baseline on these
  inputs, not a universal proof for every image.
- Synthetic rendering and floating-point transforms can differ across machines;
  cross-machine comparisons must use the stored image hashes.
