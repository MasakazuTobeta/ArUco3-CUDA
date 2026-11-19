# AprilTag CUDA detector

## Purpose

Detect square AprilTags from a pitched device grayscale image and return IDs,
Hamming distances and oriented corners to the host. The detector retains its
workspace across independent frames.

## Scope

The detector accepts the four square OpenCV AprilTag dictionaries. Their
provenance is in the [dictionary record](../../docs/code-provenance.md#apriltag-family-codewords).
The installed [C++ API](../../include/aruco3cuda/apriltag/gpu/detector.hpp) defines
the configuration and stream contract. NVIDIA cuAprilTags is a separately
installed comparison library. [Benchmark procedure](../benchmark/README.md)

## Current state

For ordinary video frames the GPU initially samples every third pixel along both axes and thresholds from local
tile minima and maxima. Regions with insufficient contrast remain unclassified.
It labels black and white regions separately, groups boundary samples by the
two regions they separate, and uses eight directional extrema to seed each
quadrilateral. These stages follow the boundary-segmentation approach described
in [AprilTag 2](https://april.eecs.umich.edu/pdfs/wang2016iros.pdf) and the
[AprilTag 3 paper](https://april.eecs.umich.edu/pdfs/krogius2019iros.pdf).
The CUDA implementation is original.

Each active quadrilateral gets a warp. Lanes sample signed gradients along
the four full-resolution edges, fit lines and intersect adjacent lines. A
projective transform maps the cells into the original image. Separate affine
brightness models for the black border and outer white ring allow illumination
to vary across the tag. Samples outside the image are omitted from fitting the
brightness planes. An edge with too little gradient evidence, a degenerate fit
or an invalid payload sample rejects the quad.

After a successful decode, the marker grid provides a scale for a second edge
fit. Each normal sample searches for a signed luminance transition within the
configured radius in cells. The warp selects a line supported across the edge,
refits using gradient strength and residuals, and intersects the four lines.
It keeps the updated corners only when the same oriented codeword still decodes
with equal or lower Hamming distance. This handles coarse corners displaced by
clutter along one side.

Adaptive mode uses one full-resolution pass for small input images (up to
36,864 pixels by default). On larger images it checks the sampling density of
plausible coarse codewords.
When the shortest projected side has fewer than two coarse samples per cell
and the word is within one bit of the configured decoding limit, the GPU
segments the same frame at full resolution and appends the resulting quads.
The extra bit triggers a search only. Both passes apply the configured limit
to accepted detections. A small tag can still be missed when the coarse image
has no plausible candidate. Fixed full-resolution sampling is available
for scenes where that trade-off matters. A decoded quad must also span at least one source pixel per cell along the
minimum projected altitude. Adaptive mode reserves full-resolution
scratch when the detector is constructed and reuses it for both passes.

The GPU searches all four codeword rotations. With default settings 36h11
accepts up to two bit errors and the other families require an exact word.
After stream completion, the host reads decoded quads directly from mapped
page-locked memory. Same-ID quads are merged only if corresponding corner positions
agree relative to the marker's shortest side. The ordering prefers fewer bit
errors, then more boundary samples, with coordinates as a deterministic tie
breaker. Spatially distinct tags with the same ID are retained.

Configure with `-DARUCO3CUDA_BUILD_APRILTAG_GPU=ON` and build
`aruco3cuda_apriltag_gpu_detect`. For example, from the repository root:

```bash
cmake --preset native -DARUCO3CUDA_BUILD_REFERENCE=OFF -DARUCO3CUDA_BUILD_TESTS=OFF \
  -DARUCO3CUDA_BUILD_APRILTAG_GPU=ON
cmake --build --preset native --target aruco3cuda_apriltag_gpu_detect
build/native/apriltag/gpu/aruco3cuda_apriltag_gpu_detect DICT_APRILTAG_36h11 frame.pgm
```

The CLI takes binary 8-bit PGM images. Within a sequence of matching dimensions it
reuses the detector and device frame. It accepts `--decimate N` before the first
image. The default `0` uses adaptive sampling. Values `1` through `4` select
a single fixed spacing. The API's `fine_samples_per_cell_` changes the
adaptive trigger. `small_frame_pixels_` controls the full-resolution input
area limit and `min_pixels_per_cell_` controls the minimum projected cell
resolution. Capacity exhaustion in either pass rejects the whole call.
`cluster_slots_` can specify a power of two. Its default scales the table with
segmentation area. Each boundary lookup tries at most `max_cluster_probes_`
slots (128 by default, configurable up to 4096). An unusually long collision
chain rejects that frame even if some slots are free. The bound keeps dense
textures from stalling the GPU after saturation. `max_tags_` limits raw
decodes before duplicate removal.

### Workspace and failure checks

On DGX Spark (GB10), Nsight Systems traced three consecutive 3840×2160
grayscale noise frames through the same 36h11 detector. Its eight device
workspace allocations and single mapped output allocation all preceded the
first detection. Replaying the frames allocated no additional device memory.

| Segmentation | Device workspace | Mapped host output | Caller’s pitched image |
|---|---:|---:|---:|
| Adaptive (default) | 251,865,968 B | 45,056 B | 8,847,360 B |
| Fixed spacing 3 | 30,448,496 B | 45,056 B | 8,847,360 B |

The detector returned no tags for the noise image and ID 17 on a pitched
3841×2160 positive image. A 3840×2160 one-pixel checkerboard exhausted the
default hash probe budget and raised the documented overflow exception.
The synthetic test also retries a saturated detector with a valid marker.

For memory measurements, run the CLI under
`nsys profile --trace=cuda --cuda-memory-usage=true` with the same PGM
three times and compare the CUDA allocation events. On GB10, running
`compute-sanitizer --tool TOOL --error-exitcode 17 aruco3cuda_apriltag_gpu_test`
separately for `memcheck`, `racecheck`, `initcheck` and `synccheck` reported
zero errors and zero race hazards.

## Goals

Compare against NVIDIA on the same prepared pixels, including error correction,
repeated IDs, empty backgrounds and images with multiple tags. Record resident
input and host-input latency separately. A change to segmentation or geometry
requires repeating the quality comparison as well as timing.

## Open questions

Severe projective distortion, clipping or low-contrast edges can leave too few
samples for a quad. The published validation covers the inputs and hardware
identified in the evaluation report. Deployment recall depends on the camera,
tag size and illumination.
