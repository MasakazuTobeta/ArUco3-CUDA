# Experimental CPU detector

## Purpose

Evaluate whether the GPU implementation's representation and work scheduling insights can
improve host detection beyond the pinned OpenCV CPU baseline. This is an experimental route,
not an installed replacement for the CUDA public API.

## Scope

`CpuDetector` accepts host `CV_8UC1` matrices, including pitched images and ROIs, and returns
host IDs and corners. It performs no CUDA operations and its library has no CUDA dependency.
Pose estimation, temporal tracking and altered detection thresholds are outside this module.
OpenCV's global thread setting is respected, not changed by the detector.

## Current state

The compatibility control uses the same contour extraction, candidate grouping,
containment traversal, representative retry and refinement rules as `hybrid/`. Host
`buildPyramid`, `resize` and `adaptiveThreshold` replace the CUDA preprocessing route.
The reference version is OpenCV 4.14.0, commit
`0654a42e19215ef25b1d367d822f3c630447e7c7`.

Independent `CpuOptions` switches expose these experiments:

| Switch | Implementation | Compatibility requirement |
| --- | --- | --- |
| `reuse_pyramid_` | Retain buffers and generate pyramid levels only when decoding or refinement needs their pixels. Compute all level dimensions before decoding. | Every generated level uses the same `pyrDown` operation, size and default border as `buildPyramid`. No previous-frame pixels are reused. |
| `exact_dictionary_lookup_` | Precompute the complete first-acceptable-ID/rotation result for each known exact dictionary word, then binary-search definite input words. | Overlapping or missing cell mask bits and unknown words use the ordinary scan; initialization uses the configured correction limit without a minimum-distance assumption. |
| `packed_dictionary_` | Cache the four rotations per ID as 64-bit words. Compute the error with two masks and a CPU population count. | Preserve ambiguous cells, the first acceptable ID, rotation tie breaking and early exact-match termination. |
| `fused_cells_` | Reuse canonical-image and ratio buffers, use `Scalar` mean/deviation outputs, and count sampled cell pixels without per-cell `Mat` headers. | Preserve warp interpolation, inner crop, Otsu, cell margins and floating-point ratio division. |
| `multiwindow_threshold_` | Use the separately tested shared-integral threshold workspace. | See [threshold design](cpu_threshold.md). |
| `dryrun_otsu_` | With `fused_cells_`, compute only the Otsu threshold and compare the original sampled pixels directly. | OpenCV `THRESH_DRYRUN` must return exactly the same threshold. No canonical-image threshold write pass is performed. |
| `component_candidates_` | Estimate one quad per eight-connected foreground region with centroid/opposite/line extrema, inlier ratio and edge support. | Explicitly changes candidate generation. It has no separate inner-boundary candidates or equivalent alternate contour retry; evaluate ground truth separately. |
| `component_prefilter_` | Label eight-connected components and remove only those with `area * 8 < min_perimeter` before contour tracing. | A contour cannot exceed eight directed neighbor states per foreground pixel; disconnected components have independent boundaries. |
| `simple_contours_` | Extract compact contours, recover their exact unit-chain length, discard those outside the perimeter limits and expand survivors before polygon approximation. | Preserve start point, traversal order, all unit-chain pixels and the original approximation epsilon. |
| `parallel_windows_` | With multiple OpenCV threads, threshold and extract contours per window in parallel, then merge in serial window order. | CCL/prefilter experiments use their serial path because their scratch is shared. One-thread mode avoids parallel setup. |
| `parallel_decode_` | Identify candidates within each containment depth and refine accepted markers in parallel, using reusable per-thread sampling buffers. | Prepare shared full levels before dispatch; sparse workers generate only private current-frame tiles. Join before ancestor suppression and preserve result order with indexed writes. |
| `spatial_grouping_` | Reject impossible candidate pairs using conservative spatial bounds before the original grouping/containment metric. | Preserve candidate order, tie behavior and every pair that can affect the original rules. |
| `filtered_contours_` | Trace and size-screen contours in reusable per-window workspaces before allocating retained paths. | Preserve exact `RETR_LIST` path order and all quad filtering decisions; see [contour design](cpu_contours.md). |
| `cached_refinement_` | Reuse Gaussian masks and patch storage while refining existing full pyramid levels. | Execute the same monitored arithmetic as the sparse helper without generating regional pyramid pixels. |
| `sparse_pyramid_` | Generate only demanded current-frame pyramid tiles, with monitored sampling and refinement footprints. | Keep global phase and coordinates, invalidate tile validity every frame, and preserve exact numerical behavior; see [sparse design](cpu_sparse_pyramid.md). |
| `profile_stages_` | Record host wall-clock boundaries. | Disabled for latency measurements; pyramid generation performed lazily is attributed to decode/refinement rather than preprocessing. |

For `DICT_APRILTAG_36h11`, the following opt-in settings provide a host
contour search with full-resolution tag sampling. They can be used together.

| Setting | Effect |
| --- | --- |
| `decode_at_source_resolution_` | Search at the configured ArUco3 scale, then decode from the input pixels. When possible, skip identical leading and trailing rows of the segmentation image, retaining the threshold window's halo. |
| `candidate_box_stride_px_` | Zero uses the configured resize. Values 2–8 sample a centered 2×2 patch on a fixed grid. Both row averages and the final average round up at a half integer. A sparse grid can miss small or oblique tags. |
| `decoded_grid_radius_px_` | Zero disables this refinement. Values 2–32 bound the local displacement search in pixels. Fit the outer edges first, then fit a projective correction from visible black-to-white transitions of the oriented payload. Accept updated corners when the original image decodes to the same ID and rotation. This requires one border cell. |
| `search_frame_edges_` | Keep the full-frame results and retry bounded regions around dark runs on the physical frame perimeter. Only the candidate image gets a bright exterior. Accept an additional near-edge observation after checking its cells on the unpadded source. Requires source-resolution decoding, grid refinement and zero minimum border distance. `frame_edge_max_pixels_` limits the total input area of these regions (720,000 pixels by default). |

The projective fit uses the tag's printed grid, signed luminance differences and a
weighted least-squares homography update. The wide outer-edge initializer is based
on this project's [CUDA edge fit](../apriltag/gpu/README.md). The codewords and
canonical orientation come from the pinned OpenCV dictionary. The [standalone AprilTag 3 CPU library](../apriltag/cpu/README.md)
has its own candidate search and build.

The retained defaults enable pyramid reuse, packed and exact-code dictionary matching,
fused cells, shared thresholding, filtered contour tracing, sparse pyramids, cached refinement,
per-window parallelism and per-depth decode/refinement parallelism. Compact contours,
component prefiltering, component candidates, Otsu dry run and spatial grouping remain opt-in.
On little-endian Arm64, filtered contour tracing finds the next run with bounded
64-bit loads. An isolated foreground pixel gets the usual visited mark and skips
tracing when its one-point contour cannot pass the configured minimum perimeter.
The experimental switches preserve the earlier implementation for paired comparison. All switches may be disabled to obtain
the compatibility control. Current results and retained/rejected experiments are recorded
in the [steady-state evaluation report](../docs/cpu-steady-state-report.md).
The [earlier evaluation](../docs/cpu-benchmark-report.md) remains a historical record.

The exact dictionary table depends only on the selected Dictionary and correction settings.
It is rebuilt on initialization and contains no frame-dependent detections. Its initialization
cost is quadratic in dictionary size; the logical table contains at most four entries
per ID (a codeword, ID and rotation). Vector capacity may retain the previous high-water
mark after a smaller dictionary is selected or lookup is disabled. The lookup guard
requires `not_black XOR not_white` to equal the complete valid-bit
mask, which rejects both overlapping ambiguous bits and uncovered threshold-gap bits. A hit
returns the result of the full baseline scan, including any earlier acceptable ID and the
first best rotation. Enabling lookup uses the packed representation even when its separate
switch is off.

When stage profiling is enabled, `trace_ms_` covers tracing, approximation and orientation
preparation; `group_ms_` covers grouping and containment. `candidates_ms_` retains the total,
including the optional geometry-based sparse/full cost decision between stages.

`min_parallel_candidates_` (default two, required positive) controls the minimum candidate
batch or accepted-marker count used by decode/refinement parallelism. Singleton batches always
use the serial workspace. Each worker's scaled corner vector, canonical image and cell ratios
live in detector-owned OpenCV TLS storage and are reused across frames. A containment depth
is a synchronization boundary. Workers do not change the shared full pyramid or
suppression state; sparse workers mutate only their private current-frame caches,
and shared suppression decisions occur after the join.

`threshold_parallel_min_pixels_` (default 32768) controls the minimum stripe size for
parallel shared-threshold row groups. Zero permits parallelism for every stripe; a large
value keeps that stage serial. It must be nonnegative.

`threshold_max_prefix_bytes_` defaults to 2 MiB and bounds the requested prefix-sum cache
used by shared thresholding. This is a prefix-storage budget, not a limit on the detector's
total workspace: padded stripe storage, output threshold images, pyramid levels, contours
and decode buffers are separate. The helper includes the halo and zero prefix row in that
budget and also enforces accumulator safety limits. A budget too small for one stripe
selects the OpenCV threshold fallback. Zero explicitly selects the legacy sizing that uses
only accumulator safety limits and pads the entire input image.

`sparse_tile_side_px_` (default 8, valid 1..65536) aligns regional pyramid requests.
`sparse_full_level_ratio_` (default 0.5, valid `(0,1]`) promotes a level to complete computation
when accumulated tile demand reaches that fraction. `sparse_max_work_ratio_` (default 0.3,
finite and nonnegative) compares the current candidate geometry's estimated regional pixel
work with complete levels before decoding. Above the configured ratio the ordinary full
route is selected for this frame; zero disables this cost decision. It never skips a
candidate or changes a detection threshold. Sparse pixel validity lasts only for
one detector call and is reset independently of image pointer identity. Input ROIs use the
ordinary full-level route; the non-ArUco3 strategy is also unchanged.

`component_edge_band_px_` exposes the component-candidate edge support band (default two
pixels); the minimum inlier and edge support ratios come from `DetectorConfig`. Component
candidate extraction is experimental and cannot be presented as a compatible OpenCV speedup.
Its inlier and edge-support gates can reject otherwise rectangular strokes whose rasterized
corners are rounded or beveled (observed with a four-pixel-thick OpenCV rectangle fixture).
Rendered marker frames passing the synthetic tests do not establish robustness to those
shapes or to real-image blur and occlusion.
Its estimated perimeter is stored as the length of an internal placeholder contour because
existing grouping and level selection consume only the contour length.

The detector validates configuration, image type, dimensions (each side at most 65536 and
`image.total()` at most `INT_MAX` for OpenCV integer statistics), result pointers and dictionary
names before processing. It does not apply CUDA candidate/detection caps. Output changes only
on successful detection. As in OpenCV, ArUco3 forces subpixel refinement even when the shared
refinement setting is `kNone`. Multiple markers with the same ID remain distinct. A moved-from detector reports
`kNotInitialized` from detection and can be reinitialized; unsuccessful reinitialization
invalidates the previous configuration.

## Goals

- Verify IDs, implied rotation, rejected counts and corners against actual OpenCV before
  accepting speed measurements.
- Compare every option independently and in combinations on the same host images, same
  detection configuration and same CPU/core budget.
- Preserve unfavorable cases, p50/p95/p99, startup cost and memory measurements.
- Test dimensions/stride changes to expose stale-buffer errors.

The CMake option `ARUCO3CPU_NATIVE` defaults to `OFF`. Enabling it passes `-mcpu=native`
on aarch64 or `-march=native` on x86 to the experimental library only. This is a separate
machine-specific option and does not enable fast math. The benchmark records the C++ build
flags via `ARUCO3CPU_BUILD_FLAGS`. Such binaries must not be assumed portable to a different
CPU architecture.

## Code provenance

This module adapts the repository's Apache-2.0 `hybrid/hybrid_detector.cpp` host stages and
compiles `hybrid/cpu_candidates.cpp` directly. Those stages independently reproduce the
observable behavior of the pinned Apache-2.0 OpenCV ArUco detector and dictionary; see the
[existing provenance record](../docs/code-provenance.md). The dimension planning and refinement
traversal reproduce the equations already documented in
[the detector pipeline](../docs/design/detector-pipeline.md). `packed_dictionary_` uses the
existing two-mask representation from `src/core/dictionary_match.cu`, mapped onto the byte
ordering of OpenCV's predefined tables at initialization. No dictionary is regenerated.

`cpu_component_candidates.cpp` adapts this repository's `src/core/quad_extract.cu` extreme
point search and `src/core/candidate_filter.cu` inlier/edge support tests to host memory,
using public OpenCV `connectedComponentsWithStats` for eight-connected labels. It retains
row-major points for eligible components and uses double-precision squared-distance tests
for edge support; it is an algorithmic experiment, not a promise of bit equality with the
CUDA extractor. The centroid is taken from OpenCV's component statistics and cast to float.

No external implementation was copied for lazy scheduling, reusable buffers or compact
contour expansion. OpenCV image operations are called through their public APIs. The module
has not used official ArUco GPLv3 sources. Existing third-party notices remain in
[NOTICE](../NOTICE). Future copies of OpenCV source must record their exact files and licenses
before incorporation; imgproc's per-file BSD notices must not be inferred from the project's
Apache-2.0 license.

## Open questions

- Which switches improve end-to-end latency across noisy and clean scenes on the measured CPU?
- How much improvement carries from DGX Spark to Jetson Orin and x86 hosts?
- Do real images expose compatibility gaps absent from the synthetic corpus?
- What remaining stage dominates after thresholding and lazy-pyramid scheduling?
