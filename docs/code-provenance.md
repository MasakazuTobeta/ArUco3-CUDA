# Code Provenance Record

## Purpose

Following the `Code provenance record` section of the [Intellectual Property and Licensing Policy](ip-and-licensing.md), this document records, in a traceable form, the external materials referenced as the basis for the implementation and the design.

## Scope

The scope covers papers, published specifications, source code under permissive licenses, predefined Dictionary data, and software executed as a CPU baseline.

## Current state

On the GPU side, the recorded subjects are preprocessing, thresholding, candidate extraction, the projective transform and cell sampling, Otsu and border verification, Dictionary matching, identification suppression and compaction, and subpixel refinement of the corners. The CPU side of the hybrid route is also included.

## Records

### PR-000: Investigation of the ArUco3 compatibility specification

| Item | Content |
| --- | --- |
| Implementation | [Detection pipeline design](design/detector-pipeline.md), [Public API draft](design/public-api.md) |
| Basis | The specification observed from OpenCV's public headers and source, and the ArUco3 paper |
| Source version | `opencv/opencv` branch `4.x`, branch head `6dc8e409035489769b4fe7edf3cd63f55bd23ec0` (retrieved 2026-08-27) |
| License | Apache-2.0 |
| Reused expression | None. Parameter names, default values, the order of the stages, and the formulas for the downscale factor and the number of pyramid levels were written down as a specification. The control flow, function decomposition, and comments of the source code were not reproduced |
| Patent review | Per the procedure in the [Intellectual Property and Licensing Policy](ip-and-licensing.md). The results are not published |

The referenced files and their hashes at retrieval time are as follows.

| File | SHA-256 |
| --- | --- |
| `modules/objdetect/include/opencv2/objdetect/aruco_detector.hpp` | `9e2d5bae344e1bb8dc7636430cc63310f3a9fc4e4c6013bfdd95cbade295b54d` |
| `modules/objdetect/src/aruco/aruco_detector.cpp` | `329ac3f0fd90939a23e1cbf21096352e2229a01998bd08d70ca50e17b99f11ed` |

The official ArUco GPLv3 source code has not been referenced.

### PR-001: Incorporating OpenCV into the development container

| Item | Content |
| --- | --- |
| Implementation | `docker/scripts/build-opencv.sh`, [Docker environment design](design/docker-environment.md) |
| Basis | A build for execution as the CPU baseline. No modification of the source code |
| Source version | `opencv/opencv` tag `4.14.0`, commit `0654a42e19215ef25b1d367d822f3c630447e7c7` |
| License | Apache-2.0 |
| Reused expression | None. Only built without modification and installed into the image |
| Patent review | Per the procedure in the [Intellectual Property and Licensing Policy](ip-and-licensing.md). The results are not published |

The build options are `BUILD_LIST=core,imgproc,imgcodecs,calib3d,objdetect` and `WITH_CUDA=OFF`. The source commit and the build options are recorded in `/opt/opencv/share/aruco3cuda/opencv-provenance.json` inside the image, and are embedded into the environment information JSON that `record-environment.sh` emits.

The script confirms that the cloned commit matches the specified value, and aborts the build if it does not.

### PR-002: Incorporating the CUDA Toolkit into the development container

| Item | Content |
| --- | --- |
| Implementation | `docker/scripts/install-cuda-toolkit.sh` |
| Basis | The apt packages of the CUDA Toolkit distributed by NVIDIA. No modification |
| Source version | `cuda-nvcc-13-0` 13.0.88-1 and others. The actual versions are recorded in `/opt/aruco3cuda/cuda-provenance.json` inside the image |
| License | NVIDIA CUDA Toolkit EULA |
| Reused expression | None. Not redistributed; retrieved from NVIDIA's repository at image build time |
| Patent review | Per the procedure in the [Intellectual Property and Licensing Policy](ip-and-licensing.md). The results are not published |

If the image is distributed to a third party, the redistribution conditions of the CUDA Toolkit must be checked against the EULA. At present, the assumed practice is to build on each machine.

### PR-003: Generating the Dictionary packed table

| Item | Content |
| --- | --- |
| Implementation | `tools/dictgen`, `src/dictionary/generated/dict_aruco_mip_36h12.cpp` |
| Basis | The output of OpenCV's `getPredefinedDictionary()` and `Dictionary::getBitsFromByteList()` |
| Source version | OpenCV 4.14.0, commit `0654a42e19215ef25b1d367d822f3c630447e7c7` |
| License | Apache-2.0 |
| Reused expression | The codeword data was converted into a packed representation and stored. No source code was reproduced |
| Patent review | Per the procedure in the [Intellectual Property and Licensing Policy](ip-and-licensing.md). The results are not published |

The generated output records the source OpenCV version and the regeneration procedure as header comments. Nothing was extracted from the official ArUco GPLv3 distribution. That the generated output is consistent with OpenCV is continuously verified by `aruco3cuda_dictgen --check` and `test/reference/test_dictionary_conformance.cpp`.

### PR-004: CPU-side stages of the option C hybrid route

| Item | Content |
| --- | --- |
| Implementation | `hybrid/hybrid_detector.cpp` |
| Basis | The ArUco detector of OpenCV `objdetect`. The behavior was reimplemented to the extent required for verifying compatibility |
| Source version | OpenCV 4.14.0, commit `0654a42e19215ef25b1d367d822f3c630447e7c7` |
| License | Apache-2.0 |
| Reused expression | Reimplementation of the behavior. The decision formulas, the order in which thresholds are applied, and the candidate reordering rules were made identical to OpenCV. Identifiers, comments, and function decomposition were written independently, following this project's conventions |
| Patent review | Per the procedure in the [Intellectual Property and Licensing Policy](ip-and-licensing.md). The results are not published |

The correspondence is as follows. The left column is this project, the right column is the OpenCV function.

| This project | OpenCV |
| --- | --- |
| `find_quad_candidates` | `_findMarkerContours` |
| `reorder_corners` | `_reorderCandidatesCorners` |
| `quad_perimeter`, `average_quad_distance` | the perimeter computation in `MarkerCandidateTree`, `getAverageDistance` |
| `average_module_size` | `getAverageModuleSize` |
| `quad_inside_quad` | `checkMarker1InMarker2` |
| `filter_too_close_candidates` | `ArucoDetectorImpl::filterTooCloseCandidates` |
| `find_optimal_level` | `_findOptPyrImageForCanonicalImg` |
| `extract_cell_pixel_ratio` | `_extractCellPixelRatio` |
| `count_border_errors` | `_getBorderErrors` |
| the depth traversal in `run_cpu_stages` | `ArucoDetectorImpl::identifyCandidates` |
| the corner restoration in `run_cpu_stages` | `findCornerInPyrImage`, `performCornerSubpixRefinement` |

Since identical output is required, there is no choice but to make the decision formulas and the order of application match. At the same time, what was made identical is the observable behavior; the expression of the source code was not reproduced. The referenced files and their hashes at retrieval time are the same as in PR-000.

OpenCV also supports `detectInvertedMarker` and simultaneous detection with multiple Dictionaries, but this project currently implements neither. `filter_too_close_candidates` has only the branch for the case where `detectInvertedMarker` is false.

The official ArUco GPLv3 source code has not been referenced.

### PR-005: GPU decode stages (S7 through S8)

| Item | Content |
| --- | --- |
| Implementation | `src/core/cell_sample.{hpp,cu}`, `src/core/cell_decode.{hpp,cu}`, `src/core/dictionary_match.{hpp,cu}`, `src/core/candidate_tree.{hpp,cu}`, `src/core/detection_emit.{hpp,cu}`, `src/core/corner_refine.{hpp,cu}`, `build_cell_masks` and `identify_marker` in `src/dictionary/dictionary.cpp`, the oracle in `test/reference/test_corner_refine.cpp` |
| Basis | The ArUco detector and `Dictionary::identify` of OpenCV `objdetect`; the Otsu route of `warpPerspective`, `getPerspectiveTransform`, and `threshold` in `imgproc`, plus `pointPolygonTest`, `cornerSubPix`, and `getRectSubPix`; and `meanStdDev` in `core` |
| Source version | OpenCV 4.14.0, commit `0654a42e19215ef25b1d367d822f3c630447e7c7` (retrieved 2026-08-28) |
| License | **Differs per file.** The referenced parts of `objdetect/aruco` and `core` are Apache-2.0. `cornersubpix.cpp`, `samplers.cpp`, `thresh.cpp`, `imgwarp.cpp`, and `geometry.cpp` in `imgproc` carry 3-clause BSD headers. See the table below |
| Reused expression | Reimplementation of the behavior. The decision formulas, the order of operations, the rounding rules, and the position of the suppression were made identical to OpenCV. Identifiers, comments, function decomposition, and data structures were written independently, following this project's conventions. However, the `oracle` namespace in `test_corner_refine.cpp` alone copies the arithmetic and evaluation order verbatim |
| Patent review | Per the procedure in the [Intellectual Property and Licensing Policy](ip-and-licensing.md). The results are not published |

The correspondence is as follows.

| This project | OpenCV |
| --- | --- |
| `solve_lu8` in `cell_sample.cu` | the `LUImpl` used by `getPerspectiveTransform` |
| `warp_canonical_kernel` in `cell_sample.cu` | the `INTER_NEAREST` route of `warpPerspective` |
| `otsu_threshold` in `cell_decode.cu` | `getThreshVal_Otsu_8u` |
| the inner sum and sum of squares in `cell_decode.cu` | `cv::meanStdDev` |
| the outer-border traversal in `cell_decode.cu` | `_getBorderErrors` |
| `build_cell_masks` | the construction part of `CellBitMasks` |
| `identify_marker`, `distance_to_id` in `dictionary_match.cu` | `Dictionary::identify`, `CellBitMasks::hammingDistanceToId` |
| `point_in_quad` in `candidate_tree.cu` | the float route of `pointPolygonTest` and `checkMarker1InMarker2` |
| `parent_kernel` and `depth_kernel` in `candidate_tree.cu` | the containment tree construction at the end of `filterTooCloseCandidates` |
| `suppress_kernel` in `candidate_tree.cu` | the depth traversal in `identifyCandidates` |
| the rotation cancellation in `emit_kernel` in `detection_emit.cu` | `correctCornerPosition` |
| the stage traversal in `refine_kernel` in `corner_refine.cu` | `findCornerInPyrImage` |
| `corner_sub_pix` in `corner_refine.cu` | `cv::cornerSubPix` |
| `sample_patch_inside` and `sample_patch_border` in `corner_refine.cu` | `getRectSubPix_8u32f`, `getRectSubPix_Cn_`, `adjustRect` |
| the `oracle` namespace in `test_corner_refine.cpp` | a verbatim copy of the above 2 to the host |

Since identical output is required, there is no choice but to make the decision formulas and the order of operations match. At the same time, what was made identical is the observable behavior; the expression of the source code was not reproduced. In particular, where OpenCV calls `hal::and8u` and `hal::normHamming` on byte sequences, this project writes a single `__popcll` on a 64-bit packed representation.

The referenced files and their hashes at retrieval time are as follows.

| File | SHA-256 | License in the header | Copyright holders listed |
| --- | --- | --- | --- |
| `modules/objdetect/src/aruco/aruco_dictionary.cpp` | `9fd90d079e62239683300625ad001f437278b75fce6523368ffa3e5a64198ac8` | top-level LICENSE (Apache-2.0) | not stated |
| `modules/objdetect/include/opencv2/objdetect/aruco_dictionary.hpp` | `6b14b87c13dd3629d3fcfd82e63829e41f01f7cc60c166216e0ae99851f6f42e` | top-level LICENSE (Apache-2.0) | not stated |
| `modules/objdetect/include/opencv2/objdetect/aruco_detector.hpp` | `9e2d5bae344e1bb8dc7636430cc63310f3a9fc4e4c6013bfdd95cbade295b54d` | top-level LICENSE (Apache-2.0) | not stated |
| `modules/imgproc/src/imgwarp.cpp` | `d953cdd11db1bf7b562ba7cfb8b36f2aba3198f15714e5deb40840b01ae77912` | **3-clause BSD** | Intel (2000-2008), Willow Garage (2009), Itseez (2014-2015), AMD (2026) |
| `modules/imgproc/src/thresh.cpp` | `ec40ffd08c842c946213cdd46c2b8036e43770445a4c4b797017c96fd3f91117` | **3-clause BSD** | Intel (2000-2008), Willow Garage (2009) |
| `modules/imgproc/src/cornersubpix.cpp` | `7053a847022c929fdd638bb04c93e86eef163a5b1fc454b96f44f103b37bd56c` | **3-clause BSD** | Intel (2000), OpenCV Foundation (2013) |
| `modules/imgproc/src/samplers.cpp` | `aaae361d7a4d9f7131cb4a63c179bb7ad80bc9adddaa0a4ab347dab156bd2ab7` | **3-clause BSD** | Intel (2000), OpenCV Foundation (2013) |
| `modules/imgproc/src/geometry.cpp` | `05ec6cfba5ad82ac809118f7c9b657eee3bb2467f87046c6ed2f5606a2a59996` | **3-clause BSD** | Intel (2000) |
| `modules/objdetect/src/aruco/aruco_detector.cpp` | `329ac3f0fd90939a23e1cbf21096352e2229a01998bd08d70ca50e17b99f11ed` | top-level LICENSE (Apache-2.0) | not stated |
| `modules/core/src/mean.dispatch.cpp` | `8e52e41c7bf7f1942a5c366683d2742040f6d82db770f253b7d61cb18205133a` | top-level LICENSE (Apache-2.0) | not stated |
| `modules/core/src/matrix_decomp.cpp` | `887deb3905e1f4fbcab426be4346b4ada863051579f19e3011721704dd2ac596` | top-level LICENSE (Apache-2.0) | not stated |

**OpenCV 4.x uses different license headers on a per-file basis.** The
project-wide LICENSE is Apache-2.0, but older files in `imgproc` retain
3-clause BSD headers and list Intel Corporation and others as copyright
holders. Clause 1 of the 3-clause BSD requires that redistribution in source
form preserve the copyright notice, the conditions, and the disclaimer.

The `oracle` namespace in `test_corner_refine.cpp` copies expression from two
3-clause BSD files. The later CPU contour and refinement adaptations are
recorded separately in PR-006 below.
**As a conservative measure, the corresponding copyright notices and license
text were placed in the `NOTICE` at the repository root.** Judging whether the
copy legally constitutes a derivative work is outside the scope of this record
and requires confirmation by a professional.

The official ArUco GPLv3 source code has not been referenced.

### PR-006: Further CPU contour and sparse-pyramid optimization

| Item | Content |
| --- | --- |
| Implementation | [CPU contour workspace](../cpu/cpu_contours.md), [sparse pyramid](../cpu/cpu_sparse_pyramid.md) |
| Source version | OpenCV 4.14.0, commit `0654a42e19215ef25b1d367d822f3c630447e7c7` |
| Contour basis | Adaptation of `modules/imgproc/src/contours_truco.cpp` tracing, visited flags and vectorized scan; no file-level license header, covered by the inspected pinned repository Apache-2.0 LICENSE |
| Contour modifications | Sequential tracing per threshold window, reusable vectors, immediate quadrilateral filtering, isolated ROI padding; remove the overlapping internal stripe scheduler and unused approximation modes |
| Refinement basis | Adapted arithmetic from `modules/imgproc/src/cornersubpix.cpp`, whose file header is BSD-3-Clause and names Intel (2000) and OpenCV Foundation (2013) |
| Refinement modifications | Demand-generated current-frame pyramid regions and monitored interpolation footprints; OpenCV public image operations retain numerical behavior |
| Reused expression | Source arithmetic/control flow is adapted explicitly, rather than described as an independent reimplementation; required notices are retained in `NOTICE` |

The contour source SHA-256 is
`57625f4ee96306f467ddba303ef25d965401a9fba8e7ad7ea0ed52a332008cba`.
The refinement source hash and its full BSD notices are recorded above. No
official ArUco GPLv3 source was consulted or used. These CPU changes do not alter
the baseline OpenCV build.

External photo and regression inputs are fetched from the pinned
`opencv/opencv_extra` revision recorded in the
[fixture manifest](../cpu/experiments/opencv-fixtures.json). They are used only as
local compatibility inputs, are not redistributed, and are not treated as
independently annotated ground truth. The manifest records original file hashes
and the lack of a discovered repository/per-directory license file; it does not
assign those images the source-code license.

The subsequent real-background timing evaluation uses 16 public industrial
photographs from Wikimedia Commons. Their
[manifest](../cpu/experiments/real_background_manifest.json) pins description-page
revisions, original URLs, author/license, original bytes and prepared pixels;
the [preparation record](../cpu/experiments/real_background.md) describes the exact
transform and attribution. Photos remain under ignored local data storage,
with their individual public-domain/CC BY/CC BY-SA terms. They are evaluation
inputs, not implementation sources, and are not licensed as repository code.

### PR-007: Device contour and polygon-approximation experiment

The GPU quality investigation adds an independently written directed pixel-edge
contour representation. Its topology is checked against the pinned OpenCV
public `findContours` API; it does not adapt official ArUco code.

Closed-contour polygon approximation in `src/core/contour_approx.cuh` adapts
initialization, splitting and final cleanup from OpenCV
`modules/imgproc/src/approx.cpp` at commit
`0654a42e19215ef25b1d367d822f3c630447e7c7`. That file carries the Intel (2000)
BSD-3-Clause notice, not the repository-wide Apache license. Its SHA-256 is
`d2b9fa2419c9eb6f880676e862d5b5fb38cd250b2db26da3638efc64181d4bcb`.
The complete source notice is retained in the adapted header and `NOTICE`
identifies the source and copyright holder. Modifications replace dynamically
allocated host vectors and stacks with explicitly bounded device storage and
report capacity failures instead of silently dropping paths.

This record describes an experimental implementation; validation and selection
of the final GPU route remain governed by the
[GPU optimization plan](gpu-steady-optimization-plan.md).

### Bounded CPU quadrilateral approximation

The [AprilTag CPU bounded approximation](../apriltag/small_cpu/bounded_quad_approximation.hpp)
adapts closed split ordering and cleanup from the same pinned OpenCV
`approx.cpp` described above. It uses bounded arrays and exact integer
distance comparisons for small integer contours. The helper preserves ordered
four-vertex output in ordinary search and ordered four- or five-vertex output
in supplemental search. Both use the pinned public OpenCV operation outside
the documented bounds. The Intel BSD-3-Clause source notice is retained in
the [module license](../apriltag/small_cpu/LICENSE.opencv-approx) and the source
is listed in `NOTICE`.

## Goals

- Add at least one line of record per implementation PR, keeping the basis explainable after the fact.
- When incorporating a predefined Dictionary, record the source file, commit, conversion procedure, and the hashes before and after conversion in the same table.
- When third-party code is used with modifications, reflect the required copyright and notices in `NOTICE`. **The license is determined by the file's header, not by the project as a whole.**

## Open questions

- Whether the granularity of the records should be per PR or per module.
- The granularity of what is stated in `NOTICE`. At present the source files copied from and their copyright holders are listed, but whether to treat the converted Dictionary data separately has not been decided.

## See also

- [Intellectual Property and Licensing Policy](ip-and-licensing.md)
- [Dictionary policy](dictionaries.md)

### ARM HAL segmentation resize compatibility

The scalar CUDA resize profile in `src/core/preprocess.cu` derives coordinate
quantization, interpolation rounding and dispatch predicates from pinned OpenCV
Carotene (BSD-3-Clause, NVIDIA 2015) and its KleidiCV 26.03 dependency
(Apache-2.0, Arm 2023–2026). The NEON tail and SVE block-coordinate schedules are
observable compatibility requirements. No SIMD code or official ArUco GPL code
was copied. Exact source paths, commit, influence, profile selection and oracle
conditions are recorded in the [preprocessing sidecar](../src/core/preprocess.md);
the file-level notices are preserved in [NOTICE](../NOTICE).

## AprilTag family codewords

The source is OpenCV 4.14.0's Apache-2.0
[AprilTag dictionary header](https://github.com/opencv/opencv/blob/0654a42e19215ef25b1d367d822f3c630447e7c7/modules/objdetect/src/aruco/apriltag/predefined_dictionaries_apriltag.hpp)
(SHA-256 `cf0e7d383b7194225ddc9a51b3b8eef780671862caa1ea3cf026f9a357fb1940`).
For each family, `aruco3cuda_dictgen --dictionary DICT_APRILTAG_<family> --output <file>`
packs each ID at all four rotations using OpenCV's accessor and checks the rotation
convention. The generated source hashes below cover the complete files.

| Family | SHA-256 |
| --- | --- |
| 16h5 | `2506b6c1a2cfc37710a67ccc0a199dd5daf7822961cd81aa2081d8a3a73c838e` |
| 25h9 | `22f87b07610eb5c42736ff76f7fb6c5206b75e296d372f9f9299ae09e0880585` |
| 36h10 | `e97a30f122c336d009729e14d1effaf59d55581bce054a99d930e09fbc83d6bc` |
| 36h11 | `0784a03c3c8e13a9962426a3a77849f7c15a8b12ece371c30d9b1cb1e06975db` |

The CPU detector links the BSD-licensed AprilRobotics implementation at commit
`efa633d9dc6f4ee33bb79b0592250f4aa71e90a8`. Its wrapper and the original
CUDA detector are documented with their respective implementation sources.
The separate benchmark links the user's installation of NVIDIA cuAprilTags.
[CPU implementation](../apriltag/cpu/README.md) · [CUDA implementation](../apriltag/gpu/README.md) · [NVIDIA baseline](../apriltag/benchmark/README.md)
