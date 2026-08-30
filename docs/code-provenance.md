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

The highest degree of copying in this project is the `oracle` namespace in
`test_corner_refine.cpp`, and both files it was copied from are 3-clause BSD.
**As a conservative measure, the corresponding copyright notices and license
text were placed in the `NOTICE` at the repository root.** Judging whether the
copy legally constitutes a derivative work is outside the scope of this record
and requires confirmation by a professional.

The official ArUco GPLv3 source code has not been referenced.

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
