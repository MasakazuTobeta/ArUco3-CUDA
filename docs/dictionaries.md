# Dictionary policy

## Purpose

This document defines the compatibility, generation method, data provenance, and verification method for the marker Dictionaries that ArUco3-CUDA supports.

## Scope

- The predefined ArUco Dictionaries of OpenCV 4.x
- Generation of marker images
- Generation of custom Dictionaries
- Conversion into lookup tables for CUDA

## Current state

Seventeen Dictionaries are bundled: the sixteen `DICT_NxN_*` of OpenCV 4.x, and `DICT_ARUCO_MIP_36h12`. The Dictionary is a parameter of `Detector::initialize`, so which one is used is chosen by the caller.

| Dictionary | Bits | Codes | `maxCorrectionBits` |
| --- | --- | --- | --- |
| `DICT_4X4_50` / `_100` / `_250` / `_1000` | 4 | 50 / 100 / 250 / 1000 | 1 / 1 / 1 / 0 |
| `DICT_5X5_50` / `_100` / `_250` / `_1000` | 5 | 50 / 100 / 250 / 1000 | 3 / 3 / 2 / 2 |
| `DICT_6X6_50` / `_100` / `_250` / `_1000` | 6 | 50 / 100 / 250 / 1000 | 6 / 5 / 5 / 4 |
| `DICT_7X7_50` / `_100` / `_250` / `_1000` | 7 | 50 / 100 / 250 / 1000 | 9 / 8 / 8 / 6 |
| `DICT_ARUCO_MIP_36h12` | 6 | 250 | 5 |

Measured on OpenCV 4.14.0. The packed tables come to 708 KB of generated source.

### What is deliberately not bundled

**`DICT_ARUCO_ORIGINAL`.** It contains markers that are unchanged by a 180 degree rotation, so their orientation cannot be recovered: decoding a marker rotated by 2 reports rotation 0, and by 3 reports rotation 1. Its minimum Hamming distance over all ids and rotations is also 1, meaning two of its markers differ by a single bit, while OpenCV declares one correctable bit for it. Every other bundled Dictionary satisfies `maxCorrectionBits = (minimum distance - 1) / 2`; this one does not. Bundling a Dictionary whose orientation is ambiguous would put a defect into every pose a caller computes from it.

**The four AprilTag families.** OpenCV lists `DICT_APRILTAG_16h5`, `_25h9`, `_36h10`, and `_36h11` in the same enumeration and `cv::aruco` detects them, but they are AprilTag rather than ArUco and are out of scope here. Nothing in the implementation prevents them: `tools/dictgen` generates the same packed form for them.

**The other MIP families.** `DICT_ARUCO_MIP_36h12` is the only MIP Dictionary OpenCV ships. `ARUCO_MIP_16h3` and `ARUCO_MIP_25h7` exist solely in the GPLv3 ArUco distribution, which the [code provenance rules](../CONTRIBUTING.md) rule out as a source. They cannot be reconstructed from the paper either: the codewords are the output of a search, not a formula. A Dictionary generated with `extendDictionary()` to the same parameters is a different Dictionary, and must be identified as one.

### What the marker size costs

The Dictionary reaches the pipeline as three numbers. `marker_size` sets the canonical patch side to `(bits + 2 * border) * perspective_remove_pixel_per_cell`, so 4x4 is 24 px and 7x7 is 36 px by default. `code_count` sets the length of the matching loop, which on the device runs over every id without an early exit. `max_correction_bits` only feeds a comparison inside that loop and changes no amount of work; neither does the minimum Hamming distance.

Measured on the DGX Spark at 1280x720 with 4 markers, none of this is visible: the spread between Dictionaries (14.2%) is the same size as the run-to-run spread of a single Dictionary (13.9%), and `DICT_6X6_250` and `DICT_ARUCO_MIP_36h12`, which are structurally identical, differed by 13%. Twenty times the codes cost nothing measurable.

- OpenCV's `getPredefinedDictionary()` retrieves the codewords that ship with it.
- `Dictionary::generateImageMarker()` renders an image of the specified ID from a retrieved Dictionary.
- `extendDictionary()` can generate a custom Dictionary, but it is not an API for reproducing the same codeword set as the predefined `DICT_ARUCO_MIP_36h12`, which was generated with MILP.

## Terminology distinctions

| Term | Meaning |
| --- | --- |
| Dictionary retrieval | Reading out the `bytesList`, `markerSize`, and `maxCorrectionBits` that ship with OpenCV |
| Marker image generation | Drawing a printable, evaluation-ready image including the black border, from a Dictionary and an ID |
| Custom Dictionary generation | Searching for a new codeword set from a specified marker size and code count |
| MIP Dictionary regeneration | Solving the MILP problem from the paper to obtain a codeword set equivalent or identical to the predefined MIP Dictionary |

## Approach adopted

The predefined Dictionaries of OpenCV 4.x are the authoritative source for compatibility. All sixteen `DICT_NxN_*` and `DICT_ARUCO_MIP_36h12` are bundled, through the same loader and lookup format.

The source for the predefined Dictionaries is `modules/objdetect/src/aruco/predefined_dictionaries.hpp` from OpenCV 4.x at a pinned tag or commit. We do not extract codewords, tables, or marker images from the official ArUco GPLv3 distribution.

On the CUDA side, the following are generated at build time from the authoritative codewords.

- The canonical bits for each ID
- The packed codewords for all 4 rotations
- The layout used to compute Hamming distance
- The `markerSize` and `maxCorrectionBits` metadata
- The source OpenCV version, commit, input hash, and output hash

When the generated artifacts are stored in the source tree, the Apache-2.0 attribution and the generation procedure are included in the same change.

## Verification

The following automated tests are mandatory for each Dictionary.

Verifications 1 through 5 are implemented as `test/reference/test_dictionary_conformance.cpp`. Verification 6 will be addressed when the CUDA implementation is added.

1. The ID count, marker size, and maximum correction bit count match the OpenCV baseline.
2. The packed codewords for all IDs and all 4 rotations match OpenCV's `bytesList`.
3. Decoding the marker image of every ID yields the original ID and rotation.
4. For bit flips from 1 up to around the correction limit, accept / reject matches OpenCV.
5. Recomputing the minimum Hamming distance within the Dictionary matches the nominal value.
6. The CPU and CUDA lookups return the same ID, rotation, and distance for the same input.

## Goals

- Keep every bundled Dictionary byte-for-byte compatible with OpenCV 4.x.
- Be able to explain the provenance of the Dictionary data and the generating code within the bounds of Apache-2.0.
- Guarantee by automated test that changes to the CUDA memory layout still match the authoritative source.
- Keep the API such that adding custom Dictionaries in the future will not be confused with the predefined ones.

## Open questions

- Whether to offer a Dictionary generated with `extendDictionary()` as a separate, clearly distinct feature, for callers who need parameters no predefined Dictionary provides.
- We decided to commit the codewords retrieved from OpenCV into the repository. Generating them at build time would make OpenCV a requirement for building the core, which would contradict the [architecture](architecture.md) policy that "the core minimizes its dependence on OpenCV types." The agreement between the generated artifacts and OpenCV is verified continuously with `aruco3cuda_dictgen --check` and automated tests.
- We do not reformat the generated artifacts. Reformatting would break byte-for-byte agreement with the generator's output, which would invalidate regeneration checks.
- The conditions for switching between CUDA constant memory and global memory.
- Whether to bring independent Dictionary generation with a MILP solver into future scope.

## See also

- [Intellectual property and licensing policy](ip-and-licensing.md)
- [Implementation plan](implementation-plan.md)
- [Docker environment design](design/docker-environment.md)
- [OpenCV Dictionary API](https://docs.opencv.org/4.x/d5/d0b/classcv_1_1aruco_1_1Dictionary.html)
- [OpenCV predefined dictionaries](https://github.com/opencv/opencv/blob/4.x/modules/objdetect/src/aruco/predefined_dictionaries.hpp)
- [Generation of fiducial marker dictionaries using Mixed Integer Linear Programming](https://doi.org/10.1016/j.patcog.2015.09.023)
