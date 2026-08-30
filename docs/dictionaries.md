# Dictionary policy

## Purpose

This document defines the compatibility, generation method, data provenance, and verification method for the marker Dictionaries that ArUco3-CUDA supports.

## Scope

- The predefined ArUco Dictionaries of OpenCV 4.x
- `DICT_ARUCO_MIP_36h12`
- Generation of marker images
- Generation of custom Dictionaries
- Conversion into lookup tables for CUDA

## Current state

- The CUDA detector is not implemented. The Dictionary packed table, the matching processing, and the conversion tool are implemented.
- Current OpenCV 4.x includes `DICT_ARUCO_MIP_36h12` as a predefined Dictionary.
- `DICT_ARUCO_MIP_36h12` is 6x6 bits, 250 codes, with a minimum Hamming distance of 12. Measured on OpenCV 4.14.0, it gives `markerSize = 6`, 250 rows in `bytesList`, and `maxCorrectionBits = 5`.
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

For the initial implementation, the predefined Dictionaries of OpenCV 4.x are the authoritative source for compatibility. The first mandatory target is `DICT_ARUCO_MIP_36h12`; the others — `DICT_4X4_*`, `DICT_5X5_*`, `DICT_6X6_*`, `DICT_7X7_*` — will be added incrementally using the same loader and lookup format.

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

- Make the target Dictionaries, including `DICT_ARUCO_MIP_36h12`, byte-for-byte compatible with OpenCV 4.x.
- Be able to explain the provenance of the Dictionary data and the generating code within the bounds of Apache-2.0.
- Guarantee by automated test that changes to the CUDA memory layout still match the authoritative source.
- Keep the API such that adding custom Dictionaries in the future will not be confused with the predefined ones.

## Open questions

- Which Dictionaries besides `DICT_ARUCO_MIP_36h12` are mandatory for the first release.
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
