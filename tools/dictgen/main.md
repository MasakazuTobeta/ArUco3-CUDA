# dictgen

## Purpose

Generates the packed codewords used on the CUDA side from OpenCV 4.x predefined Dictionaries, and outputs them as C++ source. Also provides `--check`, which continuously verifies that the generated files are consistent with OpenCV.

## Scope

Covers reading predefined Dictionaries, converting bit arrays to the packed representation, verifying the rotation rule, generating C++ source, and verifying that the generated files are consistent with OpenCV. Generating custom Dictionaries and codeword search by MILP are out of scope.

## Current state

- 18 predefined Dictionaries are supported.
- One run generates one Dictionary.
- The generated files are placed in `src/dictionary/generated/` and committed to the repository.

## Design decisions

### The generated files are committed to the repository

Generating them from OpenCV at build time would make OpenCV a requirement for building `core`. The [architecture](../../docs/architecture.md) sets out a policy of minimizing core's dependency on OpenCV, which this would contradict. The generated files are committed, and agreement with OpenCV is verified continuously by `--check` and `test/reference/test_dictionary_conformance.cpp`.

### `bytesList` is not unpacked by hand

OpenCV's `bytesList` is `CV_8UC4`, but its memory layout is not channel interleaving; it is a contiguous block per rotation. As described in the public header, `bytesList.ptr(i)[k*nbytes + j]` is byte j of rotation k of marker i. Reading this layout incorrectly as interleaved yields corrupted values for every rotation other than 0. To avoid such a misreading, the byte sequence is not unpacked by hand; it is obtained by passing `rotationId` to `Dictionary::getBitsFromByteList()`.

### The rotation rule is verified at generation time

The CUDA side uses a table with the 4 rotations expanded in advance, and compares the rotation of a match against OpenCV. If the rotation order of the table does not agree with OpenCV's definition, this comparison does not hold. At generation time it is confirmed that a chain of `rotate_marker_code()` reproduces rotations 1 through 3 of the table, and that rotating 4 times returns to the original; generation is aborted if they do not agree.

### The generated files are excluded from formatting

Applying `clang-format` would make them differ byte for byte from the generator's output, and verification of regeneration by `--check` would no longer hold. `cmake/Aruco3CudaOptions.cmake` excludes `/generated/` from formatting.

### `extern` is attached to namespace-scope `const`

In C++, a namespace-scope `const` has internal linkage by default. So that it can be referenced from `registry.cpp`, `extern` is stated explicitly on the definitions in the generated files.

### Identifiers are generated in kPascalCase

`CONTRIBUTING.md` specifies `kPascalCase` for constants. The `_` separators in the Dictionary name are treated as word boundaries, so `DICT_ARUCO_MIP_36h12` yields `kDictArucoMip36h12Codes`.

## Goals

- Allow multiple Dictionaries to be generated in a single run.
- Reflect the hashes of the generated files into the [code provenance record](../../docs/code-provenance.md) automatically.
- Also generate the declarations in `registry.cpp`, eliminating the manual work of adding a Dictionary.

## See also

- [Dictionary policy](../../docs/dictionaries.md)
- [Detection pipeline design](../../docs/design/detector-pipeline.md)
- [Code provenance record](../../docs/code-provenance.md)
