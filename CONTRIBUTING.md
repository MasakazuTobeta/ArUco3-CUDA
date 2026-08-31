# Contributing

This file is the set of rules shared by the humans and agents who work in this repository.

## Base Standard

- The C++ coding standard is based on MISRA C++ 2023, and the rules in this document act as repository-specific overrides.
- In CUDA C++ as well, make the responsibilities of host code and device code, ownership, synchronization points, and error boundaries explicit.
- Keep the OpenCV coding standard and Apache-2.0 compatibility in mind so that the work can be contributed to OpenCV in the future.
- Contributions to this repository are provided under the Apache License 2.0 unless an explicit exception is stated.
- Do not introduce copyleft dependencies. As a rule, added dependencies must carry a permissive license such as Apache-2.0, MIT, or BSD.
- Do not copy, adapt, or port the GPLv3 source code of official ArUco. Review the [Intellectual property and licensing policy](docs/ip-and-licensing.md) before implementing.

## Development Workflow

1. Before making a change, work out its scope of impact, dependencies, risks, verification method, and the documents that need updating.
2. Derive the requirements for a new algorithm from papers, the observable behavior of the OpenCV CPU implementation, and public specifications.
3. Wherever possible, first write tests that pin down the expected behavior.
4. Optimize performance only after passing an accuracy comparison against the CPU baseline implementation.
5. Record accuracy, kernel time, end-to-end time, and memory usage before and after an optimization.
6. When the API, configuration, evaluation conditions, or supported environments change, update the related documents in the same change.

## C++ / CUDA Coding Rules

- Use C++17 or later and apply RAII consistently.
- Do not use raw `new` / `delete`.
- Use `PascalCase` for types and classes, `snake_case` for functions and local variables, `kPascalCase` for constants, and lowercase for namespaces.
- Give member variables that persist a trailing `_`, and always prefix `this->` when referring to them.
- Use `auto` only where the type is obvious.
- Indicate the unit of a value that has one in its name or its type. For example: `elapsed_ms`, `marker_length_px`.
- Do not use dynamic memory allocation, exceptions, or implicit device-wide synchronization inside CUDA kernels.
- Check errors from CUDA API calls and kernel launches. For asynchronous work, provide a synchronization point in the test or the caller that detects failures.
- Avoid per-frame `cudaMalloc` / `cudaFree`; use a reusable workspace instead.
- Design so that `cudaStream_t` can be supplied through the public API, and avoid an implicit dependency on the default stream.
- Keep machine-specific branching on `__CUDA_ARCH__` local, and maintain a common implementation.
- Apply `clang-format` and do not leave static analysis warnings unaddressed.

## Code Provenance Rules

- Record the papers, specifications, and permissively licensed source code that an implementation is based on in `docs/ip-and-licensing.md` or in the implementation's sidecar document.
- Do not port the structure, expression, comments, or function decomposition of official ArUco GPLv3 code.
- When modifying or partially using Apache-2.0 code from OpenCV, record the file, commit, license, and the nature of the change, and preserve the required copyright and notices.
- Limit the implementation basis for the ArUco3 detection strategy to the 2018 paper and Apache-2.0 OpenCV 4.x. Do not use official ArUco GPLv3 source code as a basis for implementation, review, optimization, or test data creation.
- Treat `predefined_dictionaries.hpp` from OpenCV 4.x, with the version and commit pinned, as the authoritative source for predefined Dictionaries. Do not extract codewords, tables, or images from official ArUco GPLv3 distributions.
- When storing an OpenCV predefined Dictionary inside the repository, record the source file, commit, license, conversion procedure, and the hashes before and after conversion, and preserve the required notices.
- Do not treat a custom Dictionary generated with `extendDictionary()` or similar as if it were identical to `DICT_ARUCO_MIP_36h12`. Keep identifiers, metadata, and tests separate for predefined Dictionaries and custom Dictionaries.
- Even when the CPU implementation was merely run as a compatibility oracle, record the version and configuration used in the evaluation results.
- Do not accept a contribution whose code provenance cannot be explained.
- Treat patents separately from the source code license. Perform the necessary patent clearance before a commercial release.

## Error Handling And Validation

- Validate image type, dimensions, stride, Dictionary, configuration values, pointers, and streams at the boundary of the public API.
- Report failures in host code through exceptions or explicit status values; never continue silently.
- Do not throw exceptions from destructors, CUDA callbacks, or device code.
- Attach context to CUDA errors so that the API name, device, stream, and processing stage can be traced.
- Do not trust external input, files, environment variables, or datasets; validate size limits and formats.

## Comment Rules

- Write everything in English: documents, explanatory comments inside program files, Doxygen, commit messages, and PR bodies.
- Two exceptions. Quote legal wording in its original language and add an English gloss in parentheses; translating a designation of goods or a licence notice changes what it says. And keep non-ASCII test data that exists to exercise encoding, marking why it stays.
- For public classes and functions, document the purpose, arguments, return value, ownership, synchronization behavior, an input example, and an output example. `tools/check_doxygen.py` detects omissions. It also runs from `ctest`.
- When ownership and synchronization behavior are common to the whole class, it is acceptable to state in the class Doxygen that they "apply to all public member functions". Do not copy the same text onto every member.
- Comments should explain the design rationale, assumptions, handling of failures, and performance intent, not restate what the code does.
- For CUDA kernels, explain the mapping between threads / blocks and data, how races are avoided, and the boundary conditions.

## Testing Rules

- Test normal cases, error cases, and boundary values automatically.
- Compare IDs, rotation, corner coordinates, and missed detections against the CPU baseline implementation.
- Include representative conditions for image dimensions, stride, ROI, marker size, Dictionary, distortion, lighting, blur, and occlusion.
- Provide a test route that uses Compute Sanitizer.
- Aim for 100% C0 and C1 coverage, and record the reason for exclusion where it is not met. For CUDA device code, verify execution paths through input partitioning and boundary values, separately from host coverage. Measure with `cmake --preset coverage` followed by `cmake --build build/coverage --target coverage-report`, and record the current state and the reasons for shortfalls in the [implementation plan](docs/implementation-plan.md).
- For tests that deliberately make a CUDA API fail, include `DeliberateError` in the suite name so they can be excluded from Compute Sanitizer runs.
- A file a test reads must have exactly one writer. `ctest` runs in parallel and orders tests by the previous run's timings, and `FIXTURES_REQUIRED` only guarantees that the setup runs first - it does not stop another test from writing the file afterwards. Give every test that produces output a path of its own, and verify with `ctest --schedule-random`.
- Do not substitute performance tests for accuracy tests.

## Benchmark Rules

- Record the number of warm-up iterations, the number of measurement iterations, resolution, marker conditions, power mode, clocks, and the CUDA / driver / OpenCV versions.
- Separate `T_kernel`, `T_end_to_end`, single-frame latency, and multi-frame throughput.
- Store the median, p95, and p99, not just the mean.
- Use the same inputs and detection conditions for CPU and CUDA.
- Treat GPU-resident input and input including the transfer from the host as separate results.
- Do not cherry-pick favorable results; include conditions where the CPU is faster and report the crossover point.

## What CI Covers

GitHub Actions runs on runners without a GPU. Therefore **a green CI does not mean the detection results are correct.**

| Check | Where it runs |
| --- | --- |
| clang-format formatting diff | CI (`tools/check-format.sh`) |
| Doxygen elements in public headers | CI (`tools/check_doxygen.py`) |
| Relative links in documents | CI (`tools/check-doc-links.py`) |
| SPDX notice in source files | CI |
| Compilation for 3 architectures | CI (`ARUCO3CUDA_BUILD_REFERENCE=OFF`) |
| Standalone `find_package` build of the samples | CI (`examples/`) |
| Python tests that need no device | CI (`test/python/test_binding.py`) |
| The C++ and Python renderers agree byte for byte | CI |
| 54 tests that need no device | CI |
| **520 automated tests** | **4 physical machines** |
| **4 Compute Sanitizer tools** | **4 physical machines** |
| **Cross-check against the CPU baseline** | **4 physical machines** (requires OpenCV) |

CI does not build OpenCV because building it from source would take up most of the CI time, which is out of proportion to the goal of detecting compilation regressions. Comparison against the CPU baseline is done on physical machines.

Formatting and link checks use the same script in CI and locally. `cmake --build <dir> --target format-check` and CI both invoke the same `tools/check-format.sh`. Two separate implementations of the same check will always drift apart.

## Commit And Review

- Write commit messages in English, using the Conventional Commits format.
- Keep one commit to one purpose.
- As a rule, write the PR body in English, and include a summary, the changes, verification results, items not done, and related documents.
- Do not commit secrets, credentials, build outputs, or large videos and images used for measurement.
- Add `SPDX-License-Identifier: Apache-2.0` to new source files.
- Determine the license of third-party code from the **file header**, not from the project-wide LICENSE. OpenCV 4.x is Apache-2.0 as a whole, but older files in `imgproc` still carry a 3-clause BSD header. Preserve the copyright notice for anything copied in `NOTICE`.
