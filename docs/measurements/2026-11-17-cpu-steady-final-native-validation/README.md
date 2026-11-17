# Final native Release validation

## Purpose

Validate the final contour-padding change with the CPU library's explicit native
compiler option. This correctness run supersedes the older native checks that
preceded the padding update.

## Scope

The existing `build/cpu-native` configuration had `CMAKE_BUILD_TYPE=Release` and
`ARUCO3CPU_NATIVE=ON`. The CPU test and benchmark targets were rebuilt with one
build job. All 46 discovered Release tests then ran with randomized CTest order
and one test process at a time. Every build/test subprocess inherited affinity
CPUs 0-3. CPUs 5-8 were not used by this task.

The native benchmark executable was rebuilt for a later diagnostic but was
**not executed** here. Durations present in CTest logs and JUnit describe test
runner execution; they are not latency, throughput or speedup measurements.
Other validation work was active on CPUs 0-3, so even test duration must not be
interpreted as a performance result.

## Current state

Generated measurement results, logs and environment captures referenced below
are local-only and ignored by Git. This document and reproduction sources remain
version-controlled; a fresh checkout does not contain the recorded outputs.

The build succeeded and **46/46 tests passed**. JUnit records 46 tests and zero
failures. The recorded CPU/helper/test source fingerprints were unchanged across
the run. In particular, `cpu/cpu_contours.cpp` remained SHA256
`3508f502208042ba8902f9e828ec2b7d713685c84e303a18dd7c8282a450ff87`.

The final binaries have these SHA256 fingerprints:

| Output | SHA256 |
| --- | --- |
| Native CPU tests | `93aced0f440c3536f57436ff6360df4fd9e90ed4d0c3bbdda093d34a1cdfb774` |
| Rebuilt native benchmark | `fb0a95c5cb4c48891d9dd9450725c4036330c4a1df9bf656c1f994200d18fa36` |
| Native CPU static library | `e4257424c7bb7eb8d29b0e9aaa0beccbbd00ce4faa95efefb4f9391b58029fce` |

GCC 13.3.0 compiled the library with `-O3 -DNDEBUG -mcpu=native`, C++17 and the
repository's warning checks. The option applies to the CPU library; it does not
imply that every client translation unit uses native flags. The actual cache
settings and per-file commands are in flags.json (local `flags.json`). The compile
database lists other configured targets too; build.log (local `build.log`) identifies
the targets actually rebuilt and linked by this task.

OpenCV was 4.14.0, commit
`0654a42e19215ef25b1d367d822f3c630447e7c7`, with CUDA disabled in its recorded
provenance. Compiler/CMake/CTest versions and that provenance are retained in
toolchain.log (local `toolchain.log`).

### Evidence

- environment.json (local `environment.json`): exact commands, source fingerprints
  before/after, flags, binary hashes and result scope.
- build.log (local `build.log`): successful single-job build.
- ctest-list.log (local `ctest-list.log`): the 46 tests discovered after rebuilding.
- ctest.log (local `ctest.log`): the randomized execution order and all outcomes.
- ctest.xml (local `ctest.xml`): JUnit results.

No production sources were edited by this validation task, and no executable or
object-file build outputs are stored in this archive.

### Commands executed

These commands ran inside the existing `aruco3-cpu-opt` container, whose
repository mount is `/workspace`:

```bash
taskset -c 0-3 cmake --build build/cpu-native \
  --target aruco3cuda_cpu_tests aruco3cuda_cpu_benchmark -j1
taskset -c 0-3 ctest --test-dir build/cpu-native -N
taskset -c 0-3 ctest --test-dir build/cpu-native \
  --schedule-random --output-on-failure -j1 \
  --output-junit /workspace/docs/measurements/2026-11-17-cpu-steady-final-native-validation/ctest.xml
```

The environment record includes the outer `docker exec` arguments. The raw CTest
log preserves the chosen random order; no fixed scheduling seed was requested.

## Goals

- Verify the final padding source under the native compiler option.
- Preserve commands, flags, outcomes and exact binary/source identities.
- Keep correctness evidence separate from subsequent native-build performance
  diagnostics.

## Open questions

- This run establishes no native-versus-generic speed comparison.
- Other hardware targets and compiler configurations remain separate validation
  work.
