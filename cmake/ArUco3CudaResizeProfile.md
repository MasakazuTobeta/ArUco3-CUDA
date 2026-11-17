# CUDA segmentation resize profile

## Purpose

Make the segmentation pixel arithmetic reproducible against the selected CPU
OpenCV baseline, without tying it to the GPU model or changing a public C struct.

## Scope

CMake selection, host-HAL descriptor resolution, and metadata identity. The
arithmetic itself and its provenance are described in
[preprocess](../src/core/preprocess.md).

## Current state

`ARUCO3CUDA_RESIZE_PROFILE` accepts `GENERIC` and `OPENCV_ARM_HAL`. The fresh-build
default remains `GENERIC`: selecting ARM HAL also requires identifying the
linked OpenCV build's horizontal contraction policy. The measured target builds
select the ARM profile explicitly.

`ARUCO3CUDA_RESIZE_ARM_BACKEND` accepts `AUTO`, `NEON`, `SVE128`, `SVE256`, and
`SVE512`. It controls numerical vector/tail coordinate behavior, not the GPU's
instruction set. `AUTO` queries Linux ARM SVE2 capabilities and the calling
thread's SVE vector length on first use, then caches the result. Without SVE2 or
one of the supported 16/32/64-byte vector lengths it selects NEON. Explicit
choices bypass probing. Changing a host thread's vector length afterward does
not change the cached descriptor; use an explicit override for such experiments.
The generic profile ignores the ARM descriptor and reports `GENERIC`.

`compiled_resize_profile()` and `compiled_resize_backend()` report the core's
actual choice. `resize_sve_vector_bytes()` supplies the numerical descriptor to
preprocessing: 0 for generic/NEON, or 16/32/64. All are host-only and make no CUDA
calls. Their definitions live in core-only `resize_profile.cpp`: the optimized
CPU library separately compiles `config.cpp` and must never provide a second
copy of these build-dependent getters. The descriptor is stored with the reusable preprocessing layout.

`ARUCO3CUDA_CAROTENE_X_FMA` separately selects `FUSED` or `SEPARATE` for
Carotene's general horizontal coordinate calculation. This is a property of the
linked OpenCV build's compiler contraction, not of SVE capability. The measured
Spark build requires `SEPARATE`; the measured Orin build requires `FUSED`.
The CPU scalar models matched all 1254 cases with these respective settings.
`AUTO` is accepted only with `GENERIC` for now: the compiler building this core
cannot establish which compiler built an externally linked OpenCV library.
ARM HAL configuration therefore requires an explicit measured choice. A fresh
build retains `GENERIC` rather than inferring contraction from the core compiler.

Both measured target profiles passed the three maintained segmentation tests,
including zero-tolerance public-OpenCV comparison on random strided inputs,
vector tails, small dimensions, nondefault resize ratios, and wide coordinate
recalibration. Both explicit descriptors also passed the five retry boundary tests and all
four Compute Sanitizer tools after the overflow-safe coordinate update. These
are correctness observations, not resize or detector speed measurements.
`resize_carotene_x_fma()` supplies the boolean to preprocessing and
`compiled_carotene_x_fma()` reports `FUSED`, `SEPARATE`, or `NOT_APPLICABLE`.

Benchmark and audit JSON record all three identities. Aggregation keeps different
profiles, descriptors, and unknown historical identities separate. Existing C
configuration and result structure layouts are unchanged.

### Reproducing the measured host descriptors

The maintained `dgx-spark-opencv-arm` preset selects `OPENCV_ARM_HAL`, `SVE128`,
and `SEPARATE`; `jetson-orin-opencv-arm` selects `OPENCV_ARM_HAL`, `NEON`, and
`FUSED`. Both inherit the corresponding architecture preset and keep experimental
GPU routes disabled. They describe the validated pinned OpenCV 4.14 builds, not
all installations on those machines. Existing architecture-only presets and
fresh configurations retain `GENERIC`.

```sh
cmake --preset dgx-spark-opencv-arm -DOpenCV_DIR=/opt/opencv/lib/cmake/opencv4
cmake --build --preset dgx-spark-opencv-arm
ctest --preset dgx-spark-opencv-arm -R 'resize_profile|SegmentationTest'
```

Use `jetson-orin-opencv-arm` for the corresponding Orin commands. A different
OpenCV compiler, HAL build, or SVE vector length requires an explicit validated
override. CMake cache values remain overridable with `-D`; presets do not make
hardware-dependent decisions inside CUDA kernels.

## Goals

- Validate public CPU resize bytes for each explicit descriptor and linked baseline.
- Run the same CUDA arithmetic on both target GPUs for an explicitly selected
  profile and descriptor.
- Preserve a selectable generic control and visible identities in every result.

## Open questions

- Which additional OpenCV builds or host vector lengths need separate validated
  descriptors? No compatibility claim is made for untested HAL versions.
