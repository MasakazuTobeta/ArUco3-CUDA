# C++ detector API

## Purpose

This guide explains the operating contract of the detector. The [public headers](../../include/aruco3cuda) define its types, signatures and defaults.

## Scope

The C++ API accepts grayscale CUDA images and returns device views or host results.
The [C ABI and Python binding](c-abi-and-python.md) wrap the same detector.

## Current state

- `Detector::initialize()` accepts ArUco3 enabled with `kSubpix` refinement, or ArUco3 disabled with `kNone`.
- Corners are already ordered after undoing the matched rotation. `rotations_` records that original rotation.
- Device output has fixed capacities. After completion, a nonzero failure word invalidates the entire result. With a zero failure word, `accepted_total_ > count_` indicates marker-limit truncation.
- Use `DetectorConfig::opencv_defaults()` for OpenCV-compatible initial settings. The project defaults enable the ArUco3 detection strategy and downscaling. [Configuration fields](../../include/aruco3cuda/config.hpp)

## Goals

### Design principles

- Core does not depend on OpenCV types. Converting `cv::Mat` and `cv::cuda::GpuMat` is the adapter's responsibility.
- Every asynchronous API takes a caller-owned `cudaStream_t`. No implicit dependency on the default stream is created.
- Failures are reported through a returned `Status`. Core throws no exceptions. The adapter may convert them into exceptions to match OpenCV conventions.
- The workspace is owned by `Detector`; no per-frame allocation occurs.
- Values that carry units include the unit in their name.

### Namespace and file layout

Everything public lives in namespace `aruco3cuda`; the adapter will live in `aruco3cuda::opencv`.

| Header | What it provides | State |
| --- | --- | --- |
| [`status.hpp`](../../include/aruco3cuda/status.hpp) | `Status`, `to_string`, `last_cuda_error_message` | implemented |
| [`version.hpp`](../../include/aruco3cuda/version.hpp) | version constants and `version_string()` | implemented |
| [`types.hpp`](../../include/aruco3cuda/types.hpp) | `MemorySpace`, `ImageViewU8`, image bounds, `validate_image_view()` | implemented |
| [`config.hpp`](../../include/aruco3cuda/config.hpp) | `CornerRefineMethod`, `DetectorConfig`, `kMaxAdaptiveThresholdWindows` | implemented |
| [`dictionary.hpp`](../../include/aruco3cuda/dictionary.hpp) | `MarkerCode`, `DictionaryTable`, `DictionaryMatch`, `CellMasks`, matching | implemented |
| [`device_probe.hpp`](../../include/aruco3cuda/device_probe.hpp) | `DeviceProbeResult` and the device queries | implemented |
| [`workspace.hpp`](../../include/aruco3cuda/workspace.hpp) | `Workspace`, `WorkspaceStatistics` | implemented |
| [`detections.hpp`](../../include/aruco3cuda/detections.hpp) | `DeviceDetections`, `HostDetections` | implemented |
| [`detector.hpp`](../../include/aruco3cuda/detector.hpp) | `Detector` | implemented |
| [`util/`](../../include/aruco3cuda/util) | JSON writing, SHA-256, and statistics helpers used by the tools | implemented |
| `opencv/adapter.hpp` | conversion to and from `cv::Mat` and `cv::cuda::GpuMat` | not implemented |

### Basic types

[`status.hpp`](../../include/aruco3cuda/status.hpp) declares `Status`, the single error channel of the whole API, together with `to_string()` for logging and test assertions and `last_cuda_error_message()` for the detail behind a `kCudaError`. The enumerators separate an invalid argument, an invalid image view, an invalid configuration, an unsupported dictionary, candidate and marker overflow, a CUDA failure, and use before initialization. The CUDA message is held per thread and carries the API name, the device, the stage, and the string CUDA reported, so a failure can be traced without a debugger; it is overwritten by the next CUDA error on the same thread.

[`types.hpp`](../../include/aruco3cuda/types.hpp) declares `MemorySpace` and `ImageViewU8`, plus the image bounds and `validate_image_view()`. Three points of reasoning sit behind them:

- The memory space is captured in the type rather than in a code path. The DGX Spark GB10 and the Jetson Orin are both integrated GPUs, where host and device share the same physical memory, and yet the cost of an explicit copy, of page locking, and of managed migration still differ. That makes the space a measurement axis of the evaluation, not an implementation detail.
- `ImageViewU8` is non-owning and keeps the row stride separate from the width, so ROIs and non-contiguous row layouts are supported from the start rather than bolted on later.
- `kMaxImageWidthPx` and `kMaxImageHeightPx` exist because external input is not trusted, not because of a performance constraint. They leave ample headroom above the 3840x2160 maximum resolution of the evaluation plan. Validation runs at the boundary, since handing an invalid view straight to CUDA turns an out-of-range access into an asynchronous failure that surfaces somewhere else entirely. Validation calls no CUDA API, so it cannot confirm that a device pointer really exists.

### Configuration

[`config.hpp`](../../include/aruco3cuda/config.hpp) declares `CornerRefineMethod`, the `DetectorConfig` aggregate, and `kMaxAdaptiveThresholdWindows`. The fields are grouped by pipeline stage: adaptive thresholding, candidate filtering, bit reading and matching, corner refinement, the ArUco3 detection strategy, and the CUDA-specific limits. Every field carries its default and its rationale as a comment in the header; the notes that matter at the design level are these.

- `kMaxAdaptiveThresholdWindows` caps the number of swept windows. A binarized image is kept for every step, so an uncapped window count makes the workspace expand without bound.
- The two thresholds with no OpenCV counterpart, `min_quad_inlier_ratio_` and `min_edge_support_ratio_`, exist to reject shapes the extreme-point corner search would otherwise accept. The inlier ratio catches shapes that spill outside the estimated quadrilateral, such as circles, ellipses, and hexagons. It cannot catch concave shapes such as an L or a cross, because a side drawn between two extreme points can run outside the component while the component still fits inside the quadrilateral, so the edge support ratio checks separately that component pixels lie near each side. Both defaults were placed between the measured extremes of synthetic shapes that must pass and must be rejected; the header records those numbers.
- `valid_bit_threshold_` is held explicitly rather than left implicit. It is used in two places, border verification and dictionary matching, and the CPU reference only ever used the OpenCV default for it.
- `relative_corner_refinement_win_size_` keeps the refinement window from reaching into the neighboring cell on small markers. It only applies when ArUco3 is disabled.
- The CUDA-specific block holds the buffer capacities, the reference resolution the workspace is preallocated for, and `cuda_block_dim_`, the threads-per-block side for two-dimensional kernels. `cuda_block_dim_` exists so that device-specific tuning is a configuration value rather than a constant buried in the source; the default of 16 is a safe choice on most devices, and per-device values are still to be determined by measurement.

Validation is a member function so that it can run before a `Detector` exists:

```cpp
Status validate(std::string* out_message = nullptr) const;
```

The defaults for `use_aruco3_detection_` and `min_marker_length_ratio_original_img_` differ from the OpenCV defaults. OpenCV defaults to `useAruco3Detection = false` and `minMarkerLengthRatioOriginalImg = 0.0`, a combination under which no downscaling occurs. Because this project exists to evaluate the ArUco3 detection strategy, the defaults are changed, and `DetectorConfig::opencv_defaults()` is provided separately to return OpenCV-compatible defaults.

### Result representation

[`detections.hpp`](../../include/aruco3cuda/detections.hpp) declares the two result types, and it is the one header in the public set that includes nothing CUDA-specific.

The detector owns the storage behind `DeviceDetections`. Reinitialization, layout
changes and destruction invalidate the view. A new submission can overwrite its
contents. A consumer must order its reads after detection completes and before reuse.

Device corners use separate x and y planes indexed by `corner * capacity_ + detection`.
The kernels read and update that layout in place. `download()` fills caller-owned
`HostDetections` with eight floats per detection in full-resolution image coordinates.
Check its returned status for failure or truncation. [Result types](../../include/aruco3cuda/detections.hpp)

### Detector

[`detector.hpp`](../../include/aruco3cuda/detector.hpp) declares the `Detector` class. It issues everything from preprocessing through corner refinement onto a single stream and returns results that stay on the device.

Ownership is fixed for every public member function: the class owns the workspace and the memory behind `DeviceDetections`; the caller owns the input image and the stream. The `DictionaryTable` is copied to the device inside `initialize()`, so the caller may destroy it once that call returns. A single instance must not be used from several threads at once.

On an explicit stream, a stable frame layout and input address enable CUDA Graph replay.
A change of stream, pointer or layout requires recapture. Passing `nullptr` submits the
stages directly. Steady submissions record a completion event so later submissions
and readbacks can safely reuse detector storage across streams. Keep borrowed input
and results valid until their consumers finish. Reinitialization, layout changes,
readback and failure recovery have distinct synchronization points.
[Submission and storage lifetime](../../src/core/detector.md)

`device_detections()` returns a `Status` and fills an out parameter rather than returning a reference, because after a move it would otherwise have no way to report `kNotInitialized`. `workspace_statistics()` exists so that a test can confirm `allocation_count_` stays at 1 after `initialize()`, which is what makes "no per-frame allocation" checkable rather than merely asserted.

The default constructor, the destructor, and the move operations are all declared out of line. `Impl` is an incomplete type in the header, so the compiler cannot generate them at the point of declaration.

### OpenCV adapter

The adapter is not implemented. It is planned as a separate CMake target, so that core can be built in a configuration that does not link OpenCV.

Its planned surface is four conversions: mapping `cv::aruco::DetectorParameters` onto `DetectorConfig`, building an `aruco3cuda::DictionaryTable` from a `cv::aruco::Dictionary`, wrapping a `cv::cuda::GpuMat` as an `ImageViewU8` after checking its type and stride, and converting `HostDetections` into the `std::vector<std::vector<cv::Point2f>>` and `std::vector<int>` pair that `cv::aruco::detectMarkers` produces. The name on our side is `DictionaryTable`, not `Dictionary`.

One constraint falls out of the header contract and has to be resolved when the adapter is written: `DictionaryTable::codes_` is non-owning and, for the built-in tables, points into static storage. A table built at runtime from a `cv::aruco::Dictionary` needs its codeword storage owned somewhere that outlives the `Detector::initialize()` call.

### Usage example

The [C++ sample](../../examples/detect_image.cpp) initializes one detector, uploads
an image, submits detection on an explicit stream and downloads the result. Run it
with `--repeat 8` to reuse the detector and workspace. The [sample guide](../../examples/README.md)
includes build commands and expected output.

## Design decisions

- Returning `Status` and throwing no exceptions from core structurally prevents exceptions from escaping CUDA callbacks, destructors, and device code.
- Exposing `DeviceDetections` lets a GPU-resident pipeline consume results without host synchronization. The `CUDA-Resident` route in the [evaluation plan](../evaluation-plan.md) uses this API.
- Making `pitch_bytes_` mandatory allows ROIs and non-contiguous input to be handled from the start.
- Overflow is reported both through the return value and a result flag, so nothing is silently truncated.
- Making `Impl` a pimpl limits the public header's dependency on CUDA-specific types to `cudaStream_t`.

## Decisions made

### Public aggregate fields also carry a trailing `_`

The naming convention in `CONTRIBUTING.md` is applied as is. `DeviceProbeResult`, `DictionaryTable`, `ReferenceConfig`, `SceneSpec`, and `BenchmarkConfig` are already implemented under the same convention, and changing the convention only for public aggregates would mix two conventions within the same repository. Call sites become somewhat more verbose, but having a single convention takes priority.

### Validation returns a Status, with the reason received through an optional out parameter

`validate_image_view()` and `DetectorConfig::validate()` return a `Status` and store the reason for failure into a `std::string*`. Passing `nullptr` is allowed, in which case no string is assembled. Validation may be called every frame, so the structure avoids any allocation on the success path.

### Image failures get a dedicated Status

`kInvalidImage` is provided separately from `kInvalidArgument`. An invalid image view indicates a problem in the caller's input path, and it calls for a different response than a mistake in configuration or an index.

## Open questions

- Whether to provide an API besides `download()` that retrieves only partial results.
- Whether simultaneous matching against multiple dictionaries belongs in the initial scope.
- Whether to include the OpenCV adapter in the same library or in a separate target. Same as the open question in the [architecture](../architecture.md).
- Whether to ignore or reject unsupported parameters of `cv::aruco::DetectorParameters`.

## See also

- [Architecture](../architecture.md)
- [Detection pipeline design](detector-pipeline.md)
- [Implementation plan](../implementation-plan.md)
- [Code provenance record](../code-provenance.md)
