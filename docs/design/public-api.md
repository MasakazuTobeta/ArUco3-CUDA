# Public API draft

## Purpose

This document defines the types, ownership, synchronization behavior, and error reporting of the CUDA detector's public API, and presents a structure that keeps the OpenCV dependency out of core.

**The headers under [`include/aruco3cuda/`](../../include/aruco3cuda) are authoritative; this document explains the reasoning behind them.** Declarations are deliberately not restated here. An earlier revision carried a copy of the headers, and the copy drifted until field names, types, and signatures disagreed with the code. Where this text and a header disagree, the header wins and this document is the thing to correct. Do not reintroduce a copy: describe intent and link to the header instead.

## Scope

The scope covers the types and functions exposed by `core`, the configuration values, the result representation, and the conversion API in `adapter/opencv`. Pose estimation APIs, Python bindings, and the ROS2 interface are out of scope.

## Current state

The core API is implemented. `Status` and the CUDA error message accessor, `MemorySpace`, `ImageViewU8` with `validate_image_view()`, `CornerRefineMethod`, `DetectorConfig` with `validate()` and `opencv_defaults()`, `DictionaryTable` with the matching helpers, `DeviceProbeResult`, `Workspace` with `WorkspaceStatistics`, `DeviceDetections`, `HostDetections`, and `Detector` all exist. The per-header breakdown is in [Namespace and file layout](#namespace-and-file-layout).

Not implemented: the OpenCV adapter, and per-stage CUDA event timing (`set_stage_timing_enabled`), which appears nowhere in the code.

Known limitations of what is implemented:

- `Detector::initialize()` accepts only two combinations of settings: ArUco3 enabled with `kSubpix` refinement, and ArUco3 disabled with `kNone`. The other two are rejected, because with ArUco3 enabled and refinement off the corners would be left in downscaled coordinates, and with ArUco3 disabled there is only one level, so refinement would never run. The table in [`detector.hpp`](../../include/aruco3cuda/detector.hpp) states this.
- `DeviceDetections::rotations_` holds the rotation from before it was undone, while the corners are already ordered with it undone. A later stage that reads `rotations_` and rotates again ends up 90 degrees off.
- The GPU allocates its output storage at initialization time, so `max_candidates_` and `max_markers_` are hard limits. Truncation is reported, never silent.

The defaults match the observed specification of `DetectorParameters` in OpenCV 4.x. Only the two items related to the ArUco3 detection strategy have defaults changed to suit the evaluation goals. When defaults identical to OpenCV are required, use `DetectorConfig::opencv_defaults()`. Two candidate-filtering thresholds have no OpenCV counterpart at all: their defaults come from measurements on synthetic shapes, and the measured values that bracket each default are recorded in [`config.hpp`](../../include/aruco3cuda/config.hpp) next to the field.

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

`DeviceDetections` is a set of non-owning references to memory the `Detector` owns, so a GPU-resident pipeline can consume the results without a host synchronization. It holds no synchronization point of its own: the contents are not final until the kernels already issued have completed, and the pointers go stale when the `Detector` is destroyed or when `detect_async()` is called with input whose dimensions changed, so it has to be fetched again after every detection.

The corners are held as separate planes, structure of arrays, indexed as `(corner * capacity_) + detection`. Stages S5 through S10 all read and write that same layout, and S10 writes back in place at the same index, so an array-of-structures layout would mean rewriting kernels that have already been verified down to bit equality. This is settled: `DeviceDetections` ships plain `float*` SoA planes, which also keeps the public header free of any dependency on `vector_types.h`.

Two properties of the contents are easy to get wrong and are worth repeating here. The corners are ordered after the rotation found by dictionary matching has been undone, while `rotations_` still holds the value from before that. And the coordinates are in the full resolution of the input image, not in the coordinates of the downscaled segmentation image. The detection counts before and after truncation each live in a single device-side element, so the host cannot read them without a copy.

`HostDetections` is the value-type counterpart filled by `download()`. It flattens the corners to eight floats per detection and carries both the count before truncation and an explicit overflow flag, so a truncated result is never mistaken for a complete one.

### Detector

[`detector.hpp`](../../include/aruco3cuda/detector.hpp) declares the `Detector` class. It issues everything from preprocessing through corner refinement onto a single stream and returns results that stay on the device.

Ownership is fixed for every public member function: the class owns the workspace and the memory behind `DeviceDetections`; the caller owns the input image and the stream. The `DictionaryTable` is copied to the device inside `initialize()`, so the caller may destroy it once that call returns. A single instance must not be used from several threads at once.

Synchronization is not confined to `download()`, and the distinction matters for anyone measuring end-to-end time:

- `initialize()` synchronizes the whole device at the end, so that the dictionary transfer is complete as seen from any stream afterwards.
- `download()` waits for the stream and then fills the host buffer.
- `detect_async()` normally only issues kernels. On a frame whose input dimensions or pitch differ from the previous one it synchronizes the stream once, before the buffer layout is rebuilt.

When an explicit stream is given, one frame worth of the issue sequence is captured as a CUDA Graph, so every subsequent frame costs a single launch. A captured sequence bakes in the kernel arguments, so it is discarded and recaptured when `initialize()` is called again, when the input dimensions, pitch, or pointer change, or when the stream changes. Changing a setting requires calling `initialize()` again, so there is no path by which a baked value silently goes stale. CUDA does not permit capture on the default stream, so passing `nullptr` takes the path that issues one step at a time.

`device_detections()` returns a `Status` and fills an out parameter rather than returning a reference, because after a move it would otherwise have no way to report `kNotInitialized`. `workspace_statistics()` exists so that a test can confirm `allocation_count_` stays at 1 after `initialize()`, which is what makes "no per-frame allocation" checkable rather than merely asserted.

The default constructor, the destructor, and the move operations are all declared out of line. `Impl` is an incomplete type in the header, so the compiler cannot generate them at the point of declaration.

### OpenCV adapter

The adapter is not implemented. It is planned as a separate CMake target, so that core can be built in a configuration that does not link OpenCV.

Its planned surface is four conversions: mapping `cv::aruco::DetectorParameters` onto `DetectorConfig`, building an `aruco3cuda::DictionaryTable` from a `cv::aruco::Dictionary`, wrapping a `cv::cuda::GpuMat` as an `ImageViewU8` after checking its type and stride, and converting `HostDetections` into the `std::vector<std::vector<cv::Point2f>>` and `std::vector<int>` pair that `cv::aruco::detectMarkers` produces. The name on our side is `DictionaryTable`, not `Dictionary`.

One constraint falls out of the header contract and has to be resolved when the adapter is written: `DictionaryTable::codes_` is non-owning and, for the built-in tables, points into static storage. A table built at runtime from a `cv::aruco::Dictionary` needs its codeword storage owned somewhere that outlives the `Detector::initialize()` call.

### Usage example

Verified against the headers listed above. The `Detector` is constructed once and reused; nothing in the per-frame path allocates.

```cpp
#include <cstddef>
#include <cstdint>
#include <string>

#include "aruco3cuda/detector.hpp"

bool detect_one_frame(const std::uint8_t* device_image, int width_px, int height_px,
                      std::size_t pitch_bytes) {
    const aruco3cuda::DictionaryTable* dictionary =
            aruco3cuda::find_builtin_dictionary("DICT_ARUCO_MIP_36h12");
    if (dictionary == nullptr) {
        return false;
    }

    aruco3cuda::DetectorConfig config;
    config.max_width_px_ = 1920;
    config.max_height_px_ = 1080;

    std::string message;
    aruco3cuda::Detector detector;
    if (detector.initialize(*dictionary, config, &message) != aruco3cuda::Status::kOk) {
        return false;
    }

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        return false;
    }

    aruco3cuda::ImageViewU8 view;
    view.data_ = device_image;
    view.width_px_ = width_px;
    view.height_px_ = height_px;
    view.pitch_bytes_ = pitch_bytes;
    view.space_ = aruco3cuda::MemorySpace::kDevice;

    bool ok = detector.detect_async(view, stream, &message) == aruco3cuda::Status::kOk;

    // The results can be read on the device without synchronizing.
    aruco3cuda::DeviceDetections device_detections;
    ok = ok && detector.device_detections(&device_detections) == aruco3cuda::Status::kOk;

    // Or brought back to the host, which synchronizes the stream once.
    // kMarkerOverflow still fills out; it means the results were truncated.
    aruco3cuda::HostDetections result;
    const aruco3cuda::Status status = detector.download(&result, stream, &message);
    ok = ok && (status == aruco3cuda::Status::kOk ||
                status == aruco3cuda::Status::kMarkerOverflow);

    cudaStreamDestroy(stream);
    return ok;  // result.ids_ holds the IDs, result.corners_ eight floats per detection.
}
```

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
