# Public API draft

## Purpose

This document defines the types, ownership, synchronization behavior, and error reporting of the CUDA detector's public API, and presents a structure that keeps the OpenCV dependency out of core.

## Scope

The scope covers the types and functions exposed by `core`, the configuration values, the result representation, and the conversion API in `adapter/opencv`. Pose estimation APIs, Python bindings, and the ROS2 interface are out of scope.

## Current state

The following is implemented. Per-stage CUDA event timing (`set_stage_timing_enabled`) is not implemented.

`Status`, `MemorySpace`, `ImageViewU8`, `CornerRefineMethod`, `DetectorConfig`, and their respective bounds validation are implemented. `Dictionary` is implemented as `include/aruco3cuda/dictionary.hpp`. `DeviceDetections`, `HostDetections`, and `Detector` are also implemented.

The defaults match the observed specification of `DetectorParameters` in OpenCV 4.x. Only the two items related to the ArUco3 detection strategy have defaults changed to suit the evaluation goals. When defaults identical to OpenCV are required, use `DetectorConfig::opencv_defaults()`.

## Goals

### Design principles

- Core does not depend on OpenCV types. Converting `cv::Mat` and `cv::cuda::GpuMat` is the adapter's responsibility.
- Every asynchronous API takes a caller-owned `cudaStream_t`. No implicit dependency on the default stream is created.
- Failures are reported through a returned `Status`. Core throws no exceptions. The adapter may convert them into exceptions to match OpenCV conventions.
- The workspace is owned by `Detector`; no per-frame allocation occurs.
- Values that carry units include the unit in their name.

### Namespace and file layout

```
include/aruco3cuda/
  status.hpp       // Status, last_cuda_error_message          implemented
  version.hpp      // version information                      implemented
  types.hpp        // MemorySpace, ImageViewU8, bounds checks   implemented
  config.hpp       // CornerRefineMethod, DetectorConfig        implemented
  dictionary.hpp   // DictionaryTable, matching                 implemented
  device_probe.hpp // querying device properties                implemented
  detections.hpp   // DeviceDetections, HostDetections          implemented
  detector.hpp     // Detector                                  implemented
include/aruco3cuda/opencv/
  adapter.hpp      // conversion to/from cv::Mat, cv::cuda::GpuMat   not implemented
```

### Basic types

```cpp
// SPDX-License-Identifier: Apache-2.0
namespace aruco3cuda {

/// The memory space an input buffer lives in. The cost differs even on an integrated GPU, so it is made explicit.
enum class MemorySpace : int {
  kHostPageable = 0,
  kHostPinned = 1,
  kManaged = 2,
  kDevice = 3,
};

/// Result state of the public API. The core throws no exceptions.
enum class Status : int {
  kOk = 0,
  kInvalidImage,          ///< Pointer, size, or pitch is invalid
  kInvalidConfig,         ///< A setting is out of range or contradicts another one
  kUnsupportedDictionary,
  kCandidateOverflow,     ///< The candidate limit was exceeded; results are truncated
  kMarkerOverflow,        ///< The detection limit was exceeded; results are truncated
  kCudaError,             ///< A CUDA API call or a kernel launch failed
  kNotInitialized,
};

/// A non-owning view of an 8-bit grayscale image. Ownership stays with the caller.
struct ImageViewU8 {
  const std::uint8_t* data_ = nullptr;
  int width_px_ = 0;
  int height_px_ = 0;
  std::size_t pitch_bytes_ = 0;   ///< Row stride. It is not necessarily equal to width_px_
  MemorySpace space_ = MemorySpace::kDevice;
};

}  // namespace aruco3cuda
```

### Configuration

```cpp
// SPDX-License-Identifier: Apache-2.0
namespace aruco3cuda {

enum class CornerRefineMethod : int {
  kNone = 0,
  kSubpix = 1,
};

/// Detection settings. The defaults follow the DetectorParameters of OpenCV 4.x.
struct DetectorConfig {
  // Adaptive thresholding
  int adaptive_thresh_win_size_min_ = 3;
  int adaptive_thresh_win_size_max_ = 23;
  int adaptive_thresh_win_size_step_ = 10;
  double adaptive_thresh_constant_ = 7.0;

  // Candidate filtering
  double min_marker_perimeter_rate_ = 0.03;
  double max_marker_perimeter_rate_ = 4.0;
  double polygonal_approx_accuracy_rate_ = 0.03;
  double min_corner_distance_rate_ = 0.05;
  int min_distance_to_border_px_ = 3;
  double min_marker_distance_rate_ = 0.125;
  float min_group_distance_ = 0.21f;

  // Bit reading and matching
  int marker_border_bits_ = 1;
  int perspective_remove_pixel_per_cell_ = 4;
  double perspective_remove_ignored_margin_per_cell_ = 0.13;
  double max_erroneous_bits_in_border_rate_ = 0.35;
  double min_otsu_std_dev_ = 5.0;
  double error_correction_rate_ = 0.6;
  float valid_bit_id_threshold_ = 0.49f;

  // Corner refinement
  CornerRefineMethod corner_refine_method_ = CornerRefineMethod::kSubpix;
  int corner_refinement_win_size_px_ = 5;
  int corner_refinement_max_iterations_ = 30;
  double corner_refinement_min_accuracy_ = 0.1;

  // ArUco3 detection strategy
  bool use_aruco3_detection_ = true;
  int min_side_length_canonical_img_px_ = 32;
  float min_marker_length_ratio_original_img_ = 0.05f;

  // CUDA specific
  int max_candidates_ = 4096;      ///< Capacity of the candidate buffer. On overflow, kCandidateOverflow
  int max_markers_ = 1024;         ///< Capacity of the detection buffer. On overflow, kMarkerOverflow
  int max_width_px_ = 3840;        ///< Reference for preallocating the workspace
  int max_height_px_ = 2160;
  int candidate_block_size_ = 128; ///< Determined per device by measurement. Not a hard-coded value in the source

  /// Detects settings that contradict each other. It can be called before the Detector is created.
  Status validate() const;
};

}  // namespace aruco3cuda
```

The defaults for `use_aruco3_detection_` and `min_marker_length_ratio_original_img_` differ from the OpenCV defaults. OpenCV defaults to `useAruco3Detection = false` and `minMarkerLengthRatioOriginalImg = 0.0f`, a combination under which no downscaling occurs. Because this project exists to evaluate the ArUco3 detection strategy, the defaults are changed, and `DetectorConfig::opencv_defaults()` is provided separately to return OpenCV-compatible defaults.

### Result representation

```cpp
// SPDX-License-Identifier: Apache-2.0
namespace aruco3cuda {

/// Detection results that stay resident on the device. A GPU-resident pipeline can read them
/// without synchronization.
///
/// The corners are held as separate planes (SoA). The index is (corner * capacity_) + detection.
/// S5 through S10 all use the same layout, and S10 writes back in place at this index.
/// Switching to an AoS of float2 would mean rewriting kernels that have already been
/// verified down to bit equality. It also has the advantage that the public header does not
/// depend on vector_types.h.
struct DeviceDetections {
  std::int32_t* ids_ = nullptr;
  std::int32_t* rotations_ = nullptr;
  float* corner_x_ = nullptr;
  float* corner_y_ = nullptr;
  std::int32_t* source_ = nullptr;   ///< Index of the candidate it came from
  std::int32_t* count_ = nullptr;    ///< Detection count on the device
  std::int32_t* accepted_total_ = nullptr;  ///< Detection count before truncation
  int capacity_ = 0;
};

/// Detection results pulled out to the host.
struct HostDetections {
  std::vector<std::int32_t> ids_;
  std::vector<float> corners_;       ///< 8 elements per detection: x0,y0,...,x3,y3
  std::vector<std::int32_t> rotations_;
  std::int32_t accepted_total_ = 0;  ///< Detection count before truncation
  bool marker_overflow_ = false;
};

}  // namespace aruco3cuda
```

### Detector

```cpp
// SPDX-License-Identifier: Apache-2.0
namespace aruco3cuda {

/// CUDA implementation of the ArUco3 detection strategy.
///
/// Ownership:
///   This class owns the memory backing the workspace and the DeviceDetections.
///   The caller owns the memory of the input image.
/// Synchronization:
///   detect_async() only issues kernels onto the stream and does not synchronize the host.
///   download() waits for the stream to complete and then fills the host buffer.
class Detector {
 public:
  Detector() = default;
  ~Detector();
  Detector(const Detector&) = delete;
  Detector& operator=(const Detector&) = delete;
  Detector(Detector&&) noexcept;
  Detector& operator=(Detector&&) noexcept;

  /// Allocates the workspace and transfers the Dictionary to the device.
  /// On failure, the internal state is left unchanged.
  Status initialize(const DictionaryTable& dictionary, const DetectorConfig& config,
                    std::string* out_message = nullptr);

  /// Issues a detection onto the stream. It does not synchronize the host.
  /// Example input: 1920x1080, pitch_bytes_ = 1920, space_ = kDevice
  /// Example output: the detection count on the device is written into detections.count_
  Status detect_async(const ImageViewU8& image, cudaStream_t stream);

  /// Copies out the device-resident results of the most recent detect_async(). No
  /// synchronization takes place.
  /// It does not return a reference, because after a move it would no longer be able to
  /// return kNotInitialized.
  Status device_detections(DeviceDetections* out) const;

  /// Waits for the stream to complete and obtains the results on the host. This is the
  /// only place where synchronization takes place.
  Status download(HostDetections* out, cudaStream_t stream,
                  std::string* out_message = nullptr);

  /// How the device workspace is being used. A test confirms that allocation_count_
  /// stays at 1.
  const WorkspaceStatistics& workspace_statistics() const;

  /// Whether initialize() has succeeded.
  bool initialized() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace aruco3cuda
```

### OpenCV adapter

```cpp
// SPDX-License-Identifier: Apache-2.0
namespace aruco3cuda::opencv {

/// Maps cv::aruco::DetectorParameters onto DetectorConfig.
/// Returns kInvalidConfig if an unsupported parameter is set.
Status from_detector_parameters(const cv::aruco::DetectorParameters& params,
                                DetectorConfig& out);

/// Builds a device-side Dictionary from a cv::aruco::Dictionary.
Status from_cv_dictionary(const cv::aruco::Dictionary& dictionary, Dictionary& out);

/// Creates a non-owning view from a GpuMat. Validates contiguity and type.
Status view_from_gpu_mat(const cv::cuda::GpuMat& mat, ImageViewU8& out);

/// Converts the detection results into the same representation as OpenCV's detectMarkers.
void to_cv_output(const HostDetections& detections,
                  std::vector<std::vector<cv::Point2f>>& corners,
                  std::vector<int>& ids);

}  // namespace aruco3cuda::opencv
```

The adapter is a separate CMake target, so that core can be built in a configuration that does not link OpenCV.

### Usage example

```cpp
aruco3cuda::Detector detector;
aruco3cuda::DetectorConfig config;
config.max_width_px_ = 1920;
config.max_height_px_ = 1080;

if (detector.initialize(dictionary, config) != aruco3cuda::Status::kOk) { /* ... */ }

cudaStream_t stream = nullptr;
cudaStreamCreate(&stream);

aruco3cuda::ImageViewU8 view{d_image, 1920, 1080, pitch_bytes, aruco3cuda::MemorySpace::kDevice};
if (detector.detect_async(view, stream) != aruco3cuda::Status::kOk) { /* ... */ }

aruco3cuda::HostDetections result;
if (detector.download(result, stream) != aruco3cuda::Status::kOk) { /* ... */ }
```

## Design decisions

- Returning `Status` and throwing no exceptions from core structurally prevents exceptions from escaping CUDA callbacks, destructors, and device code.
- Exposing `DeviceDetections` lets a GPU-resident pipeline consume results without host synchronization. The `CUDA-Resident` route in the [evaluation plan](../evaluation-plan.md) uses this API.
- Making `pitch_bytes_` mandatory allows ROIs and non-contiguous input to be handled from the start.
- Overflow is reported both through the return value and a result flag, so nothing is silently truncated.
- Making `Impl` a pimpl limits the public header's dependency on CUDA-specific types to `cudaStream_t` and `float2`.

## Decisions made

### Public aggregate fields also carry a trailing `_`

The naming convention in `CONTRIBUTING.md` is applied as is. `DeviceProbeResult`, `DictionaryTable`, `ReferenceConfig`, `SceneSpec`, and `BenchmarkConfig` are already implemented under the same convention, and changing the convention only for public aggregates would mix two conventions within the same repository. Call sites become somewhat more verbose, but having a single convention takes priority.

### Validation returns a Status, with the reason received through an optional out parameter

`validate_image_view()` and `DetectorConfig::validate()` return a `Status` and store the reason for failure into a `std::string*`. Passing `nullptr` is allowed, in which case no string is assembled. Validation may be called every frame, so the structure avoids any allocation on the success path.

### Image failures get a dedicated Status

`kInvalidImage` is provided separately from `kInvalidArgument`. An invalid image view indicates a problem in the caller's input path, and it calls for a different response than a mistake in configuration or an index.

## Open questions

- Whether to use `float2` in the public header, or to define an equivalent of our own `Point2f`. This will be decided when `DeviceDetections` is implemented.
- Whether to provide an API besides `download()` that retrieves only partial results.
- Whether simultaneous matching against multiple dictionaries belongs in the initial scope.
- Whether to include the OpenCV adapter in the same library or in a separate target. Same as the open question in the [architecture](../architecture.md).
- Whether to ignore or reject unsupported parameters of `cv::aruco::DetectorParameters`.

## See also

- [Architecture](../architecture.md)
- [Detection pipeline design](detector-pipeline.md)
- [Implementation plan](../implementation-plan.md)
- [Code provenance record](../code-provenance.md)
