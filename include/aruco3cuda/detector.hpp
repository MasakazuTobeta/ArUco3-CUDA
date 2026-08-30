// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_DETECTOR_HPP
#define ARUCO3CUDA_DETECTOR_HPP

#include <cuda_runtime_api.h>

#include <memory>
#include <string>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/detections.hpp"
#include "aruco3cuda/dictionary.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"
#include "aruco3cuda/workspace.hpp"

namespace aruco3cuda {

/// CUDA implementation of the ArUco3 detection strategy.
///
/// Purpose:
///   Issues everything from preprocessing through corner refinement onto a single
///   stream and returns the results while they stay on the device. If the stages that
///   follow run on the same device, there is no need to bring the data back to the host.
///
/// Only two combinations of settings are supported.
///
/// | use_aruco3_detection_ | corner_refine_method_ | Corner coordinates |
/// | --- | --- | --- |
/// | true | kSubpix | Full resolution (recovered by climbing back up the levels) |
/// | false | kNone | Full resolution (nothing is downscaled) |
///
/// initialize() rejects the other two combinations. With ArUco3 enabled and refinement
/// off, the corners would stay in downscaled coordinates; with ArUco3 disabled there is
/// only one level, so refinement would never run even if it were enabled.
///
/// Ownership:
///   This class owns the memory behind the workspace and the DeviceDetections. The
///   caller owns the input image and the stream. The DictionaryTable is copied to the
///   device inside initialize(), so it may be destroyed once that call returns.
///   **This ownership applies to all public member functions.**
/// Synchronization:
///   Only initialize() and download() involve host synchronization. detect_async()
///   merely issues kernels and does not synchronize. That said, on a frame whose input
///   dimensions or pitch differ from the previous one, the stream is synchronized once
///   before the buffer layout is rebuilt.
///   A single instance must not be used from several threads at once.
///
/// Folding the issue sequence:
///   When an explicit stream is given, one frame worth of the issue sequence is
///   captured as a CUDA Graph, so every subsequent frame costs a single launch. A
///   captured sequence bakes in the kernel arguments, so it must always be discarded
///   and recaptured when any of the following happens: initialize() is called again;
///   the input dimensions, pitch, or pointer change; the stream changes. Changing a
///   setting requires calling initialize() again, so there is no path by which a baked
///   value silently goes stale.
///
/// Example input: initialize with DICT_ARUCO_MIP_36h12 and the default settings
/// Example output: kOk; detect_async() may be called from then on
class Detector {
public:
    Detector();
    ~Detector();
    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;
    Detector(Detector&&) noexcept;
    Detector& operator=(Detector&&) noexcept;

    /// Allocates the workspace and transfers the dictionary to the device.
    ///
    /// The workspace is allocated for the worst case implied by the configured limits
    /// (max_width_px_, max_height_px_, max_candidates_, max_markers_). detect_async()
    /// never adds an allocation afterwards.
    ///
    /// @param dictionary The dictionary used for matching. Its contents are copied to
    ///                   the device.
    /// @param config Detection settings. They are passed through validate().
    /// @param out_message On failure, receives the reason. May be nullptr.
    /// @return kOk. kUnsupportedDictionary if the dictionary is invalid,
    ///         kInvalidConfig if the settings are invalid or if allocation failed, and
    ///         kCudaError if a CUDA API call failed.
    ///
    /// Synchronization: synchronizes the whole device at the end, so that the dictionary
    ///                  transfer is complete as seen from any stream afterwards.
    ///
    /// Example input: the DICT_ARUCO_MIP_36h12 table and the default settings
    /// Example output: kOk; initialized() becomes true
    Status initialize(const DictionaryTable& dictionary, const DetectorConfig& config,
                      std::string* out_message = nullptr);

    /// Issues a detection onto the stream.
    ///
    /// **A return value of kOk does not mean nothing was truncated at a limit.**
    /// Whether truncation happened can be seen by comparing accepted_total_ and count_
    /// on the device. To learn it from the host, call download().
    ///
    /// @param image The input image. It must live in device or managed space. The
    ///              caller owns the memory, and it must stay valid until the kernels
    ///              have completed.
    /// @param stream The stream the kernels are issued on. nullptr means the default
    ///               stream. **Passing an explicit stream folds the issue sequence into
    ///               a CUDA Graph, so every subsequent frame is issued in a single
    ///               launch.** CUDA does not permit capture on the default stream, so
    ///               that takes the path issuing one step at a time.
    /// @param out_message On failure, receives the reason. May be nullptr.
    /// @return kOk. kNotInitialized before initialize(), kInvalidImage if the image is
    ///         invalid, kInvalidArgument if its dimensions exceed the limits,
    ///         kInvalidConfig if allocation failed, and kCudaError if a kernel launch
    ///         failed.
    ///
    /// Synchronization: issues the kernels asynchronously on the stream. Only on a frame
    ///                  whose input dimensions or pitch differ from the previous one is
    ///                  cudaStreamSynchronize called once, before the layout is rebuilt.
    ///
    /// Example input: a 1280x720 device image
    /// Example output: kOk; the results are written into the memory device_detections()
    ///                 points at
    Status detect_async(const ImageViewU8& image, cudaStream_t stream,
                        std::string* out_message = nullptr);

    /// Copies out the device-resident results of the most recent detect_async().
    ///
    /// No synchronization takes place. The contents of the memory pointed at are not
    /// final until the kernels already issued have completed. Call this again after
    /// every detect_async(): when the input dimensions change, the layout changes and
    /// so do the pointers.
    ///
    /// @param out On success, receives the references. The caller owns the storage.
    /// @return kOk. kInvalidArgument if out is nullptr, and kNotInitialized if
    ///         detect_async() has never been called.
    ///
    /// Synchronization: **does not synchronize.** Calls no CUDA API.
    ///
    /// Example input: called right after detect_async()
    /// Example output: kOk; out->count_ points at the detection count on the device
    Status device_detections(DeviceDetections* out) const;

    /// Waits for the stream to complete and obtains the results on the host.
    ///
    /// @param out On success, receives the results. The caller owns the storage.
    /// @param stream The stream to synchronize. Use the same one passed to detect_async().
    /// @param out_message On failure, receives the reason. May be nullptr.
    /// @return kOk. kMarkerOverflow if the results were truncated at the limit (out is
    ///         still filled in). kInvalidArgument if out is nullptr, kNotInitialized if
    ///         detect_async() has never been called, and kCudaError if a CUDA API call
    ///         failed.
    ///
    /// Synchronization: **synchronizes the stream.** This is the only place in this
    ///                  class that synchronizes.
    ///
    /// Example input: a stream that wrote two detections
    /// Example output: kOk; out->ids_ has 2 elements and out->corners_ has 16
    Status download(HostDetections* out, cudaStream_t stream, std::string* out_message = nullptr);

    /// Returns how the device workspace is being used.
    ///
    /// Confirming that allocation_count_ stays at 1 after initialize() shows that
    /// nothing is being allocated per frame.
    ///
    /// @return The statistics. All zero before initialize().
    ///
    /// Ownership: the return value refers to memory owned by this instance.
    /// Synchronization: host only, with no synchronization point.
    ///
    /// Example input: an instance on which initialize() has been called
    /// Example output: allocation_count_ = 1
    const WorkspaceStatistics& workspace_statistics() const;

    /// Whether initialize() has succeeded.
    ///
    /// @return true if it has.
    ///
    /// Ownership: holds no resource.
    /// Synchronization: host only, with no synchronization point.
    ///
    /// Example input: an instance on which initialize() has been called
    /// Example output: true
    bool initialized() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace aruco3cuda

#endif  // ARUCO3CUDA_DETECTOR_HPP
