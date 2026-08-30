// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_HYBRID_DEVICE_IMAGE_HPP
#define ARUCO3CUDA_HYBRID_DEVICE_IMAGE_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"

/// Helper that places a host image into device memory.
///
/// Purpose:
///     The detector takes an ImageViewU8 pointing at memory the device can
///     read; it never performs the transfer itself. If every caller wrote its
///     own cudaMalloc and cudaMemcpy2D, the same code would be scattered
///     around and more places would get the pitch handling wrong.
namespace aruco3cuda::hybrid {

/// Device buffer holding a single image.
///
/// Ownership: owns the device memory it allocates and frees it on destruction.
/// Synchronization: upload() performs the synchronous transfer, so the copy is
///           complete once it returns. A single instance must not be used from
///           several threads at the same time.
///
/// Example input: reserve(1280, 720) once, then upload() every frame
/// Example output: view() keeps pointing at the same device pointer; the
///           allocation happens only once
class DeviceImage {
public:
    DeviceImage();
    ~DeviceImage();
    DeviceImage(const DeviceImage&) = delete;
    DeviceImage& operator=(const DeviceImage&) = delete;
    DeviceImage(DeviceImage&&) noexcept;
    DeviceImage& operator=(DeviceImage&&) noexcept;

    /// Allocates the device buffer. Does nothing if the existing one is
    /// already large enough.
    ///
    /// @param width_px Image width. At least 1.
    /// @param height_px Image height. At least 1.
    /// @param out_message Receives the reason on failure. May be nullptr.
    /// @return kOk, or kInvalidArgument or kCudaError.
    ///
    /// Ownership: this instance owns the region it allocates.
    /// Synchronization: calls cudaMalloc. To avoid a per-frame allocation, call
    ///           it exactly once at setup time in benchmarks and in real-time
    ///           processing.
    ///
    /// Example input: 1280 and 720
    /// Example output: kOk. view() becomes valid
    Status reserve(int width_px, int height_px, std::string* out_message = nullptr);

    /// Allocates the region in the given memory space.
    ///
    /// The space changes both how the region is allocated and what upload()
    /// means.
    ///
    /// | space | allocation | upload() |
    /// | --- | --- | --- |
    /// | kDevice | cudaMallocPitch | transfers from host to device |
    /// | kManaged | cudaMallocManaged | host and device see the same region; no transfer happens |
    ///
    /// **Whether the source is page-locked is out of scope here.** The pointer
    /// passed in is read as is. Preparing a page-locked input is the caller's
    /// responsibility. If this class staged the data through an intermediate
    /// buffer, we would be measuring the cost of that extra copy rather than
    /// the effect of the input buffer's memory space.
    ///
    /// @param space Space to allocate in. kDevice, kHostPageable and
    ///              kHostPinned all allocate on the device side; only kManaged
    ///              is treated differently.
    /// @param width_px Width. At least 1.
    /// @param height_px Height. At least 1.
    /// @param out_message Receives the reason on failure. May be nullptr.
    /// @return kOk, or kInvalidArgument or kCudaError.
    ///
    /// Ownership: this instance owns the region it allocates.
    /// Synchronization: allocation only; it introduces no synchronization point.
    ///
    /// Example input: MemorySpace::kManaged, 1280, 720
    /// Example output: kOk. The view()'s space_ becomes kManaged
    Status reserve(MemorySpace space, int width_px, int height_px,
                   std::string* out_message = nullptr);

    /// Transfers an 8-bit grayscale host image to the device.
    ///
    /// @param data Pointer to the start of the host image. Must not be nullptr.
    /// @param width_px Image width. Must not exceed the reserved width.
    /// @param height_px Image height. Must not exceed the reserved height.
    /// @param source_pitch_bytes Bytes per row on the host side.
    /// @param out_message Receives the reason on failure. May be nullptr.
    /// @return kOk, or kInvalidArgument or kCudaError.
    ///
    /// Ownership: the caller owns the host memory passed in. It is not
    ///           referenced after the transfer.
    /// Synchronization: uses the synchronous cudaMemcpy2D. The transfer is
    ///           complete once it returns.
    ///
    /// Example input: a cv::Mat's data, cols, rows and step
    /// Example output: kOk. The region view() points at is updated
    Status upload(const std::uint8_t* data, int width_px, int height_px,
                  std::size_t source_pitch_bytes, std::string* out_message = nullptr);

    /// Returns a view of the current contents.
    ///
    /// @return A view of the device memory. Before reserve(), data_ is nullptr.
    ///
    /// Ownership: this instance owns the referenced region.
    /// Synchronization: none.
    ///
    /// Example input: an instance that has been reserved and uploaded to
    /// Example output: a view whose space_ is kDevice
    const ImageViewU8& view() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace aruco3cuda::hybrid

#endif  // ARUCO3CUDA_HYBRID_DEVICE_IMAGE_HPP
