// SPDX-License-Identifier: Apache-2.0
// Isolated GNU-linker wrappers exercise failure recovery without invalid device
// accesses or production test hooks. Link this file only into its own executable.
#include "aruco3cuda/detector.hpp"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "aruco3cuda/dictionary.hpp"
#include "detection_emit.hpp"
#include "test_detector_probe.hpp"

namespace {
struct FaultState {
    int malloc_countdown_ = 0;
    int event_countdown_ = 0;
    int launch_check_countdown_ = 0;
    int malloc_calls_ = 0;
    int malloc_failures_ = 0;
    int event_failures_ = 0;
    int launch_check_failures_ = 0;
    cudaEvent_t partial_tail_ = nullptr;
    cudaError_t partial_tail_status_ = cudaSuccess;
    int copy_countdown_ = 0;
    int copy_failures_ = 0;
    cudaMemcpyKind copy_kind_ = cudaMemcpyHostToDevice;
    int* copy_sink_ = nullptr;
    cudaEvent_t copy_tail_ = nullptr;
    cudaError_t copy_tail_status_ = cudaSuccess;
    bool copy_delay_ok_ = false;
    std::size_t copy_bytes_ = 0;
};
thread_local FaultState faults;

bool trigger(int* countdown) {
    if (*countdown <= 0) {
        return false;
    }
    --*countdown;
    return *countdown == 0;
}

class FaultScope {
public:
    FaultScope() { faults = FaultState{}; }
    ~FaultScope() { faults = FaultState{}; }
    FaultScope(const FaultScope&) = delete;
    FaultScope& operator=(const FaultScope&) = delete;
};

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    ~DeviceBuffer() {
        if (this->data_ != nullptr) {
            static_cast<void>(cudaFree(this->data_));
        }
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    cudaError_t allocate(std::size_t bytes) { return cudaMalloc(&this->data_, bytes); }
    void* data() const { return this->data_; }

private:
    void* data_ = nullptr;
};

// A failing assertion must not release borrowed pinned storage while the
// deliberately delayed transfer is still queued in a regressed implementation.
class PinnedBuffer {
public:
    PinnedBuffer() = default;
    ~PinnedBuffer() {
        if (this->data_ != nullptr) {
            static_cast<void>(cudaDeviceSynchronize());
            static_cast<void>(cudaFreeHost(this->data_));
        }
    }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    cudaError_t allocate(std::size_t bytes) { return cudaMallocHost(&this->data_, bytes); }
    void* data() const { return this->data_; }

private:
    void* data_ = nullptr;
};

class StreamDrain {
public:
    explicit StreamDrain(cudaStream_t stream) : stream_(stream) {}
    ~StreamDrain() { static_cast<void>(cudaStreamSynchronize(this->stream_)); }
    StreamDrain(const StreamDrain&) = delete;
    StreamDrain& operator=(const StreamDrain&) = delete;

private:
    cudaStream_t stream_ = nullptr;
};

class Stream {
public:
    Stream() = default;
    ~Stream() {
        if (this->stream_ != nullptr) {
            static_cast<void>(cudaStreamDestroy(this->stream_));
        }
    }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    cudaError_t create() {
        return cudaStreamCreateWithFlags(&this->stream_, cudaStreamNonBlocking);
    }
    cudaStream_t get() const { return this->stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

class Event {
public:
    Event() = default;
    ~Event() {
        if (this->event_ != nullptr) {
            static_cast<void>(cudaEventDestroy(this->event_));
        }
    }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    cudaError_t create() { return cudaEventCreateWithFlags(&this->event_, cudaEventDisableTiming); }
    cudaEvent_t get() const { return this->event_; }

private:
    cudaEvent_t event_ = nullptr;
};

bool has_cuda_device() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

aruco3cuda::DetectorConfig config_for(int side) {
    aruco3cuda::DetectorConfig config;
    config.max_width_px_ = side;
    config.max_height_px_ = side;
    config.max_candidates_ = 8;
    config.max_markers_ = 8;
    config.use_aruco3_detection_ = false;
    config.corner_refine_method_ = aruco3cuda::CornerRefineMethod::kNone;
    return config;
}

aruco3cuda::ImageViewU8 blank_view(const DeviceBuffer& pixels) {
    aruco3cuda::ImageViewU8 view;
    view.data_ = static_cast<const std::uint8_t*>(pixels.data());
    view.width_px_ = 128;
    view.height_px_ = 128;
    view.pitch_bytes_ = 128U;
    view.space_ = aruco3cuda::MemorySpace::kDevice;
    return view;
}

using aruco3cuda::Status;
constexpr long long kDelayCycles = 1000000000LL;
}  // namespace

extern "C" cudaError_t CUDARTAPI __real_cudaMalloc(void**, std::size_t);
extern "C" cudaError_t CUDARTAPI __real_cudaEventRecord(cudaEvent_t, cudaStream_t);
extern "C" cudaError_t CUDARTAPI __real_cudaGetLastError();
extern "C" cudaError_t CUDARTAPI __real_cudaMemcpyAsync(void*, const void*, std::size_t,
                                                        cudaMemcpyKind, cudaStream_t);

extern "C" cudaError_t CUDARTAPI __wrap_cudaMalloc(void** pointer, std::size_t bytes) {
    ++faults.malloc_calls_;
    if (trigger(&faults.malloc_countdown_)) {
        ++faults.malloc_failures_;
        return cudaErrorMemoryAllocation;
    }
    return __real_cudaMalloc(pointer, bytes);
}

extern "C" cudaError_t CUDARTAPI __wrap_cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    if (trigger(&faults.event_countdown_)) {
        ++faults.event_failures_;
        return cudaErrorInvalidResourceHandle;
    }
    return __real_cudaEventRecord(event, stream);
}

extern "C" cudaError_t CUDARTAPI __wrap_cudaGetLastError() {
    const cudaError_t actual = __real_cudaGetLastError();
    if (actual != cudaSuccess) {
        return actual;
    }
    // NVCC 11 and 13 use different launch entry points and discard their return
    // values. Inject at the common checked boundary after real work was queued.
    if (trigger(&faults.launch_check_countdown_)) {
        ++faults.launch_check_failures_;
        if (faults.partial_tail_ != nullptr) {
            // This marker precedes the detector's failure-recovery completion
            // record, so ordering it is part of the submitted-work contract.
            faults.partial_tail_status_ = __real_cudaEventRecord(faults.partial_tail_, nullptr);
        }
        return cudaErrorInvalidConfiguration;
    }
    return actual;
}

extern "C" cudaError_t CUDARTAPI __wrap_cudaMemcpyAsync(void* destination, const void* source,
                                                        std::size_t bytes, cudaMemcpyKind kind,
                                                        cudaStream_t stream) {
    if (kind != faults.copy_kind_ || !trigger(&faults.copy_countdown_)) {
        return __real_cudaMemcpyAsync(destination, source, bytes, kind, stream);
    }
    faults.copy_bytes_ = bytes;
    if (kind == cudaMemcpyHostToDevice) {
        faults.copy_delay_ok_ =
                aruco3cuda::test::enqueue_spin(kDelayCycles, faults.copy_sink_, stream);
    }
    const cudaError_t actual = __real_cudaMemcpyAsync(destination, source, bytes, kind, stream);
    if (actual != cudaSuccess) {
        return actual;
    }
    if (kind == cudaMemcpyDeviceToHost) {
        // Pageable D2H may complete inside cudaMemcpyAsync. A delayed marker
        // queued afterward still detects the required error-path stream drain.
        faults.copy_delay_ok_ =
                aruco3cuda::test::enqueue_spin(kDelayCycles, faults.copy_sink_, stream);
    }
    faults.copy_tail_status_ = __real_cudaEventRecord(faults.copy_tail_, stream);
    ++faults.copy_failures_;
    return cudaErrorUnknown;
}

TEST(DetectorFaultDeliberateError, failed_frame_growth_after_dictionary_growth_invalidates_state) {
    if (!has_cuda_device()) {
        GTEST_SKIP();
    }
    const FaultScope fault_scope;
    DeviceBuffer pixels;
    ASSERT_EQ(pixels.allocate(128U * 128U), cudaSuccess);
    ASSERT_EQ(cudaMemset(pixels.data(), 0, 128U * 128U), cudaSuccess);
    const auto* dictionary = aruco3cuda::find_builtin_dictionary("DICT_ARUCO_MIP_36h12");
    ASSERT_NE(dictionary, nullptr);
    aruco3cuda::DictionaryTable small = *dictionary;
    small.code_count_ = 1;
    aruco3cuda::Detector detector;
    std::string message;
    ASSERT_EQ(detector.initialize(small, config_for(128), &message), Status::kOk) << message;
    ASSERT_EQ(detector.detect_async(blank_view(pixels), nullptr, &message), Status::kOk) << message;
    aruco3cuda::HostDetections output;
    ASSERT_EQ(detector.download(&output, nullptr, &message), Status::kOk) << message;
    aruco3cuda::DeviceDetections device;
    ASSERT_EQ(detector.device_detections(&device), Status::kOk);

    const int calls_before = faults.malloc_calls_;
    faults.malloc_countdown_ = 2;
    EXPECT_EQ(detector.initialize(*dictionary, config_for(256), &message), Status::kCudaError);
    ASSERT_EQ(faults.malloc_failures_, 1);
    EXPECT_EQ(faults.malloc_calls_ - calls_before, 2);
    EXPECT_FALSE(detector.initialized());
    EXPECT_EQ(detector.device_detections(&device), Status::kNotInitialized);
    EXPECT_EQ(detector.detect_async(blank_view(pixels), nullptr, &message),
              Status::kNotInitialized);
    EXPECT_EQ(detector.download(&output, nullptr, &message), Status::kNotInitialized);

    ASSERT_EQ(detector.initialize(*dictionary, config_for(256), &message), Status::kOk) << message;
    ASSERT_EQ(detector.detect_async(blank_view(pixels), nullptr, &message), Status::kOk) << message;
    ASSERT_EQ(detector.download(&output, nullptr, &message), Status::kOk) << message;
    EXPECT_TRUE(output.ids_.empty());
}

TEST(DetectorFaultDeliberateError, completion_record_failure_drains_and_invalidates_submission) {
    if (!has_cuda_device()) {
        GTEST_SKIP();
    }
    const FaultScope fault_scope;
    DeviceBuffer pixels;
    DeviceBuffer sink;
    Stream stream;
    ASSERT_EQ(pixels.allocate(128U * 128U), cudaSuccess);
    ASSERT_EQ(sink.allocate(sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMemset(pixels.data(), 0, 128U * 128U), cudaSuccess);
    ASSERT_EQ(stream.create(), cudaSuccess);
    const auto* dictionary = aruco3cuda::find_builtin_dictionary("DICT_ARUCO_MIP_36h12");
    ASSERT_NE(dictionary, nullptr);
    aruco3cuda::Detector detector;
    std::string message;
    ASSERT_EQ(detector.initialize(*dictionary, config_for(128), &message), Status::kOk) << message;
    ASSERT_EQ(detector.detect_async(blank_view(pixels), stream.get(), &message), Status::kOk)
            << message;
    aruco3cuda::HostDetections output;
    ASSERT_EQ(detector.download(&output, stream.get(), &message), Status::kOk) << message;

    ASSERT_TRUE(aruco3cuda::test::enqueue_spin(kDelayCycles, static_cast<int*>(sink.data()),
                                               stream.get()));
    faults.event_countdown_ = 1;
    EXPECT_EQ(detector.detect_async(blank_view(pixels), stream.get(), &message),
              Status::kCudaError);
    ASSERT_EQ(faults.event_failures_, 1);
    EXPECT_EQ(cudaStreamQuery(stream.get()), cudaSuccess);
    EXPECT_FALSE(detector.initialized());
    aruco3cuda::DeviceDetections device;
    EXPECT_EQ(detector.device_detections(&device), Status::kNotInitialized);
    EXPECT_EQ(detector.download(&output, stream.get(), &message), Status::kNotInitialized);

    ASSERT_EQ(detector.initialize(*dictionary, config_for(128), &message), Status::kOk) << message;
    ASSERT_EQ(detector.detect_async(blank_view(pixels), stream.get(), &message), Status::kOk)
            << message;
    ASSERT_EQ(detector.download(&output, stream.get(), &message), Status::kOk) << message;
    EXPECT_TRUE(output.ids_.empty());
}

TEST(DetectorFaultDeliberateError, partial_direct_dispatch_is_ordered_before_next_stream) {
    if (!has_cuda_device()) {
        GTEST_SKIP();
    }
    const FaultScope fault_scope;
    DeviceBuffer pixels;
    DeviceBuffer sink;
    Stream next_stream;
    Event failed_submission_tail;
    ASSERT_EQ(pixels.allocate(128U * 128U), cudaSuccess);
    ASSERT_EQ(sink.allocate(sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMemset(pixels.data(), 0, 128U * 128U), cudaSuccess);
    ASSERT_EQ(next_stream.create(), cudaSuccess);
    ASSERT_EQ(failed_submission_tail.create(), cudaSuccess);
    const auto* dictionary = aruco3cuda::find_builtin_dictionary("DICT_ARUCO_MIP_36h12");
    ASSERT_NE(dictionary, nullptr);
    aruco3cuda::Detector detector;
    std::string message;
    ASSERT_EQ(detector.initialize(*dictionary, config_for(128), &message), Status::kOk) << message;
    ASSERT_EQ(detector.detect_async(blank_view(pixels), nullptr, &message), Status::kOk) << message;
    aruco3cuda::HostDetections output;
    ASSERT_EQ(detector.download(&output, nullptr, &message), Status::kOk) << message;

    ASSERT_TRUE(
            aruco3cuda::test::enqueue_spin(kDelayCycles, static_cast<int*>(sink.data()), nullptr));
    faults.partial_tail_ = failed_submission_tail.get();
    faults.launch_check_countdown_ = 3;
    EXPECT_EQ(detector.detect_async(blank_view(pixels), nullptr, &message), Status::kCudaError);
    ASSERT_EQ(faults.launch_check_failures_, 1);
    ASSERT_EQ(faults.partial_tail_status_, cudaSuccess);
    aruco3cuda::DeviceDetections device;
    EXPECT_EQ(detector.device_detections(&device), Status::kNotInitialized);
    ASSERT_EQ(detector.detect_async(blank_view(pixels), next_stream.get(), &message), Status::kOk)
            << message;
    ASSERT_EQ(detector.download(&output, next_stream.get(), &message), Status::kOk) << message;
    EXPECT_EQ(cudaEventQuery(failed_submission_tail.get()), cudaSuccess);
    EXPECT_TRUE(output.ids_.empty());
}

TEST(DetectorFaultDeliberateError, failed_dictionary_upload_drains_borrowed_pinned_codes) {
    if (!has_cuda_device()) {
        GTEST_SKIP();
    }
    const FaultScope fault_scope;
    const auto* dictionary = aruco3cuda::find_builtin_dictionary("DICT_ARUCO_MIP_36h12");
    ASSERT_NE(dictionary, nullptr);
    const std::size_t bytes =
            static_cast<std::size_t>(dictionary->code_count_) * 4U * sizeof(aruco3cuda::MarkerCode);
    PinnedBuffer codes;
    DeviceBuffer sink;
    Event tail;
    ASSERT_EQ(codes.allocate(bytes), cudaSuccess);
    ASSERT_EQ(sink.allocate(sizeof(int)), cudaSuccess);
    ASSERT_EQ(tail.create(), cudaSuccess);
    std::memcpy(codes.data(), dictionary->codes_, bytes);
    aruco3cuda::DictionaryTable borrowed = *dictionary;
    borrowed.codes_ = static_cast<const aruco3cuda::MarkerCode*>(codes.data());
    aruco3cuda::Detector detector;
    const StreamDrain cleanup(nullptr);
    faults.copy_kind_ = cudaMemcpyHostToDevice;
    faults.copy_sink_ = static_cast<int*>(sink.data());
    faults.copy_tail_ = tail.get();
    faults.copy_countdown_ = 1;
    std::string message;
    EXPECT_EQ(detector.initialize(borrowed, config_for(128), &message), Status::kCudaError);
    ASSERT_EQ(faults.copy_failures_, 1);
    EXPECT_EQ(faults.copy_bytes_, bytes);
    ASSERT_TRUE(faults.copy_delay_ok_);
    ASSERT_EQ(faults.copy_tail_status_, cudaSuccess);
    EXPECT_EQ(cudaEventQuery(tail.get()), cudaSuccess);
    EXPECT_FALSE(detector.initialized());
    aruco3cuda::DeviceDetections device;
    EXPECT_EQ(detector.device_detections(&device), Status::kNotInitialized);
    const std::string diagnostic = aruco3cuda::last_cuda_error_message();
    EXPECT_NE(diagnostic.find("dictionary_match.upload_dictionary"), std::string::npos);
    EXPECT_NE(diagnostic.find("cudaErrorUnknown"), std::string::npos);
}

TEST(DetectorFaultDeliberateError, failed_count_copies_drain_stack_storage_and_preserve_output) {
    if (!has_cuda_device()) {
        GTEST_SKIP();
    }
    const FaultScope fault_scope;
    DeviceBuffer counters;
    DeviceBuffer sink;
    Stream stream;
    Event tail;
    ASSERT_EQ(counters.allocate(2U * sizeof(std::int32_t)), cudaSuccess);
    ASSERT_EQ(sink.allocate(sizeof(int)), cudaSuccess);
    ASSERT_EQ(stream.create(), cudaSuccess);
    ASSERT_EQ(tail.create(), cudaSuccess);
    const StreamDrain cleanup(stream.get());
    const std::int32_t values[] = {1, 1};
    ASSERT_EQ(cudaMemcpy(counters.data(), values, sizeof(values), cudaMemcpyHostToDevice),
              cudaSuccess);
    aruco3cuda::DeviceDetections device;
    device.count_ = static_cast<std::int32_t*>(counters.data());
    device.accepted_total_ = device.count_ + 1;
    for (const int fail_at : {1, 2}) {
        faults.copy_kind_ = cudaMemcpyDeviceToHost;
        faults.copy_sink_ = static_cast<int*>(sink.data());
        faults.copy_tail_ = tail.get();
        faults.copy_failures_ = 0;
        faults.copy_countdown_ = fail_at;
        int count = -77;
        EXPECT_EQ(aruco3cuda::detail::read_detection_count(device, &count, stream.get()),
                  Status::kCudaError);
        ASSERT_EQ(faults.copy_failures_, 1);
        ASSERT_TRUE(faults.copy_delay_ok_);
        ASSERT_EQ(faults.copy_tail_status_, cudaSuccess);
        EXPECT_EQ(cudaEventQuery(tail.get()), cudaSuccess);
        EXPECT_EQ(count, -77);
        const std::string diagnostic = aruco3cuda::last_cuda_error_message();
        EXPECT_NE(diagnostic.find(fail_at == 1 ? "detection.read_count" : "detection.read_total"),
                  std::string::npos);
        EXPECT_NE(diagnostic.find("cudaErrorUnknown"), std::string::npos);
        ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
    }
}

TEST(DetectorFaultDeliberateError, failed_download_copies_preserve_scalar_and_vector_outputs) {
    if (!has_cuda_device()) {
        GTEST_SKIP();
    }
    const FaultScope fault_scope;
    DeviceBuffer pixels;
    DeviceBuffer sink;
    Stream stream;
    Event tail;
    ASSERT_EQ(pixels.allocate(128U * 128U), cudaSuccess);
    ASSERT_EQ(sink.allocate(sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMemset(pixels.data(), 0, 128U * 128U), cudaSuccess);
    ASSERT_EQ(stream.create(), cudaSuccess);
    ASSERT_EQ(tail.create(), cudaSuccess);
    const auto* dictionary = aruco3cuda::find_builtin_dictionary("DICT_ARUCO_MIP_36h12");
    ASSERT_NE(dictionary, nullptr);
    aruco3cuda::Detector detector;
    const StreamDrain cleanup(stream.get());
    std::string message;
    ASSERT_EQ(detector.initialize(*dictionary, config_for(128), &message), Status::kOk) << message;
    ASSERT_EQ(detector.detect_async(blank_view(pixels), stream.get(), &message), Status::kOk)
            << message;
    ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
    aruco3cuda::DeviceDetections device;
    ASSERT_EQ(detector.device_detections(&device), Status::kOk);
    const std::int32_t one = 1;
    const std::int32_t id = 17;
    const std::int32_t rotation = 2;
    ASSERT_EQ(cudaMemcpy(device.count_, &one, sizeof(one), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device.accepted_total_, &one, sizeof(one), cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device.ids_, &id, sizeof(id), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device.rotations_, &rotation, sizeof(rotation), cudaMemcpyHostToDevice),
              cudaSuccess);
    for (int corner = 0; corner < 4; ++corner) {
        const float coordinate = static_cast<float>(corner + 10);
        ASSERT_EQ(cudaMemcpy(device.corner_x_ + corner * device.capacity_, &coordinate,
                             sizeof(coordinate), cudaMemcpyHostToDevice),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpy(device.corner_y_ + corner * device.capacity_, &coordinate,
                             sizeof(coordinate), cudaMemcpyHostToDevice),
                  cudaSuccess);
    }
    // Diagnostic scalar, second count scalar, then the first corner-vector copy.
    for (const int fail_at : {1, 3, 6}) {
        faults.copy_kind_ = cudaMemcpyDeviceToHost;
        faults.copy_sink_ = static_cast<int*>(sink.data());
        faults.copy_tail_ = tail.get();
        faults.copy_failures_ = 0;
        faults.copy_countdown_ = fail_at;
        aruco3cuda::HostDetections output;
        output.ids_ = {-100};
        output.rotations_ = {3};
        output.corners_ = std::vector<float>(8U, -2.0F);
        output.accepted_total_ = 123;
        output.marker_overflow_ = true;
        const aruco3cuda::HostDetections before = output;
        EXPECT_EQ(detector.download(&output, stream.get(), &message), Status::kCudaError);
        ASSERT_EQ(faults.copy_failures_, 1);
        ASSERT_TRUE(faults.copy_delay_ok_);
        ASSERT_EQ(faults.copy_tail_status_, cudaSuccess);
        EXPECT_EQ(cudaEventQuery(tail.get()), cudaSuccess);
        EXPECT_EQ(output.ids_, before.ids_);
        EXPECT_EQ(output.rotations_, before.rotations_);
        EXPECT_EQ(output.corners_, before.corners_);
        EXPECT_EQ(output.accepted_total_, before.accepted_total_);
        EXPECT_EQ(output.marker_overflow_, before.marker_overflow_);
        const std::string diagnostic = aruco3cuda::last_cuda_error_message();
        const char* stage = fail_at == 1   ? "detector.download.failure_bits"
                            : fail_at == 3 ? "detection.read_total"
                                           : "detector.download.corner_x";
        EXPECT_NE(diagnostic.find(stage), std::string::npos);
        EXPECT_NE(diagnostic.find("cudaErrorUnknown"), std::string::npos);
        ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
    }
    aruco3cuda::HostDetections recovered;
    ASSERT_EQ(detector.download(&recovered, stream.get(), &message), Status::kOk) << message;
    EXPECT_EQ(recovered.ids_, std::vector<std::int32_t>({id}));
    EXPECT_EQ(recovered.rotations_, std::vector<std::int32_t>({rotation}));
    EXPECT_EQ(recovered.accepted_total_, one);
}
