/* SPDX-License-Identifier: Apache-2.0 */
/*
 * Purpose:
 *   A C ABI over the C++ detector, so that the library can be driven from any
 *   language with a foreign function interface. The Python package under
 *   python/aruco3cuda is built entirely on this header through ctypes, with no
 *   compiled extension module and therefore no dependency on a particular
 *   Python version.
 *
 * Naming:
 *   CONTRIBUTING.md asks for kPascalCase constants. That convention belongs to
 *   C++, where constants live in a namespace. Enumerators in a C header are
 *   global identifiers, so they take the ARUCO3CUDA_ prefix instead. This is
 *   the only place in the repository that deviates, and it deviates because a
 *   global named kOk would be a defect rather than a style choice.
 *
 * Error handling:
 *   Every call that can fail returns Aruco3CudaStatus and never throws. The C++
 *   side is wrapped so that no exception crosses this boundary, because
 *   unwinding through a C frame is undefined behavior.
 *
 * Ownership:
 *   The caller owns every buffer it passes in and every buffer it asks to be
 *   filled. The only object this library allocates is Aruco3CudaDetector, which
 *   is released with aruco3cuda_detector_destroy. **This applies to every
 *   function declared here.**
 *
 * Synchronization:
 *   Mirrors the C++ API. aruco3cuda_detector_initialize synchronizes the whole
 *   device, aruco3cuda_detector_download synchronizes the stream, and
 *   aruco3cuda_detector_detect only issues work. **This applies to every
 *   function declared here**, and each function that departs from it says so.
 */
#ifndef ARUCO3CUDA_C_ARUCO3CUDA_H
#define ARUCO3CUDA_C_ARUCO3CUDA_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define ARUCO3CUDA_C_API
#else
#define ARUCO3CUDA_C_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Result state. The values match aruco3cuda::Status one for one, and a
   static_assert in the implementation keeps them from drifting apart. */
typedef enum Aruco3CudaStatus {
    ARUCO3CUDA_STATUS_OK = 0,
    ARUCO3CUDA_STATUS_INVALID_ARGUMENT = 1,
    ARUCO3CUDA_STATUS_INVALID_IMAGE = 2,
    ARUCO3CUDA_STATUS_INVALID_CONFIG = 3,
    ARUCO3CUDA_STATUS_UNSUPPORTED_DICTIONARY = 4,
    ARUCO3CUDA_STATUS_CANDIDATE_OVERFLOW = 5,
    ARUCO3CUDA_STATUS_MARKER_OVERFLOW = 6,
    ARUCO3CUDA_STATUS_CUDA_ERROR = 7,
    ARUCO3CUDA_STATUS_NOT_INITIALIZED = 8,
    /* Raised by this layer alone, never by the C++ core. The values above are
       reserved for the mirror, so this one sits well clear of them. It means an
       exception was caught at the boundary, because letting one unwind through
       a C frame is undefined behavior. */
    ARUCO3CUDA_STATUS_INTERNAL_ERROR = 100
} Aruco3CudaStatus;

/* The memory space an input buffer lives in. Matches aruco3cuda::MemorySpace. */
typedef enum Aruco3CudaMemorySpace {
    ARUCO3CUDA_MEMORY_HOST_PAGEABLE = 0,
    ARUCO3CUDA_MEMORY_HOST_PINNED = 1,
    ARUCO3CUDA_MEMORY_MANAGED = 2,
    ARUCO3CUDA_MEMORY_DEVICE = 3
} Aruco3CudaMemorySpace;

/* Corner refinement method. Matches aruco3cuda::CornerRefineMethod. */
typedef enum Aruco3CudaCornerRefineMethod {
    ARUCO3CUDA_CORNER_REFINE_NONE = 0,
    ARUCO3CUDA_CORNER_REFINE_SUBPIX = 1
} Aruco3CudaCornerRefineMethod;

/* Largest marker size the dictionary functions accept, in bits per side. */
#define ARUCO3CUDA_MAX_MARKER_SIZE 7

/* A non-owning view of an 8-bit grayscale image, mirroring
   aruco3cuda::ImageViewU8. The memory must live in the space named by space. */
typedef struct Aruco3CudaImage {
    const uint8_t* data;
    int width_px;
    int height_px;
    size_t pitch_bytes;
    int space; /* Aruco3CudaMemorySpace */
} Aruco3CudaImage;

/* Metadata of one built-in dictionary. name points into static storage. */
typedef struct Aruco3CudaDictionaryInfo {
    const char* name;
    int marker_size;
    int code_count;
    int max_correction_bits;
} Aruco3CudaDictionaryInfo;

/* Detection settings, mirroring aruco3cuda::DetectorConfig field for field. The
   trailing underscore of the C++ members is dropped because it is a C++ member
   convention. Fill this with aruco3cuda_config_defaults rather than zeroing it:
   an all-zero config is not a valid one. */
typedef struct Aruco3CudaConfig {
    int adaptive_thresh_win_size_min_px;
    int adaptive_thresh_win_size_max_px;
    int adaptive_thresh_win_size_step_px;
    double adaptive_thresh_constant;

    double min_marker_perimeter_rate;
    double max_marker_perimeter_rate;
    double polygonal_approx_accuracy_rate;
    double min_corner_distance_rate;
    int min_distance_to_border_px;
    double min_marker_distance_rate;
    double min_group_distance;
    double min_quad_inlier_ratio;
    double min_edge_support_ratio;

    int marker_border_bits;
    int perspective_remove_pixel_per_cell;
    double perspective_remove_ignored_margin_per_cell;
    double max_erroneous_bits_in_border_rate;
    double min_otsu_std_dev;
    double error_correction_rate;
    double valid_bit_threshold;

    int corner_refine_method; /* Aruco3CudaCornerRefineMethod */
    int corner_refinement_win_size_px;
    double relative_corner_refinement_win_size;
    int corner_refinement_max_iterations;
    double corner_refinement_min_accuracy_px;

    int use_aruco3_detection; /* 0 or 1 */
    int min_side_length_canonical_img_px;
    float min_marker_length_ratio_original_img;

    int max_candidates;
    int max_markers;
    int max_width_px;
    int max_height_px;
    int cuda_block_dim;
} Aruco3CudaConfig;

/* Caller-provided destination for the host-side results.
   The three array members are inputs and must each hold capacity elements,
   except corners which must hold capacity * 8. The remaining members are
   outputs. */
typedef struct Aruco3CudaDetectionBuffer {
    int32_t* ids;
    /* x0, y0, x1, y1, x2, y2, x3, y3 per detection. */
    float* corners;
    int32_t* rotations;
    int capacity;
    /* Number of detections written. */
    int count;
    /* Number of detections before truncation. Larger than count means some were
       dropped. */
    int32_t accepted_total;
    /* 1 if the results were truncated at max_markers. */
    int truncated;
} Aruco3CudaDetectionBuffer;

/* Device pointers to the results, mirroring aruco3cuda::DeviceDetections. The
   detector owns this memory; it is only borrowed here. */
typedef struct Aruco3CudaDeviceDetections {
    void* ids;
    void* rotations;
    void* corner_x;
    void* corner_y;
    void* source;
    void* count;
    void* accepted_total;
    int capacity;
} Aruco3CudaDeviceDetections;

/* How the device workspace is being used, mirroring
   aruco3cuda::WorkspaceStatistics. */
typedef struct Aruco3CudaWorkspaceStatistics {
    size_t allocation_count;
    size_t reallocation_count;
    size_t capacity_bytes;
    size_t used_bytes;
    size_t peak_used_bytes;
    size_t exhausted_count;
} Aruco3CudaWorkspaceStatistics;

/* Opaque handle to a detector. */
typedef struct Aruco3CudaDetector Aruco3CudaDetector;

/// Returns the library version.
///
/// @return A string with static storage duration. Never NULL.
///
/// Ownership: points into static storage; the caller neither frees nor modifies it.
/// Synchronization: host only, with no synchronization point.
///
/// Example input: no arguments
/// Example output: "0.1.0"
ARUCO3CUDA_C_API const char* aruco3cuda_version_string(void);

/// Converts a status into its identifier string.
///
/// @param status The value to convert. A value outside the enumeration still
///               yields a string rather than NULL.
/// @return A string with static storage duration. Never NULL.
///
/// Ownership: points into static storage; the caller neither frees nor modifies it.
/// Synchronization: host only, with no synchronization point.
///
/// Example input: ARUCO3CUDA_STATUS_CUDA_ERROR
/// Example output: "kCudaError"
ARUCO3CUDA_C_API const char* aruco3cuda_status_string(int status);

/// Returns a description of the most recently recorded CUDA error.
///
/// @return A string held per thread, valid until the next CUDA error is recorded
///         on the same thread. Empty if nothing has been recorded. Never NULL.
///
/// Ownership: points into storage owned by the library; the caller does not free it.
/// Synchronization: host only, with no synchronization point. Independent per thread.
///
/// Example input: called right after a cudaMalloc failure was recorded
/// Example output: "api=cudaMalloc stage=... device=0 ..."
ARUCO3CUDA_C_API const char* aruco3cuda_last_cuda_error_message(void);

/// Returns the number of built-in dictionaries.
///
/// @return The count. Never 0.
///
/// Ownership: holds no resource.
/// Synchronization: host only, with no synchronization point.
///
/// Example input: no arguments
/// Example output: 1
ARUCO3CUDA_C_API size_t aruco3cuda_builtin_dictionary_count(void);

/// Returns the metadata of the built-in dictionary at the given index.
///
/// @param index At least 0 and less than aruco3cuda_builtin_dictionary_count().
/// @param out Receives the metadata. Must not be NULL.
/// @return ARUCO3CUDA_STATUS_OK, or ARUCO3CUDA_STATUS_INVALID_ARGUMENT if out is
///         NULL or the index is out of range.
///
/// Ownership: out->name points into static storage and is not freed by the caller.
/// Synchronization: host only, with no synchronization point.
///
/// Example input: index 0
/// Example output: OK, with out->name "DICT_ARUCO_MIP_36h12" and marker_size 6
ARUCO3CUDA_C_API Aruco3CudaStatus aruco3cuda_builtin_dictionary_at(size_t index,
                                                                   Aruco3CudaDictionaryInfo* out);

/// Looks up a built-in dictionary by name.
///
/// @param name The name to look for. NULL is treated as not found.
/// @param out Receives the metadata. Must not be NULL.
/// @return ARUCO3CUDA_STATUS_OK, or ARUCO3CUDA_STATUS_UNSUPPORTED_DICTIONARY if
///         the name is unknown, or ARUCO3CUDA_STATUS_INVALID_ARGUMENT if out is NULL.
///
/// Ownership: name is neither copied nor retained.
/// Synchronization: host only, with no synchronization point.
///
/// Example input: "DICT_ARUCO_MIP_36h12"
/// Example output: OK, with out->code_count 250
ARUCO3CUDA_C_API Aruco3CudaStatus aruco3cuda_find_builtin_dictionary(const char* name,
                                                                     Aruco3CudaDictionaryInfo* out);

/// Expands the bit pattern of one marker, row-major, as 0 and 1 bytes.
///
/// This is what a renderer needs in order to draw a marker, and it is the only
/// reason a caller has to reach into the dictionary at all.
///
/// @param name The dictionary to read from.
/// @param id The marker id. At least 0 and less than the dictionary's code count.
/// @param rotation 0 through 3, counterclockwise. Pass 0 for the marker as drawn.
/// @param out_bits Destination for marker_size * marker_size bytes.
/// @param capacity Number of bytes out_bits can hold.
/// @return ARUCO3CUDA_STATUS_OK, ARUCO3CUDA_STATUS_UNSUPPORTED_DICTIONARY if the
///         name is unknown, or ARUCO3CUDA_STATUS_INVALID_ARGUMENT if a pointer is
///         NULL, the id or rotation is out of range, or capacity is too small.
///
/// Ownership: does not retain the memory behind the arguments.
/// Synchronization: host only, with no synchronization point.
///
/// Example input: "DICT_ARUCO_MIP_36h12", id 42, rotation 0, capacity 36
/// Example output: OK, with out_bits holding 36 zeros and ones
ARUCO3CUDA_C_API Aruco3CudaStatus aruco3cuda_dictionary_marker_bits(const char* name, int id,
                                                                    int rotation, uint8_t* out_bits,
                                                                    size_t capacity);

/// Fills a configuration with the library defaults.
///
/// @param out Receives the defaults. Must not be NULL.
/// @return ARUCO3CUDA_STATUS_OK, or ARUCO3CUDA_STATUS_INVALID_ARGUMENT if out is NULL.
///
/// Ownership: does not retain the memory behind the arguments.
/// Synchronization: host only, with no synchronization point.
///
/// Example input: a pointer to an uninitialized Aruco3CudaConfig
/// Example output: OK, with use_aruco3_detection 1 and max_markers 1024
ARUCO3CUDA_C_API Aruco3CudaStatus aruco3cuda_config_defaults(Aruco3CudaConfig* out);

/// Checks a configuration without creating a detector.
///
/// @param config The configuration to check. Must not be NULL.
/// @param out_message Receives the reason on failure, NUL-terminated and
///                    truncated to fit. May be NULL.
/// @param capacity Number of bytes out_message can hold.
/// @return ARUCO3CUDA_STATUS_OK, or ARUCO3CUDA_STATUS_INVALID_CONFIG.
///
/// Ownership: does not retain the memory behind the arguments.
/// Synchronization: host only, with no synchronization point. Calls no CUDA API.
///
/// Example input: the defaults
/// Example output: OK
/// Example input: adaptive_thresh_win_size_min_px = 2
/// Example output: INVALID_CONFIG, with out_message containing
///                 "adaptive_thresh_win_size_min_px=2"
ARUCO3CUDA_C_API Aruco3CudaStatus aruco3cuda_config_validate(const Aruco3CudaConfig* config,
                                                             char* out_message, size_t capacity);

/// Creates a detector.
///
/// @return A handle, or NULL if allocation failed. Release it with
///         aruco3cuda_detector_destroy.
///
/// Ownership: the caller owns the returned handle.
/// Synchronization: host only, with no synchronization point. Calls no CUDA API,
///                  so this succeeds even with no device present.
///
/// Example input: no arguments
/// Example output: a non-NULL handle
ARUCO3CUDA_C_API Aruco3CudaDetector* aruco3cuda_detector_create(void);

/// Releases a detector and the device memory behind it.
///
/// @param detector The handle to release. NULL is accepted and ignored.
///
/// Ownership: the handle is invalid after this call.
/// Synchronization: releases device memory, so it must not run while kernels
///                  issued by this detector are still in flight.
///
/// Example input: a handle from aruco3cuda_detector_create
/// Example output: no return value; the handle must not be used again
ARUCO3CUDA_C_API void aruco3cuda_detector_destroy(Aruco3CudaDetector* detector);

/// Allocates the workspace and transfers the dictionary to the device.
///
/// @param detector The handle to initialize.
/// @param dictionary_name Name of a built-in dictionary.
/// @param config The settings. Must not be NULL.
/// @param out_message Receives the reason on failure. May be NULL.
/// @param capacity Number of bytes out_message can hold.
/// @return ARUCO3CUDA_STATUS_OK, UNSUPPORTED_DICTIONARY, INVALID_CONFIG, or
///         CUDA_ERROR.
///
/// Ownership: the dictionary is copied to the device, so the name need not outlive
///            the call.
/// Synchronization: **synchronizes the whole device** at the end.
///
/// Example input: "DICT_ARUCO_MIP_36h12" and the defaults
/// Example output: OK
ARUCO3CUDA_C_API Aruco3CudaStatus aruco3cuda_detector_initialize(Aruco3CudaDetector* detector,
                                                                 const char* dictionary_name,
                                                                 const Aruco3CudaConfig* config,
                                                                 char* out_message,
                                                                 size_t capacity);

/// Issues a detection onto a stream.
///
/// @param detector The handle. Must have been initialized.
/// @param image The input image, in device or managed memory. Must not be NULL.
/// @param stream A cudaStream_t, passed as void* so that this header does not
///               depend on the CUDA headers. NULL means the default stream, which
///               CUDA cannot capture as a graph.
/// @param out_message Receives the reason on failure. May be NULL.
/// @param capacity Number of bytes out_message can hold.
/// @return ARUCO3CUDA_STATUS_OK, NOT_INITIALIZED, INVALID_IMAGE, INVALID_ARGUMENT,
///         INVALID_CONFIG, or CUDA_ERROR.
///
/// Ownership: the image memory stays with the caller and must remain valid until
///            the kernels have completed.
/// Synchronization: **does not synchronize**, except once on a frame whose
///                  dimensions or pitch differ from the previous one.
///
/// Example input: a 1280x720 device image and a stream
/// Example output: OK
ARUCO3CUDA_C_API Aruco3CudaStatus aruco3cuda_detector_detect(Aruco3CudaDetector* detector,
                                                             const Aruco3CudaImage* image,
                                                             void* stream, char* out_message,
                                                             size_t capacity);

/// Waits for the stream and copies the results into caller-provided arrays.
///
/// @param detector The handle.
/// @param stream The stream to wait on. Use the same one passed to detect.
/// @param buffer Destination. Its three arrays and capacity are inputs; count,
///               accepted_total, and truncated are outputs. Must not be NULL.
/// @param out_message Receives the reason on failure. May be NULL.
/// @param capacity Number of bytes out_message can hold.
/// @return ARUCO3CUDA_STATUS_OK. MARKER_OVERFLOW if the detector truncated the
///         results, in which case the buffer is still filled. INVALID_ARGUMENT if
///         the buffer is too small for the detections found, NOT_INITIALIZED, or
///         CUDA_ERROR.
///
/// Ownership: the arrays stay with the caller.
/// Synchronization: **synchronizes the stream.**
///
/// Example input: a buffer with capacity 1024, after a frame with two markers
/// Example output: OK, with buffer->count 2
ARUCO3CUDA_C_API Aruco3CudaStatus aruco3cuda_detector_download(Aruco3CudaDetector* detector,
                                                               void* stream,
                                                               Aruco3CudaDetectionBuffer* buffer,
                                                               char* out_message, size_t capacity);

/// Borrows the device pointers to the results of the most recent detect.
///
/// A stage that runs on the same device reads the results from here instead of
/// calling download, and never touches the host.
///
/// @param detector The handle.
/// @param out Receives the pointers. Must not be NULL.
/// @return ARUCO3CUDA_STATUS_OK, INVALID_ARGUMENT, or NOT_INITIALIZED if detect
///         has never been called.
///
/// Ownership: the detector owns the memory. The pointers become invalid when the
///            detector is destroyed or the input dimensions change.
/// Synchronization: **does not synchronize.** The contents are not final until
///                  the kernels already issued have completed.
///
/// Example input: called right after detect
/// Example output: OK, with out->capacity equal to max_markers
ARUCO3CUDA_C_API Aruco3CudaStatus aruco3cuda_detector_device_detections(
        const Aruco3CudaDetector* detector, Aruco3CudaDeviceDetections* out);

/// Reports how the device workspace is being used.
///
/// @param detector The handle.
/// @param out Receives the statistics. Must not be NULL.
/// @return ARUCO3CUDA_STATUS_OK, or ARUCO3CUDA_STATUS_INVALID_ARGUMENT.
///
/// Ownership: values only; nothing is borrowed.
/// Synchronization: host only, with no synchronization point.
///
/// Example input: a handle on which initialize has been called
/// Example output: OK, with out->allocation_count 1
ARUCO3CUDA_C_API Aruco3CudaStatus aruco3cuda_detector_workspace_statistics(
        const Aruco3CudaDetector* detector, Aruco3CudaWorkspaceStatistics* out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ARUCO3CUDA_C_ARUCO3CUDA_H */
