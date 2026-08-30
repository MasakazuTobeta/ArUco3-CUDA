// SPDX-License-Identifier: Apache-2.0
//
// Implementation of the C ABI.
//
// Purpose:
//   Translate between the C++ API and a flat C surface. Three things happen at
//   this boundary and nowhere else: exceptions are stopped, C++ types are
//   converted to plain structs, and the enum values are pinned to the C++ ones
//   with static_assert so that a reordering on either side fails to compile
//   rather than silently renumbering a caller's error codes.
#include "aruco3cuda/c/aruco3cuda.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/detections.hpp"
#include "aruco3cuda/detector.hpp"
#include "aruco3cuda/dictionary.hpp"
#include "aruco3cuda/status.hpp"
#include "aruco3cuda/types.hpp"
#include "aruco3cuda/version.hpp"
#include "aruco3cuda/workspace.hpp"

namespace {

using aruco3cuda::Status;

// The C enumerators are part of a published ABI, so they cannot follow the C++
// enum by being cast at run time alone. If someone inserts a value into
// Status, these stop the build instead of shifting every caller's codes by one.
static_assert(static_cast<int>(Status::kOk) == ARUCO3CUDA_STATUS_OK, "status drift");
static_assert(static_cast<int>(Status::kInvalidArgument) == ARUCO3CUDA_STATUS_INVALID_ARGUMENT,
              "status drift");
static_assert(static_cast<int>(Status::kInvalidImage) == ARUCO3CUDA_STATUS_INVALID_IMAGE,
              "status drift");
static_assert(static_cast<int>(Status::kInvalidConfig) == ARUCO3CUDA_STATUS_INVALID_CONFIG,
              "status drift");
static_assert(static_cast<int>(Status::kUnsupportedDictionary) ==
                      ARUCO3CUDA_STATUS_UNSUPPORTED_DICTIONARY,
              "status drift");
static_assert(static_cast<int>(Status::kCandidateOverflow) == ARUCO3CUDA_STATUS_CANDIDATE_OVERFLOW,
              "status drift");
static_assert(static_cast<int>(Status::kMarkerOverflow) == ARUCO3CUDA_STATUS_MARKER_OVERFLOW,
              "status drift");
static_assert(static_cast<int>(Status::kCudaError) == ARUCO3CUDA_STATUS_CUDA_ERROR, "status drift");
static_assert(static_cast<int>(Status::kNotInitialized) == ARUCO3CUDA_STATUS_NOT_INITIALIZED,
              "status drift");

static_assert(static_cast<int>(aruco3cuda::MemorySpace::kHostPageable) ==
                      ARUCO3CUDA_MEMORY_HOST_PAGEABLE,
              "memory space drift");
static_assert(static_cast<int>(aruco3cuda::MemorySpace::kHostPinned) ==
                      ARUCO3CUDA_MEMORY_HOST_PINNED,
              "memory space drift");
static_assert(static_cast<int>(aruco3cuda::MemorySpace::kManaged) == ARUCO3CUDA_MEMORY_MANAGED,
              "memory space drift");
static_assert(static_cast<int>(aruco3cuda::MemorySpace::kDevice) == ARUCO3CUDA_MEMORY_DEVICE,
              "memory space drift");

static_assert(static_cast<int>(aruco3cuda::CornerRefineMethod::kNone) ==
                      ARUCO3CUDA_CORNER_REFINE_NONE,
              "corner refine method drift");
static_assert(static_cast<int>(aruco3cuda::CornerRefineMethod::kSubpix) ==
                      ARUCO3CUDA_CORNER_REFINE_SUBPIX,
              "corner refine method drift");

/// Copies a reason into a caller-provided buffer, truncating to fit.
void copy_message(const std::string& text, char* out, std::size_t capacity) {
    if (out == nullptr || capacity == 0U) {
        return;
    }
    const std::size_t length = std::min(text.size(), capacity - 1U);
    std::memcpy(out, text.data(), length);
    out[length] = '\0';
}

/// Converts the C configuration into the C++ one.
///
/// The two enum-valued fields arrive as int, so they are range checked here.
/// The C++ type makes them unrepresentable, which is exactly why the check has
/// to happen at the boundary rather than inside validate().
bool to_cpp_config(const Aruco3CudaConfig& in, aruco3cuda::DetectorConfig* out,
                   std::string* out_message) {
    if (in.corner_refine_method != ARUCO3CUDA_CORNER_REFINE_NONE &&
        in.corner_refine_method != ARUCO3CUDA_CORNER_REFINE_SUBPIX) {
        *out_message =
                "corner_refine_method=" + std::to_string(in.corner_refine_method) + " is not valid";
        return false;
    }
    out->adaptive_thresh_win_size_min_px_ = in.adaptive_thresh_win_size_min_px;
    out->adaptive_thresh_win_size_max_px_ = in.adaptive_thresh_win_size_max_px;
    out->adaptive_thresh_win_size_step_px_ = in.adaptive_thresh_win_size_step_px;
    out->adaptive_thresh_constant_ = in.adaptive_thresh_constant;
    out->min_marker_perimeter_rate_ = in.min_marker_perimeter_rate;
    out->max_marker_perimeter_rate_ = in.max_marker_perimeter_rate;
    out->polygonal_approx_accuracy_rate_ = in.polygonal_approx_accuracy_rate;
    out->min_corner_distance_rate_ = in.min_corner_distance_rate;
    out->min_distance_to_border_px_ = in.min_distance_to_border_px;
    out->min_marker_distance_rate_ = in.min_marker_distance_rate;
    out->min_group_distance_ = in.min_group_distance;
    out->min_quad_inlier_ratio_ = in.min_quad_inlier_ratio;
    out->min_edge_support_ratio_ = in.min_edge_support_ratio;
    out->marker_border_bits_ = in.marker_border_bits;
    out->perspective_remove_pixel_per_cell_ = in.perspective_remove_pixel_per_cell;
    out->perspective_remove_ignored_margin_per_cell_ =
            in.perspective_remove_ignored_margin_per_cell;
    out->max_erroneous_bits_in_border_rate_ = in.max_erroneous_bits_in_border_rate;
    out->min_otsu_std_dev_ = in.min_otsu_std_dev;
    out->error_correction_rate_ = in.error_correction_rate;
    out->valid_bit_threshold_ = in.valid_bit_threshold;
    out->corner_refine_method_ =
            static_cast<aruco3cuda::CornerRefineMethod>(in.corner_refine_method);
    out->corner_refinement_win_size_px_ = in.corner_refinement_win_size_px;
    out->relative_corner_refinement_win_size_ = in.relative_corner_refinement_win_size;
    out->corner_refinement_max_iterations_ = in.corner_refinement_max_iterations;
    out->corner_refinement_min_accuracy_px_ = in.corner_refinement_min_accuracy_px;
    out->use_aruco3_detection_ = in.use_aruco3_detection != 0;
    out->min_side_length_canonical_img_px_ = in.min_side_length_canonical_img_px;
    out->min_marker_length_ratio_original_img_ = in.min_marker_length_ratio_original_img;
    out->max_candidates_ = in.max_candidates;
    out->max_markers_ = in.max_markers;
    out->max_width_px_ = in.max_width_px;
    out->max_height_px_ = in.max_height_px;
    out->cuda_block_dim_ = in.cuda_block_dim;
    return true;
}

/// Converts the C++ configuration into the C one.
void to_c_config(const aruco3cuda::DetectorConfig& in, Aruco3CudaConfig* out) {
    out->adaptive_thresh_win_size_min_px = in.adaptive_thresh_win_size_min_px_;
    out->adaptive_thresh_win_size_max_px = in.adaptive_thresh_win_size_max_px_;
    out->adaptive_thresh_win_size_step_px = in.adaptive_thresh_win_size_step_px_;
    out->adaptive_thresh_constant = in.adaptive_thresh_constant_;
    out->min_marker_perimeter_rate = in.min_marker_perimeter_rate_;
    out->max_marker_perimeter_rate = in.max_marker_perimeter_rate_;
    out->polygonal_approx_accuracy_rate = in.polygonal_approx_accuracy_rate_;
    out->min_corner_distance_rate = in.min_corner_distance_rate_;
    out->min_distance_to_border_px = in.min_distance_to_border_px_;
    out->min_marker_distance_rate = in.min_marker_distance_rate_;
    out->min_group_distance = in.min_group_distance_;
    out->min_quad_inlier_ratio = in.min_quad_inlier_ratio_;
    out->min_edge_support_ratio = in.min_edge_support_ratio_;
    out->marker_border_bits = in.marker_border_bits_;
    out->perspective_remove_pixel_per_cell = in.perspective_remove_pixel_per_cell_;
    out->perspective_remove_ignored_margin_per_cell =
            in.perspective_remove_ignored_margin_per_cell_;
    out->max_erroneous_bits_in_border_rate = in.max_erroneous_bits_in_border_rate_;
    out->min_otsu_std_dev = in.min_otsu_std_dev_;
    out->error_correction_rate = in.error_correction_rate_;
    out->valid_bit_threshold = in.valid_bit_threshold_;
    out->corner_refine_method = static_cast<int>(in.corner_refine_method_);
    out->corner_refinement_win_size_px = in.corner_refinement_win_size_px_;
    out->relative_corner_refinement_win_size = in.relative_corner_refinement_win_size_;
    out->corner_refinement_max_iterations = in.corner_refinement_max_iterations_;
    out->corner_refinement_min_accuracy_px = in.corner_refinement_min_accuracy_px_;
    out->use_aruco3_detection = in.use_aruco3_detection_ ? 1 : 0;
    out->min_side_length_canonical_img_px = in.min_side_length_canonical_img_px_;
    out->min_marker_length_ratio_original_img = in.min_marker_length_ratio_original_img_;
    out->max_candidates = in.max_candidates_;
    out->max_markers = in.max_markers_;
    out->max_width_px = in.max_width_px_;
    out->max_height_px = in.max_height_px_;
    out->cuda_block_dim = in.cuda_block_dim_;
}

}  // namespace

// The handle is a struct rather than a typedef to void* so that the compiler
// catches a caller passing the wrong pointer type.
struct Aruco3CudaDetector {
    aruco3cuda::Detector detector_;
    // Kept so that download() can report the capacity the caller must provide.
    int max_markers_ = 0;
};

extern "C" {

const char* aruco3cuda_version_string(void) {
    return aruco3cuda::version_string();
}

const char* aruco3cuda_status_string(int status) {
    if (status == ARUCO3CUDA_STATUS_INTERNAL_ERROR) {
        return "kInternalError";
    }
    return aruco3cuda::to_string(static_cast<Status>(status));
}

const char* aruco3cuda_last_cuda_error_message(void) {
    return aruco3cuda::last_cuda_error_message();
}

size_t aruco3cuda_builtin_dictionary_count(void) {
    return aruco3cuda::builtin_dictionary_count();
}

Aruco3CudaStatus aruco3cuda_builtin_dictionary_at(size_t index, Aruco3CudaDictionaryInfo* out) {
    if (out == nullptr) {
        return ARUCO3CUDA_STATUS_INVALID_ARGUMENT;
    }
    const aruco3cuda::DictionaryTable* table = aruco3cuda::builtin_dictionary_at(index);
    if (table == nullptr) {
        return ARUCO3CUDA_STATUS_INVALID_ARGUMENT;
    }
    out->name = table->name_;
    out->marker_size = table->marker_size_;
    out->code_count = table->code_count_;
    out->max_correction_bits = table->max_correction_bits_;
    return ARUCO3CUDA_STATUS_OK;
}

Aruco3CudaStatus aruco3cuda_find_builtin_dictionary(const char* name,
                                                    Aruco3CudaDictionaryInfo* out) {
    if (out == nullptr) {
        return ARUCO3CUDA_STATUS_INVALID_ARGUMENT;
    }
    const aruco3cuda::DictionaryTable* table = aruco3cuda::find_builtin_dictionary(name);
    if (table == nullptr) {
        return ARUCO3CUDA_STATUS_UNSUPPORTED_DICTIONARY;
    }
    out->name = table->name_;
    out->marker_size = table->marker_size_;
    out->code_count = table->code_count_;
    out->max_correction_bits = table->max_correction_bits_;
    return ARUCO3CUDA_STATUS_OK;
}

Aruco3CudaStatus aruco3cuda_dictionary_marker_bits(const char* name, int id, int rotation,
                                                   uint8_t* out_bits, size_t capacity) {
    if (out_bits == nullptr) {
        return ARUCO3CUDA_STATUS_INVALID_ARGUMENT;
    }
    const aruco3cuda::DictionaryTable* table = aruco3cuda::find_builtin_dictionary(name);
    if (table == nullptr) {
        return ARUCO3CUDA_STATUS_UNSUPPORTED_DICTIONARY;
    }
    if (id < 0 || id >= table->code_count_ || rotation < 0 || rotation > 3) {
        return ARUCO3CUDA_STATUS_INVALID_ARGUMENT;
    }
    const size_t bit_count = static_cast<size_t>(table->bit_count());
    if (capacity < bit_count) {
        return ARUCO3CUDA_STATUS_INVALID_ARGUMENT;
    }
    const aruco3cuda::MarkerCode code =
            table->codes_[static_cast<size_t>(id) * 4U + static_cast<size_t>(rotation)];
    if (aruco3cuda::unpack_marker_code(code, table->marker_size_, out_bits) != Status::kOk) {
        return ARUCO3CUDA_STATUS_INVALID_ARGUMENT;
    }
    return ARUCO3CUDA_STATUS_OK;
}

Aruco3CudaStatus aruco3cuda_config_defaults(Aruco3CudaConfig* out) {
    if (out == nullptr) {
        return ARUCO3CUDA_STATUS_INVALID_ARGUMENT;
    }
    to_c_config(aruco3cuda::DetectorConfig(), out);
    return ARUCO3CUDA_STATUS_OK;
}

Aruco3CudaStatus aruco3cuda_config_validate(const Aruco3CudaConfig* config, char* out_message,
                                            size_t capacity) {
    if (config == nullptr) {
        copy_message("config is NULL", out_message, capacity);
        return ARUCO3CUDA_STATUS_INVALID_ARGUMENT;
    }
    try {
        aruco3cuda::DetectorConfig cpp_config;
        std::string message;
        if (!to_cpp_config(*config, &cpp_config, &message)) {
            copy_message(message, out_message, capacity);
            return ARUCO3CUDA_STATUS_INVALID_CONFIG;
        }
        const Status status = cpp_config.validate(&message);
        copy_message(message, out_message, capacity);
        return static_cast<Aruco3CudaStatus>(status);
    } catch (const std::exception& error) {
        copy_message(error.what(), out_message, capacity);
        return ARUCO3CUDA_STATUS_INTERNAL_ERROR;
    } catch (...) {
        copy_message("unknown exception", out_message, capacity);
        return ARUCO3CUDA_STATUS_INTERNAL_ERROR;
    }
}

Aruco3CudaDetector* aruco3cuda_detector_create(void) {
    // new is used here on purpose. The handle crosses into a caller that has no
    // destructors, so its lifetime cannot be expressed with RAII on this side.
    // aruco3cuda_detector_destroy is the matching release.
    return new (std::nothrow) Aruco3CudaDetector();
}

void aruco3cuda_detector_destroy(Aruco3CudaDetector* detector) {
    delete detector;
}

Aruco3CudaStatus aruco3cuda_detector_initialize(Aruco3CudaDetector* detector,
                                                const char* dictionary_name,
                                                const Aruco3CudaConfig* config, char* out_message,
                                                size_t capacity) {
    if (detector == nullptr || config == nullptr) {
        copy_message("detector or config is NULL", out_message, capacity);
        return ARUCO3CUDA_STATUS_INVALID_ARGUMENT;
    }
    try {
        const aruco3cuda::DictionaryTable* table =
                aruco3cuda::find_builtin_dictionary(dictionary_name);
        if (table == nullptr) {
            copy_message("unknown dictionary", out_message, capacity);
            return ARUCO3CUDA_STATUS_UNSUPPORTED_DICTIONARY;
        }
        aruco3cuda::DetectorConfig cpp_config;
        std::string message;
        if (!to_cpp_config(*config, &cpp_config, &message)) {
            copy_message(message, out_message, capacity);
            return ARUCO3CUDA_STATUS_INVALID_CONFIG;
        }
        const Status status = detector->detector_.initialize(*table, cpp_config, &message);
        copy_message(message, out_message, capacity);
        if (status == Status::kOk) {
            detector->max_markers_ = cpp_config.max_markers_;
        }
        return static_cast<Aruco3CudaStatus>(status);
    } catch (const std::exception& error) {
        copy_message(error.what(), out_message, capacity);
        return ARUCO3CUDA_STATUS_INTERNAL_ERROR;
    } catch (...) {
        copy_message("unknown exception", out_message, capacity);
        return ARUCO3CUDA_STATUS_INTERNAL_ERROR;
    }
}

Aruco3CudaStatus aruco3cuda_detector_detect(Aruco3CudaDetector* detector,
                                            const Aruco3CudaImage* image, void* stream,
                                            char* out_message, size_t capacity) {
    if (detector == nullptr || image == nullptr) {
        copy_message("detector or image is NULL", out_message, capacity);
        return ARUCO3CUDA_STATUS_INVALID_ARGUMENT;
    }
    try {
        aruco3cuda::ImageViewU8 view;
        view.data_ = image->data;
        view.width_px_ = image->width_px;
        view.height_px_ = image->height_px;
        view.pitch_bytes_ = image->pitch_bytes;
        view.space_ = static_cast<aruco3cuda::MemorySpace>(image->space);
        std::string message;
        const Status status =
                detector->detector_.detect_async(view, static_cast<cudaStream_t>(stream), &message);
        copy_message(message, out_message, capacity);
        return static_cast<Aruco3CudaStatus>(status);
    } catch (const std::exception& error) {
        copy_message(error.what(), out_message, capacity);
        return ARUCO3CUDA_STATUS_INTERNAL_ERROR;
    } catch (...) {
        copy_message("unknown exception", out_message, capacity);
        return ARUCO3CUDA_STATUS_INTERNAL_ERROR;
    }
}

Aruco3CudaStatus aruco3cuda_detector_download(Aruco3CudaDetector* detector, void* stream,
                                              Aruco3CudaDetectionBuffer* buffer, char* out_message,
                                              size_t capacity) {
    if (detector == nullptr || buffer == nullptr) {
        copy_message("detector or buffer is NULL", out_message, capacity);
        return ARUCO3CUDA_STATUS_INVALID_ARGUMENT;
    }
    if (buffer->capacity < 0 ||
        (buffer->capacity > 0 &&
         (buffer->ids == nullptr || buffer->corners == nullptr || buffer->rotations == nullptr))) {
        copy_message("the buffer arrays are NULL", out_message, capacity);
        return ARUCO3CUDA_STATUS_INVALID_ARGUMENT;
    }
    try {
        aruco3cuda::HostDetections result;
        std::string message;
        const Status status =
                detector->detector_.download(&result, static_cast<cudaStream_t>(stream), &message);
        copy_message(message, out_message, capacity);
        if (status != Status::kOk && status != Status::kMarkerOverflow) {
            return static_cast<Aruco3CudaStatus>(status);
        }

        const size_t found = result.ids_.size();
        if (found > static_cast<size_t>(buffer->capacity)) {
            copy_message("the buffer holds " + std::to_string(buffer->capacity) + " entries but " +
                                 std::to_string(found) + " were detected",
                         out_message, capacity);
            return ARUCO3CUDA_STATUS_INVALID_ARGUMENT;
        }
        if (found > 0U) {
            std::memcpy(buffer->ids, result.ids_.data(), found * sizeof(int32_t));
            std::memcpy(buffer->rotations, result.rotations_.data(), found * sizeof(int32_t));
            std::memcpy(buffer->corners, result.corners_.data(), found * 8U * sizeof(float));
        }
        buffer->count = static_cast<int>(found);
        buffer->accepted_total = result.accepted_total_;
        buffer->truncated = result.marker_overflow_ ? 1 : 0;
        return static_cast<Aruco3CudaStatus>(status);
    } catch (const std::exception& error) {
        copy_message(error.what(), out_message, capacity);
        return ARUCO3CUDA_STATUS_INTERNAL_ERROR;
    } catch (...) {
        copy_message("unknown exception", out_message, capacity);
        return ARUCO3CUDA_STATUS_INTERNAL_ERROR;
    }
}

Aruco3CudaStatus aruco3cuda_detector_device_detections(const Aruco3CudaDetector* detector,
                                                       Aruco3CudaDeviceDetections* out) {
    if (detector == nullptr || out == nullptr) {
        return ARUCO3CUDA_STATUS_INVALID_ARGUMENT;
    }
    aruco3cuda::DeviceDetections source;
    const Status status = detector->detector_.device_detections(&source);
    if (status != Status::kOk) {
        return static_cast<Aruco3CudaStatus>(status);
    }
    out->ids = source.ids_;
    out->rotations = source.rotations_;
    out->corner_x = source.corner_x_;
    out->corner_y = source.corner_y_;
    out->source = source.source_;
    out->count = source.count_;
    out->accepted_total = source.accepted_total_;
    out->capacity = source.capacity_;
    return ARUCO3CUDA_STATUS_OK;
}

Aruco3CudaStatus aruco3cuda_detector_workspace_statistics(const Aruco3CudaDetector* detector,
                                                          Aruco3CudaWorkspaceStatistics* out) {
    if (detector == nullptr || out == nullptr) {
        return ARUCO3CUDA_STATUS_INVALID_ARGUMENT;
    }
    const aruco3cuda::WorkspaceStatistics& source = detector->detector_.workspace_statistics();
    out->allocation_count = source.allocation_count_;
    out->reallocation_count = source.reallocation_count_;
    out->capacity_bytes = source.capacity_bytes_;
    out->used_bytes = source.used_bytes_;
    out->peak_used_bytes = source.peak_used_bytes_;
    out->exhausted_count = source.exhausted_count_;
    return ARUCO3CUDA_STATUS_OK;
}

}  // extern "C"
