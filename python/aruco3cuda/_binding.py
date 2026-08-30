# SPDX-License-Identifier: Apache-2.0
"""ctypes declarations for the C ABI in include/aruco3cuda/c/aruco3cuda.h.

Nothing here is part of the public Python surface. It exists so that __init__.py
can stay readable, and so that the one place that has to mirror the C structs
field for field is small enough to review against the header.

The mirror is checked at run time by test/python/test_binding.py, which compares
every field of the default configuration against the values in
include/aruco3cuda/config.hpp. A field inserted on one side and not the other
shifts the layout, and the comparison then fails on the shifted values rather
than passing quietly.
"""
import ctypes
import os
import pathlib

# Sonames tried in order, newest first. The plain name exists only where the
# CUDA development package is installed, so the versioned ones have to be tried
# too; a deployment machine often has just libcudart.so.13.
kCudartCandidates = (
    "libcudart.so",
    "libcudart.so.13",
    "libcudart.so.12",
    "libcudart.so.11.0",
)

kMessageCapacity = 512


class Aruco3CudaImage(ctypes.Structure):
    """Mirror of Aruco3CudaImage.

    data is declared void* rather than uint8_t* because the address usually
    arrives as a plain integer from a device allocation or from
    __cuda_array_interface__, and c_void_p accepts that directly. The two are
    the same size and alignment, so the ABI is unchanged.
    """

    _fields_ = [
        ("data", ctypes.c_void_p),
        ("width_px", ctypes.c_int),
        ("height_px", ctypes.c_int),
        ("pitch_bytes", ctypes.c_size_t),
        ("space", ctypes.c_int),
    ]


class Aruco3CudaDictionaryInfo(ctypes.Structure):
    """Mirror of Aruco3CudaDictionaryInfo."""

    _fields_ = [
        ("name", ctypes.c_char_p),
        ("marker_size", ctypes.c_int),
        ("code_count", ctypes.c_int),
        ("max_correction_bits", ctypes.c_int),
    ]


class Aruco3CudaConfig(ctypes.Structure):
    """Mirror of Aruco3CudaConfig. The order must match the header exactly."""

    _fields_ = [
        ("adaptive_thresh_win_size_min_px", ctypes.c_int),
        ("adaptive_thresh_win_size_max_px", ctypes.c_int),
        ("adaptive_thresh_win_size_step_px", ctypes.c_int),
        ("adaptive_thresh_constant", ctypes.c_double),
        ("min_marker_perimeter_rate", ctypes.c_double),
        ("max_marker_perimeter_rate", ctypes.c_double),
        ("polygonal_approx_accuracy_rate", ctypes.c_double),
        ("min_corner_distance_rate", ctypes.c_double),
        ("min_distance_to_border_px", ctypes.c_int),
        ("min_marker_distance_rate", ctypes.c_double),
        ("min_group_distance", ctypes.c_double),
        ("min_quad_inlier_ratio", ctypes.c_double),
        ("min_edge_support_ratio", ctypes.c_double),
        ("marker_border_bits", ctypes.c_int),
        ("perspective_remove_pixel_per_cell", ctypes.c_int),
        ("perspective_remove_ignored_margin_per_cell", ctypes.c_double),
        ("max_erroneous_bits_in_border_rate", ctypes.c_double),
        ("min_otsu_std_dev", ctypes.c_double),
        ("error_correction_rate", ctypes.c_double),
        ("valid_bit_threshold", ctypes.c_double),
        ("corner_refine_method", ctypes.c_int),
        ("corner_refinement_win_size_px", ctypes.c_int),
        ("relative_corner_refinement_win_size", ctypes.c_double),
        ("corner_refinement_max_iterations", ctypes.c_int),
        ("corner_refinement_min_accuracy_px", ctypes.c_double),
        ("use_aruco3_detection", ctypes.c_int),
        ("min_side_length_canonical_img_px", ctypes.c_int),
        ("min_marker_length_ratio_original_img", ctypes.c_float),
        ("max_candidates", ctypes.c_int),
        ("max_markers", ctypes.c_int),
        ("max_width_px", ctypes.c_int),
        ("max_height_px", ctypes.c_int),
        ("cuda_block_dim", ctypes.c_int),
    ]


class Aruco3CudaDetectionBuffer(ctypes.Structure):
    """Mirror of Aruco3CudaDetectionBuffer."""

    _fields_ = [
        ("ids", ctypes.POINTER(ctypes.c_int32)),
        ("corners", ctypes.POINTER(ctypes.c_float)),
        ("rotations", ctypes.POINTER(ctypes.c_int32)),
        ("capacity", ctypes.c_int),
        ("count", ctypes.c_int),
        ("accepted_total", ctypes.c_int32),
        ("truncated", ctypes.c_int),
    ]


class Aruco3CudaDeviceDetections(ctypes.Structure):
    """Mirror of Aruco3CudaDeviceDetections."""

    _fields_ = [
        ("ids", ctypes.c_void_p),
        ("rotations", ctypes.c_void_p),
        ("corner_x", ctypes.c_void_p),
        ("corner_y", ctypes.c_void_p),
        ("source", ctypes.c_void_p),
        ("count", ctypes.c_void_p),
        ("accepted_total", ctypes.c_void_p),
        ("capacity", ctypes.c_int),
    ]


class Aruco3CudaWorkspaceStatistics(ctypes.Structure):
    """Mirror of Aruco3CudaWorkspaceStatistics."""

    _fields_ = [
        ("allocation_count", ctypes.c_size_t),
        ("reallocation_count", ctypes.c_size_t),
        ("capacity_bytes", ctypes.c_size_t),
        ("used_bytes", ctypes.c_size_t),
        ("peak_used_bytes", ctypes.c_size_t),
        ("exhausted_count", ctypes.c_size_t),
    ]


def library_candidates():
    """Return the paths tried when loading the C ABI, in order.

    The copy beside this file is tried first, because the build stages the
    shared object into the package directory. That keeps a build tree working
    with nothing but PYTHONPATH, and keeps an installed copy from shadowing the
    one that was just built.
    """
    candidates = []
    override = os.environ.get("ARUCO3CUDA_LIBRARY")
    if override:
        candidates.append(override)
    here = pathlib.Path(__file__).resolve().parent
    candidates.append(str(here / "libaruco3cuda_c.so"))
    candidates.append("libaruco3cuda_c.so")
    return candidates


def load_library():
    """Load the C ABI shared object.

    Raises OSError naming every path that was tried, because "cannot open shared
    object file" on its own sends the reader looking in the wrong place.
    """
    attempts = []
    for candidate in library_candidates():
        try:
            return ctypes.CDLL(candidate)
        except OSError as error:
            attempts.append(f"  {candidate}: {error}")
    raise OSError(
        "libaruco3cuda_c.so could not be loaded. Build the project and put the\n"
        "package directory on PYTHONPATH, or point ARUCO3CUDA_LIBRARY at the\n"
        "shared object. Tried:\n" + "\n".join(attempts)
    )


def load_cudart():
    """Load the CUDA runtime, used only for device memory and streams.

    The detector itself never needs this: it is the caller's job to own the
    device image and the stream, exactly as in the C++ API. Returns None when
    the runtime is absent, so that importing the package still works on a
    machine with no CUDA installed.
    """
    override = os.environ.get("ARUCO3CUDA_CUDART")
    for candidate in ((override,) if override else ()) + kCudartCandidates:
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            continue
    return None


def declare(library):
    """Attach argument and result types to every function of the C ABI.

    Without this, ctypes assumes int for everything, which silently truncates a
    64-bit pointer on the way in and on the way out.
    """
    library.aruco3cuda_version_string.argtypes = []
    library.aruco3cuda_version_string.restype = ctypes.c_char_p

    library.aruco3cuda_status_string.argtypes = [ctypes.c_int]
    library.aruco3cuda_status_string.restype = ctypes.c_char_p

    library.aruco3cuda_last_cuda_error_message.argtypes = []
    library.aruco3cuda_last_cuda_error_message.restype = ctypes.c_char_p

    library.aruco3cuda_builtin_dictionary_count.argtypes = []
    library.aruco3cuda_builtin_dictionary_count.restype = ctypes.c_size_t

    library.aruco3cuda_builtin_dictionary_at.argtypes = [
        ctypes.c_size_t,
        ctypes.POINTER(Aruco3CudaDictionaryInfo),
    ]
    library.aruco3cuda_builtin_dictionary_at.restype = ctypes.c_int

    library.aruco3cuda_find_builtin_dictionary.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(Aruco3CudaDictionaryInfo),
    ]
    library.aruco3cuda_find_builtin_dictionary.restype = ctypes.c_int

    library.aruco3cuda_dictionary_marker_bits.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    library.aruco3cuda_dictionary_marker_bits.restype = ctypes.c_int

    library.aruco3cuda_config_defaults.argtypes = [ctypes.POINTER(Aruco3CudaConfig)]
    library.aruco3cuda_config_defaults.restype = ctypes.c_int

    library.aruco3cuda_config_validate.argtypes = [
        ctypes.POINTER(Aruco3CudaConfig),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.aruco3cuda_config_validate.restype = ctypes.c_int

    library.aruco3cuda_detector_create.argtypes = []
    library.aruco3cuda_detector_create.restype = ctypes.c_void_p

    library.aruco3cuda_detector_destroy.argtypes = [ctypes.c_void_p]
    library.aruco3cuda_detector_destroy.restype = None

    library.aruco3cuda_detector_initialize.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(Aruco3CudaConfig),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.aruco3cuda_detector_initialize.restype = ctypes.c_int

    library.aruco3cuda_detector_detect.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(Aruco3CudaImage),
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.aruco3cuda_detector_detect.restype = ctypes.c_int

    library.aruco3cuda_detector_download.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(Aruco3CudaDetectionBuffer),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    library.aruco3cuda_detector_download.restype = ctypes.c_int

    library.aruco3cuda_detector_device_detections.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(Aruco3CudaDeviceDetections),
    ]
    library.aruco3cuda_detector_device_detections.restype = ctypes.c_int

    library.aruco3cuda_detector_workspace_statistics.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(Aruco3CudaWorkspaceStatistics),
    ]
    library.aruco3cuda_detector_workspace_statistics.restype = ctypes.c_int
    return library


def declare_cudart(cudart):
    """Attach argument and result types to the handful of runtime calls used."""
    if cudart is None:
        return None
    cudart.cudaMallocPitch.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    cudart.cudaMallocPitch.restype = ctypes.c_int

    cudart.cudaFree.argtypes = [ctypes.c_void_p]
    cudart.cudaFree.restype = ctypes.c_int

    cudart.cudaMemcpy2D.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    cudart.cudaMemcpy2D.restype = ctypes.c_int

    cudart.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cudart.cudaStreamCreate.restype = ctypes.c_int

    cudart.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
    cudart.cudaStreamDestroy.restype = ctypes.c_int

    cudart.cudaGetErrorString.argtypes = [ctypes.c_int]
    cudart.cudaGetErrorString.restype = ctypes.c_char_p
    return cudart
