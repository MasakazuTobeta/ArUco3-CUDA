# SPDX-License-Identifier: Apache-2.0
"""Python interface to ArUco3-CUDA.

This package talks to the C ABI in include/aruco3cuda/c/aruco3cuda.h through
ctypes. There is no compiled extension module, so it works with any CPython 3
without being rebuilt, and the library can be built on a machine that has no
Python development headers at all.

It depends on nothing outside the standard library. numpy is not required:
anything supporting the buffer protocol can be uploaded, anything exposing
__cuda_array_interface__ can be detected in place, and results come back as
plain Python objects.

The detector takes device-resident input, exactly as the C++ API does. Uploading
is therefore explicit, through DeviceImage, rather than hidden inside detect().
On a library whose reason for existing is that results never have to leave the
device, a transfer that happens invisibly per frame would be the wrong default.

    import aruco3cuda

    config = aruco3cuda.Config(max_width_px=280, max_height_px=280)
    with aruco3cuda.Detector() as detector:
        detector.initialize("DICT_ARUCO_MIP_36h12", config)
        image = aruco3cuda.DeviceImage.from_host(pixels, 280, 280)
        with aruco3cuda.Stream() as stream:
            detector.detect(image, stream)
            for detection in detector.download(stream):
                print(detection.id, detection.corners)
"""
import ctypes
import enum

from . import _binding

__all__ = [
    "Config",
    "CornerRefine",
    "Detection",
    "Detections",
    "Detector",
    "DeviceDetections",
    "DeviceImage",
    "DictionaryInfo",
    "Error",
    "MemorySpace",
    "Status",
    "Stream",
    "WorkspaceStatistics",
    "dictionaries",
    "find_dictionary",
    "marker_bits",
    "version",
]

_lib = _binding.declare(_binding.load_library())
_cudart = _binding.declare_cudart(_binding.load_cudart())

__version__ = _lib.aruco3cuda_version_string().decode("utf-8")


class Status(enum.IntEnum):
    """Result state of the C ABI. The names follow the C enumerators."""

    OK = 0
    INVALID_ARGUMENT = 1
    INVALID_IMAGE = 2
    INVALID_CONFIG = 3
    UNSUPPORTED_DICTIONARY = 4
    CANDIDATE_OVERFLOW = 5
    MARKER_OVERFLOW = 6
    CUDA_ERROR = 7
    NOT_INITIALIZED = 8
    INTERNAL_ERROR = 100


class MemorySpace(enum.IntEnum):
    """The memory space an input buffer lives in."""

    HOST_PAGEABLE = 0
    HOST_PINNED = 1
    MANAGED = 2
    DEVICE = 3


class CornerRefine(enum.IntEnum):
    """Corner refinement method."""

    NONE = 0
    SUBPIX = 1


class Error(RuntimeError):
    """Raised when a call into the library fails.

    Carries the status so that a caller can branch on it. The text also carries
    whatever CUDA recorded, because a bare "kCudaError" says nothing about which
    API call failed.
    """

    def __init__(self, status, message="", operation=""):
        self.status = Status(status) if status in Status.__members__.values() else status
        self.message = message
        self.operation = operation
        parts = [f"{operation or 'call'} failed: {_status_name(status)}"]
        if message:
            parts.append(message)
        cuda_message = _lib.aruco3cuda_last_cuda_error_message().decode("utf-8", "replace")
        if cuda_message:
            parts.append(cuda_message)
        super().__init__("\n  ".join(parts))


class DictionaryInfo:
    """Metadata of one built-in dictionary."""

    __slots__ = ("name", "marker_size", "code_count", "max_correction_bits")

    def __init__(self, name, marker_size, code_count, max_correction_bits):
        self.name = name
        self.marker_size = marker_size
        self.code_count = code_count
        self.max_correction_bits = max_correction_bits

    def __repr__(self):
        return (
            f"DictionaryInfo(name={self.name!r}, marker_size={self.marker_size}, "
            f"code_count={self.code_count}, max_correction_bits={self.max_correction_bits})"
        )


class Detection:
    """One detected marker.

    corners holds four (x, y) pairs in the full resolution of the input image,
    already reordered by the rotation the dictionary match found. rotation is
    the value from before that reordering, so a stage that rotates again by it
    ends up 90 degrees off.
    """

    __slots__ = ("id", "rotation", "corners")

    def __init__(self, marker_id, rotation, corners):
        self.id = marker_id
        self.rotation = rotation
        self.corners = corners

    def __repr__(self):
        return f"Detection(id={self.id}, rotation={self.rotation}, corners={self.corners})"


class Detections:
    """The results of one frame, on the host.

    Iterating yields Detection objects, and len() is the number found, so the
    common case reads as a sequence. accepted_total and truncated are there for
    the case that is easy to miss: the detector stops at max_markers, and
    without checking them a truncated frame looks like a complete one.
    """

    __slots__ = ("detections", "accepted_total", "truncated")

    def __init__(self, detections, accepted_total, truncated):
        self.detections = detections
        self.accepted_total = accepted_total
        self.truncated = truncated

    def __len__(self):
        return len(self.detections)

    def __iter__(self):
        return iter(self.detections)

    def __getitem__(self, index):
        return self.detections[index]

    def __repr__(self):
        return (
            f"Detections({len(self.detections)} found, accepted_total="
            f"{self.accepted_total}, truncated={self.truncated})"
        )


class DeviceDetections:
    """Device pointers to the results, for a stage that stays on the GPU.

    The detector owns this memory. The pointers become invalid when the detector
    is closed or the input dimensions change, and the contents are not final
    until the stream has completed.
    """

    __slots__ = (
        "ids",
        "rotations",
        "corner_x",
        "corner_y",
        "source",
        "count",
        "accepted_total",
        "capacity",
    )

    def __init__(self, raw):
        self.ids = raw.ids
        self.rotations = raw.rotations
        self.corner_x = raw.corner_x
        self.corner_y = raw.corner_y
        self.source = raw.source
        self.count = raw.count
        self.accepted_total = raw.accepted_total
        self.capacity = raw.capacity

    def __repr__(self):
        return f"DeviceDetections(capacity={self.capacity}, ids=0x{self.ids or 0:x})"


class WorkspaceStatistics:
    """How the device workspace is being used.

    allocation_count staying at 1 after initialize is the property worth
    watching: it means nothing is being allocated per frame.
    """

    __slots__ = (
        "allocation_count",
        "reallocation_count",
        "capacity_bytes",
        "used_bytes",
        "peak_used_bytes",
        "exhausted_count",
    )

    def __init__(self, raw):
        self.allocation_count = raw.allocation_count
        self.reallocation_count = raw.reallocation_count
        self.capacity_bytes = raw.capacity_bytes
        self.used_bytes = raw.used_bytes
        self.peak_used_bytes = raw.peak_used_bytes
        self.exhausted_count = raw.exhausted_count

    def __repr__(self):
        return (
            f"WorkspaceStatistics(allocation_count={self.allocation_count}, "
            f"reallocation_count={self.reallocation_count}, "
            f"peak_used_bytes={self.peak_used_bytes})"
        )


class Config(_binding.Aruco3CudaConfig):
    """Detection settings.

    Subclasses the C struct so that it can be handed to the library directly and
    so that every field is readable and writable by its own name. Constructed
    with the library defaults, not with zeros, because an all-zero configuration
    is not a valid one.

    Only two combinations are accepted: the ArUco3 strategy with subpixel corner
    refinement, or neither. initialize() rejects the other two.

        Config(max_width_px=1280, max_height_px=720)
    """

    def __init__(self, **overrides):
        super().__init__()
        _check(_lib.aruco3cuda_config_defaults(ctypes.byref(self)), "config_defaults")
        for name, value in overrides.items():
            if not hasattr(type(self), name):
                raise AttributeError(f"Config has no field named {name!r}")
            setattr(self, name, value)

    def validate(self):
        """Check the settings without creating a detector.

        Raises Error with the offending field named in the message.
        """
        buffer = ctypes.create_string_buffer(_binding.kMessageCapacity)
        status = _lib.aruco3cuda_config_validate(
            ctypes.byref(self), buffer, _binding.kMessageCapacity
        )
        if status != Status.OK:
            raise Error(status, buffer.value.decode("utf-8", "replace"), "validate")

    def __repr__(self):
        fields = ", ".join(f"{name}={getattr(self, name)}" for name, _ in self._fields_)
        return f"Config({fields})"


class Stream:
    """A CUDA stream.

    Passing an explicit stream is what lets the detector capture one frame worth
    of launches as a CUDA Graph. CUDA cannot capture on the default stream, so
    passing None instead takes the path that issues each stage separately.
    """

    def __init__(self):
        _require_cudart()
        handle = ctypes.c_void_p()
        _check_cuda(_cudart.cudaStreamCreate(ctypes.byref(handle)), "cudaStreamCreate")
        self._handle = handle

    @property
    def handle(self):
        """The raw cudaStream_t, as an integer address."""
        return self._handle

    def close(self):
        """Destroy the stream. Safe to call more than once."""
        if getattr(self, "_handle", None) is not None and self._handle.value is not None:
            _cudart.cudaStreamDestroy(self._handle)
            self._handle = ctypes.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:  # pragma: no cover - never raise from a finalizer
            pass


class DeviceImage:
    """A pitched 8-bit grayscale image in device memory.

    cudaMallocPitch pads each row to the alignment the device prefers, which is
    why pitch_bytes is not the same as width_px. Assuming they are equal works
    on the host and then reads the wrong pixels on the device.
    """

    def __init__(self, width_px, height_px):
        _require_cudart()
        if width_px < 1 or height_px < 1:
            raise ValueError(f"the dimensions must be positive: {width_px}x{height_px}")
        pointer = ctypes.c_void_p()
        pitch = ctypes.c_size_t()
        _check_cuda(
            _cudart.cudaMallocPitch(
                ctypes.byref(pointer), ctypes.byref(pitch), width_px, height_px
            ),
            "cudaMallocPitch",
        )
        self._pointer = pointer
        self._pitch_bytes = pitch.value
        self._width_px = width_px
        self._height_px = height_px

    @classmethod
    def from_host(cls, data, width_px, height_px):
        """Allocate and upload in one step.

        data is anything supporting the buffer protocol, including bytes,
        bytearray, and a numpy array. It must hold width_px * height_px bytes,
        with rows contiguous.
        """
        image = cls(width_px, height_px)
        try:
            image.upload(data)
        except Exception:
            image.close()
            raise
        return image

    def upload(self, data):
        """Copy a host buffer into this image."""
        view = memoryview(data).cast("B")
        expected = self._width_px * self._height_px
        if view.nbytes != expected:
            raise ValueError(
                f"expected {expected} bytes for {self._width_px}x{self._height_px} "
                f"but received {view.nbytes}"
            )
        source = (ctypes.c_uint8 * view.nbytes).from_buffer_copy(view)
        _check_cuda(
            _cudart.cudaMemcpy2D(
                self._pointer,
                self._pitch_bytes,
                ctypes.cast(source, ctypes.c_void_p),
                self._width_px,
                self._width_px,
                self._height_px,
                1,  # cudaMemcpyHostToDevice
            ),
            "cudaMemcpy2D",
        )

    @property
    def pointer(self):
        """The device address, as an integer."""
        return self._pointer.value

    @property
    def pitch_bytes(self):
        return self._pitch_bytes

    @property
    def width_px(self):
        return self._width_px

    @property
    def height_px(self):
        return self._height_px

    @property
    def __cuda_array_interface__(self):
        """Let CuPy, PyTorch, and Numba wrap this buffer without copying."""
        return {
            "shape": (self._height_px, self._width_px),
            "typestr": "|u1",
            "data": (self._pointer.value, False),
            "strides": (self._pitch_bytes, 1),
            "version": 3,
        }

    def close(self):
        """Free the device memory. Safe to call more than once."""
        if getattr(self, "_pointer", None) is not None and self._pointer.value is not None:
            _cudart.cudaFree(self._pointer)
            self._pointer = ctypes.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:  # pragma: no cover - never raise from a finalizer
            pass


class Detector:
    """The detector.

    One instance must not be used from several threads at once, matching the
    C++ class it wraps.
    """

    def __init__(self):
        handle = _lib.aruco3cuda_detector_create()
        if not handle:
            raise MemoryError("aruco3cuda_detector_create returned NULL")
        self._handle = ctypes.c_void_p(handle)
        self._capacity = 0
        self._ids = None
        self._corners = None
        self._rotations = None

    def initialize(self, dictionary, config=None):
        """Allocate the workspace and transfer the dictionary to the device.

        dictionary is the name of a built-in dictionary. config defaults to the
        library defaults, which size the workspace for 3840x2160; pass the
        dimensions of the images actually being processed unless that is what is
        wanted.

        Synchronizes the whole device once.
        """
        config = Config() if config is None else config
        buffer = ctypes.create_string_buffer(_binding.kMessageCapacity)
        status = _lib.aruco3cuda_detector_initialize(
            self._handle,
            _encode(dictionary),
            ctypes.byref(config),
            buffer,
            _binding.kMessageCapacity,
        )
        if status != Status.OK:
            raise Error(status, buffer.value.decode("utf-8", "replace"), "initialize")
        # Allocated once, here, for the same reason the C++ workspace is: a
        # per-frame allocation on the host would undo the point of not making
        # one on the device.
        self._capacity = int(config.max_markers)
        self._ids = (ctypes.c_int32 * self._capacity)()
        self._rotations = (ctypes.c_int32 * self._capacity)()
        self._corners = (ctypes.c_float * (self._capacity * 8))()

    def detect(self, image, stream=None):
        """Issue a detection onto the stream.

        image is a DeviceImage or any object exposing __cuda_array_interface__
        with 8-bit contents, such as a CuPy or PyTorch array. Host buffers are
        rejected on purpose: use DeviceImage.from_host so that the transfer is
        visible in the calling code.

        Returns immediately. Nothing has run when it does.
        """
        view = _to_image_view(image)
        buffer = ctypes.create_string_buffer(_binding.kMessageCapacity)
        status = _lib.aruco3cuda_detector_detect(
            self._handle,
            ctypes.byref(view),
            _stream_handle(stream),
            buffer,
            _binding.kMessageCapacity,
        )
        if status != Status.OK:
            raise Error(status, buffer.value.decode("utf-8", "replace"), "detect")

    def download(self, stream=None):
        """Wait for the stream and return the results on the host.

        This is the only synchronizing call in steady state.
        """
        if self._capacity == 0:
            raise Error(Status.NOT_INITIALIZED, "initialize() has not been called", "download")
        result = _binding.Aruco3CudaDetectionBuffer()
        result.ids = self._ids
        result.rotations = self._rotations
        result.corners = self._corners
        result.capacity = self._capacity
        buffer = ctypes.create_string_buffer(_binding.kMessageCapacity)
        status = _lib.aruco3cuda_detector_download(
            self._handle,
            _stream_handle(stream),
            ctypes.byref(result),
            buffer,
            _binding.kMessageCapacity,
        )
        # MARKER_OVERFLOW means the detector truncated at max_markers. The
        # results are still valid, so it is reported through the truncated flag
        # rather than as an exception.
        if status not in (Status.OK, Status.MARKER_OVERFLOW):
            raise Error(status, buffer.value.decode("utf-8", "replace"), "download")
        detections = []
        for index in range(result.count):
            base = index * 8
            corners = tuple(
                (self._corners[base + corner * 2], self._corners[base + corner * 2 + 1])
                for corner in range(4)
            )
            detections.append(
                Detection(int(self._ids[index]), int(self._rotations[index]), corners)
            )
        return Detections(detections, int(result.accepted_total), bool(result.truncated))

    def device_detections(self):
        """Borrow the device pointers to the results of the most recent detect.

        Does not synchronize. The contents are not final until the stream has
        completed.
        """
        raw = _binding.Aruco3CudaDeviceDetections()
        _check(
            _lib.aruco3cuda_detector_device_detections(self._handle, ctypes.byref(raw)),
            "device_detections",
        )
        return DeviceDetections(raw)

    def workspace_statistics(self):
        """Report how the device workspace is being used."""
        raw = _binding.Aruco3CudaWorkspaceStatistics()
        _check(
            _lib.aruco3cuda_detector_workspace_statistics(self._handle, ctypes.byref(raw)),
            "workspace_statistics",
        )
        return WorkspaceStatistics(raw)

    def close(self):
        """Release the detector and its device memory. Safe to call twice."""
        if getattr(self, "_handle", None) is not None and self._handle.value is not None:
            _lib.aruco3cuda_detector_destroy(self._handle)
            self._handle = ctypes.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:  # pragma: no cover - never raise from a finalizer
            pass


def version():
    """Return the library version, for example "0.1.0"."""
    return __version__


def dictionaries():
    """Return the metadata of every built-in dictionary."""
    result = []
    for index in range(_lib.aruco3cuda_builtin_dictionary_count()):
        info = _binding.Aruco3CudaDictionaryInfo()
        _check(
            _lib.aruco3cuda_builtin_dictionary_at(index, ctypes.byref(info)),
            "builtin_dictionary_at",
        )
        result.append(_to_dictionary_info(info))
    return result


def find_dictionary(name):
    """Look up one built-in dictionary by name.

    Raises Error with UNSUPPORTED_DICTIONARY if the name is unknown.
    """
    info = _binding.Aruco3CudaDictionaryInfo()
    status = _lib.aruco3cuda_find_builtin_dictionary(_encode(name), ctypes.byref(info))
    if status != Status.OK:
        known = ", ".join(entry.name for entry in dictionaries())
        raise Error(status, f"unknown dictionary {name!r}. Known: {known}", "find_dictionary")
    return _to_dictionary_info(info)


def marker_bits(name, marker_id, rotation=0):
    """Return the bit pattern of one marker as rows of 0 and 1.

    This is what a renderer needs, and it is the only reason to reach into a
    dictionary directly.

        rows = marker_bits("DICT_ARUCO_MIP_36h12", 42)
        len(rows), len(rows[0])   # 6, 6
    """
    info = find_dictionary(name)
    count = info.marker_size * info.marker_size
    buffer = (ctypes.c_uint8 * count)()
    status = _lib.aruco3cuda_dictionary_marker_bits(
        _encode(name), marker_id, rotation, buffer, count
    )
    if status != Status.OK:
        raise Error(
            status,
            f"id {marker_id} or rotation {rotation} is out of range for {name} "
            f"({info.code_count} ids)",
            "marker_bits",
        )
    return [
        [int(buffer[row * info.marker_size + column]) for column in range(info.marker_size)]
        for row in range(info.marker_size)
    ]


def _status_name(status):
    return _lib.aruco3cuda_status_string(int(status)).decode("utf-8", "replace")


def _check(status, operation):
    if status != Status.OK:
        raise Error(status, "", operation)


def _check_cuda(error, operation):
    if error != 0:
        text = _cudart.cudaGetErrorString(error).decode("utf-8", "replace")
        raise Error(Status.CUDA_ERROR, f"{operation}: {text}", operation)


def _require_cudart():
    if _cudart is None:
        raise Error(
            Status.CUDA_ERROR,
            "the CUDA runtime could not be loaded. Set ARUCO3CUDA_CUDART to the "
            "path of libcudart.so if it is installed somewhere unusual.",
            "load_cudart",
        )


def _encode(value):
    return value.encode("utf-8") if isinstance(value, str) else value


def _to_dictionary_info(info):
    return DictionaryInfo(
        info.name.decode("utf-8"),
        info.marker_size,
        info.code_count,
        info.max_correction_bits,
    )


def _stream_handle(stream):
    if stream is None:
        return None
    if isinstance(stream, Stream):
        return stream.handle
    return ctypes.c_void_p(int(stream))


def _to_image_view(image):
    """Build the C image view from a DeviceImage or a __cuda_array_interface__."""
    if isinstance(image, DeviceImage):
        view = _binding.Aruco3CudaImage()
        view.data = image.pointer
        view.width_px = image.width_px
        view.height_px = image.height_px
        view.pitch_bytes = image.pitch_bytes
        view.space = int(MemorySpace.DEVICE)
        return view

    interface = getattr(image, "__cuda_array_interface__", None)
    if interface is None:
        raise TypeError(
            "detect() takes device-resident input: a DeviceImage, or any object "
            "exposing __cuda_array_interface__ such as a CuPy or PyTorch array. "
            "For a host buffer use DeviceImage.from_host(data, width, height), so "
            "that the transfer is visible where it happens."
        )
    if interface.get("typestr") not in ("|u1", "u1", "<u1", ">u1"):
        raise TypeError(
            f"the input must be 8-bit, but typestr is {interface.get('typestr')!r}"
        )
    shape = interface.get("shape")
    if not shape or len(shape) != 2:
        raise TypeError(f"the input must be two dimensional, but shape is {shape!r}")
    height_px, width_px = int(shape[0]), int(shape[1])
    pointer, read_only = interface["data"]
    del read_only  # the detector only reads the image
    strides = interface.get("strides")
    if strides is None:
        pitch_bytes = width_px
    else:
        if int(strides[1]) != 1:
            raise TypeError(
                "the columns of the input must be contiguous, but the column "
                f"stride is {strides[1]}"
            )
        pitch_bytes = int(strides[0])
    view = _binding.Aruco3CudaImage()
    view.data = pointer
    view.width_px = width_px
    view.height_px = height_px
    view.pitch_bytes = pitch_bytes
    view.space = int(MemorySpace.DEVICE)
    return view
