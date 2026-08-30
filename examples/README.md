# Samples

Two small programs that together run the whole public API without needing a camera,
an image library, or any file that is not produced here.

Each one exists twice, in C++ and in Python. The two take the same options and
print the same lines, and the tests compare their output, so neither can drift
into being the one that is actually maintained.

| Program | Needs a GPU | What it shows |
| --- | --- | --- |
| `generate_marker` | No | Turning a packed dictionary entry back into a picture |
| `detect_image` | Yes | Sizing the workspace, uploading, detecting on a stream, and reading the results |

Nothing here depends on OpenCV, and the Python pair depends on nothing outside
the standard library either. The library does not either, and a sample with a
wider dependency than the thing it demonstrates teaches the wrong lesson about
what is required.

## Build

The samples are built with the rest of the project by default, so there is nothing
extra to do.

```bash
cmake --preset native && cmake --build --preset native
# binaries in build/native/examples/
```

Pass `-DARUCO3CUDA_BUILD_EXAMPLES=OFF` to skip them. They default to off when this
project is pulled into another one with `add_subdirectory`.

### Building against an installed package

This is the path a consumer takes, and it is worth running at least once, because it
is the only form that exercises `find_package` and the exported target names.

```bash
cmake -S . -B build/install -DCMAKE_BUILD_TYPE=Release \
  -DARUCO3CUDA_BUILD_REFERENCE=OFF -DARUCO3CUDA_BUILD_TESTS=OFF \
  -DARUCO3CUDA_BUILD_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX=/your/prefix
cmake --build build/install -j && cmake --install build/install

cmake -S examples -B build/examples -DCMAKE_PREFIX_PATH=/your/prefix
cmake --build build/examples
```

`examples/CMakeLists.txt` is written to work both ways round, so the file in this
directory is the same one CI builds standalone. It enables only the `CXX` language:
the samples call the CUDA runtime but compile no device code, which is why a
consumer does not need the CUDA language enabled in their own project.

## Run, in C++

```bash
cd build/native/examples
./generate_marker --id 42 --size 200 --margin 40 --output marker.pgm
./detect_image --input marker.pgm
```

```
wrote marker.pgm: 280x280, DICT_ARUCO_MIP_36h12 id=42, marker 200 px, margin 40 px
aruco3cuda   : 0.1.0
input        : marker.pgm (280x280)
dictionary   : DICT_ARUCO_MIP_36h12 (6x6 bits, 250 ids)
configuration: ArUco3 on, corner refinement kSubpix, explicit stream, 1 iteration(s)
detectable   : markers of side 46.0 px or more in this image
workspace    : 1 allocation(s), 0 reallocation(s), peak 7.54 MiB
detections   : 1 (accepted 1)
  [0] id=42 rotation=2 corners= (39.59, 39.59) (239.41, 39.59) (239.42, 239.42) (39.59, 239.41)
```

The marker was drawn with its outer edge at 40 and 239, and the refined corners land
within 0.5 px of that. `rotation=2` is not an error: it records the rotation the
dictionary match had to undo, and the corners in `corners_` have already had it
applied. `--repeat 8` shows that the allocation count stays at 1 no matter how many
frames go through.

## Run, in Python

The Python samples are the same two programs against the same library, through
the C ABI and `ctypes`. Nothing is compiled for them, so they run against the
build tree directly.

```bash
export PYTHONPATH=build/native/python:examples/python
python3 examples/python/generate_marker.py --id 42 --size 200 --margin 40 --output marker.pgm
python3 examples/python/detect_image.py --input marker.pgm
```

The output is the same, line for line, and `generate_marker.py` writes the same
bytes as its C++ counterpart. Reading the two side by side is the point: the same
sequence of calls, in two languages.

Using the library directly is shorter than the sample, because the sample also
parses arguments and reads a file.

```python
import aruco3cuda

config = aruco3cuda.Config(max_width_px=280, max_height_px=280)
with aruco3cuda.Detector() as detector:
    detector.initialize("DICT_ARUCO_MIP_36h12", config)
    with aruco3cuda.DeviceImage.from_host(pixels, 280, 280) as image:
        with aruco3cuda.Stream() as stream:
            detector.detect(image, stream)
            for detection in detector.download(stream):
                print(detection.id, detection.corners)
```

`detect()` takes device-resident input: a `DeviceImage`, or anything exposing
`__cuda_array_interface__`, which is how a CuPy or PyTorch array is handed over
without a copy. A host buffer is rejected with a message naming
`DeviceImage.from_host`, so that the transfer is visible where it happens rather
than hidden inside every frame. The design is written up in
[C ABI and Python binding](../docs/design/c-abi-and-python.md).

## If nothing is detected

Almost always the marker is too small for the ArUco3 strategy. It segments a
downscaled image, so a marker below

    min_side_length_canonical_img_px_ + max(width, height) * min_marker_length_ratio_original_img_

pixels on a side cannot be found at all. `detect_image` prints that bound on the
`detectable` line, and `--no-aruco3` removes it at the cost of segmenting the image at
full resolution. Two of the automated tests pin this down from both sides: the same
40 px marker in an 840x840 image is missed with ArUco3 on and found with it off.

The second most common cause is a missing quiet zone. A marker whose black border
touches the edge of the image is rejected by `min_distance_to_border_px_`, which is
why `generate_marker` has a `--margin` option and why it defaults to a non-zero
value.

## Notes on the code

`pgm.hpp`, `pgm.cpp`, and `python/pgm.py` are the samples' own scaffolding, not
part of the library.
Binary PGM (P5) was chosen because it stores exactly what the detector takes, 8-bit
grayscale, so no conversion step hides between the file and the API being shown. The
parser rejects ASCII PGM (P2) and 16-bit PGM by name rather than misreading them.

Every option is validated before it reaches the library, in keeping with the rule in
[CONTRIBUTING.md](../CONTRIBUTING.md) that external input is not trusted. The
automated tests cover the nominal, error, and boundary cases of all four
programs, including the generate-then-detect round trip in each language:
[C++](../test/cli/CMakeLists.txt) and [Python](../test/python/CMakeLists.txt).

The samples are excluded from the coverage filter in
`cmake/Aruco3CudaOptions.cmake`. They are demonstration code, and measuring them
would report a number about the samples rather than about the library. The
Python files are not measured at all, which is recorded as an open question in
[C ABI and Python binding](../docs/design/c-abi-and-python.md).

### Provenance of the rendering rule

Both `generate_marker` programs reproduce the rule that
`cv::aruco::Dictionary::generateImageMarker` uses in OpenCV 4.x (Apache-2.0):
the border cells are black, an inner bit of 1 is
white and 0 is black, and the cell grid is scaled up by nearest neighbour. Only the
observable rule is reproduced; no OpenCV code is copied, and nothing here derives
from the GPLv3 ArUco distribution. The output was checked against
`generateImageMarker` for ids 0, 7, 13, 42, and 249 at sides of 8, 61, 96, 100, 200,
203, and 240 pixels, with 1 and 2 border bits, and is identical byte for byte in
every case. Three of those combinations are kept as tests that compare the C++
and Python renderers to each other on every run. See [Intellectual property and licensing](../docs/ip-and-licensing.md).
