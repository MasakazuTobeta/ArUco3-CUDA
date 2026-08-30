# Samples

Two small programs that together run the whole public API without needing a camera,
an image library, or any file that is not produced here.

| Program | Needs a GPU | What it shows |
| --- | --- | --- |
| `generate_marker` | No | Turning a packed dictionary entry back into a picture |
| `detect_image` | Yes | Sizing the workspace, uploading, detecting on a stream, and reading the results |

Neither depends on OpenCV. The library does not either, and a sample with a wider
dependency than the thing it demonstrates teaches the wrong lesson about what is
required.

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

## Run

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

`pgm.hpp` and `pgm.cpp` are the samples' own scaffolding, not part of the library.
Binary PGM (P5) was chosen because it stores exactly what the detector takes, 8-bit
grayscale, so no conversion step hides between the file and the API being shown. The
parser rejects ASCII PGM (P2) and 16-bit PGM by name rather than misreading them.

Every option is validated before it reaches the library, in keeping with the rule in
[CONTRIBUTING.md](../CONTRIBUTING.md) that external input is not trusted. The
[automated CLI tests](../test/cli/CMakeLists.txt) cover the nominal, error, and
boundary cases of both programs, including the generate-then-detect round trip.

The samples are excluded from the coverage filter in
`cmake/Aruco3CudaOptions.cmake`. They are demonstration code, and measuring them
would report a number about the samples rather than about the library.

### Provenance of the rendering rule

`generate_marker` reproduces the rule `cv::aruco::Dictionary::generateImageMarker`
uses in OpenCV 4.x (Apache-2.0): the border cells are black, an inner bit of 1 is
white and 0 is black, and the cell grid is scaled up by nearest neighbour. Only the
observable rule is reproduced; no OpenCV code is copied, and nothing here derives
from the GPLv3 ArUco distribution. The output was checked against
`generateImageMarker` for ids 0, 7, 13, 42, and 249 at sides of 8, 61, 96, 100, 200,
203, and 240 pixels, with 1 and 2 border bits, and is identical byte for byte in
every case. See [Intellectual property and licensing](../docs/ip-and-licensing.md).
