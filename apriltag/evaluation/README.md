# AprilTag evaluation

## Purpose

Compare CPU with OpenCV 4.14 and GPU with NVIDIA cuAprilTags on the same input pixels.
Check the truth marker by marker before comparing steady-state latency.

## Scope

The CPU comparison covers tag16h5, tag25h9, tag36h10 and tag36h11. NVIDIA's CUDA
backend supports tag36h11. Both detectors return IDs and ordered image corners.
Pose computation, capture latency and tracking fall outside these measurements.

## Current state

Generate independent OpenCV-rendered inputs with the existing [quality renderer](../../tools/cpu_quality/quality_scenes.md).
The following commands run from the repository root in the pinned OpenCV development environment.

```bash
cmake -S cpu/standalone -B build/quality -G Ninja -DARUCO3CUDA_BUILD_TESTS=OFF
cmake --build build/quality --target aruco3cuda_cpu_quality_corpus
build/quality/cpu_quality/aruco3cuda_cpu_quality_corpus build/corpus-36h11 640 480 DICT_APRILTAG_36h11
```

Each family has 96 frames and 136 annotated markers, including repeated IDs at
distinct locations, occlusion, lighting changes and empty frames. For an independent
payload-error check, render [AprilRobotics' published ID 17](https://github.com/AprilRobotics/apriltag-imgs/blob/f3fd9a7add5bfd82a886fc65240fdb8e3c9ac5a1/tag36h11/tag36_11_00017.png)
with zero through four flipped cells, three marker sizes and all four rotations.
The script checks the original PNG's SHA-256 and uses Pillow for nearest-neighbor scaling.

```bash
curl -fsSL https://raw.githubusercontent.com/AprilRobotics/apriltag-imgs/f3fd9a7add5bfd82a886fc65240fdb8e3c9ac5a1/tag36h11/tag36_11_00017.png -o build/tag-17.png
python3 apriltag/evaluation/damage_corpus.py build/tag-17.png build/tag-damage
```

An additional 1920×1080 NVIDIA integration fixture contains one tag36h11 marker.
Its expected ID is 0. In OpenCV corner order its nominal pixel coordinates are
(808,429), (1044,429), (1044,665), (808,665), with the upstream test allowing
2 px per coordinate. The original RGBA PNG has SHA-256
`6ea20e20dc8f25567276f0cf80f4a89ddd9d7b69707cd08e28946ca8915dbe3d`.
Use Pillow 10.2.0 to convert it to mode L and write a PGM.
[NVIDIA fixture and test](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_apriltag/blob/a32cf16997c5405edde1b4e00bfba043e3ed2e5b/isaac_ros_apriltag/test/isaac_ros_apriltag_pol_test.py)

For photographs with visible tags, use the three CC BY-SA 2.0 NASA Swarmathon
images from the pinned AprilRobotics source. The accompanying 47 reference
quads are upstream detector outputs. They permit a position-by-position comparison,
including multiple occurrences of ID 0, but have no independent subpixel
ground truth. The script verifies all six input files and converts each JPEG to
grayscale with Pillow (tested with 10.2.0).
[Photo credits and original images](https://github.com/AprilRobotics/apriltag/blob/efa633d9dc6f4ee33bb79b0592250f4aa71e90a8/test/data/README.md)

```bash
python3 apriltag/evaluation/field_corpus.py /path/to/apriltag/test/data build/field-corpus
```

## Goals

Collect observations with the [CPU comparison](../cpu/observe.py) or
[GPU sequence runner](../benchmark/README.md). The scorer requires exactly one
observation per manifest entry, including empty frames, and checks each image hash.
It uses the project's [ground-truth matcher](../../tools/evaluate/accuracy.md)
to pair by centroid before checking IDs. A rotated corner sequence counts as
an orientation error. Compare geometry on the common detections and inspect any
baseline truth lost by the candidate, even if the total recall improves.

```bash
c++ -std=c++17 -O2 -Itools/evaluate -Itools/report apriltag/evaluation/score.cpp \
  tools/evaluate/accuracy.cpp -o build/tag-score
python3 apriltag/evaluation/score.py build/corpus-36h11/manifest.json \
  build/comparison/nvidia-device/observations.jsonl \
  build/comparison/project-device/observations.jsonl --scorer build/tag-score
```

## Open questions

How do detector latency and corner accuracy generalize to annotated deployment video
across marker sizes, lighting and camera motion?
