# AprilTag CPU detector for small markers

Use this experimental tag36h11 detector with nonempty Gray8 images. Each dimension
must be below 32,768 pixels. The detector accepts row padding and regions of
interest. Results contain the ID, corrected bit count and four canonically ordered
corners in input image coordinates. Use a separate instance for each concurrent
caller.

## Build and use

Build with OpenCV 4.14.0. Supply its CMake package path via `OpenCV_DIR` if needed.

```sh
cmake -S apriltag/small_cpu -B build/apriltag-small-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build/apriltag-small-cpu --parallel
ctest --test-dir build/apriltag-small-cpu --output-on-failure
```

Consumers can use `add_subdirectory` and link
`aruco3cuda::apriltag_small_cpu`. Use `Detector` for individual images and
`VideoDetector` for presentation-order frames.
[Detector API](include/aruco3cuda/apriltag/small_cpu/detector.hpp) ·
[Video API](include/aruco3cuda/apriltag/small_cpu/video_detector.hpp)

### Ordered video search

Keep one `VideoDetector` per stream. Call `reset()` before processing an
unrelated stream. A change in image size resets the history automatically.

```cpp
#include <aruco3cuda/apriltag/small_cpu/video_detector.hpp>

aruco3cuda::apriltag_small_cpu::VideoDetector detector;
const auto detections = detector.detect(frame);
```

Enable `ARUCO3_APRILTAG_SMALL_BENCHMARK` to build the replay executable.
The benchmark also requires OpenCV imgcodecs and objdetect.

```sh
aruco3cuda_apriltag_small_sequence --search video 2 2 5 frame-001.pgm frame-002.pgm
```

[Current benchmark conditions and scores](../../Benchmark.md)
