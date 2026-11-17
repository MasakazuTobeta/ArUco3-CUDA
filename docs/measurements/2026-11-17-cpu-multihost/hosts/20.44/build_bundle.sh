#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd /workspace
cmake -S cpu/standalone -B build/cpu-portable -G Ninja -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=/opt/opencv/lib/cmake/opencv4 -DARUCO3CUDA_BUILD_TESTS=OFF -DARUCO3CPU_NATIVE=OFF
cmake --build build/cpu-portable --target aruco3cuda_cpu_benchmark --parallel 12
mkdir -p portable/bin portable/lib portable/metadata
cp build/cpu-portable/cpu/aruco3cuda_cpu_benchmark portable/bin/
ldd portable/bin/aruco3cuda_cpu_benchmark > portable/metadata/ldd-build.txt
python3 - <<'PY'
import pathlib,subprocess,shutil
binary='portable/bin/aruco3cuda_cpu_benchmark'
for line in subprocess.check_output(['ldd',binary],text=True).splitlines():
    parts=line.split()
    if len(parts)>2 and parts[1]=='=>':
        name,path=parts[0],parts[2]
        if name not in {'libc.so.6','libm.so.6','libdl.so.2','librt.so.1','libpthread.so.0'}:
            shutil.copy2(path,'portable/lib/'+name)
PY
cp /opt/opencv/share/aruco3cuda/opencv-provenance.json portable/metadata/
cp build/cpu-portable/CMakeCache.txt build/cpu-portable/compile_commands.json portable/metadata/ 2>/dev/null || cp build/cpu-portable/CMakeCache.txt portable/metadata/
cat > portable/metadata/opencv_info.cpp <<'CPP'
// SPDX-License-Identifier: Apache-2.0
#include <opencv2/core.hpp>
#include <iostream>
int main() { std::cout << cv::getBuildInformation(); }
CPP
g++ portable/metadata/opencv_info.cpp -I/opt/opencv/include/opencv4 -L/opt/opencv/lib -lopencv_core -o portable/bin/opencv_info
portable/bin/opencv_info > portable/metadata/opencv-build-info.txt
find portable -type f -exec sha256sum {} + > portable-checksums.txt
tar -czf portable.tar.gz portable portable-checksums.txt
