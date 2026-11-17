#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
mkdir -p evidence
cmake -S cpu/standalone -B build/cpu-opt -G Ninja -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR=/opt/opencv/lib/cmake/opencv4 -DARUCO3CUDA_BUILD_TESTS=OFF -DARUCO3CPU_NATIVE=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON > evidence/configure.log 2>&1
cmake --build build/cpu-opt --target aruco3cuda_cpu_benchmark --parallel 4 > evidence/build.log 2>&1
cp build/cpu-opt/CMakeCache.txt evidence/CMakeCache.txt
cp build/cpu-opt/compile_commands.json evidence/compile_commands.json
cp /opt/opencv/share/aruco3cuda/opencv-provenance.json evidence/opencv-provenance.json
ldd build/cpu-opt/cpu/aruco3cuda_cpu_benchmark > evidence/ldd.log
sha256sum build/cpu-opt/cpu/aruco3cuda_cpu_benchmark > evidence/binary.sha256
printf 'Build complete\n'
