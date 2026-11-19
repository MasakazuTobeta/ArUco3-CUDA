#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run after the primary AprilTag GPU study on this host.
set -euo pipefail
source_dir=$1
build_dir=$2
inputs=$3
original=$4
output=$5
cores=$6
shift 6
runner="$source_dir/docs/measurements/2026-11-19-performance-refresh"

cmake --build "$build_dir" -j 4 --target aruco3cuda_bench_cli aruco3cuda_gpu_sequence \
    aruco3cuda_gpu_quality_audit apriltag_cpu_compare
cat /opt/opencv/share/aruco3cuda/opencv-provenance.json
nvcc --version
date -Is

result=0
python3 "$runner/run_aruco.py" --build "$build_dir" --inputs "$inputs" \
    --output "$output/aruco" --cores "$cores" \
    --groups video fixed historical memory native quality "$@" || result=1
python3 "$runner/run_april_gpu.py" --source "$source_dir" --binaries "$build_dir/apriltag/benchmark" \
    --original-inputs "$original" --new-inputs "$inputs" --output "$output/apriltag-extra" \
    --cpu "${cores%%,*}" --sets native-photos controls || result=1
python3 "$source_dir/apriltag/cpu/measure.py" "$build_dir/apriltag/cpu/apriltag_cpu_compare" \
    "$inputs/aruco-steady" "$output/apriltag-cpu" --cores "$cores" \
    --resolutions 720p 1080p 2160p || result=1
exit "$result"
