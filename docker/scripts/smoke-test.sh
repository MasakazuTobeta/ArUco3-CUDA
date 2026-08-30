#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Purpose:
#   Confirm that the container environment is genuinely usable, going further than
#   the environment check. It verifies that nvcc can compile a translation unit
#   that includes OpenCV, that the resulting executable runs on the GPU, and that
#   the OpenCV ArUco3 detection strategy behaves as expected.
#
# Role:
#   This is a smoke test of the container environment, not a unit test of the
#   project. The project's own tests are run with `ctest`.
set -euo pipefail

readonly kCudaArch="${ARUCO3_CUDA_ARCH:?ARUCO3_CUDA_ARCH is not set}"
readonly kOpenCvPrefix="${ARUCO3_OPENCV_PREFIX:-/opt/opencv}"
# Default to the smoke source installed into the image; override with
# ARUCO3_SMOKE_DIR to use the repository directly.
kSourceDir="${ARUCO3_SMOKE_DIR:-/opt/aruco3cuda/smoke}"
if [ ! -f "${kSourceDir}/aruco3_smoke.cu" ]; then
  echo "[smoke-test] ${kSourceDir}/aruco3_smoke.cu does not exist." >&2
  echo "[smoke-test] specify the source directory with ARUCO3_SMOKE_DIR." >&2
  exit 1
fi
readonly kSourceDir

work_dir="$(mktemp -d)"
trap 'rm -rf "${work_dir}"' EXIT

echo "[smoke-test] compile: sm_${kCudaArch}"
nvcc -std=c++17 -arch="sm_${kCudaArch}" \
  "${kSourceDir}/aruco3_smoke.cu" -o "${work_dir}/aruco3_smoke" \
  -I"${kOpenCvPrefix}/include/opencv4" -L"${kOpenCvPrefix}/lib" \
  -lopencv_core -lopencv_imgproc -lopencv_objdetect

echo "[smoke-test] run"
"${work_dir}/aruco3_smoke"

echo "[smoke-test] passed"
