#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# 目的:
#   container 環境が実際に使用可能であることを、環境検査より踏み込んで確認する。
#   nvcc が OpenCV を含む translation unit を compile でき、生成した実行 file が
#   GPU 上で動作し、OpenCV の ArUco3 検出戦略が期待どおり動くことを検証する。
#
# 位置付け:
#   これは container 環境の smoke test であり、project の unit test ではない。
#   project 側の test は WP-0.1 で `ctest` として整備する。
set -euo pipefail

readonly kCudaArch="${ARUCO3_CUDA_ARCH:?ARUCO3_CUDA_ARCH が未設定}"
readonly kOpenCvPrefix="${ARUCO3_OPENCV_PREFIX:-/opt/opencv}"
# image へ install された smoke source を既定とし、
# repository を直接使う場合は ARUCO3_SMOKE_DIR で上書きする。
kSourceDir="${ARUCO3_SMOKE_DIR:-/opt/aruco3cuda/smoke}"
if [ ! -f "${kSourceDir}/aruco3_smoke.cu" ]; then
  echo "[smoke-test] ${kSourceDir}/aruco3_smoke.cu が無い。" >&2
  echo "[smoke-test] ARUCO3_SMOKE_DIR で source directory を指定すること。" >&2
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

echo "[smoke-test] 合格"
