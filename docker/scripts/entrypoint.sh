#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Purpose:
#   Reflect the mounted CUDA Toolkit into PATH and LD_LIBRARY_PATH, then run the
#   command given as arguments.
#
# Notes:
#   The environment check does not run by default. Only when
#   ARUCO3_VERIFY_ON_START=1 is verify-environment.sh run, and startup is aborted
#   if it fails.
set -euo pipefail

readonly kCudaHome="${CUDA_HOME:-/usr/local/cuda}"

export PATH="${kCudaHome}/bin:${PATH}"
export LD_LIBRARY_PATH="${kCudaHome}/lib64:${ARUCO3_OPENCV_PREFIX:-/opt/opencv}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="${kCudaHome}"
export CUDACXX="${kCudaHome}/bin/nvcc"

if [ "${ARUCO3_VERIFY_ON_START:-0}" = "1" ]; then
  /opt/aruco3cuda/scripts/verify-environment.sh
fi

if [ "$#" -eq 0 ]; then
  exec /bin/bash
fi
exec "$@"
