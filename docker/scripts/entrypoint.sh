#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# 目的:
#   mount された CUDA Toolkit を PATH と LD_LIBRARY_PATH へ反映し、
#   引数で渡された command を実行する。
#
# 備考:
#   環境検査は既定では実行しない。ARUCO3_VERIFY_ON_START=1 の場合のみ
#   verify-environment.sh を実行し、不合格なら起動を中止する。
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
