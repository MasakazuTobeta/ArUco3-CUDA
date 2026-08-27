#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# 目的:
#   CUDA Toolkit を image へ含めず host から bind mount する構成では、
#   mount 漏れや version 不整合が build 時の不可解な失敗として現れる。
#   container 起動直後にこれを検出し、原因が分かる形で失敗させる。
#
# 検査項目:
#   1. CUDA Toolkit が mount され nvcc が実行できる
#   2. NVIDIA driver library が container へ注入されている
#   3. nvcc の対象 architecture に ARUCO3_CUDA_ARCH が含まれる
#   4. host compiler が nvcc の対応範囲にある
#   5. OpenCV が見つかる
#
# 戻り値:
#   0 = 全て合格、1 = 1 つ以上が不合格
set -uo pipefail

readonly kCudaHome="${CUDA_HOME:-/usr/local/cuda}"
failures=0

fail() { echo "  [NG] $*"; failures=$((failures + 1)); }
pass() { echo "  [OK] $*"; }

echo "== 1. CUDA Toolkit の mount =="
if [ ! -x "${kCudaHome}/bin/nvcc" ]; then
  fail "${kCudaHome}/bin/nvcc が無い。host の CUDA Toolkit が mount されていない。"
  echo "       compose の volumes に host の CUDA directory を指定すること。"
  echo "       /usr/local/cuda は symlink のため、実体の directory を指定する。"
else
  nvcc_version="$("${kCudaHome}/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
  pass "nvcc ${nvcc_version} (${kCudaHome})"
fi

echo "== 2. NVIDIA driver library の注入 =="
if ! ldconfig -p 2>/dev/null | grep -q "libcuda.so.1"; then
  fail "libcuda.so.1 が無い。container runtime が nvidia でないか、"
  echo "       NVIDIA_DRIVER_CAPABILITIES が設定されていない。"
else
  pass "libcuda.so.1 を検出"
fi
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  pass "$(nvidia-smi -L | head -1)"
else
  fail "nvidia-smi が device を列挙できない。"
fi

echo "== 3. 対象 architecture =="
target_arch="${ARUCO3_CUDA_ARCH:-}"
if [ -z "${target_arch}" ]; then
  fail "ARUCO3_CUDA_ARCH が未設定。"
elif [ -x "${kCudaHome}/bin/nvcc" ]; then
  if "${kCudaHome}/bin/nvcc" --list-gpu-arch 2>/dev/null | grep -qx "compute_${target_arch}"; then
    pass "nvcc は compute_${target_arch} に対応"
  else
    fail "nvcc が compute_${target_arch} に対応していない。CUDA Toolkit の version を確認すること。"
  fi
fi

echo "== 4. host compiler =="
if command -v gcc >/dev/null 2>&1; then
  pass "$(gcc --version | head -1)"
  if [ -x "${kCudaHome}/bin/nvcc" ]; then
    probe="$(mktemp --suffix=.cu)"
    printf '__global__ void k() {}\nint main() { k<<<1,1>>>(); return 0; }\n' > "${probe}"
    if "${kCudaHome}/bin/nvcc" -std=c++17 -arch="sm_${target_arch:-87}" -c "${probe}" -o /dev/null 2>/tmp/nvcc_probe.log; then
      pass "nvcc と host compiler の組み合わせで compile できる"
    else
      fail "nvcc の試験 compile に失敗した。詳細:"
      sed 's/^/       /' /tmp/nvcc_probe.log | head -5
    fi
    rm -f "${probe}"
  fi
else
  fail "gcc が無い。"
fi

echo "== 5. OpenCV =="
opencv_provenance="${ARUCO3_OPENCV_PREFIX:-/opt/opencv}/share/aruco3cuda/opencv-provenance.json"
if [ -f "${opencv_provenance}" ]; then
  pass "opencv $(jq -r '.version + " (" + .commit[0:12] + ") with_cuda=" + .with_cuda' "${opencv_provenance}")"
else
  fail "${opencv_provenance} が無い。OpenCV が image へ install されていない。"
fi

echo
if [ "${failures}" -eq 0 ]; then
  echo "環境検査: 全て合格"
else
  echo "環境検査: ${failures} 件が不合格"
fi
exit $(( failures > 0 ? 1 : 0 ))
