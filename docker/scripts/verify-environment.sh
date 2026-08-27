#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# 目的:
#   container の CUDA 環境が実際に使用可能であることを起動直後に確認し、
#   不備を build 時の不可解な失敗ではなく明示的なエラーとして表面化させる。
#
# 検査項目:
#   1. CUDA Toolkit が使用でき nvcc が実行できる
#   2. nvcc の対象 architecture に ARUCO3_CUDA_ARCH が含まれる
#   3. host compiler と nvcc の組み合わせで compile できる
#   4. NVIDIA driver が注入され、CUDA から device を列挙できる
#   5. OpenCV が見つかる
#
# 移植性:
#   device の確認へ nvidia-smi を使用しない。Jetson の L4T には nvidia-smi が
#   存在しないため、両対象機で成立する CUDA runtime API の呼び出しで確認する。
#
# 戻り値:
#   0 = 全て合格、1 = 1 つ以上が不合格
set -uo pipefail

readonly kCudaHome="${CUDA_HOME:-/usr/local/cuda}"
readonly kCudaMode="${ARUCO3_CUDA_MODE:-pinned}"
readonly kTargetArch="${ARUCO3_CUDA_ARCH:-}"

failures=0
work_dir="$(mktemp -d)"
trap 'rm -rf "${work_dir}"' EXIT

fail() { echo "  [NG] $*"; failures=$((failures + 1)); }
pass() { echo "  [OK] $*"; }
warn() { echo "  [--] $*"; }

echo "== 1. CUDA Toolkit (mode=${kCudaMode}) =="
if [ ! -x "${kCudaHome}/bin/nvcc" ]; then
  if [ "${kCudaMode}" = "mounted" ]; then
    fail "${kCudaHome}/bin/nvcc が無い。host の CUDA Toolkit が mount されていない。"
    echo "       compose.mounted.yaml の volumes に host の CUDA directory を指定すること。"
    echo "       /usr/local/cuda は symlink のため、実体の directory を指定する。"
  else
    fail "${kCudaHome}/bin/nvcc が無い。image への CUDA install に失敗している。"
    echo "       image を再 build するか、ARUCO3_CUDA_PACKAGES の指定を確認すること。"
  fi
else
  nvcc_version="$("${kCudaHome}/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
  pass "nvcc ${nvcc_version} (${kCudaHome})"
  if [ "${kCudaMode}" = "mounted" ]; then
    warn "mounted mode は host の CUDA に依存する。測定を伴う実行は pinned mode で行うこと。"
  fi
fi
if [ -f /opt/aruco3cuda/cuda-provenance.json ]; then
  pass "CUDA provenance: $(jq -rc '.mode + " " + .nvcc_version' /opt/aruco3cuda/cuda-provenance.json)"
fi

echo "== 2. 対象 architecture =="
if [ -z "${kTargetArch}" ]; then
  fail "ARUCO3_CUDA_ARCH が未設定。"
elif [ -x "${kCudaHome}/bin/nvcc" ]; then
  if "${kCudaHome}/bin/nvcc" --list-gpu-arch 2>/dev/null | grep -qx "compute_${kTargetArch}"; then
    pass "nvcc は compute_${kTargetArch} に対応"
  else
    fail "nvcc が compute_${kTargetArch} に対応していない。CUDA Toolkit の version を確認すること。"
  fi
fi

echo "== 3. host compiler =="
if ! command -v gcc >/dev/null 2>&1; then
  fail "gcc が無い。"
else
  pass "$(gcc --version | head -1)"
fi

# device 検査と compile 検査で同じ program を使う。compile が通ることと
# 実行できることを分けて報告するため、compile と run を別段階にする。
cat > "${work_dir}/probe.cu" <<'PROBE'
// SPDX-License-Identifier: Apache-2.0
// 環境検査用の probe。device 数と主要な性質を表示する。
#include <cstdio>
#include <cuda_runtime_api.h>

__global__ void touch_kernel(int* out) { out[threadIdx.x] = static_cast<int>(threadIdx.x); }

int main() {
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) {
        std::printf("ERROR cudaGetDeviceCount %s\n", cudaGetErrorString(status));
        return 1;
    }
    if (count <= 0) {
        std::printf("ERROR no-device\n");
        return 1;
    }
    cudaDeviceProp prop{};
    status = cudaGetDeviceProperties(&prop, 0);
    if (status != cudaSuccess) {
        std::printf("ERROR cudaGetDeviceProperties %s\n", cudaGetErrorString(status));
        return 1;
    }
    // 性質の取得だけでなく、実際に kernel が動くところまで確認する。
    int* device_buffer = nullptr;
    if (cudaMalloc(&device_buffer, sizeof(int) * 32) != cudaSuccess) {
        std::printf("ERROR cudaMalloc\n");
        return 1;
    }
    touch_kernel<<<1, 32>>>(device_buffer);
    status = cudaDeviceSynchronize();
    cudaFree(device_buffer);
    if (status != cudaSuccess) {
        std::printf("ERROR kernel %s\n", cudaGetErrorString(status));
        return 1;
    }
    std::printf("OK count=%d name=%s cc=%d%d integrated=%d\n", count, prop.name, prop.major,
                prop.minor, prop.integrated);
    return 0;
}
PROBE

if [ -x "${kCudaHome}/bin/nvcc" ] && command -v gcc >/dev/null 2>&1; then
  if "${kCudaHome}/bin/nvcc" -std=c++17 -arch="sm_${kTargetArch:-87}" \
       "${work_dir}/probe.cu" -o "${work_dir}/probe" 2>"${work_dir}/compile.log"; then
    pass "nvcc と host compiler の組み合わせで compile できる"
  else
    fail "nvcc の試験 compile に失敗した。詳細:"
    sed 's/^/       /' "${work_dir}/compile.log" | head -8
  fi
fi

echo "== 4. NVIDIA driver と device =="
if ! ldconfig -p 2>/dev/null | grep -q "libcuda.so.1"; then
  fail "libcuda.so.1 が無い。container runtime が nvidia でないか、"
  echo "       NVIDIA_DRIVER_CAPABILITIES が設定されていない。"
else
  pass "libcuda.so.1 を検出"
fi

if [ -x "${work_dir}/probe" ]; then
  probe_output="$("${work_dir}/probe" 2>&1)"
  if [ "${probe_output#OK }" != "${probe_output}" ]; then
    pass "CUDA device: ${probe_output#OK }"
    device_cc="$(printf '%s' "${probe_output}" | sed -n 's/.*cc=\([0-9]*\).*/\1/p')"
    if [ -n "${kTargetArch}" ] && [ -n "${device_cc}" ] && [ "${device_cc}" != "${kTargetArch}" ]; then
      # profile と実機が食い違っている。cross build では正常だが、
      # 実機評価では profile の選択を誤っている可能性が高い。
      warn "実機の Compute Capability ${device_cc} が ARUCO3_CUDA_ARCH ${kTargetArch} と異なる。"
      warn "この機で測定するなら profile の選択を確認すること。"
    fi
  else
    fail "CUDA から device を列挙できない: ${probe_output}"
    echo "       runtime に nvidia を指定しているか、device が見えているかを確認すること。"
  fi
else
  fail "device 確認用の probe を build できなかったため device を確認できない。"
fi

# nvidia-smi は補助情報。Jetson の L4T には存在しないため、有無を合否に含めない。
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  pass "nvidia-smi: $(nvidia-smi -L | head -1)"
elif [ -r /etc/nv_tegra_release ]; then
  pass "L4T: $(head -1 /etc/nv_tegra_release)"
else
  warn "nvidia-smi なし。Jetson の L4T では正常。"
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
