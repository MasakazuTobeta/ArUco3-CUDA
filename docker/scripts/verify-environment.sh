#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Purpose:
#   Confirm right after startup that the container's CUDA environment is
#   genuinely usable, so that a defect surfaces as an explicit error rather than
#   as an inscrutable failure at build time.
#
# Checks:
#   1. The CUDA Toolkit is available and nvcc runs
#   2. ARUCO3_CUDA_ARCH is among the architectures nvcc targets
#   3. The host compiler and nvcc can compile together
#   4. The NVIDIA driver is injected and CUDA can enumerate devices
#   5. OpenCV is found
#
# Portability:
#   nvidia-smi is not used to check for a device. Jetson's L4T has no nvidia-smi,
#   so the check is a CUDA runtime API call, which works on both target machines.
#
# Return value:
#   0 = everything passed, 1 = one or more checks failed
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
    fail "${kCudaHome}/bin/nvcc does not exist. The host CUDA Toolkit is not mounted."
    echo "       Point the volumes in compose.mounted.yaml at the host CUDA directory."
    echo "       /usr/local/cuda is a symlink, so give the real directory."
  else
    fail "${kCudaHome}/bin/nvcc does not exist. Installing CUDA into the image failed."
    echo "       Rebuild the image, or check the ARUCO3_CUDA_PACKAGES setting."
  fi
else
  # Including the patch version; see the note in record-environment.sh.
  nvcc_version="$("${kCudaHome}/bin/nvcc" --version \
                   | sed -n 's/.*, V\([0-9][0-9.]*\).*/\1/p; s/.*release \([0-9][0-9.]*\).*/\1/p')"
  pass "nvcc ${nvcc_version} (${kCudaHome})"
  if [ "${kCudaMode}" = "mounted" ]; then
    warn "mounted mode depends on the host CUDA. Runs that produce measurements must use pinned mode."
  fi
fi
if [ -f /opt/aruco3cuda/cuda-provenance.json ]; then
  pass "CUDA provenance: $(jq -rc '.mode + " " + .nvcc_version' /opt/aruco3cuda/cuda-provenance.json)"
fi

echo "== 2. Target architecture =="
if [ -z "${kTargetArch}" ]; then
  fail "ARUCO3_CUDA_ARCH is not set."
elif [ -x "${kCudaHome}/bin/nvcc" ]; then
  if "${kCudaHome}/bin/nvcc" --list-gpu-arch 2>/dev/null | grep -qx "compute_${kTargetArch}"; then
    pass "nvcc supports compute_${kTargetArch}"
  else
    fail "nvcc does not support compute_${kTargetArch}. Check the CUDA Toolkit version."
  fi
fi

echo "== 3. host compiler =="
if ! command -v gcc >/dev/null 2>&1; then
  fail "gcc is not present."
else
  pass "$(gcc --version | head -1)"
fi

# The same program serves the device check and the compile check. Compiling and
# running are separate stages so that "it compiles" and "it runs" are reported
# separately.
cat > "${work_dir}/probe.cu" <<'PROBE'
// SPDX-License-Identifier: Apache-2.0
// Probe for the environment check. Prints the device count and the main
// properties.
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
    // Beyond reading the properties, confirm that a kernel actually runs.
    int* device_buffer = nullptr;
    if (cudaMalloc(&device_buffer, sizeof(int) * 32) != cudaSuccess) {
        std::printf("ERROR cudaMalloc\n");
        return 1;
    }
    touch_kernel<<<1, 32>>>(device_buffer);
    // A launch and an execution fail in different ways, and only the second one
    // reaches cudaDeviceSynchronize. A launch that never started - no code
    // compiled for this device's architecture, or a bad configuration - is
    // reported by cudaGetLastError alone, and synchronizing afterwards returns
    // success. Checking only the synchronize let this probe print OK for a
    // kernel that could not run at all, which is the one thing it exists to
    // catch.
    const cudaError_t launch = cudaGetLastError();
    status = cudaDeviceSynchronize();
    cudaFree(device_buffer);
    if (launch != cudaSuccess) {
        std::printf("ERROR kernel launch %s\n", cudaGetErrorString(launch));
        return 1;
    }
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
    pass "nvcc and the host compiler can compile together"
  else
    fail "The nvcc trial compile failed. Details:"
    sed 's/^/       /' "${work_dir}/compile.log" | head -8
  fi
fi

echo "== 4. NVIDIA driver and device =="
if ! ldconfig -p 2>/dev/null | grep -q "libcuda.so.1"; then
  fail "libcuda.so.1 is not present. Either the container runtime is not nvidia,"
  echo "       or NVIDIA_DRIVER_CAPABILITIES is not set."
else
  pass "libcuda.so.1 detected"
fi

if [ -x "${work_dir}/probe" ]; then
  probe_output="$("${work_dir}/probe" 2>&1)"
  if [ "${probe_output#OK }" != "${probe_output}" ]; then
    pass "CUDA device: ${probe_output#OK }"
    device_cc="$(printf '%s' "${probe_output}" | sed -n 's/.*cc=\([0-9]*\).*/\1/p')"
    if [ -n "${kTargetArch}" ] && [ -n "${device_cc}" ] && [ "${device_cc}" != "${kTargetArch}" ]; then
      # The profile and the actual machine disagree. That is normal for a cross
      # build, but for evaluation on real hardware it most likely means the
      # wrong profile was selected.
      warn "The machine's Compute Capability ${device_cc} differs from ARUCO3_CUDA_ARCH ${kTargetArch}."
      warn "If you intend to measure on this machine, check the profile selection."
    fi
  else
    fail "CUDA cannot enumerate any device: ${probe_output}"
    echo "       Check that the runtime is set to nvidia and that the device is visible."
  fi
else
  fail "The device probe could not be built, so the device cannot be checked."
fi

# nvidia-smi is supplementary information. It does not exist on Jetson's L4T, so
# its presence is not part of the pass/fail decision.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  pass "nvidia-smi: $(nvidia-smi -L | head -1)"
elif [ -r /etc/nv_tegra_release ]; then
  pass "L4T: $(head -1 /etc/nv_tegra_release)"
else
  warn "No nvidia-smi. That is normal on Jetson's L4T."
fi

echo "== 5. OpenCV =="
opencv_provenance="${ARUCO3_OPENCV_PREFIX:-/opt/opencv}/share/aruco3cuda/opencv-provenance.json"
if [ -f "${opencv_provenance}" ]; then
  pass "opencv $(jq -r '.version + " (" + .commit[0:12] + ") with_cuda=" + .with_cuda' "${opencv_provenance}")"
else
  fail "${opencv_provenance} does not exist. OpenCV is not installed into the image."
fi

echo
if [ "${failures}" -eq 0 ]; then
  echo "Environment check: everything passed"
else
  echo "Environment check: ${failures} check(s) failed"
fi
exit $(( failures > 0 ? 1 : 0 ))
