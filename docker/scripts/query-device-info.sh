#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Purpose:
#   Obtain the identity of the CUDA device - the device count, the name and the
#   Compute Capability - without nvidia-smi. L4T on Jetson AGX Orin has no
#   nvidia-smi, so record-environment.sh had nothing to read those from and
#   emitted empty strings in their place.
#
# Output:
#   key=value pairs, one item per line, on standard output, following the same
#   convention as query-platform-info.sh: an item that cannot be obtained is
#   omitted line and all, never padded with an empty string.
#
#     device_count=1
#     gpu_name=Orin
#     gpu_compute_capability=8.7
#
#   probe_error is the one deliberate exception to that convention. It is not a
#   measurement but the reason the items above are missing, and it is collapsed
#   to a single line because the key=value readers on the receiving side keep
#   only the first line of a value. An empty field with no recorded reason is
#   the very thing this script exists to remove, so the reason is emitted
#   whenever the identity is not.
#
#   The exit status is always 0. A machine with no visible device is a valid
#   environment to record, not a failure of the recording.
#
# Why this probe launches no kernel:
#   verify-environment.sh compiles a probe of its own that allocates memory and
#   runs a kernel. That is right for its job, which is to decide whether the
#   environment works. This script's job is to report what the device is, and
#   neither the name nor the Compute Capability depends on a kernel running.
#   Making the identity conditional on a launch would bring the empty fields
#   back on a machine that enumerates its device but cannot run code built for
#   it - the same symptom, with a different cause.
#
#   With no kernel there is also no cubin that has to match the device, so the
#   compile takes no -arch flag. That matters here: CUDA 11.4, which the
#   jetson-orin image ships, has neither -arch=native nor -arch=all, and
#   ARUCO3_CUDA_ARCH is the profile's architecture rather than the device's.
set -uo pipefail

readonly kCudaHome="${CUDA_HOME:-/usr/local/cuda}"

emit() {
  # Items with an empty value are not printed, so that the caller can tell
  # "not collected" apart.
  [ -n "${2:-}" ] && printf '%s=%s\n' "$1" "$2"
  return 0
}

# A reason has to fit on one line, and a compiler diagnostic does not.
one_line() {
  printf '%s' "${1:-}" | tr '\n\t' '  ' | sed 's/  */ /g;s/^ *//;s/ *$//'
}

nvcc="${kCudaHome}/bin/nvcc"
if [ ! -x "${nvcc}" ]; then
  nvcc="$(command -v nvcc 2>/dev/null)"
fi
if [ -z "${nvcc}" ] || [ ! -x "${nvcc}" ]; then
  emit probe_error "nvcc is not present, so the device probe cannot be built"
  exit 0
fi

work_dir="$(mktemp -d 2>/dev/null)"
if [ -z "${work_dir}" ]; then
  emit probe_error "a working directory for the device probe could not be created"
  exit 0
fi
trap 'rm -rf "${work_dir}"' EXIT

# The probe speaks the same key=value protocol as this script, so its output
# passes through unchanged and there is one format to keep in step instead of
# two.
cat > "${work_dir}/device-info.cu" <<'PROBE'
// SPDX-License-Identifier: Apache-2.0
// Reports the identity of CUDA device 0. No kernel is launched; the header of
// query-device-info.sh says why.
#include <cstdio>
#include <cuda_runtime_api.h>

int main() {
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) {
        std::printf("error=cudaGetDeviceCount failed: %s\n", cudaGetErrorString(status));
        return 1;
    }
    std::printf("device_count=%d\n", count);
    if (count <= 0) {
        std::printf("error=no CUDA device was found\n");
        return 1;
    }
    cudaDeviceProp properties{};
    status = cudaGetDeviceProperties(&properties, 0);
    if (status != cudaSuccess) {
        std::printf("error=cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(status));
        return 1;
    }
    std::printf("gpu_name=%s\n", properties.name);
    // Dotted, to match what nvidia-smi's compute_cap returns and what the
    // benchmark harness writes. Concatenating major and minor would be
    // ambiguous: Jetson AGX Thor is genuinely major 11, minor 0.
    std::printf("gpu_compute_capability=%d.%d\n", properties.major, properties.minor);
    return 0;
}
PROBE

if ! "${nvcc}" -std=c++17 -o "${work_dir}/device-info" "${work_dir}/device-info.cu" \
     2>"${work_dir}/compile.log"; then
  emit probe_error \
       "$(one_line "the device probe did not compile: $(head -3 "${work_dir}/compile.log")")"
  exit 0
fi

probe_output="$("${work_dir}/device-info" 2>&1)"
printf '%s\n' "${probe_output}" \
  | grep -E '^(device_count|gpu_name|gpu_compute_capability)=' || true

probe_error="$(printf '%s\n' "${probe_output}" | sed -n 's/^error=//p' | head -1)"
if [ -z "${probe_error}" ] \
   && ! printf '%s\n' "${probe_output}" | grep -q '^gpu_name='; then
  # The probe built and ran but said nothing recognizable. Passing that on as
  # silence would be the original defect again.
  probe_error="the device probe reported nothing: $(one_line "${probe_output}")"
fi
emit probe_error "$(one_line "${probe_error}")"

exit 0
