#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Purpose:
#   Save the environment information the evaluation plan requires, in a
#   machine-readable form. Benchmark results only become reproducible when they
#   are paired with this JSON.
#
# Usage:
#   record-environment.sh [output JSON]
#   With no output path, the JSON is written to standard output.
set -euo pipefail

readonly kCudaHome="${CUDA_HOME:-/usr/local/cuda}"
readonly kOutput="${1:--}"
readonly kScriptDir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Sibling scripts are resolved relative to this one rather than through PATH.
# The image bakes the scripts in with COPY and puts that directory on PATH,
# while the repository is bind-mounted at /workspace, so PATH would find the
# copy from the image even when this file was run from the checkout - and a
# sibling added after the image was built would not be found at all.
sibling() {
  if [ -x "${kScriptDir}/$1" ]; then
    printf '%s' "${kScriptDir}/$1"
  else
    command -v "$1" 2>/dev/null
  fi
}

# Never fails, so that a missing helper leaves a field uncollected instead of
# aborting the recording. The script runs under errexit, where an unguarded
# assignment from a failing substitution would end it with no JSON at all.
run_sibling() {
  local script
  script="$(sibling "$1")"
  shift
  if [ -n "${script}" ]; then
    "${script}" "$@" 2>/dev/null
  fi
  return 0
}

query_gpu() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu="$1" --format=csv,noheader 2>/dev/null | head -1 | sed 's/^ *//;s/ *$//'
  fi
}

# nvidia-smi reports a field it does not support on standard output, so an
# unusable answer arrives as an English sentence where the value should be. The
# Compute Capability is the field that can be checked cheaply, and it decides
# whether the pair is trusted: the name comes from the same query, so if one is
# prose the other cannot be relied on either.
is_compute_capability() {
  printf '%s' "${1:-}" | grep -qE '^[0-9]+\.[0-9]+$'
}

# On aarch64, /proc/cpuinfo has no model name, so several sources are tried in
# order.
detect_cpu_model() {
  local model=""
  if command -v lscpu >/dev/null 2>&1; then
    model="$(lscpu 2>/dev/null | sed -n 's/^Model name: *//p' | head -1)"
  fi
  if [ -z "${model}" ]; then
    model="$(sed -n 's/^model name[ \t]*: *//p' /proc/cpuinfo | head -1)"
  fi
  if [ -z "${model}" ] && [ -r /sys/firmware/devicetree/base/model ]; then
    model="$(tr -d '\0' < /sys/firmware/devicetree/base/model)"
  fi
  if [ -z "${model}" ]; then
    model="unknown ($(uname -m))"
  fi
  printf '%s' "${model}"
}

# The Jetson power mode and clock feed directly into the measurement conditions,
# so they are always recorded. The sources for power mode, clock, driver, and L4T
# release differ per machine; absorbing those differences is concentrated in
# query-platform-info.sh.
platform_info="$(run_sibling query-platform-info.sh || true)"
platform_field() {
  printf '%s\n' "${platform_info}" | sed -n "s/^$1=//p" | head -1
}
platform_release="$(platform_field platform_release)"
power_mode="$(platform_field power_mode)"
gpu_max_clock_mhz="$(platform_field gpu_max_clock_mhz)"
gpu_current_clock_mhz="$(platform_field gpu_current_clock_mhz)"
driver_version_from_platform="$(platform_field driver_version)"
platform_model="$(platform_field platform_model)"

tmp_output="$(mktemp)"
trap 'rm -f "${tmp_output}"' EXIT

opencv_provenance="${ARUCO3_OPENCV_PREFIX:-/opt/opencv}/share/aruco3cuda/opencv-provenance.json"
opencv_json='null'
if [ -f "${opencv_provenance}" ]; then
  opencv_json="$(cat "${opencv_provenance}")"
fi

# How the CUDA Toolkit is supplied, and at which version, feeds directly into
# the comparability of the measurement results.
cuda_provenance="/opt/aruco3cuda/cuda-provenance.json"
cuda_json='null'
if [ -f "${cuda_provenance}" ]; then
  cuda_json="$(cat "${cuda_provenance}")"
fi

# The GPU identity. nvidia-smi is asked first, because where it exists it costs
# nothing. L4T on Jetson AGX Orin has none, and until this fallback existed the
# two fields were simply written out empty.
gpu_name="$(query_gpu name || true)"
gpu_compute_cap="$(query_gpu compute_cap || true)"
gpu_probe_error=""
if [ -z "${gpu_name}" ] || ! is_compute_capability "${gpu_compute_cap}"; then
  rejected=""
  if [ -n "${gpu_name}${gpu_compute_cap}" ]; then
    rejected="nvidia-smi did not answer with a Compute Capability; "
  fi
  # A probe compiled against the CUDA runtime, which needs no nvidia-smi. The
  # difference from verify-environment.sh's probe is set out in the header of
  # query-device-info.sh.
  device_info="$(run_sibling query-device-info.sh || true)"
  device_field() {
    printf '%s\n' "${device_info}" | sed -n "s/^$1=//p" | head -1
  }
  probe_name="$(device_field gpu_name)"
  probe_compute_cap="$(device_field gpu_compute_capability)"
  if [ -n "${probe_name}" ] && is_compute_capability "${probe_compute_cap}"; then
    gpu_name="${probe_name}"
    gpu_compute_cap="${probe_compute_cap}"
  else
    # Neither source answered. The fields stay empty, as they always have, but
    # the reason no longer goes unrecorded: an empty field with nothing to
    # explain it is what kept this gap invisible.
    gpu_name=""
    gpu_compute_cap=""
    gpu_probe_error="$(device_field probe_error)"
    if [ -z "${gpu_probe_error}" ]; then
      gpu_probe_error="query-device-info.sh returned no device information"
    fi
    gpu_probe_error="${rejected}${gpu_probe_error}"
  fi
fi

nvcc_version=""
if [ -x "${kCudaHome}/bin/nvcc" ]; then
  # The V token carries the patch version, as in 11.4.315, where the release
  # field stops at 11.4. The V expression comes first because it replaces the
  # pattern space, so the one behind it only fires when there was no V token.
  nvcc_version="$("${kCudaHome}/bin/nvcc" --version \
                   | sed -n 's/.*, V\([0-9][0-9.]*\).*/\1/p; s/.*release \([0-9][0-9.]*\).*/\1/p')"
fi

jq -n \
  --arg recorded_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg hostname "$(hostname)" \
  --arg kernel "$(uname -r)" \
  --arg arch "$(uname -m)" \
  --arg os "$(. /etc/os-release && echo "${PRETTY_NAME}")" \
  --arg container_profile "${ARUCO3_PROFILE:-unknown}" \
  --arg cuda_arch "${ARUCO3_CUDA_ARCH:-}" \
  --arg nvcc "${nvcc_version}" \
  --arg driver "${driver_version_from_platform}" \
  --arg gpu_name "${gpu_name}" \
  --arg gpu_compute_cap "${gpu_compute_cap}" \
  --arg gpu_probe_error "${gpu_probe_error}" \
  --arg gpu_clock_max_mhz "${gpu_max_clock_mhz}" \
  --arg gpu_clock_current_mhz "${gpu_current_clock_mhz}" \
  --arg gcc "$(gcc --version | head -1)" \
  --arg cmake "$(cmake --version | head -1 | awk '{print $3}')" \
  --arg cpu_model "$(detect_cpu_model)" \
  --argjson cpu_online "$(nproc)" \
  --argjson mem_total_kb "$(awk '/MemTotal/{print $2}' /proc/meminfo)" \
  --arg power_mode "${power_mode}" \
  --arg platform_release "${platform_release}" \
  --arg platform_model "${platform_model}" \
  --argjson opencv "${opencv_json}" \
  --argjson cuda "${cuda_json}" \
  '{
     recorded_at: $recorded_at,
     host: { hostname: $hostname, os: $os, kernel: $kernel, arch: $arch },
     container: { profile: $container_profile, target_cuda_arch: $cuda_arch },
     gpu: {
       name: $gpu_name,
       compute_capability: $gpu_compute_cap,
       probe_error: $gpu_probe_error,
       driver_version: $driver,
       max_sm_clock_mhz: $gpu_clock_max_mhz,
       current_sm_clock_mhz: $gpu_clock_current_mhz,
       power_mode: $power_mode,
       platform_release: $platform_release,
       platform_model: $platform_model
     },
     cpu: { model: $cpu_model, online_cores: $cpu_online },
     memory: { total_kb: $mem_total_kb },
     toolchain: { cuda_toolkit: $nvcc, cuda_provenance: $cuda, gcc: $gcc, cmake: $cmake },
     opencv: $opencv
   }' > "${tmp_output}"

if [ "${kOutput}" = "-" ]; then
  cat "${tmp_output}"
else
  mkdir -p "$(dirname "${kOutput}")"
  mv "${tmp_output}" "${kOutput}"
  echo "[record-environment] wrote: ${kOutput}" >&2
fi
