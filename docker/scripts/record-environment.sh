#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# 目的:
#   評価計画が要求する環境情報を機械可読形式で保存する。
#   benchmark 結果は、この JSON と対にして初めて再現可能になる。
#
# 使用方法:
#   record-environment.sh [出力先 JSON]
#   出力先を省略した場合は標準出力へ書き出す。
set -euo pipefail

readonly kCudaHome="${CUDA_HOME:-/usr/local/cuda}"
readonly kOutput="${1:--}"

query_gpu() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu="$1" --format=csv,noheader 2>/dev/null | head -1 | sed 's/^ *//;s/ *$//'
  fi
}

# aarch64 の /proc/cpuinfo には model name が無いため、複数の情報源を順に試す。
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

# Jetson の power mode と clock は測定条件に直結するため必ず記録する。
# power mode、clock、driver、L4T release は取得元が機種ごとに異なる。
# 差の吸収は query-platform-info.sh へ集約する。
platform_info="$(query-platform-info.sh 2>/dev/null || true)"
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

# CUDA Toolkit の供給方式と version は測定結果の比較可能性に直結する。
cuda_provenance="/opt/aruco3cuda/cuda-provenance.json"
cuda_json='null'
if [ -f "${cuda_provenance}" ]; then
  cuda_json="$(cat "${cuda_provenance}")"
fi

nvcc_version=""
if [ -x "${kCudaHome}/bin/nvcc" ]; then
  nvcc_version="$("${kCudaHome}/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
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
  --arg gpu_name "$(query_gpu name)" \
  --arg gpu_compute_cap "$(query_gpu compute_cap)" \
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
  echo "[record-environment] 出力: ${kOutput}" >&2
fi
