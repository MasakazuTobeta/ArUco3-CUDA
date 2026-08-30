#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Purpose:
#   Obtain the power mode and clock that the evaluation plan requires as
#   measurement conditions, absorbing the differences between target machines.
#   The sources differ per machine, so those differences are concentrated here.
#
# Output:
#   key=value pairs, one item per line, on standard output. Items that cannot be
#   obtained are omitted line and all. They are not padded with an empty string
#   or 0. Having no value is not the same as having the value 0.
#
# Machine differences:
#   On DGX Spark the driver version and the SM clock come from nvidia-smi.
#   Jetson has no nvidia-smi, and the nvpmodel binary is not present in the
#   container either, so the values come from the nvpmodel state files and from
#   devfreq in sysfs.
set -uo pipefail

emit() {
  # Items with an empty value are not printed, so that the caller can tell
  # "not collected" apart.
  [ -n "${2:-}" ] && printf '%s=%s\n' "$1" "$2"
  return 0
}

# --- power mode ---
power_mode=""
if command -v nvpmodel >/dev/null 2>&1; then
  power_mode="$(nvpmodel -q 2>/dev/null | tr '\n' ' ' | sed 's/  */ /g;s/^ *//;s/ *$//')"
elif [ -r /var/lib/nvpmodel/status ]; then
  # status has the form "pmode:0000". Strip the leading zeros to get the mode
  # number.
  mode_id="$(sed -n 's/.*pmode:0*\([0-9][0-9]*\).*/\1/p' /var/lib/nvpmodel/status | head -1)"
  if [ -n "${mode_id}" ]; then
    mode_name=""
    if [ -r /etc/nvpmodel.conf ]; then
      # A definition line has the form "< POWER_MODEL ID=0 NAME=MAXN >". Search
      # for it while skipping comment lines.
      mode_name="$(grep -E "^< *POWER_MODEL +ID=${mode_id} +NAME=" /etc/nvpmodel.conf 2>/dev/null \
                   | sed -n 's/.*NAME=\([^ >]*\).*/\1/p' | head -1)"
    fi
    if [ -n "${mode_name}" ]; then
      power_mode="${mode_name} (${mode_id})"
    else
      power_mode="mode ${mode_id}"
    fi
  fi
fi
emit power_mode "${power_mode}"

# --- GPU clock ---
gpu_max_clock_mhz=""
gpu_current_clock_mhz=""
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_max_clock_mhz="$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader 2>/dev/null \
                       | head -1 | sed 's/[^0-9]//g')"
  gpu_current_clock_mhz="$(nvidia-smi --query-gpu=clocks.current.sm --format=csv,noheader 2>/dev/null \
                           | head -1 | sed 's/[^0-9]//g')"
fi
if [ -z "${gpu_max_clock_mhz}" ]; then
  # On Jetson the GPU frequency appears in devfreq under sysfs, in Hz.
  for path in /sys/devices/gpu.0/devfreq/*/max_freq; do
    [ -r "${path}" ] || continue
    gpu_max_clock_mhz="$(( $(cat "${path}") / 1000000 ))"
    current_path="$(dirname "${path}")/cur_freq"
    [ -r "${current_path}" ] && gpu_current_clock_mhz="$(( $(cat "${current_path}") / 1000000 ))"
    break
  done
fi
emit gpu_max_clock_mhz "${gpu_max_clock_mhz}"
emit gpu_current_clock_mhz "${gpu_current_clock_mhz}"

# --- driver / platform ---
driver_version=""
if command -v nvidia-smi >/dev/null 2>&1; then
  driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
fi
emit driver_version "${driver_version}"

platform_release=""
if [ -r /etc/nv_tegra_release ]; then
  platform_release="$(head -1 /etc/nv_tegra_release)"
fi
emit platform_release "${platform_release}"

# Board name. Jetson AGX Orin and Orin NX differ greatly in performance, so the
# measurement results record the board as well.
#
# The device tree lives under /sys/firmware, but Docker masks that path by
# default. Compose therefore mounts the model file directly, and that path is
# consulted first.
platform_model=""
for path in /etc/aruco3cuda-platform-model /sys/firmware/devicetree/base/model /proc/device-tree/model; do
  if [ -r "${path}" ]; then
    platform_model="$(tr -d '\0' < "${path}")"
    break
  fi
done
emit platform_model "${platform_model}"

exit 0
