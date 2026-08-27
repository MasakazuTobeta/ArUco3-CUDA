#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# 目的:
#   評価計画が測定条件として要求する power mode と clock を、対象機の違いを
#   吸収して取得する。取得元が機種ごとに異なるため、その差をここへ集約する。
#
# 出力:
#   key=value 形式を 1 行 1 項目で標準出力へ書く。取得できない項目は行ごと
#   省略する。空文字列や 0 で埋めない。値が無いことと 0 であることは違う。
#
# 機種差:
#   DGX Spark は nvidia-smi から driver version と SM clock を取得できる。
#   Jetson には nvidia-smi が無く、nvpmodel の binary も container へ入らない。
#   このため nvpmodel の状態 file と sysfs の devfreq から取得する。
set -uo pipefail

emit() {
  # 値が空の項目は出力しない。呼出側が「未取得」を判別できるようにする。
  [ -n "${2:-}" ] && printf '%s=%s\n' "$1" "$2"
  return 0
}

# --- power mode ---
power_mode=""
if command -v nvpmodel >/dev/null 2>&1; then
  power_mode="$(nvpmodel -q 2>/dev/null | tr '\n' ' ' | sed 's/  */ /g;s/^ *//;s/ *$//')"
elif [ -r /var/lib/nvpmodel/status ]; then
  # status は "pmode:0000" 形式。前置 0 を落として mode 番号を取り出す。
  mode_id="$(sed -n 's/.*pmode:0*\([0-9][0-9]*\).*/\1/p' /var/lib/nvpmodel/status | head -1)"
  if [ -n "${mode_id}" ]; then
    mode_name=""
    if [ -r /etc/nvpmodel.conf ]; then
      # 定義行は "< POWER_MODEL ID=0 NAME=MAXN >" 形式。注釈行を除いて探す。
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
  # Jetson は sysfs の devfreq に GPU 周波数が出る。単位は Hz。
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

# 基板名。Jetson AGX Orin と Orin NX では性能が大きく異なるため、
# 測定結果には基板まで記録する。
#
# device tree は /sys/firmware 配下にあるが、Docker は既定でこの経路を
# mask する。このため compose が model file を直接 mount し、
# その経路を最優先で参照する。
platform_model=""
for path in /etc/aruco3cuda-platform-model /sys/firmware/devicetree/base/model /proc/device-tree/model; do
  if [ -r "${path}" ]; then
    platform_model="$(tr -d '\0' < "${path}")"
    break
  fi
done
emit platform_model "${platform_model}"

exit 0
