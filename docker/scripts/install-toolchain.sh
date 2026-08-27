#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# 目的:
#   ArUco3-CUDA の build と評価に必要な toolchain を install する。
#   DGX Spark 向け Ubuntu 24.04 base と Jetson Orin 向け L4T Ubuntu 20.04 base の
#   どちらでも同じ script が動作することを前提とする。
#
# 引数:
#   環境変数 INSTALL_DEV_TOOLS が 1 の場合、format と静的解析の tool も install する。
#   Jetson image では formatter の version が異なるため、既定では install しない。
set -euo pipefail

readonly kRequiredCMakeMajorMinor="3.24"
export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends \
  build-essential \
  ca-certificates \
  curl \
  git \
  gnupg \
  pkg-config \
  ninja-build \
  cmake \
  libgtest-dev \
  libjpeg-dev \
  libpng-dev \
  libtiff-dev \
  zlib1g-dev \
  python3 \
  python3-numpy \
  jq

# CMake が要求 version に満たない base image では Kitware の apt repository を使用する。
# L4T r35 系は Ubuntu 20.04 であり、distro の CMake は 3.16 のため置き換えが必要になる。
current_cmake="$(cmake --version | head -1 | awk '{print $3}')"
if [ "$(printf '%s\n%s\n' "${kRequiredCMakeMajorMinor}" "${current_cmake}" | sort -V | head -1)" != "${kRequiredCMakeMajorMinor}" ]; then
  echo "[install-toolchain] CMake ${current_cmake} は要求 ${kRequiredCMakeMajorMinor} 未満のため Kitware repository を追加する"
  . /etc/os-release
  curl -fsSL https://apt.kitware.com/keys/kitware-archive-latest.asc \
    | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg
  echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ ${VERSION_CODENAME} main" \
    > /etc/apt/sources.list.d/kitware.list
  apt-get update
  apt-get install -y --no-install-recommends cmake
fi

if [ "${INSTALL_DEV_TOOLS:-0}" = "1" ]; then
  # format と静的解析は開発 image でのみ使用する。
  # Jetson image へ入れないのは、base image の LLVM version が異なると
  # 同じ source に対する format 結果が変わり、差分確認が破綻するためである。
  apt-get install -y --no-install-recommends \
    clang-format \
    clang-tidy \
    cppcheck \
    gcovr \
    gdb \
    python3-matplotlib
fi

rm -rf /var/lib/apt/lists/*

# CUDA Toolkit は image へ含めず、実行時に host から bind mount する。
# このため image build 時点で nvcc は存在しない。存在確認は
# verify-environment.sh が container 起動後に行う。
echo "[install-toolchain] cmake=$(cmake --version | head -1)"
echo "[install-toolchain] gcc=$(gcc --version | head -1)"
