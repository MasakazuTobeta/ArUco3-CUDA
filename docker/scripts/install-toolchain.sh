#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Purpose:
#   Install the toolchain needed to build and evaluate ArUco3-CUDA.
#   The same script is expected to work on both the Ubuntu 24.04 base for DGX
#   Spark and the L4T Ubuntu 20.04 base for Jetson Orin.
#
# Arguments:
#   When the environment variable INSTALL_DEV_TOOLS is 1, the formatting and
#   static analysis tools are installed as well. They are not installed by
#   default on the Jetson image because its formatter version differs.
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

# On base images whose CMake is below the required version, use the Kitware apt
# repository. The L4T r35 series is Ubuntu 20.04, whose distro CMake is 3.16, so
# it has to be replaced.
current_cmake="$(cmake --version | head -1 | awk '{print $3}')"
if [ "$(printf '%s\n%s\n' "${kRequiredCMakeMajorMinor}" "${current_cmake}" | sort -V | head -1)" != "${kRequiredCMakeMajorMinor}" ]; then
  echo "[install-toolchain] CMake ${current_cmake} is below the required ${kRequiredCMakeMajorMinor}; adding the Kitware repository"
  . /etc/os-release
  curl -fsSL https://apt.kitware.com/keys/kitware-archive-latest.asc \
    | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg
  echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ ${VERSION_CODENAME} main" \
    > /etc/apt/sources.list.d/kitware.list
  apt-get update
  apt-get install -y --no-install-recommends cmake
fi

if [ "${INSTALL_DEV_TOOLS:-0}" = "1" ]; then
  # Formatting and static analysis are used only in the development image.
  # They are kept out of the Jetson image because a different LLVM version in the
  # base image produces different formatting for the same source, which breaks
  # the difference check.
  apt-get install -y --no-install-recommends \
    clang-format \
    clang-tidy \
    cppcheck \
    gcovr \
    gdb \
    python3-matplotlib
fi

rm -rf /var/lib/apt/lists/*

# The CUDA Toolkit is not part of the image; it is bind-mounted from the host at
# run time. nvcc therefore does not exist while the image is being built. Its
# presence is checked by verify-environment.sh after the container starts.
echo "[install-toolchain] cmake=$(cmake --version | head -1)"
echo "[install-toolchain] gcc=$(gcc --version | head -1)"
