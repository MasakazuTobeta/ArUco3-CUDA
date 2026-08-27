# SPDX-License-Identifier: Apache-2.0
#
# 目的:
#   ArUco3-CUDA の build、test、評価を行う開発 image。
#
# CUDA Toolkit の供給方式:
#   ARUCO3_CUDA_MODE=pinned  (既定) image へ必要な CUDA package だけを install する。
#   ARUCO3_CUDA_MODE=mounted        実行時に host から bind mount する。
#
#   pinned を既定とするのは、測定に使用した compiler が image と一体になり、
#   host の CUDA 更新によって benchmark 結果の比較可能性が失われないためである。
#   Toolkit 全体は 4.7 GB あるが、その 3.2 GB は本 project が使用しない
#   cuBLAS、cuFFT、cuSOLVER、cuSPARSE、cuRAND であり、必要な範囲は 300 MB 程度で済む。
#
#   mounted は手元での素早い試行のために残す。この mode では image が host 環境から
#   独立しないため、測定結果を伴う実行には使用しない。
#
#   いずれの mode でも NVIDIA driver library は NVIDIA Container Toolkit が注入する。
#
# 制約 (mounted mode):
#   host の CUDA Toolkit binary を container 内で実行するため、
#   BASE_IMAGE の glibc は host CUDA Toolkit が要求する version 以上である必要がある。
#   このため profile ごとに base image を host 環境へ合わせる。
#     DGX Spark  : CUDA 13.0 / Ubuntu 24.04 -> ubuntu:24.04
#     Jetson Orin: CUDA 11.4 / Ubuntu 20.04 -> ubuntu:20.04
ARG BASE_IMAGE=ubuntu:24.04

FROM ${BASE_IMAGE} AS toolchain

ARG INSTALL_DEV_TOOLS=1
ENV INSTALL_DEV_TOOLS=${INSTALL_DEV_TOOLS}

COPY scripts/install-toolchain.sh /tmp/install-toolchain.sh
RUN /tmp/install-toolchain.sh && rm -f /tmp/install-toolchain.sh

FROM toolchain AS cuda

ARG ARUCO3_CUDA_MODE=pinned
ARG ARUCO3_CUDA_REPO_DISTRO=ubuntu2404
ARG ARUCO3_CUDA_REPO_ARCH=sbsa
ARG ARUCO3_CUDA_PACKAGES="cuda-nvcc-13-0 cuda-cccl-13-0 cuda-sanitizer-13-0 cuda-nvtx-13-0 cuda-cuobjdump-13-0 cuda-nvdisasm-13-0 cuda-profiler-api-13-0"

COPY scripts/install-cuda-toolkit.sh /tmp/install-cuda-toolkit.sh
RUN ARUCO3_CUDA_MODE="${ARUCO3_CUDA_MODE}" \
    ARUCO3_CUDA_REPO_DISTRO="${ARUCO3_CUDA_REPO_DISTRO}" \
    ARUCO3_CUDA_REPO_ARCH="${ARUCO3_CUDA_REPO_ARCH}" \
    ARUCO3_CUDA_PACKAGES="${ARUCO3_CUDA_PACKAGES}" \
      /tmp/install-cuda-toolkit.sh \
    && rm -f /tmp/install-cuda-toolkit.sh

FROM cuda AS opencv

# CPU 基準実装。ArUco 検出は OpenCV 4.x では objdetect module に含まれるため
# opencv_contrib は不要。CUDA 対応は image build 時点で nvcc が無いため無効とする。
ARG OPENCV_VERSION=4.14.0
ARG OPENCV_COMMIT=0654a42e19215ef25b1d367d822f3c630447e7c7

COPY scripts/build-opencv.sh /tmp/build-opencv.sh
RUN OPENCV_VERSION="${OPENCV_VERSION}" OPENCV_COMMIT="${OPENCV_COMMIT}" \
      /tmp/build-opencv.sh --prefix /opt/opencv \
    && rm -f /tmp/build-opencv.sh \
    && ldconfig

FROM opencv AS dev

ARG ARUCO3_PROFILE=unknown
ARG ARUCO3_CUDA_ARCH=121
ARG ARUCO3_CUDA_MODE=pinned

COPY scripts/ /opt/aruco3cuda/scripts/
COPY smoke/ /opt/aruco3cuda/smoke/
RUN chmod +x /opt/aruco3cuda/scripts/*.sh

# script を PATH へ通し、docker compose run から名前だけで呼び出せるようにする。
ENV PATH=/opt/aruco3cuda/scripts:${PATH} \
    ARUCO3_PROFILE=${ARUCO3_PROFILE} \
    ARUCO3_CUDA_ARCH=${ARUCO3_CUDA_ARCH} \
    ARUCO3_CUDA_MODE=${ARUCO3_CUDA_MODE} \
    ARUCO3_OPENCV_PREFIX=/opt/opencv \
    CUDA_HOME=/usr/local/cuda \
    OpenCV_DIR=/opt/opencv/lib/cmake/opencv4 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /workspace

ENTRYPOINT ["/opt/aruco3cuda/scripts/entrypoint.sh"]
CMD ["/bin/bash"]
