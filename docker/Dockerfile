# SPDX-License-Identifier: Apache-2.0
#
# 目的:
#   ArUco3-CUDA の build、test、評価を行う開発 image。
#
# 方針:
#   CUDA Toolkit を image へ含めず、実行時に host から bind mount する。
#   host の CUDA Toolkit は 4 GB を超えるため、image へ含めると
#   複数 profile を保持する際の disk 使用量と pull 時間が現実的でなくなる。
#   NVIDIA driver library は NVIDIA Container Toolkit が注入する。
#
# 制約:
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

FROM toolchain AS opencv

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

COPY scripts/ /opt/aruco3cuda/scripts/
COPY smoke/ /opt/aruco3cuda/smoke/
RUN chmod +x /opt/aruco3cuda/scripts/*.sh

# script を PATH へ通し、docker compose run から名前だけで呼び出せるようにする。
ENV PATH=/opt/aruco3cuda/scripts:${PATH} \
    ARUCO3_PROFILE=${ARUCO3_PROFILE} \
    ARUCO3_CUDA_ARCH=${ARUCO3_CUDA_ARCH} \
    ARUCO3_OPENCV_PREFIX=/opt/opencv \
    CUDA_HOME=/usr/local/cuda \
    OpenCV_DIR=/opt/opencv/lib/cmake/opencv4 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /workspace

ENTRYPOINT ["/opt/aruco3cuda/scripts/entrypoint.sh"]
CMD ["/bin/bash"]
