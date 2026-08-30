# SPDX-License-Identifier: Apache-2.0
#
# Purpose:
#   Development image for building, testing, and evaluating ArUco3-CUDA.
#
# How the CUDA Toolkit is supplied:
#   ARUCO3_CUDA_MODE=pinned  (default) installs only the required CUDA packages
#                                      into the image.
#   ARUCO3_CUDA_MODE=mounted           bind-mounts it from the host at run time.
#
#   pinned is the default because it makes the compiler used for the measurements
#   part of the image, so a CUDA update on the host cannot destroy the
#   comparability of the benchmark results. The full Toolkit is 4.7 GB, but
#   3.2 GB of that is cuBLAS, cuFFT, cuSOLVER, cuSPARSE, and cuRAND, none of which
#   this project uses; what we need comes to about 300 MB.
#
#   mounted is kept for quick local experiments. In that mode the image is not
#   independent of the host environment, so it must not be used for runs that
#   produce measurement results.
#
#   In either mode the NVIDIA driver libraries are injected by the NVIDIA
#   Container Toolkit.
#
# Constraint (mounted mode):
#   Since host CUDA Toolkit binaries run inside the container, the glibc in
#   BASE_IMAGE must be at least the version the host CUDA Toolkit requires. The
#   base image is therefore matched to the host environment per profile.
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

# CPU baseline implementation. In OpenCV 4.x, ArUco detection lives in the
# objdetect module, so opencv_contrib is not needed. CUDA support is disabled
# because nvcc is not present while the image is being built.
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

# Put the scripts on PATH so that docker compose run can call them by name.
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
