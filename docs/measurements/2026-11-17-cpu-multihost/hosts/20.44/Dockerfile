# SPDX-License-Identifier: Apache-2.0
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake ninja-build git python3 ca-certificates libjpeg-dev libpng-dev libtiff-dev libgtest-dev pkg-config && rm -rf /var/lib/apt/lists/*
COPY build-opencv.sh /usr/local/bin/build-opencv.sh
RUN BUILD_JOBS=16 bash /usr/local/bin/build-opencv.sh
ENV LD_LIBRARY_PATH=/opt/opencv/lib
WORKDIR /workspace
