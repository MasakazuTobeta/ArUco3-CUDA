#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Purpose:
#   Build the OpenCV used as the CPU baseline implementation at a fixed version,
#   and record the source commit and the build options as provenance.
#
# Scope:
#   Only the modules needed for ArUco3 detection are built. GUI backends and the
#   Python bindings are out of scope. CUDA support is off by default and is
#   enabled with --with-cuda only when it is needed.
#
# Usage:
#   at image build time: build-opencv.sh
#   inside a container:  build-opencv.sh --with-cuda --prefix /opt/opencv-cuda
#
# Notes:
#   The CUDA Toolkit is not part of the image but mounted at run time, so
#   --with-cuda cannot be used while the image is being built. Once conversion to
#   and from cv::cuda::GpuMat is needed, rebuild from inside a running container
#   and install into the named volume.
set -euo pipefail

readonly kOpenCvVersion="${OPENCV_VERSION:-4.14.0}"
readonly kOpenCvCommit="${OPENCV_COMMIT:-0654a42e19215ef25b1d367d822f3c630447e7c7}"
readonly kBuildJobs="${BUILD_JOBS:-$(nproc)}"

with_cuda="OFF"
prefix="/opt/opencv"
cuda_arch_bin="${CUDA_ARCH_BIN:-}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --with-cuda) with_cuda="ON"; shift ;;
    --prefix) prefix="$2"; shift 2 ;;
    --cuda-arch) cuda_arch_bin="$2"; shift 2 ;;
    *) echo "[build-opencv] unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ "${with_cuda}" = "ON" ]; then
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "[build-opencv] --with-cuda was given but nvcc was not found." >&2
    echo "[build-opencv] run this inside a container with the CUDA Toolkit mounted." >&2
    exit 1
  fi
  if [ -z "${cuda_arch_bin}" ]; then
    echo "[build-opencv] --with-cuda requires --cuda-arch or CUDA_ARCH_BIN." >&2
    exit 2
  fi
fi

work_dir="$(mktemp -d)"
trap 'rm -rf "${work_dir}"' EXIT

echo "[build-opencv] fetching opencv ${kOpenCvVersion} (${kOpenCvCommit})"
git init -q "${work_dir}/opencv"
git -C "${work_dir}/opencv" remote add origin https://github.com/opencv/opencv.git
git -C "${work_dir}/opencv" fetch -q --depth 1 origin "${kOpenCvCommit}"
git -C "${work_dir}/opencv" checkout -q FETCH_HEAD

actual_commit="$(git -C "${work_dir}/opencv" rev-parse HEAD)"
if [ "${actual_commit}" != "${kOpenCvCommit}" ]; then
  echo "[build-opencv] commit mismatch: expected ${kOpenCvCommit}, got ${actual_commit}" >&2
  exit 1
fi

# In 4.x, ArUco detection lives in the objdetect module. opencv_contrib is not
# needed. calib3d is required as a dependency of objdetect.
build_list="core,imgproc,imgcodecs,calib3d,objdetect"
if [ "${with_cuda}" = "ON" ]; then
  # cudev is required for cv::cuda::GpuMat. The heavy cuda* modules are not
  # built.
  build_list="${build_list},cudev"
fi

cmake_args=(
  -S "${work_dir}/opencv"
  -B "${work_dir}/build"
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_INSTALL_PREFIX="${prefix}"
  -DBUILD_LIST="${build_list}"
  -DBUILD_SHARED_LIBS=ON
  -DBUILD_TESTS=OFF
  -DBUILD_PERF_TESTS=OFF
  -DBUILD_EXAMPLES=OFF
  -DBUILD_DOCS=OFF
  -DBUILD_opencv_apps=OFF
  -DBUILD_opencv_python2=OFF
  -DBUILD_opencv_python3=OFF
  -DBUILD_JAVA=OFF
  -DWITH_GTK=OFF
  -DWITH_QT=OFF
  -DWITH_FFMPEG=OFF
  -DWITH_GSTREAMER=OFF
  -DWITH_V4L=OFF
  -DWITH_OPENEXR=OFF
  -DWITH_WEBP=OFF
  -DWITH_OPENJPEG=OFF
  -DWITH_PROTOBUF=OFF
  -DOPENCV_GENERATE_PKGCONFIG=ON
  -DWITH_CUDA="${with_cuda}"
)

if [ "${with_cuda}" = "ON" ]; then
  cmake_args+=(
    -DCUDA_ARCH_BIN="${cuda_arch_bin}"
    -DCUDA_ARCH_PTX=""
    -DWITH_CUDNN=OFF
    -DWITH_CUBLAS=OFF
    -DWITH_NVCUVID=OFF
  )
fi

cmake "${cmake_args[@]}"
cmake --build "${work_dir}/build" --parallel "${kBuildJobs}"
cmake --install "${work_dir}/build"

# Leave the provenance inside the image. This corresponds to the Code Provenance
# Rules in CONTRIBUTING.md.
mkdir -p "${prefix}/share/aruco3cuda"
cat > "${prefix}/share/aruco3cuda/opencv-provenance.json" <<JSON
{
  "component": "opencv",
  "version": "${kOpenCvVersion}",
  "commit": "${actual_commit}",
  "repository": "https://github.com/opencv/opencv.git",
  "license": "Apache-2.0",
  "with_cuda": "${with_cuda}",
  "cuda_arch_bin": "${cuda_arch_bin}",
  "build_list": "${build_list}",
  "modified": false
}
JSON

echo "[build-opencv] installed to: ${prefix}"
cat "${prefix}/share/aruco3cuda/opencv-provenance.json"
