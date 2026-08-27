#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# 目的:
#   CPU 基準実装として使用する OpenCV を version 固定で build し、
#   取得元 commit と build option を provenance として記録する。
#
# 対象範囲:
#   ArUco3 検出に必要な module だけを build する。GUI backend と python binding は
#   対象外とする。CUDA 対応は既定で無効とし、必要な場合のみ --with-cuda で有効化する。
#
# 使用方法:
#   image build 時:   build-opencv.sh
#   container 実行時: build-opencv.sh --with-cuda --prefix /opt/opencv-cuda
#
# 備考:
#   CUDA Toolkit は image へ含めず実行時 mount とするため、image build 時点では
#   --with-cuda を使用できない。cv::cuda::GpuMat との相互変換が必要になる段階で、
#   起動中の container 内から再 build して named volume へ install する。
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
    *) echo "[build-opencv] 未知の引数: $1" >&2; exit 2 ;;
  esac
done

if [ "${with_cuda}" = "ON" ]; then
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "[build-opencv] --with-cuda が指定されたが nvcc が見つからない。" >&2
    echo "[build-opencv] CUDA Toolkit が mount された container 内で実行すること。" >&2
    exit 1
  fi
  if [ -z "${cuda_arch_bin}" ]; then
    echo "[build-opencv] --with-cuda には --cuda-arch または CUDA_ARCH_BIN が必要。" >&2
    exit 2
  fi
fi

work_dir="$(mktemp -d)"
trap 'rm -rf "${work_dir}"' EXIT

echo "[build-opencv] opencv ${kOpenCvVersion} (${kOpenCvCommit}) を取得する"
git init -q "${work_dir}/opencv"
git -C "${work_dir}/opencv" remote add origin https://github.com/opencv/opencv.git
git -C "${work_dir}/opencv" fetch -q --depth 1 origin "${kOpenCvCommit}"
git -C "${work_dir}/opencv" checkout -q FETCH_HEAD

actual_commit="$(git -C "${work_dir}/opencv" rev-parse HEAD)"
if [ "${actual_commit}" != "${kOpenCvCommit}" ]; then
  echo "[build-opencv] commit 不一致: 期待 ${kOpenCvCommit} 実際 ${actual_commit}" >&2
  exit 1
fi

# ArUco 検出は 4.x では objdetect module に含まれる。opencv_contrib は不要。
# calib3d は objdetect の依存として必要になる。
build_list="core,imgproc,imgcodecs,calib3d,objdetect"
if [ "${with_cuda}" = "ON" ]; then
  # cudev は cv::cuda::GpuMat のために必要。重い cuda* module は build しない。
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

# 由来を image 内へ残す。CONTRIBUTING.md の Code Provenance Rules に対応する。
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

echo "[build-opencv] install 先: ${prefix}"
cat "${prefix}/share/aruco3cuda/opencv-provenance.json"
