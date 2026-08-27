#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# 目的:
#   ARUCO3_CUDA_MODE=pinned の場合に、この project が必要とする CUDA package
#   だけを image へ install する。
#
# 方針:
#   CUDA Toolkit 全体は 4.7 GB あるが、その 3.2 GB は cuBLAS、cuFFT、cuSOLVER、
#   cuSPARSE、cuRAND であり本 project では使用しない。必要なのは nvcc、cudart、
#   CCCL、compute-sanitizer、NVTX のみで、合計 300 MB 程度に収まる。
#
#   image へ固定することで、測定に使用した compiler が image と一体になり、
#   host の CUDA 更新によって benchmark 結果の比較可能性が失われることを防ぐ。
#
# 冪等性:
#   base image が既に CUDA を含む場合 (Jetson の l4t-jetpack 等) は install を
#   行わず、既存の Toolkit を対象として provenance のみ記録する。
set -euo pipefail

readonly kMode="${ARUCO3_CUDA_MODE:-pinned}"
readonly kRepoDistro="${ARUCO3_CUDA_REPO_DISTRO:-ubuntu2404}"
readonly kRepoArch="${ARUCO3_CUDA_REPO_ARCH:-sbsa}"
readonly kPackages="${ARUCO3_CUDA_PACKAGES:-cuda-nvcc-13-0 cuda-cccl-13-0 cuda-sanitizer-13-0 cuda-nvtx-13-0 cuda-cuobjdump-13-0 cuda-nvdisasm-13-0 cuda-profiler-api-13-0}"
readonly kProvenance="/opt/aruco3cuda/cuda-provenance.json"

mkdir -p "$(dirname "${kProvenance}")"

write_provenance() {
    local mode="$1"
    local packages_json="$2"
    local nvcc_version="$3"
    cat > "${kProvenance}" <<JSON
{
  "component": "cuda-toolkit",
  "mode": "${mode}",
  "nvcc_version": "${nvcc_version}",
  "packages": ${packages_json}
}
JSON
    echo "[install-cuda-toolkit] provenance: ${kProvenance}"
    cat "${kProvenance}"
}

if [ "${kMode}" = "mounted" ]; then
    # host から bind mount する。image には何も入れない。
    echo "[install-cuda-toolkit] mode=mounted のため install を行わない"
    write_provenance "mounted" "[]" "runtime-mounted"
    exit 0
fi

if [ "${kMode}" != "pinned" ]; then
    echo "[install-cuda-toolkit] 未知の ARUCO3_CUDA_MODE: ${kMode}" >&2
    echo "[install-cuda-toolkit] pinned または mounted を指定すること。" >&2
    exit 2
fi

detect_nvcc() {
    if [ -x /usr/local/cuda/bin/nvcc ]; then
        /usr/local/cuda/bin/nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p'
    elif command -v nvcc >/dev/null 2>&1; then
        nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p'
    fi
}

existing_nvcc="$(detect_nvcc)"
if [ -n "${existing_nvcc}" ]; then
    # base image が既に CUDA を含む場合。Jetson の l4t-jetpack がこれに当たる。
    echo "[install-cuda-toolkit] base image に CUDA ${existing_nvcc} が存在するため install を省略する"
    write_provenance "pinned-from-base-image" "[]" "${existing_nvcc}"
    exit 0
fi

export DEBIAN_FRONTEND=noninteractive
readonly kRepoBase="https://developer.download.nvidia.com/compute/cuda/repos/${kRepoDistro}/${kRepoArch}"

echo "[install-cuda-toolkit] NVIDIA の apt repository を追加する: ${kRepoBase}"
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl

# 署名鍵は cuda-keyring package で導入する。鍵を直接 URL から取得する方式より、
# 失効と更新の扱いが NVIDIA 側の運用に従うため安全である。
curl -fsSL -o /tmp/cuda-keyring.deb "${kRepoBase}/cuda-keyring_1.1-1_all.deb"
dpkg -i /tmp/cuda-keyring.deb
rm -f /tmp/cuda-keyring.deb

apt-get update
# shellcheck disable=SC2086
apt-get install -y --no-install-recommends ${kPackages}

# nvcc のみを install した場合、/usr/local/cuda の symlink が作られないことがある。
if [ ! -e /usr/local/cuda ]; then
    cuda_dir="$(find /usr/local -maxdepth 1 -name 'cuda-*' -type d | sort -V | tail -1)"
    if [ -z "${cuda_dir}" ]; then
        echo "[install-cuda-toolkit] /usr/local/cuda-* が見つからない" >&2
        exit 1
    fi
    ln -s "${cuda_dir}" /usr/local/cuda
    echo "[install-cuda-toolkit] symlink を作成: /usr/local/cuda -> ${cuda_dir}"
fi

# 実際に install された version を記録する。package 名は 13-0 のように
# minor version までを固定するが、patch version は repository の状態に依存する。
packages_json="["
first=1
for pkg in ${kPackages}; do
    version="$(dpkg-query -W -f='${Version}' "${pkg}" 2>/dev/null || echo unknown)"
    if [ "${first}" -eq 0 ]; then packages_json="${packages_json},"; fi
    packages_json="${packages_json}{\"name\":\"${pkg}\",\"version\":\"${version}\"}"
    first=0
done
packages_json="${packages_json}]"

rm -rf /var/lib/apt/lists/*
write_provenance "pinned" "${packages_json}" "$(detect_nvcc)"
