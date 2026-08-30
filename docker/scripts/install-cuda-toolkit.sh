#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Purpose:
#   When ARUCO3_CUDA_MODE=pinned, install into the image only the CUDA packages
#   this project needs.
#
# Rationale:
#   The full CUDA Toolkit is 4.7 GB, but 3.2 GB of that is cuBLAS, cuFFT,
#   cuSOLVER, cuSPARSE, and cuRAND, none of which this project uses. All we need
#   is nvcc, cudart, CCCL, compute-sanitizer, and NVTX, which come to about
#   300 MB in total.
#
#   Pinning them into the image makes the compiler used for the measurements part
#   of the image, which prevents a CUDA update on the host from destroying the
#   comparability of the benchmark results.
#
# Idempotency:
#   When the base image already contains CUDA (for example Jetson's l4t-jetpack),
#   nothing is installed and only the provenance of the existing Toolkit is
#   recorded.
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
    # Bind-mounted from the host. Nothing goes into the image.
    echo "[install-cuda-toolkit] mode=mounted, so nothing is installed"
    write_provenance "mounted" "[]" "runtime-mounted"
    exit 0
fi

if [ "${kMode}" != "pinned" ]; then
    echo "[install-cuda-toolkit] unknown ARUCO3_CUDA_MODE: ${kMode}" >&2
    echo "[install-cuda-toolkit] specify either pinned or mounted." >&2
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
    # The base image already contains CUDA. Jetson's l4t-jetpack is such a case.
    echo "[install-cuda-toolkit] the base image already has CUDA ${existing_nvcc}; skipping the installation"
    write_provenance "pinned-from-base-image" "[]" "${existing_nvcc}"
    exit 0
fi

export DEBIAN_FRONTEND=noninteractive
readonly kRepoBase="https://developer.download.nvidia.com/compute/cuda/repos/${kRepoDistro}/${kRepoArch}"

echo "[install-cuda-toolkit] adding the NVIDIA apt repository: ${kRepoBase}"
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl

# Install the signing key through the cuda-keyring package. That is safer than
# fetching the key from a URL directly, because revocation and rotation then
# follow NVIDIA's own operations.
curl -fsSL -o /tmp/cuda-keyring.deb "${kRepoBase}/cuda-keyring_1.1-1_all.deb"
dpkg -i /tmp/cuda-keyring.deb
rm -f /tmp/cuda-keyring.deb

apt-get update
# shellcheck disable=SC2086
apt-get install -y --no-install-recommends ${kPackages}

# When only nvcc is installed, the /usr/local/cuda symlink is sometimes not
# created.
if [ ! -e /usr/local/cuda ]; then
    cuda_dir="$(find /usr/local -maxdepth 1 -name 'cuda-*' -type d | sort -V | tail -1)"
    if [ -z "${cuda_dir}" ]; then
        echo "[install-cuda-toolkit] no /usr/local/cuda-* was found" >&2
        exit 1
    fi
    ln -s "${cuda_dir}" /usr/local/cuda
    echo "[install-cuda-toolkit] created symlink: /usr/local/cuda -> ${cuda_dir}"
fi

# Record the versions that were actually installed. The package names pin down
# to the minor version, as in 13-0, but the patch version depends on the state of
# the repository.
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
