#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Purpose:
#   When ARUCO3_CUDA_MODE=pinned, install into the image only the CUDA packages
#   this project needs.
#
# Rationale:
#   The full CUDA Toolkit is 4.7 GB for CUDA 13 and 3.7 GB for the CUDA 11.4 that
#   L4T ships, and most of that is cuBLAS, cuFFT, cuSOLVER, cuSPARSE, and cuRAND,
#   none of which this project uses. All we need is nvcc, cudart, CCCL,
#   compute-sanitizer, NVTX, and the disassembler, which comes to about 300 MB for
#   CUDA 13 on DGX Spark and 176 MB for CUDA 11.4 on Jetson Orin.
#
#   Pinning them into the image makes the compiler used for the measurements part
#   of the image, which prevents a CUDA update on the host from destroying the
#   comparability of the benchmark results.
#
# Where the packages come from:
#   ARUCO3_CUDA_REPO_FLAVOR=cuda    (default) developer.download.nvidia.com, the
#                                   ordinary CUDA repository, keyed by the
#                                   cuda-keyring package.
#   ARUCO3_CUDA_REPO_FLAVOR=jetson            repo.download.nvidia.com/jetson,
#                                   which is where the CUDA packages for L4T
#                                   live. Same package names, different host and
#                                   a different way of installing the key.
#
#   Jetson needs its own flavour because the CUDA build for Tegra is published
#   only there. The ordinary repository has no aarch64 build that runs on Tegra,
#   and its sbsa builds target server-class ARM.
#
# Idempotency:
#   When the base image already contains CUDA (for example Jetson's l4t-jetpack),
#   nothing is installed and only the provenance of the existing Toolkit is
#   recorded. That path is no longer taken by any profile in compose.yaml; it is
#   kept because BASE_IMAGE is overridable.
set -euo pipefail

readonly kMode="${ARUCO3_CUDA_MODE:-pinned}"
readonly kRepoDistro="${ARUCO3_CUDA_REPO_DISTRO:-ubuntu2404}"
readonly kRepoArch="${ARUCO3_CUDA_REPO_ARCH:-sbsa}"
readonly kPackages="${ARUCO3_CUDA_PACKAGES:-cuda-nvcc-13-0 cuda-cccl-13-0 cuda-sanitizer-13-0 cuda-nvtx-13-0 cuda-cuobjdump-13-0 cuda-nvdisasm-13-0 cuda-profiler-api-13-0}"
readonly kRepoFlavor="${ARUCO3_CUDA_REPO_FLAVOR:-cuda}"
readonly kRepoSuite="${ARUCO3_CUDA_REPO_SUITE:-r35.4}"
readonly kJetsonKeyring="/usr/share/keyrings/nvidia-jetson-ota.gpg"
readonly kProvenance="/opt/aruco3cuda/cuda-provenance.json"

mkdir -p "$(dirname "${kProvenance}")"

write_provenance() {
    local mode="$1"
    local packages_json="$2"
    local nvcc_version="$3"
    local repository="$4"
    cat > "${kProvenance}" <<JSON
{
  "component": "cuda-toolkit",
  "mode": "${mode}",
  "repository": "${repository}",
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
    write_provenance "mounted" "[]" "runtime-mounted" "none"
    exit 0
fi

if [ "${kMode}" != "pinned" ]; then
    echo "[install-cuda-toolkit] unknown ARUCO3_CUDA_MODE: ${kMode}" >&2
    echo "[install-cuda-toolkit] specify either pinned or mounted." >&2
    exit 2
fi

# The V token carries the patch version, as in 11.4.315, where the release field
# stops at 11.4. The V expression comes first because it replaces the pattern
# space, so the release expression behind it only fires when there was no V token
# to find.
detect_nvcc() {
    if [ -x /usr/local/cuda/bin/nvcc ]; then
        /usr/local/cuda/bin/nvcc --version | sed -n 's/.*, V\([0-9][0-9.]*\).*/\1/p; s/.*release \([0-9][0-9.]*\).*/\1/p'
    elif command -v nvcc >/dev/null 2>&1; then
        nvcc --version | sed -n 's/.*, V\([0-9][0-9.]*\).*/\1/p; s/.*release \([0-9][0-9.]*\).*/\1/p'
    fi
}

existing_nvcc="$(detect_nvcc)"
if [ -n "${existing_nvcc}" ]; then
    # The base image already contains CUDA. Jetson's l4t-jetpack is such a case.
    echo "[install-cuda-toolkit] the base image already has CUDA ${existing_nvcc}; skipping the installation"
    write_provenance "pinned-from-base-image" "[]" "${existing_nvcc}" "base-image"
    exit 0
fi

export DEBIAN_FRONTEND=noninteractive

# The ordinary CUDA repository. The signing key arrives through the cuda-keyring
# package, which is safer than fetching a key from a URL because revocation and
# rotation then follow NVIDIA's own operations.
add_cuda_repository() {
    local repo_base="https://developer.download.nvidia.com/compute/cuda/repos/${kRepoDistro}/${kRepoArch}"
    echo "[install-cuda-toolkit] adding the NVIDIA CUDA apt repository: ${repo_base}"
    apt-get install -y --no-install-recommends ca-certificates curl
    curl -fsSL -o /tmp/cuda-keyring.deb "${repo_base}/cuda-keyring_1.1-1_all.deb"
    dpkg -i /tmp/cuda-keyring.deb
    rm -f /tmp/cuda-keyring.deb
    kRepositoryDescription="${repo_base}"
}

# The L4T repository, which is where the CUDA build for Tegra is published.
#
# There is no cuda-keyring package here, so the armored key is fetched and
# dearmored into its own keyring and the source is bound to it with signed-by.
# apt-key would be the alternative and it is deprecated for a reason: a key added
# that way is trusted for every repository the image has, not just this one.
#
# Only the common component is added. The CUDA packages live there; the t234
# component holds the board support package and the L4T driver, and the driver is
# injected at run time by the NVIDIA Container Toolkit. Adding t234 would put
# packages within apt's reach that must not end up in this image.
add_jetson_repository() {
    local repo_base="https://repo.download.nvidia.com/jetson"
    echo "[install-cuda-toolkit] adding the L4T apt repository: ${repo_base}/common ${kRepoSuite}"
    apt-get install -y --no-install-recommends ca-certificates curl gnupg
    curl -fsSL "${repo_base}/jetson-ota-public.asc" | gpg --dearmor -o "${kJetsonKeyring}"
    chmod 0644 "${kJetsonKeyring}"
    echo "deb [signed-by=${kJetsonKeyring}] ${repo_base}/common ${kRepoSuite} main" \
        > /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
    kRepositoryDescription="${repo_base}/common ${kRepoSuite} main"
}

apt-get update
kRepositoryDescription=""
case "${kRepoFlavor}" in
    cuda) add_cuda_repository ;;
    jetson) add_jetson_repository ;;
    *)
        echo "[install-cuda-toolkit] unknown ARUCO3_CUDA_REPO_FLAVOR: ${kRepoFlavor}" >&2
        echo "[install-cuda-toolkit] specify either cuda or jetson." >&2
        exit 2
        ;;
esac

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
write_provenance "pinned" "${packages_json}" "$(detect_nvcc)" "${kRepositoryDescription}"
