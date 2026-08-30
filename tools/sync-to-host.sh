#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Purpose:
#   Synchronize the repository to a target machine used for evaluation. The same
#   commit is built and measured on three machines, so anything missed during the
#   sync destroys the record of which machine measured what.
#
# Note on exclusions:
#   Build outputs are not transferred. The exclusion patterns are anchored relative
#   to the source root. Writing them as `build*` would also match
#   `docker/scripts/build-opencv.sh`, which would then not be transferred and the
#   image build would fail with a missing-file error. rsync does not delete excluded
#   files on the receiving side, so a stale file that arrived once keeps lingering
#   and is easy to miss.
#
#   .git is not transferred either. The target machine only builds and measures; it
#   never commits.
#
# Usage:
#   tools/sync-to-host.sh <user>@<host>
#   tools/sync-to-host.sh <user>@<host> ~/ArUco3-CUDA
#
# Return value:
#   0 = synchronized and verified, 1 = failed
set -uo pipefail

readonly kUsage="usage: $0 <user@host> [destination path]"

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "${kUsage}" >&2
  exit 1
fi

readonly kTarget="$1"
readonly kRemotePath="${2:-~/ArUco3-CUDA}"
readonly kSourceDir="$(cd "$(dirname "$0")/.." && pwd)"

echo "source: ${kSourceDir}"
echo "destination: ${kTarget}:${kRemotePath}"

# --delete removes extra files on the receiving side. Build outputs are excluded, so they survive.
if ! rsync -a --delete \
    --exclude '/build/' \
    --exclude '/build-*/' \
    --exclude '.git' \
    "${kSourceDir}/" "${kTarget}:${kRemotePath}/"; then
  echo "synchronization failed" >&2
  exit 1
fi

# Confirm that every tracked file arrived. A mistake in an exclusion pattern
# silently leaves files untransferred, so verify both the count and the checksums.
readonly kManifest="$(mktemp)"
trap 'rm -f "${kManifest}"' EXIT
(cd "${kSourceDir}" && git ls-files -z | xargs -0 sha256sum) | sort -k2 > "${kManifest}"

readonly kLocalCount="$(wc -l < "${kManifest}")"
readonly kRemoteSum="$(
  ssh "${kTarget}" "cd ${kRemotePath} && find . -type f -not -path './build/*' -not -path './.git/*' -print0 | xargs -0 sha256sum 2>/dev/null" \
    | sed 's# \./# #' | sort -k2
)"

missing=0
while read -r digest path; do
  if ! printf '%s\n' "${kRemoteSum}" | grep -qF "${digest}  ${path}"; then
    echo "  mismatched or missing: ${path}"
    missing=$((missing + 1))
  fi
done < "${kManifest}"

echo "checked ${kLocalCount} tracked files"
if [ "${missing}" -ne 0 ]; then
  echo "${missing} files do not match on the destination" >&2
  exit 1
fi
echo "synchronization complete; all tracked files match"
