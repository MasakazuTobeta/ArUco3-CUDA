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
#
# The remote side hashes exactly the tracked list, sent over stdin, rather than
# walking the whole remote tree with find and matching the manifest against
# whatever comes back.
#
# The walking form reported "do not match" for files whose contents were in fact
# identical. That was intermittent and was never reproduced on demand, so no
# cause is recorded here as fact. What is certain is that it could not tell a
# short or failed listing from a real mismatch: every file the listing lacked,
# for any reason, was reported as not matching. Asking the remote for exactly
# the files being verified removes that ambiguity, and it no longer walks a tree
# that also holds build outputs, which makes it faster besides.
#
# The comparison is also whole-line now. The old one used grep for the digest
# and path as a substring, which passes when one path is a suffix of another.
#
# The list is NUL separated end to end, so a path containing a space or a quote
# cannot be split or reinterpreted by the remote shell.
readonly kManifest="$(mktemp)"
readonly kRemoteList="$(mktemp)"
readonly kRemotePaths="$(mktemp)"
trap 'rm -f "${kManifest}" "${kRemoteList}" "${kRemotePaths}"' EXIT

(cd "${kSourceDir}" && git ls-files -z | xargs -0 sha256sum) | sort > "${kManifest}"
readonly kLocalCount="$(wc -l < "${kManifest}")"

if [ "${kLocalCount}" -eq 0 ]; then
  echo "no tracked files were found; is ${kSourceDir} a git repository?" >&2
  exit 1
fi

# sha256sum reports a file it cannot open on stderr and leaves it out of the
# output, which is the signal that it did not arrive. The comparison below names
# it, so the noise is suppressed here.
(cd "${kSourceDir}" && git ls-files -z) \
  | ssh "${kTarget}" "cd ${kRemotePath} && xargs -0 sha256sum" 2>/dev/null \
  | sort > "${kRemoteList}"

if [ ! -s "${kRemoteList}" ]; then
  echo "the destination returned no checksums; the connection or the path is wrong" >&2
  exit 1
fi

# Paths only, for telling a file that arrived corrupted from one that never
# arrived. sha256sum separates the digest from the path with two spaces, so the
# path starts at the third field.
cut -d' ' -f3- < "${kRemoteList}" > "${kRemotePaths}"

missing=0
while IFS= read -r entry; do
  path="${entry#*  }"
  if grep -qxF "${path}" "${kRemotePaths}"; then
    echo "  content differs: ${path}"
  else
    echo "  did not arrive: ${path}"
  fi
  missing=$((missing + 1))
done < <(comm -23 "${kManifest}" "${kRemoteList}")

echo "checked ${kLocalCount} tracked files"
if [ "${missing}" -ne 0 ]; then
  echo "${missing} files do not match on the destination" >&2
  exit 1
fi
echo "synchronization complete; all tracked files match"
