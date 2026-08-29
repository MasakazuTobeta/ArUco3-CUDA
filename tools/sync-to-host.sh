#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# 目的:
#   評価対象の実機へ repository を同期する。3 機で同じ commit を build して
#   測定するため、同期の取りこぼしがあると「どの機で何を測ったか」が崩れる。
#
# 除外の注意:
#   build 出力は転送しない。ただし除外 pattern は転送元 root からの相対で
#   固定する。`build*` のように書くと `docker/scripts/build-opencv.sh` まで
#   一致して転送されず、image の build が「file が無い」で失敗する。
#   rsync は除外した file を受信側で削除しないため、一度届いた古い file が
#   残り続けて気付きにくい。
#
#   .git も転送しない。実機では build と測定のみを行い、commit はしない。
#
# 使用方法:
#   tools/sync-to-host.sh <user>@<host>
#   tools/sync-to-host.sh <user>@<host> ~/ArUco3-CUDA
#
# 戻り値:
#   0 = 同期して検証まで成功、1 = 失敗
set -uo pipefail

readonly kUsage="使用方法: $0 <user@host> [転送先 path]"

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "${kUsage}" >&2
  exit 1
fi

readonly kTarget="$1"
readonly kRemotePath="${2:-~/ArUco3-CUDA}"
readonly kSourceDir="$(cd "$(dirname "$0")/.." && pwd)"

echo "同期元: ${kSourceDir}"
echo "同期先: ${kTarget}:${kRemotePath}"

# --delete で受信側の余分な file を消す。build 出力は除外しているため残る。
if ! rsync -a --delete \
    --exclude '/build/' \
    --exclude '/build-*/' \
    --exclude '.git' \
    "${kSourceDir}/" "${kTarget}:${kRemotePath}/"; then
  echo "同期に失敗した" >&2
  exit 1
fi

# 追跡している file が全て届いたことを確認する。除外 pattern の書き間違いは
# 転送されない file を静かに生むため、件数と checksum の両方で確かめる。
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
    echo "  不一致または欠落: ${path}"
    missing=$((missing + 1))
  fi
done < "${kManifest}"

echo "追跡 file ${kLocalCount} 件を検査"
if [ "${missing}" -ne 0 ]; then
  echo "${missing} 件が同期先で一致しない" >&2
  exit 1
fi
echo "同期完了。全ての追跡 file が一致した"
