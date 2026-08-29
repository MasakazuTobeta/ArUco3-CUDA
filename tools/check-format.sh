#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# 目的:
#   clang-format による整形差分を確認する。CI と手元で同じ判定を使うため、
#   対象 file の選び方をこの 1 箇所へ集約する。CMake の format-check target も
#   この script を呼ぶ。
#
# 使用方法:
#   tools/check-format.sh          差分があれば失敗する
#   tools/check-format.sh --fix    整形を適用する
set -euo pipefail

readonly kRoot="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly kClangFormat="${CLANG_FORMAT:-clang-format}"

if ! command -v "${kClangFormat}" >/dev/null 2>&1; then
  echo "clang-format が見つからない。CLANG_FORMAT で path を指定できる。" >&2
  exit 1
fi

# 生成物は整形対象から外す。整形すると生成器の出力と byte 単位で一致しなくなり、
# dictgen --check による再生成の検証が成立しなくなる。
mapfile -t kSources < <(
  find "${kRoot}/include" "${kRoot}/src" "${kRoot}/reference" "${kRoot}/tools" "${kRoot}/test" \
       \( -name '*.hpp' -o -name '*.cpp' -o -name '*.cu' \) -type f 2>/dev/null \
    | grep -v '/generated/' | sort
)

if [ "${#kSources[@]}" -eq 0 ]; then
  echo "整形対象の file が見つからない" >&2
  exit 1
fi

if [ "${1:-}" = "--fix" ]; then
  "${kClangFormat}" -i "${kSources[@]}"
  echo "整形を適用した (${#kSources[@]} file)"
else
  "${kClangFormat}" --dry-run --Werror "${kSources[@]}"
  echo "整形差分なし (${#kSources[@]} file)"
fi
