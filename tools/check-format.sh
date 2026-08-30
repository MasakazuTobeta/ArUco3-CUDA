#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Purpose:
#   Check for formatting differences reported by clang-format. So that CI and local
#   runs make the same judgement, the rule for selecting the target files is kept in
#   this single place. The CMake format-check target also calls this script.
#
# Usage:
#   tools/check-format.sh          fails if there is any difference
#   tools/check-format.sh --fix    applies the formatting
set -euo pipefail

readonly kRoot="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly kClangFormat="${CLANG_FORMAT:-clang-format}"

if ! command -v "${kClangFormat}" >/dev/null 2>&1; then
  echo "clang-format not found. Set CLANG_FORMAT to point at its path." >&2
  exit 1
fi

# Generated files are excluded from formatting. Formatting them would break the
# byte-for-byte match with the generator output, which would invalidate the
# regeneration check performed by dictgen --check.
mapfile -t kSources < <(
  find "${kRoot}/include" "${kRoot}/src" "${kRoot}/reference" "${kRoot}/tools" "${kRoot}/test" \
       "${kRoot}/examples" \
       \( -name '*.hpp' -o -name '*.h' -o -name '*.cpp' -o -name '*.cu' \) -type f 2>/dev/null \
    | grep -v '/generated/' | sort
)

if [ "${#kSources[@]}" -eq 0 ]; then
  echo "no files to format were found" >&2
  exit 1
fi

if [ "${1:-}" = "--fix" ]; then
  "${kClangFormat}" -i "${kSources[@]}"
  echo "formatting applied (${#kSources[@]} files)"
else
  "${kClangFormat}" --dry-run --Werror "${kSources[@]}"
  echo "no formatting differences (${#kSources[@]} files)"
fi
