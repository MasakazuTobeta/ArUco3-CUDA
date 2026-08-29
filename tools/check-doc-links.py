#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""文書中の相対 link が実在する file を指すかを検査する。

文書を整理すると link は必ず壊れる。壊れた link は読者を行き止まりへ導き、
文書全体の信頼性を落とすため、CI で機械的に止める。

外部 URL は対象外とする。到達性の検査は network の状態に依存し、CI を
不安定にするためである。

使用方法:

    tools/check-doc-links.py [root]

戻り値:
    0  壊れた link が無い
    1  壊れた link がある
"""
import os
import re
import sys

kSkipDirs = {".git", "build", "node_modules"}
kLinkPattern = re.compile(r"\]\(([^)\s]+?)(?:#[^)]*)?\)")


def is_external(target: str) -> bool:
    return target.startswith(("http://", "https://", "mailto:", "#"))


def collect_markdown(root: str):
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in kSkipDirs]
        for name in files:
            if name.endswith(".md"):
                yield os.path.join(current, name)


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    broken = []
    checked = 0
    for path in sorted(collect_markdown(root)):
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        for match in kLinkPattern.finditer(text):
            target = match.group(1)
            if is_external(target):
                continue
            checked += 1
            resolved = os.path.normpath(os.path.join(os.path.dirname(path), target))
            if not os.path.exists(resolved):
                line = text.count("\n", 0, match.start()) + 1
                broken.append((path, line, target))

    for path, line, target in broken:
        print(f"{path}:{line}: 参照先が存在しない: {target}")
    print(f"相対 link {checked} 件を検査、壊れた link {len(broken)} 件")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
