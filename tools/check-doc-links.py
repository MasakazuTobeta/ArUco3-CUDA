#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check that the relative links in the documents point at files that exist.

Reorganizing the documents inevitably breaks links. A broken link leads the reader
into a dead end and erodes trust in the documentation as a whole, so CI stops it
mechanically.

External URLs are out of scope. Checking their reachability depends on the state of
the network and would make CI flaky.

Usage:

    tools/check-doc-links.py [root]

Return value:
    0  no broken links
    1  at least one broken link
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
        print(f"{path}:{line}: link target does not exist: {target}")
    print(f"checked {checked} relative links, {len(broken)} broken")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
