# SPDX-License-Identifier: Apache-2.0
"""Increase only generated CTest limits for slow instrumented correctness runs."""
import hashlib
import json
from pathlib import Path
import sys

build = Path(sys.argv[1])
assert build.parts[0] == "build" and ".." not in build.parts
path = build / "test" / "aruco3cuda_cpu_tests[1]_tests.cmake"
before = path.read_bytes()
count = before.count(b" TIMEOUT 600 ")
assert count == 46, (path, count)
after = before.replace(b" TIMEOUT 600 ", b" TIMEOUT 1800 ")
path.write_bytes(after)
print(json.dumps({
    "generated_path": str(path),
    "tests": count,
    "before_timeout_seconds": 600,
    "after_timeout_seconds": 1800,
    "before_sha256": hashlib.sha256(before).hexdigest(),
    "after_sha256": hashlib.sha256(after).hexdigest(),
    "reason": "Historical ASan corpus alone took 556.41 s; assertions and source CTest properties remain unchanged."
}, indent=2))
