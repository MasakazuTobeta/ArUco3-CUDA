# SPDX-License-Identifier: Apache-2.0
"""Check that a detector sequence produces usable CSV and reports output failures."""
import csv
import io
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    with tempfile.TemporaryDirectory() as scratch:
        first = Path(scratch) / 'double",comma.pgm'
        second = Path(scratch) / 'next\nlayout.pgm'
        for path, width in ((first, 17), (second, 21)):
            path.write_bytes(f"P5\n{width} 15\n255\n".encode() + bytes([255]) * width * 15)
        command = [sys.argv[1], "DICT_APRILTAG_36h11", str(first), str(second), str(first)]
        run = subprocess.run(command, capture_output=True, text=True)
        if run.returncode and any(message in run.stderr for message in
                                  ("no CUDA-capable device", "CUDA driver version is insufficient")):
            return 77
        if run.returncode:
            raise RuntimeError(run.stderr)
        if list(csv.reader(io.StringIO(run.stdout), strict=True)) != [
            ["image", str(path)] for path in (first, second, first)
        ]:
            raise ValueError("image names did not round-trip through CSV")
        if Path("/dev/full").exists():
            with open("/dev/full", "wb") as sink:
                run = subprocess.run(command, stdout=sink, stderr=subprocess.PIPE, text=True)
            if run.returncode == 0 or "failed to write" not in run.stderr:
                raise ValueError("output failure was not reported")
    return 0


if __name__ == "__main__":
    sys.exit(main())
