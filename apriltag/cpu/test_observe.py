# SPDX-License-Identifier: Apache-2.0
"""Exercise CPU observation provenance and CLI write errors with a small image."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    compare = sys.argv[1]
    observe = Path(__file__).with_name("observe.py")
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        frame = directory / "blank.pgm"
        frame.write_bytes(b"P5\n17 13\n255\n" + bytes([200]) * (17 * 13))
        scene = {"name": "blank", "path": frame.name,
                 "sha256": hashlib.sha256(frame.read_bytes()).hexdigest()}
        manifest = directory / "manifest.json"
        manifest.write_text(json.dumps({"dictionary": "DICT_APRILTAG_36h11", "scenes": [scene]}))
        subprocess.run([sys.executable, observe, compare, manifest, directory], check=True)
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        for route in ("cpu", "opencv"):
            result = json.loads((directory / f"36h11-{route}.jsonl").read_text())
            if result != {"scene": "blank", "manifest_sha256": digest, "detections": []}:
                raise ValueError(f"incomplete {route} observation")
        if Path("/dev/full").exists():
            with open("/dev/full", "wb") as sink:
                result = subprocess.run([compare, "--image", frame, "--warmup", "0", "--samples", "1"],
                                        stdout=sink, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0 or "failed to write" not in result.stderr:
                raise ValueError("failed CPU output was reported as a successful measurement")


if __name__ == "__main__":
    main()
