#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Download the pinned external compatibility inputs locally; never redistribute images."""
import argparse
import hashlib
import json
import pathlib
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/cpu-opencv-fixtures")
    args = parser.parse_args()
    manifest = json.loads(pathlib.Path(__file__).with_name("opencv-fixtures.json").read_text())
    destination = pathlib.Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for item in manifest["files"]:
        target = destination / item["name"]
        if target.exists():
            data = target.read_bytes()
        else:
            with urllib.request.urlopen(item["url"], timeout=30) as response:
                data = response.read()
        digest = hashlib.sha256(data).hexdigest()
        if digest != item["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch: {target}")
        blob = b"blob " + str(len(data)).encode("ascii") + b"\0" + data
        if hashlib.sha1(blob).hexdigest() != item["git_blob_sha1"]:
            raise RuntimeError(f"Git blob mismatch: {target}")
        if not target.exists():
            target.write_bytes(data)
        print(f"Verified {target}: {digest}")


if __name__ == "__main__":
    main()
