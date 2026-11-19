# SPDX-License-Identifier: Apache-2.0
"""Collect the two CPU routes on every checksummed scene in an annotated manifest."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--prefix", help="output filename prefix (default: family)")
    parser.add_argument("--exact", action="store_true", help="disable Hamming correction on both routes")
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    dictionary = manifest["dictionary"]
    if dictionary not in ("DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9",
                          "DICT_APRILTAG_36h10", "DICT_APRILTAG_36h11"):
        raise ValueError(f"unexpected dictionary {dictionary}")
    family = dictionary.rsplit("_", 1)[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    observations = {"cpu": [], "opencv": []}
    for scene in manifest["scenes"]:
        path = args.manifest.parent / scene["path"]
        pixels = path.read_bytes()
        if hashlib.sha256(pixels).hexdigest() != scene["sha256"]:
            raise ValueError(f"unexpected pixels in {path}")
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / path.name
            image.write_bytes(pixels)
            result = subprocess.run(
                [str(args.binary.resolve()), "--image", str(image), "--family", "tag" + family,
                 "--samples", "1", "--warmup", "0"] + (["--exact"] if args.exact else []),
                check=True, text=True, capture_output=True)
        rows = list(csv.reader(result.stdout.splitlines()))
        for route, output in observations.items():
            count, = [int(row[4]) for row in rows if row[0] == route]
            detections = [{"id": int(row[1]), "corners": list(map(float, row[2:]))}
                          for row in rows if row[0] == route + "_detection"]
            if len(detections) != count or any(len(d["corners"]) != 8 for d in detections):
                raise ValueError(f"incomplete {route} results for {path}")
            output.append(json.dumps({"scene": scene["name"], "manifest_sha256": manifest_digest,
                                      "detections": detections}, allow_nan=False))
    for route, output in observations.items():
        (args.output_dir / f"{args.prefix or family}-{route}.jsonl").write_text("\n".join(output) + "\n")


if __name__ == "__main__":
    main()
