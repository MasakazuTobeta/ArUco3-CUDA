# SPDX-License-Identifier: Apache-2.0
"""Prepare the three AprilRobotics Swarmathon regression images and reference quads."""
import argparse
import hashlib
import json
import re
from pathlib import Path

from PIL import Image


# The image and label hashes pin the contents of test/data at the CPU dependency revision.
SOURCES = {
    "33369213973_9d9bb4cc96_c": (
        "6f2888334e65440576a6424928a230b82461b1a0a42d87d3b8835d6385901481",
        "0237b369c1873bd1e603d185573266030e59945cbe7f55ddf958337eaaefd002"),
    "34085369442_304b6bafd9_c": (
        "386c70bbd4d98dc0954d053240e73824fa8e07bdf8dc7f99ea50cac872b9b065",
        "521300049f229c66f98ac35b9354b1a638ed8dd5724d20a96f17e21625905eb6"),
    "34139872896_defdb2f8d9_c": (
        "435a619d642e2d50fb32224dc8e12992b08c2a596b02deee3fba497d3bff672a",
        "2361b3340286253eb67e8cfa467a6e9bb02e49b11d1ee8ee117aae9614bbf9bb"),
}


def upstream_marker(line):
    marker_id, quads = line.split(",", 1)
    points = [tuple(map(float, pair.split())) for pair in re.findall(r"\(([^()]*)\)", quads)]
    if len(points) != 4 or any(len(p) != 2 for p in points):
        raise ValueError(f"invalid AprilRobotics reference: {line}")
    # AprilRobotics records p[0..3] counterclockwise; canonical top left is p[1].
    return {"id": int(marker_id), "corners_px": [v for i in (1, 0, 3, 2) for v in points[i]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="pinned AprilRobotics checkout's test/data")
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        parser.error("output directory must be empty")
    scenes = []
    for name, hashes in SOURCES.items():
        jpg = args.data / (name + ".jpg")
        labels = args.data / (name + ".txt")
        for path, expected in zip((jpg, labels), hashes):
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                parser.error(f"input checksum mismatch: {path}")
        pgm = args.out / (name + ".pgm")
        with Image.open(jpg) as photo:
            photo.convert("L").save(pgm)
        scenes.append({"name": name, "path": pgm.name,
                       "sha256": hashlib.sha256(pgm.read_bytes()).hexdigest(),
                       "markers": [upstream_marker(s) for s in labels.read_text().splitlines() if s]})
    manifest = {"dictionary": "DICT_APRILTAG_36h11", "scenes": scenes}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
