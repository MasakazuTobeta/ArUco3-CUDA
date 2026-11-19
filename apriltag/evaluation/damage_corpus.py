# SPDX-License-Identifier: Apache-2.0
"""Render changing 36h11 frames from AprilRobotics' independently published ID 17."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


SOURCE_SHA256 = "5050b33258299be15bf4670bb9f930ae64fffca83712b89eb8e70fd6b3e8eca3"
FLIPS = [(2, 2), (5, 7), (4, 3), (7, 5)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != SOURCE_SHA256:
        parser.error("expected the pinned tag36_11_00017.png")
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        parser.error("output directory must be empty")
    # AprilRobotics' PNG is 180 degrees from the OpenCV canonical bit order.
    original = Image.open(args.source).convert("L").transpose(Image.Transpose.ROTATE_180)
    scenes = []
    for cell in [4, 12, 23]:
        for turn in range(4):
            for damage in range(5):
                marker = original.copy()
                for point in FLIPS[:damage]:
                    marker.putpixel(point, 255 - marker.getpixel(point))
                marker = marker.rotate(90 * turn).resize((10 * cell, 10 * cell),
                                                          Image.Resampling.NEAREST)
                frame = Image.new("L", (960, 540), 225)
                x = 180 + (len(scenes) * 17) % 330
                y = 40 + (len(scenes) * 23) % 210
                frame.paste(marker, (x, y))
                lo_x, lo_y = x + cell - 0.5, y + cell - 0.5
                hi_x, hi_y = lo_x + cell * 8, lo_y + cell * 8
                clockwise = [(lo_x, lo_y), (hi_x, lo_y), (hi_x, hi_y), (lo_x, hi_y)]
                corners = [v for i in range(4) for v in clockwise[(i - turn) % 4]]
                name = f"size{cell*8}_q{turn}_damage{damage}"
                path = args.out / (name + ".pgm")
                frame.save(path)
                scenes.append({"name": name, "path": path.name, "sha256": hashlib.sha256(
                    path.read_bytes()).hexdigest(), "flipped_cells": damage,
                    "markers": [{"id": 17, "corners_px": corners}]})
    (args.out / "manifest.json").write_text(json.dumps({"dictionary": "DICT_APRILTAG_36h11",
                                                        "scenes": scenes}, indent=2) + "\n")


if __name__ == "__main__":
    main()
