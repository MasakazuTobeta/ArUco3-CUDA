# SPDX-License-Identifier: Apache-2.0
"""Check the ICRA forward board coordinate origin directly from rasterized rails."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def audit(manifest):
    residuals = [[], []]
    for scene in json.loads(manifest.read_bytes())["scenes"]:
        data = (manifest.parent / scene["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != scene["sha256"]:
            raise ValueError(f"Image digest mismatch in {scene['name']}")
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        bounds = np.array(image.shape[::-1])
        for tag in scene["markers"]:
            corners = np.array(tag["corners_px"]).reshape(4, 2)
            if not ((corners > 7).all() and (corners < bounds - 7).all()):
                continue
            for start, end in zip(corners, np.roll(corners, -1, axis=0)):
                edge = end - start
                major = int(np.argmax(abs(edge)))
                minor = 1 - major
                if abs(edge[minor]) > 0.015 or abs(edge[major]) < 25:
                    continue

                # Stay away from the corners of the print. The source is sampled
                # at integer pixel centers. Two plateaus determine each line's
                # local brightness range and either polarity can be measured.
                lo, hi = sorted([start[major], end[major]])
                along = np.arange(np.ceil(lo + .20 * (hi - lo)),
                                  np.floor(hi - .20 * (hi - lo)) + 1, dtype=int)
                published = start[minor]
                base = int(np.floor(published))
                across = base + np.arange(-3, 4)
                stripe = (image[across[None, :], along[:, None]] if major == 0
                          else image[along[:, None], across[None, :]]).astype(float)
                before, after = stripe[:, :2].mean(axis=1), stripe[:, -2:].mean(axis=1)
                contrast = before - after
                usable = ((abs(contrast) >= 160)
                          & (abs(stripe[:, 0] - stripe[:, 1]) <= 8)
                          & (abs(stripe[:, -1] - stripe[:, -2]) <= 8))
                if np.count_nonzero(usable) < 4:
                    continue
                steps = (stripe[usable] - after[usable, None]) / contrast[usable, None]
                # Normalized intensity integrated from the start of the first
                # pixel gives the edge position in the pixel-center convention.
                measured = base - 3.5 + steps.sum(axis=1)
                residuals[minor].append(float(np.median(measured) - published))

    report = {}
    for axis, entries in zip(("x", "y"), residuals):
        if not entries:
            raise ValueError(f"No fully visible, high-contrast outer rails on {axis}")
        errors = np.array(entries)
        report[axis] = {
            "rails": len(errors),
            "mean_signed_px": float(errors.mean()),
            "rms_vs_published_px": float(np.sqrt(np.mean(errors ** 2))),
            "rms_vs_published_minus_half_px": float(np.sqrt(np.mean((errors + .5) ** 2))),
        }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="prepared ICRA forward manifest")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    print(json.dumps(audit(args.manifest), indent=2))
