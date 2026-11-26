# SPDX-License-Identifier: Apache-2.0
"""Generate tag36h11 prints that can appear as holes in a stride-three candidate mask."""
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path

import cv2
import numpy as np

from dense_corpus import CODEBOOK, grids_from_codebook


SOURCE_CORNERS = np.float32([[-.5, -.5], [7.5, -.5], [7.5, 7.5], [-.5, 7.5]])


def generate(output):
    grids = grids_from_codebook(CODEBOOK)
    output.mkdir(parents=True, exist_ok=True)
    cases = product((46, 55, 64, 73, 90, 104, 122), (0, 8, 17),
                    (-2.0, -.4, 1.0, 3.0), ('halftone', 'solid'), (0, 1))
    scenes = []
    for index, (side, angle, margin, surround, shift) in enumerate(cases):
        center = (130.2 + shift, 98.3)
        corners = cv2.boxPoints((center, (side, side), angle))
        outer_side = side + 2 * margin
        paper = cv2.boxPoints((center, (outer_side, outer_side), angle))
        marker_id = 17 + index % 13
        cells = np.where(grids[marker_id], 228, 134).astype(np.uint8)
        transform = cv2.getPerspectiveTransform(SOURCE_CORNERS, corners)
        image = cv2.warpPerspective(cells, transform, (259, 201), flags=cv2.INTER_NEAREST,
                                    borderValue=228)
        mask = np.zeros_like(image)
        cv2.fillConvexPoly(mask, np.rint(paper).astype(np.int32), 255)
        if surround == 'halftone':
            # Only dots outside the paper land at the coarse sampling centers.
            dots = image[1::3, 1::3]
            dots[mask[1::3, 1::3] == 0] = 5
        else:
            image[mask == 0] = 8
        path = output / f'{index}.pgm'
        if not cv2.imwrite(str(path), image):
            raise OSError(f'could not write {path}')
        scenes.append(dict(name=str(index), path=path.name,
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           markers=[dict(id=marker_id, corners_px=corners.ravel().tolist())],
                           nominal_side_px=side, rotation_degrees=angle, margin_px=margin,
                           surround=surround, horizontal_shift_px=shift))
    manifest = output / 'manifest.json'
    manifest.write_text(json.dumps(dict(dictionary='tag36h11', scenes=scenes), indent=2) + '\n')
    print(f'{manifest}  {hashlib.sha256(manifest.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    generate(parser.parse_args().output)
