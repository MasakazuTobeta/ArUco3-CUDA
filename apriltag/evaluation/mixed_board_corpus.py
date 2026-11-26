# SPDX-License-Identifier: Apache-2.0
"""Add a labelled small tag to selected demosaiced TagSLAM camera frames."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from border_corpus import render
from dense_corpus import CODEBOOK, grids_from_codebook


def placements(base, occupied, grids, split, rng):
    height, width = base.shape
    if split == 'oblique':
        for side in (20, 24, 32, 40, 64):
            for lighting in ('normal', 'low-gradient'):
                tag_id = int(rng.integers(100, 500))
                angle = rng.uniform(-1.45, 1.45)
                u = np.array([np.cos(angle), np.sin(angle)]) * (side * .5)
                v = np.array([-np.sin(angle), np.cos(angle)]) * (side * .5)
                for _ in range(1000):
                    center = rng.uniform([100, 100], [width - 100, height - 100])
                    x, y = np.rint(center).astype(int)
                    if not occupied[y-side:y+side, x-side:x+side].any():
                        break
                else:
                    raise ValueError('No free patch for the rotated tag')
                corners = center + np.array([-u-v, u-v, u+v, -u+v]) + rng.normal(0, side*.02, (4, 2))
                image = base.astype(float)
                render(image, corners, grids[tag_id], lighting)
                marker = dict(id=tag_id, corners_px=corners.ravel().tolist(), nominal_side_px=side,
                              lighting=lighting)
                yield f'{side}-{lighting}', np.rint(image).astype(np.uint8), marker
        return
    for dark, light in (((16, 240),) if split == 'scale' else ((138, 214), (16, 91))):
        for side in (24, 40, 64):
            cell = side // 8
            span = 10 * cell
            x, y = next((x, y) for y in range(50, height-span, span//2)
                        for x in range(50, width-span, span//2)
                        if not occupied[y:y+span, x:x+span].any())
            grid = np.full((10, 10), light, dtype=np.uint8)
            grid[1:9, 1:9] = np.where(grids[491] > 0, light, dark)
            image = base.copy()
            image[y:y+span, x:x+span] = grid.repeat(cell, axis=0).repeat(cell, axis=1)
            left, top = x + cell - .5, y + cell - .5
            marker = dict(id=491, corners_px=[left, top, left+side, top, left+side, top+side,
                                             left, top+side], nominal_side_px=side)
            yield str(side) if split == 'scale' else f'{side}-{dark}', image, marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--split', choices=('scale', 'oblique', 'exposure'), required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be empty')
    args.output.mkdir(parents=True, exist_ok=True)
    source = json.loads(args.manifest.read_bytes())['scenes']
    indices = (6, 10, 13, 16, 22, 27, 31, 33) if args.split == 'oblique' else (5, 9, 14, 19, 24, 29, 34)
    grids = grids_from_codebook(CODEBOOK)
    rng = np.random.default_rng(70216)
    rows = []
    for index in indices:
        scene = source[index]
        path = args.manifest.parent / scene['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != scene['sha256']:
            raise ValueError(f'Input digest mismatch for {path}')
        with Image.open(path) as image:
            base = np.asarray(image)
        occupied = np.zeros_like(base)
        for marker in scene['markers']:
            corners = np.array(marker['corners_px']).reshape(4, 2)
            x, y = np.maximum(0, np.floor(corners.min(axis=0)-12).astype(int))
            margin = 12 if args.split == 'oblique' else 14
            right, bottom = np.ceil(corners.max(axis=0)+margin).astype(int)
            occupied[y:bottom, x:right] = 1
        for suffix, pixels, marker in placements(base, occupied, grids, args.split, rng):
            name = f"{scene['name']}-{suffix}"
            output = args.output / (name + '.pgm')
            Image.fromarray(pixels).save(output)
            rows.append({**scene, 'name': name, 'path': output.name,
                         'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                         'markers': scene['markers'] + [marker]})
    (args.output / 'manifest.json').write_text(json.dumps({'scenes': rows}))


if __name__ == '__main__':
    main()
