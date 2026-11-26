# SPDX-License-Identifier: Apache-2.0
"""Render moving low-contrast 36h11 tags under smooth, nonuniform illumination."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from dense_corpus import CODEBOOK, grids_from_codebook


def draw(image, grid, quad, contrast):
    height, width = image.shape
    low = np.maximum(np.floor(quad.min(axis=0)).astype(int) - 2, [0, 0])
    high = np.minimum(np.ceil(quad.max(axis=0)).astype(int) + 3, [width, height])
    x0, y0 = low
    x1, y1 = high
    unit = np.float32([[0, 0], [8, 0], [8, 8], [0, 8]])
    h = cv2.getPerspectiveTransform(quad.astype('f4'), unit)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    ink = np.zeros_like(xx, dtype='f4')
    for dy in (-.25, .25):
        for dx in (-.25, .25):
            x, y = xx + dx, yy + dy
            z = h[2, 0] * x + h[2, 1] * y + h[2, 2]
            i = np.floor((h[0, 0]*x + h[0, 1]*y + h[0, 2])/z).astype(int)
            j = np.floor((h[1, 0]*x + h[1, 1]*y + h[1, 2])/z).astype(int)
            inside = (i >= 0) & (i < 8) & (j >= 0) & (j < 8)
            ink += inside & (grid[np.clip(j, 0, 7), np.clip(i, 0, 7)] == 0)
    image[y0:y1, x0:x1] -= contrast * .25 * ink


def generate(root, height, seed, sensor=False):
    rng = np.random.default_rng(seed)
    grids = grids_from_codebook(CODEBOOK)
    noise = np.random.default_rng(seed + 1)
    width = height * 16 // 9
    y, x = np.ogrid[-1:1:complex(height), -1:1:complex(width)]
    ids = rng.choice(len(grids), (4, 2), replace=False)
    root.mkdir(parents=True)
    scenes = []
    for clip in range(4):
        phase = rng.uniform(-math.pi, math.pi)
        base = 136 + 23 * (math.cos(phase)*x + math.sin(phase)*y)
        # Include a lighting cross term and curved shading on separate clips.
        background = base + [0, 7*x*y, 6*(x*x+y*y), 5*np.sin(3*x+2*y)][clip]
        for frame in range(9):
            image = background.copy()
            markers = []
            for branch, (cx, cy, side, contrast) in enumerate(
                    ((.32, .50, .22, [11, 16, 23, 31][clip]), (.69, .58, .14, [18, 9, 88, 15][clip]))):
                if not (1+branch <= frame <= 7-branch):
                    continue
                theta = phase+.07*frame+1.3*branch
                center = np.array([width*(cx+.018*math.sin(frame)), height*(cy+.025*math.cos(frame))])
                a = np.array([math.cos(theta), math.sin(theta)]) * .5*height*side
                b = np.array([-a[1], a[0]])*.88
                quad = center+np.array([-a-b,a-b,a+b+.085*a,-a+b])
                code = int(ids[clip, branch])
                draw(image, grids[code], quad, contrast)
                markers.append(dict(id=code, track=str(branch), first_visible_frame=1+branch,
                                    contrast=contrast, corners_px=quad.ravel().tolist()))
            name = f'gradient-{clip}-{frame}'
            path = root / (name+'.pgm')
            if sensor:
                # Blur and read noise act after the two physical prints are projected.
                image = cv2.GaussianBlur(image, (0, 0), (0.6, 1.0, 1.4, 0.8)[clip])
                image += noise.normal(0, (0.35, 0.65, 0.9, 0.5)[clip], image.shape)
            data = f'P5\n{width} {height}\n255\n'.encode()+np.rint(image).clip(0,255).astype('u1').tobytes()
            path.write_bytes(data)
            scenes.append(dict(name=name, path=path.name, sha256=hashlib.sha256(data).hexdigest(),
                               markers=markers, clip=str(clip), frame_index=frame,
                               must_be_absent_ids=sorted(set(map(int,ids[clip]))-{m['id'] for m in markers})))
    manifest = dict(dictionary='tag36h11', width=width, height=height, seed=seed, scenes=scenes)
    if sensor:
        manifest['image_formation'] = 'Gaussian blur and independent per-frame read noise'
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--height', type=int, choices=(720,1080,2160), default=720)
    parser.add_argument('--seed', type=int, default=729681)
    parser.add_argument('--sensor', action='store_true', help='add optical blur and read noise')
    args = parser.parse_args()
    generate(args.output, args.height, args.seed, args.sensor)
