# SPDX-License-Identifier: Apache-2.0
"""Render a deterministic changing-frame tag36h11 small-marker corpus."""
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np
import cv2

SIZES = (12, 16, 20, 24, 28, 36, 48)
BACKGROUNDS = ('flat', 'grain', 'structured')
LIGHTING = ('normal', 'low_contrast')
ROTATIONS = ('upright', 'oblique')
PHASES = (0.125, 0.625)
SEED = 202611211308
WIDTH = 1280
HEIGHT = 720

def background(kind, x, y, rng):
    if kind == 'flat':
        return np.clip(179 + 0.014 * (x - WIDTH // 2) + 0.023 * (y - HEIGHT // 2), 0, 255)
    if kind == 'grain':
        smooth = 20 * np.sin(x / 55 + y / 38) + 13 * np.cos(x / 23 - y / 29)
        return np.clip(172 + smooth + rng.normal(0, 5, size=(HEIGHT, WIDTH)), 0, 255)
    bands = ((x + 2 * y) % 92 < 31)
    cells = ((x // 23 + y // 17) % 2)
    return np.clip(139 + 45 * bands + 25 * cells + rng.normal(0, 2, size=(HEIGHT, WIDTH)), 0, 255)


def homography(corners):
    unit = np.array([(0., 0.), (1., 0.), (1., 1.), (0., 1.)])
    rows = []
    vals = []
    for (x, y), (u, v) in zip(corners, unit, strict=True):
        rows += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
        vals += [u, v]
    return np.linalg.solve(rows, vals)


def render_tag(base, bits, side, rotation, phase, light, rng):
    center = np.array([rng.integers(80, WIDTH - 80) + phase, rng.integers(80, HEIGHT - 80) + 1 - phase])
    if rotation == 'oblique':
        shape = np.array([[-0.51, -0.55], [0.51, -0.39], [0.42, 0.53], [-0.49, 0.47]])
        a = np.deg2rad(31 if rng.random() < .5 else -27)
    else:
        shape = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])
        a = 0
    transform = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    corners = (side * shape) @ transform + center
    h = homography(corners)
    left, top = np.floor(corners.min(axis=0) - side * .25 - 2).astype(int)
    right, bottom = np.ceil(corners.max(axis=0) + side * .25 + 2).astype(int)
    x, y = np.meshgrid(np.arange(left, right), np.arange(top, bottom))
    patch = base[top:bottom, left:right].copy()
    samples = np.zeros_like(patch)
    for oy, ox in product((-.375, -.125, .125, .375), repeat=2):
        sx, sy = x + ox, y + oy
        divisor = h[6] * sx + h[7] * sy + 1
        u = (h[0] * sx + h[1] * sy + h[2]) / divisor
        v = (h[3] * sx + h[4] * sy + h[5]) / divisor
        tag = ((u >= 0) & (u < 1) & (v >= 0) & (v < 1))
        quiet_zone = ((u >= -.14) & (u < 1.14) & (v >= -.14) & (v < 1.14))
        ix = np.clip(np.floor(u * bits.shape[1]).astype(int), 0, bits.shape[1] - 1)
        iy = np.clip(np.floor(v * bits.shape[0]).astype(int), 0, bits.shape[0] - 1)
        white = ((bits[iy, ix] > 0) | ~tag)
        if light == 'normal':
            tag_luma = np.where(white, 239, 18)
        else:
            tag_luma = np.where(white, 157, 92) + 30 * (u - .5) + 12 * (v - .5)
        samples += np.where(quiet_zone, tag_luma, patch)
    base[top:bottom, left:right] = samples / 16
    return corners


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error('seed must be nonnegative')
    output = args.out
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("output directory must be empty")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    grids = [cv2.aruco.generateImageMarker(dictionary, i, 8) for i in range(32)]
    # Check against the 32 grids exported with the pinned OpenCV 4.14 C++ dictionary.
    expected = 'cb7998b3c1c0024fb63876eac214314eaea718a29a57ac3065d5482fec8de24a'
    if hashlib.sha256(b''.join(grid.tobytes() for grid in grids)).hexdigest() != expected:
        raise ValueError('OpenCV tag36h11 grids differ from the pinned dictionary')
    positive = list(product(SIZES, BACKGROUNDS, LIGHTING, ROTATIONS, PHASES))
    negatives = [(0, kind, 'normal', 'upright', phase) for kind, phase in product(BACKGROUNDS, PHASES)] * 4
    specifications = positive + negatives
    order = np.random.default_rng(args.seed).permutation(len(specifications))
    x = np.arange(WIDTH)[None, :]
    y = np.arange(HEIGHT)[:, None]
    records = []
    for sequence, spec_index in enumerate(order):
        spec_index = int(spec_index)
        side, kind, light, rotation, phase = specifications[spec_index]
        rng = np.random.default_rng(args.seed + 1 + spec_index)
        image = background(kind, x, y, rng)
        markers = []
        if side:
            marker_id = int(rng.integers(32))
            bits = grids[marker_id]
            corners = render_tag(image, bits, side, rotation, phase, light, rng)
            markers.append({'id': marker_id, 'corners_px': corners.ravel().tolist()})
        pixels = np.round(image).clip(0, 255).astype(np.uint8)
        name = f'heldout_{sequence:03}_{side:02}px_{kind}_{light}_{rotation}_{phase}'
        pnm = output / (name + '.pgm')
        pnm.write_bytes(f'P5\n{WIDTH} {HEIGHT}\n255\n'.encode() + pixels.tobytes())
        records.append({
            'name': name, 'path': pnm.name, 'sha256': hashlib.sha256(pnm.read_bytes()).hexdigest(),
            'width_px': WIDTH, 'height_px': HEIGHT, 'tag_side_px': side, 'lighting': light,
            'rotation': rotation, 'background': kind, 'subpixel_phase': phase, 'markers': markers,
        })
    manifest = {'dictionary': 'tag36h11', 'generator_seed': args.seed, 'scenes': records}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
