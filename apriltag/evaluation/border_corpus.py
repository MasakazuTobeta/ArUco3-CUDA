# SPDX-License-Identifier: Apache-2.0
"""Render the reproducible AprilTag boundary evaluation scenes."""
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path

import cv2
import numpy as np

from small_corpus import homography

SIDES = ('top', 'right', 'bottom', 'left')
DEPTHS = (-2.5, 0.5, 4.5, 12.5)
LIGHTING = ('normal', 'low-gradient')
ATTITUDES = ('upright', 'oblique')
CLUTTER = ('plain', 'dark-bar')
SQUARE = np.array([[-0.50, -0.50], [0.50, -0.50], [0.50, 0.50], [-0.50, 0.50]])
SKEW = np.array([[-0.49, -0.54], [0.54, -0.39], [0.42, 0.53], [-0.49, 0.48]])
SEALED_SEEDS = {'640x480': 671739142, '3840x2160': 221574661}
DICTIONARY_SHA256 = 'cb7998b3c1c0024fb63876eac214314eaea718a29a57ac3065d5482fec8de24a'


def dictionary_grids():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    grids = [cv2.aruco.generateImageMarker(dictionary, marker_id, 8) for marker_id in range(32)]
    if hashlib.sha256(b''.join(grid.tobytes() for grid in grids)).hexdigest() != DICTIONARY_SHA256:
        raise ValueError('OpenCV tag36h11 grids differ from the pinned 4.14 dictionary')
    return grids


def axis_index(side):
    return int(side in ('top', 'bottom'))


def tag_position(angle, side, cross, depth, size, width, height):
    radians = np.deg2rad(angle)
    rotate = np.array([[np.cos(radians), np.sin(radians)], [-np.sin(radians), np.cos(radians)]])
    shape = SQUARE if angle % 90 == 0 else SKEW
    corners = (size * shape @ rotate).copy()
    axis = axis_index(side)
    far = side in ('bottom', 'right')
    limit = (width, height)[axis] - 0.5 if far else -0.5
    extreme = corners[:, axis].max() if far else corners[:, axis].min()
    corners[:, axis] += limit - extreme + (depth if far else -depth)
    corners[:, 1 - axis] += cross
    return corners


def render(canvas, corners, cells, light):
    height, width = canvas.shape
    h = homography(corners)
    spread = np.ptp(corners, axis=0)
    left, top = np.maximum(0, np.floor(corners.min(axis=0) - 0.31 * spread - 2).astype(int))
    right, bottom = np.minimum([width, height], np.ceil(corners.max(axis=0) + 0.31 * spread + 2).astype(int))
    x, y = np.meshgrid(np.arange(left, right), np.arange(top, bottom))
    background = canvas[top:bottom, left:right].copy()
    samples = np.zeros_like(background)
    for ox, oy in product((-0.375, -0.125, 0.125, 0.375), repeat=2):
        xx, yy = x + ox, y + oy
        divisor = h[6] * xx + h[7] * yy + 1.0
        u = (h[0] * xx + h[1] * yy + h[2]) / divisor
        v = (h[3] * xx + h[4] * yy + h[5]) / divisor
        tag = (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
        quiet_zone = (u >= -0.11) & (u < 1.11) & (v >= -0.11) & (v < 1.11)
        i = np.clip((u * 8).astype(int), 0, 7)
        j = np.clip((v * 8).astype(int), 0, 7)
        white = (cells[j, i] == 255) | ~tag
        if light == 'normal':
            luma = np.where(white, 240, 16)
        else:
            luma = np.where(white, 164, 72) + 53 * (u - 0.5) - 17 * (v - 0.5)
        samples += np.where(quiet_zone, luma, background)
    canvas[top:bottom, left:right] = samples / 16


def add_dark_edge(canvas, side, tag, scale):
    axis = axis_index(side)
    coordinates = np.indices(canvas.shape)
    across = coordinates[1 - axis]
    along = coordinates[axis]
    bounds = tag[:, axis] if tag is not None else scale * np.array([61.0, 71.0])
    shift = 5.5 * scale
    offset = bounds.min() - shift if side in ('bottom', 'right') else bounds.max() + shift
    delta = across - (offset + 0.035 * (along - 220.0 * scale))
    mask = (np.abs(delta) <= 2.0 * scale) & (along >= 48 * scale) & (along <= 380 * scale)
    canvas[mask] = 24


def scenes_for_resolution(native):
    if not native:
        cases = [(side, depth, light, attitude, clutter, 'same-side')
                 for side, depth, light, attitude, clutter
                 in product(SIDES, DEPTHS, LIGHTING, ATTITUDES, CLUTTER)]
        cases.extend((side, depth, light, attitude, 'dark-bar', 'corner-pair')
                     for side, depth, light, attitude
                     in product(SIDES, (-2.5, 4.5), LIGHTING, ATTITUDES))
        cases.extend((side, 0, light, 'upright', 'dark-bar', 'negative')
                     for side, light in product(SIDES, LIGHTING))
        return cases
    cases = []
    for i, (side, depth, attitude) in enumerate(product(SIDES, DEPTHS, ATTITUDES)):
        cases.append((side, depth, LIGHTING[(i // 2 + i % 2) % 2], attitude,
                      CLUTTER[(i + SIDES.index(side)) % 2], 'same-side'))
    for i, (side, depth) in enumerate(product(SIDES, (-2.5, 4.5))):
        cases.append((side, depth, LIGHTING[i % 2], ATTITUDES[i // 2 % 2], 'dark-bar', 'corner-pair'))
    cases.extend((side, 0.0, 'normal', 'upright', 'dark-bar', 'negative') for side in SIDES)
    return cases


def tag_geometry(side, depth, attitude, layout, width, height, native):
    angles = (0, 90) if attitude == 'upright' else (27, -113)
    if not native:
        if layout == 'same-side':
            return [(angles[0], side, 156.35, depth, 62),
                    (angles[1], side, 238.65, max(-5.0, depth - 2), 56)]
        offsets = {'top': (width - 121.25, 120.75), 'right': (height - 120.5, width - 121.25),
                   'bottom': (121.65, height - 123.75), 'left': (122.35, 123.25)}
    else:
        if layout == 'same-side':
            center = (width, height)[1 - axis_index(side)] * 0.40
            return [(angles[0], side, center + 0.35, depth * 4.0, 62 * 4.0),
                    (angles[1], side, center + 82.3 * 4.0, max(-5.0, depth - 2.0) * 4.0, 56 * 4.0)]
        offsets = {'top': (width - 121 * 4.0, 120 * 4.0), 'right': (height - 120 * 4.0, width - 121 * 4.0),
                   'bottom': (121 * 4.0, height - 123 * 4.0), 'left': (122 * 4.0, 123 * 4.0)}
    extent = 4.0 if native else 1
    first, second = offsets[side]
    next_side = SIDES[(SIDES.index(side) + 1) % len(SIDES)]
    return [(angles[0], side, first, depth * extent, 61 * extent),
            (angles[1], next_side, second, depth * extent, 58 * extent)]


def generate(output, resolution, seed, grids):
    native = resolution == '3840x2160'
    if native:
        width, height, scale = 3840, 2160, 4.0
        original_seed = 31106428
        x = np.arange(width)[None, :]
        y = np.arange(height)[:, None]
        background = 173 + 0.009 * x + 0.006 * y + 7 * np.cos(x / 174 + y / 85)
    else:
        width, height, scale = 640, 480, 1
        original_seed = 6730909
        y, x = np.indices((height, width))
        background = 172 + 0.04 * x + 0.020 * y + 6 * np.cos(x / 61 + y / 22)
    cases = scenes_for_resolution(native)
    records = []
    for sequence, case_index in enumerate(np.random.default_rng(seed).permutation(len(cases))):
        case_index = int(case_index)
        side, depth, light, attitude, clutter, layout = cases[case_index]
        rng = np.random.default_rng(seed + case_index + 1)
        canvas = background + rng.normal(0, 2.0, (height, width))
        tags = []
        if layout != 'negative':
            geometry = tag_geometry(side, depth, attitude, layout, width, height, native)
            if seed != original_seed:
                cross_range = 10 if native else 4
                geometry = [(angle + float(rng.uniform(-7, 7)) if attitude == 'oblique' else angle,
                             tag_side, cross + float(rng.uniform(-cross_range, cross_range)),
                             clip, size * float(rng.uniform(0.92, 1.1)))
                            for angle, tag_side, cross, clip, size in geometry]
            tags = [(int(marker_id), tag_position(*params, width, height))
                    for marker_id, params in zip(rng.choice(32, 2, replace=False), geometry, strict=True)]
        if clutter == 'dark-bar':
            add_dark_edge(canvas, side, tags[0][1] if tags else None, scale)
        markers = []
        for marker_id, corners in tags:
            render(canvas, corners, grids[marker_id], light)
            visible = ((corners >= [-0.5, -0.5]) & (corners <= [width - 0.5, height - 0.5])).all(axis=1)
            markers.append({'id': marker_id, 'corners_px': corners.ravel().tolist(),
                            'visible_corners': int(visible.sum())})
        prefix = 'native' if native else 'border'
        name = f'{prefix}_{sequence:03}_{side}_{depth:g}_{light}_{attitude}_{clutter}_{layout}'.replace('.', 'p')
        path = output / (name + '.pgm')
        pixels = np.round(canvas).clip(0, 255).astype(np.uint8)
        data = f'P5\n{width} {height}\n255\n'.encode() + pixels.tobytes()
        path.write_bytes(data)
        records.append({'name': name, 'path': path.name, 'sha256': hashlib.sha256(data).hexdigest(),
                        'layout': layout, 'markers': markers, 'primary_side': side,
                        'nominal_clip_depth_px': depth * scale, 'lighting': light, 'attitude': attitude,
                        'distractor': clutter, 'width_px': width, 'height_px': height})
    # Preserve the baseline manifests' field order for byte-for-byte reproduction.
    manifest = {'seed': seed, 'dictionary': 'tag36h11'} if native else {'dictionary': 'tag36h11', 'seed': seed}
    manifest['scenes'] = records
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--resolution', choices=SEALED_SEEDS, required=True)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    seed = SEALED_SEEDS[args.resolution] if args.seed is None else args.seed
    if seed < 0:
        parser.error('seed must be nonnegative')
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        parser.error('output directory must be empty')
    generate(args.out, args.resolution, seed, dictionary_grids())


if __name__ == '__main__':
    main()
