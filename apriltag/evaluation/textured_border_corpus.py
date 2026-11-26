# SPDX-License-Identifier: Apache-2.0
"""Render independent textured edge negatives and genuine AprilTag controls."""
import argparse
from itertools import product
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from dense_corpus import CODEBOOK, grids_from_codebook

SEED = 6013129403
# Width, height, negative frames, positive frames.
CASES = ((640, 480, 32, 16), (713, 509, 32, 16), (1280, 720, 24, 12),
         (1921, 1081, 24, 12), (3840, 2160, 8, 4))
NEGATIVES = ('invalid_grid', 'broken_grid', 'polygons', 'edge_stripes')
POSITIVES = ('naked', 'matte')
MANIFEST_SHA256 = '460d494571741e0ddfb48550f594c30f4d13838c24409cc297da5b9c9e44c3f2'
SIDES = ('top', 'right', 'bottom', 'left')
UNIT = np.array(((0, 0), (8, 0), (8, 8), (0, 8)), np.float32)
SQUARE = np.array(((-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)), np.float32)


def background(width, height, surface, rng):
    y, x = np.ogrid[:height, :width]
    scale = max(1., width / 730)
    soft_noise = cv2.resize(rng.normal(0, 14, (21, 27)).astype(np.float32), (width, height),
                            interpolation=cv2.INTER_CUBIC)
    phase = rng.uniform(0, 6)
    surface %= 4
    if surface == 0:
        values = 115 + 26 * np.sin((x + 1.2 * y) / (55 * scale) + phase)
    elif surface == 1:
        values = 121 + 37 * np.cos(x / (26 * scale) + phase) * np.cos(y / (48 * scale))
    elif surface == 2:
        values = 90 + 27 * ((x + 2.7 * y) % (99 * scale) < 41 * scale) + .021 * x / scale
    else:
        values = (133 + 24 * np.sin((y - .5 * x) / (47 * scale) + phase) +
                  14 * ((x // (26 * scale) + y // (23 * scale)) % 2))
    return (values + soft_noise).astype(np.float32)


def position(width, height, side, length, clip, spin, rng, offset):
    shape = SQUARE + rng.uniform(-.102, .102, (4, 2))
    angle = rng.choice((-.62, -.31, .10, .41, .74)) + rng.uniform(-.10, .10)
    if spin is None:
        angle *= rng.choice((.35, 1.0))
    c, s = np.cos(angle), np.sin(angle)
    points = length * shape @ np.array(((c, s), (-s, c)))
    axis = int(side in ('top', 'bottom'))
    far = side in ('right', 'bottom')
    extent = (width, height)[axis]
    limit = extent - .5 if far else -.5
    edge = points[:, axis].max() if far else points[:, axis].min()
    points[:, axis] += limit - edge + (clip if far else -clip)
    points[:, 1 - axis] += (height, width)[axis] * offset + rng.uniform(-.45, .45)
    return points.astype(np.float32)


def render_grid(image, points, cells, low, high, matte, rng):
    height, width = image.shape
    spread = np.ptp(points, axis=0)
    left, top = np.maximum(0, np.floor(points.min(axis=0) - .20 * spread - 2).astype(int))
    right, bottom = np.minimum((width, height), np.ceil(points.max(axis=0) + .20 * spread + 2).astype(int))
    if left >= right or top >= bottom:
        raise ValueError('no visible print')
    x, y = np.meshgrid(np.arange(left, right, dtype=np.float32),
                       np.arange(top, bottom, dtype=np.float32))
    inverse = cv2.getPerspectiveTransform(points, UNIT).astype(np.float32)
    patch = image[top:bottom, left:right]
    ink = np.where(cells > 0, high, low) + rng.uniform(-12, 12, (8, 8))
    shadow = rng.uniform(-3.5, 3.5)
    result = np.zeros_like(patch)
    for dx, dy in product((-.375, -.125, .125, .375), repeat=2):
        xx, yy = x + dx, y + dy
        w = inverse[2, 0] * xx + inverse[2, 1] * yy + inverse[2, 2]
        u = (inverse[0, 0] * xx + inverse[0, 1] * yy + inverse[0, 2]) / w
        v = (inverse[1, 0] * xx + inverse[1, 1] * yy + inverse[1, 2]) / w
        inside = (u >= 0) & (u < 8) & (v >= 0) & (v < 8)
        columns = np.floor(u).astype(int).clip(0, 7)
        rows = np.floor(v).astype(int).clip(0, 7)
        background = patch
        if matte:
            paper = (u > -.65) & (u < 8.65) & (v > -.65) & (v < 8.65)
            background = np.where(paper, high - 15 + shadow * (v - 4), patch)
        result += np.where(inside, ink[rows, columns] + shadow * (u - 4), background)
    patch[:] = result / 16


def random_grids(dictionary, rng):
    rotations = np.concatenate([np.rot90(dictionary[:, 1:7, 1:7], i, axes=(1, 2)).reshape(-1, 36)
                                for i in range(4)]) != 0
    while True:
        noise = rng.integers(0, 2, (6, 6), dtype=np.uint8)
        distance = int(np.count_nonzero(rotations != noise.ravel(), axis=1).min())
        if distance < 7:
            continue
        cells = np.zeros((8, 8), np.uint8)
        cells[1:7, 1:7] = noise * 255
        yield cells, distance


def unrelated_edges(image, rng, kind):
    height, width = image.shape
    size = max(1, width // 620)
    for _ in range(7):
        side = SIDES[int(rng.integers(4))]
        length = float(rng.uniform(22, 140)) * size
        points = position(width, height, side, length, length * rng.uniform(.12, .38),
                          None, rng, rng.uniform(.05, .95))
        if kind == 'polygons':
            cv2.fillConvexPoly(image, np.rint(points).astype(np.int32), float(rng.integers(5, 69)))
        else:
            top, right = np.rint(points[:2]).astype(int)
            cv2.line(image, tuple(top), tuple(right), float(rng.integers(7, 92)),
                     int(rng.integers(2, 14)) * size)


def generate(output, seed=SEED):
    rng = np.random.default_rng(seed)
    dictionary = grids_from_codebook(CODEBOOK)
    false_patterns = random_grids(dictionary, rng)
    true_ids = iter(rng.choice(np.arange(32, len(dictionary)),
                               sum(case[3] for case in CASES), replace=False))
    records = []
    cv2.setNumThreads(1)
    output.mkdir(parents=True)
    for size_index, (width, height, negatives, positives) in enumerate(CASES):
        specifications = [(False, i) for i in range(negatives)] + [(True, i) for i in range(positives)]
        for order, case in enumerate(rng.permutation(len(specifications))):
            is_positive, index = specifications[case]
            kind = (POSITIVES[index % 2] if is_positive else
                    NEGATIVES[index % 4])
            image = background(width, height, index + size_index, rng)
            # Long silhouettes from nearby structures cross frame edges without tag payloads.
            unrelated_edges(image, rng, kind)
            side = SIDES[(index // 2 + size_index) % 4]
            markers, distances = [], []
            if kind not in {'polygons', 'edge_stripes'}:
                count = 1 if is_positive else 2
                for print_index in range(count):
                    scale = min(5.3, width / 625)
                    length = rng.uniform(31, 93) * scale
                    clip = length * rng.uniform(.035, .24)
                    if is_positive and index % 4 == 0:
                        clip = -2.2 * scale
                    spin = int(rng.integers(4))
                    points = position(width, height, side, length, clip, spin, rng,
                                      .30 + .40 * print_index + rng.uniform(-.037, .037))
                    if is_positive:
                        tag_id = int(next(true_ids))
                        cells = np.rot90(dictionary[tag_id], -spin)
                        ordered = np.roll(points, -spin, axis=0)
                        visible = ((ordered >= (-.5, -.5)) &
                                   (ordered <= (width - .5, height - .5))).all(axis=1)
                        markers.append({'id': tag_id, 'corners_px': ordered.ravel().tolist(),
                                        'visible_corners': int(visible.sum())})
                    else:
                        cells, distance = next(false_patterns)
                        distances.append(distance)
                    low, high = ((rng.uniform(8, 33), rng.uniform(153, 198))
                                 if index % 3 else (rng.uniform(38, 66), rng.uniform(116, 148)))
                    render_grid(image, points, cells, low, high, kind in ('matte', 'invalid_grid'), rng)
                    # Broken print panels contain dark, non-payload occluders.
                    if kind == 'broken_grid':
                        a, b = np.rint((points[0] + points[3]) * .5).astype(int), np.rint(
                            (points[1] + points[2]) * .5).astype(int)
                        cv2.line(image, tuple(a), tuple(b), float(low + 7), max(2, round(length * .055)))
            sigma = (.39, .63, .81)[index % 3] * min(2, width / 720)
            image = cv2.GaussianBlur(image, (7, 7), sigma)
            image += rng.normal(0, 2.9, image.shape)
            png = np.rint(image).clip(0, 255).astype(np.uint8)
            name = f'{width}x{height}_{order:02}_{kind}_{side}'
            data = f'P5\n{width} {height}\n255\n'.encode() + png.tobytes()
            file = name + '.pgm'
            (output / file).write_bytes(data)
            records.append({'name': name, 'path': file, 'sha256': hashlib.sha256(data).hexdigest(),
                            'width_px': width, 'height_px': height, 'case': kind,
                            'edge': side, 'non_dictionary_minimum_errors': distances,
                            'markers': markers})
    manifest = {'dictionary': 'tag36h11', 'seed': seed, 'scenes': records}
    data = (json.dumps(manifest, indent=2) + '\n').encode()
    if seed == SEED and hashlib.sha256(data).hexdigest() != MANIFEST_SHA256:
        raise ValueError('textured border manifest differs from the pinned inputs')
    (output / 'manifest.json').write_bytes(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output, args.seed)
