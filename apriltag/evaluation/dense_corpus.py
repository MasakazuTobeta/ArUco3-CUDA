# SPDX-License-Identifier: Apache-2.0
"""Render alternating sparse and dense tag36h11 arrays on a moving planar board."""
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import re

import cv2
import numpy as np


SEED = 29817436
IMAGE_SIZES = {'720p': (1280, 720), '1080p': (1920, 1080)}
TAG_SIDES = (9, 13, 20, 31, 44, 60)
THEMES = ('matte', 'textured', 'dim')
SUPERSAMPLE = 4
CODEBOOK = (Path(__file__).resolve().parents[2] /
            'src/dictionary/generated/dict_apriltag_36h11.cpp')
CODEBOOK_SHA256 = 'f967e3bd4f65b135ea4244769f270ff84c44f072443457cb4497e3eb0ad9bd8f'


def grids_from_codebook(path):
    words = [int(word, 16) for word in re.findall(r'0x([0-9a-f]{16})ULL', path.read_text())]
    if len(words) != 4 * 587:
        raise ValueError('expected four rotations of each tag36h11 codeword')
    grids = np.zeros((587, 8, 8), dtype=np.uint8)
    for index, word in enumerate(words[::4]):
        grids[index, 1:7, 1:7] = np.array(
            [255 * ((word >> bit) & 1) for bit in range(36)], dtype=np.uint8).reshape(6, 6)
    if hashlib.sha256(grids.tobytes()).hexdigest() != CODEBOOK_SHA256:
        raise ValueError('canonical grids differ from the pinned project codebook')
    return grids


def corners(width, height):
    return np.array([(0, 0), (width, 0), (width, height), (0, height)], np.float32)


def backdrop(width, height, theme, rng):
    y, x = np.ogrid[:height, :width]
    waves = np.sin(x / 79 + y / 47 + rng.uniform(0, 3))
    if theme == 'matte':
        return 151 + 12 * waves + .024 * x + .025 * y
    if theme == 'textured':
        pattern = (x // 29 + y // 21) % 2
        return 107 + 34 * pattern + 26 * waves + 29 * ((x + 2 * y) % 93 < 23)
    return 88 + 22 * waves + .011 * x + 7 * np.cos(x / 16 - y / 28)


def make_pose(board_width, board_height, width, height, theme, oblique, rng):
    source = corners(board_width, board_height)
    arranged = source.copy()
    arranged -= np.array([board_width / 2, board_height / 2])
    if oblique:
        # Converging edges create a coherent planar scale gradient over the full board.
        arranged *= np.array([[1.03, .88], [.90, .99], [.85, .85], [1.09, .97]])
    limit = {'matte': (3.0, 8.0), 'textured': (11.0, 22.0), 'dim': (6.0, 17.0)}[theme]
    angle = np.deg2rad(rng.uniform(*limit) * rng.choice((-1, 1)))
    rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]], np.float32)
    arranged = arranged @ rotation
    bounds = np.ptp(arranged, axis=0)
    arranged *= min(1, .92 * width / bounds[0], .87 * height / bounds[1])
    arranged += np.array([width * .5, height * .5]) + rng.uniform(-9, 9, 2)
    return source, arranged


def place_board(canvas, board, source, arranged):
    height, width = canvas.shape
    left, top = np.maximum(0, np.floor(arranged.min(axis=0) - 2).astype(int))
    right, bottom = np.minimum([width, height], np.ceil(arranged.max(axis=0) + 2).astype(int))
    patch = canvas[top:bottom, left:right]
    size = ((right - left) * SUPERSAMPLE, (bottom - top) * SUPERSAMPLE)
    raster = cv2.resize(patch, size, interpolation=cv2.INTER_NEAREST)
    # OpenCV coordinates are pixel centers. The board vertices describe cell edges.
    high_from = source * SUPERSAMPLE - .5
    high_to = (arranged - np.array([left, top], np.float32) + .5) * SUPERSAMPLE - .5
    homography = cv2.getPerspectiveTransform(high_from, high_to)
    cv2.warpPerspective(board, homography, size, dst=raster, flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_TRANSPARENT)
    patch[:] = cv2.resize(raster, (right - left, bottom - top), interpolation=cv2.INTER_AREA)


def generate_pair(width, height, side, theme, grid_bank, rng):
    pad = max(2, int(round(.24 * side)))
    pitch = side + 2 * pad
    cols = min(22, int(.84 * width) // pitch)
    rows = min(15, int(.76 * height) // pitch)
    board_width, board_height = cols * pitch, rows * pitch
    cells = rows * cols
    ids = rng.choice(len(grid_bank), cells, replace=False)
    spins = rng.integers(0, 4, cells)
    low = set(rng.choice(cells, max(8, round(cells * .24)), replace=False))
    source, arranged = make_pose(board_width, board_height, width, height, theme,
                                 bool(rng.integers(2)) or theme == 'textured', rng)
    background = backdrop(width, height, theme, rng).astype(np.float32)
    black, white = {'matte': (17, 226), 'textured': (44, 200), 'dim': (79, 156)}[theme]
    print_side = side * SUPERSAMPLE

    for density in ('low', 'high'):
        board = np.full((board_height * SUPERSAMPLE, board_width * SUPERSAMPLE),
                        white, dtype=np.float32)
        if density == 'high':
            arranged = arranged + np.array([.675, -.425], np.float32)
        transform = cv2.getPerspectiveTransform(source, arranged)
        markers = []
        for index in range(cells):
            if density == 'low' and index not in low:
                continue
            y, x = divmod(index, cols)
            x, y = x * pitch + pad, y * pitch + pad
            tag_id, spin = int(ids[index]), int(spins[index])
            printing = np.rot90(grid_bank[tag_id], -spin)
            tag = cv2.resize(printing, (print_side, print_side), interpolation=cv2.INTER_NEAREST)
            board[y * SUPERSAMPLE:y * SUPERSAMPLE + print_side,
                  x * SUPERSAMPLE:x * SUPERSAMPLE + print_side] = np.where(tag > 0, white, black)
            local = np.roll(corners(side, side) + np.array([x, y], np.float32), -spin, axis=0)
            image_corners = cv2.perspectiveTransform(local[None], transform)[0]
            markers.append({'id': tag_id, 'corners_px': image_corners.ravel().tolist()})

        # Apply an illumination gradient across the printed sheet before camera blur.
        gain = np.linspace(.85 if theme == 'dim' else .94, 1.10, board.shape[1], dtype=np.float32)
        board *= gain[None, :]
        image = background.copy()
        place_board(image, board, source, arranged)
        yield density, finish(image, theme, rng), markers


def finish(image, theme, rng):
    sigma = {'matte': .38, 'textured': .62, 'dim': .80}[theme]
    image = cv2.GaussianBlur(image, (7, 7), sigma)
    image += rng.normal(0, 1.5 if theme == 'matte' else 3.0, image.shape)
    return np.rint(image).clip(0, 255).astype(np.uint8)


def generate(output, resolution, seed, grids):
    width, height = IMAGE_SIZES[resolution]
    rng = np.random.default_rng(seed)
    pairs = list(product(TAG_SIDES, THEMES))
    scenes = []

    def record(name, image, side, density, theme, markers):
        file = name + '.pgm'
        data = f'P5\n{width} {height}\n255\n'.encode() + image.tobytes()
        (output / file).write_bytes(data)
        scenes.append({'name': name, 'path': file, 'sha256': hashlib.sha256(data).hexdigest(),
                       'width_px': width, 'height_px': height, 'nominal_tag_side_px': side,
                       'density': density, 'appearance': theme, 'markers': markers})

    for index, pair in enumerate(rng.permutation(len(pairs))):
        side, theme = pairs[pair]
        for density, image, markers in generate_pair(width, height, side, theme, grids, rng):
            record(f'{index:02}_{side:02}px_{theme}_{density}', image, side, density, theme, markers)
        if index % 3 == 2:
            # Edge-rich backgrounds with no valid tags exercise rejection.
            image = backdrop(width, height, theme, rng).astype(np.float32)
            for _ in range(25):
                x, y = rng.integers([0, 0], [width - 70, height - 50])
                span = int(rng.integers(8, 48))
                cv2.rectangle(image, (int(x), int(y)), (int(x + span), int(y + span)),
                              float(rng.integers(25, 75)), 2)
            record(f'{index:02}_{theme}_negative', finish(image, theme, rng), 0, 'none', theme, [])

    manifest = {'dictionary': 'tag36h11', 'seed': seed, 'resolution': resolution,
                'codebook_sha256': CODEBOOK_SHA256, 'scenes': scenes}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--resolution', choices=IMAGE_SIZES, default='720p')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--codebook', type=Path, default=CODEBOOK)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error('seed must be nonnegative')
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error('output directory must be empty')
    cv2.setNumThreads(1)
    generate(args.output, args.resolution, args.seed, grids_from_codebook(args.codebook))


if __name__ == '__main__':
    main()
