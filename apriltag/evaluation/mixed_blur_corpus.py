# SPDX-License-Identifier: Apache-2.0
"""Render eight ordered clips of moving tag36h11 prints under blur and sensor noise."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from border_corpus import render
from dense_corpus import CODEBOOK, grids_from_codebook


WIDTH, HEIGHT = 1280, 720
CENTERS = ((300, 320), (895, 242), (920, 540), (606, 540))
CAPTURES = {
    'original': (74208571, (112, 180, 266, 315, 88, 142, 210, 252),
                 (0.40, 0.85, 1.32, 1.75),
                 '0c072852fdee4850bcdedc8eab6633075265de6891c654f09a35d3d5ce711945'),
    'alternate': (62380714, (88, 211, 247, 304, 110, 158, 196, 232),
                  (0.43, 0.79, 1.54, 1.63),
                  '3f20fa7e25ab02f61a8dd5ddb1c2d95eb97520447a9483e91593b739ba83c9ee'),
}
NOISE_SIGMAS = (1.8, 4.5, 7.5, 12.0)


def background(clip, x, y):
    pattern = clip % 4
    if pattern == 0:
        image = 228 + 5 * np.sin(x / 110) + 3 * np.cos(y / 92)
    elif pattern == 1:
        image = 146 + 20 * np.cos(x / 250) + 25 * np.sin(y / 130)
    elif pattern == 2:
        image = 205 + 24 * np.sin((x + y) / 135)
    else:
        image = 142 + 24 * np.sin(x / 75) * np.cos(y / 52)
    image = image.astype(float)
    if clip % 3 == 2:
        cv2.rectangle(image, (685, 60), (1225, 660), 18, -1)
        cv2.rectangle(image, (704, 78), (1205, 638), 225, -1)
    return image


def corners(clip, frame, track, initial_angle, primary_side):
    angle = initial_angle + (frame * .037 if track % 2 else -frame * .029) + track * 1.1
    rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    shape = np.array([[-.50, -.49], [.49, -.49 + clip * .017],
                      [.43 + .04 * np.sin(frame / 6), .50], [-.48, .39 + .04 * np.cos(frame / 3)]])
    side = primary_side if track == 0 else (92, 55, 142)[track - 1]
    movement = np.array([3.5 * frame, 8 * np.sin(frame / 2)])
    quad = side * shape @ rotation + CENTERS[track] + movement
    if clip in (3, 7) and track == 0 and frame in (2, 6):
        quad[:, 0] -= quad[:, 0].min() - 2.1
    return quad


def generate(output, capture):
    seed, primary_sides, blur_sigmas, manifest_hash = CAPTURES[capture]
    output.mkdir(parents=True, exist_ok=True)
    grids = grids_from_codebook(CODEBOOK)
    rng = np.random.default_rng(seed)
    y, x = np.ogrid[:HEIGHT, :WIDTH]
    scenes = []
    for clip, primary_side in enumerate(primary_sides):
        angle = rng.uniform(-np.pi, np.pi)
        ids = rng.choice(len(grids), 4, replace=False)
        for frame in range(10):
            image = background(clip, x, y)
            markers = []
            if frame not in (0, 9):
                count = 1 if frame == 1 else 1 + clip % 4
                for track in range(count):
                    quad = corners(clip, frame, track, angle, primary_side)
                    lighting = 'low-gradient' if clip % 3 == 1 and track % 2 == 0 else 'normal'
                    marker_id = int(ids[track])
                    render(image, quad, grids[marker_id], lighting)
                    markers.append(dict(id=marker_id, corners_px=quad.ravel().tolist(),
                                        track=str(track), first_visible_frame=1 if track == 0 else 2))
            if clip % 2:
                cv2.rectangle(image, (7, 9), (140, 65), 45, -1)
                cv2.rectangle(image, (19, 18), (128, 54), 224, -1)
            blurred = cv2.GaussianBlur(image, (0, 0), blur_sigmas[clip % 4])
            noise = rng.normal(0, NOISE_SIGMAS[(clip + frame // 3) % 4], (HEIGHT, WIDTH))
            gray = np.clip(np.rint(blurred + noise), 0, 255).astype(np.uint8)
            name = f'mixed-{clip}-{frame}'
            path = output / f'{name}.pgm'
            if not cv2.imwrite(str(path), gray):
                raise OSError(f'could not write {path}')
            present = {marker['id'] for marker in markers}
            scenes.append(dict(name=name, path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               markers=markers, clip=f'mixed-{clip}', frame_index=frame,
                               must_be_absent_ids=[int(marker_id) for marker_id in ids if marker_id not in present]))
    data = (json.dumps(dict(seed=seed, dictionary='tag36h11', scenes=scenes), indent=2) + '\n').encode()
    digest = hashlib.sha256(data).hexdigest()
    if digest != manifest_hash:
        raise ValueError(f'manifest SHA-256 {digest} differs from the frozen sequence {manifest_hash}')
    (output / 'manifest.json').write_bytes(data)
    print(f'{output / "manifest.json"}  {digest}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--capture', choices=CAPTURES, default='original')
    args = parser.parse_args()
    generate(args.output, args.capture)
