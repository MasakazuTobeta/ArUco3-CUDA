# SPDX-License-Identifier: Apache-2.0
"""Project moving small and large tags onto the pinned industrial photographs."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from border_corpus import dictionary_grids, render


DEFAULT_SEED = 319783
PHOTO_MANIFEST = '072afacdf2e05e20c0fa0e2e942e5a9b2b720b14bce18333210c323070bb98f7'
SIZES = {'720p': (1280, 720), '1080p': (1920, 1080)}
SMALL_SIDES = (16, 20, 24, 28, 32, 36, 40, 48)
LARGE_SIDES = (76, 92, 108, 128, 148, 172, 192, 212)


def moving_quad(side, center, angle, time, direction):
    # The mild keystone distortion and rotation keep the projected cell grid planar.
    unit = np.array([[-.52, -.44], [.46, -.53], [.53, .47], [-.43, .54]])
    radians = np.deg2rad(angle + 0.7 * time)
    rotation = np.array([[np.cos(radians), np.sin(radians)], [-np.sin(radians), np.cos(radians)]])
    return side * unit @ rotation + center + time * direction


def render_clip(image, photo, photo_index, width, height, grids, output, seed, sensor_effects):
    rng = np.random.default_rng(seed + photo_index)
    ids = rng.choice(len(grids), 2, replace=False)
    left = rng.uniform(.19, .37) * width
    right = rng.uniform(.63, .81) * width
    sides = (SMALL_SIDES[(photo_index // 2 + 3 * (photo_index % 2)) % 8],
             LARGE_SIDES[(3 * (photo_index // 2) + photo_index % 2) % 8])
    centers = np.array([[left, rng.uniform(.24, .76) * height],
                        [right, rng.uniform(.24, .76) * height]])
    if photo_index % 4 < 2:
        centers[:, 0] = width - centers[:, 0]
    angles = rng.uniform(-33, 33, 2) + 90 * rng.integers(0, 4, 2)
    velocities = rng.uniform(-2, 2, (2, 2))
    # Cover the target with background and let the camera pan a few pixels per frame.
    scale = max((width + 32) / image.shape[1], (height + 24) / image.shape[0])
    background = cv2.resize(image, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    x = (background.shape[1] - width) // 2 - 8
    y = (background.shape[0] - height) // 2 - 4
    scenes = []
    for t in range(8):
        frame = background[y+t:y+t+height, x+2*t:x+2*t+width].astype(np.float64)
        markers = []
        for i in range(2):
            if t == 0 or (t == 1 and i == 0) or (t == 2 and i == 1):
                continue
            corners = moving_quad(sides[i], centers[i], angles[i], t, velocities[i])
            light = 'normal' if (photo_index // 2 + i) % 3 else 'low-gradient'
            render(frame, corners, grids[ids[i]], light)
            markers.append({'id': int(ids[i]), 'corners_px': corners.ravel().tolist(),
                            'nominal_side_px': sides[i], 'lighting': light})
        if sensor_effects:
            # Filter the whole image so the tag and the surrounding background share the optics.
            sigma = (.55, .70, .85, 1.0)[(photo_index // 2) % 4]
            gain = (.80, 1.12, .92, 1.03)[(photo_index // 2) % 4]
            frame = cv2.GaussianBlur(frame, (7, 7), sigma) * gain + (photo_index % 3 - 1) * 8
            frame += rng.normal(0, 2.0, frame.shape)
        name = f'{photo["name"]}_frame{t}'
        path = output / (name + '.pgm')
        data = f'P5\n{width} {height}\n255\n'.encode() + np.rint(frame).clip(0, 255).astype(np.uint8).tobytes()
        path.write_bytes(data)
        scenes.append({'name': name, 'path': path.name, 'sha256': hashlib.sha256(data).hexdigest(),
                       'width_px': width, 'height_px': height, 'background': photo['name'],
                       'clip_frame': t, 'markers': markers})
    return scenes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('photos', type=Path, help='pinned april-photos/manifest.json')
    parser.add_argument('output', type=Path)
    parser.add_argument('--resolution', choices=SIZES, required=True)
    parser.add_argument('--partition', choices=('development', 'validation'), required=True)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--sensor-effects', action='store_true',
                        help='apply blur, exposure changes and two-gray-level standard deviation noise')
    args = parser.parse_args()
    if args.seed < 0:
        parser.error('seed must be nonnegative')
    source = args.photos.read_bytes()
    if hashlib.sha256(source).hexdigest() != PHOTO_MANIFEST:
        parser.error('expected the original 16-photo Gray8 manifest')
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error('output directory must be empty')
    cv2.setNumThreads(1)
    grids = dictionary_grids()
    scenes = []
    for index, photo in enumerate(json.loads(source)['scenes']):
        if index % 2 != int(args.partition == 'validation'):
            continue
        path = args.photos.parent / photo['path']
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != photo['sha256']:
            parser.error(f'input hash mismatch: {path}')
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None or image.shape != (photo['height_px'], photo['width_px']) or image.dtype != np.uint8:
            parser.error(f'expected Gray8 image at manifest dimensions: {path}')
        scenes.extend(render_clip(image, photo, index, *SIZES[args.resolution], grids, args.output,
                                  args.seed, args.sensor_effects))
    manifest = {'dictionary': 'tag36h11', 'partition': args.partition,
                'background_manifest_sha256': PHOTO_MANIFEST, 'scenes': scenes}
    if args.seed != DEFAULT_SEED or args.sensor_effects:
        manifest.update(seed=args.seed, sensor_effects=args.sensor_effects)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
