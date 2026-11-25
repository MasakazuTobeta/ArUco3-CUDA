# SPDX-License-Identifier: Apache-2.0
"""Generate Gray8 video clips with frame-by-frame AprilTag geometry and visibility."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from border_corpus import render
from dense_corpus import CODEBOOK, grids_from_codebook

PHOTOS_SHA256 = '072afacdf2e05e20c0fa0e2e942e5a9b2b720b14bce18333210c323070bb98f7'
SPLITS = {'dev': (20261121, (0, 2, 6, 8, 10, 12), 0), 'held': (20261337, (1, 3, 7, 9, 11, 13), 17)}
RESOLUTIONS = {720: (1280, 720), 1080: (1920, 1080)}
CLIPS = ('moving-targets', 'moving-camera', 'cut-and-replace', 'negative-pan')
FPS = 30
FRAMES = 18


def digest(value):
    return hashlib.sha256(value).hexdigest()


def background_bank(manifest, entries, width, height):
    pictures = []
    for entry in entries:
        original = (manifest.parent / entry['path']).read_bytes()
        if digest(original) != entry['sha256']:
            raise RuntimeError('background digest mismatch: ' + entry['name'])
        input_image = cv2.imdecode(np.frombuffer(original, np.uint8), cv2.IMREAD_UNCHANGED)
        if input_image.dtype != np.uint8 or input_image.ndim != 2:
            raise RuntimeError('source background is not Gray8')
        scale = max((width + 160) / input_image.shape[1], (height + 128) / input_image.shape[0])
        resized = cv2.resize(
            input_image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        )
        y, x = (resized.shape[0] - height - 128) // 2, (resized.shape[1] - width - 160) // 2
        pictures.append((entry['name'], resized[y : y + height + 128, x : x + width + 160]))
    return pictures


def quad(mid, side, angle, skew):
    upright = np.array([[-0.50, -0.50], [0.50, -0.50], [0.50, 0.50], [-0.50, 0.50]])
    oblique = np.array([[-0.49, -0.53], [0.52, -0.42], [0.43, 0.54], [-0.51, 0.47]])
    theta = math.radians(angle)
    rotate = np.array([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])
    return (oblique if skew else upright) @ rotate * side + mid


def label(track, marker, mid, side, angle, mode, light='normal'):
    corners = quad(mid, side, angle, True)
    return {
        'track': track,
        'id': marker,
        'corners_px': corners.ravel().tolist(),
        'nominal_side_px': side,
        'motion': mode,
        'lighting': light,
    }


def plan(clip, frame, width, height, ids, seed_offset, shift):
    a, b, c, d, e, f, g, h, i, j, k = ids
    scale = height / 720
    start = (int(38 + shift[0]), int(43 + shift[1]))
    ox, oy = start
    labels, cut = [], False
    if clip == 'moving-targets':
        background = 0
        if 2 <= frame <= 15:
            labels.append(
                label(
                    'runner-a',
                    a,
                    (
                        0.28 * width + shift[0] + scale * (9.0 * (frame - 2)),
                        0.55 * height + shift[1] + 12 * math.sin(frame * 0.28),
                    ),
                    33 * scale,
                    11 + 1.7 * frame + seed_offset,
                    'tag',
                )
            )
        sweep = {
            4: 0.8,
            5: 0.36,
            6: 0.06,
            7: -0.26,
            8: -0.66,
            9: -0.96,
            10: -1.18,
            11: -1.22,
            12: -1.07,
            13: -0.72,
            14: -0.44,
            15: -0.10,
            16: 0.24,
            17: 0.68,
        }
        if frame in sweep:
            side = 55 * scale
            labels.append(
                label(
                    'edge-b',
                    b,
                    (width + sweep[frame] * side, 0.32 * height + shift[1]),
                    side,
                    -10 - 0.7 * frame + seed_offset,
                    'tag',
                )
            )
    elif clip == 'moving-camera':
        background = 1
        ox += round(4.1 * frame + 7 * math.sin(frame * 0.31))
        oy += round(2.3 * frame + 4 * math.sin(frame * 0.45))
        labels.extend(
            (
                label(
                    'fixed-c',
                    c,
                    (0.42 * width + 48 - ox, 0.39 * height + 43 - oy),
                    45 * scale,
                    17 + seed_offset,
                    'camera',
                ),
                label(
                    'fixed-d',
                    d,
                    (0.70 * width + 48 - ox, 0.62 * height + 43 - oy),
                    28 * scale,
                    -12 + seed_offset,
                    'camera',
                    'low-gradient',
                ),
            )
        )
    elif clip == 'cut-and-replace':
        background = 2 if frame < 6 else 3 if frame < 17 else 4
        cut = frame in (6, 17)
        same = (0.48 * width + shift[0], 0.55 * height + shift[1])
        if 1 <= frame < 6:
            labels.append(label('before-cut', e, same, 56 * scale, 15 + seed_offset, 'none'))
        if 6 <= frame < 10:
            labels.append(label('after-cut', f, same, 56 * scale, 15 + seed_offset, 'none'))
        if 10 <= frame < 14:
            labels.append(label('replaced-no-cut', g, same, 56 * scale, 15 + seed_offset, 'none'))
        if frame == 17:
            labels.append(
                label(
                    'late-after-cut',
                    h,
                    (0.68 * width, 0.35 * height),
                    50 * scale,
                    28 + seed_offset,
                    'none',
                )
            )
    elif clip == 'negative-pan':
        background = 4 if frame < 10 else 5
        cut = frame == 10
        ox += round(3.2 * frame)
        oy += round(1.9 * frame)
        if 5 <= frame < 9:
            labels.append(
                label(
                    'passing-i',
                    i,
                    (0.32 * width + 13 * (frame - 5), 0.46 * height),
                    49 * scale,
                    8 + seed_offset,
                    'tag+camera',
                )
            )
        if 6 <= frame < 8:
            labels.append(
                label(
                    'passing-j',
                    j,
                    (0.68 * width - 5 * (frame - 6), 0.59 * height),
                    35 * scale,
                    -18 + seed_offset,
                    'tag+camera',
                )
            )
        if 11 <= frame < 14:
            labels.append(
                label(
                    'passing-k',
                    k,
                    (0.56 * width - 11 * (frame - 11), 0.34 * height),
                    43 * scale,
                    5 + seed_offset,
                    'tag+camera',
                )
            )
    return background, (ox, oy), cut, labels


def generate(manifest_path, output, split, size):
    source = manifest_path.read_bytes()
    if digest(source) != PHOTOS_SHA256:
        raise RuntimeError('expected pinned industrial Gray8 photo manifest')
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('output directory must be empty')
    output.mkdir(parents=True, exist_ok=True)
    photo_manifest = json.loads(source)
    seed, indices, offset = SPLITS[split]
    width, height = RESOLUTIONS[size]
    selected = [photo_manifest['scenes'][i] for i in indices]
    photos = background_bank(manifest_path, selected, width, height)
    rng = np.random.default_rng(seed)
    ids = [int(x) for x in rng.choice(np.arange(1 + offset, 180 + offset), 11, replace=False)]
    shift = tuple(float(x) for x in rng.uniform(-9, 9, 2))
    seed_offset = float(rng.uniform(-5, 5))
    code = grids_from_codebook(CODEBOOK)
    code_sha = digest(CODEBOOK.read_bytes())
    scenes = []
    descriptions = {}
    for clip in CLIPS:
        first_rendered = {}
        first_required = {}
        seen = set()
        last = set()
        descriptions[clip] = {'events': []}
        for f in range(FRAMES):
            photo, xy, cut, all_tags = plan(clip, f, width, height, ids, seed_offset, shift)
            ox, oy = xy
            name, bg = photos[photo]
            if not (0 <= ox <= 160 and 0 <= oy <= 128):
                raise RuntimeError('pan exceeds allocated background')
            canvas = bg[oy : oy + height, ox : ox + width].astype(np.float64)
            visible = []
            partial = []
            for m in all_tags:
                box = np.array(m['corners_px']).reshape(4, 2)
                if (
                    box[:, 0].max() < 0
                    or box[:, 1].max() < 0
                    or box[:, 0].min() > width - 1
                    or box[:, 1].min() > height - 1
                ):
                    continue
                inside = (
                    box.min(axis=0).tolist()[0] >= 0
                    and box.min(axis=0).tolist()[1] >= 0
                    and box.max(axis=0).tolist()[0] < width
                    and box.max(axis=0).tolist()[1] < height
                )
                render(canvas, box, code[m['id']], m['lighting'])
                m['first_intersects_frame'] = first_rendered.setdefault(m['id'], f)
                if inside:
                    m['first_visible_frame'] = first_required.setdefault(m['id'], f)
                    m['first_visible_capture_ms'] = m['first_visible_frame'] * 1000 / FPS
                    visible.append(m)
                else:
                    partial.append(m)
            current = {m['id'] for m in visible}
            presence = current | {m['id'] for m in partial}
            events = []
            if f == 0:
                events.append('clip-start')
            if cut:
                events.append('hard-cut')
            if clip == 'cut-and-replace' and f == 10:
                events.append('same-pose-id-replacement')
            if current - last:
                events.append('tag-fully-enters')
            if last - current:
                events.append('tag-leaves-full-view')
            if not presence:
                events.append('negative')
            retired = seen - presence
            seen |= presence
            last = current
            label_name = f'{clip}-{f:03d}'
            imgname = label_name + '.png'
            pixels = np.rint(canvas).clip(0, 255).astype(np.uint8)
            success, encoded = cv2.imencode('.png', pixels, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            if not success:
                raise RuntimeError('PNG encode failed')
            data = encoded.tobytes()
            (output / imgname).write_bytes(data)
            scenes.append(
                {
                    'name': label_name,
                    'path': imgname,
                    'sha256': digest(data),
                    'width_px': width,
                    'height_px': height,
                    'clip': clip,
                    'frame_index': f,
                    'capture_time_ms': f * 1000 / FPS,
                    'background': name,
                    'camera_crop_origin_px': list(xy),
                    'events': events,
                    'must_be_absent_ids': sorted(retired),
                    'markers': visible,
                    'partial_markers': partial,
                }
            )
            if events:
                descriptions[clip]['events'].append(
                    {
                        'frame_index': f,
                        'events': events,
                        'ids': sorted(current),
                        'must_be_absent_ids': sorted(retired),
                    }
                )
    result = {
        'dictionary': 'tag36h11',
        'partition': split,
        'seed': seed,
        'fps': FPS,
        'codebook_cpp_sha256': code_sha,
        'background_manifest_sha256': PHOTOS_SHA256,
        'opencv_version': cv2.__version__,
        'numpy_version': np.__version__,
        'corner_convention': 'clockwise canonical TL,TR,BR,BL in source pixel coordinates',
        'clips': descriptions,
        'scenes': scenes,
    }
    (output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(output.name, len(scenes), sum(len(x['markers']) for x in scenes), flush=True)


def generate_transients(output):
    """Create brief appearances, same-ID copies and low-contrast fade-ins."""
    if output.exists() and any(output.iterdir()):
        raise ValueError('output directory must be empty')
    output.mkdir(parents=True, exist_ok=True)
    code = grids_from_codebook(CODEBOOK)
    y, x = np.indices((480, 640))
    background = np.clip(150 + 8 * np.sin(x / 33) + 6 * np.cos(y / 47), 0, 255)
    # Each episode specifies first and last frame, center, size and opacity increment.
    schedules = {
        'one-frame': [(3, 3, 385, 95, 22, 1.0), (14, 15, 130, 300, 42, 1.0)],
        'fade': [(1, 30, 372, 178, 48, 0.035), (13, 33, 205, 320, 32, 0.035)],
        'duplicate-id': [(1, 14, 90, 145, 48, 1.0), (5, 20, 512, 370, 36, 1.0)],
    }
    scenes = []
    for clip, episodes in schedules.items():
        previously_visible = set()
        for i in range(37):
            image = background.copy()
            markers = []
            for j, (start, end, cx, cy, side, increment) in enumerate(episodes):
                if not start <= i <= end:
                    continue
                velocity = 0.0 if clip == 'fade' else 0.2
                corners = quad((cx + (i - start) * velocity, cy), side, 13.0, True)
                marker_id = 282 if clip == 'duplicate-id' else 123 + j
                original = image.copy()
                render(image, corners, code[marker_id], 'normal')
                if clip == 'fade':
                    opacity = min(1.0, (i - start + 1) * increment)
                    image = original + (image - original) * opacity
                markers.append(
                    dict(
                        track=f'{clip}-{j}',
                        id=marker_id,
                        corners_px=corners.ravel().tolist(),
                        first_visible_frame=start,
                    )
                )
            name = f'{clip}-{i:02d}'
            pixels = np.uint8(image.clip(0, 255).round())
            success, encoded = cv2.imencode('.png', pixels)
            if not success:
                raise RuntimeError('PNG encode failed')
            data = encoded.tobytes()
            (output / (name + '.png')).write_bytes(data)
            visible = {marker['id'] for marker in markers}
            scenes.append(
                dict(
                    name=name,
                    path=name + '.png',
                    sha256=digest(data),
                    markers=markers,
                    clip=clip,
                    frame_index=i,
                    must_be_absent_ids=sorted(previously_visible - visible),
                )
            )
            previously_visible |= visible
    (output / 'manifest.json').write_text(json.dumps(dict(scenes=scenes), indent=2))


def generate_pan_entries(output, split):
    """Move a camera crop over a stationary tag until its complete border enters the view."""
    from PIL import Image, __version__ as pillow_version

    if output.exists() and any(output.iterdir()):
        raise ValueError('output directory must be empty')
    output.mkdir(parents=True, exist_ok=True)
    # width, height, world margin, tag sides, frames in each continuous clip
    width, height, overscan, sides, frame_count = {
        'dev': (480, 288, 200, (32, 72), 22),
        'held': (480, 288, 200, (29, 78), 28),
        'large': (640, 384, 450, (142, 196), 36),
        'photometric': (640, 384, 450, (30, 74), 36),
    }[split]
    seed = 837451 if split == 'dev' else 964209
    grids = grids_from_codebook(CODEBOOK)
    scenes = []
    directions = (('left', 1, 0), ('right', -1, 0), ('top', 0, 1), ('bottom', 0, -1))
    for index in range(16):
        side, dx, dy = directions[index % 4]
        span = sides[index // 4 % 2]
        angle = (0 if index < 8 else 0.29) if split == 'dev' else (-0.21 if index < 8 else 0.44)
        if split == 'dev':
            speed = 6
        elif split == 'held':
            speed = 5 + index % 3
        else:
            speed = 9 + index % 3
        marker_id = index + (20 if split == 'dev' else 140)
        rng = np.random.default_rng(seed + index)
        tiles = rng.uniform(30, 220, (98, 128)).astype(np.uint8)
        view = (width + 2 * overscan, height + 2 * overscan)
        world = np.array(Image.fromarray(tiles).resize(view, Image.Resampling.BICUBIC))
        cx = overscan + (-span / 2 - 3 if dx > 0 else width + span / 2 + 3 if dx < 0 else 0.51 * width)
        cy = overscan + (-span / 2 - 3 if dy > 0 else height + span / 2 + 3 if dy < 0 else 0.43 * height)
        rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        square = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
        corners = square @ rotation * span + (cx, cy)
        y, x = np.indices(world.shape)
        cells = (np.stack((x - cx, y - cy), axis=-1) @ rotation.T) / span + 0.5
        u, v = cells[..., 0], cells[..., 1]
        tag = (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
        surround = (u >= -0.13) & (u < 1.13) & (v >= -0.13) & (v < 1.13)
        white = ~tag | (grids[marker_id][(8 * v).astype(int).clip(0, 7),
                                        (8 * u).astype(int).clip(0, 7)] != 0)
        dark, light = 28, 230
        if split == 'photometric':
            dark, light = (39, 101) if index < 8 else (137, 212)
        elif split == 'held' and index % 3 == 0:
            dark, light = 75, 175
        world[surround] = np.where(white, light, dark)[surround]

        first_visible = None
        clip = f'{side}-{index:02d}'
        for frame_index in range(frame_count):
            x0 = overscan - frame_index * speed * dx
            y0 = overscan - frame_index * speed * dy
            q = corners - (x0, y0)
            inside = np.all((q >= -0.5) & (q <= (width - 0.5, height - 0.5)))
            intersects = np.all(q.max(axis=0) >= -0.5) and np.all(q.min(axis=0) <= (width - 0.5, height - 0.5))
            name = f'clip{index:02d}-{side}-{frame_index:02d}'
            if inside and first_visible is None:
                first_visible = frame_index
            record = dict(track='entering', id=marker_id, corners_px=q.ravel().tolist(),
                          first_visible_frame=first_visible)
            data = f'P5\n{width} {height}\n255\n'.encode() + world[y0:y0 + height, x0:x0 + width].tobytes()
            (output / (name + '.pgm')).write_bytes(data)
            scenes.append(dict(name=name, path=name + '.pgm', sha256=digest(data), clip=clip,
                               frame_index=frame_index, markers=[record] if inside else [],
                               partial_markers=[record] if intersects and not inside else []))
    manifest = dict(dictionary='tag36h11', split=split, seed=seed, numpy_version=np.__version__,
                    pillow_version=pillow_version, codebook_cpp_sha256=digest(CODEBOOK.read_bytes()),
                    corner_convention='clockwise canonical TL,TR,BR,BL in source pixel coordinates',
                    scenes=scenes)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    motion = modes.add_parser('motion', help='render moving tags on the pinned industrial photos')
    motion.add_argument('photos', type=Path)
    motion.add_argument('output', type=Path)
    motion.add_argument('--split', choices=SPLITS, required=True)
    motion.add_argument('--height', type=int, choices=RESOLUTIONS, required=True)
    transient = modes.add_parser('transients', help='render short appearances and fading tags')
    transient.add_argument('output', type=Path)
    pan = modes.add_parser('pan-entries', help='pan a camera over stationary tags entering from each boundary')
    pan.add_argument('output', type=Path)
    pan.add_argument('--split', choices=('dev', 'held', 'large', 'photometric'), required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.mode == 'motion':
        generate(args.photos, args.output, args.split, args.height)
    elif args.mode == 'transients':
        generate_transients(args.output)
    else:
        generate_pan_entries(args.output, args.split)
