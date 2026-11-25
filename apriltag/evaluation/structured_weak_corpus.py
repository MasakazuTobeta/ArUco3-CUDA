# SPDX-License-Identifier: Apache-2.0
"""Render paired weak/strong 36h11 prints on photographs and grooved substrates."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from assembly_corpus import PHOTOS, VALIDATION_PHOTOS, source_image
from dense_corpus import CODEBOOK, grids_from_codebook


WIDTH, HEIGHT = 1280, 720
PHOTO_IDS = ('LRC-2013-00757', 'AAL-4479', 'GRC-1970-C-03022',
             'GRC-1987-C-00568', 'MSFC-1701231', 'PIA11188')
CLIPS = ('left-photo_weak-smooth', 'right-photo_weak-smooth', 'left-photo_weak-on-card',
         'right-photo_weak-on-card', 'fine-ridges_weak-on-substrate', 'fine-photo_weak-on-substrate')
HELD_PHOTOS = ('AFRC2019-0134-03', 'GRC-2011-C-00334', 'jsc2022e043550',
               'PIA12463', 'GRC-1949-C-24341', 'A-42536-003')
HELD_BLUR = (.45, .85, 1.10, .55, .95, .75)
SEEDS = {'photographic': 68104723, 'lit-grooves': 28970351, 'held': 1946937087,
         'defocus-a': 173987042, 'defocus-b': 1776289013}
NOISE = {'native': 0, 'noise065': 0.65, 'noise140': 1.4}
DEFOCUS = {
    'defocus-a': ((1.65, .70, 1.14, 1.52, .98, 1.38), {'low': .32, 'moderate': .95}),
    'defocus-b': ((1.47, 1.62, .82, 1.18, 1.02, 1.73), {'low': .42, 'moderate': 1.14}),
}
MANIFEST_HASHES = {
    'defocus-a': ('0ae2b81df5f6e924cfe85880865ba0576ce4c08b35fb5f04b42c0dbb1864c3fb',
                  'a2ed1a55f3a11fcd1e1204bcebb3a8d9caa1ff3731464dfad9d83f25cac59410'),
    'defocus-b': ('5bc8cb9fcf2e96212cbba77e470163c7999264c9f76fd1d252a01654c2b2847c',
                  '6bc38f2c91b0b7116d9be7e2a43105e14e22a95e581748035e9485a75debe657'),
    'photographic': (
        'c496675b022bc191da07f3bf1de355a9d37352247e5d3ef1f55f6868f262afb8',
        '45b7a0a20143631031bc4a1cdbb927bf11b6ca7c194447a2cf5942df66690059',
        '49c63eda306c9d8a12671f963e3caf7323db24e3b97041b085a91604df8a2a38'),
    'lit-grooves': (
        'c819e029ca6290c0316f52ca2757998598c397e88fe242dc4affac2eabc44bd4',
        '50affce63832ea37b5756329f52e74730c7a93795b85d71140e14c30ee6a3427',
        '42402ac5164a4d190056f4e929b65b8fb8f504a41eb3c69df5f1f022bc8adbf4'),
    'held': (
        '4a91dd32639599426f9d91ab87ff4c7d8a1216e598f3ac499c114fc7ed4d5f2c',
        '6898865bfdad068be3269e5a02e59ba3845a204a67a4652ff16067b1ffe4f2e2',
        'd725614a61bf8ece19d8e70a8fc6437833ae1589ccb383a4e319f27641624e04'),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def substrate(photo, clip, variant, right, rng):
    py, px = rng.uniform(.02, .2), rng.uniform(.04, .24)
    height, width = photo.shape
    crop = photo[int(height * py):int(height * (py + .61)),
                 int(width * px):int(width * (px + .56))]
    texture = cv2.resize(crop, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA).astype('f4')
    texture -= texture.mean()
    texture = np.clip(texture / max(np.std(texture), 1e-5), -2.5, 2.5)
    y, x = np.mgrid[:HEIGHT, :WIDTH]
    wave = np.sin((x + 2 * y) / (350 + 29 * clip))
    if variant == 'lit-grooves':
        base = 104 + .058 * x + .013 * y + 5 * wave
        field = base + .45 * texture
        for row in range(9):
            for col in range(6):
                left = (42 if right else 704) + 84 * col + int(rng.integers(0, 23))
                top = 57 + 69 * row + int(rng.integers(-5, 6))
                length, thickness = int(rng.integers(34, 72)), 1 + clip % 4
                if clip % 3 == 2:
                    field[top-14:top-14+length, left+12:left+12+thickness] -= 8 + 2 * clip
                else:
                    field[top:top+thickness, left:left+length] -= 8 + 2 * clip
        return base, field

    base = 153 + 11 * wave + .009 * y - .008 * x + 3 * np.cos(y / 139 + x / 421)
    if clip < 4:
        split = 620 + 27 * np.sin(y / 173 + .52 * clip)
        textured = x < split if clip % 2 == 0 else x > split
        field = base + textured * (17 * texture + 6 * np.cos(x / 8 + y / 31))
    elif clip == 4:
        field = base + (1.5 * np.cos((x + .7 * y) / 6) + 1.7 * np.cos((2 * x - y) / 23) + 1.3 * texture)
    else:
        field = base + (2.4 * texture + 1.6 * np.sin((x - y) / 37) + 1.3 * np.cos(y / 12))
    return base, field


def pose(center, angle, side, clip, frame, track):
    cx = center[0] + 10 * np.cos(.37 * frame + .71 * clip)
    cy = center[1] + 8 * np.sin(.39 * frame + track)
    angle += .014 * frame
    a = .5 * side * np.array([np.cos(angle), np.sin(angle)])
    b = .42 * side * np.array([-np.sin(angle), np.cos(angle)])
    return np.float32([np.array([cx, cy]) + delta
                       for delta in (-a-b, a-.89*b, .86*a+b, -a+1.07*b)])


def print_tag(image, base, cells, quad, contrast, paper):
    center = quad.mean(axis=0)
    halo = center + 1.24 * (quad - center)
    low = np.floor(halo.min(axis=0) - 3).astype(int)
    high = np.ceil(halo.max(axis=0) + 4).astype(int)
    size = tuple(map(int, high - low))
    crop = np.s_[low[1]:high[1], low[0]:high[0]]
    # Marker cells are centred on integers. Integrate five subpixels on each axis.
    uv = np.float32([[-.5, -.5], [7.5, -.5], [7.5, 7.5], [-.5, 7.5]])

    def coverage(bitmap, corners):
        dest = np.float32((corners - low) * 5 + 2)
        transform = cv2.getPerspectiveTransform(uv, dest)
        samples = cv2.warpPerspective(bitmap, transform, (size[0] * 5, size[1] * 5),
                                      flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        return cv2.resize(samples.astype('f4'), size, interpolation=cv2.INTER_AREA)

    if paper:
        alpha = coverage(np.ones((8, 8), 'f4'), halo)
        image[crop] = image[crop] * (1 - alpha) + alpha * (base[crop] + 2)
    image[crop] -= contrast * coverage(1 - cells // 255, quad)


def generate(cache, output, variant, offline):
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('output directory must be empty')
    blur, read_noise = DEFOCUS.get(variant, (HELD_BLUR, NOISE))
    for key in read_noise:
        (output / key).mkdir()
    cv2.setNumThreads(1)
    seed = SEEDS[variant]
    rng = np.random.default_rng(seed)
    grids = grids_from_codebook(CODEBOOK)
    held = variant == 'held' or variant in DEFOCUS
    photos = {identifier: (rendition, sha) for identifier, rendition, sha in
              (VALIDATION_PHOTOS if held else PHOTOS)}
    sources = {}
    scenes = {key: [] for key in read_noise}
    for clip, identifier in enumerate(HELD_PHOTOS if held else PHOTO_IDS):
        rendition, sha = photos[identifier]
        photo = source_image(cache, identifier, rendition, sha, offline)
        sources[f'{identifier}.jpg'] = sha
        right = clip in (0, 3, 5)
        scene_type = ('lit-grooves' if clip % 2 else 'photographic') if held else variant
        base, field = substrate(photo, clip, scene_type, right, rng)
        centers = [(874 if right else 332, 267 if clip % 2 == 0 else 441),
                   (261 if right else 937, 475 if clip % 2 == 0 else 238)]
        if held:
            centers = [(cx + rng.uniform(-72, 64), cy + rng.uniform(-66, 53)) for cx, cy in centers]
            magnifications = rng.uniform(.75, 1.20, 2)
            perspective = rng.uniform(.04, .13, 2)
        angles = rng.uniform(-2.8, 2.8, 2)
        ids = rng.choice(len(grids), 4, replace=False)
        for frame in range(10):
            image = field.copy()
            markers = []
            for track, active in enumerate(((1, 2, 3, 5, 6, 8), (2, 3, 4, 5, 7, 8))):
                if frame not in active:
                    continue
                tag = int(ids[track + 2 * (frame >= 5)])
                contrast = (11, 15, 21, 29, 18, 24)[clip] if track == 0 else 102
                side = (91, 113, 136, 104, 121, 87)[clip] if track == 0 else 101
                quad = pose(centers[track], angles[track], side * magnifications[track] if held else side,
                            clip, frame, track)
                if held:
                    quad[1] += perspective[track] * (quad[2] - quad[1])
                paper = clip < 4 and ((track == 0 and clip >= 2) or (track == 1 and clip < 2))
                print_tag(image, base, grids[tag], quad, contrast, paper)
                markers.append(dict(id=tag, track='weak' if track == 0 else 'strong',
                                    contrast=contrast, paper=paper,
                                    corners_px=quad.ravel().astype(float).tolist()))
            image = cv2.GaussianBlur(image, (0, 0), blur[clip] if held else .62)
            noise_rng = np.random.default_rng(np.random.SeedSequence([seed, clip, frame]))
            noise = noise_rng.normal(0, 1, (HEIGHT, WIDTH))
            name = f's{clip}-{frame}'
            absent = sorted(set(map(int, ids)) - {m['id'] for m in markers})
            for key, sigma in read_noise.items():
                pixels = np.rint(image + sigma * noise).clip(0, 255).astype('u1')
                data = f'P5\n{WIDTH} {HEIGHT}\n255\n'.encode() + pixels.tobytes()
                filename = f'{name}.pgm'
                (output / key / filename).write_bytes(data)
                scenes[key].append(dict(name=name, path=filename, sha256=digest(data),
                                        markers=markers, clip=(f'{scene_type}-{clip}' if held else CLIPS[clip]),
                                        frame_index=frame,
                                        must_be_absent_ids=absent))
    for key, sigma in read_noise.items():
        manifest = dict(dictionary='tag36h11', width=WIDTH, height=HEIGHT, seed=seed,
                        optical_sigma_px=blur if held else .62, read_sigma=sigma, sources=sources, scenes=scenes[key])
        (output / key / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def verify(output, variant):
    _, read_noise = DEFOCUS.get(variant, (HELD_BLUR, NOISE))
    for key, expected in zip(read_noise, MANIFEST_HASHES[variant]):
        directory = output / key
        data = (directory / 'manifest.json').read_bytes()
        if digest(data) != expected:
            raise ValueError(f'{directory}: manifest digest differs from the frozen release')
        for scene in json.loads(data)['scenes']:
            path = directory / scene['path']
            if digest(path.read_bytes()) != scene['sha256']:
                raise ValueError(f'{path}: image digest mismatch')
        print(f'{directory}: {expected}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path, help='shared NASA JPEG cache')
    parser.add_argument('output', type=Path)
    parser.add_argument('--variant', required=True, choices=SEEDS)
    parser.add_argument('--offline', action='store_true', help='use cached, verified JPEGs')
    parser.add_argument('--verify-only', action='store_true', help='verify the saved manifest and all images')
    args = parser.parse_args()
    if not args.verify_only:
        generate(args.cache, args.output, args.variant, args.offline)
    verify(args.output, args.variant)
