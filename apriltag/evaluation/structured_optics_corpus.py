# SPDX-License-Identifier: Apache-2.0
"""Apply a reproducible second imaging pass to the original structured weak-print fixtures."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from dense_corpus import CODEBOOK, grids_from_codebook
from structured_weak_corpus import MANIFEST_HASHES, digest, print_tag


SEED = 932176541
READ_NOISE_SEED = 91867019
READ_NOISE_PARTITIONS = (
    ('defocus-a', 5.5, '4ff340630a397f327862b34fe93932e239b7be760823a058914c563689a82eb8'),
    ('defocus-b', 10.0, '7e0767bb749bd0072cf82c79c25f80313c8a8b8462502bc8bf6a4694cdd84c18'),
)
PARTITIONS = (
    ('photographic', 'ph0', (.83, .95), (.84, 1.24), .96, .18,
     '2f9f60ed421eb6066fba07367e74737244bba0fd254cac3428b80ddbdb47f57e'),
    ('photographic', 'ph1', (.67, .84), (1.53, .98), .84, .31,
     'ad63ebbf53b8250b3524dcea590d3b9ee4d388571f921f3471502b1a03942d84'),
    ('lit-grooves', 'gr0', (.83, .95), (.84, 1.24), .96, .18,
     'ba71ee02fadde9ac2589f33c671e763710a7fac618b89c1bb29e74e5b83763f6'),
    ('lit-grooves', 'gr1', (.67, .84), (1.53, .98), .84, .31,
     'ebdd56e4476a735d17b3a936c45b81336c5888f592f5b1a54f4d83ba4c7ce440'),
)


def checked_manifest(directory, expected):
    data = (directory / 'manifest.json').read_bytes()
    if digest(data) != expected:
        raise ValueError(f'{directory}: manifest digest mismatch')
    manifest = json.loads(data)
    for scene in manifest['scenes']:
        path = directory / scene['path']
        if digest(path.read_bytes()) != scene['sha256']:
            raise ValueError(f'{path}: image digest mismatch')
    return manifest


def generate(inputs, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError('output directory must be empty')
    rng = np.random.default_rng(SEED)
    cv2.setNumThreads(1)
    for variant, name, scale_range, blur, gain, noise, expected in PARTITIONS:
        source = inputs / variant / 'native'
        source_sha = MANIFEST_HASHES[variant][0]
        manifest = checked_manifest(source, source_sha)
        width, height = manifest['width'], manifest['height']
        dest = output / name
        dest.mkdir(parents=True)
        transforms = {}
        for scene in manifest['scenes']:
            if scene['frame_index'] == 0:
                scale = rng.uniform(*scale_range)
                degrees = rng.uniform(-7., 7.)
                offset = rng.uniform(-9., 9., 2)
                transform = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, scale)
                transform[:, 2] += offset
                transforms[scene['clip']] = transform.tolist()

            image = cv2.imread(str(source / scene['path']), cv2.IMREAD_GRAYSCALE)
            image = cv2.warpAffine(image.astype('f4'), transform, (width, height),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            image = cv2.GaussianBlur(image, (0, 0), blur[0], sigmaY=blur[1])
            image = 145. + (image - 145.) * gain
            image = np.rint(image + rng.normal(0., noise, image.shape)).clip(0, 255).astype('u1')
            for marker in scene['markers']:
                quad = np.asarray(marker['corners_px']).reshape(4, 2)
                marker['corners_px'] = (quad @ transform[:, :2].T + transform[:, 2]).ravel().tolist()
            path = dest / scene['path']
            if not cv2.imwrite(str(path), image):
                raise OSError(f'failed to write {path}')
            scene['sha256'] = digest(path.read_bytes())
        (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        checked_manifest(dest, expected)
        camera = dict(derivative_seed=SEED, source_manifest_sha256=source_sha,
                      output_manifest_sha256=expected, interpolation='linear',
                      boundary='reflect_101', extra_gaussian_sigma_xy_px=blur,
                      intensity_pivot=145., intensity_gain=gain, extra_read_sigma=noise,
                      clip_affines_xy=transforms)
        (dest / 'camera-model.json').write_text(json.dumps(camera, indent=2) + '\n')


def generate_read_noise(inputs, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError('output directory must be empty')
    cv2.setNumThreads(1)
    rng = np.random.default_rng(READ_NOISE_SEED)
    for variant, noise, expected in READ_NOISE_PARTITIONS:
        source = inputs / variant / 'low'
        source_sha = MANIFEST_HASHES[variant][0]
        manifest = checked_manifest(source, source_sha)
        manifest['source_manifest_sha256'] = source_sha
        manifest['additional_read_noise_sigma_gray8'] = noise
        dest = output / str(noise)
        dest.mkdir(parents=True)
        for scene in manifest['scenes']:
            image = cv2.imread(str(source / scene['path']), cv2.IMREAD_GRAYSCALE)
            pixels = np.rint(image.astype('f4') + rng.normal(0, noise, image.shape))
            path = dest / scene['path']
            if not cv2.imwrite(str(path), pixels.clip(0, 255).astype('u1')):
                raise OSError(f'failed to write {path}')
            scene['sha256'] = digest(path.read_bytes())
        (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        checked_manifest(dest, expected)


# These partitions keep their own noise stream, so each can be reproduced alone.
NOISE_HOLDOUTS = (
    ('held-sigma10', 'held', 'native', 10.0, 78829053, '7430b150e915486038f924f3177bd32c35744999589efbd48991de2abf856adf'),
    ('held-sigma8', 'held', 'native', 8.0, 8584627, '4cc03cdd89c3ba36fb6aa5f13ddd40e14ba2597a52c06ee0f1f1db517478e157'),
    ('grooves-sigma12', 'lit-grooves', 'native', 12.0, 6600181, 'ff428517f32bb31f96921afe907d013b4b0a7fb4fd7494c9fa435203e3ca7cb6'),
    ('edge-sigma10', 'defocus-a', 'low', 10.0, 21759411, 'd976b0b8447b6154edab26888b00f3a239085d12a64a6ae9272dd8448347eaf4'),
)


def add_edge_print(image, scene, grids):
    frame = scene['frame_index']
    clip = int(scene['name'].split('-')[0][1:])
    tag = (31, 80, 95, 441, 416, 360)[clip]
    if frame in (0, 9):
        scene['must_be_absent_ids'].append(tag)
        return image
    side = (22, 30, 42, 54, 36, 64)[clip]
    height, width = image.shape
    x = side * (-1, -.1, .3, .65, .8, 1.1, 1.3, 1.5, .6, -1)[frame]
    if clip % 2:
        x = width - 1 - x
    center = np.array([x, 75 + 73*clip + 7*frame])
    angle = .12*(clip-2) + .027*frame
    a = .5*side*np.array([np.cos(angle), np.sin(angle)])
    b = .47*side*np.array([-np.sin(angle), np.cos(angle)])
    quad = np.float32([center + v for v in (-a-b, a-b, a+b, -a+b)])
    # Render the full print on padded paper before clipping its optical footprint.
    paper = np.pad(image, 100, mode='edge')
    printed = paper.copy()
    print_tag(printed, paper, grids[tag], quad + 100, 78 + 4*clip, False)
    pixels = paper + cv2.GaussianBlur(printed - paper, (0, 0), .85)
    key = 'markers' if np.all((quad[:, 0] >= -.5) & (quad[:, 0] < width-.5)) else 'partial_markers'
    scene.setdefault(key, []).append(dict(id=tag, track='edge', corners_px=quad.ravel().tolist()))
    return pixels[100:-100, 100:-100]


def generate_noise_holdouts(inputs, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError('output directory must be empty')
    cv2.setNumThreads(1)
    grids = grids_from_codebook(CODEBOOK)
    for name, variant, subset, sigma, seed, expected in NOISE_HOLDOUTS:
        source = inputs / variant / subset
        source_sha = MANIFEST_HASHES[variant][0]
        manifest = checked_manifest(source, source_sha)
        if name == 'edge-sigma10':
            manifest.update(edge_noise_seed=seed, source_manifest_sha256=source_sha)
        else:
            manifest.update(source_manifest_sha256=source_sha, added_noise_sigma=sigma,
                            added_noise_seed=seed)
        rng = np.random.default_rng(seed)
        dest = output / name
        dest.mkdir(parents=True)
        for scene in manifest['scenes']:
            image = cv2.imread(str(source / scene['path']), cv2.IMREAD_GRAYSCALE).astype('f4')
            if name == 'edge-sigma10':
                image = add_edge_print(image, scene, grids)
            pixels = np.rint(image + rng.normal(0, sigma, image.shape)).clip(0, 255).astype('u1')
            path = dest / scene['path']
            if not cv2.imwrite(str(path), pixels):
                raise OSError(f'failed to write {path}')
            scene['sha256'] = digest(path.read_bytes())
        (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        checked_manifest(dest, expected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path, help='root of the photographic and lit-grooves outputs')
    parser.add_argument('output', type=Path)
    parser.add_argument('--verify-only', action='store_true', help='check saved manifests and pixels')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--noise-holdouts', action='store_true', help='build independent read-noise and edge-entry controls')
    mode.add_argument('--read-noise', action='store_true', help='add read noise to defocus-a and defocus-b')
    args = parser.parse_args()
    if args.verify_only:
        if args.noise_holdouts:
            for name, *_, sha in NOISE_HOLDOUTS:
                checked_manifest(args.output / name, sha)
        elif args.read_noise:
            for _, noise, sha in READ_NOISE_PARTITIONS:
                checked_manifest(args.output / str(noise), sha)
        else:
            for _, name, *_, sha in PARTITIONS:
                checked_manifest(args.output / name, sha)
    elif args.noise_holdouts:
        generate_noise_holdouts(args.inputs, args.output)
    elif args.read_noise:
        generate_read_noise(args.inputs, args.output)
    else:
        generate(args.inputs, args.output)
