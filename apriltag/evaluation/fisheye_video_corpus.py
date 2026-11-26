# SPDX-License-Identifier: Apache-2.0
"""Replay every left-camera frame in selected TUM VI AprilGrid calibration recordings."""
import argparse
import csv
import hashlib
import io
from importlib.metadata import version
import json
from pathlib import Path
import tarfile

import cv2
import numpy as np
import yaml

from dense_corpus import CODEBOOK, grids_from_codebook
from video_corpus import APRILTAG_REVISION, CORNER_ORDER, code_errors, read_probe


# The source publishes MD5 sidecars. These SHA-256 values pin the same EuRoC archives.
ARCHIVES = {
    'dataset-calib-cam1_512_16': '4c30f6a5670154197adf660c1110535a8d49b7f33e04ea8857c4d62af1ba78b1',
    'dataset-calib-cam1_1024_16': '1967189a6942ed12eb42d3e89ef0cb1c48b6749560b6bab63992d9d58c64d3ec',
    'dataset-calib-imu1_512_16': '16292cd91cc2bf8df8494ed7eebe818a8e0718e75827ed16d8c7001466b03bb5',
}
CAMCHAINS = {
    512: '3a0c0285f4d0a3a1678053b2e5912e3e250e78ac09bd009f6d7d8bba669d5703',
    1024: '609dc53e0642695dc54036f65307d3228d7b947bfc6e7452f0b43cf500afe73d',
}
PAYLOAD_CENTERS = np.array([((x + .5) / 10, (y + .5) / 10) for y in range(2, 8)
                            for x in range(2, 8)], dtype=np.float32).reshape(1, 36, 2)
HOMEPAGE = 'https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset'


def digest_file(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def archive_bytes(archive, name, maximum):
    member = archive.getmember(name)
    if not member.isfile() or not 0 < member.size <= maximum:
        raise ValueError(f'unexpected regular file or size: {name}')
    return archive.extractfile(member).read()


def write_manifest(output, provenance, scenes):
    data = (json.dumps({**provenance, 'scenes': scenes}, allow_nan=False, indent=2) + '\n').encode()
    (output / 'manifest.json').write_bytes(data)
    print(f"{output / 'manifest.json'} {hashlib.sha256(data).hexdigest()} {len(scenes)} frames")


def read_frames(archive, prefix, size, output):
    csv_data = archive_bytes(archive, f'{prefix}/data.csv', 2_000_000).decode('ascii')
    rows = csv.reader(io.StringIO(csv_data))
    if next(rows, None) != ['#timestamp [ns]', 'filename']:
        raise ValueError('unexpected camera index header')
    scenes = []
    for ordinal, (stamp, name) in enumerate(rows):
        if not stamp.isdecimal() or name != f'{stamp}.png':
            raise ValueError(f'invalid source frame name: {name}')
        timestamp = int(stamp)
        if scenes and timestamp <= scenes[-1]['timestamp_ns']:
            raise ValueError('camera timestamps must strictly increase')
        data = archive_bytes(archive, f'{prefix}/data/{name}', 4 * size * size)
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None or image.shape != (size, size) or image.dtype != np.uint16:
            raise ValueError(f'expected a single-channel 16-bit PNG: {name}')
        # Preserve the full linear response with a fixed Gray8 conversion shared across frames.
        pixels = (image >> 8).astype(np.uint8)
        pgm = f'P5\n{size} {size}\n255\n'.encode() + pixels.tobytes()
        filename = f'{stamp}.pgm'
        (output / filename).write_bytes(pgm)
        scenes.append({'name': stamp, 'path': filename, 'sha256': hashlib.sha256(pgm).hexdigest(),
                       'source_ordinal': ordinal, 'timestamp_ns': timestamp,
                       'width_px': size, 'height_px': size})
    if not scenes:
        raise ValueError('empty camera sequence')
    return scenes


def checked_marker(gray, grids, marker_id, points, method):
    if not 0 <= marker_id < 36:
        return None
    corners = np.asarray(points).reshape(4, 2)[CORNER_ORDER]
    if not np.isfinite(corners).all():
        raise ValueError('nonfinite reference corner')
    if not ((corners >= 0) & (corners <= np.array(gray.shape[::-1]) - 1)).all():
        return None
    errors = code_errors(corners, gray, grids[marker_id], PAYLOAD_CENTERS)
    if errors > 1:
        return None
    return {'id': marker_id, 'corners_px': corners.ravel().tolist(),
            'pixel_payload_errors': errors, 'reference_detector': method}


def label(reference, output, scenes):
    from aprilgrid import Detector

    if version('aprilgrid') != '0.5.0':
        raise ValueError('reference preparation requires aprilgrid==0.5.0')
    cv2.setNumThreads(1)
    board = Detector('t36h11')
    grids = grids_from_codebook(CODEBOOK)
    for first in range(0, len(scenes), 64):
        batch = scenes[first:first + 64]
        paths = [output / scene['path'] for scene in batch]
        observations = read_probe(reference, paths, kalibr_border=True)
        for scene, path, detections in zip(batch, paths, observations, strict=True):
            gray = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            candidates = [checked_marker(gray, grids, marker_id, points, 'AprilRobotics')
                          for marker_id, hamming, margin, points in detections
                          if hamming == 0 and margin >= 30]
            candidates.extend(checked_marker(gray, grids, int(tag.tag_id), tag.corners, 'aprilgrid')
                              for tag in board.detect(gray))
            by_id = {}
            conflicted = set()
            for tag in candidates:
                if tag is None:
                    continue
                marker_id = tag['id']
                if marker_id not in by_id:
                    by_id[marker_id] = tag
                    continue
                previous = by_id[marker_id]
                a = np.array(previous['corners_px']).reshape(4, 2)
                b = np.array(tag['corners_px']).reshape(4, 2)
                # A single printed grid has one instance of each ID. Exclude spatial conflicts.
                smallest_edge = np.linalg.norm(a - np.roll(a, 1, axis=0), axis=1).min()
                if np.linalg.norm(a - b, axis=1).max() > .3 * smallest_edge:
                    conflicted.add(marker_id)
            scene['markers'] = [by_id[i] for i in sorted(by_id.keys() - conflicted)]
    if not any(scene['markers'] for scene in scenes):
        raise ValueError('no independently verified two-border tags in the entire recording')


def prepare(source, output, reference):
    key = source.stem
    if source.suffix != '.tar' or key not in ARCHIVES or digest_file(source) != ARCHIVES[key]:
        raise ValueError('unrecognized TUM VI EuRoC archive or mismatched SHA-256')
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'expected an empty output directory: {output}')
    size = int(key.split('_')[1])
    with tarfile.open(source, 'r:') as archive:
        calibration = archive_bytes(archive, f'{key}/dso/camchain.yaml', 65536)
        if hashlib.sha256(calibration).hexdigest() != CAMCHAINS[size]:
            raise ValueError('unexpected TUM VI Kalibr calibration file')
        cam = yaml.safe_load(calibration)['cam0']
        if (cam['camera_model'], cam['distortion_model'], cam['resolution']) != (
                'pinhole', 'equidistant', [size, size]):
            raise ValueError('camera model or native raster does not match the sequence')
        output.mkdir(parents=True, exist_ok=True)
        scenes = read_frames(archive, f'{key}/mav0/cam0', size, output)
    provenance = {
        'dataset': 'TUM VI, Schubert et al. (IROS 2018)', 'homepage': HOMEPAGE,
        'license': 'CC BY 4.0',
        'source': f'https://vision.in.tum.de/tumvi/exported/euroc/{size}_16/{key}.tar',
        'source_sha256': ARCHIVES[key], 'camera': 'cam0', 'dictionary': 'tag36h11', 'marker_border_bits': 2,
        'image_conversion': 'uint16 >> 8; source distortion and raster geometry preserved',
        'calibration': {'archive_member': f'{key}/dso/camchain.yaml', 'sha256': CAMCHAINS[size],
                        'camera_model': cam['camera_model'], 'distortion_model': cam['distortion_model'],
                        'intrinsics': cam['intrinsics'], 'distortion_coeffs': cam['distortion_coeffs']},
        'lens': 'Lensagon BF2M2020S23, nominal 195-degree diagonal field of view',
        'selection': 'every indexed cam0 frame in monotonically increasing capture timestamp order',
        'labels': 'unlabelled camera frames',
    }
    if reference:
        label(reference, output, scenes)
        provenance['labels'] = (f'Incomplete AprilRobotics {APRILTAG_REVISION} (two-cell border, '
                                'decimate 1, sigma 0.8, edge refinement, zero corrected bits, '
                                'margin >= 30) and aprilgrid 0.5.0 t36h11 references. In-frame '
                                'corners and <= 1 native payload bit error required for both.')
    write_manifest(output, provenance, scenes)


def mixed_controls(manifest, output):
    """Add a one-cell print to 12 equally spaced exposures, retaining unobscured board labels."""
    manifest_hash = digest_file(manifest)
    corpus = json.loads(manifest.read_bytes())
    if (corpus.get('source_sha256') not in ARCHIVES.values() or corpus.get('marker_border_bits') != 2
            or any('markers' not in scene for scene in corpus['scenes'])):
        raise ValueError('expected a labelled TUM VI two-border manifest')
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'expected an empty output directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    grids, scenes = grids_from_codebook(CODEBOOK), []
    for k in range(12):
        scene = corpus['scenes'][round(k * (len(corpus['scenes']) - 1) / 11)]
        path = manifest.parent / scene['path']
        if digest_file(path) != scene['sha256']:
            raise ValueError(f'changed background: {path}')
        gray = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        h, w = gray.shape
        cell = (3, 5, 8)[k % 3] * (w // 512)
        side = 8 * cell
        span = 10 * cell
        x, y = ((w // 10, h // 7), (w // 9, 5 * h // 7),
                (7 * w // 10, h // 9), (7 * w // 10, 5 * h // 7))[k % 4]
        dark, light = (14, 30, 36)[k % 3], (215, 171, 147)[k % 3]
        marker_id = 201 + k
        cells = np.rot90(np.where(grids[marker_id] > 0, light, dark).astype(np.uint8), -(k % 4))
        gray[y:y + span, x:x + span] = light
        left, top = x + cell, y + cell
        gray[top:top + side, left:left + side] = np.repeat(np.repeat(cells, cell, 0), cell, 1)
        corners = np.array([[left, top], [left + side, top], [left + side, top + side],
                            [left, top + side]], dtype=float) - .5
        markers = []
        for tag in scene['markers']:
            q = np.asarray(tag['corners_px']).reshape(4, 2)
            if (q[:, 0].max() < x or q[:, 0].min() >= x + span or
                    q[:, 1].max() < y or q[:, 1].min() >= y + span):
                markers.append({**tag, 'marker_border_bits': 2})
        markers.append({'id': marker_id, 'marker_border_bits': 1, 'reference_detector': 'rendered',
                        'corners_px': np.roll(corners, -(k % 4), axis=0).ravel().tolist()})
        pgm = f'P5\n{w} {h}\n255\n'.encode() + gray.tobytes()
        (output / scene['path']).write_bytes(pgm)
        scenes.append({**scene, 'background_sha256': scene['sha256'],
                       'sha256': hashlib.sha256(pgm).hexdigest(), 'markers': markers})
    provenance = {k: v for k, v in corpus.items() if k != 'scenes'}
    provenance.update(marker_border_bits=[1, 2], source_manifest_sha256=manifest_hash,
                      selection='12 equally spaced cam0 exposures, endpoints included',
                      labels='One rendered single-border tag and unobscured incomplete camera references')
    write_manifest(output, provenance, scenes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='TUM VI archive, or labelled manifest with --mixed-controls')
    parser.add_argument('output', type=Path)
    parser.add_argument('--reference', type=Path, help='pinned upstream video_reference.c executable')
    parser.add_argument('--mixed-controls', action='store_true',
                        help='add a single-border print to 12 labelled camera frames')
    args = parser.parse_args()
    if args.mixed_controls:
        if args.reference:
            parser.error('--mixed-controls takes a prepared labelled manifest')
        mixed_controls(args.source, args.output)
    else:
        prepare(args.source, args.output, args.reference)
