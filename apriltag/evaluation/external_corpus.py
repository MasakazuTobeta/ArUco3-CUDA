# SPDX-License-Identifier: Apache-2.0
"""Prepare the pinned FRC video and ICRA 2020 board evaluations."""
import argparse
import csv
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request
from urllib.parse import quote

import cv2
import numpy as np


FRC_REVISION = '527979f5ee524ea769cf1d15a948fa19d389129c'
FRC_BASE = f'https://huggingface.co/datasets/wt200999/apriltag-dataset/resolve/{FRC_REVISION}/train'
FRC_METADATA_SHA256 = 'a68eb5d4cd5d9476bde8ba8f3d9e725172c55347479bd468602b83d8c5467a18'
FRC_CLIPS = (
    'FRC_ROBOT_POV_-_Southern_Cross_2024',
    'Half_Hollow_Hills_Invitational_Playoff_Match_6_2025',
    'SF1_-_WPI_-_190_360_POV',
    '2025_Field_Tour_Video__Barge',
)
# Source SHA-256 prefixes identify the selected frames in the pinned metadata.
FRC_SELECTIONS = {
    'frc-small': (
        # Half_Hollow_Hills_Invitational_Playoff_Match_9_2025
        '005ea53419ca', '0b9d5759f705', '3ae2b0dfe2ca', '7cd948c5d3ec', 'ba745d8f931b', 'd127b7c80d86',
        # WPI_2025_District_Event_Qual_Match_48_POV
        'b039a53c480b', '03d2a0bb822d', '006f41515a80', '160f6ba12a0c', '767c558791b7', '77f15a9f54da',
        # SF7_-_WPI_-_190_POV
        '04adc3e8eb29', '73146a50eae5', 'b7661070d9f0', '42583442b345', 'c7f739238756', '0d59bf732079',
        # FRC_9483___Robot_POV__2024_Milstein_Q67_
        '663e7dba4ab2', 'f441658ee69b', 'd6f9834a5cbe', 'f89dfffe0ff0', 'b6c5127e4aef', 'af29ed94fe48',
    ),
    'frc-large-held': (
        '962362c90905', '5c6b0b56e650', 'abc833fef056', 'f2c32a34254a',
        '4ae6949b061d', '6d954acba25c', 'e32e50128ec5', '125b5d031811',
        '855d75fd0cd6', 'f4d27b48cd3e', '3d4cd1b92e5c', '78515dc1aff5',
        '301e49390006', 'cf7ba97a5b35', 'c0b5a05c5932', '7b272ee088e7',
        '1ddd9d8c06fd', 'c632b370fd36', '73e3689e69f7', 'ba79f9033e88',
        'fddfa61f298f', 'b89051fafb40', '032a7388436d', '5e6356cf168a',
        'dd0596619a9e', 'e2681a69cae6', '1e025a605b5c', '5be63d0ba3f4',
        '6ab3e6853d45', 'a07ac8150448', '06c04f7b0d83', '2c164b258389',
    ),
    'frc-events': (
        # Q10_-_Southern_Maine_-_190_POV
        '03de6da0776d', '975d070d1705', 'e2da5add2693', '30070c90d979', 'b4eef4b3f3d4', '4dc657df4431',
        # Robot_POV__FIRST_Crescendo_Jackson_District_Match
        '3040fa37ecbc', 'aa8a0e92c4af', '049538ca19f5', '9ef347930c8a', '8fd308d46568', 'c1d2608cfe0d',
        # M11_-_Galileo_-_190_POV
        '3b9ccb79c01e', '0041a0b02777', '601cd261fae2', 'd80128835704', '9531eb23d8df', '9d4e4fec176f',
        # Q64_-_URI_-_190_POV
        '6d9993992dea', '84bdf0c78d13', 'be4488c24044', 'cd6bb6b0ac9a', 'd13189b9295d', '5cd053af1679',
    ),
}
ICRA_BASE = ('https://huggingface.co/datasets/NoeFontana/apriltag-validation-data/'
             'resolve/6c7653385d480351fe8235d010dedc18fbc0b992')
ICRA_SOURCES = {
    'icra-forward': ('forward.tar.xz',
                     '2e0aefa48227bb37a87464323f8689a4963aba2ff0fe0d4c3a3d73753aa7a9c6', 50, 1),
    'icra-circle': ('circle.tar.xz',
                    '47133d5d5a36619c81eb406a9dfee0e2dd0ae3cd5f36c604edbb99bf31154fe7', 181, 5),
}
MANIFEST_SHA256 = {
    'frc-small': 'b2cd4e3a2e1d03acea3d3e6dacae74ade7f192ba4efe43765f6ef17a323b2613',
    'frc-events': 'a8d426f56ca6b4addb56b998e90ee76d514c83cb3b0332c6d6bc75cdf8e83da4',
    'frc-large-held': 'b568193acf3571f174a13da6de53fef08cf2ffe3699f7aedb4a390c075197fe6',
    'frc-video': '766ce4984aec7ec3d86bd885f3ff321a5e4de88e8252605024865e4d20ae384e',
    'icra-forward': '875df351b9e73d7bf415315b52425065b3321fac0c844ea6080117a43f825cef',
    'icra-circle': '714e3e3438b7b3626833f2d5b75d845ab71f169817040c19898c26a6d4a3b6ce',
}
# Payload checks established the same input corner permutation in both sources.
CORNER_ORDER = (1, 0, 3, 2)


def checked_bytes(path, digest):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f'SHA-256 mismatch: {path}')
    return data


def verify_stream(stream, digest):
    checksum = hashlib.sha256()
    for block in iter(lambda: stream.read(1 << 20), b''):
        checksum.update(block)
    if checksum.hexdigest() != digest:
        raise ValueError(f'SHA-256 mismatch: {stream.name}')
    stream.seek(0)


@contextmanager
def open_source(cache, name, url, digest, offline):
    path = cache / name
    if not path.is_file():
        if offline:
            raise ValueError(f'missing offline input: {path}')
        download = tempfile.NamedTemporaryFile(dir=cache, delete=False)
        partial = Path(download.name)
        try:
            with download, urllib.request.urlopen(url, timeout=60) as response:
                shutil.copyfileobj(response, download)
            with partial.open('rb') as stream:
                verify_stream(stream, digest)
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
    with path.open('rb') as stream:
        verify_stream(stream, digest)
        yield stream


def get_source(cache, name, url, digest, offline):
    with open_source(cache, name, url, digest, offline) as stream:
        return stream.read()


def write_frame(output, file, data, width, height):
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (height, width):
        raise ValueError(f'unexpected source dimensions: {file}')
    pgm = f'P5\n{width} {height}\n255\n'.encode() + image.tobytes()
    (output / file).write_bytes(pgm)
    return hashlib.sha256(pgm).hexdigest()


def frc_frames(metadata, corpus='frc-video'):
    if corpus in FRC_SELECTIONS:
        order = FRC_SELECTIONS[corpus]
        wanted = set(order)
        selected = {}
        for line in metadata.splitlines():
            row = json.loads(line)
            key = row['image_sha256'][:12]
            if key in wanted:
                if key in selected:
                    raise ValueError(f'duplicate selected FRC source: {key}')
                selected[key] = row
        if set(selected) != wanted or len(wanted) != len(order):
            raise ValueError(f'incomplete FRC source selection: {corpus}')
        yield from (selected[key] for key in order)
        return
    by_clip = defaultdict(list)
    for line in metadata.splitlines():
        row = json.loads(line)
        stem = Path(row['file_name']).stem
        parts = stem[13:].rsplit('_frame_', 1)
        if len(parts) == 2 and parts[0] in FRC_CLIPS:
            by_clip[parts[0]].append((parts[1], row))
    for name in FRC_CLIPS:
        clip = [row for _, row in sorted(by_clip[name], key=lambda entry: entry[0])]
        if len(clip) < 8:
            raise ValueError(f'incomplete FRC clip: {name}')
        yield from (clip[i * (len(clip) - 1) // 7] for i in range(8))


def prepare_frc(cache, output, offline, gray_cache, corpus):
    metadata = get_source(cache, 'metadata.jsonl', f'{FRC_BASE}/metadata.jsonl',
                          FRC_METADATA_SHA256, offline)
    cached_frames = {}
    if gray_cache:
        data = checked_bytes(gray_cache / 'manifest.json', MANIFEST_SHA256[corpus])
        cached_frames = {scene['path']: scene for scene in json.loads(data)['scenes']}

    def prepare(row):
        source = Path(row['file_name']).name
        file = source[:12] + '.pgm'
        if gray_cache:
            cached = cached_frames[file]
            if cached['source_image_sha256'] != row['image_sha256']:
                raise ValueError(f'source SHA-256 mismatch: {source}')
            digest = cached['sha256']
            (output / file).write_bytes(checked_bytes(gray_cache / file, digest))
        else:
            data = get_source(cache, source, f'{FRC_BASE}/{quote(source)}', row['image_sha256'], offline)
            digest = write_frame(output, file, data, row['image_width'], row['image_height'])
        markers = [{'id': tag['tag_id'], 'corners_px': reorder(tag['corners'])}
                   for tag in row['detections'] if tag['tag_family'] == 'tag36h11']
        return {'name': Path(source).stem, 'path': file, 'sha256': digest,
                'source_image_sha256': row['image_sha256'], 'markers': markers}

    with ThreadPoolExecutor(max_workers=6) as workers:
        return list(workers.map(prepare, frc_frames(metadata, corpus)))


def reorder(corners):
    # Preserve the source annotation precision when serializing.
    return np.asarray(corners).reshape(4, 2)[list(CORNER_ORDER)].ravel().tolist()


def prepare_icra(cache, output, offline, corpus):
    archive, source_digest, frame_count, step = ICRA_SOURCES[corpus]
    expected = {f'{index:04d}.png' for index in range(frame_count)}
    selected = {f'{index:04d}.png' for index in range(0, frame_count, step)}
    width, height = 2448, 2048
    visible = np.array([[-.5, -.5], [width - .5, -.5], [width - .5, height - .5],
                        [-.5, height - .5]], np.float32)
    seen = set()
    frames = {}
    labels = defaultdict(lambda: defaultdict(dict))
    with open_source(cache, archive, f'{ICRA_BASE}/{archive}', source_digest, offline) as stream:
        with tarfile.open(fileobj=stream, mode='r|xz') as source:
            for member in source:
                if not member.isfile():
                    continue
                name = Path(member.name)
                if name.parent.name == 'pure_tags_images' and name.suffix == '.png':
                    if name.name in seen:
                        raise ValueError(f'duplicate ICRA image: {name}')
                    seen.add(name.name)
                    if name.name in selected:
                        with source.extractfile(member) as image:
                            frames[name.name] = write_frame(
                                output, name.stem + '.pgm', image.read(), width, height)
                elif name.name == 'tags.csv':
                    with source.extractfile(member) as table:
                        # Buffer only the CSV so parsing does not seek within the streamed archive.
                        rows = csv.DictReader(io.StringIO(table.read().decode()))
                        for row in rows:
                            image = row['image']
                            if image not in selected:
                                continue
                            marker_id, corner = int(row['tag_id']), int(row['corner'])
                            marker = labels[image][marker_id]
                            if corner in marker:
                                raise ValueError(f'duplicate ICRA corner: {image} {marker_id} {corner}')
                            marker[corner] = (float(row['ground_truth_x']),
                                              float(row['ground_truth_y']))
    if seen != expected or frames.keys() != selected or labels.keys() != selected:
        raise ValueError(f'unexpected ICRA source frame or annotation set: {corpus}')

    scenes = []
    degenerate = 0
    for image, digest in sorted(frames.items()):
        markers = []
        for marker_id, rows in sorted(labels[image].items()):
            if rows.keys() != {0, 1, 2, 3}:
                raise ValueError(f'incomplete ICRA marker: {image} {marker_id}')
            corners = reorder([rows[index] for index in range(4)])
            outline = np.asarray(corners, np.float32).reshape(4, 2)
            if not np.isfinite(outline).all():
                raise ValueError(f'nonfinite ICRA projection: {image} {marker_id}')
            if cv2.contourArea(outline) == 0:
                if cv2.contourArea(cv2.convexHull(outline)) != 0:
                    raise ValueError(f'self-crossing ICRA projection: {image} {marker_id}')
                degenerate += 1
                continue
            if not cv2.isContourConvex(outline):
                raise ValueError(f'nonconvex ICRA projection: {image} {marker_id}')
            area, _ = cv2.intersectConvexConvex(outline, visible)
            if area > 0:
                markers.append({'id': marker_id, 'corners_px': corners})
        scene = Path(image).stem
        scenes.append({'name': scene, 'path': scene + '.pgm', 'sha256': digest, 'markers': markers})
    return scenes, degenerate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('corpus', choices=MANIFEST_SHA256)
    parser.add_argument('output', type=Path)
    parser.add_argument('--cache', type=Path, required=True, help='directory for verified source files')
    parser.add_argument('--offline', action='store_true', help='use cached files only')
    parser.add_argument('--gray-cache', type=Path, help='verified prior output directory for the selected FRC sequence')
    args = parser.parse_args()
    if args.gray_cache and args.corpus in ICRA_SOURCES:
        parser.error('--gray-cache applies to the FRC sequences')
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        parser.error('output directory must be empty')
    args.cache.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    try:
        manifest = {'dictionary': 'tag36h11', 'original_corner_order': CORNER_ORDER}
        if args.corpus not in ICRA_SOURCES:
            if args.corpus == 'frc-large-held':
                # Keep the exact manifest bytes frozen before either detector scored the holdout.
                manifest = {'name': 'independent photographs with large tags'}
            else:
                manifest.update(release=FRC_REVISION,
                                selection=('Six stored frames per named clip, in pinned selection order'
                                           if args.corpus in FRC_SELECTIONS else
                                           'Eight evenly ranked stored reference frames per named video, in source frame order'))
            manifest['scenes'] = prepare_frc(args.cache, args.output, args.offline, args.gray_cache,
                                             args.corpus)
        else:
            manifest['source_sha256'] = ICRA_SOURCES[args.corpus][1]
            scenes, degenerate = prepare_icra(args.cache, args.output, args.offline, args.corpus)
            if args.corpus == 'icra-circle':
                manifest.update(selection='Every fifth pure_tags_images frame in filename order',
                                excluded_degenerate_projections=degenerate)
            manifest['scenes'] = scenes
        data = (json.dumps(manifest, indent=2) + '\n').encode()
        digest = hashlib.sha256(data).hexdigest()
        if digest != MANIFEST_SHA256[args.corpus]:
            raise ValueError(f'generated {args.corpus} manifest SHA-256 mismatch: {digest}')
    except (OSError, ValueError, tarfile.TarError) as error:
        parser.exit(1, f'{error}\n')
    (args.output / 'manifest.json').write_bytes(data)


if __name__ == '__main__':
    main()
