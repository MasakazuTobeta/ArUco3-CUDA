# SPDX-License-Identifier: Apache-2.0
"""Prepare consecutive camera frames with independently checked tag36h11 references."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from dense_corpus import CODEBOOK, grids_from_codebook


TINKER_URL = ('https://raw.githubusercontent.com/Tinker-Twins/AprilTag/'
              'fa2ea15d15e2065cf86013dc2fd78aecb1540df5/media/input/multiple_tags.mp4')
TINKER_SHA = '7b336ea62ecb78dfb9f48a37bd1de534b120c916f939ca1b96e01d854b87ebcd'
TINKER_WINDOWS = (('dev', 60, 179), ('holdout', 350, 469), ('transition', 880, 999))
APRILTAG_REVISION = '67c0f842a1c24fb6a7527eba77cd5c10135cf8e6'
TAGSLAM_URL = ('https://raw.githubusercontent.com/berndpfrommer/tagslam/'
               'a8afd5d2bb85e8d783831af7cedc0838e0ed53d1/example/example.bag')
TAGSLAM_SHA = 'c7cfdea3fa7c9206f73d6725c729b61a0f6d2edca778e82723e8b66b479a32ca'
TAGSLAM_CAMERA = '/pg_17274483/image_raw/compressed'
TAGSLAM_LABELS = '/detector/tags'
CORNER_ORDER = [1, 0, 3, 2]
MANIFEST_DIGESTS = {
    'tinker': {
        'dev': '9de5e1580f8be826fabadc150eddfd6cfb8a079231c01d38a27649c7a2a10928',
        'holdout': 'd4dfab36f175f814ab3559a78a09bdd0f3e9218bebd4b6470dd30b373b4328d4',
        'transition': '4e98ff046b3e5fb6d99fb8f91f6bc3655e0599e8c3763b15adadb5ef87c9c285',
    },
    'tinker-full': {'full': '4c14d1de5ae012e9013a87f6dffedf0ed29d42cc3209eb7c38ff6947f4b21520'},
    'tinker-tail': {'tail': 'd4ff132b22a597e6d5cc5aded4ac3c0410cf671d9ef99055e5dfd20584763b25'},
    'tagslam': {
        'full': 'a8349f86dbdb6cae94c36e7912d5801dee9ecd45b314fa8f3eefbe4f951e7b72',
        'dev': '0996211425840621ae87200d342bc60c73301638962d4937d63ab0f22710322e',
        'holdout': '13f3bf3bb3a6b634616aaed4444c9cd87738ce044f0da460963cebf04ce76b3b',
    },
}
UNIT = np.array([(0, 0), (1, 0), (1, 1), (0, 1)], np.float32)
SAMPLE_CENTERS = np.array([((x + .5) / 8, (y + .5) / 8) for y in range(1, 7)
                           for x in range(1, 7)], np.float32).reshape(1, 36, 2)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def code_errors(corners, gray, grid, sample_centers=SAMPLE_CENTERS):
    homography = cv2.getPerspectiveTransform(UNIT, np.asarray(corners, dtype=np.float32))
    points = cv2.perspectiveTransform(sample_centers, homography)
    if not ((points >= 0) & (points <= (gray.shape[1] - 1, gray.shape[0] - 1))).all():
        raise ValueError('reference payload centers must be visible')
    levels = cv2.remap(gray, points[:, :, 0], points[:, :, 1], cv2.INTER_LINEAR).ravel()
    threshold = (np.percentile(levels, 12) + np.percentile(levels, 88)) / 2
    expected = grid[1:7, 1:7].ravel() > 0
    return int(np.count_nonzero((levels > threshold) != expected))


def verified_tag(marker_id, corners, gray, grids, max_errors):
    if not 0 <= marker_id < len(grids):
        raise ValueError(f'unknown tag36h11 ID {marker_id}')
    errors = code_errors(corners, gray, grids[marker_id])
    if errors > max_errors:
        raise ValueError(f'printed payload mismatch for ID {marker_id}: {errors} cells')
    return {'id': marker_id, 'corners_px': np.asarray(corners).ravel().tolist(),
            'pixel_payload_errors': errors}


def write_manifest(corpus, output, provenance, partition, scenes):
    data = (json.dumps({**provenance, 'partition': partition, 'scenes': scenes}, indent=2)
            + '\n').encode()
    digest = sha256(data)
    if digest != MANIFEST_DIGESTS[corpus][partition]:
        raise ValueError(f'{corpus} {partition} manifest SHA-256 mismatch: {digest}')
    filename = 'manifest.json' if partition == 'full' else f'manifest-{partition}.json'
    (output / filename).write_bytes(data)
    matches = Counter(tag['pixel_payload_errors'] for scene in scenes for tag in scene['markers'])
    print(f'{filename} {digest} {len(scenes)} frames, {sum(matches.values())} references')


def probe_tinker(source):
    info = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_streams', '-show_frames',
        '-show_entries',
        'stream=width,height,pix_fmt,time_base,codec_name:frame=best_effort_timestamp_time',
        '-of', 'json', str(source)], text=True))
    stream = info['streams'][0]
    if (stream['width'], stream['height'], stream['pix_fmt'], stream['codec_name']) != (
            720, 480, 'yuv420p', 'h264'):
        raise ValueError(f'unexpected Tinker video format: {stream}')
    timestamps = [float(frame['best_effort_timestamp_time']) for frame in info['frames']]
    if len(timestamps) != 2122:
        raise ValueError('unexpected presentation frame count')
    return timestamps


def read_probe(reference, images, *, kalibr_border=False):
    observations, counts = [[] for _ in images], []
    options = ['--kalibr-border'] if kalibr_border else []
    text = subprocess.check_output([str(reference.resolve()), *options, *map(str, images)], text=True)
    for line in text.splitlines():
        kind, frame, *values = line.split()
        index = int(frame)
        if kind == 'frame' and len(values) == 1 and index == len(counts) < len(images):
            counts.append(int(values[0]))
        elif kind == 'tag' and len(values) == 11 and index == len(counts) - 1 and index >= 0:
            marker_id, hamming, margin = values[:3]
            observations[index].append((int(marker_id), int(hamming), float(margin),
                                        list(map(float, values[3:]))))
        else:
            raise ValueError(f'invalid AprilRobotics reference row: {line}')
    if counts != [len(items) for items in observations]:
        raise ValueError('incomplete AprilRobotics reference')
    return observations


def prepare_tinker(source, output, reference, grids, corpus):
    windows = {
        'tinker': TINKER_WINDOWS,
        'tinker-tail': (('tail', 1000, 2121),),
        'tinker-full': (('full', 0, 2121),),
    }[corpus]
    timestamps = probe_tinker(source)
    selection = '+'.join(f'between(n,{first},{last})' for _, first, last in windows)
    subprocess.run([
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-threads', '1', '-filter_threads', '1',
        '-i', str(source), '-vf', f"select='{selection}',setpts=N/TB,format=gray",
        '-fps_mode', 'passthrough', str(output / 'frame-%05d.pgm')], check=True)
    images = sorted(output.glob('frame-*.pgm'))
    ordinals = [i for _, first, last in windows for i in range(first, last + 1)]
    if len(images) != len(ordinals):
        raise ValueError('incorrect decoded frame count')
    observations = read_probe(reference, images)
    provenance = {
        'dictionary': 'tag36h11', 'marker_border_bits': 1, 'source': TINKER_URL,
        'source_sha256': TINKER_SHA, 'input_encoding': 'H.264 yuv420p, FFmpeg gray8 conversion',
        'frame_timestamp_basis': 'decoded presentation order, best_effort_timestamp_time',
        'nominal_capture_rate_hz': 30,
        'labels': (f'AprilRobotics3 {APRILTAG_REVISION} sigma 0.8 decimate1; '
                   'independent native payload verification'),
    }
    seen, records = set(), []
    for file, ordinal, detections in zip(images, ordinals, observations, strict=True):
        data = file.read_bytes()
        digest = sha256(data)
        if digest in seen:
            raise ValueError(f'repeated decoded pixels: {ordinal}')
        seen.add(digest)
        gray = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)
        if gray is None or gray.shape != (480, 720) or gray.dtype != np.uint8:
            raise ValueError(f'expected Gray8 input: {file}')
        markers = []
        for marker_id, hamming, margin, points in sorted(detections):
            if corpus != 'tinker' and hamming != 0:
                continue
            if not (0 <= marker_id <= 4 and hamming == 0 and margin >= 30):
                raise ValueError(f'unexpected reference detection in source frame {ordinal}')
            corners = np.asarray(points).reshape(4, 2)[CORNER_ORDER]
            markers.append(verified_tag(marker_id, corners, gray, grids, max_errors=1))
        records.append({'name': file.stem, 'path': file.name, 'sha256': digest,
                        'source_ordinal': ordinal, 'source_pts_seconds': timestamps[ordinal],
                        'width_px': 720, 'height_px': 480, 'markers': markers})
    for partition, first, last in windows:
        selected = [scene for scene in records if first <= scene['source_ordinal'] <= last]
        intervals = np.diff([scene['source_pts_seconds'] for scene in selected])
        if not ((intervals >= .020) & (intervals <= .050)).all():
            raise ValueError(f'unexpected frame cadence in {partition}')
        write_manifest(corpus, output, provenance, partition, selected)


def tagslam_messages(source):
    from rosbags.highlevel import AnyReader

    frames, labels = {}, {}
    with AnyReader([source]) as reader:
        connections = [item for item in reader.connections
                       if item.topic in (TAGSLAM_CAMERA, TAGSLAM_LABELS)]
        for connection, _, serialized in reader.messages(connections=connections):
            message = reader.deserialize(serialized, connection.msgtype)
            stamp = message.header.stamp
            timestamp = stamp.sec * 1_000_000_000 + stamp.nanosec
            if connection.topic == TAGSLAM_LABELS:
                if timestamp in labels:
                    raise ValueError('duplicate tag timestamp')
                labels[timestamp] = message.apriltags
            else:
                if timestamp in frames:
                    raise ValueError('duplicate camera timestamp')
                if message.format.strip() != 'bayer_rggb8; jpeg compressed':
                    raise ValueError(f'unexpected camera encoding: {message.format}')
                frames[timestamp] = (message.header.seq, message.data)
    return frames, labels


def prepare_tagslam(source, output, grids):
    frames, labels = tagslam_messages(source)
    timestamps = sorted(frames.keys() & labels.keys())
    if [frames[ts][0] for ts in timestamps] != list(range(9, 44)):
        raise ValueError('incomplete camera sequence 9–43')
    intervals = np.diff(timestamps)
    if not ((intervals > 490_000_000) & (intervals < 510_000_000)).all():
        raise ValueError('unexpected TagSLAM cadence')
    records = []
    for timestamp in timestamps:
        ordinal, payload = frames[timestamp]
        mosaic = cv2.imdecode(payload, cv2.IMREAD_UNCHANGED)
        if mosaic is None or mosaic.shape != (1200, 1920) or mosaic.dtype != np.uint8:
            raise ValueError('expected a native 1920×1200 RGGB8 mosaic')
        gray = cv2.cvtColor(mosaic, cv2.COLOR_BayerRGGB2GRAY)
        file = output / f'frame-{ordinal:03}.pgm'
        data = b'P5\n1920 1200\n255\n' + gray.tobytes()
        file.write_bytes(data)
        markers = []
        for item in labels[timestamp]:
            if item.family != 'mit' or item.bits != 6:
                raise ValueError('unexpected TagSLAM tag family')
            corners = [(item.corners[i].x, item.corners[i].y) for i in CORNER_ORDER]
            markers.append(verified_tag(item.id, corners, gray, grids, max_errors=0))
        records.append({'name': file.stem, 'path': file.name, 'sha256': sha256(data),
                        'source_sha256': sha256(payload), 'source_seq': ordinal,
                        'capture_timestamp_ns': timestamp, 'width_px': 1920,
                        'height_px': 1200, 'markers': markers})
    provenance = {
        'dictionary': 'tag36h11', 'source': TAGSLAM_URL, 'archive_sha256': TAGSLAM_SHA,
        'camera_topic': TAGSLAM_CAMERA, 'labels_topic': TAGSLAM_LABELS,
        'stored_corner_order': CORNER_ORDER,
        'input_conversion': 'bayer_rggb8 JPEG decode unchanged; cv2.COLOR_BayerRGGB2GRAY',
        'labels': ('TagSLAM recorded detector references; identities and orientation checked '
                   'against true image payload cells'),
    }
    for partition, selected in (('full', records), ('dev', records[:18]), ('holdout', records[18:])):
        write_manifest('tagslam', output, provenance, partition, selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('corpus', choices=('tinker', 'tinker-tail', 'tinker-full', 'tagslam'))
    parser.add_argument('source', type=Path, help='original MP4 or ROS1 bag')
    parser.add_argument('output', type=Path)
    parser.add_argument('--reference', type=Path, help='compiled video_reference.c for Tinker')
    args = parser.parse_args()
    if args.corpus.startswith('tinker') and args.reference is None:
        parser.error('tinker requires --reference')
    if args.corpus == 'tagslam' and args.reference is not None:
        parser.error('--reference applies to tinker')
    try:
        expected = TINKER_SHA if args.corpus.startswith('tinker') else TAGSLAM_SHA
        if sha256(args.source.read_bytes()) != expected:
            raise ValueError(f'input SHA-256 mismatch: {args.source}')
        args.output.mkdir(parents=True, exist_ok=True)
        if any(args.output.iterdir()):
            raise ValueError('output directory must be empty')
        cv2.setNumThreads(1)
        grids = grids_from_codebook(CODEBOOK)
        if args.corpus.startswith('tinker'):
            prepare_tinker(args.source, args.output, args.reference, grids, args.corpus)
        else:
            prepare_tagslam(args.source, args.output, grids)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
