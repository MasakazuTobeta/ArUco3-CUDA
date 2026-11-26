# SPDX-License-Identifier: Apache-2.0
"""Compare the public NAPA API with a project CPU replay executable on a Gray8 manifest."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evaluation'))
from score import rmse, score


def run_napa(warmup, passes, paths, min_tag_px, max_detections):
    import numpy as np
    from PIL import Image
    from napa_vision import NapaDetector

    if importlib.metadata.version('napa-vision') != '0.6.1':
        raise ValueError('measurement requires napa-vision==0.6.1')
    frames = []
    for path in paths:
        with Image.open(path) as image:
            pixels = np.array(image)
        if pixels.ndim != 2 or pixels.dtype != np.uint8:
            raise ValueError(f'expected single-channel Gray8: {path}')
        frames.append(pixels)
    detector = NapaDetector(min_tag_px=min_tag_px, max_detections=max_detections)
    for cycle in range(-warmup, passes):
        for i, frame in enumerate(frames):
            start = time.perf_counter_ns()
            detections = detector.detect(frame)
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            if cycle >= 0:
                print('ms', i, elapsed_ms)
            if cycle == 0:
                for tag in detections:
                    print('tag', i, int(tag.tag_id), *tag.corners.flatten())


def parse_run(text, names, passes):
    order = []
    durations = []
    observations = {name: [] for name in names}
    for line in text.splitlines():
        kind, number, *row = line.split()
        index = int(number)
        if not 0 <= index < len(names):
            raise ValueError('out-of-range frame index')
        if kind == 'ms' and len(row) == 1:
            duration = float(row[0])
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError('invalid latency')
            durations.append(duration)
            order.append(index)
        elif kind == 'tag' and len(row) == 9:
            observations[names[index]].append({'id': int(row[0]), 'corners': list(map(float, row[1:]))})
        else:
            raise ValueError(f'invalid measurement row: {line}')
    if order != list(range(len(names))) * passes:
        raise ValueError('timed frames are missing or out of order')
    durations.sort()
    summary = [durations[math.ceil(p * len(durations)) - 1] for p in (.5, .95, .99)]
    summary.append(1000 * len(durations) / math.fsum(durations))
    return observations, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--scorer', type=Path, required=True)
    parser.add_argument('--stride', type=int, choices=(0, *range(2, 9)), required=True)
    parser.add_argument('--search', choices=('components', 'sampled-contours', 'auto', 'video'), help='small CPU candidate search')
    parser.add_argument('--border', action='store_true', help='add bounded native frame-edge search')
    parser.add_argument('--recover', action='store_true', help='verify local video candidates after ordinary search')
    parser.add_argument('--alternate-contours', type=int,
                        help='trace sampled contours at this Gray8 threshold every second frame')
    parser.add_argument('--threshold', type=int, help='sampled-contour Gray8 threshold (default 148)')
    parser.add_argument('--grid-radius', type=int, help='component grid fit radius in pixels')
    parser.add_argument('--napa-max-detections', type=int, default=64, help='NAPA result capacity per frame')
    parser.add_argument('--napa-min-tag-px', type=int, default=22, help='NAPA minimum tag side in pixels')
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--passes', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or min(args.passes, args.repeats, args.napa_min_tag_px, args.napa_max_detections) < 1:
        parser.error('invalid warmup, passes, repeats or NAPA size/capacity')
    if args.search in ('components', 'sampled-contours') and args.stride not in (2, 3, 4):
        parser.error('sampled search requires stride 2, 3 or 4')
    if args.threshold is not None and (args.search != 'sampled-contours' or not 0 <= args.threshold <= 255):
        parser.error('threshold requires sampled-contour search and must be 0 through 255')
    if args.alternate_contours is not None and (args.search != 'components' or not args.recover or
                                                 args.stride != 2 or not 0 <= args.alternate_contours <= 255):
        parser.error('alternate contours require component search, stride 2, --recover, and a threshold from 0 to 255')
    if args.search == 'video' and (args.stride != 2 or args.recover or args.border):
        parser.error('video search requires stride 2 and owns its recovery')
    if args.search == 'auto' and args.stride != 3:
        parser.error('automatic search requires stride 3')
    if args.grid_radius is not None and (args.search != 'components' or
                                         args.grid_radius not in (0, *range(2, 33))):
        parser.error('grid radius requires component search and must be 0 or 2 through 32')
    manifest = args.manifest.read_bytes()
    scenes = json.loads(manifest)['scenes']
    annotated = all('markers' in scene for scene in scenes)
    if not annotated and any('markers' in scene for scene in scenes):
        parser.error('mixed annotated and unannotated frames')
    names = [scene['name'] for scene in scenes]
    if not names or len(set(names)) != len(names):
        parser.error('expected nonempty manifest with unique scene names')
    inputs = []
    for scene in scenes:
        path = (args.manifest.parent / scene['path']).resolve()
        if hashlib.sha256(path.read_bytes()).hexdigest() != scene['sha256']:
            parser.error(f'input digest mismatch: {path}')
        inputs.append(str(path))
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        parser.error('output directory must be empty')
    project = [str(args.native.resolve())]
    if args.search is not None:
        project.extend(['--search', args.search])
    if args.grid_radius is not None:
        project.extend(['--grid-radius', str(args.grid_radius)])
    if args.threshold is not None:
        project.extend(['--threshold', str(args.threshold)])
    if args.alternate_contours is not None:
        project.extend(['--alternate-contours', str(args.alternate_contours)])
    if args.border:
        project.append('--border')
    if args.recover:
        project.append('--recover')
    project.append(str(args.stride))
    commands = {'project': project,
                'napa': [sys.executable, __file__, '--napa-worker', '--min-tag-px', str(args.napa_min_tag_px),
                         '--max-detections', str(args.napa_max_detections)]}
    all_stats = {name: [] for name in commands}
    first = {}
    for repeat in range(args.repeats):
        for route in (('napa', 'project') if repeat % 2 == 0 else ('project', 'napa')):
            cmd = commands[route] + [str(args.warmup), str(args.passes), *inputs]
            text = subprocess.check_output(cmd, text=True, env={**os.environ, 'OPENBLAS_NUM_THREADS': '1'})
            (args.out / f'{route}-{repeat}.txt').write_text(text)
            observed, stats = parse_run(text, names, args.passes)
            if repeat and first[route] != observed:
                raise ValueError(f'{route}: detections differ across processes')
            first[route] = observed
            all_stats[route].append(stats)
    matches = {}
    result = {'manifest_sha256': hashlib.sha256(manifest).hexdigest(), 'frames': len(names),
              'candidate_sha256': hashlib.sha256(args.native.read_bytes()).hexdigest(),
              'warmup': args.warmup, 'passes': args.passes, 'processes_per_route': args.repeats,
              'stride_px': args.stride, 'napa_min_tag_px': args.napa_min_tag_px,
              'napa_max_detections': args.napa_max_detections}
    if args.border:
        result['border'] = True
    if args.recover:
        result['recover'] = True
    if args.alternate_contours is not None:
        result['alternate_contour_threshold'] = args.alternate_contours
    if args.search is not None:
        result['search'] = args.search
        if args.search == 'components':
            result['grid_radius_px'] = args.grid_radius or 0
        elif args.search == 'sampled-contours':
            result['threshold'] = 148 if args.threshold is None else args.threshold
    required = {(scene['name'], i) for scene in scenes
                for i in range(len(scene.get('markers', [])))}
    scoring_scenes = scenes
    if args.search == 'video' and annotated:
        scoring_scenes = [{**scene, 'markers': scene['markers'] + scene.get('partial_markers', [])}
                          for scene in scenes]
    confirmed = {}
    for route in commands:
        summary = dict(zip(('p50_ms', 'p95_ms', 'p99_ms', 'fps'),
                           map(statistics.median, zip(*all_stats[route]))))
        if annotated:
            matches[route], _, quality = score(scoring_scenes, first[route], str(args.scorer.resolve()))
            confirmed[route] = ((required & matches[route].keys())
                                - set(quality['id_errors']) - set(quality['rotation_errors']))
            if args.search == 'video':
                quality['required'] = len(required)
                quality['confirmed_required'] = len(confirmed[route])
                quality['stale_outputs'] = sum(
                    tag['id'] in scene.get('must_be_absent_ids', ())
                    for scene in scenes for tag in first[route][scene['name']])
        else:
            quality = {'returned': sum(map(len, first[route].values()))}
        result[route] = {**summary, **quality}
    if annotated:
        comparable = confirmed if args.search == 'video' else {
            route: matched.keys() for route, matched in matches.items()}
        common = comparable['napa'] & comparable['project']
        result['shared'] = {'count': len(common), 'napa_rmse_px': rmse(matches['napa'], common),
                            'project_rmse_px': rmse(matches['project'], common)}
        for route, other in (('project', 'napa'), ('napa', 'project')):
            result[f'{route}_only'] = sorted(comparable[route] - comparable[other])
    summary = json.dumps(result, indent=2)
    (args.out / 'summary.json').write_text(summary + '\n')
    print(summary)


if __name__ == '__main__':
    if sys.argv[1:2] == ['--napa-worker']:
        parser = argparse.ArgumentParser()
        parser.add_argument('--max-detections', type=int, default=64)
        parser.add_argument('--min-tag-px', type=int, default=22)
        parser.add_argument('warmup', type=int)
        parser.add_argument('passes', type=int)
        parser.add_argument('paths', nargs='+')
        worker = parser.parse_args(sys.argv[2:])
        run_napa(worker.warmup, worker.passes, worker.paths, worker.min_tag_px, worker.max_detections)
    else:
        main()
