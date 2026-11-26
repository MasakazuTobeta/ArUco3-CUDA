# SPDX-License-Identifier: Apache-2.0
"""Score a paired CPU replay against NAPA, official OpenCV and NVIDIA cuAprilTags."""
import argparse
import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from compare_napa_cpu import parse_run, rmse, score


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def groups(manifest, scenes):
    group = []
    current_size = None
    for scene in scenes:
        path = (manifest.parent / scene['path']).resolve()
        if digest(path) != scene['sha256']:
            raise ValueError(f'Changed input: {path}')
        with Image.open(path) as image:
            size = image.size
            if image.mode != 'L':
                raise ValueError(f'Expected Gray8: {path}')
        capacity = min(64, max(1, (128 << 20) // (size[0] * size[1])))
        if group and (size != current_size or len(group) >= capacity):
            yield group
            group = []
        group.append((scene['name'], str(path)))
        current_size = size
    if group:
        yield group


def nvidia_observations(text, names, capacity):
    rows = [json.loads(line) for line in text.splitlines()]
    if len(rows) != len(names):
        raise ValueError('Incomplete cuAprilTags replay')
    observations = {}
    for index, (row, name) in enumerate(zip(rows, names)):
        if (row['frame'], row['cycle'], row['backend']) != (index, 0, 'nvidia'):
            raise ValueError('Unexpected cuAprilTags frame')
        if len(row['tags']) >= capacity:
            raise ValueError('cuAprilTags returned at capacity')
        tags = []
        for tag in row['tags']:
            points = tag['corners']
            if len(points) != 4 or any(len(p) != 2 or not all(map(math.isfinite, p)) for p in points):
                raise ValueError('Invalid cuAprilTags corners')
            points = points[2:] + points[:2]
            tags.append({'id': tag['id'], 'corners': [v for p in points for v in p]})
        observations[name] = tags
    return observations


def collect(manifest, scenes, args):
    observed = {'opencv': {}, 'cuapriltags': {}}
    for group in groups(manifest, scenes):
        names, paths = zip(*group)
        cv = subprocess.run([args.opencv, '1', '0', '1', *paths], capture_output=True, text=True, check=True)
        observed['opencv'].update(parse_run(cv.stdout, names, 1)[0])
        # The NVIDIA replay accepts binary PGM only. Preserve Gray8 samples exactly for
        # manifests stored as PNG or another lossless format.
        with tempfile.TemporaryDirectory(prefix='apriltag-quality-') as temp:
            pgms = []
            for idx, path in enumerate(paths):
                with open(path, 'rb') as source:
                    is_pgm = source.read(2) == b'P5'
                if is_pgm:
                    pgms.append(path)
                    continue
                target = Path(temp) / f'{idx}.pgm'
                with Image.open(path) as source:
                    source.save(target, format='PPM')
                pgms.append(str(target))
            cu = subprocess.run([args.nvidia, 'host', '0', '1', '4', str(args.capacity), *pgms],
                                capture_output=True, text=True, check=True)
        observed['cuapriltags'].update(nvidia_observations(cu.stdout, names, args.capacity))
    return observed


def quality(scenes, observed, scorer, labels):
    names = {s['name'] for s in scenes}
    if any(set(route) != names for route in observed.values()):
        raise ValueError('Observation names differ from this partition')
    if labels == 'unannotated':
        return {'outputs': {route: sum(map(len, frame.values())) for route, frame in observed.items()}}
    required = {(s['name'], i) for s in scenes for i in range(len(s.get('markers', [])))}
    labelled = [{**s, 'markers': s.get('markers', []) + s.get('partial_markers', [])} for s in scenes]
    valid, matches, counts = {}, {}, {}
    for route, output in observed.items():
        matches[route], extras, summary = score(labelled, output, str(scorer))
        valid[route] = required.intersection(matches[route]).difference(summary['id_errors'], summary['rotation_errors'])
        counts[route] = dict(correct=len(valid[route]), unmatched=sum(extras.values()),
                             wrong_id=len(summary['id_errors']), wrong_rotation=len(summary['rotation_errors']),
                             stale=sum(tag['id'] in s.get('must_be_absent_ids', [])
                                       for s in scenes for tag in output[s['name']]))
    # Angle is physical incidence at the camera, independent of pixel radius and lens model.
    by_angle = {}
    episodes = {}
    for scene in scenes:
        for idx, marker in enumerate(scene.get('markers', [])):
            key = (scene['name'], idx)
            angle = marker.get('field_angle_deg')
            if angle is None and 'center_camera_m' in marker:
                x, y, z = marker['center_camera_m']
                angle = math.degrees(math.atan2(math.hypot(x, y), z))
            if angle is not None:
                start = next(t for t in (55, 45, 30, 0) if angle >= t)
                by_angle.setdefault(start, set()).add(key)
            if 'first_visible_frame' in marker:
                episode = (scene['clip'], marker['track'], marker['id'], marker['first_visible_frame'])
                episodes[key] = (episode, scene['frame_index'])

    def first_frames(keys):
        first = {}
        for key in keys & episodes.keys():
            episode, frame = episodes[key]
            first[episode] = min(first.get(episode, frame), frame)
        return first

    comparisons = {}
    for route in ('napa', 'cuapriltags', 'opencv'):
        shared = valid['project'] & valid[route]
        only = valid[route] - valid['project']
        project_error, reference_error = rmse(matches['project'], shared), rmse(matches[route], shared)
        gates = {'retain_each': not only,
                 'corner_rmse': project_error <= reference_error if shared else None,
                 'unmatched': counts['project']['unmatched'] <= counts[route]['unmatched']
                              if labels != 'incomplete' else None}
        for field in ('wrong_id', 'wrong_rotation', 'stale'):
            gates[field] = counts['project'][field] <= counts[route][field]
        comparisons[route] = dict(shared=len(shared), reference_only=sorted(only),
                                  project_only=len(valid['project'] - valid[route]), project_rmse=project_error,
                                  reference_rmse=reference_error, gates=gates)
        if episodes:
            first, other = first_frames(valid['project']), first_frames(valid[route])
            comparisons[route]['later_episodes'] = [k for k, t in sorted(other.items())
                                                    if first.get(k, math.inf) > t]
            gates['first_appearance'] = not comparisons[route]['later_episodes']
        if by_angle:
            comparisons[route]['angle_bins_deg'] = {
                start: dict(required=len(keys), shared=len(shared & keys), reference_only=len(only & keys),
                            project_rmse=rmse(matches['project'], shared & keys),
                            reference_rmse=rmse(matches[route], shared & keys))
                for start, keys in sorted(by_angle.items())}
    return dict(required=len(required), counts=counts, comparisons=comparisons)


def align_origins(observed, label_origin):
    """Express outputs in the manifest's declared pixel origin under the calibrated API settings."""
    if label_origin not in ('pixel-center', 'pixel-corner'):
        raise ValueError('An independently known reference origin is required')
    offsets = {'project': 0., 'napa': 0., 'cuapriltags': 0., 'opencv': -.5}
    if label_origin == 'pixel-corner':
        offsets = {r: d+.5 for r,d in offsets.items()}
    return {route: {name: [{**tag, 'corners': [value+offsets[route] for value in tag['corners']]}
                           for tag in tags] for name,tags in output.items()}
            for route,output in observed.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('paired', type=Path, help='compare_napa_cpu.py output for the same manifest')
    parser.add_argument('--opencv', type=Path, required=True)
    parser.add_argument('--nvidia', type=Path, required=True)
    parser.add_argument('--scorer', type=Path, required=True)
    parser.add_argument('--labels', choices=('complete', 'incomplete', 'expected-empty', 'unannotated'), required=True)
    parser.add_argument('--label-origin', choices=('as-published', 'pixel-center', 'pixel-corner'),
                        default='as-published', help='Add a calibrated-origin score when annotation provenance is known')
    parser.add_argument('--capacity', type=int, default=512)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if not 1 <= args.capacity <= 1024:
        parser.error('cuAprilTags capacity must be 1–1024')
    args.opencv, args.nvidia, args.scorer = [p.resolve() for p in (args.opencv, args.nvidia, args.scorer)]
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        parser.error('output must be empty')
    sha = digest(args.manifest)
    pair = json.loads((args.paired / 'summary.json').read_bytes())
    if pair['manifest_sha256'] != sha:
        raise ValueError('Paired observations refer to another manifest')
    scenes = json.loads(args.manifest.read_bytes())['scenes']
    names = [s['name'] for s in scenes]
    if not names or len(set(names)) != len(names):
        raise ValueError('Missing or duplicate scene names')
    if args.labels in ('complete', 'incomplete') and any('markers' not in s for s in scenes):
        raise ValueError('Missing reference annotations')
    observed = collect(args.manifest, scenes, args)
    for route in ('napa', 'project'):
        observed[route] = parse_run((args.paired / f'{route}-0.txt').read_text(), names, pair['passes'])[0]
    report = dict(candidate_sha256=pair.get('candidate_sha256', pair.get('native_sha256')),
                  manifest_sha256=sha, frames=len(scenes), labels=args.labels,
                  cuapriltags_tile=4, cuapriltags_capacity=args.capacity, nvidia_binary_sha256=digest(args.nvidia),
                  **quality(scenes, observed, args.scorer, args.labels))
    if args.label_origin != 'as-published':
        report['label_origin'] = args.label_origin
        report['origin_aligned'] = quality(scenes, align_origins(observed, args.label_origin),
                                           args.scorer, args.labels)
    if all('camera' in scene for scene in scenes):
        split = {}
        for camera in dict.fromkeys(s['camera'] for s in scenes):
            subset = [s for s in scenes if s['camera'] == camera]
            split[camera] = quality(subset, {r: {s['name']: output[s['name']] for s in subset}
                                            for r, output in observed.items()}, args.scorer, args.labels)
        report['by_camera'] = split
    for route in ('opencv', 'cuapriltags'):
        (args.out / f'{route}.json').write_text(json.dumps(observed[route]))
    (args.out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
