# SPDX-License-Identifier: Apache-2.0
"""Check corner origins for the two independent TUM VI AprilGrid reference detectors."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

from score import rmse, score

REFERENCE_OFFSET = {'aprilgrid': 0., 'AprilRobotics': -.5}


def calibrate(reference):
    import cv2
    import numpy as np
    from aprilgrid import Detector
    from dense_corpus import CODEBOOK, grids_from_codebook
    from video_corpus import CORNER_ORDER, read_probe

    grids = grids_from_codebook(CODEBOOK)
    board = Detector('t36h11')
    residuals = {source: [] for source in REFERENCE_OFFSET}
    with tempfile.TemporaryDirectory() as folder:
        paths, physical = [], []
        for marker_id in range(32):
            cell = (3, 4, 6, 8)[marker_id // 8]
            rotation = marker_id % 4
            left, top = 23 + marker_id % 7, 28 + marker_id % 5
            image = np.full((192, 256), 229, np.uint8)
            cells = np.pad(grids[marker_id], 1)
            marker = np.rot90(np.where(cells > 0, 229, 14).astype(np.uint8), -rotation)
            side = 10 * cell
            image[top:top + side, left:left + side] = np.repeat(np.repeat(marker, cell, 0), cell, 1)
            path = Path(folder) / f'{marker_id}.pgm'
            if not cv2.imwrite(str(path), image):
                raise OSError(path)
            paths.append(path)
            corners = np.array([[left, top], [left + side, top], [left + side, top + side],
                                [left, top + side]], dtype=np.float64) - .5
            physical.append(np.roll(corners, -rotation, axis=0))
        probes = read_probe(reference, paths, kalibr_border=True)
        for i, (path, tags, truth) in enumerate(zip(paths, probes, physical, strict=True)):
            other = board.detect(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))
            candidates = {'AprilRobotics': [points for tag, _, _, points in tags if tag == i],
                          'aprilgrid': [tag.corners for tag in other if tag.tag_id == i]}
            for source, found in candidates.items():
                if len(found) != 1:
                    raise ValueError(f'{source} expected one tag {i}, received {len(found)}')
                q = np.asarray(found[0]).reshape(4, 2)[CORNER_ORDER]
                residuals[source].extend(q - truth)
    result = {}
    for source, values in residuals.items():
        error = np.asarray(values)
        result[source] = {'prints': len(values) // 4, 'mean_signed_xy_px': error.mean(axis=0).tolist(),
                          'pixel_center_rmse_px': float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
                          'pixel_corner_rmse_px': float(np.sqrt(np.mean(np.sum((error - .5)**2, axis=1))))}
    return result


def evaluate(manifest, paired, four_way, scorer):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmark'))
    from compare_napa_cpu import parse_run
    from compare_video_quality import align_origins

    data = manifest.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    corpus = json.loads(data)
    if corpus.get('dataset') != 'TUM VI, Schubert et al. (IROS 2018)' or corpus.get('marker_border_bits') != 2:
        raise ValueError('expected a TUM VI two-border labelled manifest')
    scenes = corpus['scenes']
    pair = json.loads((paired / 'summary.json').read_bytes())
    four = json.loads((four_way / 'summary.json').read_bytes())
    if pair['manifest_sha256'] != digest or four['manifest_sha256'] != digest:
        raise ValueError('replays differ from the input manifest')
    candidate = pair.get('candidate_sha256', pair.get('native_sha256'))
    if not candidate or four.get('candidate_sha256') != candidate:
        raise ValueError('paired and four-way results have different candidates')
    names = [s['name'] for s in scenes]
    raw = {route: parse_run((paired / f'{route}-0.txt').read_text(), names, pair['passes'])[0]
           for route in ('napa', 'project')}
    raw.update({route: json.loads((four_way / f'{route}.json').read_bytes())
                for route in ('opencv', 'cuapriltags')})
    if any(set(frames) != set(names) for frames in raw.values()):
        raise ValueError('incomplete replay frames')
    normalized = []
    by_source = {name: set() for name in REFERENCE_OFFSET}
    for scene in scenes:
        markers = []
        for i, marker in enumerate(scene['markers']):
            source = marker['reference_detector']
            by_source[source].add((scene['name'], i))
            offset = REFERENCE_OFFSET[source]
            markers.append({**marker, 'corners_px': [x + offset for x in marker['corners_px']]})
        normalized.append({**scene, 'markers': markers})
    result = {'candidate_sha256': candidate, 'manifest_sha256': digest,
              'reference_counts': {k: len(v) for k, v in by_source.items()}}
    for mode, labels, readings in [('as_published', scenes, raw),
                                   ('source_pixel_center', normalized, align_origins(raw, 'pixel-center'))]:
        result[mode] = {}
        for route, detected in readings.items():
            matches, extras, summary = score(labels, detected, scorer)
            correct = set(matches).difference(summary['id_errors'], summary['rotation_errors'])
            result[mode][route] = {
                'correct': len(correct), 'unmatched': sum(extras.values()),
                'wrong_id': len(summary['id_errors']), 'wrong_rotation': len(summary['rotation_errors']),
                'corner_rmse_px': rmse(matches, correct),
                'by_reference_source': {
                    source: {'correct': len(correct & keys), 'corner_rmse_px': rmse(matches, correct & keys)}
                    for source, keys in by_source.items()}}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('calibrate')
    c.add_argument('reference', type=Path, help='video_reference.c probe with Kalibr support')
    s = sub.add_parser('score')
    s.add_argument('manifest', type=Path)
    s.add_argument('paired', type=Path)
    s.add_argument('four_way', type=Path)
    s.add_argument('--scorer', type=Path, required=True)
    args = p.parse_args()
    result = (calibrate(args.reference) if args.command == 'calibrate' else
              evaluate(args.manifest, args.paired, args.four_way, args.scorer.resolve()))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
