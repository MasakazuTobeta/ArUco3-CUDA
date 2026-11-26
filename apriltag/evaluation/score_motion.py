# SPDX-License-Identifier: Apache-2.0
"""Score fully visible tracks, first detection delay and stale outputs in motion_corpus replays."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmark'))
from compare_napa_cpu import parse_run
from score import rmse, score


def summarize(scenes, observations, scorer):
    # Clipped tags may be returned. Only fully visible tags are required.
    labels = [
        {**scene, 'markers': scene['markers'] + scene.get('partial_markers', [])}
        for scene in scenes
    ]
    matched, unmatched, quality = score(labels, observations, scorer)
    correct = matched.keys() - set(quality['id_errors']) - set(quality['rotation_errors'])
    required = set()
    onsets = {}
    stale = []
    for scene in scenes:
        name = scene['name']
        for index, marker in enumerate(scene['markers']):
            start = marker['first_visible_frame']
            episode = (scene['clip'], marker['track'], marker['id'], start)
            required.add((name, index))
            onsets.setdefault(episode, None)
            if (name, index) in correct and onsets[episode] is None:
                onsets[episode] = scene['frame_index'] - start
        forbidden = set(scene['must_be_absent_ids'])
        for tag in observations[name]:
            if tag['id'] in forbidden:
                stale.append((name, tag['id']))
    delays = [frames for frames in onsets.values() if frames is not None]
    return {
        'confirmed': len(required & correct),
        'required': len(required),
        'partial_confirmed': len(correct - required),
        'corner_rmse_px': rmse(matched, required & correct),
        'unmatched': sum(unmatched.values()),
        'id_errors': quality['id_errors'],
        'rotation_errors': quality['rotation_errors'],
        'stale_outputs': stale,
        'episodes': len(onsets),
        'confirmed_on_entry': delays.count(0),
        'never_confirmed': len(onsets) - len(delays),
        'max_delay_frames': max(delays, default=None),
        'onset_delays': [
            dict(clip=clip, track=track, id=marker_id, first=start, delay=frames)
            for (clip, track, marker_id, start), frames in onsets.items()
        ],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument(
        'replay', type=Path, help='text output from the sequence executable or NAPA worker'
    )
    parser.add_argument('--passes', required=True, type=int)
    parser.add_argument('--scorer', required=True, type=Path)
    args = parser.parse_args()
    data = args.manifest.read_bytes()
    scenes = json.loads(data)['scenes']
    observations, _ = parse_run(args.replay.read_text(), [s['name'] for s in scenes], args.passes)
    result = summarize(scenes, observations, args.scorer)
    result['manifest_sha256'] = hashlib.sha256(data).hexdigest()
    print(json.dumps(result, indent=2))
