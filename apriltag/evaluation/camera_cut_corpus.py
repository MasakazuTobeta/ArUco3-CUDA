# SPDX-License-Identifier: Apache-2.0
"""Interleave the blur, read-noise and grooves sequences at native frame boundaries."""
import argparse
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(output, sources):
    manifest = dict(scenes=[], sources={key: digest(path) for key, path in sources.items()})
    streams = [json.loads(path.read_bytes())['scenes'] for path in sources.values()]
    active = {key: {} for key in sources}
    (output / 'images').mkdir(parents=True, exist_ok=True)
    for cycle in zip_longest(*streams):
        for (key, source), scene in zip(sources.items(), cycle, strict=True):
            if scene is None:
                continue
            annotated, current = [], {}
            for marker in scene['markers']:
                track = scene['clip'], marker['track'], marker['id']
                first = active[key].get(track, marker.get('first_visible_frame', scene['frame_index']))
                current[track] = first
                annotated.append({**marker, 'first_visible_frame': first})
            active[key] = current
            frame = (source.parent / scene['path']).resolve()
            with frame.open('rb') as stream:
                if (stream.readline().strip(), stream.readline().strip(),
                    stream.readline().strip()) != (b'P5', b'1280 720', b'255'):
                    raise ValueError(f'Expected a 1280×720 Gray8 PGM in {frame}')
            if digest(frame) != scene['sha256']:
                raise ValueError(f'Frame SHA-256 changed in {frame}')
            name = f"{key}-{scene['name']}"
            link = output / 'images' / f'{name}.pgm'
            if link.exists():
                if link.resolve() != frame:
                    raise ValueError(f'{link} already points to another frame')
            else:
                link.symlink_to(os.path.relpath(frame, link.parent))
            manifest['scenes'].append({**scene, 'markers': annotated, 'name': name,
                                       'clip': f"{key}-{scene['clip']}", 'camera': key,
                                       'path': str(link.relative_to(output))})
    path = output / 'manifest.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'{path}  {digest(path)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('blur', type=Path)
    parser.add_argument('read_noise', type=Path)
    parser.add_argument('grooves', type=Path)
    args = parser.parse_args()
    generate(args.output, {'blur': args.blur, 'read': args.read_noise, 'groove': args.grooves})
