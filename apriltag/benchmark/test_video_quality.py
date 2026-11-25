# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import compare_video_quality as replay


class QualityReplayTest(unittest.TestCase):
    def test_normalized_nvidia_corners_and_full_capacity(self):
        row = dict(frame=0, cycle=0, backend='nvidia',
                   tags=[dict(id=23, corners=[[3, 4], [5, 6], [7, 8], [9, 10]])])
        self.assertEqual(replay.nvidia_observations(json.dumps(row), ['f'], 2)['f'],
                         [dict(id=23, corners=[7, 8, 9, 10, 3, 4, 5, 6])])
        with self.assertRaises(ValueError):
            replay.nvidia_observations(json.dumps(row), ['f'], 1)

    def test_explicit_pixel_origin(self):
        routes = {k: {'f': [dict(id=7, corners=[10.]*8)]}
                  for k in ('project', 'napa', 'cuapriltags', 'opencv')}
        centered = replay.align_origins(routes, 'pixel-center')
        self.assertEqual(centered['opencv']['f'][0]['corners'], [9.5]*8)
        self.assertEqual(centered['project']['f'][0], routes['project']['f'][0])
        corner = replay.align_origins(routes, 'pixel-corner')
        self.assertEqual(corner['opencv']['f'][0], routes['opencv']['f'][0])
        self.assertEqual(corner['napa']['f'][0]['corners'], [10.5]*8)
        with self.assertRaises(ValueError):
            replay.align_origins(routes, 'unknown')

    def test_lossless_png_to_nvidia_pgm(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pixels = bytes((0, 128, 255, 10, 13, 92))
            path = root / 'frame.png'
            Image.frombytes('L', (3, 2), pixels).save(path)
            scene = dict(name='f', path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())

            def runner(command, **kwargs):
                if command[0] == 'official-opencv':
                    self.assertEqual(Path(command[-1]), path)
                    return SimpleNamespace(stdout='ms 0 .123')
                pgm = Path(command[-1])
                self.assertTrue(pgm.read_bytes().startswith(b'P5'))
                with Image.open(pgm) as image:
                    self.assertEqual(image.tobytes(), pixels)
                return SimpleNamespace(stdout=json.dumps(dict(frame=0, cycle=0, backend='nvidia', tags=[])))

            args = SimpleNamespace(opencv='official-opencv', nvidia='official-cu', capacity=512)
            with patch.object(replay.subprocess, 'run', runner):
                self.assertEqual(replay.collect(root / 'manifest.json', [scene], args),
                                 {'opencv': {'f': []}, 'cuapriltags': {'f': []}})
                with self.assertRaises(ValueError):
                    replay.collect(root / 'manifest.json', [{**scene, 'sha256': 'changed'}], args)

    def test_only_correctly_oriented_required_locations_count(self):
        scenes = [dict(name='f', markers=[dict(id=3, field_angle_deg=60)],
                       partial_markers=[dict(id=5)])]
        observed = {r: {'f': []} for r in ('project', 'napa', 'opencv', 'cuapriltags')}
        # All routes match the partial. The project has the wrong rotation on the required tag.
        def fake_score(labelled, samples, scorer):
            self.assertEqual(len(labelled[0]['markers']), 2)
            return ({('f', 0): (1, 1), ('f', 1): (0, 0)}, {'f': 1},
                    dict(id_errors=[], rotation_errors=[('f', 0)] if samples is observed['project'] else []))

        with patch.object(replay, 'score', fake_score):
            q = replay.quality(scenes, observed, 'scorer', 'incomplete')
        self.assertEqual(q['required'], 1)
        self.assertEqual(q['counts']['project']['correct'], 0)
        for result in q['comparisons'].values():
            self.assertEqual(result['reference_only'], [('f', 0)])
            self.assertEqual(result['angle_bins_deg'][55]['reference_only'], 1)
            self.assertIsNone(result['project_rmse'])
            self.assertIsNone(result['gates']['unmatched'])
            self.assertFalse(result['gates']['wrong_rotation'])


if __name__ == '__main__':
    unittest.main()
