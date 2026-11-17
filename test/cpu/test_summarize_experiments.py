#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reject incomplete or incomparable measurement batches before publishing a summary."""
import copy
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'cpu' / 'summarize_experiments.py'


class SummarizeExperimentsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = pathlib.Path(self.temporary.name)
        (self.directory / 'environment.json').write_text(json.dumps({'independent_processes': 3}))
        self.configuration = {'type': 'environment', 'opencv_version': '4.14.0',
                              'opencv_threads': 1, 'iterations': 200, 'warmup': 30,
                              'aruco3': True, 'dictionary': 'DICT_ARUCO_MIP_36h12'}
        self.rows = [self.configuration]
        for route, latency in [('opencv', 2.0), ('optimized', 1.0)]:
            self.rows.append({'type': 'measurement', 'scene': 'scene.png', 'variant': route,
                             'sha256': 'frozen-image', 'width': 32, 'height': 24,
                             'detections': 0, 'reference_detections': 0,
                             'rejected': 1, 'reference_rejected': 1,
                             'compatibility_discrepancy': '', 'max_corner_error_px': 0,
                             'p50_ms': latency, 'p95_ms': latency + 0.1,
                             'p99_ms': latency + 0.2})
        for index in range(1, 4):
            self.write_run(index, self.rows)

    def write_run(self, index, rows):
        (self.directory / ('run-%d.jsonl' % index)).write_text(
            ''.join(json.dumps(row) + '\n' for row in rows))

    def run_summary(self):
        return subprocess.run([sys.executable, str(SCRIPT), str(self.directory)],
                              capture_output=True, text=True, timeout=10)

    def assert_rejected(self):
        result = self.run_summary()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout, '', 'Invalid input must not publish a JSON summary')

    def test_valid_batch_preserves_percentiles_and_ratio(self):
        result = self.run_summary()
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual(summary['all_scenes']['median_speedup'], 2.0)
        self.assertEqual(summary['scenes'][0]['candidate_p99_ms'], 1.2)

    def test_empty_runs_are_not_successful_experiments(self):
        for index in range(1, 4):
            self.write_run(index, [])
        self.assert_rejected()

    def test_headers_without_measurements_are_rejected(self):
        for index in range(1, 4):
            self.write_run(index, [self.configuration])
        self.assert_rejected()

    def test_mixed_detector_settings_are_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['opencv_threads'] = 4
        self.write_run(2, rows)
        self.assert_rejected()

    def test_missing_or_duplicate_configuration_is_rejected(self):
        for rows in [self.rows[1:], [self.configuration] + self.rows]:
            with self.subTest(rows=len(rows)):
                self.write_run(2, rows)
                self.assert_rejected()

    def test_invalid_latency_is_rejected(self):
        for latency in [0.0, -1.0, float('nan'), float('inf'), 9.0, True]:
            with self.subTest(latency=latency):
                rows = copy.deepcopy(self.rows)
                rows[2]['p50_ms'] = latency
                self.write_run(2, rows)
                self.assert_rejected()

    def test_changing_outputs_for_the_same_image_are_rejected(self):
        for key in ['detections', 'reference_detections', 'rejected', 'width']:
            with self.subTest(key=key):
                rows = copy.deepcopy(self.rows)
                rows[2][key] += 1
                self.write_run(2, rows)
                self.assert_rejected()

    def test_missing_route_or_duplicate_measurement_is_rejected(self):
        for rows in [self.rows[:-1], self.rows + [self.rows[-1]]]:
            with self.subTest(rows=len(rows)):
                self.write_run(2, rows)
                self.assert_rejected()

    def test_routes_must_record_the_same_image_dimensions(self):
        rows = copy.deepcopy(self.rows)
        rows[2]['width'] += 1
        for index in range(1, 4):
            self.write_run(index, rows)
        self.assert_rejected()

    def test_image_dimensions_are_required(self):
        for width in [None, 0, -1, True]:
            with self.subTest(width=width):
                rows = copy.deepcopy(self.rows)
                for row in rows[1:]:
                    row['width'] = width
                for index in range(1, 4):
                    self.write_run(index, rows)
                self.assert_rejected()

    def test_previous_route_requires_matching_image_dimensions(self):
        for width in [32, 33]:
            with self.subTest(width=width):
                rows = copy.deepcopy(self.rows)
                previous = dict(rows[-1], variant='previous', width=width)
                for index in range(1, 4):
                    self.write_run(index, rows + [previous])
                if width == 32:
                    result = self.run_summary()
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(json.loads(result.stdout)['scenes'][0]['speedup_over_previous'], 1.0)
                else:
                    self.assert_rejected()


if __name__ == '__main__':
    unittest.main()
