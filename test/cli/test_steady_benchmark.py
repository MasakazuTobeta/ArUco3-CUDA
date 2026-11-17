#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reject corrupted synthetic benchmark records without OpenCV or CUDA."""
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class SteadyBenchmarkTest(unittest.TestCase):
    summarizer = Path(__file__).resolve().parents[2] / 'tools/gpu_quality/summarize_steady.py'

    def check_fixture(self, mutate=lambda rows: None, *, cpu=False, accepted=False):
        names = ['first.png', 'second.png']
        metadata = dict(gpu_resize_profile='GENERIC', gpu_resize_backend='GENERIC',
                        gpu_carotene_x_fma='NOT_APPLICABLE', gpu_strategy_identity='test')
        environment = dict(processes=3, iterations=2, warmup=1, expected_new_metadata=metadata,
                           input_manifest={'inputs': [dict(name=n, family='synthetic',
                                                          sha256='a'*64) for n in names]})
        conditions = dict(max_candidates=16384, max_markers=1024, use_aruco3_detection=True,
                          latency_iterations=2, warmup_iterations=1, **metadata)
        statistics = dict(p50_ms=1.0, p95_ms=2.0, p99_ms=2.0)
        commands, records = [], []
        for process in range(3):
            commands.append(dict(key=str(process), variant='cpu1' if cpu else 'new-resident',
                                 mode=1, names=names, candidate_budget=None if cpu else 16384))
            rows = [dict(type='environment', opencv_threads=1, aruco3=True, iterations=2,
                         warmup=1, save_samples=True)]
            for name in names:
                if cpu:
                    for variant in ['opencv', 'optimized']:
                        rows.append(dict(type='measurement', scene=name, variant=variant,
                                         sha256='a'*64, latency_samples_ms=[1.0, 2.0],
                                         detections=4, compatibility_discrepancy='',
                                         max_corner_error_px=0, **statistics))
                else:
                    rows.append(dict(type='measurement', image_path=name, route='CUDA-Resident',
                                     image_sha256='a'*64, memory_mode='M-Device',
                                     conditions=copy.deepcopy(conditions), detection_count=4,
                                     end_to_end_samples_ms=[1.0, 2.0],
                                     end_to_end=copy.deepcopy(statistics)))
            records.append(rows)
        mutate(records)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name, value in [('environment.json', environment), ('commands.json', commands),
                                ('complete.json', dict(complete=True, commands=3,
                                                       binaries_and_inputs_unchanged=True))]:
                (directory/name).write_text(json.dumps(value), encoding='utf-8')
            for command, rows in zip(commands, records):
                (directory/(command['key']+'.jsonl')).write_text(
                    ''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
            output = directory/'summary.csv'
            result = subprocess.run([sys.executable, '-B', str(self.summarizer), str(directory),
                                     '--output', str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode == 0, accepted, result.stderr)
            if accepted:
                with output.open(encoding='utf-8', newline='') as stream:
                    summaries = list(csv.DictReader(stream))
                self.assertEqual(len(summaries), 4 if cpu else 2)
                self.assertTrue(all(row['processes'] == '3' for row in summaries))
                self.assertTrue(all(float(row['p50_ms']) == 1.0 and
                                    float(row['p95_ms']) == float(row['p99_ms']) == 2.0
                                    for row in summaries))
            else:
                self.assertFalse(output.exists(), 'corrupt records must not publish a CSV')

    def test_normal_cpu_and_gpu(self):
        for cpu in [False, True]:
            with self.subTest(cpu=cpu):
                self.check_fixture(cpu=cpu, accepted=True)

    def test_nonfinite_statistics(self):
        for cpu in [False, True]:
            for value in [float('nan'), float('inf')]:
                with self.subTest(cpu=cpu, value=value):
                    self.check_fixture(lambda rows: (rows[0][1] if cpu else
                                       rows[0][1]['end_to_end']).update(p50_ms=value), cpu=cpu)

    def test_dropped_image_and_duplicate_identity(self):
        self.check_fixture(lambda rows: [process.pop() for process in rows])
        self.check_fixture(lambda rows: rows[0].__setitem__(2, copy.deepcopy(rows[0][1])))

    def test_changed_identity_and_conditions(self):
        changes = [('route', 'CUDA-E2E'), ('image_sha256', 'b'*64),
                   ('memory_mode', 'M-Pageable'), ('max_candidates', 4096),
                   ('gpu_resize_profile', 'OPENCV_ARM_HAL')]
        for key, value in changes:
            with self.subTest(field=key):
                self.check_fixture(lambda rows: (rows[0][1]['conditions'] if key in
                                   ['max_candidates', 'gpu_resize_profile'] else
                                   rows[0][1]).update({key: value}))
        self.check_fixture(lambda rows: rows[0][1].update(variant='components'), cpu=True)


if __name__ == '__main__':
    unittest.main()
