#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the frozen real-background evaluation from the repository root/container."""
import argparse
import hashlib
import json
import pathlib
import subprocess

ROOT = pathlib.Path('docs/measurements/2026-11-17-cpu-real-background')
BINARY = 'build/cpu-opt/cpu/aruco3cuda_cpu_benchmark'
EXPECTED_BINARY = '99581bfab75461f5e11bb9687966b2ba8d96248156dd5abd69f8a9209fc6bdcb'


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def verify(directory, inputs, positive=False):
    environment = json.loads((directory / 'environment.json').read_text())
    assert environment['binary_sha256'] == EXPECTED_BINARY
    paths = sorted(directory.glob('run-*.jsonl'))
    assert len(paths) == environment['independent_processes']
    counts = {}
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        samples = [row for row in rows if row['type'] == 'measurement']
        assert len(samples) == 2 * len(inputs), (path, len(samples))
        seen = set()
        for row in samples:
            key = (row['scene'], row['variant'])
            assert key not in seen
            seen.add(key)
            assert row['sha256'] == inputs[row['scene']]
            assert not row['compatibility_discrepancy'] and row['max_corner_error_px'] == 0
            assert row['detections'] == row['reference_detections']
            assert row['rejected'] == row['reference_rejected']
            assert row['detections'] >= 0
            if positive:
                assert row['detections'] == 4
            previous = counts.setdefault(row['scene'], row['detections'])
            assert row['detections'] == previous, (path, row['scene'], 'varying detections')
        assert seen == {(name, route) for name in inputs for route in ['opencv', 'optimized']}
    return counts


def run(name, input_dir, threads, aruco3, iterations, warmup, runs, inputs, scene_filter='', positive=False):
    assert sha(BINARY) == EXPECTED_BINARY
    for name_in, digest in inputs.items():
        assert sha(pathlib.Path(input_dir) / name_in) == digest
    destination = ROOT / name
    assert not destination.exists(), destination
    command = ['python3', 'cpu/run_experiments.py', '--binary', BINARY,
               '--input-dir', str(input_dir), '--output-dir', str(destination),
               '--cpu-list', '5' if threads == 1 else '5,6,7,8', '--threads', str(threads),
               '--aruco3', str(aruco3), '--runs', str(runs), '--iterations', str(iterations),
               '--warmup', str(warmup), '--dictionary', 'DICT_ARUCO_MIP_36h12',
               '--variants', 'opencv,optimized', '--filter', scene_filter,
               '--background-note', 'No other agent build/test/detector process; unrelated host services and power policy retained']
    print('START', name, flush=True)
    subprocess.run(command, check=True)
    counts = verify(destination, inputs, positive)
    (destination / 'validation.json').write_text(json.dumps({
        'exact_comparisons': True, 'stable_detection_count_all_runs_and_routes': True,
        'detections_per_scene': counts,
        'zero_detection_scenes': [name_in for name_in, count in sorted(counts.items()) if count == 0],
        'scope': 'Compatibility/eligibility only' if runs == 1 else 'Repeated-process detector latency'
    }, indent=2) + '\n')
    if runs > 1:
        with (destination / 'summary.json').open('x') as output:
            subprocess.run(['python3', 'cpu/summarize_experiments.py', str(destination)], stdout=output, check=True)
    print('FINISHED', name, 'zero detections', sum(count == 0 for count in counts.values()), '/', len(counts), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=pathlib.Path, required=True)
    parser.add_argument('--expected-scenes', type=int, default=16)
    args = parser.parse_args()
    inputs = {path.name: sha(path) for path in sorted(args.input_dir.iterdir())
              if path.suffix.lower() in {'.png', '.jpg', '.jpeg'}}
    assert len(inputs) == args.expected_scenes
    groups = [f'{mode}-{threads}t' for mode in ['aruco3', 'fullres'] for threads in [1, 4]]
    for name in groups + [f'{kind}-{aruco3}' for kind in ['eligibility', 'positive-control'] for aruco3 in [1, 0]]:
        assert not (ROOT / name).exists(), name
    (ROOT / 'input-hashes.json').write_text(json.dumps(inputs, indent=2) + '\n')
    control_dir = pathlib.Path('data/cpu-experiment')
    control_name = 'cpu_1280x720_n4_clean.png'
    controls = {control_name: sha(control_dir / control_name)}
    for aruco3 in [1, 0]:
        run(f'positive-control-{aruco3}', control_dir, 1, aruco3, 1, 0, 1,
            controls, scene_filter='cpu_1280x720_n4_clean', positive=True)
        run(f'eligibility-{aruco3}', args.input_dir, 4, aruco3, 1, 0, 1, inputs)
    for mode, aruco3, iterations, warmup in [('aruco3', 1, 200, 30), ('fullres', 0, 50, 10)]:
        for threads in [1, 4]:
            run(f'{mode}-{threads}t', args.input_dir, threads, aruco3, iterations, warmup, 3, inputs)
    print('All planned measurement groups passed', flush=True)


if __name__ == '__main__':
    main()
