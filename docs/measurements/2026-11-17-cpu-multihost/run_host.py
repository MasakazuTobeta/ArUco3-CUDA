#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the frozen industrial-photo CPU comparison without build-tool dependencies."""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def read(path):
    try:
        return pathlib.Path(path).read_text().strip()
    except OSError:
        return 'unavailable'


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True, type=pathlib.Path)
    parser.add_argument('--output-dir', required=True, type=pathlib.Path)
    parser.add_argument('--cpu-one', required=True, type=int)
    parser.add_argument('--cpu-four', required=True)
    parser.add_argument('--manifest', default='snapshot.json', type=pathlib.Path)
    parser.add_argument('--provenance', default='/opt/opencv/share/aruco3cuda/opencv-provenance.json')
    parser.add_argument('--background-note', default='Existing host services and power policy retained')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    for path, digest in manifest['files'].items():
        if sha(path) != digest:
            raise RuntimeError('Snapshot differs: ' + path)
    binary = str(args.binary.resolve())
    binary_sha = sha(binary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save(args.output_dir / 'snapshot.json', manifest)
    save(args.output_dir / 'input-hashes.json', manifest['inputs'])
    available = os.sched_getaffinity(0)
    one = {args.cpu_one}
    four = {int(c) for c in args.cpu_four.split(',')}
    if len(four) != 4 or not (one | four) <= available:
        raise RuntimeError('Invalid CPU affinity')

    def run(name, threads, aruco3, iterations, warmup, runs, positive=False):
        destination = args.output_dir / name
        destination.mkdir(exist_ok=False)
        cpus = one if threads == 1 else four
        os.sched_setaffinity(0, cpus)
        inputs = manifest['controls'] if positive else manifest['inputs']
        input_dir = 'data/cpu-experiment' if positive else 'data/cpu-real-background'
        command = [binary, '--input-dir', input_dir, '--iterations', str(iterations),
                   '--warmup', str(warmup), '--threads', str(threads), '--aruco3', str(aruco3),
                   '--variants', 'opencv,optimized', '--dictionary', 'DICT_ARUCO_MIP_36h12',
                   '--throughput-frames', '0']
        if positive:
            command += ['--filter', 'cpu_1280x720_n4_clean']
        environment = {
            'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'command': command, 'independent_processes': runs, 'cpu_affinity': sorted(cpus),
            'platform': platform.uname()._asdict(), 'binary_sha256': binary_sha,
            'git_head': manifest['git_head'], 'snapshot_sha256': sha(args.manifest),
            'source_sha256': manifest['sources'], 'opencv_provenance': read(args.provenance),
            'cpuinfo': read('/proc/cpuinfo'), 'load_average_at_start': read('/proc/loadavg'),
            'power_mode': 'Host configuration retained; no power or clock settings changed',
            'background_load': args.background_note,
            'measurement': 'Single-frame host grayscale input to host results; no decoding/setup/warmup',
            'kernel_time': 'Not applicable: CPU only',
            'cpu_clocks_khz': {str(c): {k: read('/sys/devices/system/cpu/cpu%d/cpufreq/%s' % (c, k))
                for k in ['scaling_governor', 'scaling_cur_freq', 'scaling_max_freq', 'cpuinfo_max_freq']}
                for c in sorted(cpus)},
        }
        save(destination / 'environment.json', environment)
        print('START', name, flush=True)
        counts = {}
        for index in range(1, runs + 1):
            if sha(binary) != binary_sha:
                raise RuntimeError('Binary changed')
            path = destination / ('run-%d.jsonl' % index)
            with path.open('x') as output, (destination / ('run-%d.stderr.log' % index)).open('x') as errors:
                subprocess.run(command, stdout=output, stderr=errors, check=True)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            metadata = [row for row in rows if row['type'] == 'environment']
            assert len(metadata) == 1
            meta = metadata[0]
            assert meta['opencv_version'] == '4.14.0'
            assert meta['opencv_threads'] == threads and meta['iterations'] == iterations
            assert meta['warmup'] == warmup and meta['aruco3'] == bool(aruco3)
            assert not meta['profile_stages'] and meta['throughput_frames'] == 0
            measurements = [row for row in rows if row['type'] == 'measurement']
            assert len(measurements) == 2 * len(inputs)
            seen = set()
            for row in measurements:
                key = (row['scene'], row['variant'])
                assert key not in seen
                seen.add(key)
                assert row['sha256'] == inputs[row['scene']]
                assert not row['compatibility_discrepancy'] and row['max_corner_error_px'] == 0
                assert row['detections'] == row['reference_detections'] == (4 if positive else 0)
                assert row['rejected'] == row['reference_rejected']
                counts[row['scene']] = row['detections']
            assert seen == {(name_in, route) for name_in in inputs for route in ['opencv', 'optimized']}
            print('FINISHED', name, index, flush=True)
        save(destination / 'validation.json', {'exact_comparisons': True,
             'detections_per_scene': counts, 'all_processes_checked': runs,
             'scope': 'Initial untimed and final retained output; intermediate timed outputs not retained'})
        if runs > 1:
            with (destination / 'summary.json').open('x') as output:
                subprocess.run([sys.executable, 'cpu/summarize_experiments.py', str(destination)],
                               stdout=output, check=True)

    for aruco3 in [1, 0]:
        for threads in [1, 4]:
            run('positive-control-%d-%dt' % (aruco3, threads), threads, aruco3, 1, 0, 1, True)
        run('eligibility-%d' % aruco3, 4, aruco3, 1, 0, 1)
    for mode, aruco3, iterations, warmup in [('aruco3', 1, 200, 30), ('fullres', 0, 50, 10)]:
        for threads in [1, 4]:
            run('%s-%dt' % (mode, threads), threads, aruco3, iterations, warmup, 3)
    save(args.output_dir / 'completion.json', {'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'binary_sha256': binary_sha, 'groups': 10, 'timing_groups': 4, 'timing_rows': 384})
    print('All planned groups passed', flush=True)


if __name__ == '__main__':
    main()
