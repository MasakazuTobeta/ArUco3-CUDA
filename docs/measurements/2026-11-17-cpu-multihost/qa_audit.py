#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independently audit raw multihost records without running detector workloads."""
import argparse
import collections
import csv
import datetime
import hashlib
import json
import math
import pathlib
import statistics


def read(path):
    return json.loads(path.read_text())


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), (actual, expected)


def audit_group(directory, manifest, source):
    name = directory.name
    mode, threads = name.rsplit('-', 1)
    threads = int(threads[:-1])
    aruco3 = mode == 'aruco3'
    iterations, warmup = (200, 30) if aruco3 else (50, 10)
    environment = read(directory / 'environment.json')
    assert environment['snapshot_sha256'] == hashlib.sha256((directory.parent / 'snapshot.json').read_bytes()).hexdigest()
    assert environment['independent_processes'] == 3
    assert len(environment['cpu_affinity']) == threads
    provenance = json.loads(environment['opencv_provenance'])
    assert provenance['commit'] == '0654a42e19215ef25b1d367d822f3c630447e7c7'
    assert provenance['version'] == '4.14.0' and provenance['modified'] is False
    assert provenance['with_cuda'] == 'OFF'
    recorded_source = environment['source_sha256']
    for path, digest in source.items():
        assert recorded_source.get(path) == digest, (directory, 'source', path)
    paths = sorted(directory.glob('run-*.jsonl'))
    assert len(paths) == 3
    grouped = collections.defaultdict(lambda: collections.defaultdict(list))
    process_ratios = []
    rejected = collections.defaultdict(set)
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        configuration = [row for row in rows if row['type'] == 'environment']
        assert len(configuration) == 1
        cfg = configuration[0]
        expected_cfg = {'opencv_version': '4.14.0', 'opencv_threads': threads,
                        'aruco3': aruco3, 'iterations': iterations, 'warmup': warmup,
                        'profile_stages': False, 'throughput_frames': 0,
                        'dictionary': 'DICT_ARUCO_MIP_36h12'}
        for key, value in expected_cfg.items():
            assert cfg[key] == value, (path, key, cfg[key], value)
        assert cfg['min_side_length_canonical_img_px'] == 32
        assert cfg['min_marker_length_ratio_original_img'] == 0.050000001
        assert (cfg['threshold_window_min_px'], cfg['threshold_window_max_px'],
                cfg['threshold_window_step_px']) == (3, 23, 10)
        assert cfg['error_correction_rate'] == 0.6
        assert '-O3' in cfg['cpu_compile_flags'] and '-DNDEBUG' in cfg['cpu_compile_flags']
        assert 'native' not in cfg['cpu_compile_flags']
        samples = [row for row in rows if row['type'] == 'measurement']
        assert len(samples) == 32
        seen = {}
        for row in samples:
            key = row['scene'], row['variant']
            assert key not in seen, (path, key)
            seen[key] = row
            item = manifest[row['scene']]
            assert row['sha256'] == item['prepared_sha256']
            assert (row['width'], row['height']) == (item['width'], item['height'])
            assert row['detections'] == row['reference_detections'] == 0
            assert row['rejected'] == row['reference_rejected']
            assert row['compatibility_discrepancy'] == '' and row['max_corner_error_px'] == 0
            assert 0 < row['p50_ms'] <= row['p95_ms'] <= row['p99_ms']
            rejected[row['scene']].add(row['rejected'])
            grouped[row['scene']][row['variant']].append(row)
        assert set(seen) == {(scene, variant) for scene in manifest for variant in ['opencv', 'optimized']}
        for scene in manifest:
            process_ratios.append(seen[scene, 'opencv']['p50_ms'] / seen[scene, 'optimized']['p50_ms'])
    assert all(len(values) == 1 for values in rejected.values()), (directory, 'unstable rejected counts')
    summary = read(directory / 'summary.json')
    saved = {row['scene']: row for row in summary['scenes']}
    assert set(saved) == set(manifest)
    ratios = []
    losses = {p: [] for p in ['p50', 'p95', 'p99']}
    spreads = []
    for scene, variants in sorted(grouped.items()):
        medians = {}
        for variant in ['opencv', 'optimized']:
            for percentile in ['p50', 'p95', 'p99']:
                median = statistics.median(row[percentile + '_ms'] for row in variants[variant])
                medians[variant, percentile] = median
                prefix = 'opencv' if variant == 'opencv' else 'candidate'
                close(saved[scene][prefix + '_' + percentile + '_ms'], median)
        ratio = medians['opencv', 'p50'] / medians['optimized', 'p50']
        close(saved[scene]['speedup'], ratio)
        ratios.append(ratio)
        for percentile in losses:
            ratio_p = medians['opencv', percentile] / medians['optimized', percentile]
            if ratio_p <= 1:
                losses[percentile].append({'scene': scene, 'speedup': ratio_p})
        values = [row['p50_ms'] for row in variants['optimized']]
        spread = 100 * (max(values) - min(values)) / statistics.median(values)
        close(saved[scene]['candidate_run_spread_pct'], spread)
        spreads.append(spread)
    result = {'timed_rows': 96, 'scenes': 16, 'processes': 3,
              'binary_sha256': environment['binary_sha256'],
              'p50_wins': sum(ratio > 1 for ratio in ratios),
              'p50_speedup_min': min(ratios), 'p50_speedup_median': statistics.median(ratios),
              'p50_speedup_max': max(ratios), 'losses_or_ties': losses,
              'individual_process_p50_wins': sum(ratio > 1 for ratio in process_ratios),
              'individual_process_comparisons': len(process_ratios),
              'individual_process_min_speedup': min(process_ratios),
              'candidate_spread_median_pct': statistics.median(spreads),
              'candidate_spread_max_pct': max(spreads),
              'source_files_checked': len(source), 'all_zero_detections': True,
              'stable_rejected_counts': True, 'inputs_exact': True}
    all_scenes = summary['all_scenes']
    assert all_scenes['scenes'] == 16 and all_scenes['wins'] == result['p50_wins']
    close(all_scenes['median_speedup'], result['p50_speedup_median'])
    return result


def main():
    if not __debug__:
        raise RuntimeError('Validation requires Python assertions; disable -O/-OO and PYTHONOPTIMIZE')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('hosts', nargs='+', type=pathlib.Path)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    manifest = {row['name']: row for row in read(pathlib.Path('cpu/experiments/real_background_manifest.json'))['files']}
    source = read(pathlib.Path(__file__).with_name('qa-source-hashes.json'))
    result = {'status': 'passed', 'audited_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'method': 'Independent raw-row and summary audit; no detector/build execution.', 'hosts': {}}
    x86_bundles = {}
    for host in args.hosts:
        snapshot = read(host / 'snapshot.json')
        assert snapshot['git_head'] == 'c5c943232da465a5b0e89aec046787d8144dbab9'
        assert snapshot['inputs'] == {name: row['prepared_sha256'] for name, row in manifest.items()}
        for path, digest in source.items():
            assert snapshot['files'][path] == snapshot['sources'][path] == digest
        groups = {}
        for mode in ['aruco3', 'fullres']:
            for threads in [1, 4]:
                name = f'{mode}-{threads}t'
                groups[name] = audit_group(host / name, manifest, source)
        assert len({group['binary_sha256'] for group in groups.values()}) == 1
        completion = read(host / 'completion.json')
        assert completion['timing_groups'] == 4 and completion['timing_rows'] == 384
        assert completion['binary_sha256'] == groups['aruco3-1t']['binary_sha256']
        if host.name in {'20.41', '20.44', '10.26'}:
            checksums = {}
            for line in (host / 'portable-checksums.txt').read_text().splitlines():
                digest, path = line.split(maxsplit=1)
                checksums[path] = digest
            log_name = 'portable-checksums-verified.log' if host.name == '10.26' else 'portable-verification.log'
            verified = (host / log_name).read_text().splitlines()
            assert set(verified) == {path + ': OK' for path in checksums}
            assert checksums['portable/bin/aruco3cuda_cpu_benchmark'] == completion['binary_sha256']
            x86_bundles[host.name] = {path: digest for path, digest in checksums.items()
                                    if path.startswith('portable/lib/') or path == 'portable/bin/aruco3cuda_cpu_benchmark'}
            runtime = read(host / 'runtime-library-hashes.json')
            opencv_bundled = {pathlib.Path(path).name: digest for path, digest in checksums.items()
                              if path.startswith('portable/lib/libopencv_')}
            opencv_runtime = {name: entry['sha256'] for name, entry in runtime.items()
                              if name.startswith('libopencv_')}
            assert opencv_bundled == opencv_runtime
        controls = []
        for path in sorted(host.glob('positive*/run-*.jsonl')):
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            cfg = next(row for row in rows if row['type'] == 'environment')
            samples = [row for row in rows if row['type'] == 'measurement']
            assert len(samples) == 2
            assert {row['variant'] for row in samples} == {'opencv', 'optimized'}
            for row in samples:
                assert row['sha256'] == snapshot['controls'][row['scene']]
                assert row['detections'] == row['reference_detections'] == 4
                assert row['rejected'] == row['reference_rejected']
                assert not row['compatibility_discrepancy'] and row['max_corner_error_px'] == 0
            controls.append((cfg['aruco3'], cfg['opencv_threads']))
        assert set(controls) == {(a, t) for a in [False, True] for t in [1, 4]}, (host, controls)
        for aruco3 in [0, 1]:
            path = host / f'eligibility-{aruco3}' / 'run-1.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            cfg = next(row for row in rows if row['type'] == 'environment')
            assert cfg['aruco3'] == bool(aruco3) and cfg['iterations'] == 1 and cfg['warmup'] == 0
            samples = [row for row in rows if row['type'] == 'measurement']
            assert len(samples) == 32
            assert {(row['scene'], row['variant']) for row in samples} == {(name, route) for name in manifest for route in ['opencv', 'optimized']}
            for row in samples:
                assert row['sha256'] == manifest[row['scene']]['prepared_sha256']
                assert row['detections'] == row['reference_detections'] == 0
                assert row['rejected'] == row['reference_rejected']
                assert not row['compatibility_discrepancy'] and row['max_corner_error_px'] == 0
        result['hosts'][str(host)] = {'groups': groups, 'positive_controls': controls}
    if x86_bundles:
        first = next(iter(x86_bundles.values()))
        assert all(bundle == first for bundle in x86_bundles.values())
        result['x86_bundle_audit'] = {
            'hosts_checked': sorted(x86_bundles), 'same_binary_and_bundled_libraries': True,
            'resolved_runtime_opencv_libraries_match_bundle': True,
            'files_checked_per_host': len(first),
            'opencv_libraries_checked': sorted(path for path in first if '/libopencv_' in path),
            'limitation': 'Host/container libc is intentionally not bundled; it differs across operating systems.'}
    if len(args.hosts) == 6:
        root = pathlib.Path(__file__).resolve().parent
        aggregate = read(root / 'aggregate.json')
        assert set(aggregate['hosts']) == {host.name for host in args.hosts}
        for host in args.hosts:
            for group, checked in result['hosts'][str(host)]['groups'].items():
                recorded = aggregate['hosts'][host.name]['groups'][group]
                close(recorded['median_speedup'], checked['p50_speedup_median'])
                close(recorded['min_speedup'], checked['p50_speedup_min'])
                close(recorded['max_speedup'], checked['p50_speedup_max'])
                close(recorded['median_candidate_run_spread_pct'], checked['candidate_spread_median_pct'])
                close(recorded['max_candidate_run_spread_pct'], checked['candidate_spread_max_pct'])
                assert recorded['wins'] == checked['p50_wins']
                for percentile in ['p50', 'p95', 'p99']:
                    assert recorded['losses_or_ties'][percentile] == [r['scene'] for r in checked['losses_or_ties'][percentile]]
                    assert recorded[percentile + '_wins'] == 16 - len(checked['losses_or_ties'][percentile])
        expected_rows = {}
        for host in args.hosts:
            for group in result['hosts'][str(host)]['groups']:
                for row in read(host / group / 'summary.json')['scenes']:
                    expected_rows[host.name, group, row['scene']] = row
        assert len(aggregate['rows']) == len(expected_rows) == 384
        assert {(row['host'], row['group'], row['scene']) for row in aggregate['rows']} == set(expected_rows)
        for row in aggregate['rows']:
            expected = expected_rows[row['host'], row['group'], row['scene']]
            assert all(row[key] == value for key, value in expected.items())
        with (root / 'scene-latencies.csv').open() as source_csv:
            csv_rows = list(csv.DictReader(source_csv))
        assert len(csv_rows) == 384
        assert {(row['host'], row['group'], row['scene']) for row in csv_rows} == set(expected_rows)
        for row in csv_rows:
            expected = expected_rows[row['host'], row['group'], row['scene']]
            for key, value in expected.items():
                if isinstance(value, (int, float)):
                    close(float(row[key]), value)
                elif isinstance(value, str):
                    assert row[key] == value
        result['aggregate_audit'] = {'json_and_csv_rows_checked': 384, 'all_host_group_statistics_match_raw_recomputation': True}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'hosts_audited': len(result['hosts']),
                      'timed_rows': 384 * len(result['hosts'])}))


if __name__ == '__main__':
    main()
