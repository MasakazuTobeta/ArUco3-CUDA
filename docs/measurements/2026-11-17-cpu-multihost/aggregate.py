#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Aggregate all hosts without pooling image or process distributions."""
import csv
import json
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parent
HOSTS = {'20.23': 'DGX Spark GB10', '20.30': 'Jetson AGX Orin',
         '10.26': 'Intel N97', '20.41': 'Xeon w7-3565X',
         '20.20': 'Jetson AGX Thor', '20.44': 'Core i9-12900KS'}
GROUPS = ['aruco3-1t', 'aruco3-4t', 'fullres-1t', 'fullres-4t']


def main():
    result = {'aggregation': 'Per-image median of three process percentiles; image speedups are not pooled',
              'hosts': {}, 'rows': []}
    for host, label in HOSTS.items():
        directory = ROOT / 'hosts' / host
        completion = json.loads((directory / 'completion.json').read_text())
        assert completion['timing_rows'] == 384
        host_result = {'cpu': label, 'groups': {}}
        for group in GROUPS:
            summary = json.loads((directory / group / 'summary.json').read_text())
            environment = json.loads((directory / group / 'environment.json').read_text())
            scenes = summary['scenes']
            assert len(scenes) == 16 and summary['independent_processes'] == 3
            stats = dict(summary['all_scenes'])
            stats['cpu_affinity'] = environment['cpu_affinity']
            stats['binary_sha256'] = environment['binary_sha256']
            stats['median_candidate_run_spread_pct'] = statistics.median(s['candidate_run_spread_pct'] for s in scenes)
            stats['max_candidate_run_spread_pct'] = max(s['candidate_run_spread_pct'] for s in scenes)
            stats['losses_or_ties'] = {}
            for percentile in ['p50', 'p95', 'p99']:
                stats[percentile + '_wins'] = sum(s['opencv_' + percentile + '_ms'] > s['candidate_' + percentile + '_ms'] for s in scenes)
                stats['losses_or_ties'][percentile] = [s['scene'] for s in scenes
                    if s['opencv_' + percentile + '_ms'] <= s['candidate_' + percentile + '_ms']]
            for scene in scenes:
                result['rows'].append(dict(host=host, cpu=label, group=group, **scene))
            host_result['groups'][group] = stats
        result['hosts'][host] = host_result
    (ROOT / 'aggregate.json').write_text(json.dumps(result, indent=2) + '\n')
    with (ROOT / 'scene-latencies.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(result['rows'][0]))
        writer.writeheader()
        writer.writerows(result['rows'])
    for host, record in result['hosts'].items():
        print(host, record['cpu'], ' | '.join('%s %.3fx (%d/16)' %
              (group, record['groups'][group]['median_speedup'], record['groups'][group]['wins'])
              for group in GROUPS))


if __name__ == '__main__':
    main()
