# SPDX-License-Identifier: Apache-2.0
"""One-off aggregation; run from the repository root after all timing groups finish."""
import collections
import json
import pathlib
import statistics

ROOT = pathlib.Path('docs/measurements')
PREFIX = '2026-11-17-cpu-steady-'

def read_rows(path, kind=None):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return [r for r in rows if kind is None or r.get('type') == kind]

def med(rows, key):
    return statistics.median(r[key] for r in rows)

result = {'aggregation': 'Scene percentiles are medians of three independent process percentiles; no pooling.'}
primary = {}
for tag in ['primary', 'quality', 'fullres', 'quality-fullres', 'highid']:
    for threads in [1, 4]:
        name = f'final-{tag}-{threads}t'
        p = ROOT / (PREFIX + name)
        data = json.loads((p / 'summary.json').read_text())
        assert data['independent_processes'] == 3
        rows = data['scenes']
        assert all(not r['compatibility_discrepancies'] for r in rows)
        positive = [r for r in rows if r['reference_detections']]
        primary[name] = {
            'summary': str(p / 'summary.json'),
            'all': data['all_scenes'], 'positive': data['scenes_with_detections'],
            'previous_positive_median': med(positive, 'speedup_over_previous'),
            'previous_all_wins': sum(r['speedup_over_previous'] > 1 for r in rows),
            'previous_positive_wins': sum(r['speedup_over_previous'] > 1 for r in positive),
            'slower_opencv': [r for r in rows if r['speedup'] < 1],
            'worst_previous': min(rows, key=lambda r: r['speedup_over_previous']),
            'candidate_process_spread_median_pct': med(rows, 'candidate_run_spread_pct'),
            'candidate_process_spread_max_pct': max(r['candidate_run_spread_pct'] for r in rows),
            'slower_opencv_p95': [r['scene'] for r in rows if r['candidate_p95_ms'] > r['opencv_p95_ms']],
            'slower_opencv_p99': [r['scene'] for r in rows if r['candidate_p99_ms'] > r['opencv_p99_ms']],
        }
result['primary'] = primary

video = []
for size in ['1280x720', '3840x2160']:
    p = ROOT / (PREFIX + 'video-' + size)
    groups = collections.defaultdict(list)
    for trial in range(1, 4):
        rows = read_rows(p / f'run-{trial}.jsonl', 'video_measurement')
        assert len(rows) == 4
        for row in rows:
            assert row['exact_ids_and_ordered_corners'] and row['exact_untimed_rejected_counts']
            assert row['timed_frames_checked'] == row['frames'] == 600
            groups[(row['route'], row['layout'])].append(row)
    for (route, layout), rows in groups.items():
        assert len(rows) == 3
        item = {'size': size, 'route': route, 'layout': layout, 'workers': rows[0]['workers'],
                'opencv_threads': rows[0]['opencv_threads']}
        for key in ['service_p50_ms', 'service_p95_ms', 'service_p99_ms', 'saturated_throughput_fps']:
            item[key] = med(rows, key)
        fps = [r['saturated_throughput_fps'] for r in rows]
        item['fps_process_spread_pct'] = (max(fps) - min(fps)) / statistics.median(fps) * 100
        video.append(item)
result['video'] = video

rows = read_rows(ROOT / (PREFIX + 'resources') / 'resources.jsonl')
assert len(rows) == 108
resources = []
groups = collections.defaultdict(list)
for row in rows:
    groups[(pathlib.Path(row['image']).name, row['threads'], row['route'])].append(row)
for (scene, threads, route), values in sorted(groups.items()):
    assert len(values) == 3
    item = {'scene': scene, 'threads': threads, 'route': route}
    for key in ['setup_ms', 'first_detection_ms', 'time_to_first_result_ms', 'throughput_fps', 'peak_process_rss_kib']:
        item[key] = med(values, key)
    resources.append(item)
result['resources'] = resources

p = ROOT / (PREFIX + 'final-native')
native = collections.defaultdict(list)
for trial in range(1, 4):
    by_build = {}
    for build in ['portable', 'native']:
        rows = read_rows(p / f'{build}-{trial}.jsonl', 'measurement')
        assert len(rows) == 48
        assert all(not r['compatibility_discrepancy'] and r['max_corner_error_px'] == 0 for r in rows)
        by_build[build] = {(r['scene'], r['variant']): r for r in rows}
    for scene in sorted({key[0] for key in by_build['portable']}):
        pc = by_build['portable'][(scene, 'optimized')]['p50_ms']
        po = by_build['portable'][(scene, 'opencv')]['p50_ms']
        nc = by_build['native'][(scene, 'optimized')]['p50_ms']
        no = by_build['native'][(scene, 'opencv')]['p50_ms']
        native[scene].append({'raw_native_speedup': pc / nc, 'normalized_native_speedup': (no / nc) / (po / pc), 'opencv_drift_ratio': po / no})
result['native'] = [{'scene': scene, **{key: med(rows, key) for key in rows[0]}} for scene, rows in sorted(native.items())]

quiet = {}
for threads in [1, 4]:
    p = ROOT / (PREFIX + f'quiet-{threads}t')
    data = json.loads((p / 'summary.json').read_text())
    assert data['independent_processes'] == 3 and len(data['scenes']) == 44
    source = {}
    for tag in ['primary', 'quality']:
        rows = json.loads((ROOT / (PREFIX + f'final-{tag}-{threads}t') / 'summary.json').read_text())['scenes']
        source.update({r['scene']: r for r in rows})
    comparisons = []
    for row in data['scenes']:
        old = source[row['scene']]
        assert row['sha256'] == old['sha256'] and not row['compatibility_discrepancies']
        comparisons.append({'scene': row['scene'], 'candidate_latency_ratio': row['candidate_p50_ms'] / old['candidate_p50_ms'],
                            'opencv_latency_ratio': row['opencv_p50_ms'] / old['opencv_p50_ms'],
                            'relative_opencv_speedup_ratio': row['speedup'] / old['speedup'],
                            'previous_speedup': row['speedup_over_previous'], 'opencv_speedup': row['speedup']})
    quiet[str(threads)] = {'all': data['all_scenes'], 'positive': data['scenes_with_detections'], 'comparisons': comparisons}
result['quiet'] = quiet

out = pathlib.Path('build/cpu-steady-report-data.json')
out.write_text(json.dumps(result, indent=2) + '\n')
print(out)
