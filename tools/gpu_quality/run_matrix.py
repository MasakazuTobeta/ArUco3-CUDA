#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate the fixed GPU quality matrix. Defaults to plan-only; use --execute to run."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--build', type=Path, required=True)
p.add_argument('--source', type=Path, help='Optional immutable source export to verify')
p.add_argument('--manifest', type=Path, help='Optional source_files SHA-256 manifest; requires --source')
p.add_argument('--photos', type=Path, required=True)
p.add_argument('--external', type=Path, required=True)
p.add_argument('--chains', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--input-manifest', type=Path)
p.add_argument('--profile', choices=['OPENCV_ARM_HAL', 'GENERIC'], required=True)
p.add_argument('--backend', required=True)
p.add_argument('--fma', choices=['FUSED', 'SEPARATE', 'NOT_APPLICABLE'], required=True)
p.add_argument('--strategy', required=True)
p.add_argument('--execute', action='store_true')
a = p.parse_args()
if bool(a.source) != bool(a.manifest):
    p.error('--source and --manifest must be supplied together')
if not __debug__:
    p.error('Run without Python -O: validation assertions must remain enabled')
audit = a.build / 'tools/gpu_quality/aruco3cuda_gpu_quality_audit'
evaluator = a.build / 'tools/evaluate/aruco3cuda_evaluate'
base = [str(audit), '--max-candidates', '16384', '--max-markers', '1024']
cases = []
for mode in [1, 0]:
    opts = ['--aruco3', str(mode)]
    cases += [('quality96-' + str(mode), opts, 118, 96),
              ('photos-' + str(mode), opts + ['--images', str(a.photos)], 27, 16)]
for mode in [1, 0]:
    for dictionary in [f'DICT_{s}X{s}_{n}' for s in range(4, 8) for n in [50, 100, 250, 1000]] + ['DICT_ARUCO_MIP_36h12']:
        cases.append(('external-' + dictionary + '-' + str(mode), ['--aruco3', str(mode), '--dictionary', dictionary, '--images', str(a.external)], 19, 9))
variants = [('even-640x480', ['--width', '640', '--height', '480']), ('odd-1280x721', ['--width', '1280', '--height', '721']), ('cell2', ['--cell-size', '2']), ('cell7', ['--cell-size', '7']), ('ratio0', ['--min-marker-ratio', '0']), ('ratio01', ['--min-marker-ratio', '0.1']), ('minside16', ['--min-canonical-side', '16']), ('minside48', ['--min-canonical-side', '48']), ('refine100-001', ['--refine-iterations', '100', '--refine-accuracy', '0.01']), ('off-cell7', ['--aruco3', '0', '--cell-size', '7']), ('direct-default', ['--direct'])]
cases += [('variant-' + name, opts, 118, 96) for name, opts in variants]
for mode in [1, 0]:
    cases.append(('chains-' + str(mode), ['--aruco3', str(mode), '--dictionary', 'DICT_4X4_50', '--images', str(a.chains)], 82, 64))
plan = [{'name': n, 'command': base + opts, 'calls': calls, 'unique': unique} for n, opts, calls, unique in cases]
for mode in [1, 0]:
    plan.append({'name': 'normal91-' + str(mode), 'command': [str(evaluator), '--preset', 'full', '--use-aruco3', str(mode), '--corpus-dir', str(a.output / ('corpus-' + str(mode))), '--output', str(a.output / ('normal91-' + str(mode) + '.json'))], 'images': 91, 'capacity_note': 'Historical evaluator uses its unchanged default4096/1024.'})
if not a.execute:
    print(json.dumps({'gpu_execution': False, 'audit_cases': len(cases), 'audit_calls': sum(c[2] for c in cases), 'plan': plan}, indent=2))
    raise SystemExit(0)
a.output.mkdir(parents=True, exist_ok=False)
dump(a.output / 'plan.json', plan)
manifest = json.loads(a.manifest.read_text()) if a.manifest else {'source_files': {}}
for name, digest in manifest['source_files'].items():
    assert sha(a.source / name) == digest, ('source mismatch', name)
cache = (a.build / 'CMakeCache.txt').read_text()
assert 'ARUCO3CPU_NATIVE:BOOL=ON' in cache, 'native CPU build required'
binaries = {str(x): sha(x) for x in [audit, evaluator]}
inputs = {}
for kind, directory, expected in [('photos', a.photos, 16), ('external', a.external, 9), ('chains', a.chains, 64)]:
    paths = sorted(x for x in directory.iterdir() if x.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.pgm'])
    assert len(paths) == expected, (kind, len(paths))
    inputs[kind] = {x.name: sha(x) for x in paths}
if a.input_manifest:
    expected_inputs = json.loads(a.input_manifest.read_text())
    for kind, entries in inputs.items():
        assert entries == {k: v for k, v in expected_inputs[kind].items() if k != 'truth.jsonl'}, (kind, 'canonical input hashes differ')
    assert sha(a.chains / 'truth.jsonl') == expected_inputs['chains']['truth.jsonl']
truth = {r['name']: r for r in map(json.loads, (a.chains / 'truth.jsonl').read_text().splitlines())}
dump(a.output / 'environment.json', {'started_unix': time.time(), 'runner_sha256': sha(__file__), 'source_manifest_sha256': sha(a.manifest) if a.manifest else None, 'binary_sha256': binaries, 'cmake_cache_sha256': sha(a.build / 'CMakeCache.txt'), 'input_file_sha256': inputs, 'profile': a.profile, 'backend': a.backend, 'fma': a.fma, 'strategy': a.strategy, 'native_cpu': True, 'unannotated': 'Photos and external fixtures are compatibility observations, not ground truth.'})
errors = []
records = []
for item in plan:
    name = item['name']
    output = a.output / (name + ('.jsonl' if 'calls' in item else '.log'))
    start = time.monotonic()
    with output.open('x') as out, (a.output / (name + '.stderr')).open('x') as err:
        result = subprocess.run(item['command'], stdout=out, stderr=err)
    rec = {'name': name, 'exit': result.returncode, 'elapsed_seconds': time.monotonic() - start}
    if 'calls' not in item:
        assert result.returncode == 0, rec
        report = json.loads((a.output / (name + '.json')).read_text())
        assert report['sceneCount'] == 91
        rec['routes'] = report['routes']
        rec['device_memory'] = report.get('deviceMemory')
        cpu = report['routes']['CPU']['truth']
        gpu_name = 'CUDA'
        gpu = report['routes'][gpu_name]['truth']
        for key, direction in [('truePositive', -1), ('falsePositive', 1), ('rotationAgreed', -1), ('cornerRmsePx', 1), ('cornerMaxPx', 1)]:
            if direction * (gpu[key] - cpu[key]) > (0.001 if key.startswith('corner') else 0):
                errors.append([name, 'historical summary regression', key, cpu[key], gpu[key]])
    else:
        assert result.returncode in [0, 2], rec
        rows = list(map(json.loads, output.read_text().splitlines()))
        assert len(rows) == item['calls'] + 1 and rows[-1]['type'] == 'completion' and rows[-1]['frames'] == item['calls'], name
        assert [r['sequence'] for r in rows[:-1]] == list(range(item['calls']))
        unique = {r['name']: r for r in rows[:-1] if r['family'] != 'sequence_cut'}
        assert len(unique) == item['unique'], name
        seen = {}
        corner_deltas = []
        command = item['command']
        requested = lambda flag, default: command[command.index(flag) + 1] if flag in command else default
        expected_config = {'perspective_remove_pixel_per_cell': int(requested('--cell-size', '4')), 'min_side_length_canonical_img': int(requested('--min-canonical-side', '32')), 'min_marker_length_ratio_original_img': float(requested('--min-marker-ratio', '.05')), 'corner_refinement_max_iterations': int(requested('--refine-iterations', '30')), 'corner_refinement_min_accuracy': float(requested('--refine-accuracy', '.1'))}
        for r in rows[:-1]:
            assert (r['gpu_resize_profile'], r['gpu_resize_backend'], r['gpu_carotene_x_fma'], r['gpu_strategy_identity']) == (a.profile, a.backend, a.fma, a.strategy), (name, 'compiled strategy mismatch')
            assert r['configuration']['max_candidates'] == 16384 and r['configuration']['max_markers'] == 1024
            assert r['opencv_version'] == '4.14.0'
            assert r['aruco3'] == bool(int(requested('--aruco3', '1'))) and r['graph'] == ('--direct' not in command)
            assert r['dictionary'] == requested('--dictionary', 'DICT_ARUCO_MIP_36h12')
            for field, expected in expected_config.items():
                assert math.isclose(r['configuration'][field], expected, rel_tol=1e-6, abs_tol=1e-9), (name, field)
            assert all(math.isfinite(x) for route in ['opencv', 'optimized_cpu', 'gpu'] for d in r[route] for x in d['corners'])
            if not r['cpu_exact'] or r['failed_positive_control'] or r['gpu_discrete_regression']:
                errors.append([name, r['sequence'], 'CPU/control/discrete regression'])
            if not r['has_ground_truth'] and r['cpu_gpu_differences_at_1px']:
                errors.append([name, r['sequence'], 'unannotated compatibility difference', r['cpu_gpu_differences_at_1px']])
            identity = r['pixel_sha256']
            obs = {k: r[k] for k in ['opencv', 'optimized_cpu', 'gpu', 'configuration']}
            if identity in seen:
                assert seen[identity] == obs, (name, r['sequence'], 'changing-frame replay differs')
            seen[identity] = obs
        for image, r in unique.items():
            if r['has_ground_truth']:
                assert len(r['opencv_quality']['per_truth']) == len(r['gpu_quality']['per_truth']) == len(r['truth']), (name, image, 'incomplete per-truth metrics')
                for index, (cpu, gpu) in enumerate(zip(r['opencv_quality']['per_truth'], r['gpu_quality']['per_truth'])):
                    if cpu['outcome'] == 0 and gpu['outcome'] == 0:
                        corner_deltas.append({'name': image, 'truth_index': index, 'rmse_delta_px': math.sqrt(gpu['corner_squared_sum_px2'] / 4) - math.sqrt(cpu['corner_squared_sum_px2'] / 4), 'max_delta_px': gpu['corner_max_px'] - cpu['corner_max_px']})
            elif name.startswith('chains-'):
                gt = {d['id']: d['corners'] for d in truth[Path(image).stem]['truth']}
                scores = {}
                for route in ['opencv', 'gpu']:
                    scores[route] = {}
                    for d in r[route]:
                        ident = d['id']
                        if ident not in gt or ident in scores[route]:
                            errors.append([name, image, route, 'unknown or duplicate ID', ident])
                            continue
                        distances = [math.hypot(d['corners'][2*i] - gt[ident][2*i], d['corners'][2*i+1] - gt[ident][2*i+1]) for i in range(4)]
                        scores[route][ident] = (math.sqrt(sum(x*x for x in distances)/4), max(distances))
                for ident, cpu in scores['opencv'].items():
                    if ident not in scores['gpu']:
                        errors.append([name, image, 'missing CPU ID', ident])
                    else:
                        gpu = scores['gpu'][ident]
                        corner_deltas.append({'name': image, 'id': ident, 'rmse_delta_px': gpu[0]-cpu[0], 'max_delta_px': gpu[1]-cpu[1]})
        worse = [d for d in corner_deltas if max(d['rmse_delta_px'], d['max_delta_px']) > 0.001]
        if worse:
            errors.append([name, 'material GT corner worsening', worse])
        dump(a.output / (name + '-corner-deltas.json'), corner_deltas)
        rec.update(completion=rows[-1], unique_images=len(unique), revisits=len(rows)-1-len(seen), material_corner_worsening=len(worse), tiny_positive_corner_deltas=sum(0 < max(d['rmse_delta_px'], d['max_delta_px']) <= .001 for d in corner_deltas), totals={route: sum(len(r[route]) for r in unique.values()) for route in ['opencv', 'gpu']})
    records.append(rec)
    with (a.output / 'cases.jsonl').open('a') as f:
        f.write(json.dumps(rec) + '\n')
    print(json.dumps({k: v for k, v in rec.items() if k != 'routes'}), flush=True)
assert all(sha(path) == digest for path, digest in binaries.items()), 'binary changed during run'
for name, digest in manifest['source_files'].items():
    assert sha(a.source / name) == digest, ('source changed during run', name)
for kind, directory in [('photos', a.photos), ('external', a.external), ('chains', a.chains)]:
    assert all(sha(directory / name) == digest for name, digest in inputs[kind].items()), ('input changed during run', kind)
dump(a.output / 'summary.json', {'complete': True, 'accepted': not errors, 'cases': len(records), 'audit_calls': sum(c[2] for c in cases), 'historical_images': 182, 'errors': errors, 'binaries_unchanged': True, 'source_manifest_verified': bool(a.manifest)})
raise SystemExit(1 if errors else 0)
