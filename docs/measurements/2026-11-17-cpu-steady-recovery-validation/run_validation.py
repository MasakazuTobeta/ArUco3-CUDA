# SPDX-License-Identifier: Apache-2.0
"""Reproduce the scoped CPU allocation-recovery checks; not a timing runner."""
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PREFIX = ['docker', 'exec', 'aruco3-cpu-opt', 'taskset', '-c', '0,1,2,3']
SAN_PREFIX = ['docker', 'exec', '-e', 'ASAN_OPTIONS=detect_leaks=1:halt_on_error=1',
              '-e', 'UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1',
              'aruco3-cpu-opt', 'taskset', '-c', '0,1,2,3']
FOCUSED = ('^CpuSparsePyramidTest[.]|^CpuDetectorTest[.]'
           '(changing_size_stride_and_pixels_never_reuses_stale_data|'
           'corner_refinement_options_match_with_and_without_aruco3)$|'
           '^CpuAccuracyStressTest[.]sparse_options_and_reinitialization_preserve_current_frame_results$')
BUILD = {'release': 'build/cpu-qa-round2', 'asan': 'build/cpu-qa-round2-asan',
         'native': 'build/cpu-native', 'coverage': 'build/cpu-qa-round2-coverage'}

def run(name, args):
    print('Starting ' + name, flush=True)
    record = {'name': name, 'command': args, 'cwd': str(ROOT),
              'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    start = time.monotonic()
    with (OUT / (name + '.log')).open('w') as stream:
        result = subprocess.run(args, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    record.update(exit_code=result.returncode, elapsed_seconds=time.monotonic()-start,
                  finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (OUT / (name + '.command.json')).write_text(json.dumps(record, indent=2) + '\n')
    print('Finished ' + name + ': ' + str(result.returncode), flush=True)
    if result.returncode:
        raise RuntimeError(name + ' failed')


def build(kind):
    if kind == 'coverage':
        path = ROOT / BUILD[kind] / 'cpu/CMakeFiles/aruco3cuda_cpu.dir/cpu_sparse_pyramid.cpp.gcda'
        retained = [str(p.relative_to(ROOT)) for p in (ROOT / BUILD[kind]).rglob('*.gcda') if p != path]
        reset = {'removed': str(path.relative_to(ROOT)), 'existed': path.exists(),
                 'before_sha256': hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
                 'retained_files': sorted(retained),
                 'scope': 'Delete only the changed sparse implementation translation unit; retain all unchanged module counters.'}
        path.unlink(missing_ok=True)
        (OUT / 'coverage-counter-reset.json').write_text(json.dumps(reset, indent=2) + '\n')
    targets = ['aruco3cuda_cpu_tests']
    if kind == 'native':
        targets += ['aruco3cuda_cpu_benchmark']
    run(kind + '-build', PREFIX + ['cmake', '--build', BUILD[kind], '--target'] + targets + ['-j1'])


def tests(kind):
    prefix = SAN_PREFIX if kind == 'asan' else PREFIX
    args = prefix + ['ctest', '--test-dir', BUILD[kind], '-j1', '--output-on-failure',
                     '--output-junit', '/workspace/' + str(OUT.relative_to(ROOT)) + '/' + kind + '.junit.xml']
    args += ['--schedule-random'] if kind == 'release' else ['-R', FOCUSED]
    run(kind + '-tests', args)


def probe():
    run('allocation-probe-build', PREFIX + [
        'g++', '-std=c++17', '-O2', '-g', '-fsanitize=address,undefined',
        '-fno-omit-frame-pointer', '-Icpu', '-isystem', '/opt/opencv/include/opencv4',
        str(OUT.relative_to(ROOT) / 'allocation_recovery_probe.cpp'),
        'build/cpu-qa-round2-asan/cpu/libaruco3cuda_cpu.a', '-L/opt/opencv/lib',
        '-Wl,-rpath,/opt/opencv/lib', '-lopencv_imgproc', '-lopencv_core',
        '-Wl,--wrap=_Znwm', '-o', 'build/cpu-sparse-allocation-recovery-probe'])
    run('allocation-probe', SAN_PREFIX + ['build/cpu-sparse-allocation-recovery-probe'])
    records = [json.loads(line) for line in (OUT / 'allocation-probe.log').read_text().splitlines()]
    assert len(records) == 2 and all(r['controlled_exception'] and r['retry_exact'] for r in records)


def coverage_report():
    run('coverage-report', PREFIX + [
        'gcovr', '--root', '/workspace', '--filter', '/workspace/cpu/',
        '--exclude', '.*/(benchmark|resource_probe|video_benchmark)[.]cpp',
        '--exclude', '.*/experiments/.*', '--decisions', '--print-summary',
        '--json', '/workspace/build/cpu-qa-round2-coverage/cpu-recovery-detail.json',
        '--json-summary', '/workspace/' + str(OUT.relative_to(ROOT) / 'coverage-summary.json'),
        '--txt', '/workspace/' + str(OUT.relative_to(ROOT) / 'coverage.txt'),
        '/workspace/build/cpu-qa-round2-coverage'])


def main():
    mode = sys.argv[1]
    if mode == 'builds':
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(build, kind) for kind in BUILD]
            for future in futures:
                future.result()
    elif mode == 'release':
        tests('release')
    elif mode == 'asan':
        probe()
        tests('asan')
    elif mode == 'native':
        tests('native')
    elif mode == 'coverage':
        tests('coverage')
        coverage_report()
    elif mode == 'portable-binaries':
        run('portable-binaries-build', PREFIX + ['cmake', '--build', 'build/cpu-opt', '--target',
             'aruco3cuda_cpu_benchmark', 'aruco3cuda_cpu_video', 'aruco3cuda_cpu_resources', '-j1'])
    else:
        raise ValueError(mode)

if __name__ == '__main__':
    main()
