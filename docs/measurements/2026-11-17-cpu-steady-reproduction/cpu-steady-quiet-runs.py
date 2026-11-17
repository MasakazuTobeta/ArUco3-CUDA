# SPDX-License-Identifier: Apache-2.0
import pathlib,subprocess,json,hashlib,shutil,collections
marker=json.loads(pathlib.Path('build/cpu-steady-qa-complete').read_text())
for path, expected in marker['sha256'].items():
 assert hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()==expected, 'Validated source changed: '+path
source=pathlib.Path('data/cpu-experiment');quality=pathlib.Path('data/cpu-quality-641x479')
subset=pathlib.Path('data/cpu-steady-quiet-subset');subset.mkdir(exist_ok=False)
selected=list(sorted(source.glob('cpu_*.png')))
selected += [source/f'clean_{size}_n0_s16.png' for size in ['640x480','1280x720','1920x1080','3840x2160']]
families=collections.defaultdict(list)
for row in json.loads((quality/'manifest.json').read_text())['scenes']:families[row['family']].append(row)
for rows in families.values():selected += [quality/rows[0]['path'],quality/rows[-1]['path']]
assert len(selected)==44
manifest={'selection':'All24 extended marker scenes, one empty image at each of4 resolutions, first and last manifest entries from each of8 quality families. Selection uses scene definitions, never observed latency.','scope':'Post-validation representative confirmation. Different subset/order can change retained allocation history; this is not a causal isolation experiment.','files':[]}
for src in selected:
 dst=subset/src.name;shutil.copyfile(src,dst);manifest['files'].append({'source':str(src),'name':src.name,'sha256':hashlib.sha256(src.read_bytes()).hexdigest()})
(subset/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for threads,cpus in [(1,'5'),(4,'5,6,7,8')]:
 out=pathlib.Path(f'docs/measurements/2026-11-17-cpu-steady-quiet-{threads}t')
 args=['python3','cpu/run_experiments.py','--binary','build/cpu-opt/cpu/aruco3cuda_cpu_benchmark','--input-dir',str(subset),'--output-dir',str(out),'--cpu-list',cpus,'--threads',str(threads),'--runs','3','--iterations','200','--warmup','30','--variants','opencv,previous,optimized','--background-note','All other agent build/test/benchmark processes have finished. Unrelated host services and unchanged power/clock policy remain.']
 subprocess.run(args,check=True)
 with (out/'summary.json').open('w') as output:subprocess.run(['python3','cpu/summarize_experiments.py',str(out)],stdout=output,check=True)
 shutil.copyfile(subset/'manifest.json',out/'selection.json')
