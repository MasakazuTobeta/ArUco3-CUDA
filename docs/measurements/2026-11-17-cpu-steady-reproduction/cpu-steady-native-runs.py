# SPDX-License-Identifier: Apache-2.0
import pathlib,subprocess,json,hashlib,datetime,platform
p=pathlib.Path('docs/measurements/2026-11-17-cpu-steady-final-native');p.mkdir(parents=True,exist_ok=False)
binaries={'portable':'build/cpu-opt/cpu/aruco3cuda_cpu_benchmark','native':'build/cpu-native/cpu/aruco3cuda_cpu_benchmark'}
meta={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Diagnostic native-tuning comparison. Independent processes alternate build order; each process pairs actual OpenCV and optimized.80 samples15warmups,24 extendedscenes. No stage profiling. Normalize against that process OpenCV before comparing builds.','binary_sha256':{k:hashlib.sha256(pathlib.Path(v).read_bytes()).hexdigest() for k,v in binaries.items()},'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in pathlib.Path('cpu').glob('*') if f.suffix in ['.cpp','.hpp']},'commands':[],'cpuinfo':pathlib.Path('/proc/cpuinfo').read_text(),'platform':platform.platform(),'background':'All other agent build/test processes finished; unrelated host services retained. Performance CPU5 reserved for this measurement.'}
for trial in range(1,4):
 for label in (['native','portable'] if trial==2 else ['portable','native']):
  args=['taskset','-c','5',binaries[label],'--input-dir','data/cpu-experiment','--filter','cpu_','--aruco3','1','--threads','1','--variants','opencv,optimized','--iterations','80','--warmup','15']
  with (p/f'{label}-{trial}.jsonl').open('x') as out:subprocess.run(args,stdout=out,check=True)
  meta['commands'].append(args);print('Native diagnostic',label,trial,flush=True)
(p/'environment.json').write_text(json.dumps(meta,indent=2)+'\n')
