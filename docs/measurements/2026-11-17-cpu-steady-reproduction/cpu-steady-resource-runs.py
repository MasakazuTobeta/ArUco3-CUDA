# SPDX-License-Identifier: Apache-2.0
import datetime,hashlib,json,pathlib,subprocess,platform
p=pathlib.Path('docs/measurements/2026-11-17-cpu-steady-resources');p.mkdir(parents=True,exist_ok=False)
binary='build/cpu-opt/cpu/aruco3cuda_cpu_resources'
scenes=['cpu_1280x720_n4_clean.png','cpu_1280x720_n4_noise.png','cpu_3840x2160_n1_clean.png','cpu_3840x2160_n4_clean.png','cpu_3840x2160_n4_noise.png','cpu_3840x2160_n16_clean.png']
meta={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'platform':platform.platform(),'binary_sha256':hashlib.sha256(pathlib.Path(binary).read_bytes()).hexdigest(),'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in pathlib.Path('cpu').glob('*') if f.is_file() and f.suffix in ['.cpp','.hpp']},'inputs_sha256':{s:hashlib.sha256((pathlib.Path('data/cpu-experiment')/s).read_bytes()).hexdigest() for s in scenes},'commands':[],'scope':'Isolated process per route; RSS includes image decoding, input, OpenCV and detector; startup after decoding; initialization is not an acceptance gate','opencv_provenance':pathlib.Path('/opt/opencv/share/aruco3cuda/opencv-provenance.json').read_text(),'cpuinfo':pathlib.Path('/proc/cpuinfo').read_text(),'aslr':pathlib.Path('/proc/sys/kernel/randomize_va_space').read_text(),'power_policy':'Retained; unrelated host services remain. All other agent build/test processes finished.'}
with (p/'resources.jsonl').open('x') as out:
 for threads,cpus in [(1,'5'),(4,'5-8')]:
  for s in scenes:
   for trial in range(3):
    routes=['opencv','previous','optimized']
    if trial==1:routes.reverse()
    for route in routes:
     command=['taskset','-c',cpus,binary,'data/cpu-experiment/'+s,route,'1',str(threads),'100']
     r=json.loads(subprocess.check_output(command,text=True));r['trial']=trial+1
     out.write(json.dumps(r)+'\n');out.flush();meta['commands'].append(command)
   print('Completed resources',threads,s,flush=True)
(p/'environment.json').write_text(json.dumps(meta,indent=2)+'\n')
