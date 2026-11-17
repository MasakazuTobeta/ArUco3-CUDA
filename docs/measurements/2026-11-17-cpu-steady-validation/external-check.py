# SPDX-License-Identifier: Apache-2.0
# Reproduction of the external accuracy sweep; run inside the CPU container.
import pathlib,subprocess,json,hashlib,datetime
names=['DICT_'+str(n)+'X'+str(n)+'_'+str(m) for n in [4,5,6,7] for m in [50,100,250,1000]]+['DICT_ARUCO_ORIGINAL','DICT_ARUCO_MIP_36h12']
d=pathlib.Path('docs/measurements/2026-11-17-cpu-steady-external');d.mkdir(exist_ok=True)
binary='build/cpu-opt/cpu/aruco3cuda_cpu_benchmark'
metadata={'purpose':'Accuracy only, one sample, no speed claims','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'binary_sha256':hashlib.sha256(pathlib.Path(binary).read_bytes()).hexdigest(),'commands':[],'opencv_provenance':pathlib.Path('/opt/opencv/share/aruco3cuda/opencv-provenance.json').read_text(),'source_sha256':{}}
for name in subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard'],text=True).splitlines():
 p=pathlib.Path(name)
 if p.is_file() and p.suffix in ('.cpp','.hpp','.txt','.py'):metadata['source_sha256'][name]=hashlib.sha256(p.read_bytes()).hexdigest()
summary=[]
for aruco in [0,1]:
 for name in names:
  args=[binary,'--input-dir','data/cpu-opencv-fixtures','--variants','previous,optimized','--iterations','1','--warmup','0','--threads','4','--aruco3',str(aruco),'--dictionary',name]
  p=d/(name+'-'+str(aruco)+'.jsonl')
  if p.exists():raise RuntimeError('refuse overwrite '+str(p))
  with p.open('w') as o: subprocess.run(args,stdout=o,check=True)
  rows=[json.loads(s) for s in p.read_text().splitlines()][1:]
  metadata['commands'].append(args)
  summary.append({'dictionary':name,'aruco3':bool(aruco),'records':len(rows),'positive_scenes':sum(r['variant']=='optimized' and r['detections']>0 for r in rows),'detections':sum(r['detections'] for r in rows if r['variant']=='optimized'),'max_corner_error_px':max(r['max_corner_error_px'] for r in rows) if 'max_corner_error_px' in rows[0] else 'See raw exact checks'})
  print(name,aruco,'passed',len(rows),flush=True)
(d/'environment.json').write_text(json.dumps(metadata,indent=2)+'\n')
(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
