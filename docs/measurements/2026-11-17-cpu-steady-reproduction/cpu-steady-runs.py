# SPDX-License-Identifier: Apache-2.0
import pathlib,subprocess,json,time
root=pathlib.Path('docs/measurements')
configs=[]
for threads,cpus in [(1,'5'),(4,'5,6,7,8')]:
 for tag,inputs,filter_,aruco,iterations,warmup,dictionary in [
  ('primary','data/cpu-experiment','',1,200,30,'DICT_ARUCO_MIP_36h12'),
  ('quality','data/cpu-quality-641x479','',1,200,30,'DICT_ARUCO_MIP_36h12'),
  ('fullres','data/cpu-experiment','cpu_',0,50,10,'DICT_ARUCO_MIP_36h12'),
  ('quality-fullres','data/cpu-quality-641x479','',0,100,20,'DICT_ARUCO_MIP_36h12'),
  ('highid','build/cpu-lut-probe-1000','',1,200,30,'DICT_6X6_1000')]:
  out=root/f'2026-11-17-cpu-steady-final-{tag}-{threads}t'
  args=['python3','cpu/run_experiments.py','--binary','build/cpu-opt/cpu/aruco3cuda_cpu_benchmark','--input-dir',inputs,'--output-dir',str(out),'--cpu-list',cpus,'--threads',str(threads),'--runs','3','--iterations',str(iterations),'--warmup',str(warmup),'--aruco3',str(aruco),'--dictionary',dictionary,'--variants','opencv,previous,optimized','--filter',filter_,'--background-note','Other host services and CPU correctness/sanitizer/coverage work pinned to efficiency CPUs0-3 may run concurrently; detector timings use performance CPU5 or CPUs5-8 exclusively. Representative checks are repeated after validation ends.']
  configs.append(args)
for args in configs:
 print('START',json.dumps(args),flush=True)
 subprocess.run(args,check=True)
 out=pathlib.Path(args[args.index('--output-dir')+1])
 with (out/'summary.json').open('w') as output:subprocess.run(['python3','cpu/summarize_experiments.py',str(out)],stdout=output,check=True)
 print('FINISHED',str(out),flush=True)
