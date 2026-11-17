# SPDX-License-Identifier: Apache-2.0
import subprocess
for script in ['build/cpu-steady-video-runs.py','build/cpu-steady-resource-runs.py']:
 print('START',script,flush=True);subprocess.run(['python3',script],check=True)
for threads,cpus in [(1,'5'),(4,'5,6,7,8')]:
 args=['python3','cpu/run_experiments.py','--binary','build/cpu-opt/cpu/aruco3cuda_cpu_benchmark','--input-dir','data/cpu-experiment','--output-dir',f'docs/measurements/2026-11-17-cpu-steady-stages-{threads}t','--cpu-list',cpus,'--threads',str(threads),'--runs','1','--iterations','40','--warmup','10','--variants','opencv,previous,optimized','--filter','cpu_','--profile','--background-note','All other agent build/test processes finished; unrelated host services remain; timing uses the stated performance core budget']
 print('START stages',threads,flush=True);subprocess.run(args,check=True)
import pathlib
import time
pathlib.Path('build/cpu-steady-ready-for-failure-fix').write_text('All original-source detector timing groups complete.\n')
while not pathlib.Path('build/cpu-steady-failure-fix-complete').exists():
 time.sleep(5)
assert pathlib.Path('build/cpu-steady-qa-complete').exists(), 'Final QA marker missing; quiet confirmation must wait'
print('START post-validation quiet confirmation',flush=True)
subprocess.run(['python3','build/cpu-steady-quiet-runs.py'],check=True)
