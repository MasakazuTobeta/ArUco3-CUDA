# SPDX-License-Identifier: Apache-2.0
import datetime,hashlib,json,pathlib,platform,subprocess

def read(path):
 try:return pathlib.Path(path).read_text()
 except OSError:return None
binary=pathlib.Path('build/cpu-opt/cpu/aruco3cuda_cpu_video')
for size in ['1280x720','3840x2160']:
 destination=pathlib.Path('docs/measurements/2026-11-17-cpu-steady-video-'+size)
 destination.mkdir(parents=True,exist_ok=False)
 cpus=[5,6,7,8]
 command=['taskset','-c','5-8',str(binary),'--input-dir','data/cpu-experiment','--filter','cpu_'+size+'_','--frames','600','--workers','4','--opencv-threads','1','--warmup','36','--route','both','--layout','both','--tuning','default','--save-frames']
 paths=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard'],text=True).splitlines()
 metadata={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':command,'order':['forward','reverse','forward'],'host':platform.platform(),'cpuinfo':read('/proc/cpuinfo'),'aslr':read('/proc/sys/kernel/randomize_va_space'),'load_at_start':read('/proc/loadavg'),'cmake_cache':read('build/cpu-opt/CMakeCache.txt'),'compile_commands':read('build/cpu-opt/compile_commands.json'),'opencv_provenance':read('/opt/opencv/share/aruco3cuda/opencv-provenance.json'),'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'source_sha256':{p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in paths if pathlib.Path(p).is_file() and pathlib.Path(p).suffix in {'.cpp','.hpp','.py','.txt'}},'cpu_policy':{str(cpu):{n:read(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/{n}') for n in ['scaling_governor','scaling_max_freq','scaling_cur_freq']} for cpu in cpus},'scope':'Detector service time and finite saturated-batch throughput, no camera arrival/queue latency; startup/validation excluded','host_policy':'Retained; unrelated host services remain. All other agent build/test processes finished.'}
 (destination/'environment.json').write_text(json.dumps(metadata,indent=2)+'\n')
 for trial in range(1,4):
  assert hashlib.sha256(binary.read_bytes()).hexdigest()==metadata['binary_sha256']
  args=command+(['--reverse-order'] if trial==2 else [])
  with (destination/f'run-{trial}.jsonl').open('x') as output:
   subprocess.run(args,stdout=output,check=True)
  print('Completed',size,trial,flush=True)
