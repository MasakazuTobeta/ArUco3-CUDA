# SPDX-License-Identifier: Apache-2.0
import datetime,json,pathlib,subprocess
commands={
'hostname':['hostname'],'uname':['uname','-a'],'os':['cat','/etc/os-release'],
'cpu':['lscpu'],'topology':['lscpu','-e=CPU,CORE,SOCKET,ONLINE,MAXMHZ'],
'power':['nvpmodel','-q'],'load':['cat','/proc/loadavg'],
'processes':['ps','-eo','pid,psr,pcpu,comm','--sort=-pcpu'],
'containers':['docker','ps','--format','{{.Names}} {{.Status}}'],
'gpu':['nvidia-smi'],'l4t':['cat','/etc/nv_tegra_release']}
result={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
for name,cmd in commands.items():
 try:
  proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30)
  result[name]={'command':cmd,'returncode':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr}
 except (OSError,subprocess.TimeoutExpired) as exc: result[name]={'command':cmd,'error':str(exc)}
result['cpufreq']={}
for base in pathlib.Path('/sys/devices/system/cpu').glob('cpu[0-9]*'):
 result['cpufreq'][base.name]={}
 for name in ['scaling_governor','scaling_cur_freq','scaling_max_freq','cpuinfo_max_freq']:
  try: result['cpufreq'][base.name][name]=(base/'cpufreq'/name).read_text().strip()
  except OSError: result['cpufreq'][base.name][name]='unavailable'
print(json.dumps(result,indent=2))
