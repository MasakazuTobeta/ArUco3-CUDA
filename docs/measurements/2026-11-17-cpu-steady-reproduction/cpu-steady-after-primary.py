# SPDX-License-Identifier: Apache-2.0
import pathlib,subprocess,time,json
final=pathlib.Path('docs/measurements/2026-11-17-cpu-steady-final-highid-4t/summary.json')
def primary_active():
 for p in pathlib.Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   if (p/'cmdline').read_bytes().replace(b'\0',b' ').strip()==b'python3 build/cpu-steady-runs.py':return True
  except FileNotFoundError:pass
 return False
while primary_active():time.sleep(5)
assert final.exists(),'Primary measurement failed or was interrupted; refusing dependent runs'
assert json.loads(final.read_text())['independent_processes']==3
for script in ['build/cpu-steady-native-runs.py','build/cpu-steady-extra-runs.py']:
 print('START',script,flush=True);subprocess.run(['python3',script],check=True)
print('All dependent measurement groups completed',flush=True)
