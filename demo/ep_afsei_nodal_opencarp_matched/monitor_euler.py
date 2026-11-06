from pathlib import Path
import json,os,time
base=Path('/mnt/large2/gjh/AFSEI/demo/ep_afsei_nodal_opencarp_matched')
launch=json.loads((base/'euler_launch_20261031.json').read_text());out=Path(launch['output']);last=-1
while True:
    if (out/'report.json').exists():print('COMPLETED '+(out/'report.json').read_text(),flush=True);break
    try:p=json.loads((out/'progress.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):p={}
    if p.get('steps',-1)>last:last=p['steps'];print(json.dumps(p),flush=True)
    try:os.kill(launch['pid'],0)
    except ProcessLookupError:print('EXITED_WITHOUT_REPORT',flush=True);break
    time.sleep(10)
