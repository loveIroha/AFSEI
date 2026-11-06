from pathlib import Path
import json,time,os
root=Path('/mnt/large2/gjh/AFSEI/results/ep_afsei_nodal_opencarp_matched_20261030')
launch=json.loads(Path('/mnt/large2/gjh/AFSEI/demo/ep_afsei_nodal_opencarp_matched/launch.json').read_text())
last=-1
while True:
    report=root/'report.json'
    if report.exists():
        print('COMPLETED '+report.read_text(),flush=True);break
    path=root/'progress.json'
    if path.exists():
        try:p=json.loads(path.read_text())
        except json.JSONDecodeError:p={}
        if p.get('steps',-1)>last:
            last=p['steps'];print(json.dumps(p),flush=True)
    try:os.kill(launch['pid'],0)
    except ProcessLookupError:
        print('PROCESS_EXITED_WITHOUT_REPORT',flush=True);break
    time.sleep(10)
