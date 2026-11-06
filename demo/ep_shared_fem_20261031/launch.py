from pathlib import Path
import sys,subprocess,json
r=Path('/mnt/large2/gjh/AFSEI');out=r/'results/ep_shared_fem_20261031'
assert not (out/'launch.json').exists(),'Already launched'
with (out/'worker.log').open('w') as log:
    p=subprocess.Popen([sys.executable,'-u',str(r/'demo/ep_shared_fem_20261031/worker.py')],cwd=r,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=p.pid,physical_gpu=0,jobs=4,scope='Shared optimized AFSEI, CN20us, GPB Euler1us; 50ms each; openCARP not rerun')
(out/'launch.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
