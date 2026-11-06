from pathlib import Path
import json,subprocess
ROOT=Path('/mnt/large2/gjh/AFSEI');pair=ROOT/'results/ep_euler_fem_pair_20261031';base=ROOT/'demo/ep_afsei_nodal_opencarp_matched'
assert json.loads((pair/'preflight.json').read_text())['passed']
assert not (pair/'launch.json').exists(),'Pair already launched'
with (pair/'worker.log').open('w') as f:
    p=subprocess.Popen(['/mnt/large2/gjh/conda-envs/afsei/bin/python','-u',str(base/'run_pair_worker.py')],
        stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
(pair/'launch.json').write_text(json.dumps(dict(worker_pid=p.pid,physical_gpu=0),indent=2));print((pair/'launch.json').read_text())
