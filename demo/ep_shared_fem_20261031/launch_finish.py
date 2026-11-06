from pathlib import Path
import subprocess,sys,json
r=Path('/mnt/large2/gjh/AFSEI');out=r/'results/ep_shared_fem_20261031'
assert not (out/'postprocess_launch.json').exists()
with (out/'postprocess.log').open('w') as log:
    p=subprocess.Popen([sys.executable,'-u',str(r/'demo/ep_shared_fem_20261031/finish.py')],cwd=r,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/'postprocess_launch.json').write_text(json.dumps(dict(pid=p.pid),indent=2));print(p.pid)
