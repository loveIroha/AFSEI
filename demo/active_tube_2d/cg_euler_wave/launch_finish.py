from pathlib import Path
import sys,subprocess,json
R=Path('/mnt/large2/gjh/AFSEI');O=R/'results/tube_cg_euler_wave_20261031'
assert not (O/'postprocess_launch.json').exists()
with (O/'postprocess.log').open('w') as log:
    p=subprocess.Popen([sys.executable,'-u',str(R/'demo/active_tube_2d/cg_euler_wave/finish.py')],cwd=R,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
(O/'postprocess_launch.json').write_text(json.dumps(dict(pid=p.pid)));print(p.pid)
