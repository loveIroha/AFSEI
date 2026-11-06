from pathlib import Path
import json,time,subprocess,sys,os,tarfile
ROOT=Path('/mnt/large2/gjh/AFSEI');OUT=ROOT/'results/ep_shared_fem_20261031'
while True:
    status=json.loads((OUT/'progress.json').read_text())
    if status['status']=='failed':raise RuntimeError(status)
    if status['status']=='simulations_completed':break
    time.sleep(5)
env=os.environ.copy();env.update(MPLCONFIGDIR=str(ROOT/'.cache/matplotlib'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
subprocess.run([sys.executable,str(ROOT/'demo/ep_shared_fem_20261031/analyze.py')],cwd=ROOT,env=env,check=True)
with tarfile.open(OUT/'comparison_bundle.tar.gz','w:gz') as tar:
    for item in (OUT/'comparison').iterdir():tar.add(item,arcname=item.name)
    tar.add(OUT/'source_hashes.json',arcname='source_hashes.json')
print('ANALYSIS_AND_PACKAGE_COMPLETED',flush=True)
