"""One-shot dependent analysis for the designated grid queue, no new solves."""
from pathlib import Path
import json,os,subprocess,sys,time
root=Path('/mnt/large2/gjh/AFSEI/results/accuracy_performance_20261025')
started=time.monotonic()
while True:
    paths=[root/f'queue_gpu{i}.json' for i in (0,1)]
    states=[json.loads(p.read_text()) if p.exists() else {} for p in paths]
    if all(s.get('stage') in ('completed','failed') for s in states):break
    if time.monotonic()-started>24*3600:
        (root/'analysis_timeout.json').write_text(json.dumps(dict(simulations_not_interrupted=True,states=states),indent=2));sys.exit(1)
    time.sleep(30)
env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
sys.exit(subprocess.call([sys.executable,'-u','demo/verification/analyze_accuracy_campaign.py'],env=env))
