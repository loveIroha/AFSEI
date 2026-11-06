from pathlib import Path
import json,time,os
root=Path('/mnt/large2/gjh/AFSEI/results/ep_euler_fem_pair_20261031');last=None
while True:
    try:status=json.loads((root/'pair_progress.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):time.sleep(10);continue
    if status['status']!='running':print(json.dumps(status),flush=True);break
    method=status['method'];file=Path(status['output'])/'progress.json'
    try:p=json.loads(file.read_text())
    except (FileNotFoundError,json.JSONDecodeError):p={}
    key=(method,p.get('steps',0))
    if key!=last:last=key;print(json.dumps(dict(method=method,**p)),flush=True)
    time.sleep(10)
