"""Analyze one immutable accepted-data snapshot, never restart a simulation."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    case=Path(sys.argv[1]).resolve()
    rows=[]
    with (case/'steps.jsonl').open('rb') as f:
        for raw in f:
            if not raw.endswith(b'\n'):break
            try:row=json.loads(raw)
            except json.JSONDecodeError:raise ValueError('invalid completed accepted-step record')
            rows.append((raw,row))
    if not rows:raise ValueError('no accepted steps to analyze')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output=case/'reviews'/stamp;output.mkdir(parents=True,exist_ok=False)
    with (output/'steps.jsonl').open('wb') as f:
        for raw,_ in rows:f.write(raw)
    for name in ('configuration.json',):
        (output/name).write_bytes((case/name).read_bytes())
    # Initial checkpoint is immutable. A hard link avoids copying 84 MB; no
    # fields, metadata or solver buffers are changed by this CPU-only review.
    os.link(case/'initial_checkpoint.npz',output/'initial_checkpoint.npz')
    final=case/'report.json'
    state=json.loads((final if final.exists() else case/'progress.json').read_text())
    status={k:v for k,v in state.items() if k!='records'}
    # If the simulation finished after the prefix was captured, the shorter
    # prefix must remain a progress review, not a completed-event report.
    if state.get('end_tick') is not None and rows[-1][1]['time_ms']<state.get('end_time_ms',float('inf'))-1e-7:
        status['status']='running'
        if status.get('event_window'):status['event_window']['completed']=False
    (output/'progress.json').write_text(json.dumps(status,indent=2)+'\n')
    provenance=dict(source=str(case),snapshot=str(output),snapshot_at_utc=stamp,
        accepted_steps=len(rows),last_time_ms=rows[-1][1]['time_ms'],
        physical_time_advanced_by_review=False,initial_checkpoint_storage='hard link to immutable initial checkpoint')
    (output/'snapshot.json').write_text(json.dumps(provenance,indent=2)+'\n')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',MPLCONFIGDIR='/tmp/afsei-mpl-gao-progress')
    done=subprocess.run([sys.executable,'scripts/analyze_gao_ejection_interval.py',str(output)],env=env)
    if done.returncode:raise SystemExit(done.returncode)
    print(json.dumps(provenance,indent=2),flush=True)


if __name__=='__main__':main()
