"""Copy published phase checkpoints without touching a running solver or GPU."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,required=True);p.add_argument('--pid',type=int,required=True)
    p.add_argument('--interval',type=float,default=30.)
    a=p.parse_args()
    if a.interval<10:raise ValueError('at least 10 seconds between read-only checks')
    cmdline=Path(f'/proc/{a.pid}/cmdline')
    identity=cmdline.read_bytes()
    if b'run_afsei_cut_cycle.py' not in identity:raise ValueError('expected cycle solver PID')
    out=a.case/'phase_checkpoints';out.mkdir(exist_ok=False)
    report=dict(observer_pid=os.getpid(),solver_pid=a.pid,status='running',entries=[],
        source='copies of atomic published checkpoints; no GPU fields requested',
        started_at_utc=datetime.now(timezone.utc).isoformat())
    keys=set()
    def publish():
        tmp=out/'manifest.json.working';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(out/'manifest.json')
    def capture(path,final=False):
        if not path.exists():return
        with path.open('rb') as stream:
            with np.load(stream,allow_pickle=False) as data:meta=json.loads(str(data['metadata']))
            c=meta['cardiac'];phase=c['phase'];beat=c['beat'];mode=meta['balanced']['mode']
            tags=[f'beat{beat}_{phase}_{mode}']
            pulse=meta['config']['protocol']['pulse_starts_s'][beat-1]
            if phase=='EJECTION' and meta['time']>=pulse+.15:tags.append(f'beat{beat}_ejection_150ms_sample')
            if phase=='FILLING' and meta['tick']>=c['fill_end_tick']:tags.append(f'beat{beat}_filling_hold_start')
            if final:tags.append('final_published_state')
            new=[tag for tag in tags if tag not in keys]
            if not new:return
            name=f'{len(report["entries"]):02d}_{new[0]}_tick{meta["tick"]:08d}.npz'
            stream.seek(0)
            temporary=out/(name+'.working')
            with temporary.open('xb') as dest:shutil.copyfileobj(stream,dest,1024*1024)
            temporary.replace(out/name)
            keys.update(new)
            report['entries'].append(dict(file=name,tags=new,tick=meta['tick'],time_ms=meta['time']*1000,
                phase=phase,beat=beat,mode=mode,source_sha256=meta['sha256'],
                captured_at_utc=datetime.now(timezone.utc).isoformat()))
            publish()
    publish()
    while cmdline.exists():
        try:
            if cmdline.read_bytes()!=identity:break
            capture(a.case/'checkpoint.npz' if (a.case/'checkpoint.npz').exists() else a.case/'initial_checkpoint.npz')
        except Exception as exc:
            report['last_observation_error']=type(exc).__name__+': '+str(exc);publish()
        time.sleep(a.interval)
    final=a.case/'last_accepted_checkpoint.npz'
    if not final.exists():final=a.case/'checkpoint.npz'
    capture(final,final=True)
    report.update(status='solver-exited',finished_at_utc=datetime.now(timezone.utc).isoformat());publish()


if __name__=='__main__':main()
