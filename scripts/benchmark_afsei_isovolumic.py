"""Compare numerical execution on an immutable checkpoint; warm probe excluded."""
import argparse
import json
import hashlib
from pathlib import Path
from time import perf_counter
import torch
from afsi_torch.ep import checkpoint

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--checkpoint',required=True)
    p.add_argument('--backend',choices=('spectral','multigrid'),required=True)
    p.add_argument('--steps',type=int,default=10); p.add_argument('--output',required=True)
    p.add_argument('--reference-responses',action='store_true')
    a=p.parse_args()
    if not torch.cuda.is_available() or not 1<=a.steps<=1000:
        p.error('actual CUDA and 1..1000 steps required')
    output=Path(a.output)
    output.mkdir(parents=True,exist_ok=False)
    initial_hash=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()
    driver,old,progress=checkpoint.load(a.checkpoint)
    driver.set_execution(pressure_backend=a.backend,response_optimizations=not a.reference_responses)
    driver.profile=True
    started=perf_counter(); driver.step(old); torch.cuda.synchronize()
    cold=perf_counter()-started
    driver._reset_scratch(old)
    state=old; rows=[]
    for _ in range(a.steps):
        torch.cuda.synchronize(); started=perf_counter()
        state,info=driver.step(state)
        torch.cuda.synchronize(); elapsed=perf_counter()-started
        rows.append(dict(wall_seconds=elapsed,**info['nonlinear'],timings=info['timings']))
        progress['ep_steps']=progress.get('ep_steps',0)+info['ep']['steps']
        progress['ode_substeps']=progress.get('ode_substeps',0)+info['ep']['ode_substeps']
    progress['elapsed_seconds']=progress.get('elapsed_seconds',0.)+cold+sum(r['wall_seconds'] for r in rows)
    checkpoint.save(output/'checkpoint.npz',driver,state,progress)
    report=dict(backend=a.backend,source=a.checkpoint,source_sha256=initial_hash,
        response_optimizations=not a.reference_responses,
        cold_probe_seconds=cold,steps=a.steps,mean_seconds=sum(r['wall_seconds'] for r in rows)/a.steps,
        physical_duration_s=state.fsi.time-old.fsi.time,
        timing_scope='synchronized coupled steps; one discarded cold probe excluded',
        pressure_mmhg=state.cavity_pressure_mmhg,volume_error_ml=info['isovolumic']['volume_error_ml'],
        rejected_attempts=state.rejected_attempts-old.rejected_attempts,rows=rows)
    assert initial_hash==hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()
    (output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2),flush=True)

if __name__=='__main__': main()
