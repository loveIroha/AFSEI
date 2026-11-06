"""One changed-boundary systolic probe; not a continuation of the old experiment."""
import json,hashlib,traceback
from pathlib import Path
from time import perf_counter
from dataclasses import asdict
import torch
from afsi_torch.ep import mac_balanced_checkpoint as restart
from afsi_torch.ep.cycle_policy import FillingClock,PeriodicMonitor
from afsi_torch.ep.active_diagnostics import active_metrics


def native(value):
    if isinstance(value,torch.Tensor):return value.item() if value.numel()==1 else value.tolist()
    if isinstance(value,dict):return {k:native(v) for k,v in value.items() if not k.startswith('_')}
    if isinstance(value,(tuple,list)):return [native(v) for v in value]
    return value


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    source=Path('results/gao_cut_cycle_validation_20261026/ejection_until_closure_20261028/last_accepted_checkpoint.npz')
    out=Path('results/afsei_upgrade_20261029/systolic_connection_probe');out.mkdir(exist_ok=False)
    before=hashlib.sha256(source.read_bytes()).hexdigest();started=perf_counter()
    d,s,_=restart.load(source,cardiac=True)
    d.cut_geometry={'shared_collar_cells':3};d.filling_clock=FillingClock();d.variable_pacing=True
    d.periodic_monitor=PeriodicMonitor();d.cut_initialization={'mode':'changed-boundary-diagnostic-only'}
    report=dict(status='running',model_transition_probe=True,formal_periodic_run=False,
        source_checkpoint=str(source),start_time_ms=s.fsi.time*1000,geometry=d.cut_geometry,
        filling_clock=asdict(d.filling_clock),source_sha256_before=before)
    try:
        new,info=d.step(s,target_tick=s.tick+2,diagnostics=True)
        report.update(status='passed',end_time_ms=new.fsi.time*1000,step=info,
            active_metrics=active_metrics(d.model,d.buffer,new.fsi.x))
        restart.save(out/'diagnostic_checkpoint.npz',d,new,dict(model_transition_probe=True))
        copied,restored,_=restart.load(out/'diagnostic_checkpoint.npz',cardiac=True)
        assert copied.cut_geometry==d.cut_geometry and copied.filling_clock==d.filling_clock
        assert torch.equal(restored.fsi.x,new.fsi.x)
        report['restart_roundtrip']=True
    except Exception as exc:
        report.update(status='failed',failure=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc(),
                      attempts=getattr(exc,'attempts',[]))
    report.update(elapsed_seconds=perf_counter()-started,
        source_sha256_after=hashlib.sha256(source.read_bytes()).hexdigest())
    (out/'report.json').write_text(json.dumps(native(report),indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in native(report).items() if k!='step'},indent=2),flush=True)
    if report['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
