"""Diagnostic reinitialization of the failed cap case, not a production restart.

The old accepted fields provide initial data for a changed fixed-neck domain.
This short numerical test cannot be spliced into the original trajectory.
Formal cycles must start from their own compatible passive loading history.
"""
import argparse
from pathlib import Path
from time import perf_counter
import json
import traceback
import numpy as np
import torch
from afsi_torch.ep import mac_balanced_checkpoint as restart
from afsi_torch.ep.gao_tracts import prepare_cut_flow
from afsi_torch.ep.mac_balanced_dynamic import BalancedAFSEIStepper
from afsi_torch.ep.protocol import ticks
from afsi_torch.ep.tract_storage import TractStorage


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--duration-ms',type=float,default=1.)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    report=dict(status='running',production_restart=False,
        scope='changed-neck initial/boundary problem from old fields; never splice into a production run',records=[])
    started=perf_counter()
    try:
        d,s,_=restart.load(args.checkpoint,cardiac=True)
        e=d._fsi_stepper(d.maximum_ticks);prepare_cut_flow(e,s.fsi,s)
        old_volume=e.flow.domain.storage.volume(s.fsi.x)
        d.cut_geometry=dict(d.cut_geometry,neck_corner_trim_cells=1)
        for endpoint in d._steppers.values():endpoint._gao_tract_flows={}
        prepare_cut_flow(e,s.fsi,s)
        domain=e.flow.domain;storage=domain.storage
        cap=e._gao_material_cap;reference=TractStorage(cap,domain)
        coverage=[]
        samples={'reference':d.mesh.X,'anchor':d.gao_tract_anchor,'accepted':s.fsi.x,
                 'previous-step prediction':2*s.fsi.x-s.fsi.previous_x}
        for name,x in samples.items():
            domain.validate_wall(cap,x)
            gpu=storage.volume(x);cpu=reference.volume(x)
            if abs(gpu-cpu)>2e-11:raise ValueError('CPU/GPU clipped storage mismatch')
            coverage.append(dict(name=name,storage_ml=gpu,cpu_gpu_difference_ml=gpu-cpu))
        report.update(start_time_ms=s.fsi.time*1000,geometry=domain.metadata,coverage=coverage,
            initial_storage_change_ml=storage.volume(s.fsi.x)-old_volume)
        target=s.tick+ticks(args.duration_ms/1000,d.protocol.dt_ep_s)
        while s.tick<target:
            # Avoid the cardiac cumulative ledger from the previous geometry.
            # The unchanged EP/FSI step still checks actual finite-step volume,
            # flux, nonlinear residual, work pairing and pressure law.
            t=perf_counter();s,info=BalancedAFSEIStepper.step(d,s,target_tick=target,diagnostics=True)
            row=dict(time_ms=s.fsi.time*1000,step_seconds=perf_counter()-t,
                mass_error_ml_s=info['balanced']['mass_error_ml_s'],
                boundary_work_relative_error=info['balanced']['boundary_work_relative_error'],
                transfer_power_relative_error=info['balanced']['transfer_power_relative_error'],
                nonlinear_iterations=info['nonlinear']['iterations'],
                accepted_dt_s=info['accepted_dt_s'],rejected_attempts=info.get('rejected_attempts',[]))
            report['records'].append(row);print(json.dumps(row),flush=True)
        report.update(status='passed',end_time_ms=s.fsi.time*1000,
            max_mass_error_ml_s=max(abs(r['mass_error_ml_s']) for r in report['records']))
    except Exception as exc:
        report.update(status='failed',failure=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    report['elapsed_seconds']=perf_counter()-started
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','traceback')}),flush=True)
    if report['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
