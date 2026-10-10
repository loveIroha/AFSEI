"""Advance EP/GPB/Land with a jointly constrained weak closed MAC basal cap.

Validation only: both idealized port regions are closed, no ejection/filling
switch is made and the arterial RCR has assumed Q=0, not measured port flow.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import torch
from afsi_torch.ep import checkpoint
from afsi_torch.ep.mac_isovolumic import IsovolumicAFSEIStepper,IsovolumicState


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--steps',type=int,default=10)
    p.add_argument('--patch-width-cells',type=float,default=None)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not torch.cuda.is_available() or not 1<=a.steps<=10000:
        p.error('actual CUDA and 1..10000 steps required')
    a.output.mkdir(parents=True,exist_ok=False)
    sha=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()
    driver,state,progress=checkpoint.load(a.checkpoint)
    if not isinstance(driver,IsovolumicAFSEIStepper):
        driver=IsovolumicAFSEIStepper.from_driver(driver); state=driver.lock(state)
    if (state.cap_pressure is not None and a.patch_width_cells is not None
            and a.patch_width_cells!=driver.cap_patch_width_cells):
        p.error('cannot change the saved cap pressure partition on restart')
    driver.set_execution(closed_cap=True,cap_patch_width_cells=a.patch_width_cells)
    driver.profile=True
    start_time=state.fsi.time; initial_step=state.fsi.step
    initial_rejected=state.rejected_attempts
    elapsed_previous=progress.get('elapsed_seconds',0.)
    rows=[]; started=perf_counter()
    try:
        for _ in range(a.steps):
            torch.cuda.synchronize(); step_start=perf_counter()
            state,info=driver.step(state,diagnostics=True)
            torch.cuda.synchronize(); elapsed=perf_counter()-step_start
            rows.append(dict(step=state.fsi.step,time_s=state.fsi.time,wall_seconds=elapsed,
                cap=info['cap'],isovolumic=info['isovolumic'],nonlinear=info['nonlinear'],
                timings=info['timings'],divergence_l2=info.get('divergence_l2'),
                ib_power_error_erg_s=info.get('power_error')))
            progress['ep_steps']=progress.get('ep_steps',0)+info['ep']['steps']
            progress['ode_substeps']=progress.get('ode_substeps',0)+info['ep']['ode_substeps']
            # Publishing the large full EP checkpoint every 0.1 ms dominates
            # this short validation. Retain periodic and final/failure saves.
            if len(rows)%50==0:
                progress['elapsed_seconds']=elapsed_previous+perf_counter()-started
                checkpoint.save(a.output/'checkpoint.npz',driver,state,progress)
            print(f'Closed cap step={state.fsi.step}, t={1000*state.fsi.time:.3f}ms, '
                f'p={state.cavity_pressure_mmhg:.6g}mmHg, '
                f'flux={info["cap"]["constraint_residual_norm_ml_s"]:.3g}ml/s, '
                f'wall={elapsed:.3f}s',flush=True)
    finally:
        progress['elapsed_seconds']=elapsed_previous+perf_counter()-started
        checkpoint.save(a.output/'checkpoint.npz',driver,state,progress)
        report=dict(status='completed' if len(rows)==a.steps else 'incomplete',
            scope=__doc__,source_checkpoint=a.checkpoint,source_sha256=sha,
            accepted_steps=len(rows),start_step=initial_step,physical_duration_s=state.fsi.time-start_time,
            reached_time_s=state.fsi.time,elapsed_seconds=perf_counter()-started,
            rejected_attempts=state.rejected_attempts-initial_rejected,
            step_timing_scope='CUDA synchronized coupled step; checkpoint/output IO excluded; first includes cold setup',
            mean_seconds_per_step=sum(r['wall_seconds'] for r in rows)/len(rows) if rows else None,
            mean_seconds_after_first=sum(r['wall_seconds'] for r in rows[1:])/len(rows[1:]) if len(rows)>1 else None,
            execution=driver.execution_settings(),rows=rows,
            ep_advanced=bool(rows),myocardium_motion_advanced=bool(rows),
            closed_cap_reactions_in_same_implicit_fsi_solve=bool(rows),
            pointwise_impermeability_established=False,windkessel_actual_port_flow_connected=False,
            full_cardiac_cycle_implemented=False,three_field_integration_implemented=False)
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        assert sha==hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','scope')},indent=2),flush=True)


if __name__=='__main__': main()
